"""Pure stdlib tests for the background orchestration, with no ML imports.

Run this file directly with Python or collect it in the server pytest suite.
All checkpoints, Linux process records and evaluation artifacts are synthetic.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "t2mem_pipeline_under_test", REPOSITORY / "scripts/t2mem/run_pipeline.py"
)
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)
READY_SPEC = importlib.util.spec_from_file_location(
    "t2mem_readiness_under_test", REPOSITORY / "scripts/t2mem/check_policy_ready.py"
)
assert READY_SPEC is not None and READY_SPEC.loader is not None
readiness = importlib.util.module_from_spec(READY_SPEC)
sys.modules[READY_SPEC.name] = readiness
READY_SPEC.loader.exec_module(readiness)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def checkpoint(root: Path, step=20000, *, mode="stage1", complete=True, source=None) -> Path:
    directory = root.resolve() / str(step)
    for name in ("params", "train_state"):
        (directory / name).mkdir(parents=True)
    write_json(directory / "assets/robomme/norm_stats.json", {"actions": {}, "state": {}})
    alternating = mode == "alternating"
    cycle, offset = divmod(step, 1000)
    phase = "stage2a" if offset < 500 else "stage2b"
    phase_step = offset if offset < 500 else offset - 500
    stage2a = cycle * 500 + min(offset, 500)
    stage2b = cycle * 500 + max(offset - 500, 0)
    if alternating and complete and step == 4000:
        cycle, phase, phase_step = 3, "stage2b", 500
    metadata = {
        "format_version": 1,
        "complete": True,
        "config": {
            "name": "synthetic-pipeline-test",
            "num_devices": 6,
            "gpu_id": 1,
            "model": {"pi05": True, "memory_enabled": True, "action_horizon": 64,
                      "robot_action_dim": 8, "memory": {"detach_fast_state": False}},
            "data": {"manifest": "data/t2mem/manifest.json", "write_stride": 16,
                     "truncate_prefix": False, "max_sequence_length": None},
            "training": {"stage1_updates": 20000, "memory_updates_per_cycle": 500,
                         "policy_updates_per_cycle": 500, "alternating_cycles": 4,
                         "save_interval": 5000},
        },
        "progress": {
            "mode": mode, "phase": phase if alternating else mode,
            "global_step": step, "phase_step": phase_step if alternating else step,
            "cycle": cycle if alternating else 0,
            "stage_updates": {"stage1": 0 if alternating else step,
                              "stage2a": stage2a if alternating else 0,
                              "stage2b": stage2b if alternating else 0},
            "complete": complete,
        },
        "initialization_source": source,
        "approximations": [],
        "optimizer_stages": ["stage2a", "stage2b"] if alternating else ["stage1"],
        "sampler": {},
    }
    write_json(directory / "metadata.json", metadata)
    return directory


def evaluation(root: Path, ckpt: Path, ablation="memory-free") -> Path:
    metadata = json.loads((ckpt / "metadata.json").read_text())
    server = {
        "checkpoint": str(ckpt.resolve()), "ablation": ablation,
        "action_horizon": 64, "execution_horizon": 32, "memory_write_stride": 16,
        "checkpoint_progress": metadata["progress"],
        "initialization_source": metadata["initialization_source"],
    }
    manifest = {
        "experiment": "single-task reproduction", "tasks": ["PickXtimes"],
        "seeds": [7, 17, 27], "episodes_per_task": 50, "max_steps": 1300,
        "simulation_mode": "gpu", "cpu_render_support_verified": False,
        "server": server,
    }
    records = [
        {"task": "PickXtimes", "seed": seed, "episode_id": episode,
         "difficulty": "easy" if episode < 26 else "medium" if episode < 38 else "hard",
         "status": "timeout", "success": False, "steps": 1300}
        for seed in (7, 17, 27) for episode in range(50)
    ]
    metrics = {
        "complete": True, "expected_rollouts": 150, "valid_rollouts": 150,
        "infrastructure_errors": 0, "paper_difficulty_counts_match": True,
        "difficulty_counts": {f"PickXtimes/seed{seed}": {"easy": 26, "medium": 12, "hard": 12}
                              for seed in (7, 17, 27)},
        "tasks": {"PickXtimes": {"rollouts": 150, "successes": 0, "success_rate": 0.0}},
    }
    for name, value in (("manifest.json", manifest), ("episodes.json", records), ("metrics.json", metrics)):
        write_json(root / name, value)
    return root


def server_metadata(ckpt: Path, ablation="memory-free") -> dict:
    expected = json.loads((ckpt / "metadata.json").read_text())
    return {
        "protocol": "t2mem.v1", "tasks": ["PickXtimes"], "checkpoint": str(ckpt),
        "checkpoint_progress": expected["progress"], "ablation": ablation,
        "action_horizon": 64, "execution_horizon": 32, "memory_write_stride": 16,
        "demo_write_offset": 0, "flow_denoising_steps": 10, "schedule_override": False,
        "approximations": [], "memory_enabled": ablation != "memory-free",
        "online_write": ablation == "full",
        "preprocessing": {"normalization_sha256": hashlib.sha256(
            (ckpt / "assets/robomme/norm_stats.json").read_bytes()).hexdigest()},
    }


class PipelineArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_committed_snapshot_is_distinct_from_a_completed_training_run(self):
        unfinished = checkpoint(self.root / "stage1", 10000, complete=False)
        self.assertEqual(pipeline.latest_complete_checkpoint(unfinished.parent), unfinished)
        with self.assertRaises((ValueError, FileNotFoundError)):
            pipeline.validate_checkpoint(unfinished, "stage1", 20000)
        valid = checkpoint(unfinished.parent)
        result = pipeline.validate_checkpoint(valid, "stage1", 20000)
        self.assertTrue(result["progress"]["complete"])

    def test_latest_snapshot_ignores_interrupted_save_and_corrupt_metadata(self):
        valid = checkpoint(self.root / "stage2", 3000, mode="alternating", complete=False)
        (valid.parent / "4000/params").mkdir(parents=True)
        (valid.parent / "5000").mkdir()
        (valid.parent / "5000/metadata.json").write_text("{truncated")
        self.assertEqual(pipeline.latest_complete_checkpoint(valid.parent), valid)

    def test_stage2_rejects_wrong_stage1_source_even_when_budget_is_complete(self):
        a = checkpoint(self.root / "a")
        wrong = checkpoint(self.root / "wrong_a")
        b = checkpoint(self.root / "b", 4000, mode="alternating", source={"checkpoint": str(wrong)})
        with self.assertRaises(ValueError):
            pipeline.validate_checkpoint(b, "alternating", 4000, cycles=4, source=a)

    def test_wrong_checkpoint_mode_and_incomplete_or_underbudget_state_fail_closed(self):
        directory = checkpoint(self.root / "a")
        original = json.loads((directory / "metadata.json").read_text())
        for modify in (
            lambda m: m.update(complete=False),
            lambda m: m["progress"].update(mode="stage2a"),
            lambda m: m["progress"].update(global_step=19999),
        ):
            with self.subTest(modify=modify):
                metadata = copy.deepcopy(original)
                modify(metadata)
                write_json(directory / "metadata.json", metadata)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    pipeline.validate_checkpoint(directory, "stage1", 20000)

    def test_all_timeout_rollouts_are_valid_evaluation_not_infrastructure_failure(self):
        a = checkpoint(self.root / "a")
        results = evaluation(self.root / "evaluation", a)
        self.assertTrue(pipeline.validate_evaluation(results, a, "memory-free"))

    def test_evaluation_infrastructure_error_or_duplicate_episode_cannot_count_as_complete(self):
        a = checkpoint(self.root / "a")
        results = evaluation(self.root / "evaluation", a)
        metrics = json.loads((results / "metrics.json").read_text())
        metrics.update(complete=False, infrastructure_errors=1, valid_rollouts=149)
        write_json(results / "metrics.json", metrics)
        self.assertFalse(pipeline.validate_evaluation(results, a, "memory-free"))
        evaluation(results, a)
        records = json.loads((results / "episodes.json").read_text())
        records[-1] = records[0]
        write_json(results / "episodes.json", records)
        self.assertFalse(pipeline.validate_evaluation(results, a, "memory-free"))

    def test_existing_results_with_wrong_checkpoint_or_ablation_are_never_reused(self):
        a = checkpoint(self.root / "a")
        wrong = checkpoint(self.root / "wrong_a")
        results = evaluation(self.root / "evaluation", wrong)
        with self.assertRaises(ValueError):
            pipeline.validate_evaluation(results, a, "memory-free")
        evaluation(results, a, "full")
        with self.assertRaises(ValueError):
            pipeline.validate_evaluation(results, a, "memory-free")

    def test_partial_evaluation_can_resume_without_being_claimed_complete(self):
        a = checkpoint(self.root / "a")
        results = evaluation(self.root / "evaluation", a)
        (results / "metrics.json").unlink()
        self.assertFalse(pipeline.validate_evaluation(results, a, "memory-free"))

    def test_readiness_requires_actual_checkpoint_metadata_not_just_open_port(self):
        a = checkpoint(self.root / "a")
        good = server_metadata(a)
        readiness.validate_metadata(good, a, "memory-free")
        for modify in (
            lambda m: m.update(checkpoint=str(self.root / "other/20000")),
            lambda m: m["checkpoint_progress"].update(global_step=10000),
            lambda m: m["preprocessing"].update(normalization_sha256="wrong"),
            lambda m: m.update(protocol="other.v1"),
            lambda m: m.update(ablation="full"),
        ):
            with self.subTest(modify=modify):
                wrong = copy.deepcopy(good)
                modify(wrong)
                with self.assertRaises(ValueError):
                    readiness.validate_metadata(wrong, a, "memory-free")

    def test_no_write_readiness_rejects_online_writes_on_same_b_checkpoint(self):
        a = checkpoint(self.root / "a")
        b = checkpoint(self.root / "b", 4000, mode="alternating", source={"checkpoint": str(a)})
        good = server_metadata(b, "no-online-write")
        readiness.validate_metadata(good, b, "no-online-write")
        good["online_write"] = True
        with self.assertRaises(ValueError):
            readiness.validate_metadata(good, b, "no-online-write")


class PipelineProcessTests(unittest.TestCase):
    def test_resource_wait_pid_is_optional_and_must_be_positive(self):
        self.assertIsNone(pipeline.parse_args([]).wait_pid)
        options = pipeline.parse_args(["--wait-pid", "2033835", "--check-interval", "10"])
        self.assertEqual(options.wait_pid, 2033835)
        self.assertEqual(options.check_interval, 10.0)
        self.assertEqual(options.watch_pid, 1776511)
        for value in ("0", "-1", "bad"):
            with self.subTest(value=value), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    pipeline.parse_args(["--wait-pid", value])

    def test_queue_defaults_separate_fast_checks_from_wait_log_interval(self):
        options = pipeline.parse_args([])
        self.assertEqual(options.check_interval, 2.0)
        self.assertEqual(options.poll_interval, 600)
        self.assertEqual(options.gpu_policy, "exclusive")
        self.assertEqual(options.min_free_mib, 60000)
        options = pipeline.parse_args([
            "--devices", "6", "--check-interval", "0.5", "--gpu-policy", "memory",
            "--min-free-mib", "65000",
        ])
        self.assertEqual(options.devices, (6,))
        self.assertEqual(options.check_interval, 0.5)
        self.assertEqual(options.gpu_policy, "memory")
        self.assertEqual(options.min_free_mib, 65000)

    def test_queue_rejects_invalid_check_interval_and_memory_threshold(self):
        for flag, value in (("--check-interval", "0"), ("--check-interval", "-1"),
                            ("--check-interval", "nan"), ("--check-interval", "inf"),
                            ("--min-free-mib", "0"), ("--gpu-policy", "unknown")):
            with self.subTest(flag=flag, value=value), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    pipeline.parse_args([flag, value])

    def test_gpu_queue_ignores_other_devices_and_waits_for_selected_compute_jobs(self):
        def responses(occupied):
            def runner(command, **kwargs):
                query = next(part for part in command if part.startswith("--query-"))
                if query == "--query-gpu=index,uuid":
                    output = "0, GPU-zero\n1, GPU-one\n2, GPU-two\n"
                elif query == "--query-compute-apps=gpu_uuid,pid":
                    output = occupied
                else:
                    self.fail(f"Unexpected GPU probe: {command}")
                return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
            return runner

        self.assertTrue(pipeline.gpu_is_idle((1, 2), runner=responses("GPU-zero, 123\n")))
        self.assertFalse(pipeline.gpu_is_idle((1, 2), runner=responses("GPU-one, 456\n")))
        self.assertFalse(pipeline.gpu_is_idle((1, 6), runner=responses("")))

    def test_gpu_query_failure_never_means_devices_are_available(self):
        runner = mock.Mock(side_effect=subprocess.CalledProcessError(1, "nvidia-smi"))
        try:
            available = pipeline.gpu_is_idle((1, 2), runner=runner)
        except (RuntimeError, subprocess.CalledProcessError):
            return
        self.assertFalse(available)

    def _memory_gpu_probe(self, *, free="71533", utilization="0", present=True,
                          occupied="GPU-six, 14789\n"):
        def runner(command, **kwargs):
            query = next(part for part in command if part.startswith("--query-"))
            if query == "--query-gpu=index,uuid,memory.free,utilization.gpu":
                output = f"6, GPU-six, {free}, {utilization}\n" if present else "5, GPU-five, 80000, 0\n"
            elif query == "--query-compute-apps=gpu_uuid,pid":
                output = occupied
            else:
                self.fail(f"Unexpected GPU availability probe: {command}")
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        return runner

    def test_explicit_memory_policy_accepts_gpu_six_with_enough_free_memory_and_existing_pid(self):
        available, details = pipeline.gpu_availability(
            (6,), policy="memory", min_free_mib=60000, runner=self._memory_gpu_probe(),
        )
        self.assertTrue(available)
        self.assertEqual(details["policy"], "memory")
        self.assertEqual(details["gpus"][0]["free_mib"], 71533)
        self.assertEqual(details["gpus"][0]["compute_pids"], [14789])
        self.assertFalse(details["gpus"][0]["blocked"])
        self.assertEqual(details["gpus"][0]["block_reasons"], [])

    def test_exclusive_policy_still_rejects_existing_compute_pid_with_free_memory(self):
        available, details = pipeline.gpu_availability(
            (6,), policy="exclusive", min_free_mib=60000, runner=self._memory_gpu_probe(),
        )
        self.assertFalse(available)
        self.assertTrue(details["gpus"][0]["blocked"])
        self.assertTrue(details["gpus"][0]["block_reasons"])

    def test_unselected_gpu_unknown_metrics_do_not_block_selected_gpu_six(self):
        def runner(command, **kwargs):
            query = next(part for part in command if part.startswith("--query-"))
            output = ("0, GPU-zero, N/A, N/A\n6, GPU-six, 71533, 0\n"
                      if query == "--query-gpu=index,uuid,memory.free,utilization.gpu"
                      else "GPU-zero, N/A\nGPU-six, 14789\n")
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

        available, details = pipeline.gpu_availability((6,), "memory", 60000, runner=runner)
        self.assertTrue(available)
        self.assertEqual([record["index"] for record in details["gpus"]], [6])
        self.assertEqual(details["gpus"][0]["compute_pids"], [14789])

    def test_memory_policy_rejects_insufficient_unknown_or_missing_gpu_metrics(self):
        for values in ({"free": "59999"}, {"free": "N/A"}, {"utilization": "N/A"}, {"present": False}):
            with self.subTest(values=values):
                available, details = pipeline.gpu_availability(
                    (6,), policy="memory", min_free_mib=60000, runner=self._memory_gpu_probe(**values),
                )
                self.assertFalse(available)
                self.assertTrue(details["gpus"][0]["blocked"])
                self.assertTrue(details["gpus"][0]["block_reasons"])

    def test_memory_policy_gpu_probe_failure_never_grants_admission(self):
        for error in (subprocess.CalledProcessError(1, "nvidia-smi"),
                      subprocess.TimeoutExpired("nvidia-smi", 30)):
            with self.subTest(error=error):
                available, details = pipeline.gpu_availability(
                    (6,), policy="memory", min_free_mib=60000, runner=mock.Mock(side_effect=error),
                )
                self.assertFalse(available)
                self.assertTrue(details["probe_error"])

    def test_pid_reuse_different_uid_and_zombie_are_not_original_training(self):
        identity = {"pid": 1776511, "start_time": "123456", "uid": 1002, "state": "S"}
        self.assertTrue(pipeline.same_process(identity, dict(identity, state="R")))
        for actual in (None, dict(identity, start_time="987654"), dict(identity, uid=1003),
                       dict(identity, state="Z"), dict(identity, state="X")):
            with self.subTest(actual=actual):
                self.assertFalse(pipeline.same_process(identity, actual))

    def test_linux_proc_reader_handles_process_name_spaces_and_parentheses(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary) / "1776511"
            proc.mkdir()
            fields = ["S"] + ["0"] * 49
            fields[19] = "23456789"
            (proc / "stat").write_text("1776511 (python worker (train)) " + " ".join(fields))
            (proc / "status").write_text("Name:\tpython\nUid:\t1002\t1002\t1002\t1002\n")
            actual = pipeline.read_process_identity(1776511, proc_root=Path(temporary))
            self.assertIsNotNone(actual)
            self.assertEqual(actual["pid"], 1776511)
            self.assertEqual(str(actual["start_time"]), "23456789")
            self.assertEqual(actual["uid"], 1002)
            self.assertIsNone(pipeline.read_process_identity(1, proc_root=Path(temporary)))


class PipelineLockScopeTests(unittest.TestCase):
    def test_only_matching_exclusive_flock_owner_and_device_inode_prove_pipeline_scope(self):
        lock_path = mock.Mock()
        lock_path.stat.return_value = mock.Mock(st_dev=pipeline.os.makedev(8, 1), st_ino=123456)
        entry = mock.Mock()
        lines = (
            ("lock: 1: FLOCK ADVISORY WRITE 700003 08:01:123456 0 EOF", True),
            ("lock: 1: FLOCK ADVISORY READ 700003 08:01:123456 0 EOF", False),
            ("lock: 1: POSIX ADVISORY WRITE 700003 08:01:123456 0 EOF", False),
            ("lock: 1: FLOCK ADVISORY WRITE 700004 08:01:123456 0 EOF", False),
            ("lock: 1: FLOCK ADVISORY WRITE 700003 08:01:123457 0 EOF", False),
            ("lock: 1: FLOCK ADVISORY WRITE 700003 09:01:123456 0 EOF", False),
            ("lock: 1: FLOCK ADVISORY WRITE 700003 08:02:123456 0 EOF", False),
            ("lock: 1: FLOCK ADVISORY WRITE 700003 malformed 0 EOF", False),
        )
        with mock.patch.object(pipeline.Path, "iterdir", return_value=[entry]):
            for line, expected in lines:
                with self.subTest(line=line):
                    entry.read_text.return_value = "pos:\t0\n" + line + "\n"
                    self.assertEqual(pipeline._process_holds_pipeline_lock(700003, lock_path), expected)

    def test_unreadable_process_fdinfo_never_proves_lock_ownership(self):
        lock_path = mock.Mock()
        lock_path.stat.return_value = mock.Mock(st_dev=pipeline.os.makedev(8, 1), st_ino=123456)
        with mock.patch.object(pipeline.Path, "iterdir", side_effect=PermissionError("synthetic /proc denial")):
            self.assertFalse(pipeline._process_holds_pipeline_lock(700003, lock_path))


class PipelineReplacementTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.pid = 700003
        self.identity = {"pid": self.pid, "start_time": "123456", "uid": pipeline.os.getuid(), "state": "S"}
        self.state = {"driver_pid": self.pid, "status": "RUNNING", "stage": "WAIT_GPUS"}
        write_json(self.directory / "state.json", self.state)
        self.lock = mock.Mock()
        self.lock.path = self.directory / "pipeline.lock"
        self.lock.acquire.return_value = False
        self.pidfd = 100003
        self.stopped = False
        self.exited = False

        def read_identity(_pid):
            return None if self.exited else dict(self.identity, state="T" if self.stopped else "S")

        def send_signal(fd, signum, *_args):
            self.assertEqual(fd, self.pidfd)
            if signum == pipeline.signal.SIGSTOP:
                self.stopped = True
            elif signum == pipeline.signal.SIGTERM:
                self.exited = True
            elif signum == pipeline.signal.SIGCONT:
                self.stopped = False

        patches = (
            ("FileLock", mock.patch.object(pipeline, "FileLock", return_value=self.lock)),
            ("read_identity", mock.patch.object(pipeline, "read_process_identity", side_effect=read_identity)),
            ("kill", mock.patch.object(pipeline.os, "kill")),
            ("pidfd_open", mock.patch.object(pipeline.os, "pidfd_open", return_value=self.pidfd, create=True)),
            ("send_signal", mock.patch.object(pipeline.signal, "pidfd_send_signal", side_effect=send_signal, create=True)),
            ("install_signal", mock.patch.object(pipeline.signal, "signal")),
            ("close", mock.patch.object(pipeline.os, "close")),
            ("owns_lock", mock.patch.object(pipeline, "_process_holds_pipeline_lock", return_value=True)),
            ("read_command", mock.patch.object(pipeline.Path, "read_bytes",
                                              return_value=b"python\0scripts/t2mem/run_pipeline.py\0")),
            ("sleep", mock.patch.object(pipeline.time, "sleep")),
        )
        for name, patch in patches:
            setattr(self, name, patch.start())
            self.addCleanup(patch.stop)

    def signals(self):
        self.kill.assert_not_called()
        return [call.args[1] for call in self.send_signal.call_args_list]

    def test_replace_waiter_flag_is_explicit_and_disabled_by_default(self):
        self.assertFalse(pipeline.parse_args([]).replace_waiter)
        self.assertTrue(pipeline.parse_args(["--replace-waiter"]).replace_waiter)

    def test_idle_pipeline_lock_needs_no_process_probe_or_signal(self):
        self.lock.acquire.return_value = True
        pipeline.replace_waiting_driver(self.directory)
        self.kill.assert_not_called()
        self.read_identity.assert_not_called()
        self.read_command.assert_not_called()
        self.pidfd_open.assert_not_called()
        self.send_signal.assert_not_called()
        self.lock.release.assert_called_once()

    def test_verified_waiting_driver_is_frozen_then_terminated_and_replacement_waits_for_lock(self):
        self.lock.acquire.side_effect = [False, False, True]
        pipeline.replace_waiting_driver(self.directory)
        self.assertEqual(self.signals(), [pipeline.signal.SIGSTOP, pipeline.signal.SIGTERM, pipeline.signal.SIGCONT])
        self.pidfd_open.assert_called_once_with(self.pid, 0)
        self.owns_lock.assert_called_with(self.pid, self.directory / "pipeline.lock")
        self.sleep.assert_called_once_with(0.2)
        self.lock.release.assert_called_once()
        self.close.assert_called_once_with(self.pidfd)

    def test_replacement_refuses_active_work_or_nonrunning_status_without_signals(self):
        for stage, status in (("TRAIN_STAGE2", "RUNNING"), ("EVALUATE_A", "RUNNING"),
                              ("COMPARE", "RUNNING"), ("WAIT_GPUS", "COMPLETE")):
            with self.subTest(stage=stage, status=status):
                write_json(self.directory / "state.json", dict(self.state, stage=stage, status=status))
                with self.assertRaisesRegex(RuntimeError, "not waiting"):
                    pipeline.replace_waiting_driver(self.directory)
        self.kill.assert_not_called()
        self.read_identity.assert_not_called()
        self.send_signal.assert_not_called()

    def test_replacement_refuses_another_uid_or_unrelated_process_command(self):
        self.read_identity.side_effect = None
        self.read_identity.return_value = dict(self.identity, uid=self.identity["uid"] + 1)
        with self.assertRaisesRegex(RuntimeError, "belongs to this account"):
            pipeline.replace_waiting_driver(self.directory)
        self.read_command.assert_not_called()
        self.read_identity.return_value = self.identity
        self.read_command.return_value = b"python\0unrelated_job.py\0"
        with self.assertRaisesRegex(RuntimeError, "not the pipeline driver"):
            pipeline.replace_waiting_driver(self.directory)
        self.kill.assert_not_called()
        self.send_signal.assert_not_called()

    def test_replacement_rechecks_process_identity_to_reject_pid_reuse_before_signal(self):
        self.read_identity.side_effect = [self.identity, dict(self.identity, start_time="999999")]
        with self.assertRaises(RuntimeError):
            pipeline.replace_waiting_driver(self.directory)
        self.kill.assert_not_called()
        self.send_signal.assert_not_called()

    def test_driver_that_became_active_before_freeze_is_resumed_without_termination(self):
        initial_send = self.send_signal.side_effect

        def send(fd, signum, *args):
            initial_send(fd, signum, *args)
            if signum == pipeline.signal.SIGSTOP:
                write_json(self.directory / "state.json", dict(self.state, stage="TRAIN_STAGE2"))

        self.send_signal.side_effect = send
        with self.assertRaises(RuntimeError):
            pipeline.replace_waiting_driver(self.directory)
        self.assertEqual(self.signals(), [pipeline.signal.SIGSTOP, pipeline.signal.SIGCONT])
        self.assertFalse(self.stopped)
        self.assertFalse(self.exited)
        self.close.assert_called_once_with(self.pidfd)

    def test_replacement_refuses_when_pidfd_api_is_unavailable(self):
        unavailable_os = mock.Mock(spec=["getuid", "getpid"])
        unavailable_os.getuid.return_value = self.identity["uid"]
        unavailable_os.getpid.return_value = 999999
        with mock.patch.object(pipeline, "os", unavailable_os):
            with self.assertRaises(RuntimeError):
                pipeline.replace_waiting_driver(self.directory)
        self.send_signal.assert_not_called()
        self.kill.assert_not_called()

    def test_idle_lock_is_a_noop_even_if_pidfd_api_is_unavailable(self):
        self.lock.acquire.return_value = True
        unavailable_os = mock.Mock(spec=["getuid", "getpid"])
        with mock.patch.object(pipeline, "os", unavailable_os):
            pipeline.replace_waiting_driver(self.directory)
        self.send_signal.assert_not_called()
        self.read_identity.assert_not_called()
        self.lock.release.assert_called_once()

    def test_preexisting_stopped_driver_is_not_signalled_or_resumed(self):
        self.read_identity.side_effect = None
        for stopped_state in ("T", "t"):
            with self.subTest(state=stopped_state):
                self.read_identity.return_value = dict(self.identity, state=stopped_state)
                with self.assertRaises(RuntimeError):
                    pipeline.replace_waiting_driver(self.directory)
        self.send_signal.assert_not_called()
        self.kill.assert_not_called()

    def test_replacement_refuses_pid_that_does_not_hold_target_pipeline_lock(self):
        self.owns_lock.return_value = False
        with self.assertRaises(RuntimeError):
            pipeline.replace_waiting_driver(self.directory)
        self.send_signal.assert_not_called()
        self.kill.assert_not_called()

    def test_frozen_driver_is_always_resumed_when_revalidation_raises(self):
        self.owns_lock.side_effect = [True, RuntimeError("synthetic lock verification failure")]
        with self.assertRaisesRegex(RuntimeError, "synthetic lock verification failure"):
            pipeline.replace_waiting_driver(self.directory)
        self.assertEqual(self.signals(), [pipeline.signal.SIGSTOP, pipeline.signal.SIGCONT])
        self.assertFalse(self.stopped)
        self.close.assert_called_once_with(self.pidfd)

    def test_replacement_waits_for_old_process_exit_even_when_pipeline_lock_is_available(self):
        self.lock.acquire.side_effect = [False, True]
        original_send = self.send_signal.side_effect
        original_identity = self.read_identity.side_effect
        termination_requested = [False]
        alive_checks = [0]

        def send(fd, signum, *args):
            original_send(fd, signum, *args)
            if signum == pipeline.signal.SIGTERM:
                self.exited = False
                termination_requested[0] = True

        def identity(pid):
            if termination_requested[0] and not self.stopped:
                alive_checks[0] += 1
                self.exited = alive_checks[0] > 2
            return original_identity(pid)

        self.send_signal.side_effect = send
        self.read_identity.side_effect = identity
        pipeline.replace_waiting_driver(self.directory)
        self.assertEqual(self.lock.acquire.call_count, 2)
        self.assertEqual(self.sleep.call_count, 2)
        self.assertEqual(self.signals(), [pipeline.signal.SIGSTOP, pipeline.signal.SIGTERM, pipeline.signal.SIGCONT])

    def test_driver_that_became_nonrunning_after_freeze_is_resumed_without_termination(self):
        original_send = self.send_signal.side_effect

        def send(fd, signum, *args):
            original_send(fd, signum, *args)
            if signum == pipeline.signal.SIGSTOP:
                write_json(self.directory / "state.json", dict(self.state, status="CANCELLED"))

        self.send_signal.side_effect = send
        with self.assertRaises(RuntimeError):
            pipeline.replace_waiting_driver(self.directory)
        self.assertEqual(self.signals(), [pipeline.signal.SIGSTOP, pipeline.signal.SIGCONT])
        self.assertFalse(self.stopped)

    def test_explicit_wait_pid_allows_replacing_only_its_waiting_pipeline_driver(self):
        write_json(self.directory / "state.json", dict(
            self.state, stage="WAIT_RESOURCE_PROCESS",
            resource_process={"pid": 2033835, "start_time": "456", "uid": self.identity["uid"] + 1, "state": "R"},
        ))
        self.lock.acquire.side_effect = [False, True]
        with mock.patch.object(pipeline, "Pipeline") as driver, \
                mock.patch.object(pipeline.logging, "basicConfig"):
            pipeline.main(["--pipeline-dir", str(self.directory), "--wait-pid", "2033835", "--replace-waiter"])
        self.assertEqual(driver.call_args.args[0].wait_pid, 2033835)
        driver.return_value.run.assert_called_once()
        self.assertTrue(all(call.args == (self.pid,) for call in self.read_identity.call_args_list))
        self.assertEqual(self.signals(), [pipeline.signal.SIGSTOP, pipeline.signal.SIGTERM, pipeline.signal.SIGCONT])

    def test_lock_release_timeout_never_constructs_or_starts_a_new_pipeline(self):
        clock = [0.0]

        def now():
            clock[0] += 100
            return clock[0]

        with mock.patch.object(pipeline.time, "monotonic", side_effect=now), \
                mock.patch.object(pipeline, "Pipeline") as driver, \
                mock.patch.object(pipeline.logging, "basicConfig"), \
                mock.patch.object(pipeline.LOGGER, "exception"):
            with self.assertRaises(SystemExit) as caught:
                pipeline.main(["--pipeline-dir", str(self.directory), "--replace-waiter"])
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(self.signals(), [pipeline.signal.SIGSTOP, pipeline.signal.SIGTERM, pipeline.signal.SIGCONT])
        driver.assert_not_called()
        self.lock.release.assert_not_called()

    def test_cancelling_new_launcher_resumes_frozen_old_driver_and_starts_no_work(self):
        handlers = {}
        original_send = self.send_signal.side_effect

        def install(signum, handler):
            previous = handlers.get(signum, pipeline.signal.SIG_DFL)
            handlers[signum] = handler
            return previous

        def send(fd, signum, *args):
            original_send(fd, signum, *args)
            if signum == pipeline.signal.SIGSTOP:
                handlers[pipeline.signal.SIGTERM](pipeline.signal.SIGTERM, None)

        self.send_signal.side_effect = send
        with mock.patch.object(pipeline.signal, "signal", side_effect=install), \
                mock.patch.object(pipeline, "Pipeline") as driver, \
                mock.patch.object(pipeline.logging, "basicConfig"), \
                mock.patch.object(pipeline.LOGGER, "info"):
            with self.assertRaises(SystemExit) as caught:
                pipeline.main(["--pipeline-dir", str(self.directory), "--replace-waiter"])
        self.assertEqual(caught.exception.code, 130)
        self.assertEqual(self.signals(), [pipeline.signal.SIGSTOP, pipeline.signal.SIGCONT])
        self.assertFalse(self.stopped)
        self.close.assert_called_once_with(self.pidfd)
        driver.assert_not_called()
        self.assertTrue(all(handler == pipeline.signal.SIG_DFL for handler in handlers.values()))


class PipelineSupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        patch = mock.patch.object(pipeline, "ROOT", self.root)
        patch.start()
        self.addCleanup(patch.stop)
        options = pipeline.parse_args([
            "--stage1-root", str(self.root / "a"), "--stage2-root", str(self.root / "b"),
            "--pipeline-dir", str(self.root / "pipeline"), "--sim-python", str(self.root / "sim-python"),
        ])
        self.driver = pipeline.Pipeline(options)

    def _wait_for_resource_without_signals_or_command_reads(self):
        with mock.patch.object(pipeline.os, "kill") as kill, \
                mock.patch.object(pipeline.signal, "pidfd_send_signal", create=True) as send, \
                mock.patch.object(pipeline.Path, "read_bytes", return_value=b"") as read_command:
            try:
                self.assertIsNone(self.driver.wait_for_resource_process())
            finally:
                kill.assert_not_called()
                send.assert_not_called()
                read_command.assert_not_called()

    def test_resource_process_from_another_uid_is_only_observed_every_ten_seconds(self):
        identity = {"pid": 2033835, "start_time": "456", "uid": pipeline.os.getuid() + 1, "state": "R"}
        self.driver.options.wait_pid = identity["pid"]
        self.driver.options.check_interval = 10.0
        with mock.patch.object(pipeline, "read_process_identity", side_effect=[identity, identity, None]) as probe, \
                mock.patch.object(self.driver, "_pause") as pause:
            self._wait_for_resource_without_signals_or_command_reads()
        self.assertEqual(probe.call_args_list, [mock.call(2033835)] * 3)
        self.assertEqual(pause.call_args_list, [mock.call(10.0)] * 2)
        self.assertEqual(self.driver.state["resource_process"], identity)
        self.assertEqual(self.driver.state["stage"], "RESOURCE_PROCESS_COMPLETE")

    def test_resource_pid_reuse_owner_change_and_zombie_are_treated_as_original_process_exit(self):
        identity = {"pid": 2033835, "start_time": "456", "uid": pipeline.os.getuid() + 1, "state": "R"}
        self.driver.options.wait_pid = identity["pid"]
        for observed in (None, dict(identity, state="Z"), dict(identity, state="X"),
                         dict(identity, start_time="999"), dict(identity, uid=identity["uid"] + 1)):
            with self.subTest(observed=observed):
                self.driver.state["resource_process"] = identity
                with mock.patch.object(pipeline, "read_process_identity", return_value=observed), \
                        mock.patch.object(self.driver, "_pause") as pause:
                    self._wait_for_resource_without_signals_or_command_reads()
                pause.assert_not_called()
                self.assertEqual(self.driver.state["resource_process"], identity)

    def test_already_exited_resource_pid_does_not_wait_or_require_an_identity(self):
        self.driver.options.wait_pid = 2033835
        with mock.patch.object(pipeline, "read_process_identity", return_value=None), \
                mock.patch.object(self.driver, "_pause") as pause:
            self._wait_for_resource_without_signals_or_command_reads()
        pause.assert_not_called()
        self.assertEqual(self.driver.state["stage"], "RESOURCE_PROCESS_COMPLETE")

    def test_no_resource_pid_skips_process_observation(self):
        self.assertIsNone(self.driver.options.wait_pid)
        with mock.patch.object(pipeline, "read_process_identity") as probe, \
                mock.patch.object(self.driver, "_pause") as pause:
            self._wait_for_resource_without_signals_or_command_reads()
        probe.assert_not_called()
        pause.assert_not_called()

    def test_resource_process_permission_error_is_not_misreported_as_process_exit(self):
        self.driver.options.wait_pid = 2033835
        with mock.patch.object(pipeline, "read_process_identity", side_effect=PermissionError("synthetic /proc denial")), \
                mock.patch.object(self.driver, "_pause") as pause:
            with self.assertRaisesRegex(PermissionError, "synthetic /proc denial"):
                self._wait_for_resource_without_signals_or_command_reads()
        pause.assert_not_called()
        self.assertNotEqual(self.driver.state["stage"], "RESOURCE_PROCESS_COMPLETE")

    def test_resource_identity_survives_restart_only_for_the_same_selected_pid(self):
        identity = {"pid": 2033835, "start_time": "456", "uid": pipeline.os.getuid() + 1, "state": "R"}
        write_json(self.driver.directory / "state.json", {"resource_process": identity})
        for selected, expected in ((2033835, identity), (2033836, None), (None, None)):
            with self.subTest(selected=selected):
                self.driver.options.wait_pid = selected
                restored = pipeline.Pipeline(self.driver.options)
                self.assertEqual(restored.state.get("resource_process"), expected)

    def test_invalid_saved_resource_identity_is_not_reused(self):
        self.driver.options.wait_pid = 2033835
        write_json(self.driver.directory / "state.json", {"resource_process": "invalid identity"})
        restored = pipeline.Pipeline(self.driver.options)
        self.assertNotIn("resource_process", restored.state)

    def test_pipeline_validates_a_then_waits_for_resource_process_before_b_and_evaluations(self):
        a = checkpoint(self.root / "a")
        b = checkpoint(self.root / "b", 4000, mode="alternating", source={"checkpoint": str(a)})
        order = []

        def stage1():
            order.append("validated A")
            return a

        def resource():
            order.append("wait resource")

        def stage2(source):
            self.assertEqual(source, a)
            order.append("train B")
            return b

        def evaluate(name, _checkpoint, _ablation):
            order.append("evaluate " + name)
            return self.root / ("evaluation_" + name)

        with mock.patch.object(self.driver, "preflight"), \
                mock.patch.object(self.driver, "wait_for_training", side_effect=stage1), \
                mock.patch.object(self.driver, "wait_for_resource_process", side_effect=resource), \
                mock.patch.object(self.driver, "train_stage2", side_effect=stage2), \
                mock.patch.object(self.driver, "evaluate", side_effect=evaluate), \
                mock.patch.object(self.driver, "compare", side_effect=lambda _: order.append("compare")):
            self.driver.run()
        self.assertEqual(order, ["validated A", "wait resource", "train B", "evaluate A", "evaluate B",
                                 "evaluate C", "compare"])

    def test_simulator_python_keeps_venv_symlink_path_to_select_its_packages(self):
        target = self.root / "base_python"
        target.write_text("synthetic interpreter")
        link = self.root / "simvenv/bin/python"
        link.parent.mkdir(parents=True)
        link.symlink_to(target)
        options = pipeline.parse_args([
            "--pipeline-dir", str(self.root / "symlink_pipeline"), "--sim-python", str(link),
        ])
        pipeline.Pipeline(options)
        self.assertEqual(options.sim_python, link)
        self.assertTrue(options.sim_python.is_symlink())

    def test_watch_checks_after_two_seconds_then_requires_final_checkpoint(self):
        a = checkpoint(self.root / "a")
        identity = {"pid": 1776511, "start_time": "123", "uid": 1002, "state": "R"}
        self.driver.state["watched_process"] = identity
        with mock.patch.object(pipeline, "read_process_identity", side_effect=[identity, None]), \
                mock.patch.object(self.driver, "_pause") as pause:
            self.assertEqual(self.driver.wait_for_training(), a)
        pause.assert_called_once_with(2.0)
        self.assertEqual(self.driver.state["stage"], "STAGE1_COMPLETE")

    def test_watch_checks_frequently_without_repeating_ten_minute_wait_log(self):
        a = checkpoint(self.root / "a")
        identity = {"pid": 1776511, "start_time": "123", "uid": 1002, "state": "R"}
        self.driver.state["watched_process"] = identity
        clock = [0.0]

        def pause(seconds):
            self.assertEqual(seconds, 2.0)
            clock[0] += seconds

        with mock.patch.object(pipeline, "read_process_identity", side_effect=[identity] * 5 + [None]), \
                mock.patch.object(self.driver, "_pause", side_effect=pause) as sleep, \
                mock.patch.object(pipeline.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(pipeline.LOGGER, "info") as log:
            self.assertEqual(self.driver.wait_for_training(), a)
        self.assertEqual(sleep.call_count, 5)
        wait_logs = [call for call in log.call_args_list if "WAIT_STAGE1" in call.args]
        self.assertEqual(len(wait_logs), 1)

    def test_unchanged_training_wait_is_logged_again_after_ten_minutes(self):
        a = checkpoint(self.root / "a")
        identity = {"pid": 1776511, "start_time": "123", "uid": 1002, "state": "R"}
        self.driver.state["watched_process"] = identity
        self.driver.options.check_interval = 300
        clock = [0.0]

        def pause(seconds):
            clock[0] += seconds

        with mock.patch.object(pipeline, "read_process_identity", side_effect=[identity] * 3 + [None]), \
                mock.patch.object(self.driver, "_pause", side_effect=pause), \
                mock.patch.object(pipeline.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(pipeline.LOGGER, "info") as log:
            self.assertEqual(self.driver.wait_for_training(), a)
        wait_logs = [call for call in log.call_args_list if "WAIT_STAGE1" in call.args]
        self.assertEqual(len(wait_logs), 2)

    def test_watch_process_exiting_between_identity_and_command_read_is_harmless(self):
        a = checkpoint(self.root / "a")
        identity = {"pid": 1776511, "start_time": "123", "uid": pipeline.os.getuid(), "state": "R"}
        for error in (FileNotFoundError("process exited"), ProcessLookupError("process exited")):
            with self.subTest(error=error), \
                    mock.patch.object(pipeline, "read_process_identity", side_effect=[identity, None]), \
                    mock.patch.object(pipeline.Path, "read_bytes", side_effect=error):
                self.assertEqual(self.driver.wait_for_training(), a)

    def test_empty_cmdline_after_process_becomes_zombie_does_not_block_valid_final_weights(self):
        a = checkpoint(self.root / "a")
        identity = {"pid": 1776511, "start_time": "123", "uid": pipeline.os.getuid(), "state": "R"}
        for terminal in (None, dict(identity, state="Z"), dict(identity, state="X"),
                         dict(identity, start_time="999")):
            with self.subTest(terminal=terminal), \
                    mock.patch.object(pipeline, "read_process_identity", side_effect=[identity, terminal]), \
                    mock.patch.object(pipeline.Path, "read_bytes", return_value=b""), \
                    mock.patch.object(self.driver, "_pause") as pause:
                self.assertEqual(self.driver.wait_for_training(), a)
            pause.assert_not_called()

    def test_live_nontraining_pid_is_rejected_even_if_final_checkpoint_exists(self):
        checkpoint(self.root / "a")
        identity = {"pid": 1776511, "start_time": "123", "uid": pipeline.os.getuid(), "state": "R"}
        with mock.patch.object(pipeline, "read_process_identity", return_value=identity), \
                mock.patch.object(pipeline.Path, "read_bytes", return_value=b"python\0unrelated.py\0"):
            with self.assertRaisesRegex(ValueError, "not the T²Mem training process"):
                self.driver.wait_for_training()

    def test_exited_training_without_final_weights_never_starts_stage2(self):
        checkpoint(self.root / "a", 10000, complete=False)
        with mock.patch.object(pipeline, "read_process_identity", return_value=None), \
                mock.patch.object(self.driver, "preflight"), \
                mock.patch.object(self.driver, "train_stage2") as train:
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.driver.run()
        train.assert_not_called()
        self.assertEqual(self.driver.state["status"], "FAILED")

    def test_stage2_resumes_partial_snapshot_but_does_not_overwrite_unrestorable_run(self):
        a = checkpoint(self.root / "a")
        previous = checkpoint(self.root / "b", 1000, mode="alternating", complete=False,
                              source={"checkpoint": str(a)})

        def train(command, **kwargs):
            self.assertEqual(command[command.index("--stage") + 1], "resume")
            self.assertEqual(command[command.index("--resume") + 1], str(previous))
            self.assertNotIn("--init-checkpoint", command)
            checkpoint(self.root / "b", 4000, mode="alternating", source={"checkpoint": str(a)})
            return 0

        with mock.patch.object(self.driver, "wait_for_gpus"), \
                mock.patch.object(self.driver, "run_checked", side_effect=train):
            final = self.driver.train_stage2(a)
        self.assertEqual(final, self.root / "b/4000")
        other = self.root / "unrestorable"
        other.mkdir()
        (other / "partial-file").write_text("interrupted")
        self.driver.options.stage2_root = other
        with mock.patch.object(self.driver, "run_checked") as worker:
            with self.assertRaises(FileExistsError):
                self.driver.train_stage2(a)
        worker.assert_not_called()
        self.assertEqual((other / "partial-file").read_text(), "interrupted")

    def test_fresh_stage2_uses_exact_a_assets_and_initialization(self):
        a = checkpoint(self.root / "a")

        def train(command, **kwargs):
            self.assertEqual(command[command.index("--stage") + 1], "alternating")
            self.assertEqual(command[command.index("--init-checkpoint") + 1], str(a))
            saved = json.loads(Path(command[command.index("--config") + 1]).read_text())
            self.assertEqual(saved["config"]["data"]["norm_stats_dir"], str(a / "assets/robomme"))
            checkpoint(self.root / "b", 4000, mode="alternating", source={"checkpoint": str(a)})
            return 0

        with mock.patch.object(self.driver, "wait_for_gpus") as queue, \
                mock.patch.object(self.driver, "run_checked", side_effect=train):
            self.driver.train_stage2(a)
        queue.assert_called_once_with((1, 2, 3, 4, 5, 6), "TRAIN_STAGE2")

    def test_single_gpu_six_stage2_preserves_global_batch_and_exact_stage1_source(self):
        a = checkpoint(self.root / "a")
        metadata = json.loads((a / "metadata.json").read_text())
        metadata["config"]["training"].update(micro_batch_size=1, stage2_accumulation_steps=32)
        write_json(a / "metadata.json", metadata)
        original = (a / "metadata.json").read_bytes()
        self.driver.options.devices = (6,)

        def train(command, **kwargs):
            self.assertEqual(command[command.index("--num-devices") + 1], "1")
            self.assertEqual(command[command.index("--gpu-id") + 1], "6")
            self.assertEqual(command[command.index("--init-checkpoint") + 1], str(a))
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "6")
            config = json.loads(Path(command[command.index("--config") + 1]).read_text())["config"]
            self.assertEqual(config["training"]["micro_batch_size"], 1)
            self.assertEqual(config["training"]["stage2_accumulation_steps"], 32)
            checkpoint(self.root / "b", 4000, mode="alternating", source={"checkpoint": str(a)})
            return 0

        with mock.patch.object(self.driver, "wait_for_gpus") as queue, \
                mock.patch.object(self.driver, "run_checked", side_effect=train):
            self.assertEqual(self.driver.train_stage2(a), self.root / "b/4000")
        queue.assert_called_once_with((6,), "TRAIN_STAGE2")
        self.assertEqual((a / "metadata.json").read_bytes(), original)

    def test_gpu_zero_through_five_stage2_uses_six_devices_and_exact_stage1_source(self):
        a = checkpoint(self.root / "a")
        self.driver.options.devices = (0, 1, 2, 3, 4, 5)
        self.driver.options.policy_gpu = self.driver.options.sim_gpu = 0

        def train(command, **kwargs):
            self.assertEqual(command[command.index("--num-devices") + 1], "6")
            self.assertEqual(command[command.index("--gpu-id") + 1], "0")
            self.assertEqual(command[command.index("--init-checkpoint") + 1], str(a))
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5")
            checkpoint(self.root / "b", 4000, mode="alternating", source={"checkpoint": str(a)})
            return 0

        with mock.patch.object(self.driver, "wait_for_gpus") as queue, \
                mock.patch.object(self.driver, "run_checked", side_effect=train):
            self.assertEqual(self.driver.train_stage2(a), self.root / "b/4000")
        queue.assert_called_once_with((0, 1, 2, 3, 4, 5), "TRAIN_STAGE2")

    def test_policy_and_simulator_evaluation_both_use_physical_gpu_zero(self):
        a = checkpoint(self.root / "a")
        self.driver.options.devices = (0, 1, 2, 3, 4, 5)
        self.driver.options.policy_gpu = self.driver.options.sim_gpu = 0
        server = mock.Mock(pid=700001)

        def serve(command, **kwargs):
            self.assertIn("scripts/t2mem/serve.py", command)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(command[command.index("--checkpoint") + 1], str(a))
            return server

        def evaluate_command(command, **kwargs):
            self.assertIn("examples/robomme/eval.py", command)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(command[command.index("--sim-mode") + 1], "gpu")
            evaluation(self.driver.directory / "evaluation/A", a)
            return 0

        with mock.patch.object(self.driver, "wait_for_gpus") as queue, \
                mock.patch.object(self.driver, "_check_port_available"), \
                mock.patch.object(self.driver, "start_child", side_effect=serve), \
                mock.patch.object(self.driver, "wait_for_server"), \
                mock.patch.object(self.driver, "run_checked", side_effect=evaluate_command), \
                mock.patch.object(self.driver, "stop_child") as stop:
            self.assertEqual(self.driver.evaluate("A", a, "memory-free"), self.driver.directory / "evaluation/A")
        queue.assert_called_once_with((0,), "EVALUATE_A")
        stop.assert_called_once_with(server)

    def test_gpu_queue_checks_every_two_seconds_without_repeating_wait_log(self):
        clock = [0.0]
        gpu_lock = mock.Mock()
        gpu_lock.acquire.return_value = True
        details = {"policy": "exclusive", "gpus": [{"index": 6, "blocked": True,
                                                    "block_reasons": ["compute process"]}]}

        def pause(seconds):
            self.assertEqual(seconds, 2.0)
            clock[0] += seconds

        with mock.patch.object(pipeline, "gpu_inventory", return_value={6: "GPU-six"}), \
                mock.patch.object(pipeline, "FileLock", return_value=gpu_lock), \
                mock.patch.object(pipeline, "gpu_availability", side_effect=[(False, details)] * 5 + [(True, {})]), \
                mock.patch.object(self.driver, "_pause", side_effect=pause) as sleep, \
                mock.patch.object(pipeline.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(pipeline.LOGGER, "info") as log:
            self.driver.wait_for_gpus((6,), "TRAIN_STAGE2")
        self.assertEqual(sleep.call_count, 5)
        wait_logs = [call for call in log.call_args_list if "WAIT_GPUS" in call.args]
        self.assertEqual(len(wait_logs), 1)
        self.assertEqual(self.driver.state["stage"], "TRAIN_STAGE2")
        self.driver.release_gpus()

    def test_gpu_queue_reports_changed_blocking_pid_immediately_without_logging_memory_jitter(self):
        first = {"gpus": [{"index": 6, "blocked": True, "compute_pids": [123],
                           "free_mib": 1000, "block_reasons": ["compute_processes_present"]}]}
        jitter = copy.deepcopy(first)
        jitter["gpus"][0]["free_mib"] = 1001
        changed = copy.deepcopy(jitter)
        changed["gpus"][0]["compute_pids"] = [456]
        gpu_lock = mock.Mock()
        gpu_lock.acquire.return_value = True
        clock = [0.0]

        def pause(seconds):
            clock[0] += seconds

        with mock.patch.object(pipeline, "gpu_inventory", return_value={6: "GPU-six"}), \
                mock.patch.object(pipeline, "FileLock", return_value=gpu_lock), \
                mock.patch.object(pipeline, "gpu_availability", side_effect=[
                    (False, first), (False, jitter), (False, changed), (True, {}),
                ]), \
                mock.patch.object(self.driver, "_pause", side_effect=pause), \
                mock.patch.object(pipeline.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(pipeline.LOGGER, "info") as log:
            self.driver.wait_for_gpus((6,), "TRAIN_STAGE2")
        wait_logs = [call for call in log.call_args_list if "WAIT_GPUS" in call.args]
        self.assertEqual(len(wait_logs), 2)
        self.driver.release_gpus()

    def test_stage2_completed_by_existing_worker_while_queued_is_reused(self):
        a = checkpoint(self.root / "a")
        checkpoint(self.root / "b", 3000, mode="alternating", complete=False,
                   source={"checkpoint": str(a)})

        def queue(devices, stage):
            checkpoint(self.root / "b", 4000, mode="alternating", source={"checkpoint": str(a)})

        with mock.patch.object(self.driver, "wait_for_gpus", side_effect=queue), \
                mock.patch.object(self.driver, "run_checked") as run, \
                mock.patch.object(self.driver, "release_gpus") as release:
            final = self.driver.train_stage2(a)
        self.assertEqual(final, self.root / "b/4000")
        run.assert_not_called()
        release.assert_called_once()

    def test_failed_stage2_child_marks_pipeline_failed_and_never_evaluates(self):
        a = checkpoint(self.root / "a")
        with mock.patch.object(self.driver, "preflight"), \
                mock.patch.object(self.driver, "wait_for_training", return_value=a), \
                mock.patch.object(self.driver, "wait_for_gpus"), \
                mock.patch.object(self.driver, "run_checked", side_effect=RuntimeError("worker exit 1")), \
                mock.patch.object(self.driver, "evaluate") as evaluate:
            with self.assertRaisesRegex(RuntimeError, "worker exit 1"):
                self.driver.run()
        evaluate.assert_not_called()
        self.assertEqual(self.driver.state["status"], "FAILED")

    def test_evaluation_infrastructure_failures_have_bounded_retries_and_clean_each_server(self):
        a = checkpoint(self.root / "a")
        fake_server = mock.Mock(pid=700001)
        with mock.patch.object(self.driver, "wait_for_gpus"), \
                mock.patch.object(self.driver, "_check_port_available"), \
                mock.patch.object(self.driver, "start_child", return_value=fake_server) as spawn, \
                mock.patch.object(self.driver, "wait_for_server"), \
                mock.patch.object(self.driver, "run_checked", return_value=1) as run, \
                mock.patch.object(self.driver, "stop_child") as stop, \
                mock.patch.object(self.driver, "release_gpus") as release:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                self.driver.evaluate("A", a, "memory-free")
        self.assertEqual(spawn.call_count, 3)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(stop.call_count, 3)
        self.assertEqual(release.call_count, 3)

    def test_stalled_evaluation_retries_only_three_times(self):
        a = checkpoint(self.root / "a")
        fake_server = mock.Mock(pid=700001)
        with mock.patch.object(self.driver, "wait_for_gpus"), \
                mock.patch.object(self.driver, "_check_port_available"), \
                mock.patch.object(self.driver, "start_child", return_value=fake_server), \
                mock.patch.object(self.driver, "wait_for_server"), \
                mock.patch.object(self.driver, "run_checked", side_effect=TimeoutError("no episode progress")) as run, \
                mock.patch.object(self.driver, "stop_child") as stop:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                self.driver.evaluate("A", a, "memory-free")
        self.assertEqual(run.call_count, 3)
        self.assertEqual(stop.call_count, 3)

    def test_live_readiness_helper_metadata_error_is_fatal_despite_open_socket(self):
        a = checkpoint(self.root / "a")
        fake_server = mock.Mock(pid=700001)
        fake_server.poll.return_value = None
        write_json(self.driver.directory / "logs/ready_A.log", {
            "ready": False, "error": "ValueError: The server loaded a different checkpoint",
        })
        with mock.patch.object(pipeline.socket, "create_connection"), \
                mock.patch.object(self.driver, "run_checked", return_value=1):
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                self.driver.wait_for_server(fake_server, a, "memory-free", "A")

    def test_complete_evaluation_is_reused_without_spawning_and_wrong_checkpoint_aborts(self):
        a = checkpoint(self.root / "a")
        directory = evaluation(self.driver.directory / "evaluation/A", a)
        with mock.patch.object(self.driver, "start_child") as spawn:
            self.assertEqual(self.driver.evaluate("A", a, "memory-free"), directory)
        spawn.assert_not_called()
        wrong = checkpoint(self.root / "wrong_a")
        with mock.patch.object(self.driver, "start_child") as spawn:
            with self.assertRaises(ValueError):
                self.driver.evaluate("A", wrong, "memory-free")
        spawn.assert_not_called()

    def test_metadata_readiness_failure_terminates_server_and_does_not_run_evaluator(self):
        a = checkpoint(self.root / "a")
        fake_server = mock.Mock(pid=700001)
        with mock.patch.object(self.driver, "wait_for_gpus"), \
                mock.patch.object(self.driver, "_check_port_available"), \
                mock.patch.object(self.driver, "start_child", return_value=fake_server), \
                mock.patch.object(self.driver, "wait_for_server", side_effect=ValueError("wrong checkpoint")), \
                mock.patch.object(self.driver, "run_checked") as run, \
                mock.patch.object(self.driver, "stop_child") as stop:
            with self.assertRaisesRegex(ValueError, "wrong checkpoint"):
                self.driver.evaluate("A", a, "memory-free")
        run.assert_not_called()
        stop.assert_called_once_with(fake_server)

    def test_compare_nonzero_exit_never_marks_pipeline_complete(self):
        a = checkpoint(self.root / "a")
        b = checkpoint(self.root / "b", 4000, mode="alternating", source={"checkpoint": str(a)})
        with mock.patch.object(self.driver, "preflight"), \
                mock.patch.object(self.driver, "wait_for_training", return_value=a), \
                mock.patch.object(self.driver, "train_stage2", return_value=b), \
                mock.patch.object(self.driver, "evaluate", side_effect=[self.root / "ea", self.root / "eb", self.root / "ec"]), \
                mock.patch.object(self.driver, "run_checked", side_effect=RuntimeError("compare exit 1")):
            with self.assertRaisesRegex(RuntimeError, "compare exit 1"):
                self.driver.run()
        self.assertEqual(self.driver.state["status"], "FAILED")

    def test_stop_child_only_targets_process_groups_registered_by_this_driver(self):
        watched = mock.Mock(pid=1776511)
        owned = mock.Mock(pid=700001)
        owned.poll.return_value = None
        self.driver.children[owned.pid] = owned
        with mock.patch.object(pipeline.os, "killpg") as kill:
            self.driver.stop_child(watched)
            self.driver.stop_child(owned)
        kill.assert_called_once_with(owned.pid, pipeline.signal.SIGTERM)
        watched.poll.assert_not_called()
        self.assertNotIn(owned.pid, self.driver.children)

    def test_worker_failure_is_checked_and_dead_worker_is_removed_without_signal(self):
        child = mock.Mock(pid=700001, returncode=7)
        child.poll.return_value = 7
        self.driver.children[child.pid] = child
        with mock.patch.object(self.driver, "start_child", return_value=child), \
                mock.patch.object(pipeline.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "code 7"):
                self.driver.run_checked(["fake-worker"])
        self.assertNotIn(child.pid, self.driver.children)
        kill.assert_not_called()

    def test_evaluator_watchdog_stops_stalled_worker_without_real_waiting(self):
        child = mock.Mock(pid=700001)
        child.poll.return_value = None
        clock = [0.0]

        def pause(seconds):
            clock[0] += seconds

        with mock.patch.object(self.driver, "start_child", return_value=child), \
                mock.patch.object(self.driver, "_pause", side_effect=pause), \
                mock.patch.object(pipeline.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(pipeline, "progress_signature", return_value=None), \
                mock.patch.object(self.driver, "stop_child") as stop:
            with self.assertRaises(TimeoutError):
                self.driver.run_checked(["fake-evaluator"], progress_path=self.root / "episodes.json", idle_timeout=3)
        self.assertLessEqual(clock[0], 4)
        stop.assert_called_once_with(child)

    def test_evaluator_watchdog_resets_deadline_when_episode_progress_advances(self):
        child = mock.Mock(pid=700001, returncode=0)
        child.poll.side_effect = [None, None, None, None, 0]
        clock = [0.0]

        def pause(seconds):
            clock[0] += seconds

        with mock.patch.object(self.driver, "start_child", return_value=child), \
                mock.patch.object(self.driver, "_pause", side_effect=pause), \
                mock.patch.object(pipeline.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(pipeline, "progress_signature", side_effect=lambda _: (int(clock[0]), 1)), \
                mock.patch.object(self.driver, "stop_child") as stop:
            code = self.driver.run_checked(["fake-evaluator"], progress_path=self.root / "episodes.json", idle_timeout=2)
        self.assertEqual(code, 0)
        self.assertGreater(clock[0], 2)
        stop.assert_called_once_with(child)

    def test_policy_server_exit_terminates_its_evaluator_before_any_retry(self):
        child = mock.Mock(pid=700001)
        child.poll.return_value = None
        server = mock.Mock(pid=700002)
        server.poll.return_value = 1
        with mock.patch.object(self.driver, "start_child", return_value=child), \
                mock.patch.object(self.driver, "stop_child") as stop:
            with self.assertRaises(pipeline.WorkerDependencyExited):
                self.driver.run_checked(["fake-evaluator"], supervised_server=server)
        stop.assert_called_once_with(child)

    def test_same_pipeline_lock_rejects_second_driver_without_overwriting_status(self):
        first = pipeline.FileLock(self.driver.directory / "pipeline.lock")
        self.assertTrue(first.acquire())
        self.addCleanup(first.release)
        self.driver.update("WAIT_STAGE1", status="RUNNING")
        previous = (self.driver.directory / "state.json").read_bytes()
        with mock.patch.object(self.driver, "preflight") as preflight:
            with self.assertRaisesRegex(RuntimeError, "Another pipeline"):
                self.driver.run()
        preflight.assert_not_called()
        self.assertEqual((self.driver.directory / "state.json").read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
