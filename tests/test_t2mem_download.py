"""Server-side cache regression test, using a fake filesystem without network."""

from pathlib import Path
import runpy
import sys

import pytest

from openpi.shared import download


def test_interrupted_weight_download_is_not_cached_and_can_be_retried(tmp_path, monkeypatch):
    """A worker failure must propagate before partial weights become a cache hit."""
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    destination = tmp_path / "t2mem-download-fixture" / "weights.bin"
    destination.parent.mkdir(parents=True)

    class FileSystem:
        fail = True
        get_calls = 0

        def info(self, url):
            return {"type": "file", "size": 8, "name": url}

        def get(self, url, local_path, *, recursive):
            assert url == "gs://t2mem-download-fixture/weights.bin"
            assert not recursive
            self.get_calls += 1
            Path(local_path).write_bytes(b"partial" if self.fail else b"complete")
            if self.fail:
                raise OSError("interrupted transfer")

    filesystem = FileSystem()
    monkeypatch.setattr(download.fsspec.core, "url_to_fs", lambda url, **kwargs: (filesystem, url))
    with pytest.raises(OSError, match="interrupted transfer"):
        download.maybe_download("gs://t2mem-download-fixture/weights.bin")
    assert not destination.exists()
    assert destination.with_suffix(".partial").read_bytes() == b"partial"

    filesystem.fail = False
    result = download.maybe_download("gs://t2mem-download-fixture/weights.bin")
    assert result == destination
    assert result.read_bytes() == b"complete"
    assert not destination.with_suffix(".partial").exists()
    assert download.maybe_download("gs://t2mem-download-fixture/weights.bin") == destination
    assert filesystem.get_calls == 2


@pytest.fixture
def remote_directory(tmp_path, monkeypatch):
    """A canonical fsspec path and a fake recursive transfer, without network."""
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    url = "gs://t2mem-download-fixture/checkpoint"
    canonical = "t2mem-download-fixture/checkpoint"
    destination = tmp_path / canonical

    class FileSystem:
        remote = {"params/a": b"1234", "params/b": b"123456"}
        transferred = None
        get_calls = 0
        destination_existed = []

        def info(self, path):
            assert path == canonical
            return {"type": "directory", "size": 0, "name": canonical}

        def find(self, path, *, detail, withdirs):
            assert path == canonical and detail and not withdirs
            return {
                f"{canonical}/{relative}": {"name": f"{canonical}/{relative}", "type": "file", "size": len(data)}
                for relative, data in self.remote.items()
            }

        def get(self, path, local_path, *, recursive):
            assert path == canonical and recursive
            self.get_calls += 1
            target = Path(local_path)
            self.destination_existed.append(target.exists())
            if target.is_dir():
                # Matches the problematic retry layout of fsspec recursive get
                # into an existing directory. Staging must be removed first.
                target = target / "checkpoint"
            for relative, data in (self.remote if self.transferred is None else self.transferred).items():
                file = target / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(data)

    filesystem = FileSystem()
    monkeypatch.setattr(download.fsspec.core, "url_to_fs", lambda requested, **kwargs: (filesystem, canonical))
    return url, filesystem, destination


def test_directory_download_verifies_canonical_relative_files_and_sizes(remote_directory):
    url, filesystem, destination = remote_directory
    result = download.maybe_download(url)
    assert result == destination
    actual = {path.relative_to(result).as_posix(): path.read_bytes() for path in result.rglob("*") if path.is_file()}
    assert actual == filesystem.remote
    assert filesystem.get_calls == 1
    assert filesystem.destination_existed == [False]
    # Verified cache hits perform metadata checks without transferring again.
    assert download.maybe_download(url, verify_cache=True) == destination
    assert filesystem.get_calls == 1


def test_gcs_file_typed_directory_markers_are_ignored_but_empty_files_are_verified(remote_directory, monkeypatch):
    """gcsfs metadata can retain root and child markers despite withdirs=False."""
    url, filesystem, destination = remote_directory
    canonical = "t2mem-download-fixture/checkpoint"
    filesystem.remote = {**filesystem.remote, "params/empty-file": b""}
    original_find = filesystem.find

    def gcs_info(path):
        assert path == canonical
        return {"name": f"{canonical}/", "type": "file", "size": 0}

    def gcs_find(path, *, detail, withdirs):
        listing = original_find(path, detail=detail, withdirs=withdirs)
        for name in (f"{canonical}/", f"{canonical}/params/", f"{canonical}/params/empty-directory/"):
            listing[name] = {"name": name, "type": "file", "size": 0}
        return listing

    monkeypatch.setattr(filesystem, "info", gcs_info)
    monkeypatch.setattr(filesystem, "find", gcs_find)
    result = download.maybe_download(url)
    assert result == destination
    actual = {path.relative_to(result).as_posix(): path.read_bytes() for path in result.rglob("*") if path.is_file()}
    assert actual == filesystem.remote
    assert (result / "params/empty-file").is_file()
    assert download.maybe_download(url, verify_cache=True) == result
    assert filesystem.get_calls == 1


@pytest.mark.parametrize("corruption", ["short", "missing", "extra", "same-total"])
def test_successful_directory_worker_cannot_promote_corrupt_files_and_retry_is_clean(remote_directory, corruption):
    url, filesystem, destination = remote_directory
    bad = dict(filesystem.remote)
    if corruption == "short":
        bad["params/a"] = b"12"
    elif corruption == "missing":
        bad.pop("params/a")
    elif corruption == "extra":
        bad["orphan"] = b"unexpected"
    else:
        bad = {"params/a": b"12345", "params/b": b"12345"}
        assert sum(map(len, bad.values())) == sum(map(len, filesystem.remote.values()))
    filesystem.transferred = bad
    with pytest.raises(download.DownloadIntegrityError, match="integrity failure") as error:
        download.maybe_download(url)
    if corruption == "missing":
        assert "missing=['params/a']" in str(error.value)
    elif corruption == "extra":
        assert "unexpected=['orphan']" in str(error.value)
    else:
        assert "params/a" in str(error.value) and "expected bytes, actual bytes" in str(error.value)
    assert not destination.exists()
    assert destination.with_suffix(".partial").is_dir()
    filesystem.transferred = None
    assert download.maybe_download(url) == destination
    assert filesystem.destination_existed == [False, False]
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    assert actual == set(filesystem.remote)
    assert not destination.with_suffix(".partial").exists()


def test_short_file_with_normal_worker_return_is_not_cached_and_can_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    url = "gs://t2mem-download-fixture/weights.bin"
    canonical = "t2mem-download-fixture/weights.bin"

    class FileSystem:
        payload = b"bad"

        def info(self, path):
            assert path == canonical
            return {"type": "file", "size": 8, "name": canonical}

        def get(self, path, local_path, *, recursive):
            assert path == canonical and not recursive
            Path(local_path).write_bytes(self.payload)

    filesystem = FileSystem()
    monkeypatch.setattr(download.fsspec.core, "url_to_fs", lambda requested, **kwargs: (filesystem, canonical))
    destination = tmp_path / canonical
    with pytest.raises(download.DownloadIntegrityError, match="8, 3"):
        download.maybe_download(url)
    assert not destination.exists()
    assert destination.with_suffix(".partial").read_bytes() == b"bad"
    filesystem.payload = b"complete"
    assert download.maybe_download(url).read_bytes() == b"complete"


def test_old_bad_cache_is_reported_without_deletion_and_force_repairs_it(remote_directory):
    url, filesystem, destination = remote_directory
    (destination / "params").mkdir(parents=True)
    (destination / "params/a").write_bytes(b"short")
    (destination / "params/b").write_bytes(filesystem.remote["params/b"])
    with pytest.raises(download.DownloadIntegrityError, match="force_download=True"):
        download.maybe_download(url, verify_cache=True)
    assert filesystem.get_calls == 0
    assert (destination / "params/a").read_bytes() == b"short"
    repaired = download.maybe_download(url, force_download=True, verify_cache=True)
    assert repaired == destination
    assert (destination / "params/a").read_bytes() == filesystem.remote["params/a"]
    assert filesystem.get_calls == 1


def test_normal_cache_fast_path_needs_no_remote_connection(remote_directory, monkeypatch):
    url, _, destination = remote_directory
    download.maybe_download(url)

    def offline(*args, **kwargs):
        raise AssertionError("Normal cache hits must preserve the offline fast path")

    monkeypatch.setattr(download.fsspec.core, "url_to_fs", offline)
    assert download.maybe_download(url) == destination


def test_stale_partial_directory_is_removed_before_recursive_transfer(remote_directory):
    url, filesystem, destination = remote_directory
    scratch = destination.with_suffix(".partial")
    (scratch / "stale").mkdir(parents=True)
    (scratch / "stale/leftover").write_bytes(b"previous failed transfer")
    result = download.maybe_download(url, force_download=True)
    assert result == destination
    assert filesystem.destination_existed == [False]
    assert not (result / "stale").exists()
    assert not (result / "checkpoint").exists()


def test_caller_filesystem_options_are_preserved(remote_directory, monkeypatch):
    url, filesystem, _ = remote_directory
    calls = []

    def resolve(requested, **kwargs):
        assert requested == url
        calls.append(kwargs)
        return filesystem, "t2mem-download-fixture/checkpoint"

    monkeypatch.setattr(download.fsspec.core, "url_to_fs", resolve)
    options = {"token": "caller-token", "consistency": "size"}
    download.maybe_download(url, gs=options)
    assert calls == [{"gs": options}]


@pytest.mark.parametrize("force", [False, True])
def test_official_downloader_requests_cache_verification_and_anonymous_size_check(tmp_path, monkeypatch, force):
    calls = []

    def download_stub(url, **kwargs):
        calls.append((url, kwargs))
        return tmp_path / "verified-cache"

    monkeypatch.setattr(download, "maybe_download", download_stub)
    script = Path(__file__).resolve().parents[1] / "scripts/download_pi05_base.py"
    monkeypatch.setattr(sys, "argv", [str(script), *(["--force"] if force else [])])
    runpy.run_path(str(script), run_name="__main__")
    assert calls == [
        (
            "gs://openpi-assets/checkpoints/pi05_base",
            {
                "force_download": force,
                "verify_cache": True,
                "gs": {"token": "anon", "consistency": "size"},
            },
        )
    ]
