"""Server-side cache regression test, using a fake filesystem without network."""

from pathlib import Path

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
