import concurrent.futures
import datetime
import logging
import os
import pathlib
import re
import shutil
import stat
import time
import urllib.parse

import filelock
import fsspec
import fsspec.generic
import tqdm_loggable.auto as tqdm

# Environment variable to control cache directory path, ~/.cache/openpi will be used by default.
_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"
DEFAULT_CACHE_DIR = "~/.cache/openpi"

logger = logging.getLogger(__name__)


class DownloadIntegrityError(RuntimeError):
    """A transferred or cached entry does not match its remote file manifest."""


def get_cache_dir() -> pathlib.Path:
    cache_dir = pathlib.Path(os.getenv(_OPENPI_DATA_HOME, DEFAULT_CACHE_DIR)).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def maybe_download(url: str, *, force_download: bool = False, verify_cache: bool = False, **kwargs) -> pathlib.Path:
    """Download a file or directory from a remote filesystem to the local cache, and return the local path.

    New transfers are checked against remote filenames and byte sizes before
    becoming cache entries. Existing entries retain the offline fast path
    unless verify_cache is requested.

    It is safe to call this function concurrently from multiple processes.
    See `get_cache_dir` for more details on the cache directory.

    Args:
        url: URL to the file to download.
        force_download: If True, the file will be downloaded even if it already exists in the cache.
        verify_cache: Check an existing remote cache entry against the remote
            file manifest. A mismatch raises DownloadIntegrityError; explicitly
            set force_download=True to replace it. Requires network metadata.
        **kwargs: Additional arguments to pass to fsspec.

    Returns:
        Local path to the downloaded file or directory. That path is guaranteed to exist and is absolute.
    """
    # Don't use fsspec to parse the url to avoid unnecessary connection to the remote filesystem.
    parsed = urllib.parse.urlparse(url)

    # Short circuit if this is a local path.
    if parsed.scheme == "":
        path = pathlib.Path(url)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return path.resolve()

    cache_dir = get_cache_dir()

    local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
    local_path = local_path.resolve()

    if (
        local_path.exists()
        and not force_download
        and not verify_cache
        and not _should_invalidate_cache(cache_dir, local_path)
    ):
        return local_path

    try:
        lock_path = local_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with filelock.FileLock(lock_path):
            # Ensure consistent permissions for the lock file.
            _ensure_permissions(lock_path)
            # Recheck after acquiring the lock: another process may have
            # completed this entry while this caller was waiting.
            if local_path.exists():
                if force_download or _should_invalidate_cache(cache_dir, local_path):
                    logger.info("Replacing cached entry: %s", local_path)
                    _remove_local_entry(local_path)
                else:
                    if verify_cache:
                        fs, remote_path = fsspec.core.url_to_fs(url, **kwargs)
                        is_dir, files = _remote_file_manifest(fs, remote_path)
                        _verify_local_entry(url, local_path, is_dir, files)
                    return local_path

            # Download the data to a local cache.
            logger.info(f"Downloading {url} to {local_path}")
            scratch_path = local_path.with_suffix(".partial")
            # Existing recursive destinations can make fsspec nest the source
            # directory inside them. A retry starts from a clean staging path.
            if scratch_path.exists() or scratch_path.is_symlink():
                _remove_local_entry(scratch_path)
            _download_fsspec(url, scratch_path, **kwargs)

            shutil.move(scratch_path, local_path)
            _ensure_permissions(local_path)

    except PermissionError as e:
        msg = (
            f"Local file permission error was encountered while downloading {url}. "
            f"Please try again after removing the cached data using: `rm -rf {local_path}*`"
        )
        raise PermissionError(msg) from e

    return local_path


def _remove_local_entry(path: pathlib.Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _remote_file_manifest(fs, remote_path: str) -> tuple[bool, dict[str, int]]:
    """Snapshot canonical relative filenames and sizes from fsspec metadata."""
    info = fs.info(remote_path)
    is_dir = info["type"] == "directory" or (info.get("size") == 0 and info["name"].endswith("/"))

    def size(metadata):
        value = metadata.get("size")
        if type(value) is not int or value < 0:
            raise DownloadIntegrityError(f"Remote byte size is unavailable for {metadata.get('name', remote_path)}")
        return value

    if not is_dir:
        return False, {"": size(info)}
    root = pathlib.PurePosixPath(remote_path.rstrip("/"))
    files = {}
    for name, metadata in fs.find(remote_path, detail=True, withdirs=False).items():
        # GCS reports zero-byte trailing-slash directory markers as files,
        # including the requested root, even when find excludes directories.
        if metadata.get("type") == "directory" or (
            metadata.get("size") == 0 and metadata.get("name", name).endswith("/")
        ):
            continue
        try:
            relative = pathlib.PurePosixPath(name).relative_to(root)
        except ValueError as error:
            raise DownloadIntegrityError(f"Remote file is outside the requested directory: {name}") from error
        if not relative.parts or ".." in relative.parts:
            raise DownloadIntegrityError(f"Invalid remote relative filename: {name}")
        files[relative.as_posix()] = size(metadata)
    return True, files


def _verify_local_entry(url: str, local_path: pathlib.Path, is_dir: bool, files: dict[str, int]) -> None:
    """Reject missing, extra, or short files, even after a normal worker return."""
    if local_path.is_symlink() or (is_dir and not local_path.is_dir()) or (not is_dir and not local_path.is_file()):
        raise DownloadIntegrityError(f"Download integrity failure for {url}: wrong or missing local entry {local_path}")
    if is_dir:
        actual = {}
        for path in local_path.rglob("*"):
            if path.is_symlink():
                raise DownloadIntegrityError(f"Download integrity failure for {url}: unexpected symlink {path}")
            if path.is_file():
                actual[path.relative_to(local_path).as_posix()] = path.stat().st_size
    else:
        actual = {"": local_path.stat().st_size}
    missing = sorted(files.keys() - actual.keys())
    extra = sorted(actual.keys() - files.keys())
    mismatches = [
        (name or local_path.name, files[name], actual[name])
        for name in sorted(files.keys() & actual.keys())
        if files[name] != actual[name]
    ]
    if missing or extra or mismatches:
        raise DownloadIntegrityError(
            f"Download integrity failure for {url}: missing={missing[:5]}, unexpected={extra[:5]}, "
            f"size mismatches (file, expected bytes, actual bytes)={mismatches[:5]}. "
            "Fix disk/network problems, then use force_download=True to replace this cache."
        )


def _download_fsspec(url: str, local_path: pathlib.Path, **kwargs) -> None:
    """Transfer into staging and verify each file before cache promotion."""
    fs, remote_path = fsspec.core.url_to_fs(url, **kwargs)
    is_dir, files = _remote_file_manifest(fs, remote_path)
    total_size = sum(files.values())
    with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fs.get, remote_path, local_path, recursive=is_dir)
            while not future.done():
                current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
                pbar.update(current_size - pbar.n)
                time.sleep(1)
            # Worker exceptions and successful-but-short transfers must both
            # fail before progress completes or staging becomes a cache hit.
            future.result()
        _verify_local_entry(url, local_path, is_dir, files)
        pbar.update(total_size - pbar.n)


def _set_permission(path: pathlib.Path, target_permission: int):
    """chmod requires executable permission to be set, so we skip if the permission is already match with the target."""
    if path.stat().st_mode & target_permission == target_permission:
        logger.debug(f"Skipping {path} because it already has correct permissions")
        return
    path.chmod(target_permission)
    logger.debug(f"Set {path} to {target_permission}")


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    """Set folder permission to be read, write and searchable."""
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path) -> None:
    """Since we are sharing cache directory with containerized runtime as well as training script, we need to
    ensure that the cache directory has the correct permissions.
    """

    def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
        cache_dir = get_cache_dir()
        relative_path = path.relative_to(cache_dir)
        moving_path = cache_dir
        for part in relative_path.parts:
            _set_folder_permission(moving_path / part)
            moving_path = moving_path / part

    def _set_file_permission(file_path: pathlib.Path) -> None:
        """Set all files to be read & writable, if it is a script, keep it as a script."""
        file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        if file_path.stat().st_mode & 0o100:
            _set_permission(file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            _set_permission(file_path, file_rw)

    _setup_folder_permission_between_cache_dir_and_path(path)
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for file in files:
            file_path = root_path / file
            _set_file_permission(file_path)

        for dir in dirs:
            dir_path = root_path / dir
            _set_folder_permission(dir_path)


def _get_mtime(year: int, month: int, day: int) -> float:
    """Get the mtime of a given date at midnight UTC."""
    date = datetime.datetime(year, month, day, tzinfo=datetime.UTC)
    return time.mktime(date.timetuple())


# Map of relative paths, defined as regular expressions, to expiration timestamps (mtime format).
# Partial matching will be used from top to bottom and the first match will be chosen.
# Cached entries will be retained only if they are newer than the expiration timestamp.
_INVALIDATE_CACHE_DIRS: dict[re.Pattern, float] = {
    re.compile("openpi-assets/checkpoints/pi0_aloha_pen_uncap"): _get_mtime(2025, 2, 17),
    re.compile("openpi-assets/checkpoints/pi0_libero"): _get_mtime(2025, 2, 6),
    re.compile("openpi-assets/checkpoints/"): _get_mtime(2025, 2, 3),
}


def _should_invalidate_cache(cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
    """Invalidate the cache if it is expired. Return True if the cache was invalidated."""

    assert local_path.exists(), f"File not found at {local_path}"

    relative_path = str(local_path.relative_to(cache_dir))
    for pattern, expire_time in _INVALIDATE_CACHE_DIRS.items():
        if pattern.match(relative_path):
            # Remove if not newer than the expiration timestamp.
            return local_path.stat().st_mtime <= expire_time

    return False
