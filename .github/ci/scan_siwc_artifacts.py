"""Fail closed when SIWC credentials appear in source or built artifacts."""
from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import lzma
import os
import re
import stat
import struct
import sys
import tarfile
import zipfile
import zlib
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import IO, Literal, Protocol, cast

_CHUNK_BYTES = 64 * 1024
_MAX_SENTINEL_BYTES = 1_024
_MAX_SENTINEL_COUNT = 128
_MAX_SENTINEL_AGGREGATE_BYTES = 32 * 1024
_MAX_SENTINEL_FILE_BYTES = 64 * 1024
_MAX_METADATA_FIELD_BYTES = 16 * 1024
_MAX_METADATA_TOTAL_BYTES = 256 * 1024
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_ARCHIVE_CONTAINER_BYTES = 128 * 1024 * 1024
_MAX_ZIP_DIRECTORY_BYTES = 16 * 1024 * 1024
_MAX_ZIP_DETECTION_TAIL_BYTES = 263 * 1024
_MAX_ARCHIVE_MEMBER_BYTES = 256 * 1024 * 1024
_MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_WORK_BYTES = 768 * 1024 * 1024
_MAX_NESTED_ARCHIVE_BYTES = 32 * 1024 * 1024
_MAX_ARCHIVE_DEPTH = 3
_MAX_TRAVERSAL_DEPTH = 64
_MAX_TRAVERSAL_ENTRIES = 200_000
_MAX_SCANNED_FILES = 100_000
_MAX_TOP_LEVEL_RAW_BYTES = 2 * 1024 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 200
_RATIO_CHECK_MIN_BYTES = 1 * 1024 * 1024
_SKIP_PARTS = frozenset({
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "__pycache__", "node_modules",
})
# Prefix matches intentionally omit a trailing word boundary. This lets the
# streaming scanner reject a credential as soon as its minimum identifying
# prefix has arrived, without retaining an attacker-sized token in memory.
_PATTERNS = (
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20}"),
    re.compile(rb"\bBearer\s+[A-Za-z0-9._~-]{24}", re.IGNORECASE),
    re.compile(rb"\beyJ[A-Za-z0-9_-]{12}\.[A-Za-z0-9_-]{12}\.[A-Za-z0-9_-]{12}"),
    re.compile(rb"-----BEGIN " rb"PRIVATE KEY-----"),
)
_PATTERN_OVERLAP = 128
_ARCHIVE_SUFFIXES = (
    ".zip", ".whl", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz",
    ".tbz2", ".tar.xz", ".txz", ".tar.zst", ".tar.zstd", ".tzst",
)
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06")
_ZIP_EOCD_MAGIC = b"PK\x05\x06"
_ZIP_CENTRAL_MAGIC = b"PK\x01\x02"
_ZIP64_MAGICS = (b"PK\x06\x06", b"PK\x06\x07")
_BZIP2_MAGIC = b"BZh"
_XZ_MAGIC = b"\xfd7zXZ\x00"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class ScanError(RuntimeError):
    """An artifact could not be completely and safely scanned."""


def _is_zstd_magic(prefix: bytes) -> bool:
    if prefix.startswith(_ZSTD_MAGIC):
        return True
    return (
        len(prefix) >= 4
        and 0x184D2A50 <= int.from_bytes(prefix[:4], "little") <= 0x184D2A5F
    )


def _is_lzma_alone_magic(prefix: bytes) -> bool:
    if len(prefix) < 13 or prefix[0] >= 225:
        return False
    declared_size = int.from_bytes(prefix[5:13], "little")
    return (
        declared_size == 0xFFFFFFFFFFFFFFFF
        or declared_size <= _MAX_ARCHIVE_TOTAL_BYTES
    )


class _Hasher(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def digest(self) -> bytes: ...


@dataclass
class _Budget:
    traversal_entries: int = 0
    files: int = 0
    top_level_raw_bytes: int = 0
    members: int = 0
    expanded_bytes: int = 0
    work_bytes: int = 0
    metadata_bytes: int = 0

    def add_traversal(self, label: str) -> None:
        self.traversal_entries += 1
        if self.traversal_entries > _MAX_TRAVERSAL_ENTRIES:
            raise ScanError(f"{label}: filesystem traversal count exceeds safety limit")

    def add_file(self, size: int, label: str) -> None:
        self.files += 1
        if self.files > _MAX_SCANNED_FILES:
            raise ScanError(f"{label}: scanned file count exceeds safety limit")
        if size < 0:
            raise ScanError(f"{label}: regular file has an invalid size")
        self.top_level_raw_bytes += size
        if self.top_level_raw_bytes > _MAX_TOP_LEVEL_RAW_BYTES:
            raise ScanError(f"{label}: top-level raw bytes exceed safety limit")

    def add_member(self, size: int, compressed_size: int | None, label: str) -> None:
        self.members += 1
        if self.members > _MAX_ARCHIVE_MEMBERS:
            raise ScanError(f"{label}: archive member count exceeds safety limit")
        if size < 0 or size > _MAX_ARCHIVE_MEMBER_BYTES:
            raise ScanError(f"{label}: archive member size exceeds safety limit")
        self.expanded_bytes += size
        if self.expanded_bytes > _MAX_ARCHIVE_TOTAL_BYTES:
            raise ScanError(f"{label}: archive expanded size exceeds safety limit")
        if (
            compressed_size is not None
            and size > _RATIO_CHECK_MIN_BYTES
            and size > max(1, compressed_size) * _MAX_COMPRESSION_RATIO
        ):
            raise ScanError(f"{label}: archive compression ratio exceeds safety limit")

    def add_work(self, size: int, label: str) -> None:
        self.work_bytes += size
        if self.work_bytes > _MAX_ARCHIVE_WORK_BYTES:
            raise ScanError(f"{label}: archive scan work exceeds safety limit")

    def add_expanded_stream(self, size: int, label: str) -> None:
        self.expanded_bytes += size
        if self.expanded_bytes > _MAX_ARCHIVE_TOTAL_BYTES:
            raise ScanError(f"{label}: archive expanded size exceeds safety limit")

    def add_metadata(self, size: int, label: str) -> None:
        self.metadata_bytes += size
        if self.metadata_bytes > _MAX_METADATA_TOTAL_BYTES:
            raise ScanError(f"{label}: archive metadata exceeds aggregate safety limit")


def _contains_secret(
    data: bytes, sentinels: tuple[bytes, ...], *, sentinels_only: bool
) -> bool:
    return (
        not sentinels_only and any(pattern.search(data) for pattern in _PATTERNS)
    ) or any(sentinel in data for sentinel in sentinels)


def _safe_component(
    value: object, sentinels: tuple[bytes, ...] = (), *, sentinels_only: bool = False
) -> str:
    raw = str(value)
    encoded = raw.encode("utf-8", errors="backslashreplace")
    # Detection mode can ignore generic patterns, but labels must never echo
    # either a runtime sentinel or generic credential material.
    if _contains_secret(encoded, sentinels, sentinels_only=False):
        return "<redacted>"
    escaped = "".join(
        character
        if character.isprintable() and character not in {"\r", "\n"}
        else f"\\x{ord(character):02x}"
        for character in raw
    )
    return escaped if len(escaped) <= 500 else escaped[:497] + "..."


def _safe_label(value: object) -> str:
    return _safe_component(value)


def _display_context(
    context: tuple[object, ...],
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
) -> str:
    return "!".join(
        _safe_component(part, sentinels, sentinels_only=sentinels_only)
        for part in context
    )


def _fail(path: object, reason: str) -> ScanError:
    return ScanError(f"{_safe_label(path)}: {reason}")


def _fail_context(
    context: tuple[object, ...],
    reason: str,
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
) -> ScanError:
    return ScanError(
        f"{_display_context(context, sentinels, sentinels_only=sentinels_only)}: {reason}"
    )


def _archive_kind(name: str, prefix: bytes) -> str | None:
    lowered = name.casefold()
    if prefix.startswith(_ZIP_MAGICS):
        return "zip"
    if _is_zstd_magic(prefix):
        return "tar-zstd"
    if len(prefix) >= 263 and prefix[257:262] == b"ustar":
        return "tar"
    if _is_lzma_alone_magic(prefix):
        return "tar-lzma-alone"
    if prefix.startswith(b"\x1f\x8b"):
        return "tar-gzip"
    if prefix.startswith(_BZIP2_MAGIC):
        return "tar-bzip2"
    if prefix.startswith(_XZ_MAGIC):
        return "tar-xz"
    if lowered.endswith((".zip", ".whl")):
        return "zip"
    if lowered.endswith((".tar.zst", ".tar.zstd", ".tzst")):
        return "tar-zstd"
    if lowered.endswith((".tar.gz", ".tgz")):
        return "tar-gzip"
    if lowered.endswith((".tar.bz2", ".tbz", ".tbz2")):
        return "tar-bzip2"
    if lowered.endswith((".tar.xz", ".txz")):
        return "tar-xz"
    if lowered.endswith(".tar"):
        return "tar"
    return None


def _zip_tail_claims_archive(data: bytes) -> bool:
    """Recognize an EOCD-terminated ZIP without trusting filename or offset fields."""
    end_offset = data.rfind(_ZIP_EOCD_MAGIC)
    if end_offset < 0 or end_offset + 22 > len(data):
        return False
    comment_size = struct.unpack_from("<H", data, end_offset + 20)[0]
    exact_end = end_offset + 22 + comment_size == len(data)
    entry_count = struct.unpack_from("<H", data, end_offset + 10)[0]
    if entry_count == 0:
        return exact_end
    # Once a non-empty central directory and terminal EOCD are both present,
    # malformed EOCD lengths still claim archive structure and must fail closed.
    return data.rfind(_ZIP_CENTRAL_MAGIC, 0, end_offset) >= 0


def _stream_claims_zip(stream: IO[bytes], size: int, label: str) -> bool:
    tail_size = min(size, _MAX_ZIP_DIRECTORY_BYTES + 65_557)
    try:
        stream.seek(size - tail_size)
        tail = _read_chunk(stream, tail_size, label)
    except Exception as exc:
        raise ScanError(f"{label}: regular file is not seekable") from exc
    if len(tail) != tail_size:
        raise ScanError(f"{label}: regular file changed while being scanned")
    return _zip_tail_claims_archive(tail)


def _validate_member_name(
    name: str,
    context: tuple[object, ...],
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
) -> None:
    pure = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise _fail_context(
            context,
            "archive contains an unsafe member path",
            sentinels,
            sentinels_only=sentinels_only,
        )


def _read_chunk(
    stream: IO[bytes],
    size: int,
    label: str,
    budget: _Budget | None = None,
) -> bytes:
    try:
        chunk = stream.read(size)
    except Exception as exc:
        raise ScanError(f"{label}: stream became unreadable") from exc
    if not isinstance(chunk, bytes):
        raise ScanError(f"{label}: stream returned non-byte content")
    if budget is not None:
        budget.add_work(len(chunk), label)
    return chunk


def _scan_window(
    data: bytes,
    overlap: bytes,
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    overlap_size: int,
) -> tuple[bool, bytes]:
    window = overlap + data
    found = _contains_secret(window, sentinels, sentinels_only=sentinels_only)
    return found, window[-overlap_size:] if overlap_size else b""


def _consume_stream(
    stream: IO[bytes],
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    label: str,
    expected_size: int | None = None,
    budget: _Budget | None = None,
    initial: bytes = b"",
    capture_limit: int | None = None,
    tail_limit: int = 0,
    hasher: _Hasher | None = None,
) -> tuple[bool, bytes | None, bytes]:
    overlap_size = max(
        _PATTERN_OVERLAP if not sentinels_only else 0,
        max((len(sentinel) - 1 for sentinel in sentinels), default=0),
    )
    overlap = b""
    found = False
    total = 0
    captured = bytearray() if capture_limit is not None else None
    tail = bytearray()

    chunk = initial
    while True:
        if chunk:
            if hasher is not None:
                hasher.update(chunk)
            total += len(chunk)
            if expected_size is not None and total > expected_size:
                raise ScanError(f"{label}: archive member exceeded its declared size")
            if captured is not None:
                assert capture_limit is not None
                if len(captured) + len(chunk) > capture_limit:
                    raise ScanError(f"{label}: nested archive exceeds in-memory safety limit")
                captured.extend(chunk)
            if tail_limit:
                tail.extend(chunk)
                if len(tail) > tail_limit:
                    del tail[: len(tail) - tail_limit]
            matched, overlap = _scan_window(
                chunk,
                overlap,
                sentinels,
                sentinels_only=sentinels_only,
                overlap_size=overlap_size,
            )
            found = found or matched
        chunk = _read_chunk(stream, _CHUNK_BYTES, label, budget)
        if not chunk:
            break

    if expected_size is not None and total != expected_size:
        raise ScanError(f"{label}: archive member is truncated")
    return found, bytes(captured) if captured is not None else None, bytes(tail)


def _verify_stream_digest(
    stream: IO[bytes],
    expected_size: int,
    expected_digest: bytes,
    label: str,
    budget: _Budget,
) -> None:
    try:
        stream.seek(0)
    except Exception as exc:
        raise ScanError(f"{label}: regular file is not seekable") from exc
    hasher = hashlib.sha256()
    total = 0
    while True:
        chunk = _read_chunk(stream, _CHUNK_BYTES, label, budget)
        if not chunk:
            break
        total += len(chunk)
        if total > expected_size:
            raise ScanError(f"{label}: regular file changed while being scanned")
        hasher.update(chunk)
    if total != expected_size or hasher.digest() != expected_digest:
        raise ScanError(f"{label}: regular file changed while being scanned")


def _stream_matches(
    stream: IO[bytes],
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    label: str = "stream",
) -> bool:
    found, _, _ = _consume_stream(
        stream,
        sentinels,
        sentinels_only=sentinels_only,
        label=label,
    )
    return found


def _metadata_bytes(
    value: str | bytes,
    label: str,
    budget: _Budget,
) -> bytes:
    try:
        if isinstance(value, str):
            data = value.encode("utf-8", errors="strict")
        elif isinstance(value, bytes):
            value.decode("utf-8", errors="strict")
            data = value
        else:
            raise TypeError("metadata must be text or bytes")
    except (TypeError, UnicodeError) as exc:
        raise ScanError(f"{label}: archive metadata is not valid UTF-8 text") from exc
    if len(data) > _MAX_METADATA_FIELD_BYTES:
        raise ScanError(f"{label}: archive metadata field exceeds safety limit")
    if any(byte < 32 and byte not in {9, 10, 13} or byte == 127 for byte in data):
        raise ScanError(f"{label}: archive metadata contains unsafe binary content")
    budget.add_metadata(len(data), label)
    return data


def _scan_metadata(
    value: str | bytes,
    role: str,
    context: tuple[object, ...],
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    budget: _Budget,
    offenders: set[str],
) -> None:
    label = _display_context(context, sentinels, sentinels_only=sentinels_only)
    data = _metadata_bytes(value, label, budget)
    if _contains_secret(data, sentinels, sentinels_only=sentinels_only):
        offenders.add(
            _display_context(
                context + (f"<{role}>",),
                sentinels,
                sentinels_only=sentinels_only,
            )
        )


@contextmanager
def _opened_regular(path: Path, *, label: str | None = None) -> Generator[IO[bytes]]:
    display = label if label is not None else _safe_label(path)
    def snapshot(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise ScanError(f"{display}: symbolic links are not scanned")
        if not stat.S_ISREG(before.st_mode):
            raise ScanError(f"{display}: path is not a regular file")
        stream = path.open("rb")
        try:
            opened = os.fstat(stream.fileno())
            after_open = path.lstat()
            if (
                snapshot(before) != snapshot(opened)
                or snapshot(before) != snapshot(after_open)
                or not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(after_open.st_mode)
            ):
                raise ScanError(f"{display}: regular file changed while being opened")
            try:
                yield stream
            finally:
                descriptor_after = os.fstat(stream.fileno())
                stream.close()
                path_after = path.lstat()
                if (
                    snapshot(before) != snapshot(descriptor_after)
                    or snapshot(before) != snapshot(path_after)
                    or not stat.S_ISREG(descriptor_after.st_mode)
                    or not stat.S_ISREG(path_after.st_mode)
                ):
                    raise ScanError(f"{display}: regular file changed while being scanned")
        finally:
            if not stream.closed:
                stream.close()
    except ScanError:
        raise
    except OSError as exc:
        raise ScanError(f"{display}: unreadable regular file") from exc


def _files(
    paths: list[Path],
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    budget: _Budget,
) -> Iterator[Path]:
    def display(path: Path) -> str:
        return _display_context((path,), sentinels, sentinels_only=sentinels_only)

    def walk(directory: Path, depth: int) -> Iterator[Path]:
        if depth > _MAX_TRAVERSAL_DEPTH:
            raise ScanError(f"{display(directory)}: filesystem traversal depth exceeds safety limit")
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    candidate = Path(entry.path)
                    budget.add_traversal(display(candidate))
                    if entry.name in _SKIP_PARTS:
                        continue
                    if entry.is_symlink():
                        raise ScanError(f"{display(candidate)}: symbolic links are not scanned")
                    if entry.is_file(follow_symlinks=False):
                        yield candidate
                    elif entry.is_dir(follow_symlinks=False):
                        yield from walk(candidate, depth + 1)
                    else:
                        raise ScanError(f"{display(candidate)}: unsupported filesystem entry")
        except ScanError:
            raise
        except OSError as exc:
            raise ScanError(f"{display(directory)}: filesystem traversal failed") from exc

    for path in paths:
        try:
            budget.add_traversal(display(path))
            if path.is_symlink():
                raise ScanError(f"{display(path)}: symbolic links are not scanned")
            if path.is_dir():
                yield from walk(path, 1)
            elif path.is_file():
                yield path
            else:
                raise ScanError(
                    f"{display(path)}: path does not exist or is not a regular file"
                )
        except ScanError:
            raise
        except OSError as exc:
            raise ScanError(f"{display(path)}: filesystem traversal failed") from exc


def _scan_member_stream(
    stream: IO[bytes],
    name: str,
    expected_size: int,
    context: tuple[object, ...],
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    budget: _Budget,
    offenders: set[str],
    depth: int,
) -> None:
    label = _display_context(context, sentinels, sentinels_only=sentinels_only)
    prefix = _read_chunk(stream, min(_CHUNK_BYTES, expected_size + 1), label, budget)
    kind = _archive_kind(name, prefix[:512])
    capture_limit = (
        _MAX_NESTED_ARCHIVE_BYTES
        if expected_size <= _MAX_NESTED_ARCHIVE_BYTES
        else None
    )
    if kind is not None and expected_size > _MAX_NESTED_ARCHIVE_BYTES:
        raise ScanError(f"{label}: nested archive exceeds in-memory safety limit")
    found, captured, tail = _consume_stream(
        stream,
        sentinels,
        sentinels_only=sentinels_only,
        label=label,
        expected_size=expected_size,
        budget=budget,
        initial=prefix,
        capture_limit=capture_limit,
        tail_limit=_MAX_ZIP_DETECTION_TAIL_BYTES,
    )
    if found:
        offenders.add(label)
    if kind is None and captured is not None and _zip_tail_claims_archive(captured):
        kind = "zip"
    if kind is None and _zip_tail_claims_archive(tail):
        raise ScanError(f"{label}: nested archive exceeds in-memory safety limit")
    if kind is None:
        return
    if depth >= _MAX_ARCHIVE_DEPTH:
        raise ScanError(f"{label}: nested archive depth exceeds safety limit")
    assert captured is not None
    _scan_archive(
        io.BytesIO(captured),
        kind,
        context,
        sentinels,
        sentinels_only=sentinels_only,
        budget=budget,
        offenders=offenders,
        depth=depth + 1,
        container_size=len(captured),
    )


def _validate_extra_fields(extra: bytes, label: str, budget: _Budget) -> None:
    if len(extra) > _MAX_METADATA_FIELD_BYTES:
        raise ScanError(f"{label}: archive metadata field exceeds safety limit")
    budget.add_metadata(len(extra), label)
    offset = 0
    while offset < len(extra):
        if offset + 4 > len(extra):
            raise ScanError(f"{label}: ZIP extra field is truncated")
        field_id, field_size = struct.unpack_from("<HH", extra, offset)
        offset += 4
        if offset + field_size > len(extra):
            raise ScanError(f"{label}: ZIP extra field is truncated")
        if field_id == 0x0001:
            raise ScanError(f"{label}: ZIP64 metadata is not supported")
        offset += field_size


def _validate_zip_local_header(
    stream: IO[bytes],
    info: zipfile.ZipInfo,
    label: str,
    budget: _Budget,
    central_start: int,
) -> tuple[int, int]:
    stream.seek(info.header_offset)
    header = _read_chunk(stream, 30, label, budget)
    if len(header) != 30:
        raise ScanError(f"{label}: ZIP local header is truncated")
    (
        signature,
        _extract_version,
        flags,
        compression,
        _time,
        _date,
        crc,
        compressed_size,
        file_size,
        name_size,
        extra_size,
    ) = struct.unpack("<4s5H3L2H", header)
    if signature != b"PK\x03\x04":
        raise ScanError(f"{label}: ZIP local header signature is invalid")
    local_name = _read_chunk(stream, name_size, label, budget)
    local_extra = _read_chunk(stream, extra_size, label, budget)
    if len(local_name) != name_size or len(local_extra) != extra_size:
        raise ScanError(f"{label}: ZIP local metadata is truncated")
    encoding = "utf-8" if flags & 0x800 else "cp437"
    try:
        decoded_name = local_name.decode(encoding)
    except UnicodeError as exc:
        raise ScanError(f"{label}: ZIP local member name is invalid") from exc
    if (
        flags != info.flag_bits
        or compression != info.compress_type
        or decoded_name != info.orig_filename
    ):
        raise ScanError(f"{label}: ZIP local and central metadata are inconsistent")
    if not flags & 0x08 and (
        crc != info.CRC
        or compressed_size != info.compress_size
        or file_size != info.file_size
    ):
        raise ScanError(f"{label}: ZIP local and central sizes are inconsistent")
    _validate_extra_fields(local_extra, label, budget)
    _validate_extra_fields(info.extra, label, budget)
    data_start = info.header_offset + 30 + name_size + extra_size
    data_end = data_start + info.compress_size
    if data_end > central_start:
        raise ScanError(f"{label}: ZIP member overlaps the central directory")
    return info.header_offset, data_end


def _validate_zip_directory(info: zipfile.ZipInfo, label: str) -> None:
    if info.file_size != 0 or info.compress_size != 0:
        raise ScanError(f"{label}: ZIP directory contains a payload")
    dos_attributes = info.external_attr & 0xFFFF
    unix_type = (info.external_attr >> 16) & 0o170000
    valid = (
        info.create_system == 0 and bool(dos_attributes & 0x10)
    ) or (info.create_system == 3 and unix_type == stat.S_IFDIR)
    if not valid:
        raise ScanError(f"{label}: ZIP directory has invalid attributes")


def _validate_zip_regular(info: zipfile.ZipInfo, label: str) -> None:
    dos_attributes = info.external_attr & 0xFFFF
    unix_type = (info.external_attr >> 16) & 0o170000
    if info.create_system not in {0, 3} or dos_attributes & 0x10:
        raise ScanError(f"{label}: ZIP regular member has invalid attributes")
    if info.create_system == 3 and unix_type not in {0, stat.S_IFREG}:
        raise ScanError(f"{label}: ZIP regular member has invalid attributes")


def _scan_zip(
    stream: IO[bytes],
    context: tuple[object, ...],
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    budget: _Budget,
    offenders: set[str],
    depth: int,
) -> None:
    archive_label = _display_context(context, sentinels, sentinels_only=sentinels_only)
    try:
        original_position = stream.tell()
        stream.seek(0, os.SEEK_END)
        archive_size = stream.tell()
        if archive_size > _MAX_ARCHIVE_CONTAINER_BYTES:
            raise ScanError(f"{archive_label}: archive container exceeds safety limit")
        tail_size = min(archive_size, 65_557)
        stream.seek(archive_size - tail_size)
        tail = _read_chunk(stream, tail_size, archive_label)
        if len(tail) != tail_size:
            raise ScanError(f"{archive_label}: archive central directory is truncated")
        end_offset = tail.rfind(_ZIP_EOCD_MAGIC)
        if end_offset < 0 or end_offset + 22 > len(tail):
            raise ScanError(f"{archive_label}: archive central directory is unreadable")
        (
            _signature,
            disk_number,
            directory_disk,
            entries_on_disk,
            entry_count,
            directory_size,
            directory_offset,
            comment_size,
        ) = struct.unpack_from("<4s4H2LH", tail, end_offset)
        if end_offset + 22 + comment_size != len(tail):
            raise ScanError(f"{archive_label}: archive has unsafe trailing data")
        if disk_number or directory_disk or entries_on_disk != entry_count:
            raise ScanError(f"{archive_label}: multi-disk archives are not supported")
        if (
            entry_count == 0xFFFF
            or entries_on_disk == 0xFFFF
            or directory_size == 0xFFFFFFFF
            or directory_offset == 0xFFFFFFFF
            or any(magic in tail for magic in _ZIP64_MAGICS)
        ):
            raise ScanError(f"{archive_label}: ZIP64 metadata is not supported")
        if entry_count > _MAX_ARCHIVE_MEMBERS:
            raise ScanError(f"{archive_label}: archive member count exceeds safety limit")
        if directory_size > _MAX_ZIP_DIRECTORY_BYTES:
            raise ScanError(f"{archive_label}: archive central directory exceeds safety limit")
        absolute_end_offset = archive_size - tail_size + end_offset
        preamble_size = absolute_end_offset - directory_size - directory_offset
        if preamble_size < 0:
            raise ScanError(f"{archive_label}: archive central directory is inconsistent")
        central_start = directory_offset + preamble_size
        if central_start + directory_size != absolute_end_offset:
            raise ScanError(f"{archive_label}: archive central directory is inconsistent")
        stream.seek(original_position)
        with zipfile.ZipFile(stream) as archive:
            _scan_metadata(
                archive.comment,
                "archive comment",
                context,
                sentinels,
                sentinels_only=sentinels_only,
                budget=budget,
                offenders=offenders,
            )
            names: set[str] = set()
            regions: list[tuple[int, int]] = []
            infos = archive.infolist()
            if len(infos) != entry_count:
                raise ScanError(f"{archive_label}: ZIP member count is inconsistent")
            for info in infos:
                member_context = context + (info.filename,)
                member_label = _display_context(
                    member_context, sentinels, sentinels_only=sentinels_only
                )
                _validate_member_name(
                    info.filename,
                    member_context,
                    sentinels,
                    sentinels_only=sentinels_only,
                )
                _scan_metadata(
                    info.filename,
                    "member name",
                    member_context,
                    sentinels,
                    sentinels_only=sentinels_only,
                    budget=budget,
                    offenders=offenders,
                )
                _scan_metadata(
                    info.comment,
                    "member comment",
                    member_context,
                    sentinels,
                    sentinels_only=sentinels_only,
                    budget=budget,
                    offenders=offenders,
                )
                budget.add_member(info.file_size, info.compress_size, member_label)
                if info.filename in names:
                    raise ScanError(f"{member_label}: archive contains duplicate member paths")
                names.add(info.filename)
                if info.flag_bits & 0x1:
                    raise ScanError(f"{member_label}: archive contains an encrypted member")
                if info.flag_bits & 0x08:
                    raise ScanError(f"{member_label}: ZIP data descriptors are not supported")
                if info.compress_type not in {
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                    zipfile.ZIP_BZIP2,
                    zipfile.ZIP_LZMA,
                }:
                    raise ScanError(f"{member_label}: ZIP compression method is not supported")
                region = _validate_zip_local_header(
                    stream, info, member_label, budget, central_start
                )
                regions.append(region)
                if info.is_dir():
                    _validate_zip_directory(info, member_label)
                    continue
                _validate_zip_regular(info, member_label)
                try:
                    member_stream = archive.open(info, "r")
                except Exception as exc:
                    raise ScanError(f"{member_label}: archive member is unreadable") from exc
                with member_stream:
                    _scan_member_stream(
                        member_stream,
                        info.filename,
                        info.file_size,
                        member_context,
                        sentinels,
                        sentinels_only=sentinels_only,
                        budget=budget,
                        offenders=offenders,
                        depth=depth,
                    )
            ordered = sorted(regions)
            if any(
                previous[1] > current[0]
                for previous, current in pairwise(ordered)
            ):
                raise ScanError(f"{archive_label}: ZIP member regions overlap")
            if ordered:
                if ordered[0][0] != preamble_size or ordered[-1][1] != central_start or any(
                    previous[1] != current[0] for previous, current in pairwise(ordered)
                ):
                    raise ScanError(f"{archive_label}: ZIP contains unexplained gaps")
            elif central_start != preamble_size:
                raise ScanError(f"{archive_label}: ZIP contains unexplained gaps")
    except ScanError:
        raise
    except Exception as exc:
        raise ScanError(f"{archive_label}: archive is unreadable or corrupt") from exc


def _scan_single_gzip_payload(
    stream: IO[bytes],
    archive_label: str,
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    budget: _Budget,
    offenders: set[str],
) -> None:
    try:
        stream.seek(0)
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        overlap_size = max(
            _PATTERN_OVERLAP if not sentinels_only else 0,
            max((len(sentinel) - 1 for sentinel in sentinels), default=0),
        )
        overlap = b""
        found = False
        while True:
            compressed = _read_chunk(stream, _CHUNK_BYTES, archive_label, budget)
            if not compressed:
                break
            pending = compressed
            while pending:
                expanded = decompressor.decompress(pending, _CHUNK_BYTES)
                pending = decompressor.unconsumed_tail
                budget.add_work(len(expanded), archive_label)
                budget.add_expanded_stream(len(expanded), archive_label)
                matched, overlap = _scan_window(
                    expanded,
                    overlap,
                    sentinels,
                    sentinels_only=sentinels_only,
                    overlap_size=overlap_size,
                )
                found = found or matched
                if decompressor.unused_data:
                    raise ScanError(
                        f"{archive_label}: concatenated or trailing gzip data is not supported"
                    )
                if not expanded and not pending:
                    break
        if not decompressor.eof:
            raise ScanError(f"{archive_label}: gzip stream is truncated")
        if found:
            offenders.add(archive_label)
        stream.seek(0)
    except ScanError:
        raise
    except Exception as exc:
        raise ScanError(f"{archive_label}: gzip stream is unreadable or corrupt") from exc


def _scan_single_bounded_compressed_payload(
    stream: IO[bytes],
    archive_label: str,
    compression: str,
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    budget: _Budget,
    offenders: set[str],
) -> None:
    try:
        stream.seek(0)
        if compression == "bzip2":
            decompressor = bz2.BZ2Decompressor()
        elif compression == "xz":
            decompressor = lzma.LZMADecompressor(
                format=lzma.FORMAT_XZ,
                memlimit=_MAX_ARCHIVE_CONTAINER_BYTES,
            )
        else:  # pragma: no cover - private caller invariant
            raise AssertionError("unsupported decompressor")
        overlap_size = max(
            _PATTERN_OVERLAP if not sentinels_only else 0,
            max((len(sentinel) - 1 for sentinel in sentinels), default=0),
        )
        overlap = b""
        found = False
        while not decompressor.eof:
            compressed = _read_chunk(stream, _CHUNK_BYTES, archive_label, budget)
            if not compressed:
                break
            pending = compressed
            while pending or not decompressor.needs_input:
                expanded = decompressor.decompress(pending, _CHUNK_BYTES)
                pending = b""
                budget.add_work(len(expanded), archive_label)
                budget.add_expanded_stream(len(expanded), archive_label)
                matched, overlap = _scan_window(
                    expanded,
                    overlap,
                    sentinels,
                    sentinels_only=sentinels_only,
                    overlap_size=overlap_size,
                )
                found = found or matched
                if decompressor.eof:
                    if decompressor.unused_data:
                        raise ScanError(
                            f"{archive_label}: concatenated or trailing {compression} data is not supported"
                        )
                    break
                if not expanded and decompressor.needs_input:
                    break
        if not decompressor.eof:
            raise ScanError(f"{archive_label}: {compression} stream is truncated")
        trailing = _read_chunk(stream, 1, archive_label, budget)
        if trailing:
            raise ScanError(
                f"{archive_label}: concatenated or trailing {compression} data is not supported"
            )
        if found:
            offenders.add(archive_label)
        stream.seek(0)
    except ScanError:
        raise
    except Exception as exc:
        raise ScanError(
            f"{archive_label}: {compression} stream is unreadable or corrupt"
        ) from exc


def _verify_tar_zero_trailer(
    stream: IO[bytes], payload_end: int, label: str, budget: _Budget
) -> None:
    try:
        stream.seek(payload_end)
    except Exception as exc:
        raise ScanError(f"{label}: TAR stream is not seekable") from exc
    while True:
        chunk = _read_chunk(stream, _CHUNK_BYTES, label, budget)
        if not chunk:
            return
        if any(chunk):
            raise ScanError(f"{label}: archive has non-zero data after TAR payload")


def _scan_tar(
    stream: IO[bytes],
    context: tuple[object, ...],
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    budget: _Budget,
    offenders: set[str],
    depth: int,
    container_size: int,
    expected_compression: str | None,
) -> None:
    archive_label = _display_context(context, sentinels, sentinels_only=sentinels_only)
    expanded_before = budget.expanded_bytes
    local_expanded = 0
    try:
        stream.seek(0)
        compression_magic = _read_chunk(stream, 6, archive_label, budget)
        stream.seek(0)
        actual_compression = None
        if compression_magic.startswith(b"\x1f\x8b"):
            actual_compression = "gzip"
        elif compression_magic.startswith(_BZIP2_MAGIC):
            actual_compression = "bzip2"
        elif compression_magic.startswith(_XZ_MAGIC):
            actual_compression = "xz"
        if expected_compression is not None and actual_compression != expected_compression:
            display = "XZ" if expected_compression == "xz" else expected_compression
            raise ScanError(f"{archive_label}: {display} stream has invalid magic")
        if actual_compression == "gzip":
            _scan_single_gzip_payload(
                stream,
                archive_label,
                sentinels,
                sentinels_only=sentinels_only,
                budget=budget,
                offenders=offenders,
            )
        elif actual_compression == "bzip2":
            _scan_single_bounded_compressed_payload(
                stream,
                archive_label,
                "bzip2",
                sentinels,
                sentinels_only=sentinels_only,
                budget=budget,
                offenders=offenders,
            )
        elif actual_compression == "xz":
            _scan_single_bounded_compressed_payload(
                stream,
                archive_label,
                "xz",
                sentinels,
                sentinels_only=sentinels_only,
                budget=budget,
                offenders=offenders,
            )
        tar_modes: dict[
            str | None, Literal["r:", "r:gz", "r:bz2", "r:xz"]
        ] = {
            None: "r:",
            "gzip": "r:gz",
            "bzip2": "r:bz2",
            "xz": "r:xz",
        }
        with tarfile.open(fileobj=stream, mode=tar_modes[actual_compression]) as archive:
            names: set[str] = set()
            for info in archive:
                member_context = context + (info.name,)
                member_label = _display_context(
                    member_context, sentinels, sentinels_only=sentinels_only
                )
                _validate_member_name(
                    info.name,
                    member_context,
                    sentinels,
                    sentinels_only=sentinels_only,
                )
                for role, value in (
                    ("member name", info.name),
                    ("user name", info.uname),
                    ("group name", info.gname),
                    ("link name", info.linkname),
                ):
                    if value:
                        _scan_metadata(
                            value,
                            role,
                            member_context,
                            sentinels,
                            sentinels_only=sentinels_only,
                            budget=budget,
                            offenders=offenders,
                        )
                for key, value in info.pax_headers.items():
                    _scan_metadata(
                        key,
                        "PAX key",
                        member_context,
                        sentinels,
                        sentinels_only=sentinels_only,
                        budget=budget,
                        offenders=offenders,
                    )
                    _scan_metadata(
                        value,
                        "PAX value",
                        member_context,
                        sentinels,
                        sentinels_only=sentinels_only,
                        budget=budget,
                        offenders=offenders,
                    )
                budget.add_member(info.size, None, member_label)
                if not info.isdir():
                    local_expanded += info.size
                    if (
                        local_expanded > _RATIO_CHECK_MIN_BYTES
                        and local_expanded
                        > max(1, container_size) * _MAX_COMPRESSION_RATIO
                    ):
                        raise ScanError(
                            f"{archive_label}: archive compression ratio exceeds safety limit"
                        )
                if info.name in names:
                    raise ScanError(f"{member_label}: archive contains duplicate member paths")
                names.add(info.name)
                if info.isdir():
                    if info.size != 0:
                        raise ScanError(
                            f"{member_label}: TAR directory contains a payload"
                        )
                    continue
                if not info.isfile():
                    raise ScanError(f"{member_label}: archive contains a non-regular member")
                try:
                    member_stream = archive.extractfile(info)
                except Exception as exc:
                    raise ScanError(f"{member_label}: archive member is unreadable") from exc
                if member_stream is None:
                    raise ScanError(f"{member_label}: archive member is unreadable")
                with member_stream:
                    _scan_member_stream(
                        member_stream,
                        info.name,
                        info.size,
                        member_context,
                        sentinels,
                        sentinels_only=sentinels_only,
                        budget=budget,
                        offenders=offenders,
                        depth=depth,
                    )
            _verify_tar_zero_trailer(
                cast(IO[bytes], archive.fileobj),
                archive.offset,
                archive_label,
                budget,
            )
        expanded = budget.expanded_bytes - expanded_before
        if (
            expanded > _RATIO_CHECK_MIN_BYTES
            and expanded > max(1, container_size) * _MAX_COMPRESSION_RATIO
        ):
            raise ScanError(f"{archive_label}: archive compression ratio exceeds safety limit")
    except ScanError:
        raise
    except Exception as exc:
        raise ScanError(f"{archive_label}: archive is unreadable or corrupt") from exc


def _scan_archive(
    stream: IO[bytes],
    kind: str,
    context: tuple[object, ...],
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    budget: _Budget,
    offenders: set[str],
    depth: int,
    container_size: int,
) -> None:
    if kind == "zip":
        _scan_zip(
            stream,
            context,
            sentinels,
            sentinels_only=sentinels_only,
            budget=budget,
            offenders=offenders,
            depth=depth,
        )
    elif kind == "tar-zstd":
        archive_label = _display_context(
            context, sentinels, sentinels_only=sentinels_only
        )
        raise ScanError(f"{archive_label}: zstd-compressed TAR is not supported")
    elif kind == "tar-lzma-alone":
        archive_label = _display_context(
            context, sentinels, sentinels_only=sentinels_only
        )
        raise ScanError(f"{archive_label}: LZMA-alone compressed TAR is not supported")
    else:
        expected_compression = kind.removeprefix("tar-") if kind != "tar" else None
        _scan_tar(
            stream,
            context,
            sentinels,
            sentinels_only=sentinels_only,
            budget=budget,
            offenders=offenders,
            depth=depth,
            container_size=container_size,
            expected_compression=expected_compression,
        )


def _scan_path(
    path: Path,
    sentinels: tuple[bytes, ...],
    *,
    sentinels_only: bool,
    budget: _Budget,
    offenders: set[str],
) -> None:
    context: tuple[object, ...] = (path,)
    _scan_metadata(
        str(path),
        "archive or file name",
        context,
        sentinels,
        sentinels_only=sentinels_only,
        budget=budget,
        offenders=offenders,
    )
    label = _display_context(context, sentinels, sentinels_only=sentinels_only)
    with _opened_regular(path, label=label) as stream:
        try:
            container_size = os.fstat(stream.fileno()).st_size
        except OSError as exc:
            raise ScanError(f"{label}: regular file size is unreadable") from exc
        budget.add_file(container_size, label)
        prefix = _read_chunk(stream, 512, label, budget)
        kind = _archive_kind(path.name, prefix)
        if kind is not None and container_size > _MAX_ARCHIVE_CONTAINER_BYTES:
            raise ScanError(f"{label}: archive container exceeds safety limit")
        hasher = hashlib.sha256()
        found, _captured, _tail = _consume_stream(
            stream,
            sentinels,
            sentinels_only=sentinels_only,
            label=label,
            expected_size=container_size,
            budget=budget,
            initial=prefix,
            hasher=hasher,
        )
        _verify_stream_digest(
            stream, container_size, hasher.digest(), label, budget
        )
        if kind is None and _stream_claims_zip(stream, container_size, label):
            kind = "zip"
        if found:
            offenders.add(label)
        if kind is None:
            return
        stream.seek(0)
        _scan_archive(
            stream,
            kind,
            context,
            sentinels,
            sentinels_only=sentinels_only,
            budget=budget,
            offenders=offenders,
            depth=1,
            container_size=container_size,
        )


def scan(
    paths: list[Path], sentinels: tuple[bytes, ...] = (), *, sentinels_only: bool = False
) -> list[str]:
    if len(sentinels) > _MAX_SENTINEL_COUNT:
        raise ScanError("sentinel count exceeds safety limit")
    if sum(map(len, sentinels)) > _MAX_SENTINEL_AGGREGATE_BYTES:
        raise ScanError("sentinel aggregate exceeds safety limit")
    if any(not sentinel or len(sentinel) > _MAX_SENTINEL_BYTES for sentinel in sentinels):
        raise ScanError("sentinel is empty or exceeds the safety limit")
    offenders: set[str] = set()
    budget = _Budget()
    for path in _files(
        paths, sentinels, sentinels_only=sentinels_only, budget=budget
    ):
        _scan_path(
            path,
            sentinels,
            sentinels_only=sentinels_only,
            budget=budget,
            offenders=offenders,
        )
    return sorted(offenders)


def _read_sentinels(path: Path) -> tuple[bytes, ...]:
    values: list[bytes] = []
    seen: set[bytes] = set()
    aggregate = 0
    file_bytes = 0
    with _opened_regular(path, label="<sentinel manifest>") as stream:
        try:
            manifest_size = os.fstat(stream.fileno()).st_size
        except OSError as exc:
            raise _fail("<sentinel manifest>", "sentinel manifest size is unreadable") from exc
        if manifest_size > _MAX_SENTINEL_FILE_BYTES:
            raise _fail("<sentinel manifest>", "sentinel manifest file exceeds safety limit")
        while True:
            try:
                raw = stream.readline(_MAX_SENTINEL_BYTES + 2)
            except Exception as exc:
                raise _fail("<sentinel manifest>", "sentinel manifest became unreadable") from exc
            if not isinstance(raw, bytes):
                raise _fail("<sentinel manifest>", "sentinel manifest returned non-byte content")
            if not raw:
                break
            file_bytes += len(raw)
            if file_bytes > _MAX_SENTINEL_FILE_BYTES:
                raise _fail("<sentinel manifest>", "sentinel manifest file exceeds safety limit")
            terminated = raw.endswith(b"\n")
            value = raw[:-1] if terminated else raw
            if value.endswith(b"\r"):
                value = value[:-1]
            if len(value) > _MAX_SENTINEL_BYTES:
                raise _fail("<sentinel manifest>", "sentinel manifest contains an overlong line")
            if not value:
                raise _fail("<sentinel manifest>", "sentinel manifest contains an empty sentinel")
            try:
                text = value.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise _fail("<sentinel manifest>", "sentinel manifest is not valid UTF-8 text") from exc
            if any(ord(character) < 32 or ord(character) == 127 for character in text):
                raise _fail("<sentinel manifest>", "sentinel manifest contains unsafe text")
            if value in seen:
                raise _fail("<sentinel manifest>", "sentinel manifest contains duplicate sentinels")
            values.append(value)
            seen.add(value)
            if len(values) > _MAX_SENTINEL_COUNT:
                raise _fail("<sentinel manifest>", "sentinel manifest count exceeds safety limit")
            aggregate += len(value)
            if aggregate > _MAX_SENTINEL_AGGREGATE_BYTES:
                raise _fail("<sentinel manifest>", "sentinel manifest aggregate exceeds safety limit")
    return tuple(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--sentinels-file", type=Path)
    parser.add_argument("--sentinels-only", action="store_true")
    args = parser.parse_args()
    try:
        sentinels = _read_sentinels(args.sentinels_file) if args.sentinels_file else ()
        if args.sentinels_only and not sentinels:
            raise ScanError("--sentinels-only requires a non-empty --sentinels-file")
        offenders = scan(args.paths, sentinels, sentinels_only=args.sentinels_only)
    except ScanError as exc:
        print(f"SIWC artifact scan failed closed: {exc}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - CLI must fail closed without traceback.
        print("SIWC artifact scan failed closed: unexpected scanner failure", file=sys.stderr)
        return 2
    if offenders:
        for offender in offenders:
            print(f"SIWC credential material detected: {offender}")
        return 1
    print("SIWC artifact scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
