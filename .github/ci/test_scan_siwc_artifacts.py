from __future__ import annotations

import bz2
import importlib.util
import io
import lzma
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from typing import Literal
from unittest.mock import patch

ROOT = Path(__file__).parents[2]
SCANNER = ROOT / ".github" / "ci" / "scan_siwc_artifacts.py"


class SIWCArtifactScannerTest(unittest.TestCase):
    @staticmethod
    def _scanner_module():
        spec = importlib.util.spec_from_file_location("scan_siwc_artifacts", SCANNER)
        assert spec is not None and spec.loader is not None
        scanner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = scanner
        spec.loader.exec_module(scanner)
        return scanner

    @staticmethod
    def _zip_bytes(name: str, content: bytes, *, comment: bytes = b"") -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.comment = comment
            archive.writestr(name, content)
        return output.getvalue()

    @staticmethod
    def _tar_bytes(
        name: str,
        content: bytes,
        *,
        gzip: bool = False,
        compression: str | None = None,
    ) -> bytes:
        output = io.BytesIO()
        selected = "gz" if gzip else compression
        modes: dict[str | None, Literal["w", "w:gz", "w:bz2", "w:xz"]] = {
            None: "w",
            "gz": "w:gz",
            "bz2": "w:bz2",
            "xz": "w:xz",
        }
        mode = modes[selected]
        with tarfile.open(fileobj=output, mode=mode) as archive:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        return output.getvalue()

    @staticmethod
    def _zip_directory_bytes(
        payload: bytes,
        *,
        compression: int,
        external_attr: int = (0o40775 << 16) | 0x10,
        create_system: int = 3,
    ) -> bytes:
        output = io.BytesIO()
        info = zipfile.ZipInfo("package/directory/")
        info.create_system = create_system
        info.external_attr = external_attr
        info.compress_type = compression
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(info, payload)
        return output.getvalue()

    def _run(self, *paths: Path, sentinel: bytes | None = None) -> subprocess.CompletedProcess[str]:
        arguments = [sys.executable, str(SCANNER)]
        if sentinel is not None:
            sentinel_file = paths[0].parent / "sentinels"
            sentinel_file.write_bytes(sentinel + b"\n")
            arguments.extend(("--sentinels-file", str(sentinel_file)))
        arguments.extend(str(path) for path in paths)
        return subprocess.run(arguments, capture_output=True, text=True, check=False)

    def test_scans_regular_files_and_distribution_members_for_runtime_sentinels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "guide.md").write_text("securo-siwc --secret-stdin\n", encoding="utf-8")
            wheel = root / "securo_siwc-0.1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("securo_siwc/safe.txt", "no credentials here")
            sentinel = b"runtime-refresh-sentinel-that-must-never-leak"
            sentinel_file = root / "sentinels"
            sentinel_file.write_bytes(sentinel + b"\n")

            safe = subprocess.run(
                [
                    sys.executable,
                    str(SCANNER),
                    "--sentinels-file",
                    str(sentinel_file),
                    str(docs),
                    str(wheel),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(safe.returncode, 0, safe.stdout + safe.stderr)

            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr("securo_siwc/leak.log", sentinel)
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(SCANNER),
                    "--sentinels-file",
                    str(sentinel_file),
                    str(docs),
                    str(wheel),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(rejected.returncode, 1)
            self.assertIn("leak.log", rejected.stdout)

    def test_rejects_generic_bearer_and_private_key_material(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "runtime.log"
            artifact.write_text(
                "Authorization: " "Bea" "rer abcdefghijklmnopqrstuvwxyz123456\n"
                "-----BEGIN " "PRIVATE KEY-----\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(SCANNER), str(artifact)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("runtime.log", result.stdout)

    def test_scans_regular_files_larger_than_twenty_megabytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"large-regular-runtime-sentinel"
            artifact = root / "large.bin"
            artifact.write_bytes(b"x" * 20_000_001 + sentinel)

            result = self._run(artifact, sentinel=sentinel)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("large.bin", result.stdout)

    def test_scans_large_zip_and_wheel_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"large-wheel-runtime-sentinel"
            for name in ("artifact.zip", "artifact.whl"):
                artifact = root / name
                with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_STORED) as archive:
                    archive.writestr("package/large.bin", b"x" * 20_000_001 + sentinel)
                result = self._run(artifact, sentinel=sentinel)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("package/large.bin", result.stdout)

    def test_scans_large_tar_and_sdist_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"large-sdist-runtime-sentinel"
            for name, mode in (("artifact.tar", "w"), ("artifact.tar.gz", "w:gz")):
                artifact = root / name
                content = os.urandom(20_000_001) + sentinel
                with tarfile.open(artifact, mode) as archive:
                    info = tarfile.TarInfo("package/large.bin")
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
                result = self._run(artifact, sentinel=sentinel)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("package/large.bin", result.stdout)

    def test_finds_sentinel_crossing_stream_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"cross-chunk-runtime-sentinel"
            artifact = root / "boundary.bin"
            artifact.write_bytes(b"x" * (64 * 1024 - 7) + sentinel)

            result = self._run(artifact, sentinel=sentinel)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_corrupt_archive_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "corrupt.whl"
            artifact.write_bytes(b"PK\x03\x04not-a-readable-wheel")

            result = self._run(artifact)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("artifact scan failed closed", result.stderr)
            self.assertIn("corrupt.whl", result.stderr)

    def test_unreadable_regular_file_fails_closed(self):
        scanner = self._scanner_module()
        artifact = Path("unreadable-runtime.log")
        original_open = Path.open

        def denied(path: Path, *args, **kwargs):
            if path == artifact:
                raise PermissionError("denied")
            return original_open(path, *args, **kwargs)

        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "open", denied),
            self.assertRaisesRegex(scanner.ScanError, "unreadable-runtime.log"),
        ):
            scanner.scan([artifact])

    def test_bad_zip_member_crc_fails_closed_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "bad-crc.whl"
            with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("package/data.bin", b"crc payload")
            with zipfile.ZipFile(artifact) as archive:
                info = archive.getinfo("package/data.bin")
            raw = bytearray(artifact.read_bytes())
            data_offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
            raw[data_offset] ^= 0x01
            artifact.write_bytes(raw)

            result = self._run(artifact)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("bad-crc.whl!package/data.bin", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_stream_runtime_error_has_member_context(self):
        scanner = self._scanner_module()

        class BrokenStream(io.BytesIO):
            failure: Exception

            def read(self, size: int | None = -1) -> bytes:
                raise self.failure

        for failure in (
            RuntimeError("decompressor internals must not escape"),
            zlib.error("invalid compressed data"),
            tarfile.ReadError("truncated tar member"),
            zipfile.BadZipFile("bad member CRC"),
        ):
            with self.subTest(failure=type(failure).__name__):
                stream = BrokenStream()
                stream.failure = failure
                with self.assertRaisesRegex(scanner.ScanError, "artifact.whl!member.bin"):
                    scanner._stream_matches(
                        stream, (), sentinels_only=False, label="artifact.whl!member.bin"
                    )

    def test_truncated_deflated_zip_member_fails_closed_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "truncated-member.zip"
            with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("package/data.bin", os.urandom(64 * 1024))
            with zipfile.ZipFile(artifact) as archive:
                info = archive.getinfo("package/data.bin")
            raw = bytearray(artifact.read_bytes())
            truncated_size = info.compress_size - 2
            struct.pack_into("<L", raw, info.header_offset + 18, truncated_size)
            central_offset = raw.index(b"PK\x01\x02")
            struct.pack_into("<L", raw, central_offset + 20, truncated_size)
            artifact.write_bytes(raw)

            result = self._run(artifact)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("truncated-member.zip!package/data.bin", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_truncated_tar_member_fails_closed_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "truncated.tar.gz"
            raw = self._tar_bytes("package/data.bin", os.urandom(256 * 1024), gzip=True)
            artifact.write_bytes(raw[:-128])

            result = self._run(artifact)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("truncated.tar.gz", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_scans_zip_names_and_comments_without_echoing_secret_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"runtime-metadata-sentinel"
            artifacts = []

            archive_name = root / f"artifact-{sentinel.decode()}.zip"
            archive_name.write_bytes(self._zip_bytes("safe.txt", b"safe"))
            artifacts.append(archive_name)

            member_name = root / "member-name.zip"
            member_name.write_bytes(self._zip_bytes(f"package/{sentinel.decode()}.txt", b"safe"))
            artifacts.append(member_name)

            archive_comment = root / "archive-comment.zip"
            archive_comment.write_bytes(self._zip_bytes("safe.txt", b"safe", comment=sentinel))
            artifacts.append(archive_comment)

            member_comment = root / "member-comment.whl"
            with zipfile.ZipFile(member_comment, "w") as archive:
                info = zipfile.ZipInfo("package/safe.txt")
                info.comment = sentinel
                archive.writestr(info, b"safe")
            artifacts.append(member_comment)

            for artifact in artifacts:
                with self.subTest(artifact=artifact.name):
                    result = self._run(artifact, sentinel=sentinel)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

    def test_scans_bounded_tar_pax_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"runtime-pax-sentinel"
            artifact = root / "metadata.tar"
            with tarfile.open(artifact, "w", format=tarfile.PAX_FORMAT) as archive:
                info = tarfile.TarInfo("package/safe.txt")
                info.size = 4
                info.pax_headers = {"securo.note": sentinel.decode()}
                archive.addfile(info, io.BytesIO(b"safe"))

            result = self._run(artifact, sentinel=sentinel)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

    def test_binary_and_oversized_zip_metadata_fail_closed(self):
        scanner = self._scanner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, comment in (
                ("binary.zip", b"safe\x00not-text"),
                ("oversized.zip", b"x" * (scanner._MAX_METADATA_FIELD_BYTES + 1)),
            ):
                artifact = root / name
                artifact.write_bytes(self._zip_bytes("safe.txt", b"safe", comment=comment))
                with self.subTest(name=name):
                    result = self._run(artifact)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertNotIn(comment.decode(errors="ignore"), result.stderr)

    def test_recursively_scans_zip_inside_wheel_and_tar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"nested-runtime-sentinel"
            inner = self._zip_bytes("deep/leak.bin", sentinel)
            wheel = root / "outer.whl"
            wheel.write_bytes(self._zip_bytes("package/inner.zip", inner))
            tar_path = root / "outer.tar"
            tar_path.write_bytes(self._tar_bytes("package/inner.zip", inner))

            for artifact in (wheel, tar_path):
                with self.subTest(artifact=artifact.name):
                    result = self._run(artifact, sentinel=sentinel)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn("deep/leak.bin", result.stdout)

    def test_nested_archive_depth_and_compression_bombs_fail_closed(self):
        scanner = self._scanner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = self._zip_bytes("safe.txt", b"safe")
            for depth in range(scanner._MAX_ARCHIVE_DEPTH + 1):
                nested = self._zip_bytes(f"level-{depth}.zip", nested)
            too_deep = root / "too-deep.zip"
            too_deep.write_bytes(nested)

            bomb = root / "bomb.zip"
            bomb.write_bytes(
                self._zip_bytes("nested.zip", self._zip_bytes("zeros.bin", b"0" * 2_000_000))
            )

            for artifact, reason in ((too_deep, "depth"), (bomb, "compression ratio")):
                with self.subTest(artifact=artifact.name):
                    result = self._run(artifact)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn(reason, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_nested_archives_share_member_expansion_and_work_budgets(self):
        scanner = self._scanner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inner = self._zip_bytes("deep/data.bin", b"x" * 1_000)
            artifact = root / "shared-budget.zip"
            with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("outer.txt", b"safe")
                archive.writestr("inner.zip", inner)

            with (
                patch.object(scanner, "_MAX_ARCHIVE_MEMBERS", 2),
                self.assertRaisesRegex(scanner.ScanError, "member count"),
            ):
                scanner.scan([artifact])
            with (
                patch.object(
                    scanner,
                    "_MAX_ARCHIVE_TOTAL_BYTES",
                    len(inner) + len(b"safe") + 500,
                ),
                self.assertRaisesRegex(scanner.ScanError, "expanded size"),
            ):
                scanner.scan([artifact])
            with (
                patch.object(scanner, "_MAX_ARCHIVE_WORK_BYTES", 100),
                self.assertRaisesRegex(scanner.ScanError, "scan work"),
            ):
                scanner.scan([artifact])

    def test_sentinel_manifest_limits_fail_closed_without_echoing_values(self):
        scanner = self._scanner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "safe.bin"
            artifact.write_bytes(b"safe")
            aggregate_lines = [
                f"{index:03d}-".encode() + b"a" * 296
                for index in range(scanner._MAX_SENTINEL_COUNT)
            ]
            cases = {
                "too-many": (
                    b"\n".join(
                        f"sentinel-{index}".encode()
                        for index in range(scanner._MAX_SENTINEL_COUNT + 1)
                    )
                    + b"\n",
                    "manifest count",
                ),
                "aggregate": (
                    b"\n".join(aggregate_lines) + b"\n",
                    "manifest aggregate",
                ),
                "file": (
                    b"y" * (scanner._MAX_SENTINEL_FILE_BYTES + 1),
                    "manifest file",
                ),
                "overlong-line": (
                    b"z" * (scanner._MAX_SENTINEL_BYTES + 1) + b"\n",
                    "manifest contains an overlong line",
                ),
                "empty": (b"valid\n\n", "manifest contains an empty sentinel"),
                "duplicate": (
                    b"duplicate-value\nduplicate-value\n",
                    "manifest contains duplicate sentinels",
                ),
            }
            for name, (manifest, reason) in cases.items():
                sentinel_file = root / f"sentinels-{name}"
                sentinel_file.write_bytes(manifest)
                with self.subTest(name=name):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(SCANNER),
                            "--sentinels-file",
                            str(sentinel_file),
                            str(artifact),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn(reason, result.stderr)
                    self.assertNotIn("duplicate-value", result.stdout + result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_symlinks_fail_closed_and_arbitrary_binary_is_not_misclassified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.bin"
            target.write_bytes(b"safe")
            link = root / "link.bin"
            link.symlink_to(target)
            binary = root / "arbitrary.bin"
            binary.write_bytes(b"random-prefix\x00PK\x03\x04not-an-archive")

            rejected = self._run(link)
            accepted = self._run(binary)

            self.assertEqual(rejected.returncode, 2, rejected.stdout + rejected.stderr)
            self.assertIn("symbolic links", rejected.stderr)
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)

    def test_recognizes_structural_prefixed_zip_under_arbitrary_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"prefixed-bin-runtime-sentinel"
            inner = b"self-extracting preamble\x00" + self._zip_bytes("deep/leak.bin", sentinel)
            cases = {
                "payload.bin": inner,
                "nested-in-zip.zip": self._zip_bytes("opaque.bin", inner),
                "nested-in-tar.tar": self._tar_bytes("opaque.data", inner),
            }
            for name, payload in cases.items():
                artifact = root / name
                artifact.write_bytes(payload)
                with self.subTest(name=name):
                    result = self._run(artifact, sentinel=sentinel)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn("deep/leak.bin", result.stdout)

            generic = root / "credential-payload.bin"
            generic.write_bytes(
                b"#!/bin/sh\nself extracting\n"
                + self._zip_bytes("payload.txt", b"sk-" + b"1234567890abcdefghij")
            )
            result = self._run(generic)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("payload.txt", result.stdout)

    def test_structural_zip_claim_with_malformed_eocd_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = bytearray(b"preamble" + self._zip_bytes("safe.txt", b"safe"))
            eocd = base.rindex(b"PK\x05\x06")
            bad_offset = bytearray(base)
            struct.pack_into("<L", bad_offset, eocd + 16, 0xFFFFFF00)
            bad_comment = bytearray(base)
            struct.pack_into("<H", bad_comment, eocd + 20, 123)
            for name, raw in (("offset.bin", bad_offset), ("comment.bin", bad_comment)):
                artifact = root / name
                artifact.write_bytes(raw)
                with self.subTest(name=name):
                    result = self._run(artifact)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("archive", result.stderr)

    def test_rejects_every_zip_data_descriptor_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in (
                b"",
                struct.pack("<4sLLL", b"PK\x07\x08", 0, 4, 4),
                struct.pack("<LQQ", 0, 4, 4),
            ):
                raw = bytearray(self._zip_bytes("safe.txt", b"safe"))
                local = raw.index(b"PK\x03\x04")
                central = raw.index(b"PK\x01\x02")
                struct.pack_into("<H", raw, local + 6,
                                 struct.unpack_from("<H", raw, local + 6)[0] | 0x08)
                struct.pack_into("<H", raw, central + 8,
                                 struct.unpack_from("<H", raw, central + 8)[0] | 0x08)
                if suffix:
                    raw[central:central] = suffix
                    eocd = raw.rindex(b"PK\x05\x06")
                    struct.pack_into("<L", raw, eocd + 16, central + len(suffix))
                artifact = root / f"descriptor-{len(suffix)}.zip"
                artifact.write_bytes(raw)
                with self.subTest(size=len(suffix)):
                    result = self._run(artifact)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("data descriptor", result.stderr)

    def test_rejects_zip64_encryption_unsupported_methods_and_member_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mutations = {}
            base = bytearray(self._zip_bytes("safe.txt", b"safe"))
            local = base.index(b"PK\x03\x04")
            central = base.index(b"PK\x01\x02")
            eocd = base.index(b"PK\x05\x06")

            encrypted = bytearray(base)
            struct.pack_into("<H", encrypted, local + 6,
                             struct.unpack_from("<H", encrypted, local + 6)[0] | 1)
            struct.pack_into("<H", encrypted, central + 8,
                             struct.unpack_from("<H", encrypted, central + 8)[0] | 1)
            mutations["encrypted.zip"] = encrypted

            unsupported = bytearray(base)
            struct.pack_into("<H", unsupported, local + 8, 99)
            struct.pack_into("<H", unsupported, central + 10, 99)
            mutations["unsupported.zip"] = unsupported

            zip64 = bytearray(base)
            struct.pack_into("<L", zip64, eocd + 16, 0xFFFFFFFF)
            mutations["zip64.zip"] = zip64

            gap = bytearray(base)
            gap[central:central] = b"unexplained"
            gap_eocd = gap.rindex(b"PK\x05\x06")
            struct.pack_into("<L", gap, gap_eocd + 16, central + len(b"unexplained"))
            mutations["gap.zip"] = gap

            for name, payload in mutations.items():
                artifact = root / name
                artifact.write_bytes(payload)
                with self.subTest(name=name):
                    result = self._run(artifact)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_symlink_error_does_not_echo_secret_path_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"secret-symlink-name-sentinel"
            target = root / "target.bin"
            target.write_bytes(b"safe")
            link = root / f"{sentinel.decode()}.zip"
            link.symlink_to(target)

            result = self._run(link, sentinel=sentinel)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("symbolic links", result.stderr)
            self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

    def test_archive_magic_without_extension_fails_closed_when_corrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, magic in (
                ("zip.bin", b"PK\x03\x04broken"),
                ("gzip.bin", b"\x1f\x8b\x08\x00broken"),
                ("tar.bin", b"x" * 257 + b"ustar\x00" + b"broken"),
            ):
                artifact = root / name
                artifact.write_bytes(magic)
                with self.subTest(name=name):
                    result = self._run(artifact)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn(name, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    @staticmethod
    def _zip_with_distinct_extra_sentinel(
        sentinel: bytes, *, location: str, preamble: bytes = b""
    ) -> bytes:
        extra_payload = b"x" * len(sentinel)
        info = zipfile.ZipInfo("package/safe.txt")
        info.extra = struct.pack("<HH", 0xCAFE, len(extra_payload)) + extra_payload
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(info, b"safe")
        raw = bytearray(output.getvalue())
        local = raw.index(b"PK\x03\x04")
        central = raw.index(b"PK\x01\x02")
        if location == "local":
            name_size, _extra_size = struct.unpack_from("<HH", raw, local + 26)
            offset = local + 30 + name_size + 4
        else:
            name_size, _extra_size = struct.unpack_from("<HH", raw, central + 28)
            offset = central + 46 + name_size + 4
        raw[offset : offset + len(sentinel)] = sentinel
        return preamble + bytes(raw)

    def test_scans_zip_local_and_central_extra_fields_and_preamble(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"zip-structural-runtime-sentinel"
            cases = {
                "local-extra.zip": self._zip_with_distinct_extra_sentinel(
                    sentinel, location="local"
                ),
                "central-extra.zip": self._zip_with_distinct_extra_sentinel(
                    sentinel, location="central"
                ),
                "preamble.zip": self._zip_with_distinct_extra_sentinel(
                    b"harmless-structural-placeholder",
                    location="central",
                    preamble=sentinel,
                ),
            }
            for name, payload in cases.items():
                artifact = root / name
                artifact.write_bytes(payload)
                with self.subTest(name=name):
                    result = self._run(artifact, sentinel=sentinel)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

    def test_scans_tar_trailer_and_nested_archive_structural_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"nested-structural-runtime-sentinel"
            tar_trailer = self._tar_bytes("safe.txt", b"safe") + sentinel
            inner_zip = self._zip_with_distinct_extra_sentinel(
                sentinel, location="central"
            )
            cases = {
                "trailer.tar": tar_trailer,
                "nested-tar.zip": self._zip_bytes("inner.tar", tar_trailer),
                "nested-extra.tar": self._tar_bytes("inner.zip", inner_zip),
                "nested-preamble.zip": self._zip_bytes(
                    "inner.zip", sentinel + self._zip_bytes("safe.txt", b"safe")
                ),
            }
            for name, payload in cases.items():
                artifact = root / name
                artifact.write_bytes(payload)
                with self.subTest(name=name):
                    result = self._run(artifact, sentinel=sentinel)
                    expected = 2 if name in {"trailer.tar", "nested-tar.zip"} else 1
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    if expected == 2:
                        self.assertIn("after TAR payload", result.stderr)
                    self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

    def test_manifest_errors_never_echo_manifest_path_or_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret = "manifest-filename-secret-sentinel"
            manifest = root / secret
            manifest.write_bytes(b"duplicate-secret\nduplicate-secret\n")
            artifact = root / "safe.bin"
            artifact.write_bytes(b"safe")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCANNER),
                    "--sentinels-file",
                    str(manifest),
                    str(artifact),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("<sentinel manifest>", result.stderr)
            self.assertNotIn(secret, result.stdout + result.stderr)
            self.assertNotIn("duplicate-secret", result.stdout + result.stderr)

    def test_same_inode_mutation_of_already_read_bytes_fails_closed(self):
        scanner = self._scanner_module()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "mutable.bin"
            artifact.write_bytes(b"a" * (scanner._CHUNK_BYTES * 2))
            original = scanner._read_chunk
            changed = False

            def mutate_after_first_read(stream, size, label, budget=None):
                nonlocal changed
                chunk = original(stream, size, label, budget)
                if chunk and not changed:
                    changed = True
                    with artifact.open("r+b", buffering=0) as writer:
                        writer.write(b"b")
                    os.utime(artifact, None)
                return chunk

            with (
                patch.object(scanner, "_read_chunk", mutate_after_first_read),
                self.assertRaisesRegex(scanner.ScanError, "changed while being scanned"),
            ):
                scanner.scan([artifact])

    def test_many_top_level_files_share_file_raw_and_work_budgets(self):
        scanner = self._scanner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                (root / f"{index}.bin").write_bytes(b"safe")
            with (
                patch.object(scanner, "_MAX_TRAVERSAL_ENTRIES", 2),
                self.assertRaisesRegex(scanner.ScanError, "traversal count"),
            ):
                scanner.scan([root])
            with (
                patch.object(scanner, "_MAX_SCANNED_FILES", 2),
                self.assertRaisesRegex(scanner.ScanError, "file count"),
            ):
                scanner.scan([root])
            with (
                patch.object(scanner, "_MAX_TOP_LEVEL_RAW_BYTES", 11),
                self.assertRaisesRegex(scanner.ScanError, "top-level raw bytes"),
            ):
                scanner.scan([root])
            with (
                patch.object(scanner, "_MAX_ARCHIVE_WORK_BYTES", 11),
                self.assertRaisesRegex(scanner.ScanError, "scan work"),
            ):
                scanner.scan([root])

    def test_rejects_zip_local_central_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "mismatch.zip"
            raw = bytearray(self._zip_bytes("safe.txt", b"safe"))
            local = raw.index(b"PK\x03\x04")
            name_size = struct.unpack_from("<H", raw, local + 26)[0]
            self.assertEqual(name_size, len(b"safe.txt"))
            raw[local + 30 : local + 30 + name_size] = b"evil.txt"
            artifact.write_bytes(raw)

            result = self._run(artifact)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("local and central metadata", result.stderr)

    def test_rejects_concatenated_gzip_tar_streams_without_echoing_trailer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"concatenated-gzip-runtime-sentinel"
            artifact = root / "concatenated.tar.gz"
            artifact.write_bytes(
                self._tar_bytes("safe.txt", b"safe", gzip=True)
                + self._tar_bytes("trailer.txt", sentinel, gzip=True)
            )

            result = self._run(artifact, sentinel=sentinel)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("concatenated or trailing gzip", result.stderr)
            self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

    def test_zip_directories_must_have_no_payload_and_valid_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"zip-directory-payload-sentinel"
            cases = {
                "stored.zip": self._zip_directory_bytes(
                    sentinel, compression=zipfile.ZIP_STORED
                ),
                "deflated.zip": self._zip_directory_bytes(
                    sentinel, compression=zipfile.ZIP_DEFLATED
                ),
                "bad-attrs.zip": self._zip_directory_bytes(
                    b"", compression=zipfile.ZIP_STORED, external_attr=0o100644 << 16
                ),
                "unknown-host-attrs.zip": self._zip_directory_bytes(
                    b"",
                    compression=zipfile.ZIP_STORED,
                    external_attr=0x10,
                    create_system=42,
                ),
                "empty-deflated.zip": self._zip_directory_bytes(
                    b"", compression=zipfile.ZIP_DEFLATED
                ),
                "nested.zip": self._zip_bytes(
                    "inner.zip",
                    self._zip_directory_bytes(
                        sentinel, compression=zipfile.ZIP_DEFLATED
                    ),
                ),
                "nested-stored.zip": self._zip_bytes(
                    "inner.zip",
                    self._zip_directory_bytes(
                        sentinel, compression=zipfile.ZIP_STORED
                    ),
                ),
            }
            for name, payload in cases.items():
                artifact = root / name
                artifact.write_bytes(payload)
                with self.subTest(name=name):
                    result = self._run(artifact, sentinel=sentinel)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("ZIP directory", result.stderr)
                    self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

            valid_windows = root / "valid-windows-directory.zip"
            valid_windows.write_bytes(
                self._zip_directory_bytes(
                    b"",
                    compression=zipfile.ZIP_STORED,
                    external_attr=0x10,
                    create_system=0,
                )
            )
            result = self._run(valid_windows)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            valid_unix = root / "valid-unix-directory-without-dos-bit.zip"
            valid_unix.write_bytes(
                self._zip_directory_bytes(
                    b"",
                    compression=zipfile.ZIP_STORED,
                    external_attr=0o040775 << 16,
                    create_system=3,
                )
            )
            result = self._run(valid_unix)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_zip_regular_members_reject_directory_and_nonregular_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                ("dos-directory-bit.zip", 0, 0x10),
                ("unix-directory-mode.zip", 3, (0o040755 << 16) | 0x10),
                ("unix-symlink-mode.zip", 3, 0o120777 << 16),
            )
            for name, create_system, external_attr in cases:
                artifact = root / name
                info = zipfile.ZipInfo("regular.txt")
                info.create_system = create_system
                info.external_attr = external_attr
                with zipfile.ZipFile(artifact, "w") as archive:
                    archive.writestr(info, b"safe")
                with self.subTest(name=name):
                    result = self._run(artifact)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("attributes", result.stderr)

    def test_zip_directories_use_real_sizes_and_share_member_budgets(self):
        scanner = self._scanner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._zip_directory_bytes(
                b"x", compression=zipfile.ZIP_STORED
            )
            artifact = root / "directory.zip"
            artifact.write_bytes(payload)
            with (
                patch.object(scanner, "_MAX_ARCHIVE_TOTAL_BYTES", 0),
                self.assertRaisesRegex(scanner.ScanError, "expanded size"),
            ):
                scanner.scan([artifact])

            two_directories = root / "two-directories.zip"
            with zipfile.ZipFile(two_directories, "w") as archive:
                for name in ("one/", "two/"):
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (0o40775 << 16) | 0x10
                    archive.writestr(info, b"")
            with (
                patch.object(scanner, "_MAX_ARCHIVE_MEMBERS", 1),
                self.assertRaisesRegex(scanner.ScanError, "member count"),
            ):
                scanner.scan([two_directories])

    def test_zip_directory_still_validates_local_and_central_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "directory-mismatch.zip"
            raw = bytearray(
                self._zip_directory_bytes(b"", compression=zipfile.ZIP_STORED)
            )
            local = raw.index(b"PK\x03\x04")
            struct.pack_into("<L", raw, local + 18, 1)
            artifact.write_bytes(raw)

            result = self._run(artifact)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("local and central sizes", result.stderr)

    def test_zip_directory_rejects_data_descriptor_local_size_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "directory-descriptor.zip"
            raw = bytearray(
                self._zip_directory_bytes(b"", compression=zipfile.ZIP_STORED)
            )
            local = raw.index(b"PK\x03\x04")
            central = raw.index(b"PK\x01\x02")
            local_flags = struct.unpack_from("<H", raw, local + 6)[0] | 0x08
            central_flags = struct.unpack_from("<H", raw, central + 8)[0] | 0x08
            struct.pack_into("<H", raw, local + 6, local_flags)
            struct.pack_into("<H", raw, central + 8, central_flags)
            struct.pack_into("<LL", raw, local + 18, 123, 456)
            artifact.write_bytes(raw)

            result = self._run(artifact)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("data descriptor", result.stderr)

    def test_scans_bzip2_and_xz_tar_aliases_magic_and_nested_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"compressed-tar-runtime-sentinel"
            bz_payload = self._tar_bytes(
                "package/leak.bin", sentinel, compression="bz2"
            )
            xz_payload = self._tar_bytes(
                "package/leak.bin", sentinel, compression="xz"
            )
            cases = {
                "artifact.tar.bz2": bz_payload,
                "artifact.tbz": bz_payload,
                "artifact.tbz2": bz_payload,
                "artifact.tar.xz": xz_payload,
                "artifact.txz": xz_payload,
                "extensionless-bzip": bz_payload,
                "disguised.data": xz_payload,
                "disguised-as-zip.zip": bz_payload,
                "nested-bzip.zip": self._zip_bytes("payload.bin", bz_payload),
                "nested-xz.tar": self._tar_bytes("payload.dat", xz_payload),
            }
            for name, payload in cases.items():
                artifact = root / name
                artifact.write_bytes(payload)
                with self.subTest(name=name):
                    result = self._run(artifact, sentinel=sentinel)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn("leak.bin", result.stdout)
                    self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

    def test_scans_compressed_tar_header_padding_trailer_and_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"raw-compressed-tar-sentinel"
            header_raw = self._tar_bytes(f"package/{sentinel.decode()}", b"safe")
            padding_raw = bytearray(self._tar_bytes("safe.bin", b"x" * 65_017))
            padding_raw[65_529 : 65_529 + len(sentinel)] = sentinel
            trailer_raw = self._tar_bytes("safe.bin", b"safe") + sentinel
            for compression, compress in (("bz2", bz2.compress), ("xz", lzma.compress)):
                for location, raw in (
                    ("header", header_raw),
                    ("padding-boundary", bytes(padding_raw)),
                    ("trailer", trailer_raw),
                ):
                    artifact = root / f"{location}.tar.{compression}"
                    artifact.write_bytes(compress(raw))
                    with self.subTest(compression=compression, location=location):
                        result = self._run(artifact, sentinel=sentinel)
                        expected = 1 if location == "header" else 2
                        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                        if expected == 2:
                            self.assertIn("after TAR payload", result.stderr)
                        self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

    def test_rejects_concatenated_corrupt_truncated_and_bomb_compressed_tars(self):
        scanner = self._scanner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"compressed-tar-error-secret"
            for compression, compress, suffix in (
                ("bzip2", bz2.compress, "bz2"),
                ("xz", lzma.compress, "xz"),
            ):
                raw = self._tar_bytes("safe.txt", b"safe")
                concatenated = root / f"concatenated.tar.{suffix}"
                concatenated.write_bytes(compress(raw) + compress(sentinel))
                result = self._run(concatenated, sentinel=sentinel)
                with self.subTest(compression=compression, case="concatenated"):
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("concatenated or trailing", result.stderr)
                    self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

                truncated = root / f"truncated.tar.{suffix}"
                truncated.write_bytes(compress(raw)[:-4])
                result = self._run(truncated)
                with self.subTest(compression=compression, case="truncated"):
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("truncated", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

                corrupt = root / f"corrupt.tar.{suffix}"
                payload = bytearray(compress(raw))
                payload[len(payload) // 2] ^= 0xFF
                corrupt.write_bytes(payload)
                result = self._run(corrupt)
                with self.subTest(compression=compression, case="corrupt"):
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("unreadable or corrupt", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

                ratio_bomb = root / f"ratio-bomb.tar.{suffix}"
                ratio_bomb.write_bytes(
                    compress(self._tar_bytes("zeros.bin", b"0" * 2_000_000))
                )
                result = self._run(ratio_bomb)
                with self.subTest(compression=compression, case="compression-ratio"):
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("compression ratio", result.stderr)

                bomb = root / f"bomb.tar.{suffix}"
                bomb.write_bytes(compress(raw))
                with (
                    self.subTest(compression=compression, case="expanded-budget"),
                    patch.object(scanner, "_MAX_ARCHIVE_TOTAL_BYTES", 1_000),
                    self.assertRaisesRegex(scanner.ScanError, "expanded size"),
                ):
                    scanner.scan([bomb])
                with (
                    self.subTest(compression=compression, case="work-budget"),
                    patch.object(scanner, "_MAX_ARCHIVE_WORK_BYTES", 100),
                    self.assertRaisesRegex(scanner.ScanError, "scan work"),
                ):
                    scanner.scan([bomb])

                second = root / f"second.tar.{suffix}"
                second.write_bytes(compress(raw))
                with (
                    self.subTest(compression=compression, case="global-expanded-budget"),
                    patch.object(scanner, "_MAX_ARCHIVE_TOTAL_BYTES", 15_000),
                    self.assertRaisesRegex(scanner.ScanError, "expanded size"),
                ):
                    scanner.scan([bomb, second])

    def test_zstd_tar_suffix_and_magic_fail_closed_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in (
                ("archive.tar.zst", b"not-zstd"),
                ("archive.tar.zstd", b"not-zstd"),
                ("archive.tzst", b"not-zstd"),
                ("zstd-magic.bin", b"\x28\xb5\x2f\xfdnot-zstd"),
                ("zstd-skippable.bin", b"\x50\x2a\x4d\x18not-zstd"),
            ):
                artifact = root / name
                artifact.write_bytes(payload)
                with self.subTest(name=name):
                    result = self._run(artifact)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("zstd-compressed TAR is not supported", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_lzma_alone_tar_xz_fails_closed_without_secret_echo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"lzma-alone-error-secret"
            safe_tar = self._tar_bytes("safe.txt", b"safe")
            artifact = root / f"{sentinel.decode()}.tar.xz"
            artifact.write_bytes(
                lzma.compress(safe_tar, format=lzma.FORMAT_ALONE)
                + lzma.compress(sentinel, format=lzma.FORMAT_ALONE)
            )

            result = self._run(artifact, sentinel=sentinel)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("LZMA-alone compressed TAR is not supported", result.stderr)
            self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_lzma_alone_tar_cannot_hide_padding_top_level_or_nested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"lzma-alone-padding-sentinel"
            raw = bytearray(self._tar_bytes("safe.txt", b"x"))
            raw[513 : 513 + len(sentinel)] = sentinel
            payload = lzma.compress(raw, format=lzma.FORMAT_ALONE)
            small_dictionary_payload = bytearray(payload)
            struct.pack_into("<I", small_dictionary_payload, 1, 2_048)
            cases = {
                "hidden.tar": payload,
                "extensionless-lzma": payload,
                "nested.zip": self._zip_bytes("hidden.tar", payload),
                "nested-small-dictionary.zip": self._zip_bytes(
                    "artifact.bin", bytes(small_dictionary_payload)
                ),
            }
            for name, content in cases.items():
                artifact = root / name
                artifact.write_bytes(content)
                with self.subTest(name=name):
                    result = self._run(artifact, sentinel=sentinel)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_rejects_tar_content_after_logical_eof_in_all_compressions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"second-tar-nested-sentinel"
            inner = self._zip_bytes("secret.txt", sentinel)
            concatenated = self._tar_bytes("safe.txt", b"safe") + self._tar_bytes(
                "inner.zip", inner
            )
            cases = {
                "concatenated.tar": concatenated,
                "concatenated.tar.gz": zlib.compress(concatenated, wbits=31),
                "concatenated.tar.bz2": bz2.compress(concatenated),
                "concatenated.tar.xz": lzma.compress(concatenated),
                "nested.zip": self._zip_bytes("concatenated.tar", concatenated),
            }
            for name, content in cases.items():
                artifact = root / name
                artifact.write_bytes(content)
                with self.subTest(name=name):
                    result = self._run(artifact, sentinel=sentinel)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("after TAR payload", result.stderr)
                    self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

    def test_rejects_tar_directories_with_payload_and_charges_real_size(self):
        scanner = self._scanner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = b"tar-directory-payload-sentinel"
            inner = self._zip_bytes("secret.txt", sentinel)
            output = io.BytesIO()
            with tarfile.open(fileobj=output, mode="w") as archive:
                info = tarfile.TarInfo("directory/")
                info.type = tarfile.DIRTYPE
                info.size = len(inner)
                archive.addfile(info, io.BytesIO(inner))
            raw = output.getvalue()
            cases = {
                "directory.tar": raw,
                "directory.tar.gz": zlib.compress(raw, wbits=31),
                "directory.tar.bz2": bz2.compress(raw),
                "directory.tar.xz": lzma.compress(raw),
                "nested.zip": self._zip_bytes("directory.tar", raw),
            }
            for name, content in cases.items():
                artifact = root / name
                artifact.write_bytes(content)
                with self.subTest(name=name):
                    result = self._run(artifact, sentinel=sentinel)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("TAR directory contains a payload", result.stderr)
                    self.assertNotIn(sentinel.decode(), result.stdout + result.stderr)

            artifact = root / "budget.tar"
            artifact.write_bytes(raw)
            with (
                patch.object(scanner, "_MAX_ARCHIVE_TOTAL_BYTES", len(inner) - 1),
                self.assertRaisesRegex(scanner.ScanError, "expanded size"),
            ):
                scanner.scan([artifact])


if __name__ == "__main__":
    unittest.main()
