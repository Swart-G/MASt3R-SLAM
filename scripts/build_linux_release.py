#!/usr/bin/env python3
"""Build the self-extracting Linux launcher and append its two payloads."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shutil
import struct
import subprocess
import tempfile
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAGIC = b"M3SLAM1!"
INCLUDE = (
    "LICENSE.md",
    "README.md",
    "main.py",
    "pyproject.toml",
    "setup.py",
    "config",
    "desktop/download_checkpoints.py",
    "desktop/prepare_build_environment.py",
    "mast3r_slam",
    "protocol",
    "resources",
    "thirdparty/mast3r",
    "thirdparty/in3d",
    "thirdparty/eigen/Eigen",
    "thirdparty/eigen/unsupported/Eigen",
)
EXCLUDED_PARTS = {
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "build",
    "dist",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".o", ".so"}


def included_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for relative_name in INCLUDE:
        source = ROOT / relative_name
        if not source.exists():
            raise FileNotFoundError(f"required bundle input is missing: {source}")
        candidates = [source] if source.is_file() else source.rglob("*")
        for candidate in candidates:
            relative = candidate.relative_to(ROOT)
            if not candidate.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if candidate.suffix in EXCLUDED_SUFFIXES or any(part.endswith(".egg-info") for part in relative.parts):
                continue
            files.append(candidate)
    return sorted(set(files), key=lambda item: item.as_posix())


def make_archive(destination: pathlib.Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = included_files()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in files:
            relative = source.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if os.access(source, os.X_OK) else 0o644) << 16
            data = source.read_bytes()
            archive.writestr(info, data, compresslevel=9)
            digest.update(relative.encode("utf-8") + b"\0" + data)
    return digest.hexdigest(), len(files)


def build(output: pathlib.Path, uv_binary: pathlib.Path, compiler: str) -> None:
    if not uv_binary.is_file():
        raise FileNotFoundError(f"uv binary not found: {uv_binary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mast3r-linux-release-") as temporary_name:
        temporary = pathlib.Path(temporary_name)
        archive = temporary / "application.zip"
        base = temporary / "launcher"
        version, file_count = make_archive(archive)
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-O2",
                "-s",
                "-Wall",
                "-Wextra",
                "-Werror",
                f'-DBUNDLE_VERSION="{version[:16]}"',
                str(ROOT / "desktop/launcher.c"),
                "-o",
                str(base),
            ],
            check=True,
        )
        temporary_output = output.with_suffix(output.suffix + ".tmp")
        with temporary_output.open("wb") as result:
            with base.open("rb") as source:
                shutil.copyfileobj(source, result)
            with uv_binary.open("rb") as source:
                shutil.copyfileobj(source, result)
            with archive.open("rb") as source:
                shutil.copyfileobj(source, result)
            result.write(struct.pack("<8sQQ", MAGIC, uv_binary.stat().st_size, archive.stat().st_size))
            result.flush()
            os.fsync(result.fileno())
        temporary_output.chmod(0o755)
        os.replace(temporary_output, output)
        print(f"Bundled {file_count} application files (version {version[:16]}).")
        print(f"Created {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB).")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--uv", type=pathlib.Path, required=True)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = parser.parse_args()
    build(args.output.resolve(), args.uv.resolve(), args.cc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
