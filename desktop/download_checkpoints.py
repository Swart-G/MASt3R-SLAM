#!/usr/bin/env python3
"""Resumable first-run downloader for the MASt3R model assets."""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import time
import urllib.request


DEFAULT_BASE_URL = "https://download.europe.naverlabs.com/ComputerVision/MASt3R"
CHECKPOINTS = (
    (
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2",
    ),
    (
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
        "000a324f06f9369ab3e32d03dd4b8a838d48e98e3f3bf9d7dd034dce1bfd4c50",
    ),
    (
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
        "700af9e31ad0ca6159955dc80b4fd6e14234780b52f5c91884f6553991933c1a",
    ),
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: pathlib.Path, expected: str) -> None:
    if destination.is_file() and sha256(destination) == expected:
        print(f"Already verified: {destination.name}", flush=True)
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url)
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resumed = offset > 0 and getattr(response, "status", None) == 206
                if not resumed:
                    offset = 0
                mode = "ab" if resumed else "wb"
                remaining = response.headers.get("Content-Length")
                total = offset + int(remaining) if remaining else None
                completed = offset
                last_report = time.monotonic()
                print(f"Downloading {destination.name}", flush=True)
                with partial.open(mode) as output:
                    while block := response.read(8 * 1024 * 1024):
                        output.write(block)
                        completed += len(block)
                        now = time.monotonic()
                        if now - last_report >= 2:
                            mib = completed / 1024 / 1024
                            suffix = f" / {total / 1024 / 1024:.1f} MiB" if total else " MiB"
                            print(f"  {mib:.1f}{suffix}", flush=True)
                            last_report = now
            actual = sha256(partial)
            if actual != expected:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch: expected {expected}, received {actual}")
            os.replace(partial, destination)
            print(f"Verified: {destination.name}", flush=True)
            return
        except Exception as error:
            print(f"Attempt {attempt}/3 failed: {error}", flush=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 2)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: download_checkpoints.py DESTINATION", file=sys.stderr)
        return 2
    destination = pathlib.Path(arguments[0])
    destination.mkdir(parents=True, exist_ok=True)
    base_url = os.environ.get("MAST3R_SLAM_CHECKPOINT_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    try:
        for filename, expected in CHECKPOINTS:
            download(f"{base_url}/{filename}", destination / filename, expected)
    except Exception as error:
        print(f"Checkpoint download failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
