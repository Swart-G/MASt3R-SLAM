"""Process one dataset image through the live-network processor."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from mast3r_slam.mobile.frame_source import NetworkFrame
from mast3r_slam.mobile.mast3r_processor import Mast3rProcessor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise SystemExit("JPEG encoding failed")
    processor = Mast3rProcessor(args.config, args.device)
    started = time.monotonic()
    result = processor.process(
        NetworkFrame(0, time.time_ns(), encoded.tobytes()), relocalizing=False
    )
    checkpoint = processor.checkpoint()
    print(
        {
            "tracking": result.tracking.value,
            "keyframes": result.keyframe_count,
            "preview_bytes": len(result.preview_jpeg or b""),
            "checkpoint_arrays": sorted(checkpoint.arrays),
            "seconds": round(time.monotonic() - started, 2),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
