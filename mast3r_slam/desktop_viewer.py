from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np


def _use_xwayland_for_open3d() -> None:
    """Avoid Open3D 0.19's broken native Wayland/GLEW window path."""
    if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
        if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
            os.environ.pop("WAYLAND_DISPLAY", None)
            os.environ["XDG_SESSION_TYPE"] = "x11"


def _checkpoint_geometry(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct world-space points and RGB colors from a paused checkpoint."""
    try:
        stored = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read checkpoint: {path}") from error

    with stored:
        missing = {"X", "T_WC", "uimg"}.difference(stored.files)
        if missing:
            raise RuntimeError(f"checkpoint has no reconstruction data: {', '.join(sorted(missing))}")
        points = stored["X"]
        poses = stored["T_WC"]
        images = stored["uimg"]
        count = min(len(points), len(poses), len(images))
        if count == 0:
            raise RuntimeError("checkpoint contains no keyframes")

        points_per_frame = points[0].reshape(-1, 3).shape[0]
        world = np.empty((count * points_per_frame, 3), dtype=np.float32)
        colors = np.empty_like(world)
        for index in range(count):
            camera_points = np.asarray(points[index], dtype=np.float32).reshape(-1, 3)
            frame_colors = np.asarray(images[index], dtype=np.float32).reshape(-1, 3)
            if len(camera_points) != points_per_frame or len(frame_colors) != points_per_frame:
                raise RuntimeError("checkpoint point map and image sizes do not match")

            # lietorch Sim3 storage is [tx, ty, tz, qx, qy, qz, qw, scale].
            pose = np.asarray(poses[index], dtype=np.float32).reshape(-1)
            if len(pose) != 8:
                raise RuntimeError("checkpoint contains an invalid Sim3 pose")
            translation, quaternion, scale = pose[:3], pose[3:7], pose[7]
            quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
            vector, scalar = quaternion[:3], quaternion[3]
            rotated = camera_points + 2 * np.cross(
                vector, np.cross(vector, camera_points) + scalar * camera_points
            )
            start = index * points_per_frame
            stop = start + points_per_frame
            world[start:stop] = scale * rotated + translation
            colors[start:stop] = np.clip(frame_colors, 0.0, 1.0)

    finite = np.isfinite(world).all(axis=1) & np.isfinite(colors).all(axis=1)
    if not finite.any():
        raise RuntimeError("checkpoint contains no finite reconstruction points")
    if not finite.all():
        world, colors = world[finite], colors[finite]
    return world, colors


def _load_cloud(o3d, requested_path: pathlib.Path):
    if requested_path.is_file():
        cloud = o3d.io.read_point_cloud(str(requested_path))
        if not cloud.has_points():
            raise RuntimeError(f"point cloud is empty or invalid: {requested_path}")
        return cloud

    checkpoint = requested_path.parent / "checkpoint.npz"
    if not checkpoint.is_file():
        raise RuntimeError(
            f"reconstruction is not finalized and its checkpoint is missing: {requested_path.parent}"
        )
    print("Finalized PLY is absent; building the view from the saved checkpoint...", flush=True)
    points, colors = _checkpoint_geometry(checkpoint)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    args = parser.parse_args()
    _use_xwayland_for_open3d()
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise SystemExit("Cannot open the 3D viewer: no graphical display is available")
    try:
        import open3d as o3d
    except ImportError as error:
        raise SystemExit("Install open3d to use the standalone project viewer") from error
    try:
        cloud = _load_cloud(o3d, pathlib.Path(args.ply))
    except RuntimeError as error:
        raise SystemExit(f"Cannot open reconstruction: {error}") from error
    o3d.visualization.draw_geometries([cloud], window_name="MASt3R-SLAM reconstruction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
