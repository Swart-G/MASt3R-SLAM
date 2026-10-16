from __future__ import annotations

import numpy as np


def rotate_pinhole_intrinsics(
    intrinsics: tuple[float, ...] | list[float],
    width: int,
    height: int,
    rotation_degrees: int,
) -> tuple[tuple[float, float, float, float], tuple[int, int]]:
    """Rotate Camera2 intrinsics with the clockwise image rotation used by the server."""

    if len(intrinsics) < 4:
        focal = max(width, height) * 0.8660254  # approximately a 60 degree FOV
        values = (focal, focal, (width - 1) * 0.5, (height - 1) * 0.5)
    else:
        values = tuple(float(value) for value in intrinsics[:4])
    fx, fy, cx, cy = values
    rotation = rotation_degrees % 360
    if rotation == 0:
        return (fx, fy, cx, cy), (width, height)
    if rotation == 90:
        return (fy, fx, height - 1 - cy, cx), (height, width)
    if rotation == 180:
        return (fx, fy, width - 1 - cx, height - 1 - cy), (width, height)
    if rotation == 270:
        return (fy, fx, cy, width - 1 - cx), (height, width)
    raise ValueError("camera orientation must be a multiple of 90 degrees")


def project_camera_points(
    points: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project OpenCV camera coordinates (+X right, +Y down, +Z forward)."""

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-4)
    indices = np.flatnonzero(valid)
    visible = points[valid]
    if not len(visible):
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, indices
    fx, fy, cx, cy = intrinsics
    source_width, source_height = source_size
    output_width, output_height = output_size
    scale_x = output_width / max(source_width, 1)
    scale_y = output_height / max(source_height, 1)
    u = (fx * visible[:, 0] / visible[:, 2] + cx) * scale_x
    v = (fy * visible[:, 1] / visible[:, 2] + cy) * scale_y
    inside = (u >= 0) & (u < output_width) & (v >= 0) & (v < output_height)
    return u[inside], v[inside], visible[inside, 2], indices[inside]


class OffscreenMapRenderer:
    """CPU renderer of the accumulated map from the tracked frame's camera pose."""

    def __init__(self, width: int = 960, height: int = 540) -> None:
        self.width = width
        self.height = height

    def render(
        self,
        camera_points: np.ndarray,
        colors: np.ndarray,
        *,
        intrinsics: tuple[float, ...] | list[float] = (),
        image_size: tuple[int, int] = (1280, 720),
        rotation_degrees: int = 0,
    ) -> bytes:
        import cv2

        points = np.asarray(camera_points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if len(points) != len(colors):
            raise ValueError("points and colors must have the same length")
        camera, oriented_size = rotate_pinhole_intrinsics(
            intrinsics, image_size[0], image_size[1], rotation_degrees
        )
        u, v, depth, indices = project_camera_points(
            points, camera, oriented_size, (self.width, self.height)
        )
        canvas = np.full((self.height, self.width, 3), 18, dtype=np.uint8)
        if len(indices):
            x = np.rint(u).astype(np.int32).clip(0, self.width - 1)
            y = np.rint(v).astype(np.int32).clip(0, self.height - 1)
            visible_colors = colors[indices][:, ::-1]
            # A 3x3 splat keeps the reconstructed view readable while a nearest
            # sample per pixel provides deterministic depth occlusion.
            for offset_y in (-1, 0, 1):
                py = (y + offset_y).clip(0, self.height - 1)
                for offset_x in (-1, 0, 1):
                    px = (x + offset_x).clip(0, self.width - 1)
                    linear = py * self.width + px
                    near_to_far = np.argsort(depth, kind="stable")
                    _, first = np.unique(linear[near_to_far], return_index=True)
                    selected = near_to_far[first]
                    canvas[py[selected], px[selected]] = visible_colors[selected]
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 84])
        if not ok:
            raise RuntimeError("failed to encode offscreen preview")
        return encoded.tobytes()
