from __future__ import annotations

import numpy as np


OPENCV_FROM_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def sim3_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert lietorch [t, qx, qy, qz, qw, scale] to a 4x4 Sim3."""

    values = np.asarray(pose, dtype=np.float64).reshape(-1)
    if len(values) != 8:
        raise ValueError("a Sim3 pose must contain eight values")
    q = values[3:7]
    q /= max(float(np.linalg.norm(q)), np.finfo(np.float64).eps)
    x, y, z, w = q
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = values[7] * rotation
    matrix[:3, 3] = values[:3]
    return matrix


def ar_world_from_slam_map(
    ar_world_from_gl_camera: np.ndarray,
    slam_map_from_cv_camera: np.ndarray,
) -> np.ndarray:
    """Align a saved SLAM map with ARCore world coordinates.

    ARCore camera poses use OpenGL axes (+Y up, camera looks -Z); MASt3R
    camera poses use OpenCV axes (+Y down, camera looks +Z). The returned Sim3
    retains the map scale and can be used directly as the mesh model matrix.
    """

    ar_pose = np.asarray(ar_world_from_gl_camera, dtype=np.float64).reshape(4, 4)
    slam_pose = np.asarray(slam_map_from_cv_camera, dtype=np.float64).reshape(4, 4)
    return ar_pose @ OPENCV_FROM_OPENGL @ np.linalg.inv(slam_pose)


def from_column_major(values: list[float] | tuple[float, ...]) -> np.ndarray:
    if len(values) != 16:
        raise ValueError("a pose matrix must contain 16 values")
    return np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")


def to_column_major(matrix: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(matrix).reshape(16, order="F"))
