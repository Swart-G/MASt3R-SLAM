from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import struct
import uuid
from collections.abc import Iterable

import numpy as np

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - installation dependency
    zstd = None


MAGIC = b"M3PC1"
MESH_MAGIC = b"M3MS1"
MAX_POINTS = 750_000


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotChunk:
    index: int
    filename: str
    compressed_size: int
    uncompressed_size: int
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotManifest:
    snapshot_id: str
    point_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    encoding: str
    chunks: tuple[SnapshotChunk, ...]


def filter_and_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    confidence_threshold: float = 0.0,
    voxel_size: float = 0.01,
    max_points: int = MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("points and colors must have the same length")
    valid = np.isfinite(points).all(axis=1)
    confidence_values = None
    if confidence is not None:
        confidence_values = np.asarray(confidence, dtype=np.float32).reshape(-1)
        valid &= confidence_values >= confidence_threshold
    points, colors = points[valid], colors[valid]
    if confidence_values is not None:
        confidence_values = confidence_values[valid]
    if not len(points):
        raise ValueError("snapshot contains no valid points")
    if voxel_size > 0:
        origin = points.min(axis=0)
        voxels = np.floor((points - origin) / voxel_size).astype(np.int64)
        if confidence_values is None:
            _, indices = np.unique(voxels, axis=0, return_index=True)
            indices.sort()
        else:
            # Stable deterministic representative: highest-confidence point in
            # each voxel, rather than whichever input happens to occur first.
            order = np.lexsort(
                (-confidence_values, voxels[:, 2], voxels[:, 1], voxels[:, 0])
            )
            _, first = np.unique(voxels[order], axis=0, return_index=True)
            indices = order[first]
        points, colors = points[indices], colors[indices]
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points, colors = points[indices], colors[indices]
    return points, colors


def create_snapshot(
    output_dir: str | pathlib.Path,
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    confidence_threshold: float = 0.0,
    voxel_size: float = 0.01,
    max_points: int = MAX_POINTS,
    points_per_chunk: int = 100_000,
) -> SnapshotManifest:
    if zstd is None:
        raise RuntimeError("zstandard is required to create mobile snapshots")
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    points, colors = filter_and_downsample(
        points,
        colors,
        confidence,
        confidence_threshold=confidence_threshold,
        voxel_size=voxel_size,
        max_points=max_points,
    )
    bounds_min, bounds_max = points.min(axis=0), points.max(axis=0)
    chunks: list[SnapshotChunk] = []
    compressor = zstd.ZstdCompressor(level=5)
    for index, start in enumerate(range(0, len(points), points_per_chunk)):
        chunk_points = points[start : start + points_per_chunk]
        chunk_colors = colors[start : start + points_per_chunk]
        local_min, local_max = chunk_points.min(axis=0), chunk_points.max(axis=0)
        span = np.maximum(local_max - local_min, np.finfo(np.float32).eps)
        quantized = np.rint((chunk_points - local_min) / span * 65535).astype("<u2")
        header = MAGIC + struct.pack(
            "<I6f", len(chunk_points), *local_min.tolist(), *local_max.tolist()
        )
        raw = header + quantized.tobytes(order="C") + chunk_colors.tobytes(order="C")
        compressed = compressor.compress(raw)
        filename = f"chunk-{index:04d}.m3pc.zst"
        (output / filename).write_bytes(compressed)
        chunks.append(
            SnapshotChunk(
                index,
                filename,
                len(compressed),
                len(raw),
                hashlib.sha256(compressed).hexdigest(),
            )
        )
    manifest = SnapshotManifest(
        uuid.uuid4().hex,
        len(points),
        tuple(float(x) for x in bounds_min),
        tuple(float(x) for x in bounds_max),
        "m3pc-v1:xyz-u16-local,rgb-u8,zstd",
        tuple(chunks),
    )
    (output / "manifest.json").write_text(
        json.dumps(dataclasses.asdict(manifest), indent=2, sort_keys=True)
    )
    return manifest


def decode_chunk(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    if zstd is None:
        raise RuntimeError("zstandard is required to decode mobile snapshots")
    raw = zstd.ZstdDecompressor().decompress(data)
    header_size = len(MAGIC) + struct.calcsize("<I6f")
    if raw[: len(MAGIC)] != MAGIC or len(raw) < header_size:
        raise ValueError("invalid mobile point-cloud chunk")
    count, *bounds = struct.unpack("<I6f", raw[len(MAGIC) : header_size])
    xyz_size = count * 3 * 2
    rgb_size = count * 3
    if len(raw) != header_size + xyz_size + rgb_size:
        raise ValueError("truncated mobile point-cloud chunk")
    quantized = np.frombuffer(raw, dtype="<u2", count=count * 3, offset=header_size).reshape(-1, 3)
    colors = np.frombuffer(raw, dtype=np.uint8, count=count * 3, offset=header_size + xyz_size).reshape(-1, 3).copy()
    local_min = np.asarray(bounds[:3], dtype=np.float32)
    local_max = np.asarray(bounds[3:], dtype=np.float32)
    points = local_min + quantized.astype(np.float32) / 65535.0 * (local_max - local_min)
    return points, colors


def triangulate_grid(
    world_points: np.ndarray,
    camera_points: np.ndarray,
    confidence: np.ndarray,
    shape: tuple[int, int],
    *,
    confidence_quantile: float = 0.2,
    max_relative_edge: float = 0.08,
) -> np.ndarray:
    """Build surface triangles from an organized MASt3R point map.

    Connectivity comes from neighboring image pixels. Triangles crossing a
    depth discontinuity are rejected, preventing foreground and background
    objects from being joined by long sheets.
    """

    height, width = shape
    world = np.asarray(world_points, dtype=np.float32).reshape(height, width, 3)
    camera = np.asarray(camera_points, dtype=np.float32).reshape(height, width, 3)
    scores = np.asarray(confidence, dtype=np.float32).reshape(height, width)
    finite_scores = scores[np.isfinite(scores)]
    if not len(finite_scores):
        return np.empty((0, 3, 3), dtype=np.float32)
    threshold = float(np.quantile(finite_scores, confidence_quantile))
    valid = (
        np.isfinite(world).all(axis=2)
        & np.isfinite(camera).all(axis=2)
        & (camera[:, :, 2] > 1e-4)
        & (scores >= threshold)
    )
    ids = np.arange(height * width, dtype=np.int64).reshape(height, width)
    top_left = ids[:-1, :-1].reshape(-1)
    top_right = ids[:-1, 1:].reshape(-1)
    bottom_left = ids[1:, :-1].reshape(-1)
    bottom_right = ids[1:, 1:].reshape(-1)
    faces = np.concatenate(
        (
            np.column_stack((top_left, top_right, bottom_right)),
            np.column_stack((top_left, bottom_right, bottom_left)),
        ),
        axis=0,
    )
    flat_valid = valid.reshape(-1)
    faces = faces[flat_valid[faces].all(axis=1)]
    if not len(faces):
        return np.empty((0, 3, 3), dtype=np.float32)
    camera_triangles = camera.reshape(-1, 3)[faces]
    edges = np.stack(
        (
            np.linalg.norm(camera_triangles[:, 1] - camera_triangles[:, 0], axis=1),
            np.linalg.norm(camera_triangles[:, 2] - camera_triangles[:, 1], axis=1),
            np.linalg.norm(camera_triangles[:, 0] - camera_triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    depth = np.median(np.linalg.norm(camera_triangles, axis=2), axis=1)
    continuous = edges.max(axis=1) <= np.maximum(depth * max_relative_edge, 1e-4)
    return world.reshape(-1, 3)[faces[continuous]]


def create_mesh_snapshot(
    output_dir: str | pathlib.Path,
    surfaces: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]],
    *,
    max_vertices: int = MAX_POINTS,
    triangles_per_chunk: int = 50_000,
    surface_count: int | None = None,
) -> SnapshotManifest:
    """Create a mobile mesh while retaining at most one keyframe plus one chunk."""

    if zstd is None:
        raise RuntimeError("zstandard is required to create mobile snapshots")
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    max_triangles = max_vertices // 3
    per_surface = max(1, max_triangles // max(surface_count or 1, 1))
    compressor = zstd.ZstdCompressor(level=5)
    chunks: list[SnapshotChunk] = []
    pending: list[np.ndarray] = []
    pending_count = 0
    total_triangles = 0
    bounds_min: np.ndarray | None = None
    bounds_max: np.ndarray | None = None

    def write_chunk(triangles: np.ndarray) -> None:
        nonlocal bounds_min, bounds_max
        chunk_vertices = triangles.reshape(-1, 3)
        local_min, local_max = chunk_vertices.min(axis=0), chunk_vertices.max(axis=0)
        bounds_min = local_min if bounds_min is None else np.minimum(bounds_min, local_min)
        bounds_max = local_max if bounds_max is None else np.maximum(bounds_max, local_max)
        span = np.maximum(local_max - local_min, np.finfo(np.float32).eps)
        quantized = np.rint((chunk_vertices - local_min) / span * 65535).astype("<u2")
        header = MESH_MAGIC + struct.pack(
            "<I6f", len(triangles), *local_min.tolist(), *local_max.tolist()
        )
        raw = header + quantized.tobytes(order="C")
        compressed = compressor.compress(raw)
        index = len(chunks)
        filename = f"chunk-{index:04d}.m3ms.zst"
        (output / filename).write_bytes(compressed)
        chunks.append(
            SnapshotChunk(index, filename, len(compressed), len(raw), hashlib.sha256(compressed).hexdigest())
        )

    for world, camera, confidence, shape in surfaces:
        if total_triangles >= max_triangles:
            break
        triangles = triangulate_grid(world, camera, confidence, shape)
        if len(triangles) > per_surface:
            indices = np.linspace(0, len(triangles) - 1, per_surface, dtype=np.int64)
            triangles = triangles[indices]
        triangles = triangles[: max_triangles - total_triangles]
        cursor = 0
        while cursor < len(triangles):
            take = min(triangles_per_chunk - pending_count, len(triangles) - cursor)
            pending.append(triangles[cursor : cursor + take])
            pending_count += take
            total_triangles += take
            cursor += take
            if pending_count == triangles_per_chunk:
                write_chunk(np.concatenate(pending, axis=0))
                pending.clear()
                pending_count = 0
        del triangles, world, camera, confidence
    if pending_count:
        write_chunk(np.concatenate(pending, axis=0))
    if not chunks or bounds_min is None or bounds_max is None:
        raise ValueError("snapshot contains no valid mesh triangles")
    manifest = SnapshotManifest(
        uuid.uuid4().hex,
        total_triangles * 3,
        tuple(float(value) for value in bounds_min),
        tuple(float(value) for value in bounds_max),
        "m3mesh-v1:triangles-xyz-u16-local,zstd",
        tuple(chunks),
    )
    (output / "manifest.json").write_text(
        json.dumps(dataclasses.asdict(manifest), indent=2, sort_keys=True)
    )
    return manifest


def decode_mesh_chunk(data: bytes) -> np.ndarray:
    if zstd is None:
        raise RuntimeError("zstandard is required to decode mobile snapshots")
    raw = zstd.ZstdDecompressor().decompress(data)
    header_size = len(MESH_MAGIC) + struct.calcsize("<I6f")
    if raw[: len(MESH_MAGIC)] != MESH_MAGIC or len(raw) < header_size:
        raise ValueError("invalid mobile mesh chunk")
    triangle_count, *bounds = struct.unpack("<I6f", raw[len(MESH_MAGIC) : header_size])
    vertex_count = triangle_count * 3
    xyz_size = vertex_count * 3 * 2
    if len(raw) != header_size + xyz_size:
        raise ValueError("truncated mobile mesh chunk")
    quantized = np.frombuffer(
        raw, dtype="<u2", count=vertex_count * 3, offset=header_size
    ).reshape(-1, 3)
    local_min = np.asarray(bounds[:3], dtype=np.float32)
    local_max = np.asarray(bounds[3:], dtype=np.float32)
    vertices = local_min + quantized.astype(np.float32) / 65535.0 * (local_max - local_min)
    return vertices.reshape(-1, 3, 3)
