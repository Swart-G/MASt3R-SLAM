from __future__ import annotations

import collections.abc
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import struct
import tempfile
import uuid
import zipfile
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from .camera import CameraIdentity


PROJECT_FORMAT_VERSION = 1
CHECKPOINT_FORMAT_VERSION = 1
ProgressCallback = Callable[[float, str], None]


class ProjectError(RuntimeError):
    pass


class CorruptProject(ProjectError):
    pass


class IncompatibleCheckpoint(ProjectError):
    pass


@dataclasses.dataclass(slots=True)
class Checkpoint:
    """Portable state required to reconstruct a running SLAM graph.

    ``arrays`` deliberately is a Mapping rather than a dict. Live processors
    can expose one CPU array at a time and loaded checkpoints can mmap arrays
    from the NPZ instead of materialising the complete state in host RAM.
    """

    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    rgb_keyframes: Mapping[str, bytes] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class RevisionInfo:
    number: int
    state: str
    keyframes: int
    created_at: str
    path: pathlib.Path


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _progress(callback: ProgressCallback | None, fraction: float, detail: str) -> None:
    if callback is not None:
        callback(float(max(0.0, min(1.0, fraction))), detail)


def _read_npy_header(stream) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        # NumPy's 3.0 header uses the same parser and differs only in encoding.
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise IncompatibleCheckpoint("object arrays are not supported")
    return tuple(int(value) for value in shape), bool(fortran), dtype


class LazyNpzArrays(collections.abc.Mapping[str, np.ndarray]):
    """Array mapping backed by a checkpoint NPZ.

    New ZIP_STORED entries are returned as read-only memmaps. Legacy compressed
    entries fall back to NumPy's per-member loader, so at most the requested
    array is resident at any point.
    """

    def __init__(self, path: pathlib.Path, names: list[str]) -> None:
        self.path = path
        self._names = tuple(names)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._names:
            raise KeyError(name)
        member = f"{name}.npy"
        with zipfile.ZipFile(self.path) as archive:
            try:
                info = archive.getinfo(member)
            except KeyError as error:
                raise CorruptProject(f"checkpoint array is missing: {name}") from error
            if info.compress_type == zipfile.ZIP_STORED:
                return self._memmap(info)
            with archive.open(info) as stream:
                return np.lib.format.read_array(stream, allow_pickle=False)

    def _memmap(self, info: zipfile.ZipInfo) -> np.ndarray:
        with self.path.open("rb") as stream:
            stream.seek(info.header_offset)
            local = stream.read(30)
            if len(local) != 30 or local[:4] != b"PK\x03\x04":
                raise CorruptProject("invalid NPZ local header")
            filename_size, extra_size = struct.unpack_from("<HH", local, 26)
            npy_offset = info.header_offset + 30 + filename_size + extra_size
            stream.seek(npy_offset)
            shape, fortran, dtype = _read_npy_header(stream)
            data_offset = stream.tell()
        order = "F" if fortran else "C"
        return np.memmap(self.path, mode="r", dtype=dtype, offset=data_offset, shape=shape, order=order)


class _NpyMemberRows:
    """Sequential axis-0 reader for one NPY member, including compressed NPZ."""

    def __init__(self, archive: zipfile.ZipFile, name: str) -> None:
        try:
            self.stream = archive.open(f"{name}.npy")
        except KeyError as error:
            raise ProjectError(f"checkpoint has no reconstruction array: {name}") from error
        self.shape, self.fortran, self.dtype = _read_npy_header(self.stream)
        if not self.shape:
            raise ProjectError(f"checkpoint reconstruction array is scalar: {name}")
        if self.fortran:
            self.stream.close()
            raise ProjectError(f"Fortran-ordered checkpoint array is unsupported for export: {name}")
        self.row_shape = self.shape[1:]
        self.row_bytes = int(np.prod(self.row_shape, dtype=np.int64)) * self.dtype.itemsize

    def read(self) -> np.ndarray:
        remaining = self.row_bytes
        chunks: list[bytes] = []
        while remaining:
            block = self.stream.read(remaining)
            if not block:
                raise CorruptProject("checkpoint array is truncated")
            chunks.append(block)
            remaining -= len(block)
        return np.frombuffer(b"".join(chunks), dtype=self.dtype).reshape(self.row_shape)

    def close(self) -> None:
        self.stream.close()


def _sim3_apply(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if len(pose) != 8:
        raise ProjectError("checkpoint contains an invalid Sim3 pose")
    vector = pose[3:6]
    scalar = pose[6]
    norm = max(float(np.linalg.norm(pose[3:7])), 1e-12)
    vector = vector / norm
    scalar = scalar / norm
    rotated = points + 2 * np.cross(vector, np.cross(vector, points) + scalar * points)
    return pose[7] * rotated + pose[:3]


class ProjectStore:
    def __init__(self, root: os.PathLike[str] | str) -> None:
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._recover_projects()

    def create(
        self,
        name: str,
        camera: CameraIdentity,
        *,
        model_id: str,
        config_fingerprint: str,
    ) -> str:
        project_id = f"{dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
        path = self.root / project_id
        path.mkdir()
        (path / "revisions").mkdir()
        manifest = {
            "format_version": PROJECT_FORMAT_VERSION,
            "project_id": project_id,
            "name": name,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "state": "active",
            "latest_revision": 0,
            "model_id": model_id,
            "config_fingerprint": config_fingerprint,
            "camera": dataclasses.asdict(camera),
            "revisions": [],
        }
        _atomic_json(path / "manifest.json", manifest)
        return project_id

    def save_revision(
        self,
        project_id: str,
        checkpoint: Checkpoint,
        *,
        state: str = "saved",
        trajectory: bytes | None = None,
        ply: bytes | None = None,
        thumbnail: bytes | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> RevisionInfo:
        """Atomically save a complete immutable revision.

        ``state`` remains accepted for source compatibility, but an immutable
        disk revision is always ``saved``. Live PAUSED/SCANNING state belongs to
        the session machine, not the project manifest.
        """

        del state
        project = self._project_path(project_id)
        manifest = self._manifest(project)
        number = int(manifest["latest_revision"]) + 1
        revisions = project / "revisions"
        final_dir = revisions / f"{number:04d}"
        if final_dir.exists():
            raise ProjectError(f"revision already exists: {number}")
        temp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f".{number:04d}.", suffix=".tmp", dir=revisions))
        names = sorted(checkpoint.arrays)
        try:
            _progress(progress_callback, 0.02, "Preparing checkpoint")
            self._write_npz(temp_dir / "checkpoint.npz", checkpoint.arrays, names, progress_callback)
            _atomic_json(
                temp_dir / "checkpoint.json",
                {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "metadata": checkpoint.metadata,
                    "array_names": names,
                    "storage": "zip-stored-npy",
                },
            )
            keyframe_dir = temp_dir / "keyframes"
            keyframe_dir.mkdir()
            for name, jpeg in checkpoint.rgb_keyframes.items():
                (keyframe_dir / pathlib.Path(name).name).write_bytes(jpeg)
            for name, content in {
                "trajectory.txt": trajectory,
                "reconstruction.ply": ply,
                "thumbnail.jpg": thumbnail,
            }.items():
                if content is not None:
                    (temp_dir / name).write_bytes(content)
            _progress(progress_callback, 0.82, "Verifying revision")
            files = sorted(path for path in temp_dir.rglob("*") if path.is_file())
            checksums = {str(path.relative_to(temp_dir)): _sha256(path) for path in files}
            _atomic_json(temp_dir / "checksums.json", checksums)
            self._fsync_tree(temp_dir)
            os.replace(temp_dir, final_dir)
            self._fsync_directory(revisions)
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        entry = {
            "number": number,
            "state": "saved",
            "keyframes": int(checkpoint.metadata.get("keyframe_count", 0)),
            "created_at": _utc_now(),
        }
        manifest["latest_revision"] = number
        manifest["state"] = "saved"
        manifest["updated_at"] = entry["created_at"]
        manifest["revisions"].append(entry)
        _atomic_json(project / "manifest.json", manifest)
        _progress(progress_callback, 1.0, "Project saved")
        return RevisionInfo(path=final_dir, **entry)

    @staticmethod
    def _write_npz(
        path: pathlib.Path,
        arrays: Mapping[str, np.ndarray],
        names: list[str],
        callback: ProgressCallback | None,
    ) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            total = max(len(names), 1)
            for index, name in enumerate(names):
                array = np.asarray(arrays[name])
                if array.dtype.hasobject:
                    raise IncompatibleCheckpoint(f"object checkpoint array is unsupported: {name}")
                with archive.open(f"{name}.npy", "w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, array, allow_pickle=False)
                del array
                _progress(callback, 0.05 + 0.72 * (index + 1) / total, f"Saving {name}")

    def load_revision(
        self,
        project_id: str,
        revision: int | None = None,
        *,
        expected_model_id: str | None = None,
        expected_config_fingerprint: str | None = None,
    ) -> Checkpoint:
        project = self._project_path(project_id)
        manifest = self._manifest(project)
        if manifest["format_version"] != PROJECT_FORMAT_VERSION:
            raise IncompatibleCheckpoint("unsupported project format")
        if expected_model_id and manifest["model_id"] != expected_model_id:
            raise IncompatibleCheckpoint("MASt3R model does not match")
        if expected_config_fingerprint and manifest["config_fingerprint"] != expected_config_fingerprint:
            raise IncompatibleCheckpoint("critical SLAM configuration does not match")
        number = revision or int(manifest["latest_revision"])
        if number < 1:
            raise ProjectError("project has no checkpoint")
        path = project / "revisions" / f"{number:04d}"
        self._verify(path)
        try:
            description = json.loads((path / "checkpoint.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CorruptProject("checkpoint description cannot be read") from error
        if description["format_version"] != CHECKPOINT_FORMAT_VERSION:
            raise IncompatibleCheckpoint("unsupported checkpoint format")
        names = description.get("array_names")
        if not isinstance(names, list):
            with np.load(path / "checkpoint.npz", allow_pickle=False) as stored:
                names = list(stored.files)
        keyframe_path = path / "keyframes"
        keyframes = {
            item.name: item.read_bytes()
            for item in keyframe_path.glob("*")
            if item.is_file()
        }
        return Checkpoint(LazyNpzArrays(path / "checkpoint.npz", names), description["metadata"], keyframes)

    def export_ply(
        self,
        project_id: str,
        target: os.PathLike[str] | str,
        progress_callback: ProgressCallback | None = None,
    ) -> pathlib.Path:
        """Export the latest complete revision without constructing a full cloud."""

        project = self._project_path(project_id)
        manifest = self._manifest(project)
        if manifest.get("state") != "saved" or int(manifest.get("latest_revision", 0)) < 1:
            raise ProjectError("project has no complete saved revision")
        revision = self.revision_path(project_id)
        self._verify(revision)
        destination = pathlib.Path(target).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        temporary = pathlib.Path(temporary_name)
        try:
            existing = revision / "reconstruction.ply"
            if existing.is_file():
                _progress(progress_callback, 0.05, "Copying reconstruction")
                with existing.open("rb") as source, temporary.open("wb") as output:
                    size = max(existing.stat().st_size, 1)
                    copied = 0
                    while True:
                        block = source.read(4 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        copied += len(block)
                        _progress(progress_callback, 0.05 + 0.9 * copied / size, "Copying reconstruction")
                    output.flush()
                    os.fsync(output.fileno())
            else:
                checkpoint = revision / "checkpoint.npz"
                if not checkpoint.is_file():
                    raise ProjectError("saved revision has neither PLY nor checkpoint")
                self._export_checkpoint_ply(checkpoint, temporary, progress_callback)
            os.replace(temporary, destination)
            self._fsync_directory(destination.parent)
            _progress(progress_callback, 1.0, "Export complete")
            return destination
        except ProjectError:
            temporary.unlink(missing_ok=True)
            raise
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise ProjectError(f"PLY export failed: {error}") from error

    @staticmethod
    def _export_checkpoint_ply(
        checkpoint: pathlib.Path,
        output: pathlib.Path,
        callback: ProgressCallback | None,
    ) -> None:
        with zipfile.ZipFile(checkpoint) as archive:
            try:
                with archive.open("T_WC.npy") as stream:
                    poses = np.lib.format.read_array(stream, allow_pickle=False)
            except KeyError as error:
                raise ProjectError("checkpoint has no reconstruction pose array: T_WC") from error
            points_reader = _NpyMemberRows(archive, "X")
            colors_reader = _NpyMemberRows(archive, "uimg")
            try:
                count = min(points_reader.shape[0], colors_reader.shape[0], len(poses))
                if count < 1:
                    raise ProjectError("checkpoint contains no keyframes")
                valid_counts: list[int] = []
                for index in range(count):
                    points = np.asarray(points_reader.read(), dtype=np.float32).reshape(-1, 3)
                    colors = np.asarray(colors_reader.read()).reshape(-1, 3)
                    if len(points) != len(colors):
                        raise ProjectError("checkpoint point map and image sizes do not match")
                    world = _sim3_apply(points, poses[index])
                    valid_counts.append(int((np.isfinite(world).all(axis=1) & np.isfinite(colors).all(axis=1)).sum()))
                    _progress(callback, 0.04 + 0.38 * (index + 1) / count, "Counting reconstruction vertices")
            finally:
                points_reader.close()
                colors_reader.close()

        vertex_count = sum(valid_counts)
        if vertex_count < 1:
            raise ProjectError("checkpoint contains no finite reconstruction points")
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {vertex_count}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        ).encode("ascii")
        vertex_dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        with output.open("wb") as destination, zipfile.ZipFile(checkpoint) as archive:
            destination.write(header)
            points_reader = _NpyMemberRows(archive, "X")
            colors_reader = _NpyMemberRows(archive, "uimg")
            try:
                for index in range(len(valid_counts)):
                    points = np.asarray(points_reader.read(), dtype=np.float32).reshape(-1, 3)
                    colors = np.asarray(colors_reader.read()).reshape(-1, 3)
                    world = _sim3_apply(points, poses[index])
                    valid = np.isfinite(world).all(axis=1) & np.isfinite(colors).all(axis=1)
                    world = world[valid]
                    colors = colors[valid]
                    if np.issubdtype(colors.dtype, np.floating):
                        colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
                    else:
                        colors = np.clip(colors, 0, 255).astype(np.uint8)
                    vertices = np.empty(len(world), dtype=vertex_dtype)
                    vertices["x"], vertices["y"], vertices["z"] = world.T
                    vertices["red"], vertices["green"], vertices["blue"] = colors.T
                    destination.write(vertices.tobytes())
                    _progress(callback, 0.45 + 0.5 * (index + 1) / len(valid_counts), "Writing reconstruction")
            finally:
                points_reader.close()
                colors_reader.close()
            destination.flush()
            os.fsync(destination.fileno())

    def list_projects(self, *, include_empty: bool = False) -> list[dict[str, Any]]:
        result = []
        for manifest_path in self.root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
                if not include_empty and int(manifest.get("latest_revision", 0)) < 1:
                    continue
                latest = manifest["revisions"][-1] if manifest["revisions"] else {}
                revision = manifest_path.parent / "revisions" / f"{int(manifest['latest_revision']):04d}"
                result.append(
                    {
                        "project_id": manifest["project_id"],
                        "name": manifest["name"],
                        "updated_at": manifest["updated_at"],
                        "state": manifest["state"],
                        "revision": manifest["latest_revision"],
                        "keyframes": latest.get("keyframes", 0),
                        "has_ply": (revision / "reconstruction.ply").is_file(),
                        "size_bytes": sum(p.stat().st_size for p in manifest_path.parent.rglob("*") if p.is_file()),
                    }
                )
            except (OSError, KeyError, json.JSONDecodeError, ValueError):
                continue
        return sorted(result, key=lambda item: item["updated_at"], reverse=True)

    def revision_path(self, project_id: str, revision: int | None = None) -> pathlib.Path:
        project = self._project_path(project_id)
        manifest = self._manifest(project)
        number = revision or int(manifest["latest_revision"])
        return project / "revisions" / f"{number:04d}"

    def project_path(self, project_id: str) -> pathlib.Path:
        return self._project_path(project_id)

    def read_manifest(self, project_id: str) -> dict[str, Any]:
        return self._manifest(self._project_path(project_id))

    def delete_project(self, project_id: str) -> None:
        shutil.rmtree(self._project_path(project_id))

    def _recover_projects(self) -> None:
        """Recover the newest verified revision and discard only safe temp dirs."""

        for manifest_path in self.root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
                revisions_dir = manifest_path.parent / "revisions"
                temporary_dirs = list(revisions_dir.glob(".*.tmp"))
                numbered = sorted(
                    (item for item in revisions_dir.glob("[0-9][0-9][0-9][0-9]") if item.is_dir()),
                    key=lambda item: item.name,
                    reverse=True,
                )
                has_orphan = bool(numbered) and int(numbered[0].name) != int(manifest.get("latest_revision", 0))
                if manifest.get("state") not in {"active", "paused"} and not temporary_dirs and not has_orphan:
                    continue
                valid_path = next((item for item in numbered if self._is_valid_revision(item)), None)
                if valid_path is None:
                    # Keep empty/broken projects and their temporary evidence for diagnostics.
                    continue
                valid = int(valid_path.name)
                entries = [entry for entry in manifest.get("revisions", []) if int(entry.get("number", 0)) <= valid]
                known = {int(entry["number"]): entry for entry in entries}
                if valid not in known:
                    description = json.loads((valid_path / "checkpoint.json").read_text())
                    entries.append({
                        "number": valid,
                        "state": "saved",
                        "keyframes": int(description.get("metadata", {}).get("keyframe_count", 0)),
                        "created_at": dt.datetime.fromtimestamp(valid_path.stat().st_mtime, dt.timezone.utc).isoformat(),
                    })
                changed = manifest.get("state") != "saved" or int(manifest.get("latest_revision", 0)) != valid
                for entry in entries:
                    if entry.get("state") != "saved":
                        entry["state"] = "saved"
                        changed = True
                manifest["latest_revision"] = valid
                manifest["state"] = "saved"
                manifest["revisions"] = sorted(entries, key=lambda entry: int(entry["number"]))
                if changed:
                    manifest["updated_at"] = manifest["revisions"][-1]["created_at"]
                    _atomic_json(manifest_path, manifest)
                for temporary in temporary_dirs:
                    if temporary.is_dir():
                        shutil.rmtree(temporary, ignore_errors=True)
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                continue

    @classmethod
    def _is_valid_revision(cls, path: pathlib.Path) -> bool:
        try:
            cls._verify(path)
            return True
        except ProjectError:
            return False

    def _project_path(self, project_id: str) -> pathlib.Path:
        if pathlib.Path(project_id).name != project_id:
            raise ProjectError("invalid project id")
        path = self.root / project_id
        if not path.is_dir():
            raise ProjectError(f"unknown project: {project_id}")
        return path

    @staticmethod
    def _manifest(project: pathlib.Path) -> dict[str, Any]:
        try:
            return json.loads((project / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CorruptProject("manifest cannot be read") from error

    @staticmethod
    def _verify(path: pathlib.Path) -> None:
        try:
            checksums = json.loads((path / "checksums.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CorruptProject("checksums are missing") from error
        for name, expected in checksums.items():
            file_path = path / name
            if not file_path.is_file() or _sha256(file_path) != expected:
                raise CorruptProject(f"checksum mismatch: {name}")

    @staticmethod
    def _fsync_tree(path: pathlib.Path) -> None:
        for item in path.rglob("*"):
            if item.is_file():
                with item.open("rb") as stream:
                    os.fsync(stream.fileno())
        ProjectStore._fsync_directory(path)

    @staticmethod
    def _fsync_directory(path: pathlib.Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
