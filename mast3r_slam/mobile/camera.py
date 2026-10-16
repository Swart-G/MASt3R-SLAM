from __future__ import annotations

import dataclasses
import hashlib
import json


@dataclasses.dataclass(frozen=True, slots=True)
class CameraIdentity:
    device_model: str
    physical_camera_id: str
    width: int
    height: int
    crop: tuple[int, int, int, int]
    intrinsics: tuple[float, float, float, float]
    distortion: tuple[float, ...] = ()

    @property
    def fingerprint(self) -> str:
        value = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class CameraCompatibility:
    compatible: bool
    differences: tuple[str, ...]

    @property
    def requires_confirmation(self) -> bool:
        return not self.compatible


def compare_cameras(saved: CameraIdentity, current: CameraIdentity) -> CameraCompatibility:
    differences: list[str] = []
    for field in ("device_model", "physical_camera_id", "width", "height", "crop"):
        if getattr(saved, field) != getattr(current, field):
            differences.append(field)
    if any(abs(a - b) > 1e-3 for a, b in zip(saved.intrinsics, current.intrinsics)):
        differences.append("intrinsics")
    if len(saved.distortion) != len(current.distortion) or any(
        abs(a - b) > 1e-4 for a, b in zip(saved.distortion, current.distortion)
    ):
        differences.append("distortion")
    return CameraCompatibility(not differences, tuple(differences))
