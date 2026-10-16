from __future__ import annotations

import io
import threading
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .engine import ArAlignmentFrame, ProcessResult, Tracking
from .frame_source import NetworkFrame
from .projects import Checkpoint
from .renderer import OffscreenMapRenderer
from .snapshot import create_mesh_snapshot


class _ProcessorArrays(Mapping[str, np.ndarray]):
    """Expose live CUDA state as one disposable CPU array at a time."""

    TENSOR_FIELDS = (
        "dataset_idx", "img", "uimg", "img_shape", "img_true_shape", "T_WC",
        "X", "C", "N", "N_updates", "feat", "pos", "K",
    )

    def __init__(self, processor: "Mast3rProcessor", count: int) -> None:
        self.processor = processor
        self.count = count
        self.names = self.TENSOR_FIELDS + tuple(f"graph_{name}" for name in processor.GRAPH_ARRAYS)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str) -> np.ndarray:
        if name.startswith("graph_"):
            graph_name = name.removeprefix("graph_")
            if graph_name not in self.processor.GRAPH_ARRAYS:
                raise KeyError(name)
            return self.processor._cpu(getattr(self.processor.factor_graph, graph_name))
        if name not in self.TENSOR_FIELDS:
            raise KeyError(name)
        tensor = getattr(self.processor.keyframes, name)
        return self.processor._cpu(tensor if name == "K" else tensor[: self.count])


class _ProcessorRgb(Mapping[str, bytes]):
    def __init__(self, processor: "Mast3rProcessor", count: int) -> None:
        self.processor = processor
        ids = processor._cpu(processor.keyframes.dataset_idx[:count]).reshape(-1)
        self.names = tuple(f"{int(value):012d}.jpg" for value in ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str) -> bytes:
        try:
            index = self.names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        import cv2

        image = (self.processor._cpu(self.processor.keyframes.uimg[index]) * 255).clip(0, 255).astype(np.uint8)
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"cannot encode keyframe {name}")
        return jpeg.tobytes()


class _ThreadValue:
    def __init__(self, value: Any) -> None:
        self.value = value


class _ThreadStateManager:
    """Subset of multiprocessing.Manager used by SharedKeyframes.

    The mobile processor is intentionally single-process: the engine owns one
    processing thread and serializes processor access. Spawning a Manager here
    adds no safety, but makes first-frame initialization depend on inherited
    terminal/socket file descriptors.
    """

    @staticmethod
    def RLock() -> threading.RLock:
        return threading.RLock()

    @staticmethod
    def Value(_typecode: str, value: Any) -> _ThreadValue:
        return _ThreadValue(value)


class Mast3rProcessor:
    """Synchronous adapter around the existing tracker and CUDA factor graph.

    Imports of torch, lietorch and MASt3R are intentionally delayed until the
    first frame/checkpoint, keeping project management and CLI discovery usable
    on machines without a configured GPU.
    """

    GRAPH_ARRAYS = (
        "ii",
        "jj",
        "idx_ii2jj",
        "idx_jj2ii",
        "valid_match_j",
        "valid_match_i",
        "Q_ii2jj",
        "Q_jj2ii",
    )

    def __init__(self, config_path: str = "config/base.yaml", device: str = "cuda:0") -> None:
        self.config_path = config_path
        self.device = device
        self.model: Any = None
        self.manager: Any = None
        self.keyframes: Any = None
        self.tracker: Any = None
        self.factor_graph: Any = None
        self.retrieval: Any = None
        self._last_frame: Any = None
        self._current_packet: NetworkFrame | None = None
        self._timestamps: dict[int, int] = {}
        self._renderer = OffscreenMapRenderer()
        self._shape: tuple[int, int] | None = None

    def process(self, packet: NetworkFrame, *, relocalizing: bool) -> ProcessResult:
        import cv2
        import lietorch

        from mast3r_slam.frame import create_frame
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        self._current_packet = packet
        encoded = np.frombuffer(packet.jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("invalid JPEG frame")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rotations = (packet.orientation_degrees % 360) // 90
        if rotations:
            image = np.rot90(image, k=(4 - rotations) % 4).copy()
        image = image.astype(np.float32) / 255.0
        self._ensure_initialized(image=image)
        previous_pose = (
            lietorch.Sim3.Identity(1, device=self.device)
            if self._last_frame is None
            else self._last_frame.T_WC
        )
        frame = create_frame(packet.frame_id, image, previous_pose, device=self.device)
        self._timestamps[packet.frame_id] = packet.timestamp_ns

        if len(self.keyframes) == 0:
            X, C = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X, C)
            self._append_and_optimize(frame)
            self._last_frame = frame
            return self._result(frame, Tracking.TRACKING, True)

        if relocalizing:
            X, C = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X, C)
            success = self._relocalize(frame)
            if success:
                self._last_frame = frame
                return self._result(frame, Tracking.TRACKING, True)
            return self._result(
                frame,
                Tracking.RELOCALIZING,
                False,
                guidance="Вернитесь к уже отснятой области",
            )

        add_keyframe, _, lost = self.tracker.track(frame)
        if lost:
            return self._result(
                frame,
                Tracking.LOST,
                False,
                guidance="Вернитесь к уже отснятой области",
            )
        if add_keyframe and len(self.keyframes) < self.keyframes.buffer:
            self._append_and_optimize(frame)
        self._last_frame = frame
        return self._result(frame, Tracking.TRACKING, True)

    def checkpoint(self) -> Checkpoint:
        if self.keyframes is None:
            return Checkpoint({}, {"keyframe_count": 0, "timestamps_ns": {}})
        count = len(self.keyframes)
        return Checkpoint(
            _ProcessorArrays(self, count),
            {
                "keyframe_count": count,
                "timestamps_ns": {str(key): value for key, value in self._timestamps.items()},
                "shape": list(self._shape or (0, 0)),
            },
            _ProcessorRgb(self, count),
        )

    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        import torch

        count = int(checkpoint.metadata["keyframe_count"])
        if count == 0:
            return
        shape = tuple(int(value) for value in checkpoint.metadata["shape"])
        self._ensure_initialized(shape=shape)
        tensor_fields = (
            "dataset_idx", "img", "uimg", "img_shape", "img_true_shape", "T_WC",
            "X", "C", "N", "N_updates", "feat", "pos", "K",
        )
        for name in tensor_fields:
            target = getattr(self.keyframes, name)
            stored = checkpoint.arrays[name]
            source = torch.from_numpy(stored).to(target.device, dtype=target.dtype)
            if name == "K":
                target[:] = source
            else:
                target[:count] = source
            del source, stored
        self.keyframes.n_size.value = count
        for name in self.GRAPH_ARRAYS:
            stored = checkpoint.arrays[f"graph_{name}"]
            current = getattr(self.factor_graph, name)
            setattr(self.factor_graph, name, torch.from_numpy(stored).to(current.device, dtype=current.dtype))
            del stored
        # The retrieval index is derived state: rebuild it from persisted MASt3R features.
        for index in range(count):
            self.retrieval.update(
                self.keyframes[index],
                add_after_query=True,
                k=0,
                min_thresh=0.0,
            )
        self._last_frame = self.keyframes[count - 1]
        self._timestamps = {int(key): int(value) for key, value in checkpoint.metadata["timestamps_ns"].items()}

    def finalize(self) -> bytes | None:
        if self.keyframes is None or len(self.keyframes) == 0:
            return None
        trajectory = io.StringIO()
        from mast3r_slam.lietorch_utils import as_SE3

        for index in range(len(self.keyframes)):
            frame = self.keyframes[index]
            values = as_SE3(frame.T_WC).data.detach().cpu().numpy().reshape(-1)
            timestamp = self._timestamps.get(frame.frame_id, frame.frame_id) / 1e9
            trajectory.write(f"{timestamp} {' '.join(str(float(v)) for v in values)}\n")
        return trajectory.getvalue().encode()

    def mobile_snapshot(self, output_dir: str):
        count = len(self.keyframes)

        def surfaces():
            for index in range(count):
                frame = self.keyframes[index]
                height, width = (int(value) for value in frame.img_shape.flatten())
                camera = frame.X_canon.detach().cpu().numpy().reshape(-1, 3)
                world = frame.T_WC.act(frame.X_canon).detach().cpu().numpy().reshape(-1, 3)
                confidence = frame.get_average_conf().detach().cpu().numpy().reshape(-1)
                yield world, camera, confidence, (height, width)

        return create_mesh_snapshot(output_dir, surfaces(), surface_count=count)

    def align_ar(self, packet: ArAlignmentFrame) -> tuple[np.ndarray, float] | None:
        """Pose-only retrieval/matching that never appends to the saved graph."""

        import cv2
        import torch

        from mast3r_slam.frame import create_frame
        from mast3r_slam.mast3r_utils import mast3r_inference_mono, mast3r_match_asymmetric

        if self.keyframes is None or len(self.keyframes) == 0:
            return None
        bgr = cv2.imdecode(np.frombuffer(packet.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("invalid AR alignment JPEG")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Encode once before retrieval. Mono point inference also populates feat/pos.
        frame = create_frame(packet.frame_id, image, self.keyframes[len(self.keyframes) - 1].T_WC, device=self.device)
        # Mono inference populates feat/pos for retrieval. Its point map is not
        # fused: the asymmetric candidate match below is the sole pose input.
        mast3r_inference_mono(self.model, frame)
        candidates = self.retrieval.update(
            frame,
            add_after_query=False,
            k=self._config["retrieval"]["k"],
            min_thresh=self._config["retrieval"]["min_thresh"],
        )
        if not candidates:
            return None
        keyframe = self.keyframes[int(candidates[0])]
        frame.T_WC = keyframe.T_WC.clone()
        idx_f2k, valid_match_k, Xff, Cff, Qff, _Xkf, _Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe
        )
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]
        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)
        frame.update_pointmap(Xff, Cff)
        use_calib = self._config["use_calib"]
        K = keyframe.K if use_calib else None
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.tracker.get_points_poses(
            frame, keyframe, idx_f2k, frame.img.shape[-2:], use_calib, K
        )
        cfg = self.tracker.cfg
        valid = valid_match_k & (Cf > cfg["C_conf"]) & (Ck > cfg["C_conf"]) & (Qk > cfg["Q_conf"])
        quality = float(valid.float().mean().item())
        if quality < float(self._config["reloc"]["min_match_frac"]):
            return None
        try:
            if use_calib:
                pose, _ = self.tracker.opt_pose_calib_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, frame.img.shape[-2:]
                )
            else:
                pose, _ = self.tracker.opt_pose_ray_dist_sim3(Xf, Xk, T_WCf, T_WCk, Qk, valid)
        except Exception:
            return None
        return pose.data.detach().cpu().numpy().reshape(-1), quality

    def _ensure_initialized(
        self,
        *,
        image: np.ndarray | None = None,
        shape: tuple[int, int] | None = None,
    ) -> None:
        if self.model is not None:
            return
        import torch

        from mast3r_slam.config import config, load_config
        from mast3r_slam.frame import SharedKeyframes
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, resize_img
        from mast3r_slam.tracker import FrameTracker

        if not torch.cuda.is_available() and self.device.startswith("cuda"):
            raise RuntimeError("CUDA is required for live MASt3R-SLAM processing")
        load_config(self.config_path)
        if shape is None:
            if image is None:
                raise ValueError("an image or checkpoint shape is required")
            resized = resize_img(image, 512)["img"]
            shape = tuple(int(value) for value in resized.shape[-2:])
        self._shape = shape
        self.manager = _ThreadStateManager()
        self.keyframes = SharedKeyframes(self.manager, *shape, device=self.device)
        self.model = load_mast3r(device=self.device)
        self.model.share_memory()
        self.tracker = FrameTracker(self.model, self.keyframes, self.device)
        self.factor_graph = FactorGraph(self.model, self.keyframes, None, self.device)
        self.retrieval = load_retriever(self.model, device=self.device)
        self._config = config

    def _append_and_optimize(self, frame: Any) -> None:
        self.keyframes.append(frame)
        index = len(self.keyframes) - 1
        previous = [index - 1] if index else []
        retrieved = self.retrieval.update(
            frame,
            add_after_query=True,
            k=self._config["retrieval"]["k"],
            min_thresh=self._config["retrieval"]["min_thresh"],
        )
        neighbors = sorted(set(previous + retrieved) - {index})
        if neighbors:
            self.factor_graph.add_factors(
                neighbors,
                [index] * len(neighbors),
                self._config["local_opt"]["min_match_frac"],
            )
            self._solve()

    def _relocalize(self, frame: Any) -> bool:
        candidates = self.retrieval.update(
            frame,
            add_after_query=False,
            k=self._config["retrieval"]["k"],
            min_thresh=self._config["retrieval"]["min_thresh"],
        )
        if not candidates:
            return False
        self.keyframes.append(frame)
        index = len(self.keyframes) - 1
        success = self.factor_graph.add_factors(
            [index] * len(candidates),
            list(candidates),
            self._config["reloc"]["min_match_frac"],
            is_reloc=self._config["reloc"]["strict"],
        )
        if not success:
            self.keyframes.pop_last()
            return False
        self.keyframes.T_WC[index] = self.keyframes.T_WC[candidates[0]].clone()
        self.retrieval.update(
            frame,
            add_after_query=True,
            k=self._config["retrieval"]["k"],
            min_thresh=self._config["retrieval"]["min_thresh"],
        )
        self._solve()
        return True

    def _solve(self) -> None:
        if self._config["use_calib"]:
            self.factor_graph.solve_GN_calib()
        else:
            self.factor_graph.solve_GN_rays()

    def _result(self, frame: Any, tracking: Tracking, updated: bool, guidance: str = "") -> ProcessResult:
        preview = None
        if tracking != Tracking.LOST:
            points, colors, _ = self._map_arrays(max_points=350_000, camera_frame=frame)
            packet = self._current_packet
            if packet is None:
                raise RuntimeError("preview frame metadata is unavailable")
            preview = self._renderer.render(
                points,
                colors,
                intrinsics=packet.intrinsics,
                image_size=(packet.width, packet.height),
                rotation_degrees=packet.orientation_degrees,
            )
        return ProcessResult(frame.frame_id, tracking, preview, updated, len(self.keyframes), guidance)

    def _map_arrays(
        self, max_points: int | None = None, camera_frame: Any | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        point_sets, color_sets, confidence_sets = [], [], []
        for index in range(len(self.keyframes)):
            frame = self.keyframes[index]
            points = frame.T_WC.act(frame.X_canon)
            if camera_frame is not None:
                points = camera_frame.T_WC.inv().act(points)
            point_sets.append(points.detach().cpu().numpy().reshape(-1, 3))
            color_sets.append((frame.uimg.detach().cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3))
            confidence_sets.append(frame.get_average_conf().detach().cpu().numpy().reshape(-1))
        points = np.concatenate(point_sets)
        colors = np.concatenate(color_sets)
        confidence = np.concatenate(confidence_sets)
        if max_points and len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
            points, colors, confidence = points[indices], colors[indices], confidence[indices]
        return points, colors, confidence

    @staticmethod
    def _cpu(tensor: Any) -> np.ndarray:
        return tensor.detach().cpu().numpy().copy()
