from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mast3r_slam.desktop_viewer import _checkpoint_geometry
from mast3r_slam.mobile.adaptive import AdaptiveBitrate
from mast3r_slam.mobile.ar_alignment import ar_world_from_slam_map, from_column_major, sim3_matrix, to_column_major
from mast3r_slam.mobile.camera import CameraIdentity, compare_cameras
from mast3r_slam.mobile.engine import ArAlignmentFrame, ArAlignmentStatus, ProcessResult, SlamEngine, Tracking
from mast3r_slam.mobile.frame_source import NetworkFrame, NetworkFrameSource
from mast3r_slam.mobile.mast3r_processor import _ThreadStateManager
from mast3r_slam.mobile.projects import Checkpoint, CorruptProject, ProjectError, ProjectStore
from mast3r_slam.mobile.renderer import project_camera_points, rotate_pinhole_intrinsics
from mast3r_slam.mobile.snapshot import (
    create_mesh_snapshot,
    create_snapshot,
    decode_chunk,
    decode_mesh_chunk,
    filter_and_downsample,
    triangulate_grid,
)
from mast3r_slam.mobile.state_machine import InvalidTransition, SessionMachine, SessionPhase
from protocol import mast3r_slam_pb2 as wire


CAMERA = CameraIdentity("phone", "0", 1280, 720, (0, 0, 1280, 720), (900, 900, 640, 360))


class FrameSourceTests(unittest.TestCase):
    def test_only_newest_waiting_frame_survives(self):
        source = NetworkFrameSource()
        self.assertTrue(source.submit(NetworkFrame(1, 1, b"one")).accepted)
        result = source.submit(NetworkFrame(2, 2, b"two"))
        self.assertEqual(result.replaced_frame_id, 1)
        self.assertEqual(source.dropped, 1)
        self.assertEqual(source.get(0).frame_id, 2)
        self.assertIsNone(source.get(0.001))

    def test_reordered_frame_is_rejected(self):
        source = NetworkFrameSource()
        source.submit(NetworkFrame(2, 2, b""))
        self.assertFalse(source.submit(NetworkFrame(1, 1, b"")).accepted)

    def test_reset_discards_pending_and_can_restart_frame_ids(self):
        source = NetworkFrameSource()
        source.submit(NetworkFrame(10, 10, b"stale"))
        source.reset(reset_sequence=True)
        self.assertIsNone(source.pending_frame_id)
        self.assertTrue(source.submit(NetworkFrame(1, 11, b"fresh")).accepted)
        self.assertEqual(source.get(0).frame_id, 1)


class MobileProcessorStateTests(unittest.TestCase):
    def test_live_state_manager_is_thread_local_and_needs_no_subprocess(self):
        manager = _ThreadStateManager()
        value = manager.Value("i", 3)
        value.value += 2
        self.assertEqual(value.value, 5)
        lock = manager.RLock()
        with lock:
            with lock:
                value.value += 1
        self.assertEqual(value.value, 6)


class StateMachineTests(unittest.TestCase):
    def test_happy_path_and_idempotency(self):
        state = SessionMachine()
        self.assertEqual(state.paired(), SessionPhase.READY)
        first = state.command("a", "start")
        repeated = state.command("a", "start")
        self.assertFalse(first.repeated)
        self.assertTrue(repeated.repeated)
        state.command("b", "pause")
        state.command("c", "enter_ar")
        state.ar_ready()
        state.command("d", "stop")
        state.finalized()
        self.assertEqual(state.phase, SessionPhase.SAVED)

    def test_invalid_transition_and_command_reuse(self):
        state = SessionMachine(); state.paired()
        with self.assertRaises(InvalidTransition):
            state.command("a", "pause")
        state.command("x", "start")
        with self.assertRaises(InvalidTransition):
            state.command("x", "pause")

    def test_disconnect_reconnect_is_paused(self):
        state = SessionMachine(); state.paired(); state.command("a", "start")
        state.disconnect()
        self.assertEqual(state.phase, SessionPhase.DISCONNECTED)
        self.assertEqual(state.paired(), SessionPhase.PAUSED)


class PolicyTests(unittest.TestCase):
    def test_adaptive_order(self):
        controller = AdaptiveBitrate()
        self.assertGreater(controller.update(latency_ms=100, dropped=False).jpeg_quality, 80)
        for _ in range(10):
            profile = controller.update(latency_ms=100, dropped=False)
        self.assertEqual(profile.jpeg_quality, 90)
        self.assertGreater(profile.fps, 10)
        old_quality = profile.jpeg_quality
        overloaded = controller.update(latency_ms=1000, dropped=True)
        self.assertLess(overloaded.fps, profile.fps)
        self.assertEqual(overloaded.jpeg_quality, old_quality)

    def test_camera_difference_requires_confirmation(self):
        changed = CameraIdentity("phone", "2", 1280, 720, CAMERA.crop, CAMERA.intrinsics)
        result = compare_cameras(CAMERA, changed)
        self.assertFalse(result.compatible)
        self.assertIn("physical_camera_id", result.differences)


class ProtocolTests(unittest.TestCase):
    def test_frame_round_trip(self):
        outgoing = wire.ClientMessage(
            frame=wire.FramePacket(
                frame_id=42,
                timestamp_ns=123,
                orientation_degrees=90,
                intrinsics=[1, 2, 3, 4],
                jpeg=b"jpeg",
            )
        )
        incoming = wire.ClientMessage.FromString(outgoing.SerializeToString())
        self.assertEqual(incoming.WhichOneof("payload"), "frame")
        self.assertEqual(incoming.frame.frame_id, 42)
        self.assertEqual(incoming.frame.jpeg, b"jpeg")

    def test_ar_alignment_v2_round_trip(self):
        pose = list(np.eye(4, dtype=np.float32).reshape(-1, order="F"))
        outgoing = wire.ClientMessage(
            ar_alignment_frame=wire.ArAlignmentFrame(
                frame_id=7,
                jpeg=b"jpeg",
                intrinsics=[500, 500, 320, 240],
                ar_world_from_camera=pose,
            )
        )
        incoming = wire.ClientMessage.FromString(outgoing.SerializeToString())
        self.assertEqual(incoming.WhichOneof("payload"), "ar_alignment_frame")
        self.assertEqual(len(incoming.ar_alignment_frame.ar_world_from_camera), 16)


class ProjectStoreTests(unittest.TestCase):
    def test_round_trip_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = store.create("room", CAMERA, model_id="model", config_fingerprint="cfg")
            checkpoint = Checkpoint(
                {"poses": np.eye(4, dtype=np.float32), "confidence": np.arange(8, dtype=np.float32)},
                {"keyframe_count": 2, "edges": [[0, 1]]},
                {"000.jpg": b"jpeg"},
            )
            revision = store.save_revision(project, checkpoint, state="paused")
            loaded = store.load_revision(project, expected_model_id="model", expected_config_fingerprint="cfg")
            np.testing.assert_array_equal(loaded.arrays["poses"], checkpoint.arrays["poses"])
            np.testing.assert_array_equal(loaded.arrays["confidence"], checkpoint.arrays["confidence"])
            self.assertEqual(loaded.metadata, checkpoint.metadata)
            self.assertEqual(loaded.rgb_keyframes["000.jpg"], b"jpeg")
            self.assertIsInstance(loaded.arrays["poses"], np.memmap)
            self.assertEqual(store.read_manifest(project)["state"], "saved")
            (revision.path / "checkpoint.npz").write_bytes(b"corrupt")
            with self.assertRaises(CorruptProject):
                store.load_revision(project)

    def test_failed_save_keeps_previous_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = store.create("room", CAMERA, model_id="m", config_fingerprint="c")
            checkpoint = Checkpoint({"x": np.arange(3)}, {"keyframe_count": 1})
            store.save_revision(project, checkpoint, state="paused")
            with mock.patch.object(ProjectStore, "_write_npz", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    store.save_revision(project, checkpoint, state="paused")
            self.assertEqual(store.list_projects()[0]["revision"], 1)
            np.testing.assert_array_equal(store.load_revision(project).arrays["x"], np.arange(3))

    def test_delete_project_removes_only_validated_project(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = store.create("room", CAMERA, model_id="m", config_fingerprint="c")
            project_path = store.project_path(project)
            store.delete_project(project)
            self.assertFalse(project_path.exists())
            self.assertEqual(store.list_projects(), [])
            with self.assertRaises(ProjectError):
                store.delete_project("../outside")

    def test_empty_project_is_kept_for_diagnostics_but_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = store.create("empty", CAMERA, model_id="m", config_fingerprint="c")
            self.assertEqual(store.list_projects(), [])
            self.assertEqual(store.list_projects(include_empty=True)[0]["project_id"], project)

    def test_export_checkpoint_existing_ply_and_atomic_failure(self):
        poses = np.asarray([[0, 0, 0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 1, 2]], dtype=np.float32)
        points = np.asarray([[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]]], dtype=np.float32)
        colors = np.asarray([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(Path(directory) / "projects")
            project = store.create("room", CAMERA, model_id="m", config_fingerprint="c")
            store.save_revision(project, Checkpoint({"T_WC": poses, "X": points, "uimg": colors}, {"keyframe_count": 2}))
            target = Path(directory) / "room.ply"
            progress = []
            store.export_ply(project, target, lambda value, detail: progress.append((value, detail)))
            self.assertIn(b"element vertex 4", target.read_bytes()[:256])
            self.assertEqual(progress[-1][0], 1.0)

            revision = store.revision_path(project)
            existing = revision / "reconstruction.ply"
            existing.write_bytes(b"existing-ply")
            checksums = json.loads((revision / "checksums.json").read_text())
            checksums["reconstruction.ply"] = hashlib.sha256(b"existing-ply").hexdigest()
            (revision / "checksums.json").write_text(json.dumps(checksums))
            store.export_ply(project, target)
            self.assertEqual(target.read_bytes(), b"existing-ply")

            existing.unlink()
            checksums.pop("reconstruction.ply")
            (revision / "checksums.json").write_text(json.dumps(checksums))
            target.write_bytes(b"last-good")
            with mock.patch.object(store, "_export_checkpoint_ply", side_effect=OSError("synthetic")):
                with self.assertRaisesRegex(ProjectError, "PLY export failed"):
                    store.export_ply(project, target)
            self.assertEqual(target.read_bytes(), b"last-good")
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_export_rejects_unsaved_and_recovers_paused_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            empty = store.create("empty", CAMERA, model_id="m", config_fingerprint="c")
            with self.assertRaisesRegex(ProjectError, "complete saved"):
                store.export_ply(empty, Path(directory) / "empty.ply")
            project = store.create("room", CAMERA, model_id="m", config_fingerprint="c")
            store.save_revision(project, Checkpoint({"x": np.arange(3)}, {"keyframe_count": 1}))
            manifest_path = store.project_path(project) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["state"] = "paused"
            manifest["revisions"][-1]["state"] = "paused"
            manifest_path.write_text(json.dumps(manifest))
            temporary = store.project_path(project) / "revisions" / ".0002.interrupted.tmp"
            temporary.mkdir()
            (temporary / "checkpoint.npz").write_bytes(b"partial")
            recovered = ProjectStore(directory)
            self.assertEqual(recovered.read_manifest(project)["state"], "saved")
            self.assertFalse(temporary.exists())

    def test_recovery_adopts_complete_revision_renamed_before_manifest_update(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = store.create("room", CAMERA, model_id="m", config_fingerprint="c")
            first = store.save_revision(project, Checkpoint({"x": np.arange(3)}, {"keyframe_count": 1}))
            second = first.path.parent / "0002"
            shutil.copytree(first.path, second)
            recovered = ProjectStore(directory)
            manifest = recovered.read_manifest(project)
            self.assertEqual(manifest["latest_revision"], 2)
            self.assertEqual(manifest["state"], "saved")


class DesktopViewerTests(unittest.TestCase):
    def test_paused_checkpoint_is_converted_to_world_space_geometry(self):
        points = np.asarray([[[1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        # 90 degrees around Z, then scale by two and translate by (10, 20, 30).
        poses = np.asarray(
            [[[10, 20, 30, 0, 0, np.sqrt(0.5), np.sqrt(0.5), 2]]], dtype=np.float32
        )
        colors = np.asarray([[[[1, 0, 0], [0, 0.5, 1]]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.npz"
            np.savez_compressed(checkpoint, X=points, T_WC=poses, uimg=colors)
            world, rgb = _checkpoint_geometry(checkpoint)
        np.testing.assert_allclose(world, [[10, 22, 30], [8, 20, 30]], atol=1e-5)
        np.testing.assert_allclose(rgb, [[1, 0, 0], [0, 0.5, 1]])


class SnapshotTests(unittest.TestCase):
    def test_voxel_keeps_highest_confidence_point(self):
        points = np.asarray([[0.001, 0, 0], [0.002, 0, 0]], dtype=np.float32)
        colors = np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        selected_points, selected_colors = filter_and_downsample(
            points, colors, np.asarray([0.1, 0.9]), voxel_size=0.01
        )
        self.assertEqual(len(selected_points), 1)
        np.testing.assert_array_equal(selected_colors[0], [0, 255, 0])

    def test_quantized_chunk_round_trip(self):
        rng = np.random.default_rng(7)
        points = rng.normal(size=(1000, 3)).astype(np.float32)
        colors = rng.integers(0, 256, size=(1000, 3), dtype=np.uint8)
        confidence = np.ones(1000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            manifest = create_snapshot(directory, points, colors, confidence, voxel_size=0, points_per_chunk=400)
            self.assertEqual(manifest.point_count, 1000)
            restored_points, restored_colors = [], []
            for chunk in manifest.chunks:
                compressed = (Path(directory) / chunk.filename).read_bytes()
                self.assertEqual(hashlib.sha256(compressed).hexdigest(), chunk.sha256)
                xyz, rgb = decode_chunk(compressed)
                restored_points.append(xyz); restored_colors.append(rgb)
            restored_points = np.concatenate(restored_points)
            restored_colors = np.concatenate(restored_colors)
            np.testing.assert_allclose(restored_points, points, atol=1e-4)
            np.testing.assert_array_equal(restored_colors, colors)

    def test_organized_pointmap_becomes_surface_mesh_without_depth_bridges(self):
        yy, xx = np.mgrid[:3, :3]
        camera = np.column_stack((xx.reshape(-1) * 0.01, yy.reshape(-1) * 0.01, np.full(9, 2.0)))
        triangles = triangulate_grid(camera, camera, np.ones(9), (3, 3))
        self.assertEqual(triangles.shape, (8, 3, 3))
        discontinuous = camera.copy(); discontinuous[:, 2] = np.where(xx.reshape(-1) == 2, 8.0, 2.0)
        split = triangulate_grid(discontinuous, discontinuous, np.ones(9), (3, 3))
        self.assertLess(len(split), len(triangles))

    def test_mesh_snapshot_quantized_round_trip(self):
        yy, xx = np.mgrid[:4, :4]
        vertices = np.column_stack(
            (xx.reshape(-1) * 0.01, yy.reshape(-1) * 0.01, np.full(16, 2.0))
        ).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            manifest = create_mesh_snapshot(
                directory, [(vertices, vertices, np.ones(16), (4, 4))], triangles_per_chunk=4
            )
            self.assertTrue(manifest.encoding.startswith("m3mesh-v1"))
            restored = np.concatenate(
                [decode_mesh_chunk((Path(directory) / chunk.filename).read_bytes()) for chunk in manifest.chunks]
            )
            self.assertEqual(manifest.point_count, restored.size // 3)
            self.assertEqual(restored.shape, (18, 3, 3))
            self.assertTrue(np.isfinite(restored).all())


class CameraPreviewTests(unittest.TestCase):
    def test_current_camera_projection_uses_pose_aligned_pinhole_view(self):
        points = np.asarray([[0, 0, 2], [0.5, 0, 2], [0, 0, -1]], dtype=np.float32)
        u, v, depth, indices = project_camera_points(
            points, (100, 100, 50, 25), (100, 50), (200, 100)
        )
        np.testing.assert_allclose(u, [100, 150], atol=1e-5)
        np.testing.assert_allclose(v, [50, 50], atol=1e-5)
        np.testing.assert_allclose(depth, [2, 2])
        np.testing.assert_array_equal(indices, [0, 1])

    def test_camera_intrinsics_follow_clockwise_frame_rotation(self):
        rotated, size = rotate_pinhole_intrinsics((800, 700, 600, 300), 1280, 720, 90)
        self.assertEqual(size, (720, 1280))
        self.assertEqual(rotated, (700, 800, 419, 600))


class ArAlignmentMathTests(unittest.TestCase):
    def test_axes_scale_and_column_major_round_trip(self):
        slam = sim3_matrix(np.asarray([1, 2, 3, 0, 0, 0, 1, 2], dtype=np.float32))
        ar_camera = np.eye(4)
        ar_camera[:3, 3] = [10, 20, 30]
        aligned = ar_world_from_slam_map(ar_camera, slam)
        camera_origin_slam = slam @ np.asarray([0, 0, 0, 1], dtype=np.float64)
        np.testing.assert_allclose(
            aligned @ camera_origin_slam,
            ar_camera @ np.asarray([0, 0, 0, 1]),
            atol=1e-6,
        )
        np.testing.assert_allclose(from_column_major(to_column_major(aligned)), aligned)


class FakeProcessor:
    def __init__(self):
        self.frames = []
        self.loaded = False

    def process(self, frame, *, relocalizing):
        self.frames.append((frame.frame_id, relocalizing))
        return ProcessResult(frame.frame_id, Tracking.TRACKING, b"preview", True, len(self.frames))

    def checkpoint(self):
        return Checkpoint({"ids": np.asarray([x[0] for x in self.frames])}, {"keyframe_count": len(self.frames)})

    def load_checkpoint(self, checkpoint): self.loaded = True
    def finalize(self): return b"trajectory"
    def mobile_snapshot(self, output_dir): raise NotImplementedError
    def align_ar(self, frame): return None


class FailingProcessor(FakeProcessor):
    def process(self, frame, *, relocalizing):
        raise RuntimeError("synthetic failure")


class AligningProcessor(FakeProcessor):
    def align_ar(self, frame):
        return np.asarray([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32), 0.75


class EngineTests(unittest.TestCase):
    def test_ar_alignment_returns_model_matrix_only_after_match(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = SlamEngine(AligningProcessor(), ProjectStore(directory), model_id="m", config_fingerprint="c")
            engine.start("start", "scan", CAMERA)
            engine.pause("pause")
            engine.session.command("ar", "enter_ar")
            engine.session.ar_ready()
            pose = tuple(float(value) for value in np.eye(4).reshape(-1, order="F"))
            result = engine.align_ar(ArAlignmentFrame(1, 1, b"jpeg", (1, 1, 0, 0), pose, 1, 1))
            self.assertEqual(result.status, ArAlignmentStatus.ALIGNED)
            self.assertEqual(result.quality, 0.75)
            self.assertEqual(len(result.ar_world_from_slam_map), 16)
            engine.close()

    def test_continue_requires_camera_change_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = SlamEngine(FakeProcessor(), ProjectStore(directory), model_id="m", config_fingerprint="c")
            engine.start("start", "scan", CAMERA)
            engine.pause("pause")
            changed = CameraIdentity("phone", "2", 1280, 720, CAMERA.crop, CAMERA.intrinsics)
            with self.assertRaisesRegex(ValueError, "confirmation required"):
                engine.resume("resume-rejected", changed)
            self.assertEqual(engine.session.phase, SessionPhase.PAUSED)
            engine.resume("resume-confirmed", changed, camera_change_confirmed=True)
            self.assertEqual(engine.session.phase, SessionPhase.SCANNING)
            engine.close()

    def test_lifecycle_and_processing_event(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = FakeProcessor()
            engine = SlamEngine(processor, ProjectStore(directory), model_id="m", config_fingerprint="c")
            done = threading.Event(); events = []
            engine.on("frame_processed", lambda event: (events.append(event), done.set()))
            project = engine.start("start", "scan", CAMERA)
            engine.submit(NetworkFrame(10, 1, b"jpeg"))
            self.assertTrue(done.wait(1))
            self.assertEqual(events[0].data["processed_frame_id"], 10)
            engine.pause("pause")
            engine.stop("stop")
            self.assertEqual(engine.session.phase, SessionPhase.SAVED)
            self.assertEqual(ProjectStore(directory).list_projects()[0]["revision"], 1)
            self.assertFalse((ProjectStore(directory).revision_path(project) / "reconstruction.ply").exists())
            engine.close()

    def test_pause_is_immediate_and_resume_waits_for_background_save(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            engine = SlamEngine(FakeProcessor(), store, model_id="m", config_fingerprint="c")
            engine.start("start", "scan", CAMERA)
            gate = threading.Event()
            original = store.save_revision

            def slow_save(*args, **kwargs):
                gate.wait(2)
                return original(*args, **kwargs)

            with mock.patch.object(store, "save_revision", side_effect=slow_save):
                started = time.monotonic()
                engine.pause("pause")
                self.assertLess(time.monotonic() - started, 0.2)
                self.assertEqual(engine.session.phase, SessionPhase.PAUSED)
                resumed = threading.Event()
                thread = threading.Thread(target=lambda: (engine.resume("resume", CAMERA), resumed.set()))
                thread.start()
                self.assertFalse(resumed.wait(0.1))
                gate.set()
                self.assertTrue(resumed.wait(2))
                thread.join()
            engine.close()

    def test_failed_pause_save_stays_paused_and_stop_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            engine = SlamEngine(FakeProcessor(), store, model_id="m", config_fingerprint="c")
            engine.start("start", "scan", CAMERA)
            original = store.save_revision
            attempts = 0

            def flaky_save(*args, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError("disk temporarily unavailable")
                return original(*args, **kwargs)

            with mock.patch.object(store, "save_revision", side_effect=flaky_save):
                engine.pause("pause")
                with self.assertRaisesRegex(RuntimeError, "not completed successfully"):
                    engine._wait_for_save()
                self.assertEqual(engine.session.phase, SessionPhase.PAUSED)
                engine.stop("stop")
            self.assertEqual(engine.session.phase, SessionPhase.SAVED)
            self.assertEqual(attempts, 2)
            self.assertEqual(store.read_manifest(engine.project_id)["latest_revision"], 1)
            engine.close()

    def test_ar_snapshot_failure_returns_to_paused(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = SlamEngine(FakeProcessor(), ProjectStore(directory), model_id="m", config_fingerprint="c")
            engine.start("start", "scan", CAMERA)
            engine.pause("pause")
            with self.assertRaises(NotImplementedError):
                engine.enter_ar("ar")
            self.assertEqual(engine.session.phase, SessionPhase.PAUSED)
            with self.assertRaises(NotImplementedError):
                engine.enter_ar("ar")
            self.assertEqual(engine.session.phase, SessionPhase.PAUSED)
            engine.close()

    def test_failed_stop_rolls_back_and_same_command_id_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            engine = SlamEngine(FakeProcessor(), store, model_id="m", config_fingerprint="c")
            engine.start("start", "scan", CAMERA)
            with mock.patch.object(store, "save_revision", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(RuntimeError, "not completed successfully"):
                    engine.stop("stop")
            self.assertEqual(engine.session.phase, SessionPhase.PAUSED)
            engine.stop("stop")
            self.assertEqual(engine.session.phase, SessionPhase.SAVED)
            engine.close()

    def test_processing_failure_is_exposed_and_written_to_project_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = SlamEngine(FailingProcessor(), ProjectStore(directory), model_id="m", config_fingerprint="c")
            project = engine.start("start", "scan", CAMERA)
            engine.submit(NetworkFrame(1, 1, b"jpeg"))
            deadline = time.monotonic() + 1
            while engine.session.phase != SessionPhase.ERROR and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(engine.session.phase, SessionPhase.ERROR)
            self.assertEqual(engine.last_error, "RuntimeError: synthetic failure")
            journal = (ProjectStore(directory).project_path(project) / "journal.log").read_text()
            self.assertIn("PROCESSING_FAILED", journal)
            self.assertIn("RuntimeError: synthetic failure", journal)
            self.assertIn("Traceback", journal)
            engine.close()

    def test_close_saves_dirty_state_after_processing_error(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            engine = SlamEngine(FailingProcessor(), store, model_id="m", config_fingerprint="c")
            project = engine.start("start", "scan", CAMERA)
            engine.submit(NetworkFrame(1, 1, b"jpeg"))
            deadline = time.monotonic() + 1
            while engine.session.phase != SessionPhase.ERROR and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(engine.session.phase, SessionPhase.ERROR)
            engine.close()
            self.assertEqual(store.read_manifest(project)["latest_revision"], 1)


if __name__ == "__main__":
    unittest.main()
