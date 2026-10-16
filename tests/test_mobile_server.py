from __future__ import annotations

import asyncio
import json
import socket
import tempfile
import time
import unittest
from unittest import mock

from aiohttp import ClientSession, TCPConnector

from mast3r_slam.mobile.engine import ProcessResult, SlamEngine, Tracking
from mast3r_slam.mobile.projects import Checkpoint, ProjectStore
from mast3r_slam.mobile.server import MobileServer
from protocol import mast3r_slam_pb2 as wire


class Processor:
    def process(self, frame, *, relocalizing):
        return ProcessResult(frame.frame_id, Tracking.TRACKING, b"preview", True, 1)

    def checkpoint(self): return Checkpoint({}, {"keyframe_count": 1})
    def load_checkpoint(self, checkpoint): pass
    def finalize(self): return b"trajectory"
    def mobile_snapshot(self, output_dir): raise NotImplementedError
    def align_ar(self, frame): return None


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        probe = socket.socket(); probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]; probe.close()
        self.engine = SlamEngine(Processor(), ProjectStore(self.temp.name), model_id="m", config_fingerprint="c")
        self.server = MobileServer(self.engine, host="127.0.0.1", port=self.port)
        with mock.patch.object(self.server, "_publish_mdns"):
            await self.server.start()

    async def asyncTearDown(self):
        await self.server.stop()
        self.temp.cleanup()

    async def test_protobuf_websocket_pair_start_and_ack(self):
        async with ClientSession(connector=TCPConnector(ssl=False)) as client:
            async with client.ws_connect(f"https://127.0.0.1:{self.port}/ws") as ws:
                hello = wire.ClientMessage(
                    hello=wire.Hello(protocol_version=2, token=self.server.token, device_id="phone")
                )
                await ws.send_bytes(hello.SerializeToString())
                state = wire.ServerMessage.FromString((await ws.receive()).data)
                self.assertEqual(state.session_state.phase, wire.READY)
                camera = wire.ClientMessage(
                    camera_config=wire.CameraConfig(
                        device_model="phone", physical_camera_id="0", width=1280, height=720,
                        crop_width=1280, crop_height=720, intrinsics=[900, 900, 640, 360],
                    )
                )
                await ws.send_bytes(camera.SerializeToString())
                start = wire.ClientMessage(command=wire.Command(command_id="start", type=wire.START))
                await ws.send_bytes(start.SerializeToString())
                started = wire.ServerMessage.FromString((await ws.receive()).data)
                self.assertEqual(started.session_state.phase, wire.SCANNING)
                frame = wire.ClientMessage(frame=wire.FramePacket(frame_id=7, timestamp_ns=1, jpeg=b"jpeg"))
                await ws.send_bytes(frame.SerializeToString())
                received = []
                for _ in range(3):
                    item = wire.ServerMessage.FromString((await asyncio.wait_for(ws.receive(), 1)).data)
                    received.append(item)
                    if item.WhichOneof("payload") == "frame_ack": break
                ack = next(item.frame_ack for item in received if item.WhichOneof("payload") == "frame_ack")
                self.assertTrue(ack.accepted)
                self.assertEqual(ack.frame_id, 7)

    async def test_health_exposes_last_processing_error(self):
        self.engine.last_error = "RuntimeError: diagnostic"
        async with ClientSession(connector=TCPConnector(ssl=False)) as client:
            async with client.get(f"https://127.0.0.1:{self.port}/health") as response:
                self.assertEqual(response.status, 200)
                payload = await response.json()
        self.assertEqual(payload["error"], "RuntimeError: diagnostic")

    async def test_slow_pause_does_not_trigger_heartbeat_disconnect(self):
        original_pause = self.engine.pause

        def slow_pause(command_id):
            time.sleep(2.25)
            original_pause(command_id)

        async with ClientSession(connector=TCPConnector(ssl=False)) as client:
            async with client.ws_connect(f"https://127.0.0.1:{self.port}/ws") as ws:
                await ws.send_bytes(
                    wire.ClientMessage(
                        hello=wire.Hello(protocol_version=2, token=self.server.token, device_id="phone")
                    ).SerializeToString()
                )
                await ws.receive()
                await ws.send_bytes(
                    wire.ClientMessage(
                        camera_config=wire.CameraConfig(
                            device_model="phone", physical_camera_id="0", width=1280, height=720,
                            crop_width=1280, crop_height=720, intrinsics=[900, 900, 640, 360],
                        )
                    ).SerializeToString()
                )
                await ws.send_bytes(
                    wire.ClientMessage(command=wire.Command(command_id="start", type=wire.START)).SerializeToString()
                )
                await ws.receive()
                with mock.patch.object(self.engine, "pause", side_effect=slow_pause):
                    await ws.send_bytes(
                        wire.ClientMessage(command=wire.Command(command_id="pause", type=wire.PAUSE)).SerializeToString()
                    )
                    paused = wire.ServerMessage.FromString((await asyncio.wait_for(ws.receive(), 4)).data)
                self.assertEqual(paused.session_state.phase, wire.PAUSED)
                self.assertFalse(ws.closed)

    async def test_pairing_watchdog_starts_only_after_hello(self):
        async with ClientSession(connector=TCPConnector(ssl=False)) as client:
            async with client.ws_connect(f"https://127.0.0.1:{self.port}/ws") as ws:
                await asyncio.sleep(2.25)
                await ws.send_bytes(
                    wire.ClientMessage(
                        hello=wire.Hello(protocol_version=2, token=self.server.token, device_id="phone")
                    ).SerializeToString()
                )
                state = wire.ServerMessage.FromString((await asyncio.wait_for(ws.receive(), 1)).data)
                self.assertEqual(state.session_state.phase, wire.READY)
                self.assertFalse(ws.closed)

    async def test_send_tolerates_transport_closing_race(self):
        class ClosingSocket:
            closed = False

            async def send_bytes(self, payload):
                raise ConnectionResetError("closing transport")

        self.server._socket = ClosingSocket()
        await self.server._send(wire.ServerMessage())
        self.server._socket = None

    def test_advertised_address_uses_normal_default_route_not_policy_vpn(self):
        route = [{"dst": "default", "dev": "enp5s0", "prefsrc": "192.168.1.9", "metric": 100}]
        completed = mock.Mock(stdout=json.dumps(route))
        with mock.patch("mast3r_slam.mobile.server.subprocess.run", return_value=completed):
            self.assertEqual(MobileServer._advertised_address(), "192.168.1.9")


if __name__ == "__main__":
    unittest.main()
