"""Smoke-test Web Panel → Android protocol → GPU → project → PLY."""

import argparse
import asyncio
import pathlib
import socket
import tempfile
import time
import uuid
import cv2
from aiohttp import ClientSession, TCPConnector
from aiohttp.test_utils import TestClient, TestServer
from mast3r_slam.mobile.projects import ProjectStore
from mast3r_slam.web.app import create_app, PANEL_KEY
from protocol import PROTOCOL_VERSION, mast3r_slam_pb2 as wire


async def main(args):
    with tempfile.TemporaryDirectory(prefix="mast3r-web-smoke-") as directory:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        app = create_app(
            ProjectStore(directory),
            phone_port=port,
            config=args.config,
            device=args.device,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        panel = app[PANEL_KEY]
        try:
            headers = {"X-Panel-Token": panel.csrf}
            response = await client.post(
                "/api/session", json={"name": "GPU smoke test"}, headers=headers
            )
            assert response.status == 202
            await panel.task
            assert panel.status()["phase"] == "READY", panel.error
            async with ClientSession(connector=TCPConnector(ssl=False)) as phone:
                async with phone.ws_connect(f"https://127.0.0.1:{port}/ws") as ws:
                    await ws.send_bytes(
                        wire.ClientMessage(
                            hello=wire.Hello(
                                protocol_version=PROTOCOL_VERSION,
                                token=panel.server.token,
                                device_id="web-smoke-test",
                            )
                        ).SerializeToString()
                    )
                    await ws.receive()
                    await ws.send_bytes(
                        wire.ClientMessage(
                            camera_config=wire.CameraConfig(
                                device_model="smoke",
                                physical_camera_id="0",
                                width=640,
                                height=480,
                                crop_width=640,
                                crop_height=480,
                                intrinsics=[517.3, 516.5, 318.6, 255.3],
                            )
                        ).SerializeToString()
                    )
                    await ws.send_bytes(
                        wire.ClientMessage(
                            command=wire.Command(
                                command_id=str(uuid.uuid4()), type=wire.START
                            )
                        ).SerializeToString()
                    )
                    await ws.receive()

                    async def heartbeat():
                        while True:
                            await ws.send_bytes(
                                wire.ClientMessage(
                                    heartbeat=wire.Heartbeat()
                                ).SerializeToString()
                            )
                            await asyncio.sleep(0.5)

                    pulse = asyncio.create_task(heartbeat())
                    try:
                        image = cv2.imread(str(args.image))
                        if image is None:
                            raise RuntimeError(f"Cannot read image: {args.image}")
                        image = cv2.resize(image, (640, 480))
                        ok, jpeg = cv2.imencode(".jpg", image)
                        assert ok
                        await ws.send_bytes(
                            wire.ClientMessage(
                                frame=wire.FramePacket(
                                    frame_id=1,
                                    timestamp_ns=time.time_ns(),
                                    jpeg=jpeg.tobytes(),
                                    width=640,
                                    height=480,
                                    physical_camera_id="0",
                                )
                            ).SerializeToString()
                        )
                        for _ in range(600):
                            if panel.preview:
                                break
                            if panel.error:
                                raise RuntimeError(panel.error)
                            await asyncio.sleep(0.2)
                        assert panel.preview, "no processed preview"
                        response = await client.get("/api/preview.jpg")
                        assert response.status == 200
                        print(
                            "GPU preview:",
                            len(await response.read()),
                            "bytes",
                            panel.stats,
                            flush=True,
                        )
                        response = await client.post(
                            "/api/commands/pause", json={}, headers=headers
                        )
                        assert response.status == 200, await response.text()
                        response = await client.post(
                            "/api/commands/stop", json={}, headers=headers
                        )
                        assert response.status == 200, await response.text()
                        projects = await (await client.get("/api/projects")).json()
                        assert len(projects) == 1
                        response = await client.get(
                            "/api/projects/" + projects[0]["project_id"] + "/export"
                        )
                        assert response.status == 200, await response.text()
                        size = 0
                        async for block in response.content.iter_chunked(1024 * 1024):
                            size += len(block)
                        assert size > 100
                        print(
                            "Saved revision:",
                            projects[0]["revision"],
                            "keyframes:",
                            projects[0]["keyframes"],
                            "PLY bytes:",
                            size,
                            flush=True,
                        )
                    finally:
                        pulse.cancel()
                        await asyncio.gather(pulse, return_exceptions=True)
        finally:
            await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=pathlib.Path)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--device", default="cuda:0")
    asyncio.run(main(parser.parse_args()))
