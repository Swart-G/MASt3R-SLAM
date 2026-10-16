from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import subprocess
import sys

from .mobile.projects import ProjectError, ProjectStore


DEFAULT_PROJECTS = pathlib.Path(os.environ.get("MAST3R_SLAM_PROJECTS", "projects"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mast3r-slam", description="Mobile MASt3R-SLAM scanner")
    parser.add_argument("--projects-dir", type=pathlib.Path, default=DEFAULT_PROJECTS)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("tui", help="открыть терминальный интерфейс")
    panel = sub.add_parser("web", help="запустить веб-панель")
    panel.add_argument("--host", default="127.0.0.1")
    panel.add_argument("--port", type=int, default=8080)
    panel.add_argument("--phone-port", type=int, default=8765)
    panel.add_argument("--config", default="config/base.yaml")
    panel.add_argument("--device", default="cuda:0")
    serve = sub.add_parser("serve", help="start the paired phone server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--config", default="config/base.yaml")
    serve.add_argument("--device", default="cuda:0")
    serve.add_argument("--continue", dest="project_id")
    sub.add_parser("projects", help="list saved projects")
    view = sub.add_parser("view", help="open a completed reconstruction")
    view.add_argument("project_id")
    cont = sub.add_parser("continue", help="continue a project after relocalization")
    cont.add_argument("project_id")
    cont.add_argument("--host", default="0.0.0.0")
    cont.add_argument("--port", type=int, default=8765)
    cont.add_argument("--config", default="config/base.yaml")
    export = sub.add_parser("export", help="copy the latest project PLY")
    export.add_argument("project_id")
    export.add_argument("output", nargs="?", type=pathlib.Path)
    legacy = sub.add_parser("legacy", help="run the original dataset/video/RealSense pipeline")
    legacy.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = ProjectStore(args.projects_dir)
    try:
        if args.command in {None, "tui"}:
            return _run_tui(store)
        if args.command == "web":
            from .web.app import run_panel

            return run_panel(store, host=args.host, port=args.port, phone_port=args.phone_port,
                             config=args.config, device=args.device)
        if args.command == "projects":
            projects = store.list_projects()
            if not projects:
                print("No projects")
            for item in projects:
                mib = item["size_bytes"] / 1024 / 1024
                print(
                    f"{item['project_id']}  {item['updated_at']}  {item['state']:<7} "
                    f"rev={item['revision']} kf={item['keyframes']} {mib:.1f} MiB  {item['name']}"
                )
            return 0
        if args.command == "export":
            target = args.output or pathlib.Path(f"{args.project_id}.ply")
            exported = store.export_ply(args.project_id, target)
            print(exported)
            return 0
        if args.command == "view":
            source = store.revision_path(args.project_id) / "reconstruction.ply"
            if not source.is_file() and not (source.parent / "checkpoint.npz").is_file():
                raise ProjectError("project has neither a finalized PLY nor a saved checkpoint")
            return subprocess.call([sys.executable, "-m", "mast3r_slam.desktop_viewer", str(source)])
        if args.command == "legacy":
            root = pathlib.Path(__file__).resolve().parents[1]
            legacy_args = args.args[1:] if args.args[:1] == ["--"] else args.args
            return subprocess.call([sys.executable, str(root / "main.py"), *legacy_args])
        if args.command in {"serve", "continue"}:
            project_id = args.project_id if args.command in {"serve", "continue"} else None
            return asyncio.run(
                _serve(
                    store,
                    host=args.host,
                    port=args.port,
                    config_path=args.config,
                    device=getattr(args, "device", "cuda:0"),
                    project_id=project_id,
                )
            )
    except (ProjectError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


async def _serve(store, *, host: str, port: int, config_path: str, device: str, project_id: str | None) -> int:
    from .mobile.engine import SlamEngine
    from .mobile.mast3r_processor import Mast3rProcessor
    from .mobile.server import MobileServer

    config_bytes = pathlib.Path(config_path).read_bytes()
    processor = Mast3rProcessor(config_path, device)
    engine = SlamEngine(
        processor,
        store,
        model_id="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        config_fingerprint=hashlib.sha256(config_bytes).hexdigest(),
    )
    if project_id:
        engine.load_project(project_id)
    server = MobileServer(engine, host=host, port=port)
    await server.start()
    print("Pair the Android app with:")
    _print_qr(server.pairing_uri)
    print(server.pairing_uri)
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await server.stop()
    return 0


def _print_qr(value: str) -> None:
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(value)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except ImportError:
        pass


def _run_tui(store: ProjectStore) -> int:
    try:
        from .tui import run_tui
    except ImportError:
        print("Textual is not installed; use `mast3r-slam serve` or `mast3r-slam projects`.")
        return 2
    return run_tui(store)


if __name__ == "__main__":
    raise SystemExit(main())
