from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _downloader_module():
    path = ROOT / "desktop" / "download_checkpoints.py"
    spec = importlib.util.spec_from_file_location("download_checkpoints", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_preparer_module():
    path = ROOT / "desktop" / "prepare_build_environment.py"
    spec = importlib.util.spec_from_file_location("prepare_build_environment", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckpointDownloaderTests(unittest.TestCase):
    def test_download_and_existing_file(self):
        module = _downloader_module()
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            content = (b"mast3r-slam-checkpoint" * 1024) + b"end"
            source = directory / "source.bin"
            destination = directory / "destination.bin"
            source.write_bytes(content)
            expected = hashlib.sha256(content).hexdigest()

            module.download(source.as_uri(), destination, expected)
            self.assertEqual(destination.read_bytes(), content)

            source.unlink()
            module.download(source.as_uri(), destination, expected)
            self.assertEqual(destination.read_bytes(), content)

    def test_checksum_failure_removes_partial(self):
        module = _downloader_module()
        with tempfile.TemporaryDirectory() as directory_name, patch.object(module.time, "sleep"):
            directory = pathlib.Path(directory_name)
            source = directory / "source.bin"
            destination = directory / "destination.bin"
            source.write_bytes(b"wrong")

            with self.assertRaises(RuntimeError):
                module.download(source.as_uri(), destination, "0" * 64)

            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".bin.part").exists())


class BuildEnvironmentTests(unittest.TestCase):
    def test_torch_header_patch_is_idempotent(self):
        module = _build_preparer_module()
        with tempfile.TemporaryDirectory() as directory_name:
            header = pathlib.Path(directory_name) / "List_inl.h"
            header.write_text("prefix\n" + module.OLD_LIST_OPERATOR + "suffix\n")
            self.assertTrue(module.patch_torch_list_header(header))
            self.assertFalse(module.patch_torch_list_header(header))
            self.assertIn(module.FIXED_LIST_OPERATOR, header.read_text())

    def test_local_lietorch_dependency_handles_spaces(self):
        module = _build_preparer_module()
        with tempfile.TemporaryDirectory(prefix="MASt3R New Folder ") as directory_name:
            root = pathlib.Path(directory_name)
            pyproject = root / "pyproject.toml"
            pyproject.write_text(
                '[project]\ndependencies = [\n'
                '    "lietorch @ https://example.invalid/lietorch.zip",\n]\n'
            )
            lietorch = root / "thirdparty" / "lietorch"
            lietorch.mkdir(parents=True)
            self.assertTrue(module.patch_project_dependency(pyproject, lietorch))
            self.assertIn(lietorch.resolve().as_uri(), pyproject.read_text())
            self.assertFalse(module.patch_project_dependency(pyproject, lietorch))


class LauncherIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("MAST3R_SLAM_TEST_BINARY")
        cls.binary = pathlib.Path(configured) if configured else None

    def test_first_run_creates_directory_and_persistent_log(self):
        if self.binary is None:
            self.skipTest("MAST3R_SLAM_TEST_BINARY is not set")
        with tempfile.TemporaryDirectory() as directory_name:
            app_dir = pathlib.Path(directory_name) / "application"
            environment = {**os.environ, "PATH": "/missing", "MAST3R_SLAM_APP_DIR": str(app_dir)}
            result = subprocess.run(
                [self.binary, "--install-only"],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("[1/8]", result.stdout)
            self.assertTrue((app_dir / ".bootstrap" / "uv").is_file())
            log = (app_dir / "install.log").read_text()
            self.assertIn("cannot execute c++", log)
            self.assertFalse((app_dir / ".installed").exists())

    def test_installed_relaunch_forwards_arguments(self):
        if self.binary is None:
            self.skipTest("MAST3R_SLAM_TEST_BINARY is not set")
        version = subprocess.check_output([self.binary, "--bootstrap-version"], text=True).strip()
        with tempfile.TemporaryDirectory() as directory_name:
            app_dir = pathlib.Path(directory_name) / "application"
            python = app_dir / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            python.chmod(0o755)
            (app_dir / ".installed").write_text(version + "\n")
            result = subprocess.run(
                [self.binary, "projects"],
                env={**os.environ, "MAST3R_SLAM_APP_DIR": str(app_dir)},
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            )
            self.assertEqual(result.stdout.splitlines(), ["-m", "mast3r_slam.cli", "projects"])


if __name__ == "__main__":
    unittest.main()
