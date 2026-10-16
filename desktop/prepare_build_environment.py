#!/usr/bin/env python3
"""Apply compatibility fixes needed before compiling CUDA extensions."""

from __future__ import annotations

import pathlib
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile


OLD_LIST_OPERATOR = """template<class T>
typename List<T>::internal_reference_type List<T>::operator[](size_type pos) {
  static_cast<void>(impl_->list.at(pos)); // Throw the exception if it is out of range.
  return {impl_->list.begin() + static_cast<typename decltype(impl_->list)::difference_type>(pos)};
}
"""

FIXED_LIST_OPERATOR = """template<class T>
typename List<T>::internal_reference_type List<T>::operator[](size_type pos) {
  static_cast<void>(impl_->list.at(pos)); // Throw the exception if it is out of range.
  using difference_type = std::vector<IValue>::difference_type;
  return {impl_->list.begin() + static_cast<difference_type>(pos)};
}
"""

LIETORCH_URL = "https://github.com/princeton-vl/lietorch/archive/e7df86554156b36846008d8ddbcc4d8521a16554.zip"
LIETORCH_SHA256 = "0963d5d5a8791c801c4ce24d840c1509c12e16cf7299901a5ea8a44cacd8659a"


def patch_torch_list_header(header: pathlib.Path) -> bool:
    source = header.read_text()
    if FIXED_LIST_OPERATOR in source:
        return False
    if OLD_LIST_OPERATOR not in source:
        raise RuntimeError(f"unsupported torch List_inl.h layout: {header}")
    header.write_text(source.replace(OLD_LIST_OPERATOR, FIXED_LIST_OPERATOR, 1))
    return True


def prepare_lietorch(app_dir: pathlib.Path) -> pathlib.Path:
    target = app_dir / "thirdparty" / "lietorch"
    if not (target / "setup.py").is_file():
        print("Downloading the pinned Lietorch source…", flush=True)
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=".lietorch-") as temporary_name:
            temporary = pathlib.Path(temporary_name)
            archive_path = temporary / "lietorch.zip"
            digest = hashlib.sha256()
            with urllib.request.urlopen(LIETORCH_URL, timeout=60) as response, archive_path.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
            if digest.hexdigest() != LIETORCH_SHA256:
                raise RuntimeError("Lietorch archive checksum mismatch")
            extracted = temporary / "source"
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extracted)
            roots = [item for item in extracted.iterdir() if item.is_dir()]
            if len(roots) != 1 or not (roots[0] / "setup.py").is_file():
                raise RuntimeError("unexpected Lietorch archive layout")
            if target.exists() or target.is_symlink():
                if target.is_symlink() or target.is_file():
                    target.unlink()
                else:
                    shutil.rmtree(target)
            os.replace(roots[0], target)
    eigen = target / "eigen"
    if eigen.is_symlink() or eigen.is_file():
        eigen.unlink()
    elif eigen.is_dir():
        shutil.rmtree(eigen)
    eigen.symlink_to("../eigen", target_is_directory=True)
    return target


def patch_project_dependency(pyproject: pathlib.Path, lietorch: pathlib.Path) -> bool:
    source = pyproject.read_text()
    replacement = f'    "lietorch @ {lietorch.resolve().as_uri()}",'
    lines = source.splitlines()
    matched = [index for index, line in enumerate(lines) if line.strip().startswith('"lietorch @ ')]
    if len(matched) != 1:
        raise RuntimeError(f"could not locate the Lietorch dependency in {pyproject}")
    changed = lines[matched[0]] != replacement
    lines[matched[0]] = replacement
    pyproject.write_text("\n".join(lines) + "\n")
    return changed


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: prepare_build_environment.py APP_DIR", file=sys.stderr)
        return 2
    app_dir = pathlib.Path(sys.argv[1]).resolve()
    try:
        import torch

        header = pathlib.Path(torch.__file__).parent / "include" / "ATen" / "core" / "List_inl.h"
        changed = patch_torch_list_header(header)
        lietorch = prepare_lietorch(app_dir)
        dependency_changed = patch_project_dependency(app_dir / "pyproject.toml", lietorch)
    except Exception as error:
        print(f"Could not prepare the PyTorch build headers: {error}", file=sys.stderr)
        return 1
    print("Patched PyTorch headers for GCC 15+." if changed else "PyTorch build headers are ready.")
    print("Pinned Lietorch source and Eigen headers are ready.")
    if dependency_changed:
        print("Configured the application to build the local Lietorch source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
