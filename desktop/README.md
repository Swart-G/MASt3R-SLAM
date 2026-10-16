# Linux desktop layer

`launcher.c` is a native first-run bootstrap and the permanent entrypoint. The
release builder appends two payloads to the ELF: a standalone `uv` executable
and a deterministic ZIP containing the application. No repository checkout or
system Python is needed on the target machine.

On first launch it creates `mast3r-slam-app` next to the executable, then:

1. checks disk space, the C++ compiler, CUDA toolkit and NVIDIA driver;
2. installs a private Python 3.11 runtime and creates `venv`;
3. extracts the bundled sources and installs a GPU-matched PyTorch build;
4. selects a CUDA-compatible host compiler, applies the PyTorch/GCC 15+
   compatibility fix and builds every CUDA dependency for the visible GPU;
5. downloads and SHA-256-verifies the three model checkpoints;
6. imports the application and CUDA extension before marking setup complete.

The full-screen ANSI TUI includes live subprocess output. The same output is
persisted to `mast3r-slam-app/install.log`. Interrupted downloads resume, and an
interrupted installation is retried on the next launch. Once installed, the
launcher directly executes `venv/bin/python -m mast3r_slam.cli`.

Build only the Linux artifact with:

```bash
scripts/build_linux_release.sh
```

Set `UV_BINARY=/path/to/uv` when `uv` is not on `PATH`. Useful launcher-only
options are `--install-only`, `--print-install-dir` and `--bootstrap-version`. Set the absolute
`MAST3R_SLAM_APP_DIR` environment variable to override the adjacent directory.

Choose `mast3r-slam tui` for the terminal interface (also the default without
arguments), or `mast3r-slam web` for the local Bootstrap panel. For the Docker
Compose workflow see [the web panel guide](../docs/web-panel.md).

Other installed commands are `serve`, `projects`, `view`, `continue`, `export`,
and `legacy` (forward options after `legacy --`).
