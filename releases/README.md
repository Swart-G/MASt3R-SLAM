# Local release artifacts

Expected files:

- `mast3r-slam-android-debug.apk` — installable Android debug APK;
- `mast3r-slam-linux-x86_64` — native console launcher;
- `SHA256SUMS` — SHA-256 checksums for both artifacts.

The Linux file is a self-extracting first-run installer and launcher. It embeds
the application sources and `uv`; it does not need a repository checkout or a
preinstalled Python. Keep the binary anywhere in a writable directory and run:

```bash
./mast3r-slam-linux-x86_64
```

The first run creates `mast3r-slam-app/` beside the binary, displays every
installation step and live log in a TUI, and then opens the application. It
downloads Python packages and about 2.9 GiB of model checkpoints. Allow at
least 12 GiB of free space. A working NVIDIA driver, CUDA toolkit (`nvcc`), C++
compiler and internet connection are required. Later runs start immediately.

The complete first-run log is saved as `mast3r-slam-app/install.log`. To only
prepare the installation, use `./mast3r-slam-linux-x86_64 --install-only`.

The APK is debug-signed for direct device installation and is not intended for
Google Play publication.

Install it over USB with:

```bash
adb install -r releases/mast3r-slam-android-debug.apk
```
