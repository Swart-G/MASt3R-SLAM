#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_dir="$project_root/releases"
android_apk="$project_root/android-app/app/build/outputs/apk/debug/app-debug.apk"

mkdir -p "$release_dir"

"$project_root/scripts/build_linux_release.sh" \
  "$release_dir/mast3r-slam-linux-x86_64"

if [[ ! -f "$android_apk" ]]; then
  echo "Android APK is missing: $android_apk" >&2
  echo "Build it first with: cd android-app && ./gradlew :app:assembleDebug" >&2
  exit 2
fi

install -m 0644 "$android_apk" "$release_dir/mast3r-slam-android-debug.apk"
(
  cd "$release_dir"
  sha256sum mast3r-slam-android-debug.apk mast3r-slam-linux-x86_64 > SHA256SUMS
)

echo "Release artifacts:"
ls -lh "$release_dir/mast3r-slam-android-debug.apk" \
  "$release_dir/mast3r-slam-linux-x86_64" \
  "$release_dir/SHA256SUMS"
