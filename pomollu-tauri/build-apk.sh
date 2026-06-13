#!/usr/bin/env bash
# Build the Pomollu Android APK (GeckoView + Tauri 2).
#
# Prerequisites:
#   - ANDROID_HOME + ANDROID_NDK_ROOT (NDK r27+; HomuHomu833/android-ndk-custom
#     r30 works on aarch64 Linux hosts)
#   - rustup target add aarch64-linux-android
#   - npm i -g @tauri-apps/cli  (or use npx)
#   - One-time: `tauri android init` in this directory, then apply the
#     android-overlay/ files (step 2 below re-applies them on every build).
set -euo pipefail
cd "$(dirname "$0")"

GEN_MAIN=gen/android/app/src/main

# 1. Frontend → dist/
(cd ../pomollu-frontend && npm run build)
rm -rf dist && cp -r ../pomollu-frontend/build dist

# 2. Overlay GeckoView integration onto the generated Android project.
#    (tauri android init regenerates gen/ — the overlay must win.)
mkdir -p "$GEN_MAIN/java/com/shittimplana/pomollu" "$GEN_MAIN/assets"
cp -r android-overlay/java/com/shittimplana/pomollu/. "$GEN_MAIN/java/com/shittimplana/pomollu/"
cp -r android-overlay/assets/ipc-extension "$GEN_MAIN/assets/"

# 3. Frontend assets for the in-app AssetServer.
cp -r dist/. "$GEN_MAIN/assets/"

# 4. Build the APK (aarch64 only — GeckoView artifacts are per-ABI).
tauri android build -- --apk --target aarch64

APK=gen/android/app/build/outputs/apk/universal/release/app-universal-release-unsigned.apk
[ -f "$APK" ] && du -h "$APK" || { echo "APK not found at $APK" >&2; exit 1; }

# 5. Optional signing: ./build-apk.sh --sign (expects pomollu.keystore)
if [ "${1:-}" = "--sign" ]; then
  "${ANDROID_HOME}/build-tools/35.0.0/apksigner" sign \
    --ks pomollu.keystore --ks-key-alias pomollu \
    --out "pomollu-v${VERSION:-dev}.apk" "$APK"
  echo "signed: pomollu-v${VERSION:-dev}.apk"
fi
