#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$SCRIPT_DIR/../pomollu-engine"
WASM_DIR="$SCRIPT_DIR/src/lib/wasm"

echo "building wasm..."
wasm-pack build --release --target bundler "$ENGINE_DIR"

echo "copying wasm to $WASM_DIR..."
cp "$ENGINE_DIR/pkg/pomollu_engine.js" "$WASM_DIR/"
cp "$ENGINE_DIR/pkg/pomollu_engine.d.ts" "$WASM_DIR/"
cp "$ENGINE_DIR/pkg/pomollu_engine_bg.js" "$WASM_DIR/"
cp "$ENGINE_DIR/pkg/pomollu_engine_bg.wasm" "$WASM_DIR/"
cp "$ENGINE_DIR/pkg/pomollu_engine_bg.wasm.d.ts" "$WASM_DIR/"
cp "$ENGINE_DIR/pkg/package.json" "$WASM_DIR/"

echo "building web..."
cd "$SCRIPT_DIR"
npm run build

echo "done"
