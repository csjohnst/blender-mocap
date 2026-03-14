#!/bin/bash
# scripts/build_addon.sh — Package the addon as a .zip for Blender installation
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/dist"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Create zip with the blender_mocap/ directory at the root
cd "$PROJECT_DIR"
zip -r "$BUILD_DIR/blender_mocap.zip" blender_mocap/ \
    -x "blender_mocap/__pycache__/*" \
    -x "blender_mocap/capture_server/__pycache__/*"

echo "Built: $BUILD_DIR/blender_mocap.zip"
