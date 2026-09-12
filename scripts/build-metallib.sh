#!/usr/bin/env bash
# Build MLX's Metal library for SwiftPM command-line builds.
#
# Frigate vendors MLX in JIT mode: most kernels are compiled from source at runtime, but
# MLX still loads a small precompiled library at device init (the same kernel set its
# CMake builds when MLX_METAL_JIT=ON). `swift build` doesn't compile .metal files, so this
# script builds `mlx.metallib` from the vendored sources and places it next to every
# build product (MLX looks for a colocated mlx.metallib first).
#
# Usage: scripts/build-metallib.sh [debug|release|all]   (default: all)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Frigate is a sibling checkout (path dependency); fall back to a SwiftPM checkout.
if [[ -d "$ROOT/../Frigate/Sources/Cmlx/mlx" ]]; then
  MLX_SRC="$(cd "$ROOT/../Frigate/Sources/Cmlx/mlx" && pwd)"
else
  MLX_SRC="$ROOT/.build/checkouts/Frigate/Sources/Cmlx/mlx"
fi
KERNELS="$MLX_SRC/mlx/backend/metal/kernels"
OUT="$ROOT/.build/metallib"
CONFIGS="${1:-all}"

if [[ ! -d "$KERNELS" ]]; then
  echo "MLX sources not found — run 'swift package resolve' first." >&2
  exit 1
fi

mkdir -p "$OUT"
FLAGS=(-x metal -Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions)
KERNEL_LIST=(arg_reduce conv gemv layer_norm random rms_norm rope scaled_dot_product_attention fence)

AIRS=()
for k in "${KERNEL_LIST[@]}"; do
  src="$KERNELS/$k.metal"
  air="$OUT/$k.air"
  if [[ ! -f "$air" || "$src" -nt "$air" ]]; then
    echo "  metal  $k"
    xcrun -sdk macosx metal "${FLAGS[@]}" -c "$src" -I"$MLX_SRC" -o "$air"
  fi
  AIRS+=("$air")
done
xcrun -sdk macosx metallib "${AIRS[@]}" -o "$OUT/mlx.metallib"
echo "built $OUT/mlx.metallib"

place() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  cp "$OUT/mlx.metallib" "$dir/mlx.metallib"
  echo "  → $dir/mlx.metallib"
  # Test bundles load MLX from inside the bundle.
  for bundle in "$dir"/*.xctest; do
    [[ -d "$bundle/Contents/MacOS" ]] && cp "$OUT/mlx.metallib" "$bundle/Contents/MacOS/mlx.metallib" \
      && echo "  → $bundle/Contents/MacOS/mlx.metallib"
  done
}

for cfg in debug release; do
  if [[ "$CONFIGS" == "all" || "$CONFIGS" == "$cfg" ]]; then
    place "$ROOT/.build/$cfg"
  fi
done
