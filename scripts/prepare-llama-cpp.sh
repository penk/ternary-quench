#!/usr/bin/env bash
set -euo pipefail

# Known-good upstream used for the Qwen3.8 Hybrid GGUF release.
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-4d9176092d00586775af140581bb0b558ddc4389}"
LLAMA_CPP_DIR="${1:-vendor/llama.cpp}"
BUILD_JOBS="${BUILD_JOBS:-8}"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone --filter=blob:none --no-checkout \
    https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
elif ! git -C "$LLAMA_CPP_DIR" diff --quiet || \
     ! git -C "$LLAMA_CPP_DIR" diff --cached --quiet; then
  echo "refusing to replace a modified llama.cpp checkout: $LLAMA_CPP_DIR" >&2
  exit 1
fi

git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin "$LLAMA_CPP_COMMIT"
git -C "$LLAMA_CPP_DIR" checkout --detach FETCH_HEAD

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
  -DGGML_CCACHE=OFF
cmake --build "$LLAMA_CPP_DIR/build" \
  --target llama-cli llama-server \
  -j "$BUILD_JOBS"

echo "llama.cpp ready at $LLAMA_CPP_DIR ($LLAMA_CPP_COMMIT)"
