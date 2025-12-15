#!/usr/bin/env bash

# Run the DSE sweep test, then compile a LaTeX figure.
# Usage: ./run_dse_and_figure.sh [path/to/tex]  (default: validation.tex)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Build Ramulator2 as a library if it's missing.
RAMULATOR_DIR="$ROOT/bluespec/ramulator2"
RAMULATOR_LIB="$RAMULATOR_DIR/libramulator.so"

build_ramulator() {
  if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake not found; please install it to build ramulator2." >&2
    exit 1
  fi
  if ! command -v make >/dev/null 2>&1; then
    echo "make not found; please install it to build ramulator2." >&2
    exit 1
  fi

  echo "Building ramulator2 (library) ..."
  (
    set -euo pipefail
    cd "$RAMULATOR_DIR"
    mkdir -p build
    cd build
    cmake ..
    make -j"$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"
    # Copy outputs up one level so BSC can find them via -L./ramulator2.
    # The exact output path can vary; try common locations and fail if none found
    # and no existing parent-level library is present.
    LIB_SRC=""
    for candidate in "./libramulator.so" "./ramulator2/libramulator.so" "./lib/libramulator.so"; do
      if [[ -f "$candidate" ]]; then
        LIB_SRC="$candidate"
        break
      fi
    done
    if [[ -n "$LIB_SRC" ]]; then
      cp -f "$LIB_SRC" ../libramulator.so
    elif [[ -f ../libramulator.so ]]; then
      echo "libramulator.so already present; not copying from build dir."
    else
      echo "libramulator.so not found after build; looked in ./, ./ramulator2/, ./lib/" >&2
      exit 1
    fi
    # Also copy the executable for convenience if it was built.
    if [[ -f ./ramulator2 ]]; then
      cp -f ./ramulator2 ../
    fi
  )
  echo "ramulator2 build complete."
}

if [[ ! -f "$RAMULATOR_LIB" ]]; then
  build_ramulator
else
  echo "ramulator2 library already present; skip rebuild."
fi

cargo test run_dse_sweep -- --nocapture

TEX_FILE="${1:-validation.tex}"
if [[ ! -f "$TEX_FILE" ]]; then
  echo "TeX file not found: $TEX_FILE" >&2
  exit 1
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk not found; please install it to build the figure." >&2
  exit 1
fi

# Build the PDF in the TeX file's directory to keep outputs together.
TEX_DIR="$(cd "$(dirname "$TEX_FILE")" && pwd)"
TEX_BASENAME="$(basename "$TEX_FILE")"

(
  cd "$TEX_DIR"
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$TEX_BASENAME"
)
