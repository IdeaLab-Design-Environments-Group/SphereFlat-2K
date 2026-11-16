#!/usr/bin/env bash
# Build MoebiusRegistration's SphereMap on macOS arm64 (Homebrew gcc + fftw).
#
# Clones (if needed) and checks out the pinned commit, applies
# third_party/patches/*.patch idempotently, builds only Bin/Linux/SphereMap,
# and prints the absolute path of the binary on the last line of stdout.
#
# Requires: brew install fftw gcc
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_URL="https://github.com/mkazhdan/MoebiusRegistration"
PIN="2fac0c7"
SRC="$ROOT/third_party/MoebiusRegistration"
PATCHES="$ROOT/third_party/patches"
JOBS="${JOBS:-8}"

if [[ ! -d "$SRC/.git" ]]; then
  git clone "$REPO_URL" "$SRC" >&2
fi
cd "$SRC"
if ! git cat-file -e "${PIN}^{commit}" 2>/dev/null; then
  git fetch origin >&2
fi
# Only detach to the pin when HEAD differs; keeps an already-patched tree intact.
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "${PIN}^{commit}")" ]]; then
  git checkout -q "$PIN"
fi

shopt -s nullglob
for p in "$PATCHES"/*.patch; do
  if git apply --reverse --check "$p" 2>/dev/null; then
    echo "patch already applied: $(basename "$p")" >&2
  else
    git apply "$p"
    echo "patch applied: $(basename "$p")" >&2
  fi
done

BREW_PREFIX="$(brew --prefix)"
GCC_PREFIX="$(brew --prefix gcc)"
# Highest-versioned Homebrew g++-NN (Apple's g++ is clang, which lacks libgomp).
CXX="$(ls "$GCC_PREFIX"/bin/g++-[0-9]* 2>/dev/null | sort -V | tail -1)"
CC="${CXX/g++-/gcc-}"
if [[ -z "$CXX" || ! -x "$CXX" ]]; then
  echo "error: Homebrew g++ not found under $GCC_PREFIX/bin (brew install gcc)" >&2
  exit 1
fi
if [[ ! -f "$BREW_PREFIX/include/fftw3.h" ]]; then
  echo "error: fftw3.h not found under $BREW_PREFIX/include (brew install fftw)" >&2
  exit 1
fi
echo "compiler: $CXX ($("$CXX" -dumpfullversion))" >&2

# The Makefile appends its own -fpermissive -fopenmp ... to CFLAGS and
# -lgomp -lfftw3 -lfftw3f to LFLAGS; environment values are kept and extended.
# The release flags normally come from the 'all' target, so pass them here.
mkdir -p Bin/Linux/Objects/SOFT1.0 Bin/Linux/Objects/SphereMap
CFLAGS="-O3 -DRELEASE -funroll-loops -ffast-math -g" \
LFLAGS="-O3 -g -L$BREW_PREFIX/lib" \
  make -j"$JOBS" CC="$CC" CXX="$CXX" INCLUDE="$BREW_PREFIX/include -IInclude" \
  Bin/Linux/SphereMap >&2

BIN="$SRC/Bin/Linux/SphereMap"
test -x "$BIN"
echo "$BIN"
