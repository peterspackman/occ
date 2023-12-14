#!/usr/bin/env bash
BUILD_DIR="build"
MIN_VERSION="10.15"
ARCH="x86_64"
NAME="macos"

# override architecture if provided as argument
if [ $# -gt 0 ]; then
    ARCH="$1"
fi
if [ $# -gt 1 ]; then
   NAME="$2"
fi

cmake . -B"${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_BLAS=ON -DENABLE_HOST_OPT=OFF -GNinja \
  -DCMAKE_OSX_ARCHITECTURES="${ARCH}" \
  -DCMAKE_CXX_FLAGS="-O2 -mmacosx-version-min=${MIN_VERSION}" \
  -DCMAKE_C_FLAGS="-O2 -mmacosx-version-min=${MIN_VERSION}" -DUSE_OPENMP=OFF \
  -DCPACK_SYSTEM_NAME="${NAME}"

cmake --build "${BUILD_DIR}" --target occ

cd "${BUILD_DIR}" && cpack -G TXZ && cd -
