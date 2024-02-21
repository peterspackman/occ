#!/usr/bin/env bash
BUILD_DIR="build"
ARCH="x86_64"
NAME="linux"

if [ $# -gt 0 ]; then
    ARCH="$1"
fi
if [ $# -gt 1 ]; then
   NAME="$2"
fi

cmake . -B"${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
	-DENABLE_HOST_OPT=OFF -GNinja -DUSE_OPENMP=OFF \
	-DCMAKE_CXX_FLAGS="-O2 -march=skylake -static -static-libgcc -static-libstdc++" \
	-DCMAKE_C_FLAGS="-march=skylake -O2 -static -static-libgcc" -DBUILD_DOCS=ON \
	-DCPACK_SYSTEM_NAME="${NAME}"

cmake --build "${BUILD_DIR}"

cd "${BUILD_DIR}" && cpack -G TXZ && cd -
