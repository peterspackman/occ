#!/usr/bin/env bash
BUILD_DIR="build"
ARCH="x86_64"
NAME="windows"

if [ $# -gt 0 ]; then
    ARCH="$1"
fi
if [ $# -gt 1 ]; then
   NAME="$2"
fi
IMAGE="dockcross/windows-static-x64-posix"

echo "Pulling image..."
docker run --rm "${IMAGE}" > ./dockcross

chmod +x ./dockcross

./dockcross cmake -B ${BUILD_DIR} -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=skylake -O3" \
    -DCMAKE_C_FLAGS="-march=skylake -O3" \
    -DENABLE_HOST_OPT=OFF -DGG_NO_PRAGMA=ON \
    -DCPACK_SYSTEM_NAME="${NAME}"

./dockcross cmake --build ${BUILD_DIR} --target occ

./dockcross bash -c "cd build && cpack -G TXZ"

echo "Packaged and done"
