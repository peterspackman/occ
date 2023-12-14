#!/usr/bin/env bash
#
IMAGE="dockcross/windows-static-x64-posix"
BUILD_DIR="build"
echo "Pulling image..."
docker run --rm "${IMAGE}" > ./dockcross

chmod +x ./dockcross

./dockcross cmake -B ${BUILD_DIR} -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=skylake -O3" -DCMAKE_C_FLAGS="-march=skylake -O3" -DENABLE_HOST_OPT=OFF -DGG_NO_PRAGMA=ON

./dockcross cmake --build ${BUILD_DIR} occ

./dockcross bash -c "cd build && cpack -G TXZ"

echo "Packaged and done"
