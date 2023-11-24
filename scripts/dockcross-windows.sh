#!/usr/bin/env bash
#
IMAGE="dockcross/windows-static-x64-posix"
echo "Pulling image..."
docker run --rm "${IMAGE}" > ./dockcross

chmod +x ./dockcross

./dockcross cmake -B windows-build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=skylake -O3" -DCMAKE_C_FLAGS="-march=skylake -O3" -DENABLE_HOST_OPT=OFF -DGG_NO_PRAGMA=ON

./dockcross ninja -C windows-build occ

./dockcross bash -c "cd windows-build && cpack -G TXZ"

echo "Packaged and done"
