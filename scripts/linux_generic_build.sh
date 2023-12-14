BUILD_DIR="build"
cmake . -B"${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DENABLE_HOST_OPT=OFF -GNinja -DCMAKE_CXX_FLAGS="-O2 -march=skylake -static-libgcc -static-libstdc++" -DCMAKE_C_FLAGS="-march=skylake -O2 -static-libgcc"
cmake --build "${BUILD_DIR}" --target occ
cd "${BUILD_DIR}" && cpack -G TXZ && cd -
