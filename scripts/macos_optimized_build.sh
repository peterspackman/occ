BUILD_DIR="build-optimized"
cmake . -B"${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_BLAS=ON -DENABLE_HOST_OPT=ON -GNinja -DCMAKE_CXX_FLAGS="-O3 -ffast-math -march=native" -DCMAKE_C_FLAGS="-O3 -ffast-math -march=native"
cmake --build "${BUILD_DIR}" --target occ
cd "${BUILD_DIR}" && cpack -G TXZ && cd -
