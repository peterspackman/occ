#!/usr/bin/env bash
NAME="wasm"
BUILD_DIR="wasm"

emcmake cmake . -B"${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -GNinja \
  -DCPACK_SYSTEM_NAME="${NAME}" -DENABLE_JS_BINDINGS=ON -DUSE_SYSTEM_EIGEN=OFF \
  -DCMAKE_CXX_FLAGS="-msimd128" -DCMAKE_C_FLAGS="-msimd128"

cmake --build "${BUILD_DIR}" --target occjs --target occ.wasm
