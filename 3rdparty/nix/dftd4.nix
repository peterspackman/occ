{
  stdenv,
  lib,
  fetchFromGitHub,
  gfortran,
  buildType ? "cmake",
  cmake,
  meson,
  ninja,
  pkg-config,
  python3,
  blas,
  lapack,
  mctc-lib,
  mstore,
  multicharge,
  eigen,
}:

assert !blas.isILP64 && !lapack.isILP64;
assert (
  builtins.elem buildType [
    "meson"
    "cmake"
  ]
);

stdenv.mkDerivation rec {
  pname = "dftd4";
  version = "2.2.0";
  src = fetchFromGitHub {
    owner = "peterspackman";
    repo = "cpp-d4";
    rev = "main";
    hash = "sha256-aBWw9DGt0s5nGvmHQZzuZDPTWRGCsaKCrp4NylHOzqw=";
  };

  nativeBuildInputs = [
    gfortran
    pkg-config
    python3
  ]
  ++ lib.optionals (buildType == "meson") [
    meson
    ninja
  ]
  ++ lib.optional (buildType == "cmake") cmake;

  buildInputs = [
    blas
    lapack
    eigen
  ];

  propagatedBuildInputs = [
    mctc-lib
    mstore
    multicharge
  ];

  cmakeFlags = [
    (lib.strings.cmakeBool "BUILD_SHARED_LIBS" (!stdenv.hostPlatform.isStatic))
    "-DDFTD4_USE_EIGEN=ON"
    "-DBUILD_SHARED_LIBS=OFF"
  ];

  doCheck = true;

  postPatch = ''
    patchShebangs --build \
      config/install-mod.py \
      app/tester.py
  '';

  preCheck = ''
    export OMP_NUM_THREADS=2
  '';
  postInstall = ''
    # 3. SIMPLIFIED: Everything is already in $out. 
    # Just fix the missing Config file in place.

    TARGET_DIR="$out/lib/cmake/dftd4"

    # Ensure the directory exists (just in case)
    mkdir -p "$TARGET_DIR"

    cat > "$TARGET_DIR/dftd4Config.cmake" <<'EOF'
      include(''${CMAKE_CURRENT_LIST_DIR}/dftd4Targets.cmake)
    EOF
  '';
  meta = {
    description = "Generally Applicable Atomic-Charge Dependent London Dispersion Correction";
    mainProgram = "dftd4";
    license = with lib.licenses; [
      lgpl3Plus
      gpl3Plus
    ];
    homepage = "https://github.com/grimme-lab/dftd4";
    platforms = lib.platforms.linux;
    maintainers = [ lib.maintainers.sheepforce ];
  };
}
