{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  fast-float,
  enableShared ? !stdenv.hostPlatform.isStatic,
}:
# Coppied from https://github.com/nix-community/nur-combined/tree/main/repos/foolnotion
stdenv.mkDerivation rec {
  pname = "scnlib";
  version = "4.0.1";

  src = fetchFromGitHub {
    owner = "eliaskosunen";
    repo = "scnlib";
    rev = "v${version}";
    sha256 = "sha256-qEZAWhtvhKMkh7fk1yD17ErWGCpztEs0seV4AkBOy1I=";
  };

  nativeBuildInputs = [ cmake ];
  buildInputs = [ fast-float ];
  postPatch = ''
    substituteInPlace src/scn/impl.cpp \
      --replace "fast_float::scientific" "fast_float::chars_format::scientific" \
      --replace "fast_float::fixed" "fast_float::chars_format::fixed" \
      --replace "fast_float::general" "fast_float::chars_format::general" \
      --replace "fast_float::hex" "fast_float::chars_format::hex"
  '';

  cmakeFlags = [
    "-DSCN_TESTS=OFF"
    "-DSCN_EXAMPLES=OFF"
    "-DSCN_BENCHMARKS=OFF"
    "-DSCN_BENCHMARKS_BUILDTIME=OFF"
    "-DSCN_BENCHMARKS_BINARYSIZE=OFF"
    "-DSCN_USE_EXTERNAL_BENCHMARK=ON"
    "-DSCN_USE_EXTERNAL_FAST_FLOAT=ON"
    "-DENABLE_FULL=OFF"
    "-DSCN_INSTALL=ON"
    "-DBUILD_SHARED_LIBS=${if enableShared then "ON" else "OFF"}"
  ];
  # postInstall = ''
  #   ln -s $out/lib/cmake/scn/scn-config.cmake $out/lib/cmake/scn/scnlibConfig.cmake
  # '';
  meta = with lib; {
    description = "Modern C++ library for replacing scanf and std::istream";
    homepage = "https://scnlib.readthedocs.io/";
    license = licenses.asl20;
    platforms = platforms.all;
    #maintainers = with maintainers; [ foolnotion ];
  };
}
