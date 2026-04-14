{
  stdenv,
  lib,
  fetchFromGitHub,
  cmake,
  blas,
  python3,
  src,
  version,
}:

stdenv.mkDerivation rec {
  pname = "libcint";
  inherit version;
  inherit src;

  postPatch = ''
    sed -i 's/libcint.so/libcint${stdenv.hostPlatform.extensions.sharedLibrary}/g' testsuite/*.py
    mkdir -p $out/lib/cmake/libcint
    cat > $out/lib/cmake/libcint/libcintConfig.cmake <<EOF
      set(Libcint_FOUND TRUE)
      # Use placeholder "out" to refer to this package's installation path
      set(LIBCINT_INCLUDE_DIRS "${placeholder "out"}/include")
      set(LIBCINT_LIBRARIES "${placeholder "out"}/lib/libcint.a")
      
      if(NOT TARGET libcint::libcint)
        add_library(libcint::libcint UNKNOWN IMPORTED)
        set_target_properties(libcint::libcint PROPERTIES
          IMPORTED_LOCATION "\''${LIBCINT_LIBRARIES}"
          INTERFACE_INCLUDE_DIRECTORIES "\''${LIBCINT_INCLUDE_DIRS}"
        )
      endif()
  '';
  nativeBuildInputs = [ cmake ];
  buildInputs = [ blas ];
  cmakeFlags = [
    "-DCMAKE_INSTALL_PREFIX=" # ends up double-adding /nix/store/... prefix, this avoids issue
    "-DWITH_FORTRAN=OFF"
    "-DWITH_CINT2_INTERFACE=OFF"
    "-DENABLE_STATIC=ON"
    "-DBUILD_SHARED_LIBS=OFF"
    "-DPYPZPX=ON"
    "-DBUILD_MARCH_NATIVE=ON"
    "-DWITH_RANGE_COULOMB=ON"
  ];
  env.NIX_CFLAGS_COMPILE = "-Wno-implicit-function-declaration -Wno-deprecated-non-prototype -D_GNU_SOURCE";
  strictDeps = true;

  doCheck = true;
  nativeCheckInputs = [ python3.pkgs.numpy ];

  meta = {
    description = "General GTO integrals for quantum chemistry";
    longDescription = ''
      libcint is an open source library for analytical Gaussian integrals.
      It provides C/Fortran API to evaluate one-electron / two-electron
      integrals for Cartesian / real-spheric / spinor Gaussian type functions.
    '';
    homepage = "http://wiki.sunqm.net/libcint";
    downloadPage = "https://github.com/sunqm/libcint";
    changelog = "https://github.com/sunqm/libcint/blob/master/ChangeLog";
    license = lib.licenses.bsd2;
    maintainers = [ ];
    platforms = lib.platforms.unix;
  };
}
