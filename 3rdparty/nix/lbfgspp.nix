{
  lib,
  stdenv,
  cmake,
  eigen,
  version,
  src,
}:

stdenv.mkDerivation {
  pname = "lbfgspp";
  inherit src;
  inherit version;
  nativeBuildInputs = [ cmake ];

  propagatedBuildInputs = [ eigen ];

  cmakeFlags = [
    "-DCMAKE_INSTALL_INCLUDEDIR=include"
  ];

  meta = with lib; {
    description = "A header-only C++ library for L-BFGS and L-BFGS-B algorithms";
    homepage = "https://github.com/yixuan/LBFGSpp";
    license = licenses.mit;
    platforms = platforms.all;
    maintainers = [ ];
  };
}
