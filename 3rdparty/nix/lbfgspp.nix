{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  eigen,
}:

stdenv.mkDerivation rec {
  pname = "lbfgspp";
  version = "0.4.0"; # Check GitHub for the latest release

  src = fetchFromGitHub {
    owner = "yixuan";
    repo = "LBFGSpp";
    rev = "v${version}";
    hash = "sha256-PUzZ2jUVgHq1LJDWIWW93KnV7vBBEdZlspOrE5TcYBc=";
  };

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
