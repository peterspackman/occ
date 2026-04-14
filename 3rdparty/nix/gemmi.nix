{ pkgs, stdenv, ... }:
stdenv.mkDerivation rec {
  pname = "gemmi";
  version = "0.6.5";
  src = pkgs.fetchFromGitHub {
    owner = "project-gemmi";
    repo = "gemmi";
    tag = "v${version}";
    hash = "sha256-JJ6YBsdL3J+d0ihuJ2Nowp40c7FkDdfTqBhDrxWgSFw=";
  };
  dontBuild = true;
  dontConfigure = true;

  installPhase = ''
      mkdir -p $out/include
      cp -r include/* $out/include/
      TARGET_DIR="$out/lib/cmake/gemmi"
      mkdir -p "$TARGET_DIR"
      cat > "$TARGET_DIR/gemmiConfig.cmake" <<EOF
      if(NOT TARGET gemmi::gemmi)
        add_library(gemmi::gemmi INTERFACE IMPORTED)
      endif()

      set_target_properties(gemmi::gemmi PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "$out/include"
      )
    EOF
  '';
}
