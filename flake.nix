{
  description = "Open Computational Chemestry";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      fast-float-6_1_6 = pkgs.fast-float.overrideAttrs (old: rec {
        version = "6.1.6";
        src = pkgs.fetchFromGitHub {
          owner = "fastfloat";
          repo = "fast_float";
          rev = "v${version}";
          hash = "sha256-MEJMPQZZZhOFiKlPAKIi0zVzaJBvjAlbSyg3wLOQ1fg=";
        };
      });
      dftd4 = pkgs.callPackage ./3rdparty/nix/dftd4.nix { };
      lbfgspp = pkgs.callPackage ./3rdparty/nix/lbfgspp.nix { };
      libcint = pkgs.callPackage ./3rdparty/nix/libcint.nix { };
      scnlib = pkgs.callPackage ./3rdparty/nix/scnlib.nix {
        fast-float = fast-float-6_1_6;
      };
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

    in
    {
      # 1. The Build Artifact (nix build)
      packages.${system} = {
        default = pkgs.stdenv.mkDerivation {
          name = "occ";
          src = ./.;
          buildInputs = [
            pkgs.cmake
            # Project dependencies installed with cpm.cmake
            pkgs.cpm-cmake
            pkgs.pkg-config
            pkgs.spdlog
            pkgs.onetbb
            pkgs.tomlplusplus
            pkgs.unordered_dense
            pkgs.fmt
            pkgs.nlohmann_json
            pkgs.eigen_3_4_0
            pkgs.libxc
            pkgs.cli11
            pkgs.gemmi
            pkgs.zlib
            scnlib
            dftd4
            libcint
            lbfgspp
            fast-float-6_1_6
          ];
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
          ];
          cmakeFlags = [
            "-GNinja"
            "-DCPM_DOWNLOAD_LOCATION=${pkgs.cpm-cmake}/share/cpm/CPM.cmake"
            "-DUSE_SYSTEM_EIGEN=ON"
            "-DUSE_SYSTEM_LIBXC=ON"
            "-DCPM_USE_LOCAL_PACKAGES=ON"
            "-DNIX_BUILD=ON"
          ];
        };
        dftd4 = dftd4;
      };

      # 2. The Development Environment (nix develop)
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          cmake
          cargo
          gcc
          clang-tools
          spdlog
          lbfgspp
          gdb
          libcint
          dftd4
          pkg-config
        ];

      };
    };
}
