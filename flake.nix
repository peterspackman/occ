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
      gemmi = pkgs.callPackage ./3rdparty/nix/gemmi.nix { };
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

    in
    {
      # 1. The Build Artifact (nix build)
      packages.${system} = {
        default = pkgs.stdenv.mkDerivation {
          name = "occ";
          src = ./.;
          postPatch = ''
            rm cmake/FindLibxc.cmake
          '';
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
            pkgs.libxc.dev
            pkgs.cli11
            gemmi
            pkgs.ccache
            pkgs.gfortran.cc.lib
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
            pkgs.ccache
          ];
          env = {
            CCACHE_DIR = "/var/cache/ccache";
            CCACHE_BASEDIR = "$NIX_BUILD_TOP";
            CCACHE_SLOPPINESS = "locale,time_macros";
          };
          cmakeFlags = [
            "-GNinja"
            "-DCPM_DOWNLOAD_LOCATION=${pkgs.cpm-cmake}/share/cpm/CPM.cmake"
            "-DUSE_SYSTEM_EIGEN=ON"
            "-DUSE_SYSTEM_LIBXC=ON"
            "-DCPM_USE_LOCAL_PACKAGES=ON"
            "-DCMAKE_C_COMPILER_LAUNCHER=${pkgs.ccache}/bin/ccache"
            "-DCMAKE_CXX_COMPILER_LAUNCHER=${pkgs.ccache}/bin/ccache"
            "-DNIX_BUILD=ON"
          ];
          NIX_LDFLAGS = "-lquadmath";
        };
        gemmi = gemmi;
        libxc = pkgs.libxc.dev;
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
          pkg-config
        ];

      };
    };
}
