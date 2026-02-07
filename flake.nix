{
  description = "Open Computational Chemestry";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    fast-float-src = {
      url = "github:fastfloat/fast_float?ref=v6.1.6";
      flake = false;
    };
    dftd4-src = {
      url = "github:peterspackman/cpp-d4?ref=main";
      flake = false;
    };
    lbfgspp-src = {
      url = "github:yixuan/LBFGSpp?ref=v0.4.0";
      flake = false;
    };
    libcint-src = {
      url = "github:peterspackman/libcint";
      flake = false;
    };
    gemmi-src = {
      url = "github:project-gemmi/gemmi?ref=v0.6.5";
      flake = false;
    };
  };
  outputs =
    {
      self,
      nixpkgs,
      fast-float-src,
      dftd4-src,
      lbfgspp-src,
      libcint-src,
      gemmi-src,
    }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      fast-float-6_1_6 = pkgs.fast-float.overrideAttrs {
        src = fast-float-src;
        version = "6.1.6";
      };
      scnlib = pkgs.callPackage ./3rdparty/nix/scnlib.nix {
        fast-float = fast-float-6_1_6;
      };
      dftd4 = pkgs.callPackage ./3rdparty/nix/dftd4.nix {
        src = dftd4-src;
        version = "main";
      };
      lbfgspp = pkgs.callPackage ./3rdparty/nix/lbfgspp.nix {
        src = lbfgspp-src;
        version = "0.4.0";
      };
      libcint = pkgs.callPackage ./3rdparty/nix/libcint.nix {
        version = "6.1.2";
        src = libcint-src;
      };
      gemmi = pkgs.callPackage ./3rdparty/nix/gemmi.nix {
        version = "0.6.5";
        src = gemmi-src;
      };
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
