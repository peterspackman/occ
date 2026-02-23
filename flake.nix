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
      python = pkgs.python314.withPackages (ps: [ ps.nanobind ]);
      CMAKE_PREFIX_PATH = "${python}/${python.sitePackages}";
    in
    {
      # 1. The Build Artifact (nix build)
      packages.${system} = {
        python = python;
        default = pkgs.stdenv.mkDerivation {
          name = "occ";
          src = ./.;
          postPatch = ''
            rm cmake/FindLibxc.cmake
          '';
          buildInputs = with pkgs; [
            cmake
            python
            cpm-cmake
            pkg-config
            spdlog
            onetbb
            tomlplusplus
            unordered_dense
            fmt
            nlohmann_json
            eigen_3_4_0
            libxc.dev
            cli11
            gemmi
            ccache
            gfortran.cc.lib
            zlib
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
            CMAKE_PREFIX_PATH = "${CMAKE_PREFIX_PATH}";
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
            "-DWITH_PYTHON_BINDINGS=ON"
          ];
          NIX_LDFLAGS = "-lquadmath";
        };
        gemmi = gemmi;
        libxc = pkgs.libxc.dev;
      };

      # 2. The Development Environment (nix develop)
      devShells.${system}.default = pkgs.mkShell {
        inputsFrom = [ self.packages.${system}.default ];

        # Extra development-only tools
        nativeBuildInputs = with pkgs; [
          llvm
          gcc
          gdb
          ninja
          clang-tools
          lld
        ];

        shellHook =
          let
            # List all dependencies that CMake needs to find
            deps = [
              pkgs.spdlog
              pkgs.onetbb
              pkgs.tomlplusplus
              pkgs.unordered_dense
              pkgs.fmt
              pkgs.nlohmann_json
              pkgs.eigen_3_4_0
              pkgs.libxc
              pkgs.cli11
              gemmi
              scnlib
              dftd4
              libcint
              lbfgspp
              fast-float-6_1_6
            ];
            runtimeLibs = with pkgs; [
              stdenv.cc.cc.lib
              zlib
              glib
            ];
            buildInputs = with pkgs; [
              uv
            ];
          in
          ''
            # Construct CMAKE_PREFIX_PATH by joining the store paths with semicolons
            export CMAKE_PREFIX_PATH="${
              pkgs.lib.makeSearchPathOutput "dev" "" deps
            }:${pkgs.lib.makeBinPath deps}"

            # Also export PKG_CONFIG_PATH just in case some libs don't have CMake configs
            export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" deps}:./cmake"

            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}"
          '';

      };
    };
}
