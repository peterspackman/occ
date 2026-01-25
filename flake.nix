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
      cli11-src = pkgs.fetchFromGitHub {
        owner = "CLIUtils";
        repo = "CLI11";
        rev = "v2.4.2";
        hash = "sha256-73dfpZDnKl0cADM4LTP3/eDFhwCdiHbEaGRF7ZyWsdQ=";
      };
      gemmi = pkgs.gemmi.overrideAttrs (old: rec {
        version = "0.6.5";
        src = pkgs.fetchFromGitHub {
          owner = "project-gemmi";
          repo = "gemmi";
          tag = "v${version}";
          hash = "sha256-JJ6YBsdL3J+d0ihuJ2Nowp40c7FkDdfTqBhDrxWgSFw=";
        };

      });
      dftd4 = pkgs.callPackage ./3rdparty/nix/dftd4.nix { };
      scnlib = pkgs.callPackage ./3rdparty/nix/scnlib.nix {
        fast-float = fast-float-6_1_6;
      };
      #      libcint = pkgs.callPackage ./3rdparty/nix/libcint.nix { };
      libcint_src = pkgs.fetchFromGitHub {
        owner = "peterspackman";
        repo = "libcint";
        rev = "master";
        hash = "sha256-JWk1B+Fz5nHxnGI5WlSynNqvkqIRvkbba8Nx3I5Tziw=";
      };
      lbfgspp-src = pkgs.fetchFromGitHub {
        owner = "yixuan";
        repo = "LBFGSpp";
        rev = "master";
        hash = "sha256-PUzZ2jUVgHq1LJDWIWW93KnV7vBBEdZlspOrE5TcYBc=";
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
            scnlib
          ];
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
          ];
          cmakeFlags = [
            "-GNinja"
            "-DCPM_DOWNLOAD_LOCATION=${pkgs.cpm-cmake}/share/cpm/CPM.cmake"
            "-DCPM_USE_LOCAL_PACKAGES=ON"
            "-DCPM_spdlog_SOURCE=${pkgs.spdlog.src}"
            "-DCPM_oneTBB_SOURCE=${pkgs.onetbb.src}"
            "-DCPM_scnlib_SOURCE=${scnlib.src}"
            "-DCPM_eigen3_SOURCE=${pkgs.eigen_3_4_0.src}"
            "-DCPM_gemmi_SOURCE=${gemmi.src}"
            "-DCPM_CLI11_SOURCE=${cli11-src}"
            "-DCPM_fmt_SOURCE=${pkgs.fmt.src}"
            "-DCPM_nlohmann_json_SOURCE=${pkgs.nlohmann_json.src}"
            "-DCPM_Libxc_SOURCE=${pkgs.libxc.src}"
            "-DCPM_libcint_SOURCE=${libcint_src}"
            "-DCPM_unordered_dense_SOURCE=${pkgs.unordered_dense.src}"
            "-DCPM_tomlplusplus_SOURCE=${pkgs.tomlplusplus.src}"
            "-DCPM_dftd4_cpp_SOURCE=${dftd4.src}"
            "-DCPM_LBFGSpp_SOURCE=${lbfgspp-src}"
            "-DFETCHCONTENT_SOURCE_DIR_FAST_FLOAT=${fast-float-6_1_6.src}"
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
          gdb
        ];

        # Environment variables for the shell
        shellHook = ''
          exec zsh
        '';
      };
    };
}
