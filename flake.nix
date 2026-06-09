{
  description = "C/C++ development environment for x86_64-linux";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };
      rustToolchain = pkgs.rust-bin.stable."1.88.0".default;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          gcc
          lld
          cmake
          gnumake
          pkg-config
          python3
          python3Packages.pytest
          jupyter
          doxygen
        ];
        buildInputs = with pkgs; [
          boost
          catch2
          rustToolchain
          llvmPackages.openmp
          stdenv.cc.cc.lib # provides libstdc++.so.6
        ];
        shellHook = ''
          export LD_LIBRARY_PATH="${
            pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.llvmPackages.openmp
            ]
          }:$LD_LIBRARY_PATH"

          export CMAKE_MODULE_PATH="$PWD/.venv/share/Pytest/cmake:$CMAKE_MODULE_PATH"
          export CMAKE_PREFIX_PATH="$PWD/.venv"
          ln -sfn ${pkgs.micromamba}/bin/micromamba micromamba
        '';
      };
    };
}
