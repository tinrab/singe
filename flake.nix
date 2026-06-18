{
  description = "Singe development environment.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        hostPkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };

        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
          overlays = [ cudaOverlay ];
        };

        cudaOverlay = final: prev: {
          cudaPackages_13_2 = prev.cudaPackages_13_2.overrideScope (
            cfinal: cprev: {
              #   cudnn = final.stdenv.mkDerivation rec {
              #     pname = "cudnn";
              #     version = "9.21.1.3";

              #     src = final.fetchurl {
              #       url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${version}_cuda13-archive.tar.xz";
              #       sha256 = "IfF5g3M+uiGsCGFTIEQgWpTfkrQoP5uv3O4fThnAmB0=";
              #     };

              #     nativeBuildInputs = [ final.autoPatchelfHook ];

              #     buildInputs = [
              #       final.zlib
              #     ]
              #     ++ (with cfinal; [
              #       cuda_cudart
              #       libcublas
              #     ]);

              #     sourceRoot = "cudnn-linux-x86_64-${version}_cuda13-archive";

              #     installPhase = ''
              #       runHook preInstall

              #       mkdir -p $out/lib $out/include

              #       cp -r lib/*     $out/lib/
              #       cp -r include/* $out/include/
              #       cp -r LICENSE   $out/ 2>/dev/null || true

              #       # cuDNN 9's top-level dispatcher libraries load sibling cuDNN
              #       # components at runtime, so they need an RPATH back to $out/lib
              #       # even when LD_LIBRARY_PATH is not populated.
              #       for library in $out/lib/*.so $out/lib/*.so.*; do
              #         [ -e "$library" ] || continue
              #         current_rpath="$(patchelf --print-rpath "$library" || true)"
              #         if [ -n "$current_rpath" ]; then
              #           patchelf --set-rpath "$out/lib:$current_rpath" "$library"
              #         else
              #           patchelf --set-rpath "$out/lib" "$library"
              #         fi
              #       done

              #       runHook postInstall
              #     '';

              #     meta = {
              #       description = "NVIDIA cuDNN (tarball override)";
              #       license = final.lib.licenses.unfree;
              #       platforms = final.lib.platforms.linux;
              #     };
              #   };

              libcutensor = final.stdenv.mkDerivation rec {
                pname = "libcutensor";
                version = "2.6.0.4";

                src = final.fetchurl {
                  url = "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-${version}_cuda13-archive.tar.xz";
                  sha256 = "9wADpIF9o9svH3cmMowxMyraiaSWRG/jC5D0FOzNpIc=";
                };

                nativeBuildInputs = [ final.autoPatchelfHook ];

                buildInputs = with cfinal; [
                  cuda_cudart
                  libcublas
                  nccl
                ];

                sourceRoot = "libcutensor-linux-x86_64-${version}_cuda13-archive";

                installPhase = ''
                  runHook preInstall

                  mkdir -p $out/lib $out/include

                  cp -r lib/*     $out/lib/
                  cp -r include/* $out/include/
                  cp -r LICENSE   $out/ 2>/dev/null || true

                  runHook postInstall
                '';

                meta = {
                  description = "NVIDIA libcutensor (tarball override)";
                  license = final.lib.licenses.unfree;
                  platforms = final.lib.platforms.linux;
                };
              };

              libcudss = final.stdenv.mkDerivation rec {
                pname = "libcudss";
                version = "0.8.0.10";

                src = final.fetchurl {
                  url = "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-${version}_cuda13-archive.tar.xz";
                  sha256 = "uhj1/YDcu+kF0VjKrFswYdhIRCu1q9R3tfKWtCV6STc=";
                };

                nativeBuildInputs = [ final.autoPatchelfHook ];

                buildInputs =
                  with cfinal;
                  [
                    cuda_cudart
                    libcublas
                    nccl
                  ]
                  ++ [
                    hostPkgs.openmpi
                  ];

                sourceRoot = "libcudss-linux-x86_64-${version}_cuda13-archive";

                installPhase = ''
                  runHook preInstall

                  mkdir -p $out/lib $out/include

                  cp -r lib/*     $out/lib/
                  cp -r include/* $out/include/
                  cp -r LICENSE   $out/ 2>/dev/null || true

                  runHook postInstall
                '';

                meta = {
                  description = "NVIDIA libcudss (tarball override)";
                  license = final.lib.licenses.unfree;
                  platforms = final.lib.platforms.linux;
                };
              };
            }
          );
        };

        cudaPkgs = pkgs.cudaPackages_13_2;
        llvmPkgs = pkgs.llvmPackages_20;

        cudaLibs =
          with cudaPkgs;
          [
            libcublas
            libcusolver
            libcusparse
            libcudss
            cuda_cudart
            cuda_cupti
            cuda_nvrtc
            cuda_nvtx
            cudnn
            cuda_nvml_dev
            libcufft
            libcutensor
            nccl
            libnvvm
            libnpp
            libcurand
            libcufile
          ]
          ++ [
            hostPkgs.openmpi
          ];

        driverLibraryPath = "/run/opengl-driver/lib";
        runtimeLibraryPath = pkgs.lib.makeLibraryPath cudaLibs;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "cuda-13-dev";

          packages = [
            # Compiler pieces
            cudaPkgs.cuda_nvcc

            # Libraries
            cudaPkgs.libcublas
            cudaPkgs.libcusolver
            cudaPkgs.libcusparse
            cudaPkgs.libcudss
            cudaPkgs.cuda_cudart
            cudaPkgs.cuda_cupti
            cudaPkgs.cuda_nvrtc
            cudaPkgs.cuda_nvtx
            cudaPkgs.cudnn
            cudaPkgs.cuda_nvml_dev
            cudaPkgs.libcufft
            cudaPkgs.libcutensor
            cudaPkgs.nccl
            cudaPkgs.libnvvm
            cudaPkgs.libnpp
            cudaPkgs.libcurand
            cudaPkgs.libcufile

            hostPkgs.openmpi

            # Tools
            pkgs.gcc15
            llvmPkgs.clang
            pkgs.cmake
            pkgs.pkg-config
          ];

          shellHook = ''
            export CUDA_PATH=${cudaPkgs.cudatoolkit}
            export CUDA_CCBIN="${pkgs.gcc15}/bin/gcc"

            export CUDA_DRIVER_LIBRARY_PATH="${driverLibraryPath}"
            export CUDA_RUNTIME_LIBRARY_PATH="${runtimeLibraryPath}"
            export LD_LIBRARY_PATH="${driverLibraryPath}:${runtimeLibraryPath}:''${LD_LIBRARY_PATH:-}"
            export NIX_LD_LIBRARY_PATH="${driverLibraryPath}:${runtimeLibraryPath}:''${NIX_LD_LIBRARY_PATH:-}"

            export LIBCLANG_PATH="${llvmPkgs.libclang.lib}/lib"

            # export RUSTFLAGS="-C link-arg=-fuse-ld=bfd"

            export CUPTI_ROOT="${cudaPkgs.cuda_cupti}"
            export CUPTI_PATH="${cudaPkgs.cuda_cupti.lib}/lib"
            export CUPTI_INCLUDE_PATH="${cudaPkgs.cuda_cupti.include}/include"

            export NVRTC_ROOT="${cudaPkgs.cuda_nvrtc}"
            export NVRTC_PATH="${cudaPkgs.cuda_nvrtc.lib}/lib"
            export NVRTC_INCLUDE_PATH="${cudaPkgs.cuda_nvrtc.include}/include"

            export NVTX_ROOT="${cudaPkgs.cuda_nvtx}"
            export NVTX_PATH="${cudaPkgs.cuda_nvtx.lib}/lib"
            export NVTX_INCLUDE_PATH="${cudaPkgs.cuda_nvtx.include}/include"

            export NVML_ROOT="${cudaPkgs.cuda_nvml_dev}"
            export NVML_PATH="${cudaPkgs.cuda_nvml_dev.stubs}/lib"
            export NVML_INCLUDE_PATH="${cudaPkgs.cuda_nvml_dev.include}/include"

            export CUDNN_ROOT="${cudaPkgs.cudnn}"
            export CUDNN_PATH="${cudaPkgs.cudnn.lib}/lib"
            export CUDNN_INCLUDE_PATH="${cudaPkgs.cudnn.include}/include"

            export CUFFT_ROOT="${cudaPkgs.libcufft}"
            export CUFFT_PATH="${cudaPkgs.libcufft.lib}/lib"
            export CUFFT_INCLUDE_PATH="${cudaPkgs.libcufft.include}/include"

            export CUTENSOR_ROOT="${cudaPkgs.libcutensor}"
            export CUTENSOR_PATH="${cudaPkgs.libcutensor}/lib"
            export CUTENSOR_INCLUDE_PATH="${cudaPkgs.libcutensor}/include"

            export CUSOLVER_ROOT="${cudaPkgs.libcusolver}"
            export CUSOLVER_PATH="${cudaPkgs.libcusolver.lib}/lib"
            export CUSOLVER_INCLUDE_PATH="${cudaPkgs.libcusolver.include}/include"

            export CUSPARSE_ROOT="${cudaPkgs.libcusparse}"
            export CUSPARSE_PATH="${cudaPkgs.libcusparse.lib}/lib"
            export CUSPARSE_INCLUDE_PATH="${cudaPkgs.libcusparse.include}/include"

            export CUDSS_ROOT="${cudaPkgs.libcudss}"
            export CUDSS_PATH="${cudaPkgs.libcudss}/lib"
            export CUDSS_INCLUDE_PATH="${cudaPkgs.libcudss}/include"

            export NCCL_ROOT="${cudaPkgs.nccl}"
            export NCCL_PATH="${cudaPkgs.nccl}/lib"
            export NCCL_INCLUDE_PATH="${cudaPkgs.nccl.dev}/include"

            export NVVM_ROOT="${cudaPkgs.libnvvm}"
            export NVVM_PATH="${cudaPkgs.libnvvm}/lib"
            export NVVM_INCLUDE_PATH="${cudaPkgs.libnvvm}/include"

            export NPP_ROOT="${cudaPkgs.libnpp}"
            export NPP_PATH="${cudaPkgs.libnpp.lib}/lib"
            export NPP_INCLUDE_PATH="${cudaPkgs.libnpp.include}/include"

            export CURAND_ROOT="${cudaPkgs.libcurand}"
            export CURAND_PATH="${cudaPkgs.libcurand.lib}/lib"
            export CURAND_INCLUDE_PATH="${cudaPkgs.libcurand.include}/include"

            export CUFILE_ROOT="${cudaPkgs.libcufile}"
            export CUFILE_PATH="${cudaPkgs.libcufile.lib}/lib"
            export CUFILE_INCLUDE_PATH="${cudaPkgs.libcufile.include}/include"
          '';
        };

        apps = {
          format = {
            type = "app";
            program = "${pkgs.writeShellScript "format-nix" ''
              find . -name "*.nix" -type f -exec ${pkgs.nixfmt}/bin/nixfmt {} +
            ''}";
          };
        };
      }
    );
}
