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
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };

        lib = pkgs.lib;
        llvmPkgs = pkgs.llvmPackages_20;

        cudaRedistBase = "https://developer.download.nvidia.com/compute/cuda/redist";
        cudnnRedistBase = "https://developer.download.nvidia.com/compute/cudnn/redist";
        cudssRedistBase = "https://developer.download.nvidia.com/compute/cudss/redist";
        cutensorRedistBase = "https://developer.download.nvidia.com/compute/cutensor/redist";
        ncclGencodeString = lib.concatStringsSep " " [
          "-gencode=arch=compute_120,code=sm_120"
          "-gencode=arch=compute_120,code=compute_120"
        ];

        fetchRedist =
          {
            base,
            path,
            sha256,
          }:
          pkgs.fetchurl {
            url = "${base}/${path}";
            inherit sha256;
          };

        unpackRedist =
          {
            name,
            src,
          }:
          pkgs.runCommand name
            {
              inherit src;
              nativeBuildInputs = [
                pkgs.autoPatchelfHook
                pkgs.patchelf
              ];
              buildInputs = [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
              ];
              autoPatchelfIgnoreMissingDeps = [
                "libcuda.so.1"
                "libnvidia-ml.so.1"
                "libnvidia-ptxjitcompiler.so.1"
              ];
            }
            ''
                mkdir -p "$out"
                tar xf "$src" --strip-components=1 -C "$out"

              if [ -d "$out/lib" ] && [ ! -e "$out/lib64" ]; then
                ln -s lib "$out/lib64"
              fi

              if [ -d "$out/nvvm/lib64" ] && [ ! -e "$out/lib" ]; then
                ln -s nvvm/lib64 "$out/lib"
              fi

              if [ -d "$out/nvvm/include" ] && [ ! -e "$out/include" ]; then
                ln -s nvvm/include "$out/include"
              fi

              while IFS= read -r file; do
                if patchelf --print-interpreter "$file" >/dev/null 2>&1; then
                  patchelf --set-interpreter "${pkgs.glibc}/lib/ld-linux-x86-64.so.2" "$file"
                fi
              done < <(find "$out" -type f -perm -0100)
            '';

        cudaRedist =
          {
            name,
            version,
            sha256,
          }:
          unpackRedist {
            name = "${name}-${version}";
            src = fetchRedist {
              base = cudaRedistBase;
              path = "${name}/linux-x86_64/${name}-linux-x86_64-${version}-archive.tar.xz";
              inherit sha256;
            };
          };

        cudaVariantRedist =
          {
            base,
            name,
            version,
            cudaVersion,
            sha256,
          }:
          unpackRedist {
            name = "${name}-${version}-cuda${cudaVersion}";
            src = fetchRedist {
              inherit base sha256;
              path = "${name}/linux-x86_64/${name}-linux-x86_64-${version}_cuda${cudaVersion}-archive.tar.xz";
            };
          };

        cudaPackages = rec {
          cccl = cudaRedist {
            name = "cccl";
            version = "13.3.4.3.1";
            sha256 = "6b7516074f42f80dd3feb9c11318c8e1c5ab56cdd9f18fff4c5ace2e7e0a8209";
          };

          cuda_crt = cudaRedist {
            name = "cuda_crt";
            version = "13.4.92";
            sha256 = "c969e61ded12dbf0cf20ace72867f1622ec0e8ea1e1d305e27761012025f7aaa";
          };

          cuda_cudart = cudaRedist {
            name = "cuda_cudart";
            version = "13.4.92";
            sha256 = "0ac5dbc538d04e9983bc493b410cce4b459e1ee9f5f6654b6464ef7b3e14a8b5";
          };

          cuda_culibos = cudaRedist {
            name = "cuda_culibos";
            version = "13.4.92";
            sha256 = "241c618719e0613ce73b781c4e84b5eccf04e5baa3bae57de07980edb68a0e55";
          };

          cuda_cupti = cudaRedist {
            name = "cuda_cupti";
            version = "13.4.92";
            sha256 = "d182ce791affb8cc33e68c773d05f74b1719548d7f24e12b73525cae75d9d51e";
          };

          cuda_nvcc = cudaRedist {
            name = "cuda_nvcc";
            version = "13.4.92";
            sha256 = "60998f40cc9df5826b2a846e2dcc328133fbc5ef539398ce0a21cf5dcac7193d";
          };

          cuda_nvdisasm = cudaRedist {
            name = "cuda_nvdisasm";
            version = "13.4.92";
            sha256 = "d0af2b49c3ba6b4f64f3e284e0229d234aa88beeba8e033a8a4fe5db7c4cc3b6";
          };

          cuda_nvml_dev = cudaRedist {
            name = "cuda_nvml_dev";
            version = "13.4.92";
            sha256 = "3ef4df4c820f56a659883b0a0d5c7d2e2086f87e981a455f16498fedcd3bb9ca";
          };

          cuda_nvrtc = cudaRedist {
            name = "cuda_nvrtc";
            version = "13.4.92";
            sha256 = "defc747cfa5953b86f4c67414fc349d21149e9d513c096cb89c64b8ec4c0c932";
          };

          cuda_nvtx = cudaRedist {
            name = "cuda_nvtx";
            version = "13.4.92";
            sha256 = "d767ac8b8bb40f5d58602ce900e8bf2fe073c1ebb4a7d405ff67674009fa83ce";
          };

          cuda_profiler_api = cudaRedist {
            name = "cuda_profiler_api";
            version = "13.4.92";
            sha256 = "9d405e9b0fd9a5c29e680d6f6750b9ca79b7df91e253d4dffea3495057a75f08";
          };

          cuda_tileiras = cudaRedist {
            name = "cuda_tileiras";
            version = "13.4.92";
            sha256 = "4b5f41dc0c99f2cb4dd20090c5a20a7756a6cfbf07c8dd21e960edec2e6ee3be";
          };

          libcublas = cudaRedist {
            name = "libcublas";
            version = "13.8.0.4";
            sha256 = "bd6ffcce561672da6abb928214a6694f392741e7d1e2d7dfde0da4bd651706b6";
          };

          libcufft = cudaRedist {
            name = "libcufft";
            version = "12.4.0.43";
            sha256 = "f7f8ac2cc67714989a3c8140cda39b1b7ed7d462a491e1ad3f53adb777f5070f";
          };

          libcufile = cudaRedist {
            name = "libcufile";
            version = "1.19.1.55";
            sha256 = "8a79272aa67dd952107384a7d0d6f6466c319419ddca156bcfe15d186a23b293";
          };

          libcurand = cudaRedist {
            name = "libcurand";
            version = "10.4.4.72";
            sha256 = "6f725f9187dc5f675308b830f786c163f5b1b499d86986f94fb60aa701c13c0b";
          };

          libcusolver = cudaRedist {
            name = "libcusolver";
            version = "12.3.4.7";
            sha256 = "dd89888e75628ada3fece05eb642d8e492667fdf3714c4659104c9636863f6ec";
          };

          libcusparse = cudaRedist {
            name = "libcusparse";
            version = "12.8.6.72";
            sha256 = "1c1fe7f04d74878e1cabc6d0161882caac766c2872e469743a1993366572640b";
          };

          libnpp = cudaRedist {
            name = "libnpp";
            version = "13.2.0.58";
            sha256 = "6e219a5472d28d7e038e88c2a5d1ab18f936f3a13cc8dd4c62f08c0e6233780a";
          };

          libnvvm = cudaRedist {
            name = "libnvvm";
            version = "13.4.92";
            sha256 = "0e619cf3d5b27f7c812665599ca9761fc296e56129b24a85d822a5d8a1a3bd75";
          };

          libnvjitlink = cudaRedist {
            name = "libnvjitlink";
            version = "13.4.92";
            sha256 = "e24557473e5e2e1dc591209558523deaca48b32123fe2b822f994b8e691c8ae3";
          };

          cudnn = cudaVariantRedist {
            base = cudnnRedistBase;
            name = "cudnn";
            version = "9.27.0.42";
            cudaVersion = "13";
            sha256 = "dd3245ba3aca8593934fc946d08649ad3a5ee8500e8da3870bc23bf7ddccd7f9";
          };

          libcudss = cudaVariantRedist {
            base = cudssRedistBase;
            name = "libcudss";
            version = "0.8.0.10";
            cudaVersion = "13";
            sha256 = "ba18f5fd80dcbbe905d158caac5b3061d848442bb5abd477b5f296b4257a4937";
          };

          libcutensor = cudaVariantRedist {
            base = cutensorRedistBase;
            name = "libcutensor";
            version = "2.8.1.0";
            cudaVersion = "13";
            sha256 = "03051950d98911ce21d977fd7ef697b6ff963b885df8844a326e25b9d400ae7a";
          };

          # CUDA 13.4 supports GCC through version 15.
          nccl = pkgs.gcc15Stdenv.mkDerivation rec {
            pname = "nccl";
            version = "2.32.3-1";

            src = pkgs.fetchFromGitHub {
              owner = "NVIDIA";
              repo = "nccl";
              tag = "v${version}";
              hash = "sha256-xUllfdWAL0Ee9P9T9CZC2ddkPRnSXZXgwApgO398i6g=";
            };

            nativeBuildInputs = [
              pkgs.python3
              pkgs.which
              cuda_nvcc
            ];

            buildInputs = [
              cccl
              cuda_cudart
              pkgs.openssl
            ];

            env = lib.optionalAttrs (ncclGencodeString != "") {
              NVCC_GENCODE = ncclGencodeString;
            };

            postPatch = ''
              patchShebangs ./src/misc/generate_git_version.py
              patchShebangs ./src/device/generate.py
              patchShebangs ./src/device/symmetric/generate.py

              substituteInPlace ./makefiles/common.mk \
                --replace-fail '-ccbin $(CXX)' ""
            '';

            makeFlags = [
              "CXXSTD=-std=c++17"
              "TLS_BACKEND=OPENSSL3"
              "CUDA_HOME=${cudaCompilerToolkit}"
              "CUDA_INC=${cudaCompilerToolkit}/include"
              "CUDA_LIB=${cudaCompilerToolkit}/lib"
              "PREFIX=$(out)"
            ];

            enableParallelBuilding = true;
          };

          cudaCompilerToolkit = pkgs.symlinkJoin {
            name = "cuda-toolkit-13.4.2";
            paths = [
              cccl
              cuda_crt
              cuda_cudart
              cuda_culibos
              cuda_nvcc
              cuda_nvdisasm
              cuda_nvrtc
              cuda_nvtx
              cuda_profiler_api
              cuda_tileiras
              libnvjitlink
              libnvvm
            ];
            postBuild = ''
              if [ -d "$out/lib" ] && [ ! -e "$out/lib64" ]; then
                ln -s lib "$out/lib64"
              fi
            '';
          };

          cudatoolkit = pkgs.symlinkJoin {
            name = "cuda-toolkit-13.4.2";
            paths = [
              cudaCompilerToolkit
              libcublas
            ];
            postBuild = ''
              if [ -d "$out/lib" ] && [ ! -e "$out/lib64" ]; then
                ln -s lib "$out/lib64"
              fi
            '';
          };
        };

        pathOf = package: "${package}/lib";
        includeOf = package: "${package}/include";

        cudaLibs = with cudaPackages; [
          cuda_cudart
          cuda_cupti
          cuda_nvrtc
          cuda_nvtx
          cuda_nvml_dev
          cudnn
          libcublas
          libcudss
          libcufft
          libcufile
          libcurand
          libcusolver
          libcusparse
          libcutensor
          libnpp
          libnvjitlink
          libnvvm
          nccl
          pkgs.openssl
          pkgs.zlib
          pkgs.openmpi
          pkgs.stdenv.cc.cc.lib
        ];

        driverLibraryPath = "/run/opengl-driver/lib";
        runtimeLibraryPath = "${lib.makeLibraryPath cudaLibs}";
        frameworkTools = with pkgs; [
          cargo
          rustc
          rustfmt
          clippy
          just
          uv
          git
          protobuf
          python314
          cacert
          actionlint
          ripgrep
          util-linux
          nixfmt
        ];
      in
      {
        packages = {
          cudaToolkit = cudaPackages.cudatoolkit;
        };

        devShells.default = pkgs.mkShell {
          name = "cuda-13.4.2-dev";
          hardeningDisable = [ "fortify" ];

          packages = [
            cudaPackages.cudatoolkit

            pkgs.gcc15
            llvmPkgs.clang
            pkgs.rust-bindgen
            pkgs.cmake
            pkgs.pkg-config
          ]
          ++ cudaLibs;

          shellHook = ''
            export CUDA_PATH="${cudaPackages.cudatoolkit}"
            export CUDA_HOME="${cudaPackages.cudatoolkit}"
            export CUDA_TOOLKIT_PATH="${cudaPackages.cudatoolkit}"
            export CUDA_CCBIN="${pkgs.gcc15}/bin/gcc"

            export PATH="${cudaPackages.cudatoolkit}/bin:$PATH"

            export CUDA_DRIVER_LIBRARY_PATH="${driverLibraryPath}"
            export CUDA_RUNTIME_LIBRARY_PATH="${runtimeLibraryPath}"
            export LD_LIBRARY_PATH="${driverLibraryPath}:${runtimeLibraryPath}:''${LD_LIBRARY_PATH:-}"
            export NIX_LD_LIBRARY_PATH="${driverLibraryPath}:${runtimeLibraryPath}:''${NIX_LD_LIBRARY_PATH:-}"

            export LIBCLANG_PATH="${llvmPkgs.libclang.lib}/lib"

            # export RUSTFLAGS="-C link-arg=-fuse-ld=bfd"

            export CUPTI_ROOT="${cudaPackages.cuda_cupti}"
            export CUPTI_PATH="${pathOf cudaPackages.cuda_cupti}"
            export CUPTI_INCLUDE_PATH="${includeOf cudaPackages.cuda_cupti}"

            export NVRTC_ROOT="${cudaPackages.cuda_nvrtc}"
            export NVRTC_PATH="${pathOf cudaPackages.cuda_nvrtc}"
            export NVRTC_INCLUDE_PATH="${includeOf cudaPackages.cuda_nvrtc}"

            export NVTX_ROOT="${cudaPackages.cuda_nvtx}"
            export NVTX_PATH="${pathOf cudaPackages.cuda_nvtx}"
            export NVTX_INCLUDE_PATH="${includeOf cudaPackages.cuda_nvtx}"

            export NVML_ROOT="${cudaPackages.cuda_nvml_dev}"
            export NVML_PATH="${pathOf cudaPackages.cuda_nvml_dev}/stubs"
            export NVML_INCLUDE_PATH="${includeOf cudaPackages.cuda_nvml_dev}"

            export CUDNN_ROOT="${cudaPackages.cudnn}"
            export CUDNN_PATH="${pathOf cudaPackages.cudnn}"
            export CUDNN_INCLUDE_PATH="${includeOf cudaPackages.cudnn}"

            export CUFFT_ROOT="${cudaPackages.libcufft}"
            export CUFFT_PATH="${pathOf cudaPackages.libcufft}"
            export CUFFT_INCLUDE_PATH="${includeOf cudaPackages.libcufft}"

            export CUTENSOR_ROOT="${cudaPackages.libcutensor}"
            export CUTENSOR_PATH="${pathOf cudaPackages.libcutensor}"
            export CUTENSOR_INCLUDE_PATH="${includeOf cudaPackages.libcutensor}"

            export CUSOLVER_ROOT="${cudaPackages.libcusolver}"
            export CUSOLVER_PATH="${pathOf cudaPackages.libcusolver}"
            export CUSOLVER_INCLUDE_PATH="${includeOf cudaPackages.libcusolver}"

            export CUSPARSE_ROOT="${cudaPackages.libcusparse}"
            export CUSPARSE_PATH="${pathOf cudaPackages.libcusparse}"
            export CUSPARSE_INCLUDE_PATH="${includeOf cudaPackages.libcusparse}"

            export CUDSS_ROOT="${cudaPackages.libcudss}"
            export CUDSS_PATH="${pathOf cudaPackages.libcudss}"
            export CUDSS_INCLUDE_PATH="${includeOf cudaPackages.libcudss}"

            export NCCL_ROOT="${cudaPackages.nccl}"
            export NCCL_PATH="${pathOf cudaPackages.nccl}"
            export NCCL_INCLUDE_PATH="${includeOf cudaPackages.nccl}"

            export NVVM_ROOT="${cudaPackages.libnvvm}"
            export NVVM_PATH="${pathOf cudaPackages.libnvvm}"
            export NVVM_INCLUDE_PATH="${includeOf cudaPackages.libnvvm}"

            export NPP_ROOT="${cudaPackages.libnpp}"
            export NPP_PATH="${pathOf cudaPackages.libnpp}"
            export NPP_INCLUDE_PATH="${includeOf cudaPackages.libnpp}"

            export CURAND_ROOT="${cudaPackages.libcurand}"
            export CURAND_PATH="${pathOf cudaPackages.libcurand}"
            export CURAND_INCLUDE_PATH="${includeOf cudaPackages.libcurand}"

            export CUFILE_ROOT="${cudaPackages.libcufile}"
            export CUFILE_PATH="${pathOf cudaPackages.libcufile}"
            export CUFILE_INCLUDE_PATH="${includeOf cudaPackages.libcufile}"
          '';
        };

        apps = {
          format = {
            type = "app";
            program = "${pkgs.writeShellScript "format-nix" ''
              find . -name "*.nix" -type f -exec ${pkgs.nixfmt}/bin/nixfmt {} +
            ''}";
            meta.description = "Format Nix files";
          };
        };
      }
    );
}
