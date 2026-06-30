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
        nvcompRedistBase = "https://developer.download.nvidia.com/compute/nvcomp/redist";
        nvimgcodecRedistBase = "https://developer.download.nvidia.com/compute/nvimgcodec/redist";
        nvtiffRedistBase = "https://developer.download.nvidia.com/compute/nvtiff/redist";
        nvjpeg2000RedistBase = "https://developer.download.nvidia.com/compute/nvjpeg2000/redist";
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

        cudaMajorRedist =
          {
            base,
            name,
            version,
            cudaVersion,
            sha256,
          }:
          pkgs.runCommand "${name}-${version}-cuda${cudaVersion}"
            {
              src = fetchRedist {
                inherit base sha256;
                path = "${name}/linux-x86_64/${name}-linux-x86_64-${version}-archive.tar.xz";
              };
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

              for path in bin cmake extensions include lib64; do
                if [ -e "$out/${cudaVersion}/$path" ] && [ ! -e "$out/$path" ]; then
                  ln -s "${cudaVersion}/$path" "$out/$path"
                fi
              done

              if [ -d "$out/lib64" ] && [ ! -e "$out/lib" ]; then
                ln -s lib64 "$out/lib"
              fi

              rpath="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.glibc}/lib:$out/lib:$out/extensions"

              while IFS= read -r file; do
                if patchelf --print-needed "$file" >/dev/null 2>&1; then
                  patchelf --set-rpath "$rpath" "$file"
                fi

                if patchelf --print-interpreter "$file" >/dev/null 2>&1; then
                  patchelf --set-interpreter "${pkgs.glibc}/lib/ld-linux-x86-64.so.2" "$file"
                fi
              done < <(find "$out" -type f -perm -0100)
            '';

        cudaPackages = rec {
          cccl = cudaRedist {
            name = "cccl";
            version = "13.3.3.3.1";
            sha256 = "67746da12f16229ac4ebde78ce7895e42b069d1d3e2ae2d2d25f90bc43679d68";
          };

          cuda_crt = cudaRedist {
            name = "cuda_crt";
            version = "13.3.33";
            sha256 = "4755d36d24c6ef7697a2d3e1dbb23c4562c9c0d97d48390d4cbd8ab32dec5b5f";
          };

          cuda_cudart = cudaRedist {
            name = "cuda_cudart";
            version = "13.3.29";
            sha256 = "1e59c4888267d27ba1a9bd0f3669a6439db1334a96e754cd9013c7c73e18dc9d";
          };

          cuda_culibos = cudaRedist {
            name = "cuda_culibos";
            version = "13.3.33";
            sha256 = "db36b1381bb3105f2f13591393131d93f501958ebd44e9d0d52234cbbaf8cbd0";
          };

          cuda_cupti = cudaRedist {
            name = "cuda_cupti";
            version = "13.3.35";
            sha256 = "8ec73c3063039a780a45d136329563a5c15e3a034b1b082196cd8d98add053ba";
          };

          cuda_nvcc = cudaRedist {
            name = "cuda_nvcc";
            version = "13.3.33";
            sha256 = "93b098bda4a562ebf3541523ce82adc43f106a81dcf28bcbf8f0d8e093d1c66f";
          };

          cuda_nvdisasm = cudaRedist {
            name = "cuda_nvdisasm";
            version = "13.3.29";
            sha256 = "95b7f617ea11bb983a7150827a14ff4488cba10f0171fa8ae2e845bd58823456";
          };

          cuda_nvml_dev = cudaRedist {
            name = "cuda_nvml_dev";
            version = "13.3.29";
            sha256 = "b62508923dadc0ac79fb1846eeb444296813cb077964e0c8348ba051bc0160aa";
          };

          cuda_nvrtc = cudaRedist {
            name = "cuda_nvrtc";
            version = "13.3.33";
            sha256 = "9e8f78278215babd1236b137252424ca7912c185bd093201f5d97f7dd763b74a";
          };

          cuda_nvtx = cudaRedist {
            name = "cuda_nvtx";
            version = "13.3.29";
            sha256 = "7c7c2567e35de98b5bf92bef06b97ccc90cd472ca44b9b1558b25812a54df64e";
          };

          cuda_profiler_api = cudaRedist {
            name = "cuda_profiler_api";
            version = "13.3.27";
            sha256 = "5aa4df91651f19c2c7c3ac0531cdcd96e18f5227a60efb81b18ca282be719780";
          };

          cuda_tileiras = cudaRedist {
            name = "cuda_tileiras";
            version = "13.3.36";
            sha256 = "1b055db199f806c746d53331200ccd8480bfdddd14638ed2911f30ee0cc4447b";
          };

          libcublas = cudaRedist {
            name = "libcublas";
            version = "13.5.1.27";
            sha256 = "35a898360520d6101ffcaf36c0d04496b54d4fc2afb82f7fce44218c54513808";
          };

          libcufft = cudaRedist {
            name = "libcufft";
            version = "12.3.0.29";
            sha256 = "b2404952a5d630fbbc13e12d36975ef87a9c05c71c321c2cb5b3820789e47462";
          };

          libcufile = cudaRedist {
            name = "libcufile";
            version = "1.18.0.66";
            sha256 = "be4b6376f9e407176f5bc1802c873a79ba629bc2b0b928a7b059471ef0d90081";
          };

          libcurand = cudaRedist {
            name = "libcurand";
            version = "10.4.3.29";
            sha256 = "0218e62ab413e435dcd0274ec8e63b62214e6aba8519201061d1597e73caadbb";
          };

          libcusolver = cudaRedist {
            name = "libcusolver";
            version = "12.2.2.18";
            sha256 = "100b49bc0b6372fe17c1fea3400dbb4cd12313ad326dea516fa26eb1d20c1a32";
          };

          libcusparse = cudaRedist {
            name = "libcusparse";
            version = "12.8.1.7";
            sha256 = "c258aab32bd5c3f19eb9e82be368501b8e24c21d5825f2ab86473008a26ba0c7";
          };

          libnpp = cudaRedist {
            name = "libnpp";
            version = "13.1.2.48";
            sha256 = "12fefdcae4c94a7b977b399b87b2a717ce8a11464a7ed55b8b2518c3261c08b4";
          };

          libnvjpeg = cudaRedist {
            name = "libnvjpeg";
            version = "13.2.1.68";
            sha256 = "22610703f7bc4e57d9b7b2f39ea7055c7495b603ab8fbf107b6d735a16746d4e";
          };

          libnvjpeg_2k = cudaVariantRedist {
            base = nvjpeg2000RedistBase;
            name = "libnvjpeg_2k";
            version = "0.10.0.49";
            cudaVersion = "13";
            sha256 = "31f52bb363a63089552954b055fbbe84f8ac64e4f9850c9d9a2366338fe15f12";
          };

          libnvtiff = cudaVariantRedist {
            base = nvtiffRedistBase;
            name = "libnvtiff";
            version = "0.6.0.78";
            cudaVersion = "13";
            sha256 = "8018c8dcff3bcac9f0e74827d312f67169b9fd01b40b2bbb431dd9bf8a5c38aa";
          };

          libnvvm = cudaRedist {
            name = "libnvvm";
            version = "13.3.33";
            sha256 = "fc9c1fd5844e44c0e5eeb051378c1b13cf0e3bb3fe4966d5103c38885424f802";
          };

          libnvjitlink = cudaRedist {
            name = "libnvjitlink";
            version = "13.2.78";
            sha256 = "75fb264ce48651095972ab3aaaefab0809f69245f859aabbf2113f4c5e01d400";
          };

          nvimgcodec = cudaMajorRedist {
            base = nvimgcodecRedistBase;
            name = "nvimgcodec";
            version = "0.8.0.22";
            cudaVersion = "13";
            sha256 = "4da5a5ae724e5bb35e3c567babce9367bcdae6b594c03a4d7802e399f2d81342";
          };

          cudnn = cudaVariantRedist {
            base = cudnnRedistBase;
            name = "cudnn";
            version = "9.22.0.52";
            cudaVersion = "13";
            sha256 = "6853561e3fbb545e2d25ad567876eadbfff4db49e210abd285e82b55bcb982b0";
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
            version = "2.6.0.4";
            cudaVersion = "13";
            sha256 = "f70003a4817da3db2f1f7726328c31332ada89a496446fe30b90f414eccda487";
          };

          nvcomp = cudaVariantRedist {
            base = nvcompRedistBase;
            name = "nvcomp";
            version = "5.2.0.10";
            cudaVersion = "13";
            sha256 = "2dd6c184c79fa5402c9b63a274e778d4b52e8d736ee927da81f07c1f8bed12ff";
          };

          nccl = pkgs.stdenv.mkDerivation rec {
            pname = "nccl";
            version = "2.28.7-1";

            src = pkgs.fetchFromGitHub {
              owner = "NVIDIA";
              repo = "nccl";
              tag = "v${version}";
              hash = "sha256-NM19OiBBGmv3cGoVoRLKSh9Y59hiDoei9NIrRnTqWeA=";
            };

            nativeBuildInputs = [
              pkgs.python3
              pkgs.which
              cuda_nvcc
            ];

            buildInputs = [
              cccl
              cuda_cudart
            ];

            env = lib.optionalAttrs (ncclGencodeString != "") {
              NVCC_GENCODE = ncclGencodeString;
            };

            postPatch = ''
              patchShebangs ./src/device/generate.py
              patchShebangs ./src/device/symmetric/generate.py

              substituteInPlace ./makefiles/common.mk \
                --replace-fail '-ccbin $(CXX)' ""
            '';

            makeFlags = [
              "CXXSTD=-std=c++17"
              "CUDA_HOME=${cudaCompilerToolkit}"
              "CUDA_INC=${cudaCompilerToolkit}/include"
              "CUDA_LIB=${cudaCompilerToolkit}/lib"
              "PREFIX=$(out)"
            ];

            enableParallelBuilding = true;
          };

          cudaCompilerToolkit = pkgs.symlinkJoin {
            name = "cuda-toolkit-13.3";
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
            name = "cuda-toolkit-13.3";
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
          libnvjpeg
          libnvjpeg_2k
          libnvtiff
          libnvjitlink
          libnvvm
          nvcomp
          nvimgcodec
          nccl
          pkgs.openmpi
          pkgs.stdenv.cc.cc.lib
        ];

        nvimgcodecExtensionPath = "${cudaPackages.nvimgcodec}/extensions";
        extraRuntimeLibraryPaths = [
          "${cudaPackages.libnvtiff}/lib/13"
        ];
        driverLibraryPath = "/run/opengl-driver/lib";
        runtimeLibraryPath = "${lib.makeLibraryPath cudaLibs}:${lib.concatStringsSep ":" extraRuntimeLibraryPaths}:${nvimgcodecExtensionPath}";
      in
      {
        packages = {
          cudaToolkit = cudaPackages.cudatoolkit;
          nvcomp = cudaPackages.nvcomp;
          nvimgcodec = cudaPackages.nvimgcodec;
        };

        devShells.default = pkgs.mkShell {
          name = "cuda-13.3-dev";
          hardeningDisable = [ "fortify" ];

          packages = [
            cudaPackages.cudatoolkit

            pkgs.gcc15
            llvmPkgs.clang
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

            export NVIMGCODEC_ROOT="${cudaPackages.nvimgcodec}"
            export NVIMGCODEC_PATH="${pathOf cudaPackages.nvimgcodec}"
            export NVIMGCODEC_INCLUDE_PATH="${includeOf cudaPackages.nvimgcodec}"
            export NVIMGCODEC_EXTENSIONS_PATH="${nvimgcodecExtensionPath}"

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

            export NVCOMP_ROOT="${cudaPackages.nvcomp}"
            export NVCOMP_PATH="${pathOf cudaPackages.nvcomp}"
            export NVCOMP_INCLUDE_PATH="${includeOf cudaPackages.nvcomp}"

            export NVVM_ROOT="${cudaPackages.libnvvm}"
            export NVVM_PATH="${pathOf cudaPackages.libnvvm}"
            export NVVM_INCLUDE_PATH="${includeOf cudaPackages.libnvvm}"

            export NPP_ROOT="${cudaPackages.libnpp}"
            export NPP_PATH="${pathOf cudaPackages.libnpp}"
            export NPP_INCLUDE_PATH="${includeOf cudaPackages.libnpp}"

            export NVJPEG_ROOT="${cudaPackages.libnvjpeg}"
            export NVJPEG_PATH="${pathOf cudaPackages.libnvjpeg}"
            export NVJPEG_INCLUDE_PATH="${includeOf cudaPackages.libnvjpeg}"

            export NVJPEG2K_ROOT="${cudaPackages.libnvjpeg_2k}"
            export NVJPEG2K_PATH="${pathOf cudaPackages.libnvjpeg_2k}"
            export NVJPEG2K_INCLUDE_PATH="${includeOf cudaPackages.libnvjpeg_2k}"

            export NVTIFF_ROOT="${cudaPackages.libnvtiff}"
            export NVTIFF_PATH="${pathOf cudaPackages.libnvtiff}"
            export NVTIFF_INCLUDE_PATH="${includeOf cudaPackages.libnvtiff}"

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
