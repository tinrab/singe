/// Toolkit root variables, checked in this order when no component-specific root matches.
pub const COMMON_CUDA_ENV_VARS: [&str; 2] = ["CUDA_PATH", "CUDA_HOME"];

/// Environment variables whose changes can affect CUDA dependency discovery or linking.
pub const CUDA_DISCOVERY_ENV_VARS: [&str; 69] = [
    "CUDA_PATH",
    "CUDA_HOME",
    "CUBLAS_PATH",
    "CUBLAS_ROOT",
    "CUDNN_INCLUDE_PATH",
    "CUDNN_PATH",
    "CUDNN_ROOT",
    "CUDSS_INCLUDE_PATH",
    "CUDSS_PATH",
    "CUDSS_ROOT",
    "CUFFT_INCLUDE_PATH",
    "CUFFT_PATH",
    "CUFFT_ROOT",
    "CURAND_PATH",
    "CURAND_ROOT",
    "CUPTI_INCLUDE_PATH",
    "CUPTI_PATH",
    "CUPTI_ROOT",
    "CUSOLVER_PATH",
    "CUSOLVER_ROOT",
    "CUSPARSE_PATH",
    "CUSPARSE_ROOT",
    "CUTENSOR_INCLUDE_PATH",
    "CUTENSOR_PATH",
    "CUTENSOR_ROOT",
    "CUFILE_INCLUDE_PATH",
    "CUFILE_PATH",
    "LD_LIBRARY_PATH",
    "NCCL_INCLUDE_PATH",
    "NCCL_PATH",
    "NCCL_ROOT",
    "NPP_INCLUDE_PATH",
    "NPP_PATH",
    "NPP_ROOT",
    "NIX_LD_LIBRARY_PATH",
    "NVCC",
    "NVML_INCLUDE_PATH",
    "NVML_PATH",
    "NVML_ROOT",
    "NVCOMP_INCLUDE_PATH",
    "NVCOMP_PATH",
    "NVCOMP_ROOT",
    "NVJPEG_INCLUDE_PATH",
    "NVJPEG_PATH",
    "NVJPEG_ROOT",
    "NVJPEG2K_INCLUDE_PATH",
    "NVJPEG2K_PATH",
    "NVJPEG2K_ROOT",
    "NVIMGCODEC_INCLUDE_PATH",
    "NVIMGCODEC_PATH",
    "NVIMGCODEC_ROOT",
    "NVIMGCODEC_EXTENSIONS_PATH",
    "NVJITLINK_INCLUDE_PATH",
    "NVJITLINK_PATH",
    "NVJITLINK_ROOT",
    "NVRTC_INCLUDE_PATH",
    "NVRTC_PATH",
    "NVRTC_ROOT",
    "NVVM_INCLUDE_PATH",
    "NVVM_PATH",
    "NVVM_ROOT",
    "PATH",
    "ProgramFiles",
    "SystemRoot",
    "CUDA_DRIVER_LIBRARY_PATH",
    "CUDA_RUNTIME_LIBRARY_PATH",
    "LIBCLANG_PATH",
    "NIX_LDFLAGS",
    "RUSTFLAGS",
];

/// Emits Cargo rebuild directives for environment variables used by CUDA discovery.
///
/// Build scripts should call this before using the discovery helpers so Cargo
/// reruns them when CUDA, NVIDIA library, compiler, or linker environment
/// variables change.
pub fn emit_rerun_if_env_changed() {
    for var_name in CUDA_DISCOVERY_ENV_VARS {
        println!("cargo:rerun-if-env-changed={var_name}");
    }
}
