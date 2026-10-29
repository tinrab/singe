#[cfg(target_os = "windows")]
pub(crate) fn cuda_runtime_lib_patterns() -> Vec<&'static str> {
    vec!["cudart*.lib", "cudart*.dll"]
}

#[cfg(target_os = "linux")]
pub fn cuda_runtime_lib_patterns() -> Vec<&'static str> {
    vec!["libcudart.so", "libcudart.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn nvrtc_lib_patterns() -> Vec<&'static str> {
    vec!["nvrtc*.lib", "nvrtc*.dll"]
}

#[cfg(target_os = "linux")]
pub fn nvrtc_lib_patterns() -> Vec<&'static str> {
    vec!["libnvrtc.so", "libnvrtc.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn nvvm_lib_patterns() -> Vec<&'static str> {
    vec!["nvvm*.lib", "nvvm*.dll"]
}

#[cfg(target_os = "linux")]
pub fn nvvm_lib_patterns() -> Vec<&'static str> {
    vec!["libnvvm.so", "libnvvm.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn nvml_lib_patterns() -> Vec<&'static str> {
    vec!["nvml.lib", "nvml.dll", "nvidia-ml.lib", "nvidia-ml.dll"]
}

#[cfg(target_os = "linux")]
pub fn nvml_lib_patterns() -> Vec<&'static str> {
    vec!["libnvidia-ml.so", "libnvidia-ml.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn cuda_driver_lib_patterns() -> Vec<&'static str> {
    vec!["nvcuda.dll"] // The core driver library
}

#[cfg(target_os = "linux")]
pub fn cuda_driver_lib_patterns() -> Vec<&'static str> {
    vec!["libcuda.so", "libcuda.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn cublas_lib_patterns() -> Vec<&'static str> {
    vec!["cublas*.lib", "cublas*.dll"]
}

#[cfg(target_os = "linux")]
pub fn cublas_lib_patterns() -> Vec<&'static str> {
    vec!["libcublas.so", "libcublas.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn curand_lib_patterns() -> Vec<&'static str> {
    vec!["curand*.lib", "curand*.dll"]
}

#[cfg(target_os = "linux")]
pub fn curand_lib_patterns() -> Vec<&'static str> {
    vec!["libcurand.so", "libcurand.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn cusolver_lib_patterns() -> Vec<&'static str> {
    vec!["cusolver*.lib", "cusolver*.dll"]
}

#[cfg(target_os = "linux")]
pub fn cusolver_lib_patterns() -> Vec<&'static str> {
    vec!["libcusolver.so", "libcusolver.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn cudss_lib_patterns() -> Vec<&'static str> {
    vec!["cudss*.lib", "cudss*.dll"]
}

#[cfg(target_os = "linux")]
pub fn cudss_lib_patterns() -> Vec<&'static str> {
    vec!["libcudss.so", "libcudss.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn nvcomp_lib_patterns() -> Vec<&'static str> {
    vec!["nvcomp*.lib", "nvcomp*.dll"]
}

#[cfg(target_os = "linux")]
pub fn nvcomp_lib_patterns() -> Vec<&'static str> {
    vec!["libnvcomp.so", "libnvcomp.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn nvjpeg_lib_patterns() -> Vec<&'static str> {
    vec!["nvjpeg*.lib", "nvjpeg*.dll"]
}

#[cfg(target_os = "linux")]
pub fn nvjpeg_lib_patterns() -> Vec<&'static str> {
    vec!["libnvjpeg.so", "libnvjpeg.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn nvjpeg2k_lib_patterns() -> Vec<&'static str> {
    vec!["nvjpeg2k*.lib", "nvjpeg2k*.dll"]
}

#[cfg(target_os = "linux")]
pub fn nvjpeg2k_lib_patterns() -> Vec<&'static str> {
    vec!["libnvjpeg2k.so", "libnvjpeg2k.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn cusparse_lib_patterns() -> Vec<&'static str> {
    vec!["cusparse*.lib", "cusparse*.dll"]
}

#[cfg(target_os = "linux")]
pub fn cusparse_lib_patterns() -> Vec<&'static str> {
    vec!["libcusparse.so", "libcusparse.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn cutensor_lib_patterns() -> Vec<&'static str> {
    vec!["cutensor*.lib", "cutensor*.dll"]
}

#[cfg(target_os = "linux")]
pub fn cutensor_lib_patterns() -> Vec<&'static str> {
    vec!["libcutensor.so", "libcutensor.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn cufile_lib_patterns() -> Vec<&'static str> {
    vec!["cufile*.lib", "cufile*.dll"]
}

#[cfg(target_os = "linux")]
pub fn cufile_lib_patterns() -> Vec<&'static str> {
    vec!["libcufile.so", "libcufile.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn cudnn_lib_patterns() -> Vec<&'static str> {
    vec!["cudnn*.lib", "cudnn*.dll"]
}

#[cfg(target_os = "linux")]
pub fn cudnn_lib_patterns() -> Vec<&'static str> {
    vec!["libcudnn.so", "libcudnn.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn cupti_lib_patterns() -> Vec<&'static str> {
    vec!["cupti*.lib", "cupti*.dll"]
}

#[cfg(target_os = "linux")]
pub fn cupti_lib_patterns() -> Vec<&'static str> {
    vec!["libcupti.so", "libcupti.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn nccl_lib_patterns() -> Vec<&'static str> {
    vec!["nccl*.lib", "nccl*.dll"]
}

#[cfg(target_os = "linux")]
pub fn nccl_lib_patterns() -> Vec<&'static str> {
    vec!["libnccl.so", "libnccl.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn npp_lib_patterns() -> Vec<&'static str> {
    vec!["npp*.lib", "npp*.dll"]
}

#[cfg(target_os = "linux")]
pub fn npp_lib_patterns() -> Vec<&'static str> {
    vec!["libnppc.so", "libnppc.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn nvimagecodec_lib_patterns() -> Vec<&'static str> {
    vec!["nvimgcodec*.lib", "nvimgcodec*.dll"]
}

#[cfg(target_os = "linux")]
pub fn nvimagecodec_lib_patterns() -> Vec<&'static str> {
    vec!["libnvimgcodec.so", "libnvimgcodec.so.*"]
}

#[cfg(target_os = "windows")]
pub(crate) fn nvtiff_lib_patterns() -> Vec<&'static str> {
    vec!["nvtiff*.lib", "nvtiff*.dll"]
}

#[cfg(target_os = "linux")]
pub fn nvtiff_lib_patterns() -> Vec<&'static str> {
    vec!["libnvtiff.so", "libnvtiff.so.*"]
}
