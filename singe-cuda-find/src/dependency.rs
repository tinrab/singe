use std::{
    env::{self},
    path::{Path, PathBuf},
    result::Result as StdResult,
};

use glob::glob;

use crate::{
    error::{FindError, Result},
    patterns::{
        cublas_lib_patterns, cuda_driver_lib_patterns, cuda_runtime_lib_patterns,
        cudnn_lib_patterns, cudss_lib_patterns, cufile_lib_patterns, cupti_lib_patterns,
        curand_lib_patterns, cusolver_lib_patterns, cusparse_lib_patterns, cutensor_lib_patterns,
        nccl_lib_patterns, npp_lib_patterns, nvcomp_lib_patterns, nvimagecodec_lib_patterns,
        nvjpeg_lib_patterns, nvjpeg2k_lib_patterns, nvml_lib_patterns, nvrtc_lib_patterns,
        nvtiff_lib_patterns, nvvm_lib_patterns,
    },
    search::{command_in_path, driver_search_paths, find_dependency, potential_roots},
    version::{
        cuda_major_version, parse_cublas_version_from_header, parse_cuda_version_from_header,
        parse_cudnn_version_from_header, parse_cudss_version_from_header,
        parse_cufile_version_from_library_path, parse_cupti_version_from_header,
        parse_curand_version_from_header, parse_cusolver_version_from_header,
        parse_cusparse_version_from_header, parse_cutensor_version_from_header,
        parse_nccl_version_from_header, parse_npp_version_from_header,
        parse_nvcomp_version_from_header, parse_nvimagecodec_version_from_header,
        parse_nvjpeg_version_from_header, parse_nvjpeg2k_version_from_header,
        parse_nvtiff_version_from_header,
    },
};

/// Represents the located paths and version for a CUDA dependency.
#[derive(Debug, Clone)]
pub struct Dependency {
    /// The path to the directory containing the header files.
    pub include_path: PathBuf,
    /// The path to the directory containing the library files.
    pub library_path: PathBuf,
    /// The detected version string, such as `"13.0.0"`.
    pub version: Option<String>,
}

/// Emits Cargo link-search and include metadata for a discovered dependency.
///
/// Discovery errors panic with a clear message so build scripts fail at the
/// actual discovery problem instead of silently falling through to later linker
/// failures. `Ok(None)` remains non-fatal to allow system linkers to resolve
/// libraries from their normal search paths.
///
/// # Panics
///
/// Panics when `result` contains a discovery error.
#[must_use]
pub fn emit_dependency_metadata(result: Result<Option<Dependency>>) -> Option<Dependency> {
    let dependency = match result {
        Ok(dependency) => dependency?,
        Err(error) => panic!("{error}"),
    };
    println!(
        "cargo:rustc-link-search={}",
        dependency.library_path.display()
    );
    println!("cargo:include={}", dependency.include_path.display());
    Some(dependency)
}

/// Emits Cargo link-search metadata for a discovered library directory.
///
/// Discovery errors panic with a clear message. `Ok(None)` remains non-fatal to
/// allow system linkers to resolve libraries from their normal search paths.
///
/// # Panics
///
/// Panics when `result` contains a discovery error.
#[must_use]
pub fn emit_library_path_metadata(result: Result<Option<PathBuf>>) -> Option<PathBuf> {
    let path = match result {
        Ok(path) => path?,
        Err(error) => panic!("{error}"),
    };
    println!("cargo:rustc-link-search={}", path.display());
    Some(path)
}

/// Emits Cargo link-lib metadata for each library name.
pub fn emit_link_libraries(libraries: &[&str]) {
    for library in libraries {
        println!("cargo:rustc-link-lib={library}");
    }
}

/// Emits Cargo static link-lib metadata for each library name.
pub fn emit_static_link_libraries(libraries: &[&str]) {
    for library in libraries {
        println!("cargo:rustc-link-lib=static={library}");
    }
}

/// Attempts to find the CUDA Runtime headers and library.
///
/// The search checks `CUDA_PATH`/`CUDA_HOME` and common CUDA Toolkit roots for
/// `cuda_runtime_api.h` plus a matching `cudart` library.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_cuda_runtime() -> Result<Option<Dependency>> {
    find_dependency(
        "CUDA Runtime",
        "cuda_runtime_api.h",
        &["cuda.h"],
        &cuda_runtime_lib_patterns(),
        parse_cuda_version_from_header,
        &[],
        None,
    )
}

/// Attempts to find the NVRTC headers and library.
///
/// The search checks `CUDA_PATH`/`CUDA_HOME` and common CUDA Toolkit roots for
/// `nvrtc.h` plus a matching `nvrtc` library.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_nvrtc() -> Result<Option<Dependency>> {
    find_dependency(
        "NVRTC",
        "nvrtc.h",
        &["nvrtc.h"],
        &nvrtc_lib_patterns(),
        parse_cuda_version_from_header,
        &[],
        None,
    )
}

/// Attempts to find the nvJitLink headers and shared library.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_nvjitlink() -> Result<Option<Dependency>> {
    if let Some(dependency) = find_dependency_from_explicit_paths(
        "nvJitLink.h",
        &["libnvJitLink.so*", "nvJitLink*.lib"],
        parse_cuda_version_from_header,
        &["NVJITLINK_INCLUDE_PATH"],
        &["NVJITLINK_PATH"],
    )? {
        return Ok(Some(dependency));
    }
    find_dependency(
        "NVJITLINK",
        "nvJitLink.h",
        &["nvJitLink.h"],
        &["libnvJitLink.so*", "nvJitLink*.lib"],
        parse_cuda_version_from_header,
        &["NVJITLINK_PATH", "NVJITLINK_ROOT"],
        None,
    )
}

/// Attempts to find the NVVM headers and library.
///
/// The search checks `NVVM_ROOT`/`NVVM_PATH`/`NVVM_INCLUDE_PATH`,
/// `CUDA_PATH`/`CUDA_HOME`, and common CUDA Toolkit roots for `nvvm.h` plus a
/// matching `nvvm` library.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_nvvm() -> Result<Option<Dependency>> {
    if let Some(dependency) = find_dependency_from_explicit_paths(
        "nvvm.h",
        &nvvm_lib_patterns(),
        parse_cuda_version_from_header,
        &["NVVM_INCLUDE_PATH"],
        &["NVVM_PATH"],
    )? {
        return Ok(Some(dependency));
    }

    find_dependency(
        "NVVM",
        "nvvm.h",
        &["nvvm.h"],
        &nvvm_lib_patterns(),
        parse_cuda_version_from_header,
        &["NVVM_PATH", "NVVM_ROOT"],
        None,
    )
}

/// Attempts to find the NVML driver library directory.
///
/// NVML is supplied by the NVIDIA driver rather than the CUDA Toolkit, so this
/// searches driver library paths such as `PATH` on Windows and
/// `LD_LIBRARY_PATH`, `NIX_LD_LIBRARY_PATH`, and standard library directories on
/// Linux.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_nvml() -> Result<Option<PathBuf>> {
    let lib_patterns = nvml_lib_patterns();
    let search_paths = driver_search_paths();

    for dir in search_paths {
        if !dir.is_dir() {
            continue;
        }

        for pattern in &lib_patterns {
            let glob_pattern = dir.join(pattern).to_string_lossy().into_owned();
            match glob(&glob_pattern) {
                Ok(mut entries) => {
                    if entries.find_map(StdResult::ok).is_some() {
                        return Ok(Some(dir));
                    }
                }
                Err(error) => {
                    return Err(FindError::glob("NVML", glob_pattern, &error));
                }
            }
        }
    }

    Ok(None)
}

/// Attempts to find the `nvcc` compiler executable.
///
/// The search checks `NVCC`, then `PATH`, then common CUDA Toolkit roots under
/// their `bin` directories.
///
/// # Errors
///
/// This search currently returns no errors. A missing executable produces `Ok(None)`.
pub fn find_nvcc() -> Result<Option<PathBuf>> {
    if let Some(path) = env::var_os("NVCC") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(Some(path));
        }
    }

    if let Some(path) = command_in_path("nvcc") {
        return Ok(Some(path));
    }

    Ok(potential_roots(&[], "CUDA Toolkit")
        .into_iter()
        .map(|root| {
            #[cfg(target_os = "windows")]
            let candidate = root.join("bin").join("nvcc.exe");

            #[cfg(not(target_os = "windows"))]
            let candidate = root.join("bin").join("nvcc");

            candidate
        })
        .find(|path| path.is_file()))
}

/// Attempts to find the CUDA Driver library directory.
///
/// This searches system and driver locations for `libcuda.so` or `nvcuda.dll`.
/// The result is not necessarily inside the CUDA Toolkit because the driver
/// library is normally installed by the NVIDIA display/compute driver.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_cuda_driver_path() -> Result<Option<PathBuf>> {
    let lib_patterns = cuda_driver_lib_patterns();
    let search_paths = driver_search_paths();

    for dir in search_paths {
        if !dir.is_dir() {
            continue;
        }

        for pattern in &lib_patterns {
            let glob_pattern = dir.join(pattern).to_string_lossy().into_owned();
            match glob(&glob_pattern) {
                Ok(mut entries) => {
                    if entries.find_map(StdResult::ok).is_some() {
                        return Ok(Some(dir));
                    }
                }
                Err(error) => {
                    return Err(FindError::glob("CUDA Driver", glob_pattern, &error));
                }
            }
        }
    }

    Ok(None)
}

/// Attempts to find the cuBLAS headers and library.
///
/// The search prefers `CUBLAS_PATH`/`CUBLAS_ROOT`, then `CUDA_PATH`/`CUDA_HOME`,
/// then common CUDA installation roots.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_cublas() -> Result<Option<Dependency>> {
    find_dependency(
        "cuBLAS",
        "cublas_v2.h",
        &["cublas_v2.h"],
        &cublas_lib_patterns(),
        parse_cublas_version_from_header,
        &["CUBLAS_PATH", "CUBLAS_ROOT"],
        None,
    )
}

/// Attempts to find the cuRAND headers and library.
///
/// The search prefers `CURAND_PATH`/`CURAND_ROOT`, then `CUDA_PATH`/`CUDA_HOME`,
/// then common CUDA installation roots.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_curand() -> Result<Option<Dependency>> {
    find_dependency(
        "cuRAND",
        "curand.h",
        &["curand.h"],
        &curand_lib_patterns(),
        parse_curand_version_from_header,
        &["CURAND_PATH", "CURAND_ROOT"],
        None,
    )
}

/// Attempts to find the cuSOLVER headers and library.
///
/// The search prefers `CUSOLVER_PATH`/`CUSOLVER_ROOT`, then
/// `CUDA_PATH`/`CUDA_HOME`, then common CUDA installation roots.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_cusolver() -> Result<Option<Dependency>> {
    find_dependency(
        "cuSOLVER",
        "cusolverDn.h",
        &["cusolver_common.h"],
        &cusolver_lib_patterns(),
        parse_cusolver_version_from_header,
        &["CUSOLVER_PATH", "CUSOLVER_ROOT"],
        None,
    )
}

/// Attempts to find the cuDSS headers and library.
///
/// Explicit `CUDSS_INCLUDE_PATH` plus `CUDSS_PATH` take precedence. The
/// fallback search checks `CUDSS_PATH`/`CUDSS_ROOT`, CUDA Toolkit roots, and
/// common cuDSS installation roots.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_cudss() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "cudss.h",
        &cudss_lib_patterns(),
        parse_cudss_version_from_header,
        &["CUDSS_INCLUDE_PATH"],
        &["CUDSS_PATH"],
    )? {
        return Ok(Some(dep));
    }

    find_dependency(
        "cuDSS",
        "cudss.h",
        &["cudss.h"],
        &cudss_lib_patterns(),
        parse_cudss_version_from_header,
        &["CUDSS_PATH", "CUDSS_ROOT"],
        None,
    )
}

/// Attempts to find the nvCOMP headers and library.
///
/// Explicit `NVCOMP_INCLUDE_PATH` plus `NVCOMP_PATH` take precedence. The
/// fallback search checks `NVCOMP_PATH`/`NVCOMP_ROOT`, CUDA Toolkit roots, and
/// common nvCOMP installation roots.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_nvcomp() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "nvcomp.h",
        &nvcomp_lib_patterns(),
        parse_nvcomp_version_from_header,
        &["NVCOMP_INCLUDE_PATH"],
        &["NVCOMP_PATH"],
    )? {
        return Ok(Some(dep));
    }

    find_dependency(
        "nvCOMP",
        "nvcomp.h",
        &["nvcomp/version.h", "nvcomp.h"],
        &nvcomp_lib_patterns(),
        parse_nvcomp_version_from_header,
        &["NVCOMP_PATH", "NVCOMP_ROOT"],
        None,
    )
}

/// Attempts to find the nvJPEG headers and library.
///
/// Explicit `NVJPEG_INCLUDE_PATH` plus `NVJPEG_PATH` take precedence. The
/// fallback search checks `NVJPEG_PATH`/`NVJPEG_ROOT`, CUDA Toolkit roots, and
/// common CUDA installation roots.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_nvjpeg() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "nvjpeg.h",
        &nvjpeg_lib_patterns(),
        parse_nvjpeg_version_from_header,
        &["NVJPEG_INCLUDE_PATH"],
        &["NVJPEG_PATH"],
    )? {
        return Ok(Some(dep));
    }

    find_dependency(
        "nvJPEG",
        "nvjpeg.h",
        &["nvjpeg.h"],
        &nvjpeg_lib_patterns(),
        parse_nvjpeg_version_from_header,
        &["NVJPEG_PATH", "NVJPEG_ROOT"],
        None,
    )
}

/// Attempts to find the nvJPEG 2000 headers and library.
///
/// Explicit `NVJPEG2K_INCLUDE_PATH` plus `NVJPEG2K_PATH` take precedence. The
/// fallback search checks `NVJPEG2K_PATH`/`NVJPEG2K_ROOT`, CUDA Toolkit roots,
/// and nvJPEG 2000 library directories that include the detected CUDA major
/// version such as `lib/13`.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_nvjpeg2k() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "nvjpeg2k.h",
        &nvjpeg2k_lib_patterns(),
        parse_nvjpeg2k_version_from_header,
        &["NVJPEG2K_INCLUDE_PATH"],
        &["NVJPEG2K_PATH"],
    )? {
        return Ok(Some(dep));
    }

    let primary_search_result = find_dependency(
        "nvJPEG 2000",
        "nvjpeg2k.h",
        &["nvjpeg2k_version.h", "nvjpeg2k.h"],
        &nvjpeg2k_lib_patterns(),
        parse_nvjpeg2k_version_from_header,
        &["NVJPEG2K_PATH", "NVJPEG2K_ROOT"],
        None,
    )?;

    if primary_search_result.is_some() {
        return Ok(primary_search_result);
    }

    cuda_major_version()?.map_or(Ok(None), |cuda_major_version| {
        find_dependency(
            "nvJPEG 2000 (versioned path)",
            "nvjpeg2k.h",
            &["nvjpeg2k_version.h", "nvjpeg2k.h"],
            &nvjpeg2k_lib_patterns(),
            parse_nvjpeg2k_version_from_header,
            &["NVJPEG2K_PATH", "NVJPEG2K_ROOT"],
            Some(&cuda_major_version),
        )
    })
}

/// Attempts to find the nvTIFF headers and library.
///
/// Explicit `NVTIFF_INCLUDE_PATH` plus `NVTIFF_PATH` take precedence. The
/// fallback search checks `NVTIFF_PATH`/`NVTIFF_ROOT`, CUDA Toolkit roots, and
/// nvTIFF library directories that include the detected CUDA major version such
/// as `lib/13`.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_nvtiff() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "nvtiff.h",
        &nvtiff_lib_patterns(),
        parse_nvtiff_version_from_header,
        &["NVTIFF_INCLUDE_PATH"],
        &["NVTIFF_PATH"],
    )? {
        return Ok(Some(dep));
    }

    let primary_search_result = find_dependency(
        "nvTIFF",
        "nvtiff.h",
        &["nvtiff_version.h", "nvtiff.h"],
        &nvtiff_lib_patterns(),
        parse_nvtiff_version_from_header,
        &["NVTIFF_PATH", "NVTIFF_ROOT"],
        None,
    )?;

    if primary_search_result.is_some() {
        return Ok(primary_search_result);
    }

    cuda_major_version()?.map_or(Ok(None), |cuda_major_version| {
        find_dependency(
            "nvTIFF (versioned path)",
            "nvtiff.h",
            &["nvtiff_version.h", "nvtiff.h"],
            &nvtiff_lib_patterns(),
            parse_nvtiff_version_from_header,
            &["NVTIFF_PATH", "NVTIFF_ROOT"],
            Some(&cuda_major_version),
        )
    })
}

/// Attempts to find the cuSPARSE headers and library.
///
/// The search prefers `CUSPARSE_PATH`/`CUSPARSE_ROOT`, then
/// `CUDA_PATH`/`CUDA_HOME`, then common CUDA installation roots.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_cusparse() -> Result<Option<Dependency>> {
    find_dependency(
        "cuSPARSE",
        "cusparse.h",
        &["cusparse.h"],
        &cusparse_lib_patterns(),
        parse_cusparse_version_from_header,
        &["CUSPARSE_PATH", "CUSPARSE_ROOT"],
        None,
    )
}

/// Attempts to find the cuTENSOR library.
///
/// Explicit `CUTENSOR_INCLUDE_PATH` plus `CUTENSOR_PATH` take precedence. The
/// fallback search checks `CUTENSOR_PATH`/`CUTENSOR_ROOT`, CUDA Toolkit roots,
/// and cuTENSOR library directories that include the detected CUDA major version
/// such as `lib/13`.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_cutensor() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "cutensor.h",
        &cutensor_lib_patterns(),
        parse_cutensor_version_from_header,
        &["CUTENSOR_INCLUDE_PATH"],
        &["CUTENSOR_PATH"],
    )? {
        return Ok(Some(dep));
    }

    // First, try the standard search logic.
    let primary_search_result = find_dependency(
        "cuTENSOR",
        "cutensor.h",
        &["cutensor.h"], // Header for version check.
        &cutensor_lib_patterns(),
        parse_cutensor_version_from_header,
        &["CUTENSOR_PATH", "CUTENSOR_ROOT"], // Potential specific env vars.
        None, // Don't require a specific major version for this initial generic search.
    )?;

    if primary_search_result.is_some() {
        return Ok(primary_search_result);
    }

    // If standard search failed, try searching specifically in versioned lib paths (e.g., lib/11, lib/12).
    cuda_major_version()?.map_or(Ok(None), |cuda_major_version| {
        find_dependency(
            "cuTENSOR (versioned path)",
            "cutensor.h",
            &["cutensor.h"],
            &cutensor_lib_patterns(),
            parse_cutensor_version_from_header,
            &["CUTENSOR_PATH", "CUTENSOR_ROOT"],
            Some(&cuda_major_version),
        )
    })
}

/// Attempts to find the cuFile headers and library.
///
/// Explicit `CUFILE_INCLUDE_PATH` plus `CUFILE_PATH` take precedence. The
/// fallback search checks `CUFILE_PATH`/`CUFILE_ROOT`, CUDA Toolkit roots, and
/// common cuFile installation roots.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_cufile() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "cufile.h",
        &cufile_lib_patterns(),
        parse_cufile_version_from_library_path,
        &["CUFILE_INCLUDE_PATH"],
        &["CUFILE_PATH"],
    )? {
        return Ok(Some(dep));
    }

    find_dependency(
        "cuFile",
        "cufile.h",
        &["cufile.h"],
        &cufile_lib_patterns(),
        parse_cufile_version_from_library_path,
        &["CUFILE_PATH", "CUFILE_ROOT"],
        None,
    )
}

/// Attempts to find the cuDNN headers and library.
///
/// Explicit `CUDNN_INCLUDE_PATH` plus `CUDNN_PATH` take precedence. The fallback
/// search checks `CUDNN_PATH`/`CUDNN_ROOT`, CUDA Toolkit roots, and common cuDNN
/// installation roots.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_cudnn() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "cudnn.h",
        &cudnn_lib_patterns(),
        parse_cudnn_version_from_header,
        &["CUDNN_INCLUDE_PATH"],
        &["CUDNN_PATH"],
    )? {
        return Ok(Some(dep));
    }

    find_dependency(
        "cuDNN",
        "cudnn.h",
        &["cudnn_version.h", "cudnn.h"],
        &cudnn_lib_patterns(),
        parse_cudnn_version_from_header,
        &["CUDNN_PATH", "CUDNN_ROOT"],
        None,
    )
}

/// Attempts to find the NCCL headers and library.
///
/// Explicit `NCCL_INCLUDE_PATH` plus `NCCL_PATH` take precedence. The fallback
/// search checks `NCCL_PATH`/`NCCL_ROOT`, CUDA Toolkit roots, and common NCCL
/// installation roots.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_nccl() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "nccl.h",
        &nccl_lib_patterns(),
        parse_nccl_version_from_header,
        &["NCCL_INCLUDE_PATH"],
        &["NCCL_PATH"],
    )? {
        return Ok(Some(dep));
    }

    find_dependency(
        "NCCL",
        "nccl.h",
        &["nccl.h"],
        &nccl_lib_patterns(),
        parse_nccl_version_from_header,
        &["NCCL_PATH", "NCCL_ROOT"],
        None,
    )
}

/// Attempts to find the NPP headers and library.
///
/// Explicit `NPP_INCLUDE_PATH` plus `NPP_PATH` take precedence. The fallback
/// search checks `NPP_PATH`/`NPP_ROOT`, CUDA Toolkit roots, and common NPP
/// installation roots.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_npp() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "npp.h",
        &npp_lib_patterns(),
        parse_npp_version_from_header,
        &["NPP_INCLUDE_PATH"],
        &["NPP_PATH"],
    )? {
        return Ok(Some(dep));
    }

    find_dependency(
        "NPP",
        "npp.h",
        &["npp.h"],
        &npp_lib_patterns(),
        parse_npp_version_from_header,
        &["NPP_PATH", "NPP_ROOT"],
        None,
    )
}

/// Attempts to find the CUPTI headers and library.
///
/// Explicit `CUPTI_INCLUDE_PATH` plus `CUPTI_PATH`/`CUPTI_ROOT` take precedence.
/// The fallback search checks `CUPTI_PATH`/`CUPTI_ROOT`, CUDA Toolkit roots,
/// and common CUDA installation locations.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_cupti() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "cupti.h",
        &cupti_lib_patterns(),
        parse_cupti_version_from_header,
        &["CUPTI_INCLUDE_PATH"],
        &["CUPTI_PATH", "CUPTI_ROOT"],
    )? {
        return Ok(Some(dep));
    }

    find_dependency(
        "CUPTI",
        "cupti.h",
        &["cupti.h"],
        &cupti_lib_patterns(),
        parse_cupti_version_from_header,
        &["CUPTI_PATH", "CUPTI_ROOT"],
        None,
    )
}

/// Attempts to find the nvImageCodec headers and library.
///
/// Explicit `NVIMGCODEC_INCLUDE_PATH` plus `NVIMGCODEC_PATH` take precedence.
/// The fallback search checks `NVIMGCODEC_PATH`/`NVIMGCODEC_ROOT`, CUDA Toolkit
/// roots, and nvImageCodec library directories that include the detected CUDA
/// major version such as `13/include`.
///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_nvimagecodec() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "nvimgcodec.h",
        &nvimagecodec_lib_patterns(),
        parse_nvimagecodec_version_from_header,
        &["NVIMGCODEC_INCLUDE_PATH"],
        &["NVIMGCODEC_PATH"],
    )? {
        return Ok(Some(dep));
    }

    let primary_search_result = find_dependency(
        "nvImageCodec",
        "nvimgcodec.h",
        &["nvimgcodec_version.h", "nvimgcodec.h"],
        &nvimagecodec_lib_patterns(),
        parse_nvimagecodec_version_from_header,
        &["NVIMGCODEC_PATH", "NVIMGCODEC_ROOT"],
        None,
    )?;

    if primary_search_result.is_some() {
        return Ok(primary_search_result);
    }

    cuda_major_version()?.map_or(Ok(None), |cuda_major_version| {
        find_dependency(
            "nvImageCodec (versioned path)",
            "nvimgcodec.h",
            &["nvimgcodec_version.h", "nvimgcodec.h"],
            &nvimagecodec_lib_patterns(),
            parse_nvimagecodec_version_from_header,
            &["NVIMGCODEC_PATH", "NVIMGCODEC_ROOT"],
            Some(&cuda_major_version),
        )
    })
}

///
/// # Errors
///
/// Returns [`FindError`] if a library search path produces an invalid glob pattern.
/// Missing headers or libraries produce `Ok(None)`.
pub fn find_dependency_from_explicit_paths(
    primary_header: &str,
    lib_patterns: &[&str],
    version_parser: fn(&Path) -> Option<String>,
    include_env_vars: &[&str],
    library_env_vars: &[&str],
) -> Result<Option<Dependency>> {
    let include_path = include_env_vars.iter().find_map(|var_name| {
        env::var_os(var_name)
            .map(PathBuf::from)
            .filter(|path| path.is_dir())
    });
    let Some(include_path) = include_path else {
        return Ok(None);
    };
    let library_path = library_env_vars.iter().find_map(|var_name| {
        env::var_os(var_name)
            .map(PathBuf::from)
            .filter(|path| path.is_dir())
    });
    let Some(library_path) = library_path else {
        return Ok(None);
    };

    if !include_path.join(primary_header).is_file() {
        return Ok(None);
    }

    let mut found_lib = false;
    for pattern in lib_patterns {
        let glob_pattern = library_path.join(pattern).to_string_lossy().into_owned();
        match glob(&glob_pattern) {
            Ok(mut entries) => {
                if entries.find_map(StdResult::ok).is_some() {
                    found_lib = true;
                    break;
                }
            }
            Err(error) => {
                return Err(FindError::glob(
                    "explicit CUDA dependency",
                    glob_pattern,
                    &error,
                ));
            }
        }
    }

    if !found_lib {
        return Ok(None);
    }

    Ok(Some(Dependency {
        version: version_parser(&include_path.join(primary_header)),
        include_path,
        library_path,
    }))
}
