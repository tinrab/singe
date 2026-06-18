//! CUDA Toolkit and NVIDIA library discovery helpers for Singe build scripts and
//! wrapper crates.
//!
//! This crate centralizes the path probing used by the Singe CUDA ecosystem. It
//! looks for headers, libraries, and tools through explicit environment
//! variables, the active `PATH` or dynamic library path, and common CUDA
//! installation roots. The result is a small [`Dependency`] value containing
//! include and library directories plus a detected version when the header
//! exposes one.
//!
//! # Discovery
//!
//! Toolkit components prefer component-specific roots such as `CUBLAS_PATH` or
//! `CUSOLVER_ROOT`, then fall back to `CUDA_PATH` or `CUDA_HOME`, then common
//! platform locations. Driver-owned libraries such as CUDA Driver and NVML are
//! searched in driver library paths because they are not necessarily installed
//! inside the CUDA Toolkit.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cuda = singe_cuda_find::find_cuda_runtime()?
//!     .expect("CUDA runtime headers and library were not found");
//!
//! println!("cargo:include={}", cuda.include_path.display());
//! println!("cargo:rustc-link-search=native={}", cuda.library_path.display());
//!
//! if let Some(version) = cuda.version {
//!     println!("cargo:warning=found CUDA runtime {version}");
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashSet;
use std::{
    env::{self, split_paths},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use glob::glob;
use regex::Regex;

/// Error returned when CUDA/NVIDIA dependency discovery cannot complete.
#[derive(Debug, Clone)]
pub struct FindError {
    /// Component being searched, such as `"CUDA Runtime"` or `"cuDNN"`.
    pub component: String,
    /// Human-readable reason discovery failed.
    pub message: String,
}

impl FindError {
    fn glob(
        component: impl Into<String>,
        pattern: impl Into<String>,
        error: glob::PatternError,
    ) -> Self {
        Self {
            component: component.into(),
            message: format!("invalid glob pattern `{}`: {error}", pattern.into()),
        }
    }
}

impl std::fmt::Display for FindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} discovery failed: {}", self.component, self.message)
    }
}

impl std::error::Error for FindError {}

/// Result type used by CUDA discovery helpers.
pub type Result<T> = std::result::Result<T, FindError>;

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
pub fn find_cuda_runtime() -> Result<Option<Dependency>> {
    find_dependency(
        "CUDA Runtime",
        "cuda_runtime_api.h",
        &["cuda.h"],
        &get_cuda_runtime_lib_patterns(),
        parse_cuda_version_from_header,
        &[],
        None,
    )
}

/// Attempts to find the NVRTC headers and library.
///
/// The search checks `CUDA_PATH`/`CUDA_HOME` and common CUDA Toolkit roots for
/// `nvrtc.h` plus a matching `nvrtc` library.
pub fn find_nvrtc() -> Result<Option<Dependency>> {
    find_dependency(
        "NVRTC",
        "nvrtc.h",
        &["nvrtc.h"],
        &get_nvrtc_lib_patterns(),
        parse_cuda_version_from_header,
        &[],
        None,
    )
}

/// Attempts to find the NVVM headers and library.
///
/// The search checks `NVVM_ROOT`/`NVVM_PATH`/`NVVM_INCLUDE_PATH`,
/// `CUDA_PATH`/`CUDA_HOME`, and common CUDA Toolkit roots for `nvvm.h` plus a
/// matching `nvvm` library.
pub fn find_nvvm() -> Result<Option<Dependency>> {
    if let Some(dependency) = find_dependency_from_explicit_paths(
        "nvvm.h",
        &get_nvvm_lib_patterns(),
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
        &get_nvvm_lib_patterns(),
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
pub fn find_nvml() -> Result<Option<PathBuf>> {
    let lib_patterns = get_nvml_lib_patterns();
    let search_paths = get_driver_search_paths();

    for dir in search_paths {
        if !dir.is_dir() {
            continue;
        }

        for pattern in &lib_patterns {
            let glob_pattern = dir.join(pattern).to_string_lossy().into_owned();
            match glob(&glob_pattern) {
                Ok(entries) => {
                    if entries.filter_map(std::result::Result::ok).next().is_some() {
                        return Ok(Some(dir));
                    }
                }
                Err(err) => {
                    return Err(FindError::glob("NVML", glob_pattern, err));
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

    Ok(get_potential_roots(&[], "CUDA Toolkit")
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
pub fn find_cuda_driver_path() -> Result<Option<PathBuf>> {
    let lib_patterns = get_cuda_driver_lib_patterns();
    let search_paths = get_driver_search_paths();

    for dir in search_paths {
        if !dir.is_dir() {
            continue;
        }

        for pattern in &lib_patterns {
            let glob_pattern = dir.join(pattern).to_string_lossy().into_owned();
            match glob(&glob_pattern) {
                Ok(entries) => {
                    if entries.filter_map(std::result::Result::ok).next().is_some() {
                        return Ok(Some(dir));
                    }
                }
                Err(err) => {
                    return Err(FindError::glob("CUDA Driver", glob_pattern, err));
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
pub fn find_cublas() -> Result<Option<Dependency>> {
    find_dependency(
        "cuBLAS",
        "cublas_v2.h",
        &["cublas_v2.h"],
        &get_cublas_lib_patterns(),
        parse_cublas_version_from_header,
        &["CUBLAS_PATH", "CUBLAS_ROOT"],
        None,
    )
}

/// Attempts to find the cuRAND headers and library.
///
/// The search prefers `CURAND_PATH`/`CURAND_ROOT`, then `CUDA_PATH`/`CUDA_HOME`,
/// then common CUDA installation roots.
pub fn find_curand() -> Result<Option<Dependency>> {
    find_dependency(
        "cuRAND",
        "curand.h",
        &["curand.h"],
        &get_curand_lib_patterns(),
        parse_curand_version_from_header,
        &["CURAND_PATH", "CURAND_ROOT"],
        None,
    )
}

/// Attempts to find the cuSOLVER headers and library.
///
/// The search prefers `CUSOLVER_PATH`/`CUSOLVER_ROOT`, then
/// `CUDA_PATH`/`CUDA_HOME`, then common CUDA installation roots.
pub fn find_cusolver() -> Result<Option<Dependency>> {
    find_dependency(
        "cuSOLVER",
        "cusolverDn.h",
        &["cusolver_common.h"],
        &get_cusolver_lib_patterns(),
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
pub fn find_cudss() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "cudss.h",
        &get_cudss_lib_patterns(),
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
        &get_cudss_lib_patterns(),
        parse_cudss_version_from_header,
        &["CUDSS_PATH", "CUDSS_ROOT"],
        None,
    )
}

/// Attempts to find the cuSPARSE headers and library.
///
/// The search prefers `CUSPARSE_PATH`/`CUSPARSE_ROOT`, then
/// `CUDA_PATH`/`CUDA_HOME`, then common CUDA installation roots.
pub fn find_cusparse() -> Result<Option<Dependency>> {
    find_dependency(
        "cuSPARSE",
        "cusparse.h",
        &["cusparse.h"],
        &get_cusparse_lib_patterns(),
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
pub fn find_cutensor() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "cutensor.h",
        &get_cutensor_lib_patterns(),
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
        &get_cutensor_lib_patterns(),
        parse_cutensor_version_from_header,
        &["CUTENSOR_PATH", "CUTENSOR_ROOT"], // Potential specific env vars.
        None, // Don't require a specific major version for this initial generic search.
    )?;

    if primary_search_result.is_some() {
        return Ok(primary_search_result);
    }

    // If standard search failed, try searching specifically in versioned lib paths (e.g., lib/11, lib/12).
    get_cuda_major_version()?.map_or(Ok(None), |cuda_major_version| {
        find_dependency(
            "cuTENSOR (versioned path)",
            "cutensor.h",
            &["cutensor.h"],
            &get_cutensor_lib_patterns(),
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
pub fn find_cufile() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "cufile.h",
        &get_cufile_lib_patterns(),
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
        &get_cufile_lib_patterns(),
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
pub fn find_cudnn() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "cudnn.h",
        &get_cudnn_lib_patterns(),
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
        &get_cudnn_lib_patterns(),
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
pub fn find_nccl() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "nccl.h",
        &get_nccl_lib_patterns(),
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
        &get_nccl_lib_patterns(),
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
pub fn find_npp() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "npp.h",
        &get_npp_lib_patterns(),
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
        &get_npp_lib_patterns(),
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
pub fn find_cupti() -> Result<Option<Dependency>> {
    if let Some(dep) = find_dependency_from_explicit_paths(
        "cupti.h",
        &get_cupti_lib_patterns(),
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
        &get_cupti_lib_patterns(),
        parse_cupti_version_from_header,
        &["CUPTI_PATH", "CUPTI_ROOT"],
        None,
    )
}

fn find_dependency_from_explicit_paths(
    primary_header: &str,
    lib_patterns: &[&str],
    version_parser: fn(&Path) -> Option<String>,
    include_env_vars: &[&str],
    library_env_vars: &[&str],
) -> Result<Option<Dependency>> {
    let include_path = include_env_vars.iter().find_map(|var_name| {
        env::var(var_name)
            .ok()
            .map(PathBuf::from)
            .filter(|path| path.is_dir())
    });
    let Some(include_path) = include_path else {
        return Ok(None);
    };
    let library_path = library_env_vars.iter().find_map(|var_name| {
        env::var(var_name)
            .ok()
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
            Ok(entries) => {
                if entries.filter_map(std::result::Result::ok).next().is_some() {
                    found_lib = true;
                    break;
                }
            }
            Err(err) => {
                return Err(FindError::glob(
                    "explicit CUDA dependency",
                    glob_pattern,
                    err,
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

const COMMON_CUDA_ENV_VARS: [&str; 2] = ["CUDA_PATH", "CUDA_HOME"];

const CUDA_DISCOVERY_ENV_VARS: [&str; 53] = [
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

#[cfg(target_os = "windows")]
fn get_cuda_runtime_lib_patterns() -> Vec<&'static str> {
    vec!["cudart*.lib", "cudart*.dll"]
}

#[cfg(target_os = "linux")]
fn get_cuda_runtime_lib_patterns() -> Vec<&'static str> {
    vec!["libcudart.so", "libcudart.so.*"]
}

#[cfg(target_os = "windows")]
fn get_nvrtc_lib_patterns() -> Vec<&'static str> {
    vec!["nvrtc*.lib", "nvrtc*.dll"]
}

#[cfg(target_os = "linux")]
fn get_nvrtc_lib_patterns() -> Vec<&'static str> {
    vec!["libnvrtc.so", "libnvrtc.so.*"]
}

#[cfg(target_os = "windows")]
fn get_nvvm_lib_patterns() -> Vec<&'static str> {
    vec!["nvvm*.lib", "nvvm*.dll"]
}

#[cfg(target_os = "linux")]
fn get_nvvm_lib_patterns() -> Vec<&'static str> {
    vec!["libnvvm.so", "libnvvm.so.*"]
}

#[cfg(target_os = "windows")]
fn get_nvml_lib_patterns() -> Vec<&'static str> {
    vec!["nvml.lib", "nvml.dll", "nvidia-ml.lib", "nvidia-ml.dll"]
}

#[cfg(target_os = "linux")]
fn get_nvml_lib_patterns() -> Vec<&'static str> {
    vec!["libnvidia-ml.so", "libnvidia-ml.so.*"]
}

#[cfg(target_os = "windows")]
fn get_cuda_driver_lib_patterns() -> Vec<&'static str> {
    vec!["nvcuda.dll"] // The core driver library
}

#[cfg(target_os = "linux")]
fn get_cuda_driver_lib_patterns() -> Vec<&'static str> {
    vec!["libcuda.so", "libcuda.so.*"]
}

#[cfg(target_os = "windows")]
fn get_cublas_lib_patterns() -> Vec<&'static str> {
    vec!["cublas*.lib", "cublas*.dll"]
}

#[cfg(target_os = "linux")]
fn get_cublas_lib_patterns() -> Vec<&'static str> {
    vec!["libcublas.so", "libcublas.so.*"]
}

#[cfg(target_os = "windows")]
fn get_curand_lib_patterns() -> Vec<&'static str> {
    vec!["curand*.lib", "curand*.dll"]
}

#[cfg(target_os = "linux")]
fn get_curand_lib_patterns() -> Vec<&'static str> {
    vec!["libcurand.so", "libcurand.so.*"]
}

#[cfg(target_os = "windows")]
fn get_cusolver_lib_patterns() -> Vec<&'static str> {
    vec!["cusolver*.lib", "cusolver*.dll"]
}

#[cfg(target_os = "linux")]
fn get_cusolver_lib_patterns() -> Vec<&'static str> {
    vec!["libcusolver.so", "libcusolver.so.*"]
}

#[cfg(target_os = "windows")]
fn get_cudss_lib_patterns() -> Vec<&'static str> {
    vec!["cudss*.lib", "cudss*.dll"]
}

#[cfg(target_os = "linux")]
fn get_cudss_lib_patterns() -> Vec<&'static str> {
    vec!["libcudss.so", "libcudss.so.*"]
}

#[cfg(target_os = "windows")]
fn get_cusparse_lib_patterns() -> Vec<&'static str> {
    vec!["cusparse*.lib", "cusparse*.dll"]
}

#[cfg(target_os = "linux")]
fn get_cusparse_lib_patterns() -> Vec<&'static str> {
    vec!["libcusparse.so", "libcusparse.so.*"]
}

#[cfg(target_os = "windows")]
fn get_cutensor_lib_patterns() -> Vec<&'static str> {
    vec!["cutensor*.lib", "cutensor*.dll"]
}

#[cfg(target_os = "linux")]
fn get_cutensor_lib_patterns() -> Vec<&'static str> {
    vec!["libcutensor.so", "libcutensor.so.*"]
}

#[cfg(target_os = "windows")]
fn get_cufile_lib_patterns() -> Vec<&'static str> {
    vec!["cufile*.lib", "cufile*.dll"]
}

#[cfg(target_os = "linux")]
fn get_cufile_lib_patterns() -> Vec<&'static str> {
    vec!["libcufile.so", "libcufile.so.*"]
}

#[cfg(target_os = "windows")]
fn get_cudnn_lib_patterns() -> Vec<&'static str> {
    vec!["cudnn*.lib", "cudnn*.dll"]
}

#[cfg(target_os = "linux")]
fn get_cudnn_lib_patterns() -> Vec<&'static str> {
    vec!["libcudnn.so", "libcudnn.so.*"]
}

#[cfg(target_os = "windows")]
fn get_cupti_lib_patterns() -> Vec<&'static str> {
    vec!["cupti*.lib", "cupti*.dll"]
}

#[cfg(target_os = "linux")]
fn get_cupti_lib_patterns() -> Vec<&'static str> {
    vec!["libcupti.so", "libcupti.so.*"]
}

#[cfg(target_os = "windows")]
fn get_nccl_lib_patterns() -> Vec<&'static str> {
    vec!["nccl*.lib", "nccl*.dll"]
}

#[cfg(target_os = "linux")]
fn get_nccl_lib_patterns() -> Vec<&'static str> {
    vec!["libnccl.so", "libnccl.so.*"]
}

#[cfg(target_os = "windows")]
fn get_npp_lib_patterns() -> Vec<&'static str> {
    vec!["npp*.lib", "npp*.dll"]
}

#[cfg(target_os = "linux")]
fn get_npp_lib_patterns() -> Vec<&'static str> {
    vec!["libnppc.so", "libnppc.so.*"]
}

/// Finds a specific CUDA-related dependency.
fn find_dependency(
    lib_name: &str,
    primary_header: &str,
    version_headers: &[&str],
    lib_patterns: &[&str],
    primary_version_parser: fn(&Path) -> Option<String>,
    specific_env_vars: &[&str],
    cuda_major_version_hint: Option<&str>,
) -> Result<Option<Dependency>> {
    let potential_roots = get_potential_roots(specific_env_vars, lib_name);

    for root in potential_roots {
        if let Some(dep) = check_root_for_dependency(
            lib_name,
            &root,
            primary_header,
            version_headers,
            lib_patterns,
            primary_version_parser,
            cuda_major_version_hint,
        )? {
            return Ok(Some(dep));
        }
    }

    Ok(None)
}

/// Checks a specific root directory for the necessary components of a dependency.
fn check_root_for_dependency(
    lib_name: &str,
    root: &Path,
    primary_header: &str,
    version_headers: &[&str],
    lib_patterns: &[&str],
    primary_version_parser: fn(&Path) -> Option<String>,
    cuda_major_version_hint: Option<&str>,
) -> Result<Option<Dependency>> {
    let include_path = root.join("include");

    if !include_path.join(primary_header).is_file() {
        return Ok(None);
    }

    let Some(lib_path_to_check) = find_library_path(root, cuda_major_version_hint) else {
        return Ok(None);
    };

    let mut found_lib = false;
    for pattern in lib_patterns {
        let glob_pattern = lib_path_to_check
            .join(pattern)
            .to_string_lossy()
            .into_owned();
        match glob(&glob_pattern) {
            Ok(entries) => {
                if entries.filter_map(std::result::Result::ok).next().is_some() {
                    found_lib = true;
                    break;
                }
            }
            Err(err) => {
                return Err(FindError::glob(lib_name, glob_pattern, err));
            }
        }
    }

    if !found_lib {
        return Ok(None);
    }

    let mut version = None;
    for header_name in version_headers {
        let header_path = include_path.join(header_name);
        if header_path.is_file() {
            version = primary_version_parser(&header_path);
            if version.is_some() {
                break;
            }
        }
    }

    Ok(Some(Dependency {
        include_path,
        library_path: lib_path_to_check,
        version,
    }))
}

/// Finds the most likely library subdirectory within a CUDA root.
/// If `cuda_major_version_hint` is provided, it prioritizes checking versioned paths like `lib/<major_version>`.
fn find_library_path(root: &Path, cuda_major_version_hint: Option<&str>) -> Option<PathBuf> {
    let mut potential_lib_dirs = Vec::new();

    // If hint provided, check versioned path first.
    if let Some(major_version) = cuda_major_version_hint {
        #[cfg(target_os = "windows")]
        {
            // Order matters. Check specific versioned path first if applicable.
            potential_lib_dirs.push(root.join("lib").join(major_version)); // e.g. C:/.../cutensor/lib/12
            potential_lib_dirs.push(root.join("lib").join("x64")); // Standard toolkit path
        }
        #[cfg(target_os = "linux")]
        {
            // Order matters: check specific versioned path first if applicable
            potential_lib_dirs.push(root.join("lib").join(major_version)); // e.g. /usr/local/cutensor/lib/12
            potential_lib_dirs.push(root.join("lib64")); // Standard toolkit path
        }
    }

    // Add standard non-versioned paths.
    #[cfg(target_os = "windows")]
    {
        potential_lib_dirs.push(root.join("lib").join("x64"));
        potential_lib_dirs.push(root.join("lib"));
    }
    #[cfg(target_os = "linux")]
    {
        potential_lib_dirs.push(root.join("lib64"));
        potential_lib_dirs.push(root.join("lib"));
    }

    // Check root itself.
    potential_lib_dirs.push(root.to_path_buf());

    // Iterate through potential paths and return the first one that exists and is a directory.
    potential_lib_dirs.into_iter().find(|path| path.is_dir())
}

/// Gathers potential root directories from environment variables and common locations.
fn get_potential_roots(specific_env_vars: &[&str], lib_name_hint: &str) -> Vec<PathBuf> {
    let mut roots = Vec::new();

    // Specific environment variables first.
    for var_name in specific_env_vars {
        if let Ok(path_str) = env::var(var_name) {
            let path = PathBuf::from(path_str);
            if path.is_dir() {
                roots.push(path);
            }
        }
    }

    // Common CUDA environment variables.
    for var_name in &COMMON_CUDA_ENV_VARS {
        if let Ok(path_str) = env::var(var_name) {
            let path = PathBuf::from(path_str);
            if path.is_dir() {
                roots.push(path);
            }
        }
    }

    // Standard system paths using glob.
    #[cfg(target_os = "windows")]
    let glob_patterns = [
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*",
        // Add potential dedicated paths if needed on Windows, e.g., C:/Program Files/NVIDIA/cuDNN/..., etc.
    ];

    #[cfg(target_os = "linux")]
    let mut glob_patterns = vec![
        "/usr/local/cuda-*", // Common install pattern
        "/opt/cuda-*",       // Alternative common pattern
        "/usr/local/cuda",   // Default symlink/install
        "/opt/cuda",
        // Broader system paths with lower priority.
        "/usr",
        "/usr/local",
    ];

    // Add specific known paths for libraries like cuDNN/cuTensor/cuBLAS if their env vars weren't set.
    #[cfg(target_os = "linux")]
    {
        if lib_name_hint.contains("cuDNN") && specific_env_vars.iter().all(|v| env::var(v).is_err())
        {
            glob_patterns.push("/usr/local/cudnn");
        }
        if lib_name_hint.contains("cuTENSOR")
            && specific_env_vars.iter().all(|v| env::var(v).is_err())
        {
            // Add the potential root for versioned search.
            glob_patterns.push("/usr/local/cutensor");
        }
        if lib_name_hint.contains("cuFile")
            && specific_env_vars.iter().all(|v| env::var(v).is_err())
        {
            glob_patterns.push("/usr/local/cufile");
        }
        if lib_name_hint.contains("cuBLAS")
            && specific_env_vars.iter().all(|v| env::var(v).is_err())
        {
            // cuBLAS is usually part of toolkit, but check just in case.
            glob_patterns.push("/usr/local/cublas");
        }
        if lib_name_hint.contains("cuRAND")
            && specific_env_vars.iter().all(|v| env::var(v).is_err())
        {
            glob_patterns.push("/usr/local/curand");
        }
        if lib_name_hint.contains("cuSOLVER")
            && specific_env_vars.iter().all(|v| env::var(v).is_err())
        {
            glob_patterns.push("/usr/local/cusolver");
        }
        if lib_name_hint.contains("cuSPARSE")
            && specific_env_vars.iter().all(|v| env::var(v).is_err())
        {
            glob_patterns.push("/usr/local/cusparse");
        }
        if lib_name_hint.contains("NCCL") && specific_env_vars.iter().all(|v| env::var(v).is_err())
        {
            glob_patterns.push("/usr/local/nccl");
        }
        if lib_name_hint.contains("NPP") && specific_env_vars.iter().all(|v| env::var(v).is_err()) {
            glob_patterns.push("/usr/local/npp");
        }
    }

    for pattern in &glob_patterns {
        if let Ok(entries) = glob(pattern) {
            for entry in entries.filter_map(std::result::Result::ok) {
                if entry.is_dir() {
                    roots.push(entry);
                }
            }
        }
    }

    // Deduplicate and normalize paths.
    let mut unique_roots = Vec::new();
    for root in roots {
        let canonical = fs::canonicalize(&root).unwrap_or_else(|_| root.clone());
        if !unique_roots.contains(&canonical) {
            unique_roots.push(canonical);
        }
    }

    unique_roots
}

#[cfg(target_os = "windows")]
fn get_driver_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut seen_paths = HashSet::new();

    // PATH environment variable.
    if let Ok(path_var) = env::var("PATH") {
        for path in split_paths(&path_var) {
            if path.is_dir() && seen_paths.insert(path.clone()) {
                paths.push(path);
            }
        }
    }

    // Standard System directories.
    if let Ok(system_root) = env::var("SystemRoot") {
        let system32 = PathBuf::from(&system_root).join("System32");
        if system32.is_dir() && seen_paths.insert(system32.clone()) {
            paths.push(system32);
        }
        // For 32-bit DLLs on 64-bit systems, though nvcuda.dll is usually 64-bit.
        if cfg!(target_arch = "x86_64") {
            let syswow64 = PathBuf::from(&system_root).join("SysWOW64");
            if syswow64.is_dir() && seen_paths.insert(syswow64.clone()) {
                paths.push(syswow64);
            }
        }
    } else {
        // Fallback if SystemRoot isn't set.
        let system32_fallback = PathBuf::from("C:/Windows/System32");
        if system32_fallback.is_dir() && seen_paths.insert(system32_fallback.clone()) {
            paths.push(system32_fallback);
        }
        if cfg!(target_arch = "x86_64") {
            let syswow64_fallback = PathBuf::from("C:/Windows/SysWOW64");
            if syswow64_fallback.is_dir() && seen_paths.insert(syswow64_fallback.clone()) {
                paths.push(syswow64_fallback);
            }
        }
    }

    // Explicit NVIDIA Driver locations.
    // These paths might vary greatly. Focusing on System32/PATH is usually sufficient.
    let program_files = env::var("ProgramFiles").unwrap_or_else(|_| "C:/Program Files".to_string());
    let nvidia_driver_base = PathBuf::from(program_files).join("NVIDIA Corporation/Driver");
    if nvidia_driver_base.is_dir() && seen_paths.insert(nvidia_driver_base.clone()) {
        paths.push(nvidia_driver_base);
    }
    // Check for NVSMI directory.
    let nvsmi_path = PathBuf::from(program_files).join("NVIDIA Corporation/NVSMI");
    if nvsmi_path.is_dir() && seen_paths.insert(nvsmi_path.clone()) {
        paths.push(nvsmi_path);
    }

    paths
}

#[cfg(target_os = "linux")]
fn get_driver_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut stub_paths = Vec::new();
    let mut seen_paths = HashSet::new();

    // LD_LIBRARY_PATH environment variable.
    if let Ok(ld_path_var) = env::var("LD_LIBRARY_PATH") {
        for path in split_paths(&ld_path_var).filter(|p| p.is_dir()) {
            if seen_paths.insert(path.clone()) {
                paths.push(path);
            }
        }
    }

    for var_name in &COMMON_CUDA_ENV_VARS {
        if let Ok(root) = env::var(var_name) {
            let root = PathBuf::from(root);
            for candidate in [root.join("lib"), root.join("lib64")] {
                if candidate.is_dir() && seen_paths.insert(candidate.clone()) {
                    paths.push(candidate);
                }
            }
            for candidate in [
                root.join("lib").join("stubs"),
                root.join("lib64").join("stubs"),
            ] {
                if candidate.is_dir() && seen_paths.insert(candidate.clone()) {
                    stub_paths.push(candidate);
                }
            }
        }
    }

    // Nix systems often expose the real NVIDIA driver through this path even when
    // the CUDA toolkit only provides stub libcuda libraries.
    if let Ok(ld_path_var) = env::var("NIX_LD_LIBRARY_PATH") {
        for path in split_paths(&ld_path_var).filter(|p| p.is_dir()) {
            if seen_paths.insert(path.clone()) {
                paths.push(path);
            }
        }
    }

    // Standard system library paths, ordered from driver-specific locations to
    // broader system directories.
    let standard_paths = [
        // NixOS/OpenGL driver mount points.
        "/run/opengl-driver/lib",
        "/run/current-system/sw/lib",
        // Multiarch paths first
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
        // Standard 64-bit paths
        "/usr/lib64",
        "/lib64",
        "/usr/lib",
        "/lib",
        // Less common, but possible
        "/usr/local/lib64",
        "/usr/local/lib",
        // Paths potentially used by NVIDIA drivers directly (often symlinked from above)
        // Globbing might be better here, but let's list common static ones.
        "/usr/lib/nvidia",
        "/usr/lib64/nvidia",
        // Check paths listed in ld.so.conf (requires parsing or calling ldconfig -p)
        // This is more complex, relying on standard paths covers most cases.
    ];

    for p_str in &standard_paths {
        let path = PathBuf::from(p_str);
        if path.is_dir() && seen_paths.insert(path.clone()) {
            paths.push(path);
        }
    }

    // We intentionally avoid shelling out to ldconfig here; the fixed path list
    // keeps discovery deterministic and works in minimal build environments.
    // if let Ok(output) = Command::new("ldconfig").arg("-p").output() {
    //     if output.status.success() {
    //         let stdout = String::from_utf8_lossy(&output.stdout);
    //         // Parse lines like: libcuda.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so.1
    //         // Extract unique directory paths containing libcuda.so
    //     }
    // }

    paths.extend(stub_paths);
    paths
}

/// Attempts to get the CUDA major version, such as `"12"` or `"13"`.
/// Tries CUDA Runtime first, then falls back to nvcc.
fn get_cuda_major_version() -> Result<Option<String>> {
    let version_string = find_cuda_runtime()?
        .and_then(|dep| dep.version)
        .or_else(parse_cuda_version_from_nvcc);

    Ok(version_string.and_then(|v| v.split('.').next().map(String::from)))
}

fn parse_version_from_header_defines(
    header_path: &Path,
    major_define: &str,
    minor_define: &str,
    patch_define: &str,
) -> Option<String> {
    let content = fs::read_to_string(header_path).ok()?;
    let mut major = None;
    let mut minor = None;
    let mut patch = None;

    // Relaxed regex to allow for comments after the value.
    let define_re =
        |name: &str| -> Regex { Regex::new(&format!(r"#define\s+{name}\s+(\d+).*")).unwrap() };

    let major_re = define_re(major_define);
    let minor_re = define_re(minor_define);
    let patch_re = define_re(patch_define);

    for line in content.lines() {
        let trimmed_line = line.trim_start();
        if !trimmed_line.starts_with("#define") {
            continue;
        }

        if major.is_none()
            && let Some(caps) = major_re.captures(trimmed_line)
        {
            major = caps.get(1).map(|m| m.as_str().to_string());
            continue; // Move to next line once found
        }
        if minor.is_none()
            && let Some(caps) = minor_re.captures(trimmed_line)
        {
            minor = caps.get(1).map(|m| m.as_str().to_string());
            continue;
        }
        if patch.is_none()
            && let Some(caps) = patch_re.captures(trimmed_line)
        {
            patch = caps.get(1).map(|m| m.as_str().to_string());
            // Don't break here, allow finding others if defines are out of order.
        }
        if major.is_some() && minor.is_some() && patch.is_some() {
            break;
        }
    }

    match (major, minor, patch) {
        (Some(maj), Some(min), Some(pat)) => Some(format!("{maj}.{min}.{pat}")),
        (Some(maj), Some(min), None) => Some(format!("{maj}.{min}")),
        _ => None,
    }
}

fn parse_cuda_version_from_header(header_path: &Path) -> Option<String> {
    let content = fs::read_to_string(header_path).ok()?;
    let version_re = Regex::new(r"#define\s+CUDA_VERSION\s+(\d+)").unwrap();

    for line in content.lines() {
        if let Some(caps) = version_re.captures(line)
            && let Some(match_str) = caps.get(1)
        {
            let version_code: u32 = match_str.as_str().parse().ok()?;
            // Format XXXXX into M.m (e.g., 11080 -> 11.8).
            let major = version_code / 1000;
            let minor = (version_code % 1000) / 10;
            return Some(format!("{major}.{minor}"));
        }
    }

    None
}

fn parse_cublas_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CUBLAS_VER_MAJOR",
        "CUBLAS_VER_MINOR",
        "CUBLAS_VER_PATCH",
    )
}

fn parse_curand_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CURAND_VER_MAJOR",
        "CURAND_VER_MINOR",
        "CURAND_VER_PATCH",
    )
}

fn parse_cusolver_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CUSOLVER_VER_MAJOR",
        "CUSOLVER_VER_MINOR",
        "CUSOLVER_VER_PATCH",
    )
}

fn parse_cudss_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CUDSS_VERSION_MAJOR",
        "CUDSS_VERSION_MINOR",
        "CUDSS_VERSION_PATCH",
    )
}

fn parse_cusparse_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CUSPARSE_VER_MAJOR",
        "CUSPARSE_VER_MINOR",
        "CUSPARSE_VER_PATCH",
    )
}

fn parse_cudnn_version_from_header(header_path: &Path) -> Option<String> {
    let version_header_path = header_path.with_file_name("cudnn_version.h");
    let path_to_parse = if version_header_path.is_file() {
        &version_header_path
    } else {
        header_path
    };

    parse_version_from_header_defines(
        path_to_parse,
        "CUDNN_MAJOR",
        "CUDNN_MINOR",
        "CUDNN_PATCHLEVEL",
    )
}

fn parse_cupti_version_from_header(header_path: &Path) -> Option<String> {
    let content = fs::read_to_string(header_path).ok()?;
    let api_version_re = Regex::new(r"#define\s+CUPTI_API_VERSION\s+(\d+)").ok()?;

    if let Some(captures) = api_version_re.captures(&content) {
        return captures.get(1).map(|capture| capture.as_str().to_string());
    }

    parse_version_from_header_defines(
        header_path,
        "CUPTI_MAJOR_VERSION",
        "CUPTI_MINOR_VERSION",
        "CUPTI_PATCHLEVEL",
    )
}

fn parse_cutensor_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CUTENSOR_MAJOR",
        "CUTENSOR_MINOR",
        "CUTENSOR_PATCH",
    )
}

fn parse_cufile_version_from_library_path(header_path: &Path) -> Option<String> {
    if let Ok(library_path) = env::var("CUFILE_PATH")
        && let Some(version) = parse_cufile_version_from_library_dir(Path::new(&library_path))
    {
        return Some(version);
    }

    let include_path = header_path.parent()?;
    for library_path in [
        include_path.parent()?.join("lib"),
        include_path.parent()?.join("lib64"),
    ] {
        if let Some(version) = parse_cufile_version_from_library_dir(&library_path) {
            return Some(version);
        }
    }

    None
}

fn parse_cufile_version_from_library_dir(library_path: &Path) -> Option<String> {
    let mut candidates = Vec::new();

    let glob_pattern = library_path
        .join("libcufile.so.*")
        .to_string_lossy()
        .into_owned();
    let Ok(entries) = glob(&glob_pattern) else {
        return None;
    };

    for entry in entries.filter_map(std::result::Result::ok) {
        if let Some(file_name) = entry.file_name().and_then(|name| name.to_str()) {
            candidates.push(file_name.to_owned());
        }
    }

    let version_re = Regex::new(r"^libcufile\.so\.(\d+)\.(\d+)\.(\d+)$").unwrap();
    candidates
        .into_iter()
        .filter_map(|file_name| {
            let caps = version_re.captures(&file_name)?;
            let major: u32 = caps.get(1)?.as_str().parse().ok()?;
            let minor: u32 = caps.get(2)?.as_str().parse().ok()?;
            let patch: u32 = caps.get(3)?.as_str().parse().ok()?;
            Some((major, minor, patch))
        })
        .max()
        .map(|(major, minor, patch)| format!("{major}.{minor}.{patch}"))
}

fn parse_nccl_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(header_path, "NCCL_MAJOR", "NCCL_MINOR", "NCCL_PATCH")
}

fn parse_npp_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "NPP_VER_MAJOR",
        "NPP_VER_MINOR",
        "NPP_VER_PATCH",
    )
}

fn parse_cuda_version_from_nvcc() -> Option<String> {
    let nvcc = find_nvcc().ok()??;
    let output = Command::new(nvcc).arg("--version").output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version_re = Regex::new(r"V(\d+\.\d+\.\d+)").unwrap();

    if let Some(caps) = version_re.captures(&stdout)
        && let Some(full_version) = caps.get(1)
    {
        return Some(full_version.as_str().to_string());
    }

    let release_re = Regex::new(r"release\s+(\d+\.\d+)").unwrap();
    if let Some(caps) = release_re.captures(&stdout)
        && let Some(release_version) = caps.get(1)
    {
        return Some(format!("{}.0", release_version.as_str()));
    }

    None
}

fn command_in_path(command: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;

    split_paths(&path_var).find_map(|entry| {
        #[cfg(target_os = "windows")]
        let candidate = entry.join(format!("{command}.exe"));

        #[cfg(not(target_os = "windows"))]
        let candidate = entry.join(command);

        candidate.is_file().then_some(candidate)
    })
}
