use std::{
    collections::HashSet,
    env::{self, split_paths},
    fs,
    path::{Path, PathBuf},
    result::Result as StdResult,
};

use glob::glob;

use crate::{
    dependency::Dependency,
    environment::COMMON_CUDA_ENV_VARS,
    error::{FindError, Result},
};

/// Finds a specific CUDA-related dependency.
pub fn find_dependency(
    lib_name: &str,
    primary_header: &str,
    version_headers: &[&str],
    lib_patterns: &[&str],
    primary_version_parser: fn(&Path) -> Option<String>,
    specific_env_vars: &[&str],
    cuda_major_version_hint: Option<&str>,
) -> Result<Option<Dependency>> {
    let potential_roots = potential_roots(specific_env_vars, lib_name);

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
pub fn check_root_for_dependency(
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
            Ok(mut entries) => {
                if entries.find_map(StdResult::ok).is_some() {
                    found_lib = true;
                    break;
                }
            }
            Err(error) => {
                return Err(FindError::glob(lib_name, glob_pattern, &error));
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
pub fn find_library_path(root: &Path, cuda_major_version_hint: Option<&str>) -> Option<PathBuf> {
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
pub fn potential_roots(specific_env_vars: &[&str], lib_name_hint: &str) -> Vec<PathBuf> {
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
        if lib_name_hint.contains("nvCOMP")
            && specific_env_vars.iter().all(|v| env::var(v).is_err())
        {
            glob_patterns.push("/usr/local/nvcomp");
        }
    }

    for pattern in &glob_patterns {
        if let Ok(entries) = glob(pattern) {
            for entry in entries.filter_map(StdResult::ok) {
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
pub(crate) fn driver_search_paths() -> Vec<PathBuf> {
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
pub fn driver_search_paths() -> Vec<PathBuf> {
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

pub fn command_in_path(command: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;

    split_paths(&path_var).find_map(|entry| {
        #[cfg(target_os = "windows")]
        let candidate = entry.join(format!("{command}.exe"));

        #[cfg(not(target_os = "windows"))]
        let candidate = entry.join(command);

        candidate.is_file().then_some(candidate)
    })
}
