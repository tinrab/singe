use std::{
    env::{self},
    fs,
    path::Path,
    process::Command,
    result::Result as StdResult,
};

use glob::glob;
use regex::Regex;

use crate::{
    dependency::{find_cuda_runtime, find_nvcc},
    error::Result,
};

/// Attempts to get the CUDA major version, such as `"12"` or `"13"`.
/// Tries CUDA Runtime first, then falls back to nvcc.
pub fn cuda_major_version() -> Result<Option<String>> {
    let version_string = find_cuda_runtime()?
        .and_then(|dep| dep.version)
        .or_else(parse_cuda_version_from_nvcc);

    Ok(version_string.and_then(|v| v.split('.').next().map(String::from)))
}

pub fn parse_version_from_header_defines(
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
    let define_re = |name: &str| -> Regex {
        Regex::new(&format!(r"#define\s+{name}\s+(\d+).*")).expect("static regex is valid")
    };

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

pub fn parse_cuda_version_from_header(header_path: &Path) -> Option<String> {
    let content = fs::read_to_string(header_path).ok()?;
    let version_re = Regex::new(r"#define\s+CUDA_VERSION\s+(\d+)").expect("static regex is valid");

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

pub fn parse_cublas_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CUBLAS_VER_MAJOR",
        "CUBLAS_VER_MINOR",
        "CUBLAS_VER_PATCH",
    )
}

pub fn parse_curand_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CURAND_VER_MAJOR",
        "CURAND_VER_MINOR",
        "CURAND_VER_PATCH",
    )
}

pub fn parse_cusolver_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CUSOLVER_VER_MAJOR",
        "CUSOLVER_VER_MINOR",
        "CUSOLVER_VER_PATCH",
    )
}

pub fn parse_cudss_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CUDSS_VERSION_MAJOR",
        "CUDSS_VERSION_MINOR",
        "CUDSS_VERSION_PATCH",
    )
}

pub fn parse_nvcomp_version_from_header(header_path: &Path) -> Option<String> {
    let sibling_version_path = header_path.with_file_name("version.h");
    let nested_version_path = header_path.parent()?.join("nvcomp/version.h");
    let path_to_parse = if sibling_version_path.is_file() {
        &sibling_version_path
    } else if nested_version_path.is_file() {
        &nested_version_path
    } else {
        header_path
    };

    parse_version_from_header_defines(
        path_to_parse,
        "NVCOMP_VER_MAJOR",
        "NVCOMP_VER_MINOR",
        "NVCOMP_VER_PATCH",
    )
}

pub fn parse_cusparse_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CUSPARSE_VER_MAJOR",
        "CUSPARSE_VER_MINOR",
        "CUSPARSE_VER_PATCH",
    )
}

pub fn parse_cudnn_version_from_header(header_path: &Path) -> Option<String> {
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

pub fn parse_cupti_version_from_header(header_path: &Path) -> Option<String> {
    let content = fs::read_to_string(header_path).ok()?;
    let api_version_re =
        Regex::new(r"#define\s+CUPTI_API_VERSION\s+(\d+)").expect("static regex is valid");

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

pub fn parse_cutensor_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "CUTENSOR_MAJOR",
        "CUTENSOR_MINOR",
        "CUTENSOR_PATCH",
    )
}

pub fn parse_nvimagecodec_version_from_header(header_path: &Path) -> Option<String> {
    let version_header_path = header_path.with_file_name("nvimgcodec_version.h");
    let path_to_parse = if version_header_path.is_file() {
        &version_header_path
    } else {
        header_path
    };

    parse_version_from_header_defines(
        path_to_parse,
        "NVIMGCODEC_VER_MAJOR",
        "NVIMGCODEC_VER_MINOR",
        "NVIMGCODEC_VER_PATCH",
    )
}

pub fn parse_nvjpeg_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "NVJPEG_VER_MAJOR",
        "NVJPEG_VER_MINOR",
        "NVJPEG_VER_PATCH",
    )
}

pub fn parse_nvjpeg2k_version_from_header(header_path: &Path) -> Option<String> {
    let version_header_path = header_path.with_file_name("nvjpeg2k_version.h");
    let path_to_parse = if version_header_path.is_file() {
        &version_header_path
    } else {
        header_path
    };

    parse_version_from_header_defines(
        path_to_parse,
        "NVJPEG2K_VER_MAJOR",
        "NVJPEG2K_VER_MINOR",
        "NVJPEG2K_VER_PATCH",
    )
}

pub fn parse_nvtiff_version_from_header(header_path: &Path) -> Option<String> {
    let version_header_path = header_path.with_file_name("nvtiff_version.h");
    let path_to_parse = if version_header_path.is_file() {
        &version_header_path
    } else {
        header_path
    };

    parse_version_from_header_defines(
        path_to_parse,
        "NVTIFF_VER_MAJOR",
        "NVTIFF_VER_MINOR",
        "NVTIFF_VER_PATCH",
    )
}

pub fn parse_cufile_version_from_library_path(header_path: &Path) -> Option<String> {
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

pub fn parse_cufile_version_from_library_dir(library_path: &Path) -> Option<String> {
    let mut candidates = Vec::new();

    let glob_pattern = library_path
        .join("libcufile.so.*")
        .to_string_lossy()
        .into_owned();
    let Ok(entries) = glob(&glob_pattern) else {
        return None;
    };

    for entry in entries.filter_map(StdResult::ok) {
        if let Some(file_name) = entry.file_name().and_then(|name| name.to_str()) {
            candidates.push(file_name.to_owned());
        }
    }

    let version_re =
        Regex::new(r"^libcufile\.so\.(\d+)\.(\d+)\.(\d+)$").expect("static regex is valid");
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

pub fn parse_nccl_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(header_path, "NCCL_MAJOR", "NCCL_MINOR", "NCCL_PATCH")
}

pub fn parse_npp_version_from_header(header_path: &Path) -> Option<String> {
    parse_version_from_header_defines(
        header_path,
        "NPP_VER_MAJOR",
        "NPP_VER_MINOR",
        "NPP_VER_PATCH",
    )
}

pub fn parse_cuda_version_from_nvcc() -> Option<String> {
    let nvcc = find_nvcc().ok()??;
    let output = Command::new(nvcc).arg("--version").output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version_re = Regex::new(r"V(\d+\.\d+\.\d+)").expect("static regex is valid");

    if let Some(caps) = version_re.captures(&stdout)
        && let Some(full_version) = caps.get(1)
    {
        return Some(full_version.as_str().to_string());
    }

    let release_re = Regex::new(r"release\s+(\d+\.\d+)").expect("static regex is valid");
    if let Some(caps) = release_re.captures(&stdout)
        && let Some(release_version) = caps.get(1)
    {
        return Some(format!("{}.0", release_version.as_str()));
    }

    None
}

#[cfg(test)]
mod tests {
    use std::{env, fs, path::PathBuf, process};

    use super::*;

    fn write_header(name: &str, contents: &str) -> PathBuf {
        let directory = env::temp_dir().join(format!("singe-cuda-find-{}", process::id()));
        fs::create_dir_all(&directory).expect("temporary directory is writable");
        let path = directory.join(name);
        fs::write(&path, contents).expect("temporary header is writable");
        path
    }

    #[test]
    fn test_parses_cuda_version_define() {
        let header = write_header("cuda.h", "#define CUDA_VERSION 13040 /* 13.4 */\n");
        assert_eq!(
            parse_cuda_version_from_header(&header).as_deref(),
            Some("13.4")
        );
    }

    #[test]
    fn test_parses_major_minor_patch_defines() {
        let header = write_header(
            "cublas_api.h",
            "#define CUBLAS_VER_MAJOR 13\n#define CUBLAS_VER_MINOR 8 // minor\n#define CUBLAS_VER_PATCH 2\n",
        );
        assert_eq!(
            parse_cublas_version_from_header(&header).as_deref(),
            Some("13.8.2")
        );
    }

    #[test]
    fn test_missing_header_has_no_version() {
        let header = env::temp_dir().join("singe-cuda-find-missing/cuda.h");
        assert_eq!(parse_cuda_version_from_header(&header), None);
    }
}
