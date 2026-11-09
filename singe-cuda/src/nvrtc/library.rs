use std::path::PathBuf;

/// Returns the path of the NVRTC shared library, loading it if it is not yet.
/// Returns `None` if NVRTC does not load.
#[must_use]
pub fn library_path() -> Option<PathBuf> {
    singe_cuda_sys::nvrtc_paths().ok()?.into_iter().next()
}
