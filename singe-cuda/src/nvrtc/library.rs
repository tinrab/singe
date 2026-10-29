use std::path::PathBuf;

/// Returns the loaded NVRTC shared-library path when the platform can identify it.
/// Linux resolves the actual loader symbol, including preloaded overrides, without opening another library.
/// Other platforms and static-library deployments may return `None`.
#[must_use]
pub fn library_path() -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        linux::library_path()
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{
        ffi::{CStr, OsStr, c_char, c_int, c_void},
        mem::MaybeUninit,
        os::unix::ffi::OsStrExt,
        path::PathBuf,
        ptr,
    };

    #[repr(C)]
    struct LibraryInfo {
        filename: *const c_char,
        _base: *mut c_void,
        _symbol_name: *const c_char,
        _symbol_address: *mut c_void,
    }

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;

        fn dladdr(address: *const c_void, info: *mut LibraryInfo) -> c_int;
    }

    pub fn library_path() -> Option<PathBuf> {
        let mut info = MaybeUninit::<LibraryInfo>::uninit();
        // SAFETY: a null handle searches the Linux loader's global symbol scope.
        // dladdr receives writable storage matching Dl_info and initializes it on success.
        let info = unsafe {
            let symbol = dlsym(ptr::null_mut(), c"nvrtcVersion".as_ptr());
            if symbol.is_null() || dladdr(symbol, info.as_mut_ptr()) == 0 {
                return None;
            }
            info.assume_init()
        };
        if info.filename.is_null() {
            return None;
        }
        // SAFETY: the linked NVRTC library remains loaded; the loader owns this terminated filename.
        let filename = unsafe { CStr::from_ptr(info.filename) };
        if filename.is_empty() {
            return None;
        }
        Some(PathBuf::from(OsStr::from_bytes(filename.to_bytes())))
    }
}
