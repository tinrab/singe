// Loads NVIDIA libraries at run time, so a crate that calls them builds without them and its binary starts without them.
// A missing library or symbol becomes a value of the function's return type, never a failed link or start.

mod libraries;

use std::{
    env,
    ffi::OsString,
    fmt,
    path::{Path, PathBuf},
    ptr,
    sync::OnceLock,
};

// One library behind a set of declarations, made of every file in `files`, each of which must load.
// A symbol is looked up in each file in order, as NPP's functions are spread over eleven.
// The NVIDIA libraries have constructors of their own, such as `Library::cusparse`, which name their files on each platform.
pub struct Library {
    name: &'static str,
    files: &'static [&'static str],
    loaded: OnceLock<Result<Vec<File>, LoadError>>,
}

// A file of a library, loaded, and where the loader found it.
struct File {
    library: libloading::Library,
    path: PathBuf,
}

impl Library {
    #[must_use]
    pub const fn new(name: &'static str, files: &'static [&'static str]) -> Self {
        Self {
            name,
            files,
            loaded: OnceLock::new(),
        }
    }

    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    // Loads every file once, and keeps them loaded for the life of the process, which every cached function relies on.
    pub fn load(&self) -> Result<(), LoadError> {
        self.files().map(|_| ())
    }

    // Where each file loaded from, in order, loading them if they are not yet.
    pub fn paths(&self) -> Result<Vec<PathBuf>, LoadError> {
        Ok(self.files()?.iter().map(|file| file.path.clone()).collect())
    }

    fn files(&self) -> Result<&[File], LoadError> {
        self.loaded
            .get_or_init(|| {
                if self.files.is_empty() {
                    return Err(LoadError {
                        name: self.name,
                        attempts: Vec::new(),
                    });
                }

                self.files
                    .iter()
                    .map(|file| open(self.name, file))
                    .collect()
            })
            .as_deref()
            .map_err(Clone::clone)
    }

    // Looks up `name`, a symbol's name ending in a nul byte, as a function of type `F`.
    //
    // # Safety
    //
    // `F` must be the symbol's actual type.
    #[must_use]
    pub unsafe fn symbol<F: Copy>(&self, name: &str) -> Option<F> {
        let files = self.files().ok()?;
        files.iter().find_map(|file| {
            // SAFETY: the caller guarantees `F` is the symbol's type, and `files` keeps the library loaded.
            unsafe { file.library.get::<F>(name.as_bytes()) }
                .ok()
                .map(|symbol| *symbol)
        })
    }

    // Why a function of this library is missing: its library, or only its symbol.
    #[must_use]
    pub fn absence(&self) -> Absence {
        if self.files().is_ok() {
            Absence::Symbol
        } else {
            Absence::Library
        }
    }
}

impl fmt::Debug for Library {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Library")
            .field("name", &self.name)
            .field("files", &self.files)
            .finish_non_exhaustive()
    }
}

// Opens `file` by name, through the system loader's search, and then in the directories NVIDIA installs to.
fn open(name: &'static str, file: &'static str) -> Result<File, LoadError> {
    let mut attempts = Vec::new();
    for candidate in candidates(file) {
        // SAFETY: loading an NVIDIA library runs only its initializers, which have no preconditions.
        match unsafe { libloading::Library::new(&candidate) } {
            Ok(library) => {
                let (library, path) = locate(library);
                return Ok(File {
                    path: path.unwrap_or_else(|| PathBuf::from(&candidate)),
                    library,
                });
            }
            Err(error) => attempts.push(Attempt {
                path: PathBuf::from(candidate),
                reason: error.to_string(),
            }),
        }
    }
    Err(LoadError { name, attempts })
}

// Asks the Linux loader for the file it opened, which a bare file name does not say.
#[cfg(target_os = "linux")]
fn locate(library: libloading::Library) -> (libloading::Library, Option<PathBuf>) {
    use std::{
        ffi::{CStr, OsStr, c_char, c_int, c_void},
        os::unix::ffi::OsStrExt,
    };

    use libloading::os::unix::Library as UnixLibrary;

    // The start of glibc's `struct link_map`, which is all `dlinfo` callers may read.
    #[repr(C)]
    struct LinkMap {
        address: usize,
        name: *const c_char,
    }

    const RTLD_DI_LINKMAP: c_int = 2;

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlinfo(handle: *mut c_void, request: c_int, info: *mut c_void) -> c_int;
    }

    let handle = UnixLibrary::from(library).into_raw();
    let mut map: *const LinkMap = ptr::null();
    // SAFETY: `handle` is a live handle `dlopen` returned, and `RTLD_DI_LINKMAP` writes one pointer to `map`.
    let found = unsafe { dlinfo(handle, RTLD_DI_LINKMAP, (&raw mut map).cast()) } == 0;
    let path = (found && !map.is_null())
        .then(|| {
            // SAFETY: the loader owns the map and its nul-terminated name while the library stays loaded.
            let name = unsafe { CStr::from_ptr((*map).name) };
            (!name.is_empty()).then(|| PathBuf::from(OsStr::from_bytes(name.to_bytes())))
        })
        .flatten();
    // SAFETY: `handle` came from `into_raw` above and is owned again here.
    let library = unsafe { UnixLibrary::from_raw(handle) }.into();
    (library, path)
}

#[cfg(not(target_os = "linux"))]
fn locate(library: libloading::Library) -> (libloading::Library, Option<PathBuf>) {
    (library, None)
}

fn candidates(file: &str) -> Vec<OsString> {
    let mut candidates = vec![OsString::from(file)];
    candidates.extend(
        directories()
            .into_iter()
            .map(|directory| directory.join(file))
            .filter(|path| path.exists())
            .map(PathBuf::into_os_string),
    );
    candidates
}

// Where NVIDIA installs libraries on this platform, searched after the system loader's own search.
fn directories() -> Vec<PathBuf> {
    let mut roots = ["CUDA_PATH", "CUDA_HOME"]
        .into_iter()
        .filter_map(env::var_os)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if cfg!(target_os = "windows") {
        // CUDA 13 moved its DLLs from `bin` to `bin\x64`; the driver's and NVML's are in `System32`, which the system search covers.
        return roots
            .iter()
            .flat_map(|root| [root.join("bin").join("x64"), root.join("bin")])
            .collect();
    }

    roots.push(PathBuf::from("/usr/local/cuda"));
    let mut directories = roots
        .iter()
        .flat_map(|root| {
            ["lib64", "lib", "nvvm/lib64"]
                .into_iter()
                .map(|directory| root.join(directory))
        })
        .collect::<Vec<_>>();
    // NixOS puts the driver's libraries here.
    directories.push(Path::new("/run/opengl-driver/lib").to_path_buf());
    directories
}

// A library that would not load, with each place tried and why it failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("failed to load {name}: {}", describe(attempts))]
pub struct LoadError {
    pub name: &'static str,
    pub attempts: Vec<Attempt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attempt {
    pub path: PathBuf,
    pub reason: String,
}

fn describe(attempts: &[Attempt]) -> String {
    if attempts.is_empty() {
        return "it has no library on this platform".to_owned();
    }

    attempts
        .iter()
        .map(|attempt| attempt.reason.clone())
        .collect::<Vec<_>>()
        .join("; ")
}

// Why a function could not be called.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Absence {
    Library,
    Symbol,
}

// What a function returns when it cannot be called.
pub trait Missing {
    fn missing(absence: Absence) -> Self;
}

impl Missing for () {
    fn missing(_: Absence) -> Self {}
}

impl<T> Missing for *mut T {
    fn missing(_: Absence) -> Self {
        ptr::null_mut()
    }
}

impl<T> Missing for *const T {
    fn missing(_: Absence) -> Self {
        ptr::null()
    }
}

macro_rules! missing_zero {
    ($($integer:ty),*) => {$(
        impl Missing for $integer {
            fn missing(_: Absence) -> Self {
                0
            }
        }
    )*};
}

missing_zero!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

// Implements `Missing` for a library's status type, with the status for a missing library and for a missing symbol.
#[macro_export]
macro_rules! missing_status {
    ($status:ty, library: $library:expr, symbol: $symbol:expr $(,)?) => {
        impl $crate::Missing for $status {
            fn missing(absence: $crate::Absence) -> Self {
                match absence {
                    $crate::Absence::Library => $library,
                    $crate::Absence::Symbol => $symbol,
                }
            }
        }
    };
}

// Turns the declarations of an `extern "C"` block, as bindgen writes them, into functions of the same names and signatures that look up their symbol in `$library` on first call.
#[macro_export]
macro_rules! functions {
    ($library:path; $(
        $(#[$meta:meta])*
        pub fn $name:ident($($argument:ident: $type:ty),* $(,)?) $(-> $result:ty)?;
    )*) => {$(
        $(#[$meta])*
        pub unsafe extern "C" fn $name($($argument: $type),*) $(-> $result)? {
            type Function = unsafe extern "C" fn($($type),*) $(-> $result)?;
            static FUNCTION: ::std::sync::OnceLock<::core::option::Option<Function>> =
                ::std::sync::OnceLock::new();
            // SAFETY: `Function` is the declaration bindgen wrote from the library's header.
            let function = FUNCTION.get_or_init(|| unsafe {
                $library.symbol::<Function>(::core::concat!(::core::stringify!($name), "\0"))
            });
            match function {
                // SAFETY: the caller upholds the C function's contract.
                ::core::option::Option::Some(function) => unsafe { function($($argument),*) },
                ::core::option::Option::None => $crate::Missing::missing($library.absence()),
            }
        }
    )*};
    // A variable the library exports, read through a function of its name, since its value is only known once the library loads.
    ($library:path; $(#[$meta:meta])* pub static $name:ident: $type:ty;) => {
        $(#[$meta])*
        pub unsafe fn $name() -> $type {
            // The variable's address, kept as an integer so the cache can be shared between threads.
            static ADDRESS: ::std::sync::OnceLock<::core::option::Option<usize>> =
                ::std::sync::OnceLock::new();
            // SAFETY: the variable is declared with the type bindgen wrote from the library's header.
            let address = ADDRESS.get_or_init(|| unsafe {
                $library
                    .symbol::<*const $type>(::core::concat!(::core::stringify!($name), "\0"))
                    .map(|address| address as usize)
            });
            match address {
                // SAFETY: the address is the loaded library's variable, which stays loaded.
                ::core::option::Option::Some(address) => unsafe { *(*address as *const $type) },
                ::core::option::Option::None => $crate::Missing::missing($library.absence()),
            }
        }
    };
}
