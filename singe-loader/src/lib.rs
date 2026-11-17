//! Loads the NVIDIA libraries behind Singe's `*-sys` crates at run time, so a program that uses them builds and starts on a machine without them.
//! A function whose library or symbol is missing returns a fixed value of its return type instead of failing to link.
//!
//! This crate provides:
//!
//! - **[`Library`]:** one library made of one or more files, with a `const` constructor per NVIDIA library, such as [`Library::cublas()`] or [`Library::nvjpeg2k()`], and [`Library::new`] for any other.
//! - **[`functions!`]:** turns the declarations of a bindgen `extern "C"` block into functions of the same names and signatures that look up their symbol on first call.
//! - **[`Missing`]:** what a function returns when it cannot be called: zero for integers, null for pointers, and nothing for [`()`](prim@unit), with [`missing_status!`] and [`missing_value!`] to implement it for a library's own types.
//! - **[`take_absence`]:** returns and clears the [`Absence`] that the last status-returning call on the current thread recorded, so a safe wrapper can turn that status into a load error.
//! - **[`LoadError`]:** a library with no files on this platform, a library that would not load, with each [`Attempt`] the loader made, or a missing symbol.
//!
//! # Example
//!
//! Load cuBLAS and print where each of its files was found, or each place the loader tried:
//!
//! ```rust,no_run
//! use singe_loader::{Library, LoadError};
//!
//! static CUBLAS: Library = Library::cublas();
//!
//! fn main() -> anyhow::Result<()> {
//!     match CUBLAS.paths() {
//!         Ok(paths) => {
//!             for path in paths {
//!                 println!("{} loaded from {}", CUBLAS.name(), path.display());
//!             }
//!         }
//!         Err(LoadError::NotFound { library, attempts }) => {
//!             for attempt in &attempts {
//!                 println!("{library}: {}: {}", attempt.path.display(), attempt.reason);
//!             }
//!         }
//!         Err(error) => return Err(error.into()),
//!     }
//!     Ok(())
//! }
//! ```
//!
//! A library loads at most once and stays loaded for the life of the process, so the functions that cached its symbols stay valid.
//! Its load result is cached too: a library that failed to load returns the same [`LoadError`] on every later call.
//!
//! # Search order
//!
//! The loader tries each file by name through the system loader's search first, then in the library directories under `CUDA_PATH` and `CUDA_HOME`, and on Linux under `/usr/local/cuda` and in `/run/opengl-driver/lib`.
//! The NVIDIA libraries have file names only on Linux and Windows; elsewhere, loading one returns [`LoadError::Unsupported`].

mod libraries;

use std::{
    cell::Cell,
    env,
    ffi::{CStr, OsString},
    fmt,
    path::{Path, PathBuf},
    ptr,
    sync::OnceLock,
};

/// One library behind a set of declarations, made of every file in `files`, each of which must load.
///
/// A symbol is looked up in each file in order, as NPP's functions are spread over eleven.
/// The NVIDIA libraries have constructors of their own, such as [`Library::cusparse`], which name their files on each platform.
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
    /// Creates a library named `name` from the files that make it up on this platform.
    #[must_use]
    pub const fn new(name: &'static str, files: &'static [&'static str]) -> Self {
        Self {
            name,
            files,
            loaded: OnceLock::new(),
        }
    }

    /// Returns the library's name, such as `cuSPARSE`.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Loads every file once, and keeps them loaded for the life of the process, which every cached function relies on.
    ///
    /// # Errors
    ///
    /// Returns an error if the library has no files on this platform or a file would not load.
    pub fn load(&self) -> Result<(), LoadError> {
        self.files().map(|_| ())
    }

    /// Returns where each file loaded from, in order, loading them if they are not yet.
    ///
    /// # Errors
    ///
    /// Returns an error if the library would not load.
    pub fn paths(&self) -> Result<Vec<PathBuf>, LoadError> {
        Ok(self.files()?.iter().map(|file| file.path.clone()).collect())
    }

    fn files(&self) -> Result<&[File], LoadError> {
        self.loaded
            .get_or_init(|| {
                if self.files.is_empty() {
                    return Err(LoadError::Unsupported { library: self.name });
                }

                self.files
                    .iter()
                    .map(|file| open(self.name, file))
                    .collect()
            })
            .as_deref()
            .map_err(Clone::clone)
    }

    /// Looks up the symbol `name` in each file in order, as a value of type `F`.
    ///
    /// Returns `None` if the library would not load or no file exports `name`.
    ///
    /// # Safety
    ///
    /// `F` must be the symbol's actual type.
    #[doc(hidden)]
    #[must_use]
    pub unsafe fn symbol<F: Copy>(&self, name: &CStr) -> Option<F> {
        let files = self.files().ok()?;
        files.iter().find_map(|file| {
            // SAFETY: the caller guarantees `F` is the symbol's type, and `files` keeps the library loaded.
            unsafe { file.library.get::<F>(name.to_bytes_with_nul()) }
                .ok()
                .map(|symbol| *symbol)
        })
    }

    /// Returns why the function `symbol` of this library is missing: its library, or only its symbol.
    #[doc(hidden)]
    #[must_use]
    pub fn absence(&'static self, symbol: &'static str) -> Absence {
        if self.files().is_ok() {
            Absence::Symbol {
                library: self,
                name: symbol,
            }
        } else {
            Absence::Library(self)
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
    Err(LoadError::NotFound {
        library: name,
        attempts,
    })
}

// Asks glibc's loader for the file it opened, which a bare file name does not say.
// `dlinfo` and the layout of `struct link_map` are glibc's, so other C libraries skip the lookup.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn locate(library: libloading::Library) -> (libloading::Library, Option<PathBuf>) {
    use std::{
        ffi::{OsStr, c_char, c_int, c_void},
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

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
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

/// A library or symbol that would not load.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum LoadError {
    /// The library has no files on this platform.
    #[error("{library} is not available on this platform")]
    Unsupported {
        /// Name of the library.
        library: &'static str,
    },
    /// A file of the library would not load, with each place tried and why it failed.
    #[error("failed to load {library}: {}", describe(attempts))]
    NotFound {
        /// Name of the library.
        library: &'static str,
        /// Each place tried, in order.
        attempts: Vec<Attempt>,
    },
    /// The library loaded but lacks a symbol, as an older version of it would.
    #[error("{library} has no symbol `{symbol}`")]
    MissingSymbol {
        /// Name of the library.
        library: &'static str,
        /// Name of the missing symbol.
        symbol: &'static str,
    },
}

/// One place the loader tried a library file, and why it failed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Attempt {
    /// The path or file name tried.
    pub path: PathBuf,
    /// The system loader's reason.
    pub reason: String,
}

fn describe(attempts: &[Attempt]) -> String {
    attempts
        .iter()
        .map(|attempt| attempt.reason.clone())
        .collect::<Vec<_>>()
        .join("; ")
}

/// Why a function could not be called.
#[derive(Debug, Clone, Copy)]
pub enum Absence {
    /// Its library would not load.
    Library(&'static Library),
    /// Its library loaded but lacks the symbol `name`.
    Symbol {
        /// The library looked in.
        library: &'static Library,
        /// Name of the missing symbol.
        name: &'static str,
    },
}

impl Absence {
    /// Returns the library the function belongs to.
    #[must_use]
    pub const fn library(self) -> &'static Library {
        match self {
            Self::Library(library) | Self::Symbol { library, .. } => library,
        }
    }

    /// Returns the error describing this absence.
    #[must_use]
    pub fn into_error(self) -> LoadError {
        match self {
            Self::Library(library) => match library.load() {
                Err(error) => error,
                // A library's load result is cached, so one recorded as absent never loads later.
                Ok(()) => LoadError::Unsupported {
                    library: library.name,
                },
            },
            Self::Symbol { library, name } => LoadError::MissingSymbol {
                library: library.name,
                symbol: name,
            },
        }
    }
}

thread_local! {
    static ABSENCE: Cell<Option<Absence>> = const { Cell::new(None) };
}

/// Returns and clears the absence the last status-returning function called on this thread recorded.
///
/// A function whose library or symbol is missing records it before returning its status, and one that runs clears the record.
#[must_use]
pub fn take_absence() -> Option<Absence> {
    ABSENCE.take()
}

#[doc(hidden)]
pub fn record_absence(absence: Absence) {
    ABSENCE.set(Some(absence));
}

#[doc(hidden)]
pub fn clear_absence() {
    ABSENCE.set(None);
}

/// What a function returns when it cannot be called.
pub trait Missing: Sized {
    /// Returns the value standing for a call that could not be made for `absence`.
    fn missing(absence: Absence) -> Self;

    /// Passes through the result of a call that was made.
    ///
    /// Status types override it to clear the thread's recorded absence, so a later failure is not blamed on an earlier missing function.
    #[doc(hidden)]
    #[must_use]
    fn made(self) -> Self {
        self
    }
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

/// Implements [`Missing`] for a library's status type, with the status for a missing library and for a missing symbol.
///
/// A missing call records its [`Absence`] for [`take_absence`], and a call that runs clears the record.
#[macro_export]
macro_rules! missing_status {
    ($status:ty, library: $library:expr, symbol: $symbol:expr $(,)?) => {
        impl $crate::Missing for $status {
            fn missing(absence: $crate::Absence) -> Self {
                $crate::record_absence(absence);
                match absence {
                    $crate::Absence::Library(_) => $library,
                    $crate::Absence::Symbol { .. } => $symbol,
                }
            }

            fn made(self) -> Self {
                $crate::clear_absence();
                self
            }
        }
    };
}

/// Implements [`Missing`] for a value type that is not a status, returning `$value` and recording nothing.
#[macro_export]
macro_rules! missing_value {
    ($type:ty, $value:expr $(,)?) => {
        impl $crate::Missing for $type {
            fn missing(_: $crate::Absence) -> Self {
                $value
            }
        }
    };
}

// The `&CStr` naming a function or variable, for `Library::symbol`.
#[doc(hidden)]
#[macro_export]
macro_rules! symbol_name {
    ($name:ident) => {{
        const NAME: &::core::ffi::CStr = match ::core::ffi::CStr::from_bytes_with_nul(
            ::core::concat!(::core::stringify!($name), "\0").as_bytes(),
        ) {
            ::core::result::Result::Ok(name) => name,
            ::core::result::Result::Err(_) => {
                ::core::panic!("a symbol name has no interior nul byte")
            }
        };
        NAME
    }};
}

/// Turns the declarations of an `extern "C"` block, as bindgen writes them, into functions of the same names and signatures that look up their symbol in `$library` on first call.
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
                $library.symbol::<Function>($crate::symbol_name!($name))
            });
            match function {
                // SAFETY: `function` has the declared signature, and this function's caller meets the C function's preconditions.
                ::core::option::Option::Some(function) => $crate::Missing::made(unsafe { function($($argument),*) }),
                ::core::option::Option::None => $crate::Missing::missing($library.absence(::core::stringify!($name))),
            }
        }
    )*};
    // A variable the library exports, read through a function of its name, since its value is only known once the library loads.
    ($library:path; $(#[$meta:meta])* pub static $name:ident: $type:ty;) => {
        $(#[$meta])*
        pub unsafe fn $name() -> $type {
            // The variable's address, kept as an exposed integer so the cache can be shared between threads.
            static ADDRESS: ::std::sync::OnceLock<::core::option::Option<usize>> =
                ::std::sync::OnceLock::new();
            // SAFETY: the variable is declared with the type bindgen wrote from the library's header.
            let address = ADDRESS.get_or_init(|| unsafe {
                $library
                    .symbol::<*const $type>($crate::symbol_name!($name))
                    .map(<*const $type>::expose_provenance)
            });
            match address {
                // SAFETY: the address is the loaded library's variable, which stays loaded for the life of the process.
                ::core::option::Option::Some(address) => unsafe {
                    *::core::ptr::with_exposed_provenance::<$type>(*address)
                },
                ::core::option::Option::None => {
                    $crate::Missing::missing($library.absence(::core::stringify!($name)))
                }
            }
        }
    };
}
