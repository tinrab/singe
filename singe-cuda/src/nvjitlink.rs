//! Runtime linking of PTX, cubin and link-time optimization IR.

use std::{ffi::CString, mem, ptr, result::Result as StdResult};

use singe_core::{impl_enum_display, report_cleanup_failure};
use singe_cuda_sys::nvjitlink as sys;

use crate::{
    architecture::GpuArchitecture,
    error::{Error, Result},
};

singe_core::wrapper_status! {
    library = "nvjitlink",
    error = Error::NvJitLink,
    raw = sys::nvJitLinkResult,
    success = NVJITLINK_SUCCESS,
    absence = singe_cuda_sys::take_absence,
    check = check_nvjitlink,
    diagnostic = String::new,
    unknown = "UNKNOWN_NVJITLINK_STATUS",
    /// API call result codes returned by nvJitLink.
    pub enum Status: u32 {
        /// Successful completion.
        Success = NVJITLINK_SUCCESS => "success",
        /// Unrecognized Option
        UnrecognizedOption = NVJITLINK_ERROR_UNRECOGNIZED_OPTION => "unrecognized option",
        /// Option `-arch=sm_NN` not specified
        MissingArch = NVJITLINK_ERROR_MISSING_ARCH => "missing arch",
        /// Invalid Input
        InvalidInput = NVJITLINK_ERROR_INVALID_INPUT => "invalid input",
        /// Issue during PTX Compilation
        PtxCompile = NVJITLINK_ERROR_PTX_COMPILE => "ptx compile",
        /// Issue during NVVM Compilation
        NvvmCompile = NVJITLINK_ERROR_NVVM_COMPILE => "nvvm compile",
        /// Internal Error
        Internal = NVJITLINK_ERROR_INTERNAL => "internal",
        /// Issue with Thread Pool
        Threadpool = NVJITLINK_ERROR_THREADPOOL => "threadpool",
        /// Unrecognized Input
        UnrecognizedInput = NVJITLINK_ERROR_UNRECOGNIZED_INPUT => "unrecognized input",
        /// Finalizer Error
        Finalize = NVJITLINK_ERROR_FINALIZE => "finalize",
        /// Null Input
        NullInput = NVJITLINK_ERROR_NULL_INPUT => "null input",
        /// Incompatible Options
        IncompatibleOptions = NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS => "incompatible options",
        /// Incorrect Input Type
        IncorrectInputType = NVJITLINK_ERROR_INCORRECT_INPUT_TYPE => "incorrect input type",
        /// Arch Mismatch
        ArchMismatch = NVJITLINK_ERROR_ARCH_MISMATCH => "arch mismatch",
        /// Outdated Library
        OutdatedLibrary = NVJITLINK_ERROR_OUTDATED_LIBRARY => "outdated library",
        /// Missing Fatbin
        MissingFatbin = NVJITLINK_ERROR_MISSING_FATBIN => "missing fatbin",
        /// Unrecognized -arch value
        UnrecognizedArch = NVJITLINK_ERROR_UNRECOGNIZED_ARCH => "unrecognized arch",
        /// Unsupported -arch value
        UnsupportedArch = NVJITLINK_ERROR_UNSUPPORTED_ARCH => "unsupported arch",
        /// Requires -lto
        LtoNotEnabled = NVJITLINK_ERROR_LTO_NOT_ENABLED => "lto not enabled",
    }
}

/// The kind of inputs that can be passed to the linker.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
#[non_exhaustive]
pub enum InputType {
    /// CUDA binaries.
    Cubin = sys::nvJitLinkInputType::NVJITLINK_INPUT_CUBIN.0,
    /// PTX.
    Ptx = sys::nvJitLinkInputType::NVJITLINK_INPUT_PTX.0,
    /// LTO-IR.
    LtoIr = sys::nvJitLinkInputType::NVJITLINK_INPUT_LTOIR.0,
    /// Fatbin.
    Fatbin = sys::nvJitLinkInputType::NVJITLINK_INPUT_FATBIN.0,
    /// Host object.
    Object = sys::nvJitLinkInputType::NVJITLINK_INPUT_OBJECT.0,
    /// Host library.
    Library = sys::nvJitLinkInputType::NVJITLINK_INPUT_LIBRARY.0,
    /// Index file.
    Index = sys::nvJitLinkInputType::NVJITLINK_INPUT_INDEX.0,
    /// Dynamically chooses from the valid types.
    Any = sys::nvJitLinkInputType::NVJITLINK_INPUT_ANY.0,
}

impl InputType {
    fn raw(self) -> sys::nvJitLinkInputType {
        match self {
            Self::Cubin => sys::nvJitLinkInputType::NVJITLINK_INPUT_CUBIN,
            Self::Ptx => sys::nvJitLinkInputType::NVJITLINK_INPUT_PTX,
            Self::LtoIr => sys::nvJitLinkInputType::NVJITLINK_INPUT_LTOIR,
            Self::Fatbin => sys::nvJitLinkInputType::NVJITLINK_INPUT_FATBIN,
            Self::Object => sys::nvJitLinkInputType::NVJITLINK_INPUT_OBJECT,
            Self::Library => sys::nvJitLinkInputType::NVJITLINK_INPUT_LIBRARY,
            Self::Index => sys::nvJitLinkInputType::NVJITLINK_INPUT_INDEX,
            Self::Any => sys::nvJitLinkInputType::NVJITLINK_INPUT_ANY,
        }
    }
}

/// Output produced after linking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    /// Linked CUDA binary.
    Cubin,
    /// PTX after linking; requires link-time optimization.
    Ptx,
    /// Linked LTO-IR in LTOIR container format, rather than raw bitcode.
    /// Requires link-time optimization.
    LtoIr,
}

/// Supported link options.
/// Architecture is required by nvJitLink.
#[derive(Debug, Clone)]
pub struct LinkOptions {
    /// Target architecture; nvJitLink requires `-arch=sm_NN`.
    pub architecture: GpuArchitecture,
    /// Do link time optimization.
    pub lto: bool,
    /// Linked output to retrieve after completion.
    pub output: OutputKind,
    /// Generate line information.
    pub line_info: bool,
    /// Additional nvJitLink options, passed without shell interpretation.
    pub raw_options: Vec<String>,
}

impl LinkOptions {
    /// Creates linker settings for the target architecture, with cubin output, link-time optimization disabled, and no additional options.
    #[must_use]
    pub fn new(architecture: GpuArchitecture) -> Self {
        Self {
            architecture,
            lto: false,
            output: OutputKind::Cubin,
            line_info: false,
            raw_options: Vec::new(),
        }
    }
}

/// Owned linked output and informational diagnostics.
#[derive(Debug)]
pub struct LinkedArtifact {
    /// Format of the owned linked output.
    pub kind: OutputKind,
    bytes: Vec<u8>,
    /// Linker informational diagnostics.
    /// Empty if the native log is unavailable or its retrieval failed.
    pub info_log: String,
}

impl LinkedArtifact {
    /// Borrows the linked output bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consumes the artifact and returns the owned linked output bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Linker handle used during linking operations.
#[derive(Debug)]
pub struct Linker {
    handle: sys::nvJitLinkHandle,
    output: OutputKind,
}

/// Raw ownership of a linker and the output mode it was created with.
///
/// Destroy `handle` with nvJitLink, or re-adopt the value with [`Linker::from_raw`].
#[derive(Debug)]
pub struct RawLinker {
    /// Native linker handle.
    pub handle: sys::nvJitLinkHandle,
    /// Output mode the linker was created with.
    pub output: OutputKind,
}

impl Linker {
    /// Creates an instance of the linker with the given input options.
    ///
    /// # Errors
    ///
    /// Returns an error if PTX or LTO IR output is requested without link-time optimization, the option count does not fit a `u32`, an option contains an interior NUL byte, nvJitLink cannot create the linker, or it returns a null handle.
    pub fn create(options: LinkOptions) -> Result<Self> {
        if options.output != OutputKind::Cubin && !options.lto {
            return Err(Error::out_of_range("output kind"));
        }

        let mut arguments = vec![format!("-arch={}", options.architecture)];
        if options.lto {
            arguments.push("-lto".into());
        }

        if options.output == OutputKind::Ptx {
            arguments.push("-ptx".into());
        }

        if options.line_info {
            arguments.push("-lineinfo".into());
        }

        arguments.extend(options.raw_options);
        let args = arguments
            .into_iter()
            .map(CString::new)
            .collect::<StdResult<Vec<_>, _>>()?;
        let pointers = args.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a local output, and `pointers` holds the counted NUL-terminated options in `args`, which outlive the call.
        let code = unsafe {
            sys::__nvJitLinkCreate_13_4(
                &raw mut handle,
                pointers
                    .len()
                    .try_into()
                    .map_err(|_| Error::out_of_range("linker option count"))?,
                pointers.as_ptr(),
            )
        };
        let linker = Self {
            handle,
            output: options.output,
        };
        linker.check(code)?;
        if linker.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(linker)
    }

    /// Borrows the linker handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvJitLinkHandle {
        self.handle
    }

    /// Takes ownership of a linker created with the specified output mode.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be live and uniquely owned, and its options must match `raw.output`.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taking the raw owner by value transfers ownership of its handle"
    )]
    pub unsafe fn from_raw(raw: RawLinker) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: raw.handle,
            output: raw.output,
        })
    }

    /// Transfers ownership of the linker handle and its output mode to the caller.
    #[must_use]
    pub fn into_raw(mut self) -> RawLinker {
        RawLinker {
            handle: mem::replace(&mut self.handle, ptr::null_mut()),
            output: self.output,
        }
    }

    /// Adds data to the link.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL or the native linker rejects the input image, format, or handle.
    /// Native failure status and linker diagnostics are retained in `Error::NvJitLink`.
    pub fn add_data(&mut self, kind: InputType, bytes: &[u8], name: &str) -> Result<()> {
        let name = CString::new(name)?;
        // SAFETY: `self` owns the live linker exclusively, and `bytes` and the NUL-terminated `name` are borrowed for the call, which copies them.
        self.check(unsafe {
            sys::__nvJitLinkAddData_13_4(
                self.handle,
                kind.raw(),
                bytes.as_ptr().cast(),
                bytes.len(),
                name.as_ptr(),
            )
        })
    }

    /// Adds a file to the link.
    ///
    /// # Errors
    ///
    /// Returns an error if `file` contains an interior NUL or the native linker cannot read or accept the file for the selected input type.
    /// Native failure status and linker diagnostics are retained in `Error::NvJitLink`.
    pub fn add_file(&mut self, kind: InputType, file: &str) -> Result<()> {
        let file = CString::new(file)?;
        // SAFETY: `self` owns the live linker exclusively, and `file` is a NUL-terminated path borrowed for the call.
        self.check(unsafe { sys::__nvJitLinkAddFile_13_4(self.handle, kind.raw(), file.as_ptr()) })
    }

    /// Does the actual link and retrieves the linked output.
    ///
    /// # Errors
    ///
    /// Returns an error if native linking fails or the selected output size or contents cannot be retrieved.
    /// The consumed linker is destroyed on both success and failure; successful output owns its bytes independently.
    pub fn complete(self) -> Result<LinkedArtifact> {
        // SAFETY: `self` owns the live linker, consumed by this call.
        self.check(unsafe { sys::__nvJitLinkComplete_13_4(self.handle) })?;
        let mut size = 0;
        // SAFETY: the linker completed above, and `size` is a local output.
        self.check(unsafe {
            match self.output {
                OutputKind::Cubin => {
                    sys::__nvJitLinkGetLinkedCubinSize_13_4(self.handle, &raw mut size)
                }
                OutputKind::Ptx => {
                    sys::__nvJitLinkGetLinkedPtxSize_13_4(self.handle, &raw mut size)
                }
                OutputKind::LtoIr => {
                    sys::__nvJitLinkGetLinkedLTOIRSize_13_4(self.handle, &raw mut size)
                }
            }
        })?;
        let mut bytes = vec![0; size];
        // SAFETY: `bytes` holds the `size` bytes the linker reported for the same output kind.
        self.check(unsafe {
            match self.output {
                OutputKind::Cubin => {
                    sys::__nvJitLinkGetLinkedCubin_13_4(self.handle, bytes.as_mut_ptr().cast())
                }
                OutputKind::Ptx => {
                    sys::__nvJitLinkGetLinkedPtx_13_4(self.handle, bytes.as_mut_ptr().cast())
                }
                OutputKind::LtoIr => {
                    sys::__nvJitLinkGetLinkedLTOIR_13_4(self.handle, bytes.as_mut_ptr().cast())
                }
            }
        })?;
        Ok(LinkedArtifact {
            kind: self.output,
            bytes,
            info_log: self.log(false),
        })
    }

    /// Returns the informational log generated by the linker.
    #[must_use]
    pub fn info_log(&self) -> String {
        self.log(false)
    }

    fn check(&self, status: sys::nvJitLinkResult) -> Result<()> {
        Error::check_nvjitlink(status).map_err(|error| match error {
            Error::NvJitLink { code, .. } => Error::NvJitLink {
                code,
                diagnostic: self.log(true),
            },
            error => error,
        })
    }

    fn log(&self, error: bool) -> String {
        if self.handle.is_null() {
            return String::new();
        }
        // SAFETY: the handle is non-null and owned by `self`, `size` is a local output, and `bytes` holds the `size` bytes reported for the same log.
        unsafe {
            let mut size = 0;
            let result = if error {
                sys::__nvJitLinkGetErrorLogSize_13_4(self.handle, &raw mut size)
            } else {
                sys::__nvJitLinkGetInfoLogSize_13_4(self.handle, &raw mut size)
            };
            if result != sys::nvJitLinkResult::NVJITLINK_SUCCESS || size == 0 {
                return String::new();
            }

            let mut bytes = vec![0u8; size];
            let result = if error {
                sys::__nvJitLinkGetErrorLog_13_4(self.handle, bytes.as_mut_ptr().cast())
            } else {
                sys::__nvJitLinkGetInfoLog_13_4(self.handle, bytes.as_mut_ptr().cast())
            };
            if result != sys::nvJitLinkResult::NVJITLINK_SUCCESS {
                return String::new();
            }
            String::from_utf8_lossy(&bytes)
                .trim_end_matches('\0')
                .to_owned()
        }
    }
}

impl Drop for Linker {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: this wrapper owns the non-null linker, and nvJitLink nulls the handle field it destroys.
            let status = unsafe { sys::__nvJitLinkDestroy_13_4(&raw mut self.handle) };
            if status != sys::nvJitLinkResult::NVJITLINK_SUCCESS {
                // The linker log is not read through a handle whose destruction failed.
                let error = Error::from(Status::from(status));
                report_cleanup_failure("destroy", "nvjitlink linker", &error);
            }
        }
    }
}

impl_enum_display!(InputType, {
    InputType::Cubin => "NVJITLINK_INPUT_CUBIN",
    InputType::Ptx => "NVJITLINK_INPUT_PTX",
    InputType::LtoIr => "NVJITLINK_INPUT_LTOIR",
    InputType::Fatbin => "NVJITLINK_INPUT_FATBIN",
    InputType::Object => "NVJITLINK_INPUT_OBJECT",
    InputType::Library => "NVJITLINK_INPUT_LIBRARY",
    InputType::Index => "NVJITLINK_INPUT_INDEX",
    InputType::Any => "NVJITLINK_INPUT_ANY",
});
