//! Runtime linking of PTX, cubin and link-time optimization IR.

use std::{ffi::CString, mem, ptr, result::Result as StdResult};

use singe_core::{impl_enum_display, status_enum};
use singe_cuda_sys::nvjitlink as sys;

use crate::{
    architecture::GpuArchitecture,
    error::{Error, Result},
};

status_enum! {
    /// API call result codes returned by nvJitLink.
    pub enum Status: u32 => sys::nvJitLinkResult, unknown = "UNKNOWN_NVJITLINK_STATUS" {
        /// Successful completion.
        Success = NVJITLINK_SUCCESS,
        /// Unrecognized Option
        UnrecognizedOption = NVJITLINK_ERROR_UNRECOGNIZED_OPTION,
        /// Option `-arch=sm_NN` not specified
        MissingArch = NVJITLINK_ERROR_MISSING_ARCH,
        /// Invalid Input
        InvalidInput = NVJITLINK_ERROR_INVALID_INPUT,
        /// Issue during PTX Compilation
        PtxCompile = NVJITLINK_ERROR_PTX_COMPILE,
        /// Issue during NVVM Compilation
        NvvmCompile = NVJITLINK_ERROR_NVVM_COMPILE,
        /// Internal Error
        Internal = NVJITLINK_ERROR_INTERNAL,
        /// Issue with Thread Pool
        Threadpool = NVJITLINK_ERROR_THREADPOOL,
        /// Unrecognized Input
        UnrecognizedInput = NVJITLINK_ERROR_UNRECOGNIZED_INPUT,
        /// Finalizer Error
        Finalize = NVJITLINK_ERROR_FINALIZE,
        /// Null Input
        NullInput = NVJITLINK_ERROR_NULL_INPUT,
        /// Incompatible Options
        IncompatibleOptions = NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS,
        /// Incorrect Input Type
        IncorrectInputType = NVJITLINK_ERROR_INCORRECT_INPUT_TYPE,
        /// Arch Mismatch
        ArchMismatch = NVJITLINK_ERROR_ARCH_MISMATCH,
        /// Outdated Library
        OutdatedLibrary = NVJITLINK_ERROR_OUTDATED_LIBRARY,
        /// Missing Fatbin
        MissingFatbin = NVJITLINK_ERROR_MISSING_FATBIN,
        /// Unrecognized -arch value
        UnrecognizedArch = NVJITLINK_ERROR_UNRECOGNIZED_ARCH,
        /// Unsupported -arch value
        UnsupportedArch = NVJITLINK_ERROR_UNSUPPORTED_ARCH,
        /// Requires -lto
        LtoNotEnabled = NVJITLINK_ERROR_LTO_NOT_ENABLED,
    }
}

impl Status {
    /// Static description of this status.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::UnrecognizedOption => "unrecognized option",
            Self::MissingArch => "missing arch",
            Self::InvalidInput => "invalid input",
            Self::PtxCompile => "ptx compile",
            Self::NvvmCompile => "nvvm compile",
            Self::Internal => "internal",
            Self::Threadpool => "threadpool",
            Self::UnrecognizedInput => "unrecognized input",
            Self::Finalize => "finalize",
            Self::NullInput => "null input",
            Self::IncompatibleOptions => "incompatible options",
            Self::IncorrectInputType => "incorrect input type",
            Self::ArchMismatch => "arch mismatch",
            Self::OutdatedLibrary => "outdated library",
            Self::MissingFatbin => "missing fatbin",
            Self::UnrecognizedArch => "unrecognized arch",
            Self::UnsupportedArch => "unsupported arch",
            Self::LtoNotEnabled => "lto not enabled",
            Self::Unknown(_) => "unknown nvjitlink error",
        }
    }
}

/// The kind of inputs that can be passed to the linker.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
#[non_exhaustive]
pub enum InputType {
    /// CUDA binaries.
    Cubin = sys::nvJitLinkInputType::NVJITLINK_INPUT_CUBIN as _,
    /// PTX.
    Ptx = sys::nvJitLinkInputType::NVJITLINK_INPUT_PTX as _,
    /// LTO-IR.
    LtoIr = sys::nvJitLinkInputType::NVJITLINK_INPUT_LTOIR as _,
    /// Fatbin.
    Fatbin = sys::nvJitLinkInputType::NVJITLINK_INPUT_FATBIN as _,
    /// Host object.
    Object = sys::nvJitLinkInputType::NVJITLINK_INPUT_OBJECT as _,
    /// Host library.
    Library = sys::nvJitLinkInputType::NVJITLINK_INPUT_LIBRARY as _,
    /// Index file.
    Index = sys::nvJitLinkInputType::NVJITLINK_INPUT_INDEX as _,
    /// Dynamically chooses from the valid types.
    Any = sys::nvJitLinkInputType::NVJITLINK_INPUT_ANY as _,
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

/// Supported link options. Architecture is required by nvJitLink.
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
    /// Linker informational diagnostics. Empty if the native log is unavailable or its retrieval failed.
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

impl Linker {
    /// Creates an instance of the linker with the given input options.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if PTX or LTO IR output is requested without link-time optimization, or the option count exceeds the native integer range. Also returns an error for interior NUL bytes in option strings, a native linker initialization failure, or a null returned handle.
    pub fn create(options: LinkOptions) -> Result<Self> {
        if options.output != OutputKind::Cubin && !options.lto {
            return Err(Error::InvalidValue);
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
        // SAFETY: `__nvJitLinkCreate_13_4` receives valid output locations and pointers into buffers borrowed for the call.
        let code = unsafe {
            sys::__nvJitLinkCreate_13_4(
                &raw mut handle,
                pointers.len().try_into().map_err(|_| Error::InvalidValue)?,
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
    /// The handle must be live and uniquely owned, and its options must match `output`.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `handle` is null.
    pub unsafe fn from_raw(handle: sys::nvJitLinkHandle, output: OutputKind) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle, output })
    }

    /// Transfers ownership of the linker handle to the caller.
    #[must_use]
    pub fn into_raw(mut self) -> sys::nvJitLinkHandle {
        mem::replace(&mut self.handle, ptr::null_mut())
    }

    /// Adds data to the link.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL or the native linker rejects the input image, format, or handle. Native failure status and linker diagnostics are retained in `Error::NvJitLink`.
    pub fn add_data(&mut self, kind: InputType, bytes: &[u8], name: &str) -> Result<()> {
        let name = CString::new(name)?;
        // SAFETY: `__nvJitLinkAddData_13_4` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
    /// Returns an error if `file` contains an interior NUL or the native linker cannot read or accept the file for the selected input type. Native failure status and linker diagnostics are retained in `Error::NvJitLink`.
    pub fn add_file(&mut self, kind: InputType, file: &str) -> Result<()> {
        let file = CString::new(file)?;
        // SAFETY: `__nvJitLinkAddFile_13_4` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        self.check(unsafe { sys::__nvJitLinkAddFile_13_4(self.handle, kind.raw(), file.as_ptr()) })
    }

    /// Does the actual link and retrieves the linked output.
    ///
    /// # Errors
    ///
    /// Returns an error if native linking fails or the selected output size or contents cannot be retrieved. The consumed linker is destroyed on both success and failure; successful output owns its bytes independently.
    pub fn complete(self) -> Result<LinkedArtifact> {
        // SAFETY: `__nvJitLinkComplete_13_4` receives live handles owned by the wrappers.
        self.check(unsafe { sys::__nvJitLinkComplete_13_4(self.handle) })?;
        let mut size = 0;
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
        // SAFETY: `__nvJitLinkGetLinkedCubin_13_4`, `__nvJitLinkGetLinkedPtx_13_4`, `__nvJitLinkGetLinkedLTOIR_13_4` receive live handles owned by the wrappers and valid output locations.
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
        if status == sys::nvJitLinkResult::NVJITLINK_SUCCESS {
            Ok(())
        } else {
            Err(Error::NvJitLink {
                code: Status::from(status as u32),
                message: self.log(true),
            })
        }
    }

    fn log(&self, error: bool) -> String {
        if self.handle.is_null() {
            return String::new();
        }
        // SAFETY: `__nvJitLinkGetErrorLogSize_13_4`, `__nvJitLinkGetInfoLogSize_13_4`, `__nvJitLinkGetErrorLog_13_4`, `__nvJitLinkGetInfoLog_13_4` receive live handles owned by the wrappers and valid output locations.
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
            // SAFETY: `__nvJitLinkDestroy_13_4` receives live handles owned by the wrappers and valid output locations.
            unsafe {
                let status = sys::__nvJitLinkDestroy_13_4(&raw mut self.handle);
                if status != sys::nvJitLinkResult::NVJITLINK_SUCCESS {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "failed to destroy nvjitlink linker: {}",
                        Status::from(status as u32)
                    );
                }
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
