use std::{
    cell::{RefCell, UnsafeCell},
    ffi::{CStr, CString, c_char, c_void},
    ptr, result,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use singe_core::report_cleanup_failure;
use singe_cuda_sys::nvrtc as sys;

use super::{CompileOptions, Status};
use crate::{
    error::{Error, Result},
    module::ModuleImage,
    try_nvrtc,
    utility::{bytes_to_string, to_i32, to_usize},
};

/// Borrowed CUDA header source and the virtual include name used by NVRTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Header<'a> {
    /// Header source text.
    /// Program builders copy this text when the header is added.
    pub source: &'a str,
    /// Include name by which CUDA source refers to this header.
    pub include_name: &'a str,
}

#[derive(Debug, Clone)]
struct OwnedHeader {
    source: String,
    include_name: String,
}

/// Owned NVRTC program with copied source and virtual headers.
/// The native program is created lazily on the first operation that needs its handle and destroyed when this wrapper is dropped.
/// Builder changes made after materialization do not recreate the existing native program.
#[derive(Debug)]
pub struct Program {
    source: String,
    name: Option<String>,
    headers: Vec<OwnedHeader>,
    handle: UnsafeCell<sys::nvrtcProgram>,
    // The flag a registered cancellation callback reads, kept until the callback is cleared or the program is destroyed.
    cancel_flag: RefCell<Option<Arc<AtomicBool>>>,
}

/// Raw ownership of an NVRTC program, including the cancellation flag its flow callback may still read.
///
/// Keep `cancel_flag` alive until the handle is destroyed or re-adopted with [`Program::from_raw`].
#[derive(Debug)]
pub struct RawProgram {
    /// Native program handle.
    pub handle: sys::nvrtcProgram,
    /// Flag passed to a flow callback that is still registered on the handle, if any.
    pub cancel_flag: Option<Arc<AtomicBool>>,
}

impl Program {
    /// Creates a lazy NVRTC program from CUDA source text.
    ///
    /// The raw NVRTC handle is created when the program is compiled or otherwise materialized.
    #[must_use]
    pub fn from_source(source: &str) -> Self {
        Self {
            source: source.to_string(),
            name: None,
            headers: Vec::new(),
            handle: UnsafeCell::new(ptr::null_mut()),
            cancel_flag: RefCell::new(None),
        }
    }

    /// Copies the source filename used in diagnostics when the native program is first created.
    /// An already materialized program keeps its existing name.
    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Copies a virtual header's source and include name for native program creation.
    /// Add headers before materializing the program.
    #[must_use]
    pub fn with_header(mut self, header: Header<'_>) -> Self {
        self.headers.push(OwnedHeader {
            source: header.source.to_string(),
            include_name: header.include_name.to_string(),
        });
        self
    }

    /// Copies virtual headers in slice order for native program creation.
    /// Add headers before materializing the program.
    #[must_use]
    pub fn with_headers(mut self, headers: &[Header<'_>]) -> Self {
        self.headers
            .extend(headers.iter().map(|header| OwnedHeader {
                source: header.source.to_string(),
                include_name: header.include_name.to_string(),
            }));
        self
    }

    /// Takes ownership of a raw NVRTC program.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid NVRTC program that is not owned by any other wrapper.
    /// Any flow callback registered on it must be NVRTC's default, a callback that reads no payload, or this crate's cancellation callback with `raw.cancel_flag` as its payload.
    /// The returned wrapper destroys the handle on drop.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawProgram) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            source: String::new(),
            name: None,
            headers: Vec::new(),
            handle: UnsafeCell::new(raw.handle),
            cancel_flag: RefCell::new(raw.cancel_flag),
        })
    }

    /// Renders `options` and compiles the program synchronously.
    /// Call [`Self::log`] to inspect compiler diagnostics after a successful or failed compilation.
    ///
    /// With a flag set through [`CompileOptions::cancel_flag()`], another thread can cancel compilation by setting the flag to true.
    /// The wrapper then registers its own native callback and attempts to replace it with a no-op callback after compilation, including failed attempts, and the program keeps a clone of the flag until that replacement succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, an option contains an interior NUL, the option count does not fit an `i32`, or NVRTC rejects an option or fails compilation.
    /// Compilation diagnostics remain available from [`Self::log`].
    /// Errors from registering or clearing the cancellation callback are also returned.
    /// A compilation error takes precedence over a later callback-clear error.
    pub fn compile(&self, options: &CompileOptions<'_>) -> Result<()> {
        let arguments = options.as_arguments();
        let argument_refs = arguments.iter().map(String::as_str).collect::<Vec<_>>();
        let Some(cancel) = &options.cancel_flag else {
            return self.compile_raw(&argument_refs);
        };

        let cancel = Arc::clone(cancel);
        // SAFETY: the program handle is live, and the payload points into `cancel`, which the program keeps below for as long as NVRTC can call the callback with it.
        unsafe {
            try_nvrtc!(sys::nvrtcSetFlowCallback(
                self.handle()?,
                Some(cancel_if_requested_callback),
                Arc::as_ptr(&cancel).cast_mut().cast(),
            ))?;
        }

        // NVRTC no longer references a previously kept flag, so dropping it here is sound.
        *self.cancel_flag.borrow_mut() = Some(cancel);

        let compile_result = self.compile_raw(&argument_refs);
        let clear_result = self.clear_flow_callback();

        match (compile_result, clear_result) {
            (Err(error), _) | (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    /// Registers a name expression for a `__global__`, `__device__`, or `__constant__` symbol.
    ///
    /// The identical name expression string must be provided to [`Program::lowered_name`] after compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if `name_expression` contains an interior NUL byte, if the program handle is invalid, or if NVRTC rejects the expression.
    pub fn add_name_expression(&self, name_expression: &str) -> Result<()> {
        let name_expression = CString::new(name_expression)?;
        // SAFETY: `self` owns the live program, and `name_expression` is a NUL-terminated string borrowed for the call, which copies it.
        unsafe {
            try_nvrtc!(sys::nvrtcAddNameExpression(
                self.handle()?,
                name_expression.as_ptr(),
            ))
        }
    }

    /// Returns the lowered (mangled) name for a registered name expression.
    ///
    /// The identical name expression must have been previously provided to [`Program::add_name_expression`].
    ///
    /// # Errors
    ///
    /// Returns an error if `name_expression` contains an interior NUL byte, if the program handle is invalid, or if NVRTC cannot return the lowered name.
    pub fn lowered_name(&self, name_expression: &str) -> Result<String> {
        let name_expression = CString::new(name_expression)?;
        let mut lowered_name = ptr::null();
        // SAFETY: `self` owns the live program, the name is NUL-terminated, and on success NVRTC points `lowered_name` at a NUL-terminated string the program owns, copied out before `self` can drop.
        unsafe {
            try_nvrtc!(sys::nvrtcGetLoweredName(
                self.handle()?,
                name_expression.as_ptr(),
                &raw mut lowered_name,
            ))?;
            Ok(CStr::from_ptr(lowered_name).to_string_lossy().into_owned())
        }
    }

    /// Returns the PTX bytes produced by the last compilation, including the native trailing NUL byte.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`.
    /// The requested code format must have been generated by the preceding compilation.
    pub fn ptx(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetPTXSize, sys::nvrtcGetPTX)
    }

    /// Copies the last compilation's PTX bytes into an owned module image.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, the native output size cannot be represented as `usize`, or the output is empty.
    /// The requested code format must have been generated by the preceding compilation.
    pub fn ptx_image(&self) -> Result<ModuleImage<'static>> {
        ModuleImage::from_compiler_output(self.ptx()?)
    }

    /// Returns the last compilation's PTX as a string, removing its trailing NUL and replacing invalid UTF-8 lossily.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`.
    /// The requested code format must have been generated by the preceding compilation.
    pub fn ptx_string(&self) -> Result<String> {
        Ok(bytes_to_string(self.ptx()?))
    }

    /// Returns the GPU machine-code binary produced by the last compilation for a real GPU architecture target.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`.
    /// The requested code format must have been generated by the preceding compilation.
    pub fn cubin(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetCUBINSize, sys::nvrtcGetCUBIN)
    }

    /// Copies the last compilation's GPU machine-code binary into an owned module image.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, the native output size cannot be represented as `usize`, or the output is empty.
    /// The requested code format must have been generated by the preceding compilation.
    pub fn cubin_image(&self) -> Result<ModuleImage<'static>> {
        ModuleImage::from_compiler_output(self.cubin()?)
    }

    /// Returns the `cuda_tile` IR generated by the previous compilation of this program.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`.
    /// The requested code format must have been generated by the preceding compilation.
    pub fn tile_ir(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetTileIRSize, sys::nvrtcGetTileIR)
    }

    /// Returns the device link-time intermediate representation produced by the last compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`.
    /// The requested code format must have been generated by the preceding compilation.
    pub fn lto_ir(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetLTOIRSize, sys::nvrtcGetLTOIR)
    }

    /// Returns the `OptiX` intermediate representation produced by the last compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`.
    /// The requested code format must have been generated by the preceding compilation.
    pub fn optix_ir(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetOptiXIRSize, sys::nvrtcGetOptiXIR)
    }

    /// Returns the program compilation log, removing its trailing NUL and replacing invalid UTF-8 lossily.
    /// The log may contain diagnostics from either a successful or failed compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the log, or the native log size cannot be represented as `usize`.
    pub fn log(&self) -> Result<String> {
        Ok(bytes_to_string(self.bytes(
            sys::nvrtcGetProgramLogSize,
            sys::nvrtcGetProgramLog,
        )?))
    }

    /// Returns the PCH creation status.
    ///
    /// [`Status::Success`] indicates that the PCH was successfully created.
    /// [`Status::NoPchCreateAttempted`] indicates that no PCH creation was attempted, either because PCH was not requested during the preceding compile call, or automatic PCH processing was requested, and the compiler chose not to create a PCH file.
    /// [`Status::PchCreateHeapExhausted`] indicates that a PCH file could potentially have been created, but the compiler ran out space in the PCH heap.
    /// In this scenario, use [`Program::pch_heap_size_required`] to query the required heap size, reallocate the heap with [`set_pch_heap_size`](super::set_pch_heap_size), and retry PCH creation by compiling a new [`Program`].
    /// [`Status::PchCreate`] indicates that an error condition prevented the PCH file from being created.
    ///
    /// # Errors
    ///
    /// Returns an error if the program handle is invalid.
    pub fn pch_create_status(&self) -> Result<Status> {
        // SAFETY: `self` owns the live program.
        unsafe { Ok(sys::nvrtcGetPCHCreateStatus(self.handle()?).into()) }
    }

    /// Returns the PCH heap size required to compile the given program.
    ///
    /// # Errors
    ///
    /// Returns an error if the program handle is invalid or if the returned size is not valid because [`Program::pch_create_status`] did not return [`Status::Success`] or [`Status::PchCreateHeapExhausted`].
    pub fn pch_heap_size_required(&self) -> Result<usize> {
        let mut size = 0;
        // SAFETY: `self` owns the live program, and `size` is a local output.
        unsafe {
            try_nvrtc!(sys::nvrtcGetPCHHeapSizeRequired(
                self.handle()?,
                &raw mut size
            ))?;
        }
        to_usize(size, "NVRTC output byte count")
    }

    /// Returns the raw NVRTC program handle, creating it first if needed.
    ///
    /// If the program has not been materialized yet, this creates the raw handle first.
    ///
    /// # Errors
    ///
    /// Returns an error if NVRTC cannot create the program handle.
    pub fn materialize_raw(&self) -> Result<sys::nvrtcProgram> {
        self.handle()
    }

    /// Returns the raw NVRTC program handle only if it has already been materialized.
    pub const fn as_raw_if_materialized(&self) -> Option<sys::nvrtcProgram> {
        // SAFETY: `Program` is not `Sync`, and no `&mut` to the cell is live while this shared borrow exists.
        let handle = unsafe { *self.handle.get() };
        if handle.is_null() { None } else { Some(handle) }
    }

    /// Transfers ownership of the raw NVRTC program to the caller.
    ///
    /// If the program has not been materialized yet, this creates the raw handle first.
    /// The host copies of the source and headers are dropped, since NVRTC keeps its own.
    /// The caller becomes responsible for destroying the returned handle.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails because source, filenames, or headers contain an interior NUL, the header count exceeds the native integer range, or NVRTC cannot create the program.
    pub fn into_raw(mut self) -> Result<RawProgram> {
        let handle = self.handle()?;
        // A null handle makes `Drop` skip destruction, while the host strings drop normally.
        *self.handle.get_mut() = ptr::null_mut();
        Ok(RawProgram {
            handle,
            cancel_flag: self.cancel_flag.get_mut().take(),
        })
    }

    fn compile_raw(&self, options: &[&str]) -> Result<()> {
        let options = options
            .iter()
            .map(|option| CString::new(*option))
            .collect::<result::Result<Vec<_>, _>>()?;
        let option_ptrs = options
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();

        // SAFETY: `self` owns the live program, and `option_ptrs` holds the counted NUL-terminated options in `options`, which outlive the call.
        unsafe {
            try_nvrtc!(sys::nvrtcCompileProgram(
                self.handle()?,
                to_i32(option_ptrs.len(), "NVRTC option count")?,
                if option_ptrs.is_empty() {
                    ptr::null()
                } else {
                    option_ptrs.as_ptr()
                },
            ))
        }
    }

    fn bytes(
        &self,
        get_size: unsafe extern "C" fn(sys::nvrtcProgram, *mut sys::size_t) -> sys::nvrtcResult,
        get_data: unsafe extern "C" fn(sys::nvrtcProgram, *mut c_char) -> sys::nvrtcResult,
    ) -> Result<Vec<u8>> {
        let mut size = 0;
        // SAFETY: `self` owns the live program, and `size` is a local output.
        unsafe {
            try_nvrtc!(get_size(self.handle()?, &raw mut size))?;
        }

        let mut bytes = vec![0u8; to_usize(size, "NVRTC output byte count")?];
        if bytes.is_empty() {
            return Ok(bytes);
        }

        // SAFETY: `bytes` holds the `size` bytes NVRTC reported for the same output of the live program.
        unsafe {
            try_nvrtc!(get_data(self.handle()?, bytes.as_mut_ptr().cast()))?;
        }
        Ok(bytes)
    }

    fn handle(&self) -> Result<sys::nvrtcProgram> {
        // SAFETY: `Program` is not `Sync`, and no other reference into the cell is live during lazy creation.
        unsafe {
            let handle = self.handle.get();
            if (*handle).is_null() {
                *handle = self.create_handle()?;
            }
            Ok(*handle)
        }
    }

    fn create_handle(&self) -> Result<sys::nvrtcProgram> {
        let source = CString::new(self.source.as_str())?;
        let name = self.name.as_deref().map(CString::new).transpose()?;
        let header_sources = self
            .headers
            .iter()
            .map(|header| CString::new(header.source.as_str()))
            .collect::<result::Result<Vec<_>, _>>()?;
        let include_names = self
            .headers
            .iter()
            .map(|header| CString::new(header.include_name.as_str()))
            .collect::<result::Result<Vec<_>, _>>()?;
        let header_ptrs = header_sources
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        let include_name_ptrs = include_names
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();

        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a local output, and the source, name, and the counted header and include-name arrays point at NUL-terminated strings that outlive the call.
        unsafe {
            try_nvrtc!(sys::nvrtcCreateProgram(
                &raw mut handle,
                source.as_ptr(),
                name.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
                to_i32(self.headers.len(), "NVRTC header count")?,
                if header_ptrs.is_empty() {
                    ptr::null()
                } else {
                    header_ptrs.as_ptr()
                },
                if include_name_ptrs.is_empty() {
                    ptr::null()
                } else {
                    include_name_ptrs.as_ptr()
                },
            ))?;
        }

        Ok(handle)
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: `Program` is not `Sync`, so `&mut self` is the only access to the cell, and a non-null handle is owned by this wrapper.
        unsafe {
            let handle = self.handle.get();
            if !(*handle).is_null()
                && let Err(error) = try_nvrtc!(sys::nvrtcDestroyProgram(handle))
            {
                report_cleanup_failure("destroy", "nvrtc program", &error);
            }
        }
    }
}

/// Owned NVRTC compilation output tagged with its code format.
#[derive(Debug)]
#[non_exhaustive]
pub enum CompilationArtifact {
    /// PTX assembly, requiring driver JIT compilation before execution.
    Ptx(
        /// Owned output image bytes.
        ModuleImage<'static>,
    ),
    /// GPU machine code for a concrete GPU architecture.
    Cubin(
        /// Owned output image bytes.
        ModuleImage<'static>,
    ),
    /// Device link-time intermediate representation, requiring linking before loading.
    LtoIr(
        /// Owned output bytes.
        Vec<u8>,
    ),
    /// `OptiX` intermediate representation, which CUDA cannot load.
    OptixIr(
        /// Owned output bytes.
        Vec<u8>,
    ),
    /// The `cuda_tile` IR generated by the previous compilation.
    TileIr(
        /// Owned output image bytes.
        ModuleImage<'static>,
    ),
}

impl CompilationArtifact {
    /// Borrows the output bytes regardless of their compilation format.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Ptx(image) | Self::Cubin(image) | Self::TileIr(image) => image.as_bytes(),
            Self::LtoIr(bytes) | Self::OptixIr(bytes) => bytes.as_slice(),
        }
    }

    /// Consumes the artifact and returns its module image, or `Err(self)` for LTO IR and `OptiX` IR, which CUDA cannot load.
    ///
    /// # Errors
    ///
    /// Returns the artifact itself if it is LTO IR or `OptiX` IR.
    pub fn into_image(self) -> result::Result<ModuleImage<'static>, Self> {
        match self {
            Self::Ptx(image) | Self::Cubin(image) | Self::TileIr(image) => Ok(image),
            Self::LtoIr(_) | Self::OptixIr(_) => Err(self),
        }
    }
}

/// Compilation output format requested from an NVRTC program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutputKind {
    /// PTX assembly, requiring driver JIT compilation before execution.
    Ptx,
    /// GPU machine code for a concrete GPU architecture.
    Cubin,
    /// Device link-time intermediate representation, requiring linking before loading.
    LtoIr,
    /// `OptiX` intermediate representation.
    OptixIr,
    /// The `cuda_tile` IR generated by the previous compilation.
    TileIr,
}

impl Program {
    /// Returns the requested compilation output as an owned, format-tagged artifact.
    ///
    /// # Errors
    ///
    /// Returns an error if the selected output cannot be retrieved or is empty.
    pub fn artifact(&self, kind: OutputKind) -> Result<CompilationArtifact> {
        fn nonempty_output(bytes: Vec<u8>) -> Result<Vec<u8>> {
            if bytes.is_empty() {
                return Err(Error::empty("nvrtc output"));
            }
            Ok(bytes)
        }

        Ok(match kind {
            OutputKind::Ptx => CompilationArtifact::Ptx(self.ptx_image()?),
            OutputKind::Cubin => CompilationArtifact::Cubin(self.cubin_image()?),
            OutputKind::LtoIr => CompilationArtifact::LtoIr(nonempty_output(self.lto_ir()?)?),
            OutputKind::OptixIr => CompilationArtifact::OptixIr(nonempty_output(self.optix_ir()?)?),
            OutputKind::TileIr => {
                CompilationArtifact::TileIr(ModuleImage::from_compiler_output(self.tile_ir()?)?)
            }
        })
    }
}

unsafe extern "C" fn noop_compile_callback(payload: *mut c_void, reserved: *mut c_void) -> i32 {
    let _ = payload;
    let _ = reserved;
    0
}

unsafe extern "C" fn cancel_if_requested_callback(
    payload: *mut c_void,
    reserved: *mut c_void,
) -> i32 {
    let _ = reserved;
    if payload.is_null() {
        return 0;
    }

    // SAFETY: the payload is the cancellation flag registered with this callback, which its program keeps alive while the callback is registered.
    let cancel = unsafe { &*payload.cast::<AtomicBool>() };
    i32::from(cancel.load(Ordering::Relaxed))
}

impl Program {
    // Replaces a cancellation callback with a no-op callback, then releases the flag it read.
    pub(crate) fn clear_flow_callback(&self) -> Result<()> {
        // SAFETY: the program handle is live, and the no-op callback reads no payload.
        unsafe {
            try_nvrtc!(sys::nvrtcSetFlowCallback(
                self.handle()?,
                Some(noop_compile_callback),
                ptr::null_mut(),
            ))?;
        }

        *self.cancel_flag.borrow_mut() = None;
        Ok(())
    }
}
