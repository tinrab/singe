use std::{ffi::CString, mem, ptr, result};

use singe_core::{impl_enum_display, static_str_from_c_ptr, status_enum};
use singe_cuda_sys::nvvm as sys;

use crate::{
    architecture::GpuArchitecture,
    error::{Error, Result},
    module::ModuleImage,
    try_nvvm,
    utility::{to_i32, to_usize},
};

// TODO: move to a core version type
/// Major and minor version of the loaded NVVM compiler library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    /// Major NVVM library version component.
    pub major: i32,
    /// Minor NVVM library version component.
    pub minor: i32,
}

/// NVVM IR and debug metadata versions supported by the loaded compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IrVersion {
    /// Major supported NVVM IR version.
    pub major: i32,
    /// Minor supported NVVM IR version.
    pub minor: i32,
    /// Major supported debug metadata version.
    pub debug_major: i32,
    /// Minor supported debug metadata version.
    pub debug_minor: i32,
}

status_enum! {
    /// NVVM result code returned by compiler operations.
    pub enum Status: u32 => sys::nvvmResult, unknown = "UNKNOWN_NVVM_STATUS" {
        /// The operation completed successfully.
        Success = NVVM_SUCCESS,
        /// NVVM could not allocate the required memory.
        OutOfMemory = NVVM_ERROR_OUT_OF_MEMORY,
        /// NVVM could not create the program.
        ProgramCreationFailure = NVVM_ERROR_PROGRAM_CREATION_FAILURE,
        /// The input IR version is incompatible with the NVVM library.
        IrVersionMismatch = NVVM_ERROR_IR_VERSION_MISMATCH,
        /// An input argument is invalid.
        InvalidInput = NVVM_ERROR_INVALID_INPUT,
        /// The program handle is invalid.
        InvalidProgram = NVVM_ERROR_INVALID_PROGRAM,
        /// The input NVVM IR is invalid.
        InvalidIr = NVVM_ERROR_INVALID_IR,
        /// A compilation option is invalid.
        InvalidOption = NVVM_ERROR_INVALID_OPTION,
        /// The program contains no IR modules.
        NoModuleInProgram = NVVM_ERROR_NO_MODULE_IN_PROGRAM,
        /// NVVM compilation failed; the program log contains diagnostics.
        Compilation = NVVM_ERROR_COMPILATION,
        /// The operation was cancelled.
        Cancelled = NVVM_ERROR_CANCELLED,
    }
}

impl Status {
    /// Library description of this status.
    #[must_use]
    pub fn description(self) -> &'static str {
        let Ok(raw) = sys::nvvmResult::try_from(self) else {
            return "unknown nvvm error";
        };
        // SAFETY: `nvvmGetErrorString` returns null or a static NUL-terminated string.
        unsafe { static_str_from_c_ptr(sys::nvvmGetErrorString(raw), "unknown nvvm error") }
    }
}

/// Native optimization level for NVVM IR compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OptimizationLevel {
    /// Disables compiler optimizations.
    Zero,
    /// Enables the native default optimization level.
    Three,
}

impl_enum_display!(OptimizationLevel, {
    Self::Zero => "0",
    Self::Three => "3",
});

/// Typed NVVM compiler options. Unset optional values leave native defaults in effect; raw options are appended after typed options. Typed compilation and verification reject concrete GPU architecture targets.
#[derive(Debug, Clone, Default)]
pub struct CompileOptions<'a> {
    /// Whether to emit full device debug information. Native full debug support requires optimization level zero and IR debug metadata.
    pub device_debug: bool,
    /// Optimization effort applied to the input IR.
    pub optimization_level: Option<OptimizationLevel>,
    /// Virtual GPU architecture target. Typed compilation and verification reject a concrete architecture.
    pub gpu_architecture: Option<GpuArchitecture>,
    /// Whether single-precision denormal values are flushed to zero.
    pub flush_to_zero: Option<bool>,
    /// Whether single-precision square roots use the precise native implementation.
    pub precise_square_root: Option<bool>,
    /// Whether single-precision division uses the precise native implementation.
    pub precise_division: Option<bool>,
    /// Whether floating-point multiply and add operations may be contracted.
    pub fma: Option<bool>,
    /// Switch case-density threshold, in percent, from zero through 101. The builder clamps larger values to 101.
    pub jump_table_density: Option<u8>,
    /// Whether to generate link-time intermediate representation instead of PTX.
    pub generate_lto: bool,
    /// Additional native options appended after the typed options.
    pub raw_options: Vec<&'a str>,
}

impl<'a> CompileOptions<'a> {
    /// Creates an empty compiler option set that leaves native defaults in effect.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            device_debug: false,
            optimization_level: None,
            gpu_architecture: None,
            flush_to_zero: None,
            precise_square_root: None,
            precise_division: None,
            fma: None,
            jump_table_density: None,
            generate_lto: false,
            raw_options: Vec::new(),
        }
    }

    /// Enables or disables full device debug information. Native full debug support requires optimization level zero and IR debug metadata.
    #[must_use]
    pub fn device_debug(mut self, value: bool) -> Self {
        self.device_debug = value;
        self
    }

    /// Sets the native optimization level.
    #[must_use]
    pub fn optimization_level(mut self, value: OptimizationLevel) -> Self {
        self.optimization_level = Some(value);
        self
    }

    /// Sets the virtual GPU architecture target. Concrete targets are rejected when typed options are compiled or verified.
    #[must_use]
    pub fn gpu_architecture(mut self, value: GpuArchitecture) -> Self {
        self.gpu_architecture = Some(value);
        self
    }

    /// Controls whether single-precision denormal values are flushed to zero.
    #[must_use]
    pub fn flush_to_zero(mut self, value: bool) -> Self {
        self.flush_to_zero = Some(value);
        self
    }

    /// Selects precise or approximate single-precision square roots.
    #[must_use]
    pub fn precise_square_root(mut self, value: bool) -> Self {
        self.precise_square_root = Some(value);
        self
    }

    /// Selects precise or approximate single-precision division.
    #[must_use]
    pub fn precise_division(mut self, value: bool) -> Self {
        self.precise_division = Some(value);
        self
    }

    /// Controls contraction of floating-point multiply and add operations.
    #[must_use]
    pub fn fma(mut self, value: bool) -> Self {
        self.fma = Some(value);
        self
    }

    /// Sets the switch case-density percentage threshold, clamping values above 101 to 101.
    #[must_use]
    pub fn jump_table_density(mut self, value: u8) -> Self {
        self.jump_table_density = Some(value.min(101));
        self
    }

    /// Selects link-time intermediate representation generation instead of PTX.
    #[must_use]
    pub fn generate_lto(mut self, value: bool) -> Self {
        self.generate_lto = value;
        self
    }

    /// Appends a native option string after the typed options.
    #[must_use]
    pub fn raw_option(mut self, value: &'a str) -> Self {
        self.raw_options.push(value);
        self
    }

    /// Renders compiler settings as owned native option strings without validating or compiling them.
    #[must_use]
    pub fn as_arguments(&self) -> Vec<String> {
        let mut arguments = Vec::new();
        if self.device_debug {
            arguments.push(String::from("-g"));
        }
        if let Some(value) = self.optimization_level {
            arguments.push(format!("-opt={value}"));
        }
        if let Some(value) = self.gpu_architecture {
            arguments.push(format!("-arch={value}"));
        }
        if let Some(value) = self.flush_to_zero {
            arguments.push(format!("-ftz={}", flag_bit(value)));
        }
        if let Some(value) = self.precise_square_root {
            arguments.push(format!("-prec-sqrt={}", flag_bit(value)));
        }
        if let Some(value) = self.precise_division {
            arguments.push(format!("-prec-div={}", flag_bit(value)));
        }
        if let Some(value) = self.fma {
            arguments.push(format!("-fma={}", flag_bit(value)));
        }
        if let Some(value) = self.jump_table_density {
            arguments.push(format!("-jump-table-density={value}"));
        }
        if self.generate_lto {
            arguments.push(String::from("-gen-lto"));
        }
        arguments.extend(self.raw_options.iter().map(|value| (*value).to_string()));
        arguments
    }

    fn validate(&self) -> Result<()> {
        if let Some(architecture) = self.gpu_architecture
            && !architecture.is_virtual()
        {
            return Err(Error::InvalidValue);
        }
        Ok(())
    }
}

/// Borrowed NVVM IR bytes and a module name for addition to a compiler program.
#[derive(Debug, Clone, Copy)]
pub struct Module<'a> {
    /// NVVM IR bytes in the representation accepted by the selected module-addition API.
    pub ir: &'a [u8],
    /// Module name used in compiler diagnostics. Interior NUL bytes are rejected when adding it.
    pub name: &'a str,
}

/// Owned NVVM program containing IR modules to link, verify, and compile. The native program is destroyed when this wrapper is dropped.
#[derive(Debug)]
pub struct Program {
    handle: sys::nvvmProgram,
}

impl Program {
    /// Creates an empty native NVVM compiler program.
    ///
    /// # Errors
    ///
    /// Returns an error if NVVM cannot create the program or returns a null handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `nvvmCreateProgram` receives valid output locations.
        unsafe {
            try_nvvm!(sys::nvvmCreateProgram(&raw mut handle))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
    }

    /// Takes ownership of a raw NVVM program handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `nvvmProgram` that is not owned by any other
    /// wrapper. The returned wrapper destroys it with `nvvmDestroyProgram`.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `handle` is null.
    pub unsafe fn from_raw(handle: sys::nvvmProgram) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle })
    }

    /// Adds an NVVM IR module in text or bitcode form to the program.
    ///
    /// # Errors
    ///
    /// Returns an error if the IR byte slice is empty, the module name contains an interior NUL, or NVVM rejects the module or program.
    pub fn add_module(&mut self, module: Module<'_>) -> Result<()> {
        self.add_module_raw(&module, false)
    }

    /// Adds a bitcode module whose symbols are loaded only as required by eagerly added modules. A program must contain at least one eagerly added module.
    ///
    /// # Errors
    ///
    /// Returns an error if the IR byte slice is empty, the module name contains an interior NUL, or NVVM rejects the module or program.
    pub fn lazy_add_module(&mut self, module: Module<'_>) -> Result<()> {
        self.add_module_raw(&module, true)
    }

    /// Links the added IR modules and compiles the result using native option strings. Output is PTX unless link-time IR generation is requested.
    ///
    /// # Errors
    ///
    /// Returns an error if an option contains an interior NUL, the option count exceeds the native integer range, or NVVM rejects the program or compilation options or fails compilation. Compiler diagnostics are available from `log`.
    pub fn compile(&self, options: &[&str]) -> Result<()> {
        self.compile_raw(sys::nvvmCompileProgram, options)
    }

    /// Validates typed options, links the IR modules, and compiles the program.
    ///
    /// # Errors
    ///
    /// Returns an error if an option contains an interior NUL, the option count exceeds the native integer range, or NVVM rejects the program or compilation options or fails compilation. Compiler diagnostics are available from `log`. Also rejects a concrete GPU target in `options.gpu_architecture`.
    pub fn compile_with_options(&self, options: &CompileOptions<'_>) -> Result<()> {
        options.validate()?;
        let arguments = options.as_arguments();
        let argument_refs = arguments.iter().map(String::as_str).collect::<Vec<_>>();
        self.compile(&argument_refs)
    }

    /// Verifies the added NVVM IR modules using native option strings.
    ///
    /// # Errors
    ///
    /// Returns an error if an option contains an interior NUL, the option count exceeds the native integer range, or NVVM rejects the options or finds invalid IR. Verification diagnostics are available from `log`.
    pub fn verify(&self, options: &[&str]) -> Result<()> {
        self.compile_raw(sys::nvvmVerifyProgram, options)
    }

    /// Validates typed options and verifies the added NVVM IR modules.
    ///
    /// # Errors
    ///
    /// Returns an error if an option contains an interior NUL, the option count exceeds the native integer range, or NVVM rejects the options or finds invalid IR. Verification diagnostics are available from `log`. Also rejects a concrete GPU target in `options.gpu_architecture`.
    pub fn verify_with_options(&self, options: &CompileOptions<'_>) -> Result<()> {
        options.validate()?;
        let arguments = options.as_arguments();
        let argument_refs = arguments.iter().map(String::as_str).collect::<Vec<_>>();
        self.verify(&argument_refs)
    }

    /// Returns owned bytes of the last compiled PTX or link-time IR output. Native PTX output includes its trailing NUL byte.
    ///
    /// # Errors
    ///
    /// Returns an error if NVVM cannot query or copy the requested output, or the returned byte count cannot be represented as `usize`.
    pub fn compiled_result(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvvmGetCompiledResultSize, sys::nvvmGetCompiledResult)
    }

    /// Copies the last compiled output into an owned module image.
    ///
    /// # Errors
    ///
    /// Returns an error if NVVM cannot query or copy the requested output, or the returned byte count cannot be represented as `usize`.
    pub fn compiled_image(&self) -> Result<ModuleImage<'static>> {
        Ok(ModuleImage::from_vec(self.compiled_result()?))
    }

    /// Interprets the last compiled output as text, removing a trailing NUL and replacing invalid UTF-8 lossily. Intended for PTX output.
    ///
    /// # Errors
    ///
    /// Returns an error if NVVM cannot query or copy the requested output, or the returned byte count cannot be represented as `usize`.
    pub fn compiled_string(&self) -> Result<String> {
        Ok(bytes_to_string(self.compiled_result()?))
    }

    /// Returns the compiler or verification log, removing a trailing NUL and replacing invalid UTF-8 lossily.
    ///
    /// # Errors
    ///
    /// Returns an error if NVVM cannot query or copy the requested output, or the returned byte count cannot be represented as `usize`.
    pub fn log(&self) -> Result<String> {
        Ok(bytes_to_string(self.bytes(
            sys::nvvmGetProgramLogSize,
            sys::nvvmGetProgramLog,
        )?))
    }

    /// Borrows the native program handle without transferring ownership. It remains valid until the wrapper is dropped or converted with `into_raw`.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvvmProgram {
        self.handle
    }

    /// Transfers ownership of the raw NVVM program handle to the caller.
    ///
    /// The caller becomes responsible for destroying it with
    /// `nvvmDestroyProgram`.
    #[must_use]
    pub fn into_raw(self) -> sys::nvvmProgram {
        let handle = self.handle;
        mem::forget(self);
        handle
    }

    fn add_module_raw(&mut self, module: &Module<'_>, lazy: bool) -> Result<()> {
        if module.ir.is_empty() {
            return Err(Error::InvalidValue);
        }
        let name = CString::new(module.name)?;
        let function = if lazy {
            sys::nvvmLazyAddModuleToProgram
        } else {
            sys::nvvmAddModuleToProgram
        };

        // SAFETY: the program handle is live, and the buffer and name are borrowed for the call.
        unsafe {
            try_nvvm!(function(
                self.handle,
                module.ir.as_ptr().cast(),
                module.ir.len() as _,
                name.as_ptr(),
            ))
        }
    }

    fn compile_raw(
        &self,
        function: unsafe extern "C" fn(sys::nvvmProgram, i32, *mut *const i8) -> sys::nvvmResult,
        options: &[&str],
    ) -> Result<()> {
        let options = options
            .iter()
            .map(|option| CString::new(*option))
            .collect::<result::Result<Vec<_>, _>>()?;
        let mut option_ptrs = options
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();

        // SAFETY: the program handle is live, and the buffer and name are borrowed for the call.
        unsafe {
            try_nvvm!(function(
                self.handle,
                to_i32(option_ptrs.len(), "NVVM option count")?,
                if option_ptrs.is_empty() {
                    ptr::null_mut()
                } else {
                    option_ptrs.as_mut_ptr()
                },
            ))
        }
    }

    fn bytes(
        &self,
        get_size: unsafe extern "C" fn(sys::nvvmProgram, *mut u64) -> sys::nvvmResult,
        get_data: unsafe extern "C" fn(sys::nvvmProgram, *mut i8) -> sys::nvvmResult,
    ) -> Result<Vec<u8>> {
        let mut size = 0;
        // SAFETY: the program handle is live, and `size` is a writable local.
        unsafe {
            try_nvvm!(get_size(self.handle, &raw mut size))?;
        }

        let mut bytes = vec![0u8; to_usize(size, "NVVM output byte count")?];
        if bytes.is_empty() {
            return Ok(bytes);
        }

        // SAFETY: `bytes` has the size the library reported for this output.
        unsafe {
            try_nvvm!(get_data(self.handle, bytes.as_mut_ptr().cast()))?;
        }
        Ok(bytes)
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: `nvvmDestroyProgram` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            if !self.handle.is_null()
                && let Err(error) = try_nvvm!(sys::nvvmDestroyProgram(&raw mut self.handle))
            {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy nvvm program: {error}");
            }
        }
    }
}

/// Returns the loaded NVVM compiler library version.
///
/// # Errors
///
/// Returns an error if NVVM cannot report its library version.
pub fn version() -> Result<Version> {
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: `nvvmVersion` receives valid output locations.
    unsafe {
        try_nvvm!(sys::nvvmVersion(&raw mut major, &raw mut minor))?;
    }
    Ok(Version { major, minor })
}

/// Returns the NVVM IR and debug metadata versions supported by the loaded compiler.
///
/// # Errors
///
/// Returns an error if NVVM cannot report its supported IR versions.
pub fn ir_version() -> Result<IrVersion> {
    let mut major = 0;
    let mut minor = 0;
    let mut debug_major = 0;
    let mut debug_minor = 0;
    // SAFETY: `nvvmIRVersion` receives valid output locations.
    unsafe {
        try_nvvm!(sys::nvvmIRVersion(
            &raw mut major,
            &raw mut minor,
            &raw mut debug_major,
            &raw mut debug_minor,
        ))?;
    }
    Ok(IrVersion {
        major,
        minor,
        debug_major,
        debug_minor,
    })
}

/// Returns the LLVM major version supported for a virtual GPU architecture.
///
/// # Errors
///
/// Returns `Error::InvalidValue` for a concrete architecture target, or a native NVVM error if the version query fails.
pub fn llvm_version(architecture: GpuArchitecture) -> Result<i32> {
    if !architecture.is_virtual() {
        return Err(Error::InvalidValue);
    }
    llvm_version_for_architecture(&architecture.to_string())
}

/// Returns the LLVM major version supported for the supplied native architecture string.
///
/// # Errors
///
/// Returns an error if `architecture` contains an interior NUL or NVVM rejects the architecture or version query.
pub fn llvm_version_for_architecture(architecture: &str) -> Result<i32> {
    let architecture = CString::new(architecture)?;
    let mut major = 0;
    // SAFETY: `nvvmLLVMVersion` receives valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_nvvm!(sys::nvvmLLVMVersion(architecture.as_ptr(), &raw mut major,))?;
    }
    Ok(major)
}

fn flag_bit(value: bool) -> u8 {
    u8::from(value)
}

fn bytes_to_string(mut bytes: Vec<u8>) -> String {
    if bytes.last() == Some(&0) {
        bytes.pop();
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_options_build_expected_arguments() {
        let options = CompileOptions::new()
            .device_debug(true)
            .optimization_level(OptimizationLevel::Zero)
            .gpu_architecture(GpuArchitecture::Compute90)
            .flush_to_zero(true)
            .precise_square_root(false)
            .precise_division(true)
            .fma(false)
            .jump_table_density(200)
            .generate_lto(true)
            .raw_option("-custom");

        assert_eq!(
            options.as_arguments(),
            vec![
                "-g",
                "-opt=0",
                "-arch=compute_90",
                "-ftz=1",
                "-prec-sqrt=0",
                "-prec-div=1",
                "-fma=0",
                "-jump-table-density=101",
                "-gen-lto",
                "-custom",
            ]
        );
    }

    #[test]
    fn test_version_queries_are_available() {
        let version = version().unwrap();
        assert!(version.major > 0);

        let ir_version = ir_version().unwrap();
        assert!(ir_version.major > 0);

        let llvm_version = llvm_version(GpuArchitecture::Compute90).unwrap();
        assert!(llvm_version > 0);
    }

    #[test]
    fn test_real_architecture_is_rejected_for_nvvm_options() {
        let options = CompileOptions::new().gpu_architecture(GpuArchitecture::Sm90);
        assert!(matches!(options.validate(), Err(Error::InvalidValue)));
        assert!(matches!(
            llvm_version(GpuArchitecture::Sm90),
            Err(Error::InvalidValue)
        ));
    }
}
