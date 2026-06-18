use std::{
    ffi::{CStr, CString},
    fmt, ptr, result,
};

use singe_core::impl_enum_display;
use singe_cuda_sys::nvvm as sys;

use crate::{
    architecture::GpuArchitecture,
    error::{Error, Result},
    module::ModuleImage,
    try_nvvm,
};

// TODO: move to a core version type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    pub major: i32,
    pub minor: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IrVersion {
    pub major: i32,
    pub minor: i32,
    pub debug_major: i32,
    pub debug_minor: i32,
}

/// NVVM result code returned by compiler operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    Success,
    OutOfMemory,
    ProgramCreationFailure,
    IrVersionMismatch,
    InvalidInput,
    InvalidProgram,
    InvalidIr,
    InvalidOption,
    NoModuleInProgram,
    Compilation,
    Cancelled,
    Unknown(u32),
}

impl Status {
    pub fn description(self) -> String {
        match sys::nvvmResult::try_from(self.raw()) {
            Ok(status) => unsafe {
                let ptr = sys::nvvmGetErrorString(status);
                if ptr.is_null() {
                    String::from("unknown nvvm error")
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            },
            Err(_) => String::from("unknown nvvm error"),
        }
    }

    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::nvvmResult::NVVM_SUCCESS as _,
            Self::OutOfMemory => sys::nvvmResult::NVVM_ERROR_OUT_OF_MEMORY as _,
            Self::ProgramCreationFailure => {
                sys::nvvmResult::NVVM_ERROR_PROGRAM_CREATION_FAILURE as _
            }
            Self::IrVersionMismatch => sys::nvvmResult::NVVM_ERROR_IR_VERSION_MISMATCH as _,
            Self::InvalidInput => sys::nvvmResult::NVVM_ERROR_INVALID_INPUT as _,
            Self::InvalidProgram => sys::nvvmResult::NVVM_ERROR_INVALID_PROGRAM as _,
            Self::InvalidIr => sys::nvvmResult::NVVM_ERROR_INVALID_IR as _,
            Self::InvalidOption => sys::nvvmResult::NVVM_ERROR_INVALID_OPTION as _,
            Self::NoModuleInProgram => sys::nvvmResult::NVVM_ERROR_NO_MODULE_IN_PROGRAM as _,
            Self::Compilation => sys::nvvmResult::NVVM_ERROR_COMPILATION as _,
            Self::Cancelled => sys::nvvmResult::NVVM_ERROR_CANCELLED as _,
            Self::Unknown(code) => code,
        }
    }
}

impl TryFrom<u32> for Status {
    type Error = u32;

    fn try_from(code: u32) -> result::Result<Self, u32> {
        match code {
            code if code == sys::nvvmResult::NVVM_SUCCESS as u32 => Ok(Self::Success),
            code if code == sys::nvvmResult::NVVM_ERROR_OUT_OF_MEMORY as u32 => {
                Ok(Self::OutOfMemory)
            }
            code if code == sys::nvvmResult::NVVM_ERROR_PROGRAM_CREATION_FAILURE as u32 => {
                Ok(Self::ProgramCreationFailure)
            }
            code if code == sys::nvvmResult::NVVM_ERROR_IR_VERSION_MISMATCH as u32 => {
                Ok(Self::IrVersionMismatch)
            }
            code if code == sys::nvvmResult::NVVM_ERROR_INVALID_INPUT as u32 => {
                Ok(Self::InvalidInput)
            }
            code if code == sys::nvvmResult::NVVM_ERROR_INVALID_PROGRAM as u32 => {
                Ok(Self::InvalidProgram)
            }
            code if code == sys::nvvmResult::NVVM_ERROR_INVALID_IR as u32 => Ok(Self::InvalidIr),
            code if code == sys::nvvmResult::NVVM_ERROR_INVALID_OPTION as u32 => {
                Ok(Self::InvalidOption)
            }
            code if code == sys::nvvmResult::NVVM_ERROR_NO_MODULE_IN_PROGRAM as u32 => {
                Ok(Self::NoModuleInProgram)
            }
            code if code == sys::nvvmResult::NVVM_ERROR_COMPILATION as u32 => Ok(Self::Compilation),
            code if code == sys::nvvmResult::NVVM_ERROR_CANCELLED as u32 => Ok(Self::Cancelled),
            code => Err(code),
        }
    }
}

impl From<sys::nvvmResult> for Status {
    fn from(status: sys::nvvmResult) -> Self {
        Self::try_from(status as u32).unwrap_or_else(Self::Unknown)
    }
}

impl TryFrom<Status> for sys::nvvmResult {
    type Error = Status;

    fn try_from(status: Status) -> result::Result<Self, Status> {
        match status {
            Status::Success => Ok(Self::NVVM_SUCCESS),
            Status::OutOfMemory => Ok(Self::NVVM_ERROR_OUT_OF_MEMORY),
            Status::ProgramCreationFailure => Ok(Self::NVVM_ERROR_PROGRAM_CREATION_FAILURE),
            Status::IrVersionMismatch => Ok(Self::NVVM_ERROR_IR_VERSION_MISMATCH),
            Status::InvalidInput => Ok(Self::NVVM_ERROR_INVALID_INPUT),
            Status::InvalidProgram => Ok(Self::NVVM_ERROR_INVALID_PROGRAM),
            Status::InvalidIr => Ok(Self::NVVM_ERROR_INVALID_IR),
            Status::InvalidOption => Ok(Self::NVVM_ERROR_INVALID_OPTION),
            Status::NoModuleInProgram => Ok(Self::NVVM_ERROR_NO_MODULE_IN_PROGRAM),
            Status::Compilation => Ok(Self::NVVM_ERROR_COMPILATION),
            Status::Cancelled => Ok(Self::NVVM_ERROR_CANCELLED),
            Status::Unknown(_) => Err(status),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("NVVM_SUCCESS"),
            Self::OutOfMemory => f.write_str("NVVM_ERROR_OUT_OF_MEMORY"),
            Self::ProgramCreationFailure => f.write_str("NVVM_ERROR_PROGRAM_CREATION_FAILURE"),
            Self::IrVersionMismatch => f.write_str("NVVM_ERROR_IR_VERSION_MISMATCH"),
            Self::InvalidInput => f.write_str("NVVM_ERROR_INVALID_INPUT"),
            Self::InvalidProgram => f.write_str("NVVM_ERROR_INVALID_PROGRAM"),
            Self::InvalidIr => f.write_str("NVVM_ERROR_INVALID_IR"),
            Self::InvalidOption => f.write_str("NVVM_ERROR_INVALID_OPTION"),
            Self::NoModuleInProgram => f.write_str("NVVM_ERROR_NO_MODULE_IN_PROGRAM"),
            Self::Compilation => f.write_str("NVVM_ERROR_COMPILATION"),
            Self::Cancelled => f.write_str("NVVM_ERROR_CANCELLED"),
            Self::Unknown(code) => write!(f, "UNKNOWN_NVVM_STATUS({code})"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OptimizationLevel {
    Zero,
    Three,
}

impl_enum_display!(OptimizationLevel, {
    Self::Zero => "0",
    Self::Three => "3",
});

#[derive(Debug, Clone, Default)]
pub struct CompileOptions<'a> {
    pub device_debug: bool,
    pub optimization_level: Option<OptimizationLevel>,
    pub gpu_architecture: Option<GpuArchitecture>,
    pub flush_to_zero: Option<bool>,
    pub precise_square_root: Option<bool>,
    pub precise_division: Option<bool>,
    pub fma: Option<bool>,
    pub jump_table_density: Option<u8>,
    pub generate_lto: bool,
    pub raw_options: Vec<&'a str>,
}

impl<'a> CompileOptions<'a> {
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

    pub fn device_debug(mut self, value: bool) -> Self {
        self.device_debug = value;
        self
    }

    pub fn optimization_level(mut self, value: OptimizationLevel) -> Self {
        self.optimization_level = Some(value);
        self
    }

    pub fn gpu_architecture(mut self, value: GpuArchitecture) -> Self {
        self.gpu_architecture = Some(value);
        self
    }

    pub fn flush_to_zero(mut self, value: bool) -> Self {
        self.flush_to_zero = Some(value);
        self
    }

    pub fn precise_square_root(mut self, value: bool) -> Self {
        self.precise_square_root = Some(value);
        self
    }

    pub fn precise_division(mut self, value: bool) -> Self {
        self.precise_division = Some(value);
        self
    }

    pub fn fma(mut self, value: bool) -> Self {
        self.fma = Some(value);
        self
    }

    pub fn jump_table_density(mut self, value: u8) -> Self {
        self.jump_table_density = Some(value.min(101));
        self
    }

    pub fn generate_lto(mut self, value: bool) -> Self {
        self.generate_lto = value;
        self
    }

    pub fn raw_option(mut self, value: &'a str) -> Self {
        self.raw_options.push(value);
        self
    }

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

#[derive(Debug)]
pub struct Module<'a> {
    pub ir: &'a [u8],
    pub name: &'a str,
}

#[derive(Debug)]
pub struct Program {
    handle: sys::nvvmProgram,
}

impl Program {
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
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
    pub unsafe fn from_raw(handle: sys::nvvmProgram) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle })
    }

    pub fn add_module(&mut self, module: Module<'_>) -> Result<()> {
        self.add_module_raw(module, false)
    }

    pub fn lazy_add_module(&mut self, module: Module<'_>) -> Result<()> {
        self.add_module_raw(module, true)
    }

    pub fn compile(&self, options: &[&str]) -> Result<()> {
        self.compile_raw(sys::nvvmCompileProgram, options)
    }

    pub fn compile_with_options(&self, options: &CompileOptions<'_>) -> Result<()> {
        options.validate()?;
        let arguments = options.as_arguments();
        let argument_refs = arguments.iter().map(String::as_str).collect::<Vec<_>>();
        self.compile(&argument_refs)
    }

    pub fn verify(&self, options: &[&str]) -> Result<()> {
        self.compile_raw(sys::nvvmVerifyProgram, options)
    }

    pub fn verify_with_options(&self, options: &CompileOptions<'_>) -> Result<()> {
        options.validate()?;
        let arguments = options.as_arguments();
        let argument_refs = arguments.iter().map(String::as_str).collect::<Vec<_>>();
        self.verify(&argument_refs)
    }

    pub fn compiled_result(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvvmGetCompiledResultSize, sys::nvvmGetCompiledResult)
    }

    pub fn compiled_image(&self) -> Result<ModuleImage<'static>> {
        Ok(ModuleImage::from_vec(self.compiled_result()?))
    }

    pub fn compiled_string(&self) -> Result<String> {
        Ok(bytes_to_string(self.compiled_result()?))
    }

    pub fn log(&self) -> Result<String> {
        Ok(bytes_to_string(self.bytes(
            sys::nvvmGetProgramLogSize,
            sys::nvvmGetProgramLog,
        )?))
    }

    pub const fn as_raw(&self) -> sys::nvvmProgram {
        self.handle
    }

    /// Transfers ownership of the raw NVVM program handle to the caller.
    ///
    /// The caller becomes responsible for destroying it with
    /// `nvvmDestroyProgram`.
    pub fn into_raw(self) -> sys::nvvmProgram {
        let handle = self.handle;
        std::mem::forget(self);
        handle
    }

    fn add_module_raw(&mut self, module: Module<'_>, lazy: bool) -> Result<()> {
        if module.ir.is_empty() {
            return Err(Error::InvalidValue);
        }
        let name = CString::new(module.name)?;
        let function = if lazy {
            sys::nvvmLazyAddModuleToProgram
        } else {
            sys::nvvmAddModuleToProgram
        };

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

        unsafe {
            try_nvvm!(function(
                self.handle,
                option_ptrs.len() as _,
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
        unsafe {
            try_nvvm!(get_size(self.handle, &raw mut size))?;
        }

        let mut bytes = vec![0u8; size as usize];
        if bytes.is_empty() {
            return Ok(bytes);
        }

        unsafe {
            try_nvvm!(get_data(self.handle, bytes.as_mut_ptr().cast()))?;
        }
        Ok(bytes)
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                let _ = sys::nvvmDestroyProgram(&raw mut self.handle);
            }
        }
    }
}

pub fn version() -> Result<Version> {
    let mut major = 0;
    let mut minor = 0;
    unsafe {
        try_nvvm!(sys::nvvmVersion(&raw mut major, &raw mut minor))?;
    }
    Ok(Version { major, minor })
}

pub fn ir_version() -> Result<IrVersion> {
    let mut major = 0;
    let mut minor = 0;
    let mut debug_major = 0;
    let mut debug_minor = 0;
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

pub fn llvm_version(architecture: GpuArchitecture) -> Result<i32> {
    if !architecture.is_virtual() {
        return Err(Error::InvalidValue);
    }
    llvm_version_for_architecture(&architecture.to_string())
}

pub fn llvm_version_for_architecture(architecture: &str) -> Result<i32> {
    let architecture = CString::new(architecture)?;
    let mut major = 0;
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
    fn compile_options_build_expected_arguments() {
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
    fn version_queries_are_available() {
        let version = version().unwrap();
        assert!(version.major > 0);

        let ir_version = ir_version().unwrap();
        assert!(ir_version.major > 0);

        let llvm_version = llvm_version(GpuArchitecture::Compute90).unwrap();
        assert!(llvm_version > 0);
    }

    #[test]
    fn real_architecture_is_rejected_for_nvvm_options() {
        let options = CompileOptions::new().gpu_architecture(GpuArchitecture::Sm90);
        assert!(matches!(options.validate(), Err(Error::InvalidValue)));
        assert!(matches!(
            llvm_version(GpuArchitecture::Sm90),
            Err(Error::InvalidValue)
        ));
    }
}
