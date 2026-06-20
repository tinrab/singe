use std::{
    cell::UnsafeCell,
    ffi::{CStr, CString},
    fmt::{self, Display},
    ptr, result,
    sync::atomic::{AtomicBool, Ordering},
};

use singe_core::impl_enum_display;
use singe_cuda_sys::nvrtc as sys;

use crate::{
    architecture::GpuArchitecture,
    error::{Error, Result},
    module::ModuleImage,
    try_nvrtc,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    pub major: i32,
    pub minor: i32,
}

/// NVRTC result code returned by compiler operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    Success,
    OutOfMemory,
    ProgramCreationFailure,
    InvalidInput,
    InvalidProgram,
    InvalidOption,
    Compilation,
    BuiltinOperationFailure,
    NoNameExpressionsAfterCompilation,
    NoLoweredNamesBeforeCompilation,
    NameExpressionNotValid,
    InternalError,
    TimeFileWriteFailed,
    NoPchCreateAttempted,
    PchCreateHeapExhausted,
    PchCreate,
    Cancelled,
    TimeTraceFileWriteFailed,
    Unknown(u32),
}

impl Status {
    pub fn description(self) -> String {
        match sys::nvrtcResult::try_from(self.raw()) {
            Ok(status) => unsafe {
                let ptr = sys::nvrtcGetErrorString(status);
                if ptr.is_null() {
                    String::from("unknown nvrtc error")
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            },
            Err(_) => String::from("unknown nvrtc error"),
        }
    }

    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::nvrtcResult::NVRTC_SUCCESS as _,
            Self::OutOfMemory => sys::nvrtcResult::NVRTC_ERROR_OUT_OF_MEMORY as _,
            Self::ProgramCreationFailure => {
                sys::nvrtcResult::NVRTC_ERROR_PROGRAM_CREATION_FAILURE as _
            }
            Self::InvalidInput => sys::nvrtcResult::NVRTC_ERROR_INVALID_INPUT as _,
            Self::InvalidProgram => sys::nvrtcResult::NVRTC_ERROR_INVALID_PROGRAM as _,
            Self::InvalidOption => sys::nvrtcResult::NVRTC_ERROR_INVALID_OPTION as _,
            Self::Compilation => sys::nvrtcResult::NVRTC_ERROR_COMPILATION as _,
            Self::BuiltinOperationFailure => {
                sys::nvrtcResult::NVRTC_ERROR_BUILTIN_OPERATION_FAILURE as _
            }
            Self::NoNameExpressionsAfterCompilation => {
                sys::nvrtcResult::NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION as _
            }
            Self::NoLoweredNamesBeforeCompilation => {
                sys::nvrtcResult::NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION as _
            }
            Self::NameExpressionNotValid => {
                sys::nvrtcResult::NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID as _
            }
            Self::InternalError => sys::nvrtcResult::NVRTC_ERROR_INTERNAL_ERROR as _,
            Self::TimeFileWriteFailed => sys::nvrtcResult::NVRTC_ERROR_TIME_FILE_WRITE_FAILED as _,
            Self::NoPchCreateAttempted => {
                sys::nvrtcResult::NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED as _
            }
            Self::PchCreateHeapExhausted => {
                sys::nvrtcResult::NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED as _
            }
            Self::PchCreate => sys::nvrtcResult::NVRTC_ERROR_PCH_CREATE as _,
            Self::Cancelled => sys::nvrtcResult::NVRTC_ERROR_CANCELLED as _,
            Self::TimeTraceFileWriteFailed => {
                sys::nvrtcResult::NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED as _
            }
            Self::Unknown(code) => code,
        }
    }
}

impl TryFrom<u32> for Status {
    type Error = u32;

    fn try_from(code: u32) -> result::Result<Self, u32> {
        match code {
            code if code == sys::nvrtcResult::NVRTC_SUCCESS as u32 => Ok(Self::Success),
            code if code == sys::nvrtcResult::NVRTC_ERROR_OUT_OF_MEMORY as u32 => {
                Ok(Self::OutOfMemory)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_PROGRAM_CREATION_FAILURE as u32 => {
                Ok(Self::ProgramCreationFailure)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_INVALID_INPUT as u32 => {
                Ok(Self::InvalidInput)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_INVALID_PROGRAM as u32 => {
                Ok(Self::InvalidProgram)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_INVALID_OPTION as u32 => {
                Ok(Self::InvalidOption)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_COMPILATION as u32 => {
                Ok(Self::Compilation)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_BUILTIN_OPERATION_FAILURE as u32 => {
                Ok(Self::BuiltinOperationFailure)
            }
            code if code
                == sys::nvrtcResult::NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION as u32 =>
            {
                Ok(Self::NoNameExpressionsAfterCompilation)
            }
            code if code
                == sys::nvrtcResult::NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION as u32 =>
            {
                Ok(Self::NoLoweredNamesBeforeCompilation)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID as u32 => {
                Ok(Self::NameExpressionNotValid)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_INTERNAL_ERROR as u32 => {
                Ok(Self::InternalError)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_TIME_FILE_WRITE_FAILED as u32 => {
                Ok(Self::TimeFileWriteFailed)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED as u32 => {
                Ok(Self::NoPchCreateAttempted)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED as u32 => {
                Ok(Self::PchCreateHeapExhausted)
            }
            code if code == sys::nvrtcResult::NVRTC_ERROR_PCH_CREATE as u32 => Ok(Self::PchCreate),
            code if code == sys::nvrtcResult::NVRTC_ERROR_CANCELLED as u32 => Ok(Self::Cancelled),
            code if code == sys::nvrtcResult::NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED as u32 => {
                Ok(Self::TimeTraceFileWriteFailed)
            }
            code => Err(code),
        }
    }
}

impl From<sys::nvrtcResult> for Status {
    fn from(status: sys::nvrtcResult) -> Self {
        Self::try_from(status as u32).unwrap_or_else(Self::Unknown)
    }
}

impl TryFrom<Status> for sys::nvrtcResult {
    type Error = Status;

    fn try_from(status: Status) -> result::Result<Self, Status> {
        match status {
            Status::Success => Ok(Self::NVRTC_SUCCESS),
            Status::OutOfMemory => Ok(Self::NVRTC_ERROR_OUT_OF_MEMORY),
            Status::ProgramCreationFailure => Ok(Self::NVRTC_ERROR_PROGRAM_CREATION_FAILURE),
            Status::InvalidInput => Ok(Self::NVRTC_ERROR_INVALID_INPUT),
            Status::InvalidProgram => Ok(Self::NVRTC_ERROR_INVALID_PROGRAM),
            Status::InvalidOption => Ok(Self::NVRTC_ERROR_INVALID_OPTION),
            Status::Compilation => Ok(Self::NVRTC_ERROR_COMPILATION),
            Status::BuiltinOperationFailure => Ok(Self::NVRTC_ERROR_BUILTIN_OPERATION_FAILURE),
            Status::NoNameExpressionsAfterCompilation => {
                Ok(Self::NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION)
            }
            Status::NoLoweredNamesBeforeCompilation => {
                Ok(Self::NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION)
            }
            Status::NameExpressionNotValid => Ok(Self::NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID),
            Status::InternalError => Ok(Self::NVRTC_ERROR_INTERNAL_ERROR),
            Status::TimeFileWriteFailed => Ok(Self::NVRTC_ERROR_TIME_FILE_WRITE_FAILED),
            Status::NoPchCreateAttempted => Ok(Self::NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED),
            Status::PchCreateHeapExhausted => Ok(Self::NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED),
            Status::PchCreate => Ok(Self::NVRTC_ERROR_PCH_CREATE),
            Status::Cancelled => Ok(Self::NVRTC_ERROR_CANCELLED),
            Status::TimeTraceFileWriteFailed => Ok(Self::NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED),
            Status::Unknown(_) => Err(status),
        }
    }
}

impl Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("NVRTC_SUCCESS"),
            Self::OutOfMemory => f.write_str("NVRTC_ERROR_OUT_OF_MEMORY"),
            Self::ProgramCreationFailure => f.write_str("NVRTC_ERROR_PROGRAM_CREATION_FAILURE"),
            Self::InvalidInput => f.write_str("NVRTC_ERROR_INVALID_INPUT"),
            Self::InvalidProgram => f.write_str("NVRTC_ERROR_INVALID_PROGRAM"),
            Self::InvalidOption => f.write_str("NVRTC_ERROR_INVALID_OPTION"),
            Self::Compilation => f.write_str("NVRTC_ERROR_COMPILATION"),
            Self::BuiltinOperationFailure => f.write_str("NVRTC_ERROR_BUILTIN_OPERATION_FAILURE"),
            Self::NoNameExpressionsAfterCompilation => {
                f.write_str("NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION")
            }
            Self::NoLoweredNamesBeforeCompilation => {
                f.write_str("NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION")
            }
            Self::NameExpressionNotValid => f.write_str("NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID"),
            Self::InternalError => f.write_str("NVRTC_ERROR_INTERNAL_ERROR"),
            Self::TimeFileWriteFailed => f.write_str("NVRTC_ERROR_TIME_FILE_WRITE_FAILED"),
            Self::NoPchCreateAttempted => f.write_str("NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED"),
            Self::PchCreateHeapExhausted => f.write_str("NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED"),
            Self::PchCreate => f.write_str("NVRTC_ERROR_PCH_CREATE"),
            Self::Cancelled => f.write_str("NVRTC_ERROR_CANCELLED"),
            Self::TimeTraceFileWriteFailed => {
                f.write_str("NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED")
            }
            Self::Unknown(code) => write!(f, "UNKNOWN_NVRTC_STATUS({code})"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Header<'a> {
    pub source: &'a str,
    pub include_name: &'a str,
}

#[derive(Debug, Clone)]
struct OwnedHeader {
    source: String,
    include_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MacroDefinition<'a> {
    Name(&'a str),
    WithValue { name: &'a str, value: &'a str },
}

impl MacroDefinition<'_> {
    fn format(self) -> String {
        match self {
            Self::Name(name) => name.to_string(),
            Self::WithValue { name, value } => format!("{name}={value}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CppDialect {
    Cpp03,
    Cpp11,
    Cpp14,
    Cpp17,
    Cpp20,
}

impl_enum_display!(CppDialect, {
    Self::Cpp03 => "c++03",
    Self::Cpp11 => "c++11",
    Self::Cpp14 => "c++14",
    Self::Cpp17 => "c++17",
    Self::Cpp20 => "c++20",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FastCompileLevel {
    Zero,
    Min,
    Mid,
    Max,
}

impl_enum_display!(FastCompileLevel, {
    Self::Zero => "0",
    Self::Min => "min",
    Self::Mid => "mid",
    Self::Max => "max",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WarningAsErrorKind {
    AllWarnings,
    Reorder,
    DeprecatedDeclarations,
}

impl_enum_display!(WarningAsErrorKind, {
    Self::AllWarnings => "all-warnings",
    Self::Reorder => "reorder",
    Self::DeprecatedDeclarations => "deprecated-declarations",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OptimizationInfoKind {
    Inline,
}

impl_enum_display!(OptimizationInfoKind, {
    Self::Inline => "inline",
});

#[derive(Debug, Clone, Default)]
pub struct CompileOptions<'a> {
    pub gpu_architecture: Option<GpuArchitecture>,
    pub relocatable_device_code: Option<bool>,
    pub extensible_whole_program: bool,
    pub device_debug: bool,
    pub generate_line_info: bool,
    pub device_optimization: Option<bool>,
    pub fast_compile: Option<FastCompileLevel>,
    pub ptxas_options: Vec<&'a str>,
    pub max_register_count: Option<i32>,
    pub flush_to_zero: Option<bool>,
    pub precise_square_root: Option<bool>,
    pub precise_division: Option<bool>,
    pub fmad: Option<bool>,
    pub use_fast_math: bool,
    pub extra_device_vectorization: bool,
    pub modify_stack_limit: Option<bool>,
    pub dlink_time_optimization: bool,
    pub generate_optimized_lto: bool,
    pub optix_ir: bool,
    pub jump_table_density: Option<u8>,
    pub no_cache: bool,
    pub random_seed: Option<&'a str>,
    pub define_macros: Vec<MacroDefinition<'a>>,
    pub undefine_macros: Vec<&'a str>,
    pub include_paths: Vec<&'a str>,
    pub pre_include_headers: Vec<&'a str>,
    pub no_source_include: bool,
    pub cpp_dialect: Option<CppDialect>,
    pub builtin_move_forward: Option<bool>,
    pub builtin_initializer_list: Option<bool>,
    pub pch: bool,
    pub create_pch: Option<&'a str>,
    pub use_pch: Option<&'a str>,
    pub pch_dir: Option<&'a str>,
    pub pch_verbose: Option<bool>,
    pub pch_messages: Option<bool>,
    pub instantiate_templates_in_pch: Option<bool>,
    pub disable_warnings: bool,
    pub warning_as_error: Vec<WarningAsErrorKind>,
    pub restrict_pointers: bool,
    pub device_as_default_execution_space: bool,
    pub device_int128: bool,
    pub device_float128: bool,
    pub optimization_info: Vec<OptimizationInfoKind>,
    pub display_error_number: Option<bool>,
    pub diag_error: Vec<i32>,
    pub diag_suppress: Vec<i32>,
    pub diag_warn: Vec<i32>,
    pub brief_diagnostics: Option<bool>,
    pub time: Option<&'a str>,
    pub split_compile: Option<i32>,
    pub device_syntax_only: bool,
    pub minimal: bool,
    pub device_stack_protector: Option<bool>,
    pub device_time_trace: Option<&'a str>,
    pub raw_options: Vec<&'a str>,
}

impl<'a> CompileOptions<'a> {
    pub const fn new() -> Self {
        Self {
            gpu_architecture: None,
            relocatable_device_code: None,
            extensible_whole_program: false,
            device_debug: false,
            generate_line_info: false,
            device_optimization: None,
            fast_compile: None,
            ptxas_options: Vec::new(),
            max_register_count: None,
            flush_to_zero: None,
            precise_square_root: None,
            precise_division: None,
            fmad: None,
            use_fast_math: false,
            extra_device_vectorization: false,
            modify_stack_limit: None,
            dlink_time_optimization: false,
            generate_optimized_lto: false,
            optix_ir: false,
            jump_table_density: None,
            no_cache: false,
            random_seed: None,
            define_macros: Vec::new(),
            undefine_macros: Vec::new(),
            include_paths: Vec::new(),
            pre_include_headers: Vec::new(),
            no_source_include: false,
            cpp_dialect: None,
            builtin_move_forward: None,
            builtin_initializer_list: None,
            pch: false,
            create_pch: None,
            use_pch: None,
            pch_dir: None,
            pch_verbose: None,
            pch_messages: None,
            instantiate_templates_in_pch: None,
            disable_warnings: false,
            warning_as_error: Vec::new(),
            restrict_pointers: false,
            device_as_default_execution_space: false,
            device_int128: false,
            device_float128: false,
            optimization_info: Vec::new(),
            display_error_number: None,
            diag_error: Vec::new(),
            diag_suppress: Vec::new(),
            diag_warn: Vec::new(),
            brief_diagnostics: None,
            time: None,
            split_compile: None,
            device_syntax_only: false,
            minimal: false,
            device_stack_protector: None,
            device_time_trace: None,
            raw_options: Vec::new(),
        }
    }

    pub fn gpu_architecture(mut self, value: GpuArchitecture) -> Self {
        self.gpu_architecture = Some(value);
        self
    }

    pub fn relocatable_device_code(mut self, value: bool) -> Self {
        self.relocatable_device_code = Some(value);
        self
    }

    pub fn extensible_whole_program(mut self, value: bool) -> Self {
        self.extensible_whole_program = value;
        self
    }

    pub fn device_debug(mut self, value: bool) -> Self {
        self.device_debug = value;
        self
    }

    pub fn generate_line_info(mut self, value: bool) -> Self {
        self.generate_line_info = value;
        self
    }

    pub fn device_optimization(mut self, value: bool) -> Self {
        self.device_optimization = Some(value);
        self
    }

    pub fn fast_compile(mut self, value: FastCompileLevel) -> Self {
        self.fast_compile = Some(value);
        self
    }

    pub fn ptxas_option(mut self, value: &'a str) -> Self {
        self.ptxas_options.push(value);
        self
    }

    pub fn max_register_count(mut self, value: i32) -> Self {
        self.max_register_count = Some(value);
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

    pub fn fmad(mut self, value: bool) -> Self {
        self.fmad = Some(value);
        self
    }

    pub fn use_fast_math(mut self, value: bool) -> Self {
        self.use_fast_math = value;
        self
    }

    pub fn extra_device_vectorization(mut self, value: bool) -> Self {
        self.extra_device_vectorization = value;
        self
    }

    pub fn modify_stack_limit(mut self, value: bool) -> Self {
        self.modify_stack_limit = Some(value);
        self
    }

    pub fn dlink_time_optimization(mut self, value: bool) -> Self {
        self.dlink_time_optimization = value;
        self
    }

    pub fn generate_optimized_lto(mut self, value: bool) -> Self {
        self.generate_optimized_lto = value;
        self
    }

    pub fn optix_ir(mut self, value: bool) -> Self {
        self.optix_ir = value;
        self
    }

    pub fn jump_table_density(mut self, value: u8) -> Self {
        self.jump_table_density = Some(value.min(101));
        self
    }

    pub fn no_cache(mut self, value: bool) -> Self {
        self.no_cache = value;
        self
    }

    pub fn random_seed(mut self, value: &'a str) -> Self {
        self.random_seed = Some(value);
        self
    }

    pub fn define_macro(mut self, value: MacroDefinition<'a>) -> Self {
        self.define_macros.push(value);
        self
    }

    pub fn undefine_macro(mut self, value: &'a str) -> Self {
        self.undefine_macros.push(value);
        self
    }

    pub fn include_path(mut self, value: &'a str) -> Self {
        self.include_paths.push(value);
        self
    }

    pub fn pre_include_header(mut self, value: &'a str) -> Self {
        self.pre_include_headers.push(value);
        self
    }

    pub fn no_source_include(mut self, value: bool) -> Self {
        self.no_source_include = value;
        self
    }

    pub fn cpp_dialect(mut self, value: CppDialect) -> Self {
        self.cpp_dialect = Some(value);
        self
    }

    pub fn builtin_move_forward(mut self, value: bool) -> Self {
        self.builtin_move_forward = Some(value);
        self
    }

    pub fn builtin_initializer_list(mut self, value: bool) -> Self {
        self.builtin_initializer_list = Some(value);
        self
    }

    pub fn pch(mut self, value: bool) -> Self {
        self.pch = value;
        self
    }

    pub fn create_pch(mut self, value: &'a str) -> Self {
        self.create_pch = Some(value);
        self
    }

    pub fn use_pch(mut self, value: &'a str) -> Self {
        self.use_pch = Some(value);
        self
    }

    pub fn pch_dir(mut self, value: &'a str) -> Self {
        self.pch_dir = Some(value);
        self
    }

    pub fn pch_verbose(mut self, value: bool) -> Self {
        self.pch_verbose = Some(value);
        self
    }

    pub fn pch_messages(mut self, value: bool) -> Self {
        self.pch_messages = Some(value);
        self
    }

    pub fn instantiate_templates_in_pch(mut self, value: bool) -> Self {
        self.instantiate_templates_in_pch = Some(value);
        self
    }

    pub fn disable_warnings(mut self, value: bool) -> Self {
        self.disable_warnings = value;
        self
    }

    pub fn warning_as_error(mut self, value: WarningAsErrorKind) -> Self {
        self.warning_as_error.push(value);
        self
    }

    pub fn restrict_pointers(mut self, value: bool) -> Self {
        self.restrict_pointers = value;
        self
    }

    pub fn device_as_default_execution_space(mut self, value: bool) -> Self {
        self.device_as_default_execution_space = value;
        self
    }

    pub fn device_int128(mut self, value: bool) -> Self {
        self.device_int128 = value;
        self
    }

    pub fn device_float128(mut self, value: bool) -> Self {
        self.device_float128 = value;
        self
    }

    pub fn optimization_info(mut self, value: OptimizationInfoKind) -> Self {
        self.optimization_info.push(value);
        self
    }

    pub fn display_error_number(mut self, value: bool) -> Self {
        self.display_error_number = Some(value);
        self
    }

    pub fn diag_error(mut self, value: i32) -> Self {
        self.diag_error.push(value);
        self
    }

    pub fn diag_suppress(mut self, value: i32) -> Self {
        self.diag_suppress.push(value);
        self
    }

    pub fn diag_warn(mut self, value: i32) -> Self {
        self.diag_warn.push(value);
        self
    }

    pub fn brief_diagnostics(mut self, value: bool) -> Self {
        self.brief_diagnostics = Some(value);
        self
    }

    pub fn time(mut self, value: &'a str) -> Self {
        self.time = Some(value);
        self
    }

    pub fn split_compile(mut self, value: i32) -> Self {
        self.split_compile = Some(value);
        self
    }

    pub fn device_syntax_only(mut self, value: bool) -> Self {
        self.device_syntax_only = value;
        self
    }

    pub fn minimal(mut self, value: bool) -> Self {
        self.minimal = value;
        self
    }

    pub fn device_stack_protector(mut self, value: bool) -> Self {
        self.device_stack_protector = Some(value);
        self
    }

    pub fn device_time_trace(mut self, value: &'a str) -> Self {
        self.device_time_trace = Some(value);
        self
    }

    pub fn raw_option(mut self, value: &'a str) -> Self {
        self.raw_options.push(value);
        self
    }

    pub fn as_arguments(&self) -> Vec<String> {
        let mut arguments = Vec::new();

        if let Some(value) = self.gpu_architecture {
            arguments.push(format!("--gpu-architecture={value}"));
        }
        if let Some(value) = self.relocatable_device_code {
            arguments.push(format!("--relocatable-device-code={}", bool_flag(value)));
        }
        if self.extensible_whole_program {
            arguments.push(String::from("--extensible-whole-program"));
        }
        if self.device_debug {
            arguments.push(String::from("--device-debug"));
        }
        if self.generate_line_info {
            arguments.push(String::from("--generate-line-info"));
        }
        if let Some(value) = self.device_optimization
            && value
        {
            arguments.push(String::from("--dopt=on"));
        }
        if let Some(value) = self.fast_compile {
            arguments.push(format!("--Ofast-compile={value}"));
        }
        arguments.extend(
            self.ptxas_options
                .iter()
                .map(|value| format!("--ptxas-options={value}")),
        );
        if let Some(value) = self.max_register_count {
            arguments.push(format!("--maxrregcount={value}"));
        }
        if let Some(value) = self.flush_to_zero {
            arguments.push(format!("--ftz={}", bool_flag(value)));
        }
        if let Some(value) = self.precise_square_root {
            arguments.push(format!("--prec-sqrt={}", bool_flag(value)));
        }
        if let Some(value) = self.precise_division {
            arguments.push(format!("--prec-div={}", bool_flag(value)));
        }
        if let Some(value) = self.fmad {
            arguments.push(format!("--fmad={}", bool_flag(value)));
        }
        if self.use_fast_math {
            arguments.push(String::from("--use_fast_math"));
        }
        if self.extra_device_vectorization {
            arguments.push(String::from("--extra-device-vectorization"));
        }
        if let Some(value) = self.modify_stack_limit {
            arguments.push(format!("--modify-stack-limit={}", bool_flag(value)));
        }
        if self.dlink_time_optimization {
            arguments.push(String::from("--dlink-time-opt"));
        }
        if self.generate_optimized_lto {
            arguments.push(String::from("--gen-opt-lto"));
        }
        if self.optix_ir {
            arguments.push(String::from("--optix-ir"));
        }
        if let Some(value) = self.jump_table_density {
            arguments.push(format!("--jump-table-density={value}"));
        }
        if self.no_cache {
            arguments.push(String::from("--no-cache"));
        }
        if let Some(value) = self.random_seed {
            arguments.push(format!("--frandom-seed={value}"));
        }
        arguments.extend(
            self.define_macros
                .iter()
                .copied()
                .map(|value| format!("--define-macro={}", value.format())),
        );
        arguments.extend(
            self.undefine_macros
                .iter()
                .map(|value| format!("--undefine-macro={value}")),
        );
        arguments.extend(
            self.include_paths
                .iter()
                .map(|value| format!("--include-path={value}")),
        );
        arguments.extend(
            self.pre_include_headers
                .iter()
                .map(|value| format!("--pre-include={value}")),
        );
        if self.no_source_include {
            arguments.push(String::from("--no-source-include"));
        }
        if let Some(value) = self.cpp_dialect {
            arguments.push(format!("--std={value}"));
        }
        if let Some(value) = self.builtin_move_forward {
            arguments.push(format!("--builtin-move-forward={}", bool_flag(value)));
        }
        if let Some(value) = self.builtin_initializer_list {
            arguments.push(format!("--builtin-initializer-list={}", bool_flag(value)));
        }
        if self.pch {
            arguments.push(String::from("--pch"));
        }
        if let Some(value) = self.create_pch {
            arguments.push(format!("--create-pch={value}"));
        }
        if let Some(value) = self.use_pch {
            arguments.push(format!("--use-pch={value}"));
        }
        if let Some(value) = self.pch_dir {
            arguments.push(format!("--pch-dir={value}"));
        }
        if let Some(value) = self.pch_verbose {
            arguments.push(format!("--pch-verbose={}", bool_flag(value)));
        }
        if let Some(value) = self.pch_messages {
            arguments.push(format!("--pch-messages={}", bool_flag(value)));
        }
        if let Some(value) = self.instantiate_templates_in_pch {
            arguments.push(format!(
                "--instantiate-templates-in-pch={}",
                bool_flag(value)
            ));
        }
        if self.disable_warnings {
            arguments.push(String::from("--disable-warnings"));
        }
        if !self.warning_as_error.is_empty() {
            arguments.push(format!(
                "--warning-as-error={}",
                join_display(&self.warning_as_error)
            ));
        }
        if self.restrict_pointers {
            arguments.push(String::from("--restrict"));
        }
        if self.device_as_default_execution_space {
            arguments.push(String::from("--device-as-default-execution-space"));
        }
        if self.device_int128 {
            arguments.push(String::from("--device-int128"));
        }
        if self.device_float128 {
            arguments.push(String::from("--device-float128"));
        }
        arguments.extend(
            self.optimization_info
                .iter()
                .map(|value| format!("--optimization-info={value}")),
        );
        if let Some(value) = self.display_error_number {
            arguments.push(if value {
                String::from("--display-error-number")
            } else {
                String::from("--no-display-error-number")
            });
        }
        if !self.diag_error.is_empty() {
            arguments.push(format!("--diag-error={}", join_numbers(&self.diag_error)));
        }
        if !self.diag_suppress.is_empty() {
            arguments.push(format!(
                "--diag-suppress={}",
                join_numbers(&self.diag_suppress)
            ));
        }
        if !self.diag_warn.is_empty() {
            arguments.push(format!("--diag-warn={}", join_numbers(&self.diag_warn)));
        }
        if let Some(value) = self.brief_diagnostics {
            arguments.push(format!("--brief-diagnostics={}", bool_flag(value)));
        }
        if let Some(value) = self.time {
            arguments.push(format!("--time={value}"));
        }
        if let Some(value) = self.split_compile {
            arguments.push(format!("--split-compile={value}"));
        }
        if self.device_syntax_only {
            arguments.push(String::from("--fdevice-syntax-only"));
        }
        if self.minimal {
            arguments.push(String::from("--minimal"));
        }
        if let Some(value) = self.device_stack_protector {
            arguments.push(format!("--device-stack-protector={}", bool_flag(value)));
        }
        if let Some(value) = self.device_time_trace {
            arguments.push(format!("--fdevice-time-trace={value}"));
        }

        arguments.extend(self.raw_options.iter().map(|value| (*value).to_string()));
        arguments
    }
}

#[derive(Debug)]
pub struct Program {
    source: String,
    name: Option<String>,
    headers: Vec<OwnedHeader>,
    handle: UnsafeCell<sys::nvrtcProgram>,
}

impl Program {
    /// Creates a lazy NVRTC program from CUDA source text.
    ///
    /// The raw NVRTC handle is created when the program is compiled or
    /// otherwise materialized.
    pub fn from_source(source: &str) -> Self {
        Self {
            source: source.to_string(),
            name: None,
            headers: Vec::new(),
            handle: UnsafeCell::new(ptr::null_mut()),
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub fn with_header(mut self, header: Header<'_>) -> Self {
        self.headers.push(OwnedHeader {
            source: header.source.to_string(),
            include_name: header.include_name.to_string(),
        });
        self
    }

    pub fn with_headers(mut self, headers: &[Header<'_>]) -> Self {
        self.headers
            .extend(headers.iter().map(|header| OwnedHeader {
                source: header.source.to_string(),
                include_name: header.include_name.to_string(),
            }));
        self
    }

    /// Takes ownership of a raw NVRTC program handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `nvrtcProgram` that is not owned by any other
    /// wrapper. The returned wrapper destroys it with `nvrtcDestroyProgram`.
    pub unsafe fn from_raw(handle: sys::nvrtcProgram) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            source: String::new(),
            name: None,
            headers: Vec::new(),
            handle: UnsafeCell::new(handle),
        })
    }

    pub fn compile(&self, options: &[&str]) -> Result<()> {
        self.compile_raw(options)
    }

    pub fn compile_with_options(&self, options: &CompileOptions<'_>) -> Result<()> {
        let arguments = options.as_arguments();
        let argument_refs = arguments.iter().map(String::as_str).collect::<Vec<_>>();
        self.compile_raw(&argument_refs)
    }

    /// Registers a flow callback that can cancel compilation.
    ///
    /// The callback function must satisfy the following constraints:
    ///
    /// (1) It must return 1 to cancel compilation or 0 to continue.
    /// Other return values are reserved for future use.
    ///
    /// (2) It must return consistent values.
    /// Once it returns 1 at one point, it must return 1 in all following invocations during the current nvrtcCompileProgram call in progress.
    ///
    /// (3) It must be thread-safe.
    ///
    /// (4) It must not invoke any NVRTC, libNVVM, or PTX APIs.
    pub fn compile_with_options_and_cancel_flag(
        &self,
        options: &CompileOptions<'_>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        unsafe {
            try_nvrtc!(sys::nvrtcSetFlowCallback(
                self.handle()?,
                Some(cancel_if_requested_callback),
                ptr::from_ref(cancel).cast_mut().cast(),
            ))?;
        }

        let compile_result = self.compile_with_options(options);
        let clear_result = clear_flow_callback(self);

        match (compile_result, clear_result) {
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    /// Registers a name expression for a `__global__`, `__device__`, or `__constant__` symbol.
    ///
    /// The identical name expression string must be provided to [`Program::lowered_name`] after compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if `name_expression` contains an interior NUL byte, if
    /// the program handle is invalid, or if NVRTC rejects the expression.
    pub fn add_name_expression(&self, name_expression: &str) -> Result<()> {
        let name_expression = CString::new(name_expression)?;
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
    /// Returns an error if `name_expression` contains an interior NUL byte, if
    /// the program handle is invalid, or if NVRTC cannot return the lowered name.
    pub fn lowered_name(&self, name_expression: &str) -> Result<String> {
        let name_expression = CString::new(name_expression)?;
        let mut lowered_name = ptr::null();
        unsafe {
            try_nvrtc!(sys::nvrtcGetLoweredName(
                self.handle()?,
                name_expression.as_ptr(),
                &raw mut lowered_name,
            ))?;
            Ok(CStr::from_ptr(lowered_name).to_string_lossy().into_owned())
        }
    }

    pub fn ptx(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetPTXSize, sys::nvrtcGetPTX)
    }

    pub fn ptx_image(&self) -> Result<ModuleImage<'static>> {
        Ok(ModuleImage::from_vec(self.ptx()?))
    }

    pub fn ptx_string(&self) -> Result<String> {
        Ok(bytes_to_string(self.ptx()?))
    }

    pub fn cubin(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetCUBINSize, sys::nvrtcGetCUBIN)
    }

    pub fn cubin_image(&self) -> Result<ModuleImage<'static>> {
        Ok(ModuleImage::from_vec(self.cubin()?))
    }

    pub fn lto_ir(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetLTOIRSize, sys::nvrtcGetLTOIR)
    }

    pub fn lto_ir_image(&self) -> Result<ModuleImage<'static>> {
        Ok(ModuleImage::from_vec(self.lto_ir()?))
    }

    pub fn optix_ir(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetOptiXIRSize, sys::nvrtcGetOptiXIR)
    }

    pub fn optix_ir_image(&self) -> Result<ModuleImage<'static>> {
        Ok(ModuleImage::from_vec(self.optix_ir()?))
    }

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
    /// In this scenario, use [`Program::pch_heap_size_required`] to query the required heap size, reallocate the heap with [`set_pch_heap_size`], and retry PCH creation with [`sys::nvrtcCompileProgram`](singe_cuda_sys::nvrtc::nvrtcCompileProgram) on a new NVRTC program instance.
    /// [`Status::PchCreate`] indicates that an error condition prevented the PCH file from being created.
    ///
    /// # Errors
    ///
    /// Returns an error if the program handle is invalid.
    pub fn pch_create_status(&self) -> Result<Status> {
        unsafe { Ok(sys::nvrtcGetPCHCreateStatus(self.handle()?).into()) }
    }

    /// Returns the PCH heap size required to compile the given program.
    ///
    /// # Errors
    ///
    /// Returns an error if the program handle is invalid or if the returned size
    /// is not valid because [`Program::pch_create_status`] did not return
    /// [`Status::Success`] or [`Status::PchCreateHeapExhausted`].
    pub fn pch_heap_size_required(&self) -> Result<usize> {
        let mut size = 0;
        unsafe {
            try_nvrtc!(sys::nvrtcGetPCHHeapSizeRequired(
                self.handle()?,
                &raw mut size
            ))?;
        }
        Ok(size as usize)
    }

    /// Returns the raw NVRTC program handle, creating it first if needed.
    ///
    /// If the program has not been materialized yet, this creates the raw
    /// handle first.
    ///
    /// # Errors
    ///
    /// Returns an error if NVRTC cannot create the program handle.
    pub fn materialize_raw(&self) -> Result<sys::nvrtcProgram> {
        self.handle()
    }

    /// Returns the raw NVRTC program handle only if it has already been
    /// materialized.
    pub const fn as_raw_if_materialized(&self) -> Option<sys::nvrtcProgram> {
        let handle = unsafe { *self.handle.get() };
        if handle.is_null() { None } else { Some(handle) }
    }

    /// Transfers ownership of the raw NVRTC program handle to the caller.
    ///
    /// If the program has not been materialized yet, this creates the raw
    /// handle first. The caller becomes responsible for destroying the returned
    /// handle with `nvrtcDestroyProgram`.
    pub fn into_raw(self) -> Result<sys::nvrtcProgram> {
        let handle = self.handle()?;
        std::mem::forget(self);
        Ok(handle)
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

        unsafe {
            try_nvrtc!(sys::nvrtcCompileProgram(
                self.handle()?,
                option_ptrs.len() as _,
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
        get_data: unsafe extern "C" fn(sys::nvrtcProgram, *mut i8) -> sys::nvrtcResult,
    ) -> Result<Vec<u8>> {
        let mut size = 0;
        unsafe {
            try_nvrtc!(get_size(self.handle()?, &raw mut size))?;
        }

        let mut bytes = vec![0u8; size as usize];
        if bytes.is_empty() {
            return Ok(bytes);
        }

        unsafe {
            try_nvrtc!(get_data(self.handle()?, bytes.as_mut_ptr().cast()))?;
        }
        Ok(bytes)
    }

    fn handle(&self) -> Result<sys::nvrtcProgram> {
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
        unsafe {
            try_nvrtc!(sys::nvrtcCreateProgram(
                &raw mut handle,
                source.as_ptr(),
                name.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
                self.headers.len() as _,
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
        unsafe {
            let handle = self.handle.get();
            if !(*handle).is_null() {
                let _ = sys::nvrtcDestroyProgram(handle);
            }
        }
    }
}

#[non_exhaustive]
pub enum CompilationArtifact {
    Ptx(ModuleImage<'static>),
    Cubin(ModuleImage<'static>),
    LtoIr(ModuleImage<'static>),
    OptixIr(ModuleImage<'static>),
}

impl CompilationArtifact {
    pub fn image(&self) -> &ModuleImage<'static> {
        match self {
            Self::Ptx(image) | Self::Cubin(image) | Self::LtoIr(image) | Self::OptixIr(image) => {
                image
            }
        }
    }

    pub fn into_image(self) -> ModuleImage<'static> {
        match self {
            Self::Ptx(image) | Self::Cubin(image) | Self::LtoIr(image) | Self::OptixIr(image) => {
                image
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutputKind {
    Ptx,
    Cubin,
    LtoIr,
    OptixIr,
}

impl Program {
    pub fn artifact(&self, kind: OutputKind) -> Result<CompilationArtifact> {
        match kind {
            OutputKind::Ptx => {
                let image = self.ptx_image()?;
                if image.as_bytes().is_empty() {
                    return Err(Error::InvalidValue);
                }
                Ok(CompilationArtifact::Ptx(image))
            }
            OutputKind::Cubin => {
                let image = self.cubin_image()?;
                if image.as_bytes().is_empty() {
                    return Err(Error::InvalidValue);
                }
                Ok(CompilationArtifact::Cubin(image))
            }
            OutputKind::LtoIr => {
                let image = self.lto_ir_image()?;
                if image.as_bytes().is_empty() {
                    return Err(Error::InvalidValue);
                }
                Ok(CompilationArtifact::LtoIr(image))
            }
            OutputKind::OptixIr => {
                let image = self.optix_ir_image()?;
                if image.as_bytes().is_empty() {
                    return Err(Error::InvalidValue);
                }
                Ok(CompilationArtifact::OptixIr(image))
            }
        }
    }
}

/// Returns the CUDA Runtime Compilation version.
///
/// # Errors
///
/// Returns an error if NVRTC cannot report its version.
pub fn version() -> Result<Version> {
    let mut major = 0;
    let mut minor = 0;
    unsafe {
        try_nvrtc!(sys::nvrtcVersion(&raw mut major, &raw mut minor))?;
    }
    Ok(Version { major, minor })
}

pub fn supported_architectures() -> Result<Vec<i32>> {
    let mut count = 0;
    unsafe {
        try_nvrtc!(sys::nvrtcGetNumSupportedArchs(&raw mut count))?;
    }

    let mut architectures = vec![0; count as usize];
    if architectures.is_empty() {
        return Ok(Vec::new());
    }

    unsafe {
        try_nvrtc!(sys::nvrtcGetSupportedArchs(architectures.as_mut_ptr()))?;
    }
    Ok(architectures)
}

/// Returns the current size of the PCH heap.
///
/// # Errors
///
/// Returns an error if NVRTC cannot report the PCH heap size.
pub fn pch_heap_size() -> Result<usize> {
    let mut size = 0;
    unsafe {
        try_nvrtc!(sys::nvrtcGetPCHHeapSize(&raw mut size))?;
    }
    Ok(size as usize)
}

/// Sets the size of the PCH heap.
///
/// The requested size may be rounded up to a platform-dependent alignment, such as the page size.
/// If the PCH heap has already been allocated, NVRTC frees it and allocates a new heap.
///
/// # Errors
///
/// Returns an error if NVRTC rejects the requested PCH heap size.
pub fn set_pch_heap_size(size: usize) -> Result<()> {
    unsafe { try_nvrtc!(sys::nvrtcSetPCHHeapSize(size as _)) }
}

unsafe extern "C" fn noop_compile_callback(
    payload: *mut std::ffi::c_void,
    reserved: *mut std::ffi::c_void,
) -> i32 {
    let _ = payload;
    let _ = reserved;
    0
}

unsafe extern "C" fn cancel_if_requested_callback(
    payload: *mut std::ffi::c_void,
    reserved: *mut std::ffi::c_void,
) -> i32 {
    let _ = reserved;
    if payload.is_null() {
        return 0;
    }

    let cancel = unsafe { &*payload.cast::<AtomicBool>() };
    i32::from(cancel.load(Ordering::Relaxed))
}

/// Clears the NVRTC flow callback by installing a no-op callback.
///
/// Use this after a compile attempt that installed a cancellation callback.
///
/// # Errors
///
/// Returns an error if the program handle is invalid or if NVRTC rejects the
/// callback update.
pub fn clear_flow_callback(program: &Program) -> Result<()> {
    unsafe {
        try_nvrtc!(sys::nvrtcSetFlowCallback(
            program.handle()?,
            Some(noop_compile_callback),
            ptr::null_mut(),
        ))
    }
}

fn bool_flag(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn join_display(values: &[impl Display]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn join_numbers(values: &[i32]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn bytes_to_string(mut bytes: Vec<u8>) -> String {
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::{
        device::Device, error::Result, memory::DeviceMemory, module::LaunchConfig, testing,
    };

    fn current_device_sm_architecture() -> Result<GpuArchitecture> {
        let properties = Device::current()?.properties()?;
        Ok(match (properties.major, properties.minor) {
            (7, 5) => GpuArchitecture::Sm75,
            (8, 0) => GpuArchitecture::Sm80,
            (8, 6) => GpuArchitecture::Sm86,
            (8, 7) => GpuArchitecture::Sm87,
            (8, 9) => GpuArchitecture::Sm89,
            (9, 0) => GpuArchitecture::Sm90,
            (10, 0) => GpuArchitecture::Sm100,
            (10, 1) => GpuArchitecture::Sm101,
            (10, 3) => GpuArchitecture::Sm103,
            (12, 0) => GpuArchitecture::Sm120,
            (12, 1) => GpuArchitecture::Sm121,
            (major, minor) => panic!("unsupported device architecture sm_{major}{minor}"),
        })
    }

    #[test]
    fn version_is_available() {
        let version = version().unwrap();
        assert_ne!(version.major, 0);
    }

    #[test]
    fn supported_architectures_are_sorted() {
        let architectures = supported_architectures().unwrap();
        assert!(!architectures.is_empty());
        assert!(
            architectures
                .windows(2)
                .all(|window| window[0] <= window[1])
        );
    }

    #[test]
    fn compile_options_build_expected_arguments() {
        let arguments = CompileOptions::new()
            .gpu_architecture(GpuArchitecture::Compute80)
            .device_debug(true)
            .generate_line_info(true)
            .define_macro(MacroDefinition::WithValue {
                name: "FOO",
                value: "42",
            })
            .include_path("include")
            .cpp_dialect(CppDialect::Cpp20)
            .warning_as_error(WarningAsErrorKind::Reorder)
            .diag_suppress(177)
            .raw_option("--custom-flag")
            .as_arguments();

        assert_eq!(
            arguments,
            vec![
                "--gpu-architecture=compute_80",
                "--device-debug",
                "--generate-line-info",
                "--define-macro=FOO=42",
                "--include-path=include",
                "--std=c++20",
                "--warning-as-error=reorder",
                "--diag-suppress=177",
                "--custom-flag",
            ]
        );
    }

    #[test]
    fn compiles_to_ptx() {
        let program = Program::from_source(
            r#"
            extern "C" __global__ void saxpy(float a, const float* x, const float* y, float* out, size_t n) {
                size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < n) {
                    out[tid] = a * x[tid] + y[tid];
                }
            }
            "#,
        )
        .with_name("saxpy.cu");
        let options = CompileOptions::new()
            .gpu_architecture(GpuArchitecture::Compute80)
            .generate_line_info(true);
        program.compile_with_options(&options).unwrap();

        let ptx = program.ptx_string().unwrap();
        assert!(ptx.contains(".visible .entry saxpy"));
    }

    #[test]
    fn compile_with_cancel_flag_succeeds_when_not_cancelled() {
        let (_lock, _ctx) = testing::bootstrap().unwrap();
        let cancel = AtomicBool::new(false);
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop.cu");
        let options = CompileOptions::new().gpu_architecture(GpuArchitecture::Compute80);

        program
            .compile_with_options_and_cancel_flag(&options, &cancel)
            .unwrap();
        assert!(program.ptx_string().unwrap().contains("noop"));
    }

    #[test]
    fn clear_flow_callback_is_allowed_before_compilation() {
        let program =
            Program::from_source("extern \"C\" __global__ void noop() {}").with_name("noop.cu");
        clear_flow_callback(&program).unwrap();
    }

    #[test]
    fn cubin_artifact_loads_as_module() {
        let (_lock, ctx) = testing::bootstrap().unwrap();
        let program = Program::from_source(
            r#"
            extern "C" __global__ void saxpy(float a, const float* x, const float* y, float* out, size_t n) {
                size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < n) {
                    out[tid] = a * x[tid] + y[tid];
                }
            }
            "#,
        )
        .with_name("saxpy_module.cu");
        let architecture = current_device_sm_architecture().unwrap();
        let options = CompileOptions::new().gpu_architecture(architecture);
        program.compile_with_options(&options).unwrap();

        let module = ctx.load_nvrtc_module(&program, OutputKind::Cubin).unwrap();
        assert!(module.function("saxpy").is_ok());
    }

    #[test]
    fn cubin_artifact_loads_as_module_with_jit_options() {
        let (_lock, ctx) = testing::bootstrap().unwrap();
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_module_jit.cu");
        let architecture = current_device_sm_architecture().unwrap();
        let options = CompileOptions::new().gpu_architecture(architecture);
        program.compile_with_options(&options).unwrap();

        let mut info_log = [0u8; 1024];
        let mut error_log = [0u8; 1024];
        let jit_options = crate::jit::JitOptions::default()
            .with_generate_line_info(true)
            .with_info_log(&mut info_log)
            .with_error_log(&mut error_log);

        let module = ctx
            .load_nvrtc_module_with_options(&program, OutputKind::Cubin, jit_options)
            .unwrap();
        assert!(module.function("noop").is_ok());
    }

    #[test]
    fn compiles_loads_launches_and_reads_back_results() {
        let (_lock, ctx) = testing::bootstrap().unwrap();

        let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
        let mut output = vec![0.0f32; input.len()];
        let scalar = 2.5f32;
        let input_device = DeviceMemory::from_slice(&input).unwrap();
        let output_device = DeviceMemory::<f32>::zeroes(output.len()).unwrap();
        let length = input.len();

        let program = Program::from_source(
            r#"
            extern "C" __global__ void scale_add(const float* input, float* output, float alpha, size_t len) {
                size_t i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < len) {
                    output[i] = input[i] * alpha + 1.0f;
                }
            }
            "#,
        )
        .with_name("scale_add.cu");
        let architecture = current_device_sm_architecture().unwrap();
        let compile_options = CompileOptions::new().gpu_architecture(architecture);
        program.compile_with_options(&compile_options).unwrap();

        let module = ctx.load_nvrtc_module(&program, OutputKind::Cubin).unwrap();
        let function = module.function("scale_add").unwrap();

        let config = LaunchConfig::for_1d_grid(input.len(), 128);
        let input_ptr = input_device.as_ptr();
        let mut output_ptr = output_device.as_mut_ptr();

        function
            .launch(&config, (&input_ptr, &mut output_ptr, &scalar, &length))
            .unwrap();
        output_device.copy_to_host(&mut output).unwrap();

        let expected = input
            .iter()
            .map(|value| value * scalar + 1.0)
            .collect::<Vec<_>>();
        assert_eq!(output, expected);
    }

    #[test]
    fn cubin_artifact_loads_as_library() {
        let (_lock, ctx) = testing::bootstrap().unwrap();
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_library.cu");
        let architecture = current_device_sm_architecture().unwrap();
        let options = CompileOptions::new().gpu_architecture(architecture);
        program.compile_with_options(&options).unwrap();

        let library = ctx.load_nvrtc_library(&program, OutputKind::Cubin).unwrap();
        assert!(library.kernel_count().unwrap() >= 1);
    }

    #[test]
    fn lto_ir_artifact_is_available_when_requested() {
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_lto.cu");
        let options = CompileOptions::new().dlink_time_optimization(true);
        program.compile_with_options(&options).unwrap();

        let artifact = program.artifact(OutputKind::LtoIr).unwrap();
        assert!(!artifact.image().as_bytes().is_empty());
        let ptx = program.artifact(OutputKind::Ptx).unwrap();
        assert!(!ptx.image().as_bytes().is_empty());
    }

    #[test]
    fn optix_ir_artifact_is_available_when_requested() {
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_optix.cu");
        let options = CompileOptions::new().optix_ir(true);
        program.compile_with_options(&options).unwrap();

        let artifact = program.artifact(OutputKind::OptixIr).unwrap();
        assert!(!artifact.image().as_bytes().is_empty());
        let ptx = program.artifact(OutputKind::Ptx).unwrap();
        assert!(!ptx.image().as_bytes().is_empty());
    }

    #[test]
    fn cubin_artifact_requires_real_architecture() {
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_cubin.cu");
        let options = CompileOptions::new().gpu_architecture(GpuArchitecture::Compute80);
        program.compile_with_options(&options).unwrap();

        assert!(program.artifact(OutputKind::Cubin).is_err());
    }
}
