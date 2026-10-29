use crate::utility::{to_i32, to_usize};

mod library;

use std::{
    cell::UnsafeCell,
    ffi::{CStr, CString, c_void},
    fmt::Display,
    mem::forget,
    ptr, result,
    sync::atomic::{AtomicBool, Ordering},
};

pub use library::library_path;
use singe_core::{impl_enum_display, static_str_from_c_ptr, status_enum};
use singe_cuda_sys::nvrtc as sys;

use crate::{
    architecture::GpuArchitecture,
    error::{Error, Result},
    module::ModuleImage,
    try_nvrtc,
};

/// Major and minor version of the loaded NVRTC library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    /// Major NVRTC version component.
    pub major: i32,
    /// Minor NVRTC version component.
    pub minor: i32,
}

status_enum! {
    /// NVRTC result code returned by compiler operations.
    pub enum Status: u32 => sys::nvrtcResult, unknown = "UNKNOWN_NVRTC_STATUS" {
        /// The operation completed successfully.
        Success = NVRTC_SUCCESS,
        /// NVRTC could not allocate the required memory.
        OutOfMemory = NVRTC_ERROR_OUT_OF_MEMORY,
        /// NVRTC could not create the program.
        ProgramCreationFailure = NVRTC_ERROR_PROGRAM_CREATION_FAILURE,
        /// An input argument is invalid.
        InvalidInput = NVRTC_ERROR_INVALID_INPUT,
        /// The program handle is invalid.
        InvalidProgram = NVRTC_ERROR_INVALID_PROGRAM,
        /// A compilation option is invalid.
        InvalidOption = NVRTC_ERROR_INVALID_OPTION,
        /// CUDA source compilation failed; the program log contains diagnostics.
        Compilation = NVRTC_ERROR_COMPILATION,
        /// An NVRTC built-in operation failed.
        BuiltinOperationFailure = NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
        /// A name expression cannot be added after compilation.
        NoNameExpressionsAfterCompilation = NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
        /// A lowered name was requested before compiling the program.
        NoLoweredNamesBeforeCompilation = NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
        /// The name expression is invalid.
        NameExpressionNotValid = NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
        /// An internal NVRTC error occurred.
        InternalError = NVRTC_ERROR_INTERNAL_ERROR,
        /// NVRTC could not write its compilation timing file.
        TimeFileWriteFailed = NVRTC_ERROR_TIME_FILE_WRITE_FAILED,
        /// No precompiled-header creation was attempted. PCH processing may not have been requested, or automatic PCH processing chose not to create a file.
        NoPchCreateAttempted = NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED,
        /// The precompiled-header heap was exhausted. Query the required size and enlarge the heap before retrying compilation with a new program.
        PchCreateHeapExhausted = NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED,
        /// An error prevented the precompiled-header file from being created.
        PchCreate = NVRTC_ERROR_PCH_CREATE,
        /// The operation was cancelled.
        Cancelled = NVRTC_ERROR_CANCELLED,
        /// NVRTC could not write its compilation time-trace file.
        TimeTraceFileWriteFailed = NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED,
    }
}

impl Status {
    /// Library description of this status.
    #[must_use]
    pub fn description(self) -> &'static str {
        let Ok(raw) = sys::nvrtcResult::try_from(self) else {
            return "unknown nvrtc error";
        };
        // SAFETY: `nvrtcGetErrorString` returns null or a static NUL-terminated string.
        unsafe { static_str_from_c_ptr(sys::nvrtcGetErrorString(raw), "unknown nvrtc error") }
    }
}

/// Borrowed CUDA header source and the virtual include name used by NVRTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Header<'a> {
    /// Header source text. Program builders copy this text when the header is added.
    pub source: &'a str,
    /// Include name by which CUDA source refers to this header.
    pub include_name: &'a str,
}

#[derive(Debug, Clone)]
struct OwnedHeader {
    source: String,
    include_name: String,
}

/// Preprocessor macro definition emitted as an NVRTC `--define-macro` option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MacroDefinition<'a> {
    /// Defines a macro without an explicit replacement value.
    Name(
        /// Preprocessor macro name.
        &'a str,
    ),
    /// Defines a macro with an explicit replacement value.
    WithValue {
        /// Preprocessor macro name.
        name: &'a str,
        /// Macro replacement text.
        value: &'a str,
    },
}

impl MacroDefinition<'_> {
    fn format(self) -> String {
        match self {
            Self::Name(name) => name.to_string(),
            Self::WithValue { name, value } => format!("{name}={value}"),
        }
    }
}

/// C++ language standard passed to NVRTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CppDialect {
    /// Selects the C++03 language standard.
    Cpp03,
    /// Selects the C++11 language standard.
    Cpp11,
    /// Selects the C++14 language standard.
    Cpp14,
    /// Selects the C++17 language standard.
    Cpp17,
    /// Selects the C++20 language standard.
    Cpp20,
}

impl_enum_display!(CppDialect, {
    Self::Cpp03 => "c++03",
    Self::Cpp11 => "c++11",
    Self::Cpp14 => "c++14",
    Self::Cpp17 => "c++17",
    Self::Cpp20 => "c++20",
});

/// Amount of optimization disabled to improve device compilation speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FastCompileLevel {
    /// Disables fast-compile mode.
    Zero,
    /// Disables a limited set of expensive optimizations to reduce compilation time.
    Min,
    /// Disables more expensive optimizations to balance compilation time and runtime performance.
    Mid,
    /// Prioritizes compilation speed by disabling many optimizations.
    Max,
}

impl_enum_display!(FastCompileLevel, {
    Self::Zero => "0",
    Self::Min => "min",
    Self::Mid => "mid",
    Self::Max => "max",
});

/// Native warning category promoted to a compilation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WarningAsErrorKind {
    /// Promotes all warnings to errors.
    AllWarnings,
    /// Promotes constructor initialization-order warnings to errors.
    Reorder,
    /// Promotes deprecated-declaration warnings to errors.
    DeprecatedDeclarations,
}

impl_enum_display!(WarningAsErrorKind, {
    Self::AllWarnings => "all-warnings",
    Self::Reorder => "reorder",
    Self::DeprecatedDeclarations => "deprecated-declarations",
});

/// Category of native compiler optimization reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OptimizationInfoKind {
    /// Requests inlining optimization reports.
    Inline,
}

impl_enum_display!(OptimizationInfoKind, {
    Self::Inline => "inline",
});

/// Controls compilation of Tile constructs such as `__tile__` and `__tile_global__`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileCompilation {
    /// Enable support for Tile constructs and define the macro `__CUDACC_TILE__`.
    Enabled,
    /// Enable Tile parsing, but omit code generation for non-Tile code.
    TileOnly,
    /// Enable Tile parsing, but omit code generation for Tile code.
    SimtOnly,
}

/// Typed NVRTC compilation options. Unset optional values and disabled flag switches leave the corresponding native option unspecified. String lists preserve insertion order; `raw_options` are appended last. Validation occurs when the program is materialized or compiled.
#[derive(Debug, Clone, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent capabilities and option switches can be combined freely"
)]
pub struct CompileOptions<'a> {
    /// GPU architecture target used for PTX or machine-code generation.
    pub gpu_architecture: Option<GpuArchitecture>,
    /// Whether to generate relocatable device code for subsequent linking.
    pub relocatable_device_code: Option<bool>,
    /// Enables extensible whole-program compilation.
    pub extensible_whole_program: bool,
    /// Generates device debug information.
    pub device_debug: bool,
    /// Generates source line information for device code.
    pub generate_line_info: bool,
    /// Emits `--dopt=on` only for `Some(true)`. `None` and `Some(false)` leave the native default in effect.
    pub device_optimization: Option<bool>,
    /// Compilation-speed setting that trades generated-code optimization for shorter compilation.
    pub fast_compile: Option<FastCompileLevel>,
    /// Additional PTX assembler option strings, each emitted as a separate `--ptxas-options` argument.
    pub ptxas_options: Vec<&'a str>,
    /// Requested maximum register count per GPU thread, subject to native ABI minimums.
    pub max_register_count: Option<i32>,
    /// Whether single-precision denormal results are flushed to zero.
    pub flush_to_zero: Option<bool>,
    /// Whether single-precision square roots use the precise native implementation.
    pub precise_square_root: Option<bool>,
    /// Whether single-precision division uses the precise native implementation.
    pub precise_division: Option<bool>,
    /// Whether floating-point multiply and add operations may be contracted.
    pub fmad: Option<bool>,
    /// Enables the native fast-math preset.
    pub use_fast_math: bool,
    /// Enables additional device-code vectorization.
    pub extra_device_vectorization: bool,
    /// Whether NVRTC may temporarily raise the process-wide stack limit during compilation on Linux.
    pub modify_stack_limit: Option<bool>,
    /// Generates intermediate code for device link-time optimization.
    pub dlink_time_optimization: bool,
    /// Requests optimized link-time intermediate code.
    pub generate_optimized_lto: bool,
    /// Requests `OptiX` intermediate representation output.
    pub optix_ir: bool,
    /// Enable support for Tile constructs and control which code is generated.
    pub tile_compilation: Option<TileCompilation>,
    /// Consider unannotated functions and static-storage variables implicitly `__tile__`.
    pub default_tile: bool,
    /// Diagnose variables that cannot be implicitly annotated with `__tile__`.
    pub diagnose_implicit_tile_variable: bool,
    /// Case-density percentage threshold for generating a switch jump table.
    pub jump_table_density: Option<u8>,
    /// Disables native caching for both PTX and cubin generation.
    pub no_cache: bool,
    /// Seed string used for compiler-generated symbol and variable names.
    pub random_seed: Option<&'a str>,
    /// Preprocessor macro definitions emitted in insertion order.
    pub define_macros: Vec<MacroDefinition<'a>>,
    /// Preprocessor macro names to undefine.
    pub undefine_macros: Vec<&'a str>,
    /// Additional header search directories.
    pub include_paths: Vec<&'a str>,
    /// Headers included before processing the source.
    pub pre_include_headers: Vec<&'a str>,
    /// Prevents adding the source file directory to the header search path.
    pub no_source_include: bool,
    /// C++ language standard used for compilation.
    pub cpp_dialect: Option<CppDialect>,
    /// Whether NVRTC supplies built-in definitions of `std::move` and `std::forward`.
    pub builtin_move_forward: Option<bool>,
    /// Whether NVRTC supplies a built-in definition of `std::initializer_list`.
    pub builtin_initializer_list: Option<bool>,
    /// Enables automatic precompiled-header processing.
    pub pch: bool,
    /// Path of an explicitly created precompiled-header file.
    pub create_pch: Option<&'a str>,
    /// Path of an existing precompiled-header file to use.
    pub use_pch: Option<&'a str>,
    /// Directory used for automatic precompiled-header files.
    pub pch_dir: Option<&'a str>,
    /// Whether to emit verbose precompiled-header diagnostics.
    pub pch_verbose: Option<bool>,
    /// Whether to emit precompiled-header processing messages.
    pub pch_messages: Option<bool>,
    /// Whether to instantiate templates while creating a precompiled header.
    pub instantiate_templates_in_pch: Option<bool>,
    /// Suppresses native compiler warnings.
    pub disable_warnings: bool,
    /// Warning categories promoted to compilation errors.
    pub warning_as_error: Vec<WarningAsErrorKind>,
    /// Treats kernel pointer parameters as restricted pointers.
    pub restrict_pointers: bool,
    /// Treats unannotated functions as device functions.
    pub device_as_default_execution_space: bool,
    /// Enables device support for 128-bit integers.
    pub device_int128: bool,
    /// Enables device support for 128-bit floating-point types.
    pub device_float128: bool,
    /// Categories of optimization reports to request.
    pub optimization_info: Vec<OptimizationInfoKind>,
    /// Whether diagnostics display their numeric identifiers.
    pub display_error_number: Option<bool>,
    /// Diagnostic numbers promoted to errors.
    pub diag_error: Vec<i32>,
    /// Diagnostic numbers to suppress.
    pub diag_suppress: Vec<i32>,
    /// Diagnostic numbers demoted to warnings.
    pub diag_warn: Vec<i32>,
    /// Whether to use brief diagnostic output.
    pub brief_diagnostics: Option<bool>,
    /// File for compiler phase timing output.
    pub time: Option<&'a str>,
    /// Maximum number of threads used by parallel compilation.
    pub split_compile: Option<i32>,
    /// Checks device syntax without generating output code.
    pub device_syntax_only: bool,
    /// Omits texture, surface, device-runtime, device-launch, and runtime/driver type features to reduce compilation time.
    pub minimal: bool,
    /// Whether to enable device stack protection.
    pub device_stack_protector: Option<bool>,
    /// File for the device compilation time trace.
    pub device_time_trace: Option<&'a str>,
    /// Additional native option strings appended after the typed options.
    pub raw_options: Vec<&'a str>,
}

impl<'a> CompileOptions<'a> {
    /// Creates an empty option set that leaves native defaults in effect.
    #[must_use]
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
            tile_compilation: None,
            default_tile: false,
            diagnose_implicit_tile_variable: false,
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

    /// Sets the GPU architecture target for PTX or machine-code generation.
    #[must_use]
    pub fn gpu_architecture(mut self, value: GpuArchitecture) -> Self {
        self.gpu_architecture = Some(value);
        self
    }

    /// Enables or disables relocatable device code for subsequent linking.
    #[must_use]
    pub fn relocatable_device_code(mut self, value: bool) -> Self {
        self.relocatable_device_code = Some(value);
        self
    }

    /// Enables or disables extensible whole-program compilation.
    #[must_use]
    pub fn extensible_whole_program(mut self, value: bool) -> Self {
        self.extensible_whole_program = value;
        self
    }

    /// Enables or disables device debug information.
    #[must_use]
    pub fn device_debug(mut self, value: bool) -> Self {
        self.device_debug = value;
        self
    }

    /// Enables or disables device source line information.
    #[must_use]
    pub fn generate_line_info(mut self, value: bool) -> Self {
        self.generate_line_info = value;
        self
    }

    /// Requests `--dopt=on` when `value` is true. False leaves the native optimization default in effect.
    #[must_use]
    pub fn device_optimization(mut self, value: bool) -> Self {
        self.device_optimization = Some(value);
        self
    }

    /// Sets the compilation-speed tradeoff between optimization effort and generated-code performance.
    #[must_use]
    pub fn fast_compile(mut self, value: FastCompileLevel) -> Self {
        self.fast_compile = Some(value);
        self
    }

    /// Appends a PTX assembler option string as a separate `--ptxas-options` argument.
    #[must_use]
    pub fn ptxas_option(mut self, value: &'a str) -> Self {
        self.ptxas_options.push(value);
        self
    }

    /// Sets the requested maximum register count per GPU thread, subject to native ABI minimums.
    #[must_use]
    pub fn max_register_count(mut self, value: i32) -> Self {
        self.max_register_count = Some(value);
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
    pub fn fmad(mut self, value: bool) -> Self {
        self.fmad = Some(value);
        self
    }

    /// Enables or disables the native fast-math preset.
    #[must_use]
    pub fn use_fast_math(mut self, value: bool) -> Self {
        self.use_fast_math = value;
        self
    }

    /// Enables or disables additional device-code vectorization.
    #[must_use]
    pub fn extra_device_vectorization(mut self, value: bool) -> Self {
        self.extra_device_vectorization = value;
        self
    }

    /// Controls whether NVRTC may temporarily raise the process-wide stack limit during compilation on Linux.
    #[must_use]
    pub fn modify_stack_limit(mut self, value: bool) -> Self {
        self.modify_stack_limit = Some(value);
        self
    }

    /// Enables or disables device link-time intermediate-code generation.
    #[must_use]
    pub fn dlink_time_optimization(mut self, value: bool) -> Self {
        self.dlink_time_optimization = value;
        self
    }

    /// Enables or disables generation of optimized link-time intermediate code.
    #[must_use]
    pub fn generate_optimized_lto(mut self, value: bool) -> Self {
        self.generate_optimized_lto = value;
        self
    }

    /// Enables or disables `OptiX` intermediate representation output.
    #[must_use]
    pub fn optix_ir(mut self, value: bool) -> Self {
        self.optix_ir = value;
        self
    }

    /// Sets the case-density percentage threshold for generating a switch jump table.
    #[must_use]
    pub fn jump_table_density(mut self, value: u8) -> Self {
        self.jump_table_density = Some(value.min(101));
        self
    }

    /// Controls whether native PTX and cubin generation caches are disabled.
    #[must_use]
    pub fn no_cache(mut self, value: bool) -> Self {
        self.no_cache = value;
        self
    }

    /// Sets the seed string used for compiler-generated symbol and variable names.
    #[must_use]
    pub fn random_seed(mut self, value: &'a str) -> Self {
        self.random_seed = Some(value);
        self
    }

    /// Appends a preprocessor macro definition.
    #[must_use]
    pub fn define_macro(mut self, value: MacroDefinition<'a>) -> Self {
        self.define_macros.push(value);
        self
    }

    /// Appends a preprocessor macro name to undefine.
    #[must_use]
    pub fn undefine_macro(mut self, value: &'a str) -> Self {
        self.undefine_macros.push(value);
        self
    }

    /// Appends a header search directory.
    #[must_use]
    pub fn include_path(mut self, value: &'a str) -> Self {
        self.include_paths.push(value);
        self
    }

    /// Appends a header to include before processing the source.
    #[must_use]
    pub fn pre_include_header(mut self, value: &'a str) -> Self {
        self.pre_include_headers.push(value);
        self
    }

    /// Controls whether the source file directory is omitted from the header search path.
    #[must_use]
    pub fn no_source_include(mut self, value: bool) -> Self {
        self.no_source_include = value;
        self
    }

    /// Sets the C++ language standard.
    #[must_use]
    pub fn cpp_dialect(mut self, value: CppDialect) -> Self {
        self.cpp_dialect = Some(value);
        self
    }

    /// Controls whether NVRTC supplies built-in `std::move` and `std::forward` definitions.
    #[must_use]
    pub fn builtin_move_forward(mut self, value: bool) -> Self {
        self.builtin_move_forward = Some(value);
        self
    }

    /// Controls whether NVRTC supplies a built-in `std::initializer_list` definition.
    #[must_use]
    pub fn builtin_initializer_list(mut self, value: bool) -> Self {
        self.builtin_initializer_list = Some(value);
        self
    }

    /// Enables or disables automatic precompiled-header processing.
    #[must_use]
    pub fn pch(mut self, value: bool) -> Self {
        self.pch = value;
        self
    }

    /// Sets the path of a precompiled-header file to create explicitly.
    #[must_use]
    pub fn create_pch(mut self, value: &'a str) -> Self {
        self.create_pch = Some(value);
        self
    }

    /// Sets the path of an existing precompiled-header file to use.
    #[must_use]
    pub fn use_pch(mut self, value: &'a str) -> Self {
        self.use_pch = Some(value);
        self
    }

    /// Sets the directory used for automatic precompiled-header files.
    #[must_use]
    pub fn pch_dir(mut self, value: &'a str) -> Self {
        self.pch_dir = Some(value);
        self
    }

    /// Controls whether the log explains why candidate automatic precompiled headers could not be used.
    #[must_use]
    pub fn pch_verbose(mut self, value: bool) -> Self {
        self.pch_verbose = Some(value);
        self
    }

    /// Controls whether the log reports creating or using a precompiled header.
    #[must_use]
    pub fn pch_messages(mut self, value: bool) -> Self {
        self.pch_messages = Some(value);
        self
    }

    /// Controls template instantiation while creating a precompiled header.
    #[must_use]
    pub fn instantiate_templates_in_pch(mut self, value: bool) -> Self {
        self.instantiate_templates_in_pch = Some(value);
        self
    }

    /// Controls suppression of native compiler warnings.
    #[must_use]
    pub fn disable_warnings(mut self, value: bool) -> Self {
        self.disable_warnings = value;
        self
    }

    /// Appends a warning category to promote to compilation errors.
    #[must_use]
    pub fn warning_as_error(mut self, value: WarningAsErrorKind) -> Self {
        self.warning_as_error.push(value);
        self
    }

    /// Controls whether kernel pointer parameters are treated as restricted pointers.
    #[must_use]
    pub fn restrict_pointers(mut self, value: bool) -> Self {
        self.restrict_pointers = value;
        self
    }

    /// Controls whether unannotated functions are treated as device functions.
    #[must_use]
    pub fn device_as_default_execution_space(mut self, value: bool) -> Self {
        self.device_as_default_execution_space = value;
        self
    }

    /// Enables or disables device support for 128-bit integer types.
    #[must_use]
    pub fn device_int128(mut self, value: bool) -> Self {
        self.device_int128 = value;
        self
    }

    /// Enables or disables device support for 128-bit floating-point types.
    #[must_use]
    pub fn device_float128(mut self, value: bool) -> Self {
        self.device_float128 = value;
        self
    }

    /// Appends a category of optimization reports to request.
    #[must_use]
    pub fn optimization_info(mut self, value: OptimizationInfoKind) -> Self {
        self.optimization_info.push(value);
        self
    }

    /// Controls whether diagnostics display their numeric identifiers.
    #[must_use]
    pub fn display_error_number(mut self, value: bool) -> Self {
        self.display_error_number = Some(value);
        self
    }

    /// Appends a diagnostic number to promote to an error.
    #[must_use]
    pub fn diag_error(mut self, value: i32) -> Self {
        self.diag_error.push(value);
        self
    }

    /// Appends a diagnostic number to suppress.
    #[must_use]
    pub fn diag_suppress(mut self, value: i32) -> Self {
        self.diag_suppress.push(value);
        self
    }

    /// Appends a diagnostic number to demote to a warning.
    #[must_use]
    pub fn diag_warn(mut self, value: i32) -> Self {
        self.diag_warn.push(value);
        self
    }

    /// Controls whether to use brief diagnostic output.
    #[must_use]
    pub fn brief_diagnostics(mut self, value: bool) -> Self {
        self.brief_diagnostics = Some(value);
        self
    }

    /// Sets the file for compiler phase timing output.
    #[must_use]
    pub fn time(mut self, value: &'a str) -> Self {
        self.time = Some(value);
        self
    }

    /// Sets the maximum number of parallel optimization threads. Zero uses the available system threads; one disables splitting.
    #[must_use]
    pub fn split_compile(mut self, value: i32) -> Self {
        self.split_compile = Some(value);
        self
    }

    /// Enables or disables syntax-only device compilation without valid output code.
    #[must_use]
    pub fn device_syntax_only(mut self, value: bool) -> Self {
        self.device_syntax_only = value;
        self
    }

    /// Enables or disables minimal compilation, which omits texture, surface, device-runtime, device-launch, and runtime/driver type features.
    #[must_use]
    pub fn minimal(mut self, value: bool) -> Self {
        self.minimal = value;
        self
    }

    /// Enables or disables device stack protection.
    #[must_use]
    pub fn device_stack_protector(mut self, value: bool) -> Self {
        self.device_stack_protector = Some(value);
        self
    }

    /// Sets the file for the device compilation time trace.
    #[must_use]
    pub fn device_time_trace(mut self, value: &'a str) -> Self {
        self.device_time_trace = Some(value);
        self
    }

    /// Appends a native option string after the typed options.
    #[must_use]
    pub fn raw_option(mut self, value: &'a str) -> Self {
        self.raw_options.push(value);
        self
    }

    /// Renders configured options as owned NVRTC command-line argument strings. This performs no native compilation or option validation.
    #[must_use]
    #[expect(
        clippy::too_many_lines,
        reason = "NVRTC option switches are emitted from one complete option catalog"
    )]
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

        if let Some(mode) = self.tile_compilation {
            arguments.push(
                match mode {
                    TileCompilation::Enabled => "--enable-tile",
                    TileCompilation::TileOnly => "--tile-only",
                    TileCompilation::SimtOnly => "--simt-only",
                }
                .into(),
            );
        }
        if self.default_tile {
            arguments.push("--default-tile".into());
        }
        if self.diagnose_implicit_tile_variable {
            arguments.push("-diagnose-implicit-tile-var".into());
        }
        arguments.extend(self.raw_options.iter().map(|value| (*value).to_string()));
        arguments
    }
}

/// Owned NVRTC program with copied source and virtual headers. The native program is created lazily on the first operation that needs its handle and destroyed when this wrapper is dropped. Builder changes made after materialization do not recreate the existing native program.
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
    #[must_use]
    pub fn from_source(source: &str) -> Self {
        Self {
            source: source.to_string(),
            name: None,
            headers: Vec::new(),
            handle: UnsafeCell::new(ptr::null_mut()),
        }
    }

    /// Copies the source filename used in diagnostics when the native program is first created. An already materialized program keeps its existing name.
    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Copies a virtual header's source and include name for native program creation. Add headers before materializing the program.
    #[must_use]
    pub fn with_header(mut self, header: Header<'_>) -> Self {
        self.headers.push(OwnedHeader {
            source: header.source.to_string(),
            include_name: header.include_name.to_string(),
        });
        self
    }

    /// Copies virtual headers in slice order for native program creation. Add headers before materializing the program.
    #[must_use]
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
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `handle` is null.
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

    /// Compiles the program synchronously using the supplied native option strings. Call `log` to inspect compiler diagnostics after a successful or failed compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, an option contains an interior NUL, the option count exceeds the native integer range, or NVRTC rejects an option or fails compilation. Compilation diagnostics remain available from `log`.
    pub fn compile(&self, options: &[&str]) -> Result<()> {
        self.compile_raw(options)
    }

    /// Renders typed options and compiles the program synchronously. Call `log` to inspect compiler diagnostics.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, an option contains an interior NUL, the option count exceeds the native integer range, or NVRTC rejects an option or fails compilation. Compilation diagnostics remain available from `log`.
    pub fn compile_with_options(&self, options: &CompileOptions<'_>) -> Result<()> {
        let arguments = options.as_arguments();
        let argument_refs = arguments.iter().map(String::as_str).collect::<Vec<_>>();
        self.compile_raw(&argument_refs)
    }

    /// Compiles synchronously while allowing another thread to request cancellation.
    ///
    /// Set `cancel` to true to request cancellation, and leave it true until this
    /// call returns. The wrapper registers its own native callback and attempts to
    /// replace it with a no-op callback after compilation, including failed attempts.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, an option contains an interior NUL, the option count exceeds the native integer range, or NVRTC rejects an option or fails compilation. Compilation diagnostics remain available from `log`. Errors from registering or clearing the cancellation callback are also returned. A compilation error takes precedence over a later callback-clear error.
    pub fn compile_with_options_and_cancel_flag(
        &self,
        options: &CompileOptions<'_>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        // SAFETY: `nvrtcSetFlowCallback` receives live handles owned by the wrappers.
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
    /// Returns an error if `name_expression` contains an interior NUL byte, if
    /// the program handle is invalid, or if NVRTC rejects the expression.
    pub fn add_name_expression(&self, name_expression: &str) -> Result<()> {
        let name_expression = CString::new(name_expression)?;
        // SAFETY: `nvrtcAddNameExpression` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
        // SAFETY: the program handle is live, and the name is NUL-terminated.
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
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`. The requested code format must have been generated by the preceding compilation.
    pub fn ptx(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetPTXSize, sys::nvrtcGetPTX)
    }

    /// Copies the last compilation's PTX bytes into an owned module image.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`. The requested code format must have been generated by the preceding compilation.
    pub fn ptx_image(&self) -> Result<ModuleImage<'static>> {
        Ok(ModuleImage::from_vec(self.ptx()?))
    }

    /// Returns the last compilation's PTX as a string, removing its trailing NUL and replacing invalid UTF-8 lossily.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`. The requested code format must have been generated by the preceding compilation.
    pub fn ptx_string(&self) -> Result<String> {
        Ok(bytes_to_string(self.ptx()?))
    }

    /// Returns the GPU machine-code binary produced by the last compilation for a real GPU architecture target.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`. The requested code format must have been generated by the preceding compilation.
    pub fn cubin(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetCUBINSize, sys::nvrtcGetCUBIN)
    }

    /// Copies the last compilation's GPU machine-code binary into an owned module image.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`. The requested code format must have been generated by the preceding compilation.
    pub fn cubin_image(&self) -> Result<ModuleImage<'static>> {
        Ok(ModuleImage::from_vec(self.cubin()?))
    }

    /// Returns the `cuda_tile` IR generated by the previous compilation of this program.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`. The requested code format must have been generated by the preceding compilation.
    pub fn tile_ir(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetTileIRSize, sys::nvrtcGetTileIR)
    }

    /// Returns the device link-time intermediate representation produced by the last compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`. The requested code format must have been generated by the preceding compilation.
    pub fn lto_ir(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetLTOIRSize, sys::nvrtcGetLTOIR)
    }

    /// Copies the last compilation's device link-time intermediate representation into an owned module image.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`. The requested code format must have been generated by the preceding compilation.
    pub fn lto_ir_image(&self) -> Result<ModuleImage<'static>> {
        Ok(ModuleImage::from_vec(self.lto_ir()?))
    }

    /// Returns the `OptiX` intermediate representation produced by the last compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`. The requested code format must have been generated by the preceding compilation.
    pub fn optix_ir(&self) -> Result<Vec<u8>> {
        self.bytes(sys::nvrtcGetOptiXIRSize, sys::nvrtcGetOptiXIR)
    }

    /// Copies the last compilation's `OptiX` intermediate representation into an owned module image.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails, NVRTC cannot query or copy the requested output, or the native output size cannot be represented as `usize`. The requested code format must have been generated by the preceding compilation.
    pub fn optix_ir_image(&self) -> Result<ModuleImage<'static>> {
        Ok(ModuleImage::from_vec(self.optix_ir()?))
    }

    /// Returns the program compilation log, removing its trailing NUL and replacing invalid UTF-8 lossily. The log may contain diagnostics from either a successful or failed compilation.
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
    /// In this scenario, use [`Program::pch_heap_size_required`] to query the required heap size, reallocate the heap with [`set_pch_heap_size`], and retry PCH creation with [`sys::nvrtcCompileProgram`](singe_cuda_sys::nvrtc::nvrtcCompileProgram) on a new NVRTC program instance.
    /// [`Status::PchCreate`] indicates that an error condition prevented the PCH file from being created.
    ///
    /// # Errors
    ///
    /// Returns an error if the program handle is invalid.
    pub fn pch_create_status(&self) -> Result<Status> {
        // SAFETY: `nvrtcGetPCHCreateStatus` receives live handles owned by the wrappers.
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
        // SAFETY: `nvrtcGetPCHHeapSizeRequired` receives live handles owned by the wrappers and valid output locations.
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
        // SAFETY: `Program` is not `Sync`, and no `&mut` to the cell is live while this shared borrow exists.
        let handle = unsafe { *self.handle.get() };
        if handle.is_null() { None } else { Some(handle) }
    }

    /// Transfers ownership of the raw NVRTC program handle to the caller.
    ///
    /// If the program has not been materialized yet, this creates the raw
    /// handle first. The caller becomes responsible for destroying the returned
    /// handle with `nvrtcDestroyProgram`.
    ///
    /// # Errors
    ///
    /// Returns an error if lazy program creation fails because source, filenames, or headers contain an interior NUL, the header count exceeds the native integer range, or NVRTC cannot create the program.
    pub fn into_raw(self) -> Result<sys::nvrtcProgram> {
        let handle = self.handle()?;
        forget(self);
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

        // SAFETY: `nvrtcCompileProgram` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
        get_data: unsafe extern "C" fn(sys::nvrtcProgram, *mut i8) -> sys::nvrtcResult,
    ) -> Result<Vec<u8>> {
        let mut size = 0;
        // SAFETY: the program handle is live, and `size` is a writable local.
        unsafe {
            try_nvrtc!(get_size(self.handle()?, &raw mut size))?;
        }

        let mut bytes = vec![0u8; to_usize(size, "NVRTC output byte count")?];
        if bytes.is_empty() {
            return Ok(bytes);
        }

        // SAFETY: `bytes` has the size the library reported for this output.
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
        // SAFETY: `nvrtcCreateProgram` receives valid output locations and pointers into buffers borrowed for the call.
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
        // SAFETY: `nvrtcDestroyProgram` receives only plain values.
        unsafe {
            let handle = self.handle.get();
            if !(*handle).is_null()
                && let Err(error) = try_nvrtc!(sys::nvrtcDestroyProgram(handle))
            {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy nvrtc program: {error}");
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
        /// Owned output image bytes.
        ModuleImage<'static>,
    ),
    /// `OptiX` intermediate representation.
    OptixIr(
        /// Owned output image bytes.
        ModuleImage<'static>,
    ),
    /// The `cuda_tile` IR generated by the previous compilation.
    TileIr(
        /// Owned output image bytes.
        ModuleImage<'static>,
    ),
}

impl CompilationArtifact {
    /// Borrows the owned module image regardless of its compilation format.
    #[must_use]
    pub fn image(&self) -> &ModuleImage<'static> {
        match self {
            Self::Ptx(image)
            | Self::Cubin(image)
            | Self::LtoIr(image)
            | Self::OptixIr(image)
            | Self::TileIr(image) => image,
        }
    }

    /// Consumes the artifact and returns its owned module image.
    #[must_use]
    pub fn into_image(self) -> ModuleImage<'static> {
        match self {
            Self::Ptx(image)
            | Self::Cubin(image)
            | Self::LtoIr(image)
            | Self::OptixIr(image)
            | Self::TileIr(image) => image,
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
    /// Returns the requested compilation output in an owned, format-tagged module image.
    ///
    /// # Errors
    ///
    /// Propagates errors from retrieving the selected native output. Also returns `Error::InvalidValue` for an empty PTX, cubin, LTO IR, or `OptiX` IR image. Tile IR is returned without that additional empty-image check.
    pub fn artifact(&self, kind: OutputKind) -> Result<CompilationArtifact> {
        match kind {
            OutputKind::TileIr => Ok(CompilationArtifact::TileIr(ModuleImage::from_vec(
                self.tile_ir()?,
            ))),
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
    // SAFETY: `nvrtcVersion` receives valid output locations.
    unsafe {
        try_nvrtc!(sys::nvrtcVersion(&raw mut major, &raw mut minor))?;
    }
    Ok(Version { major, minor })
}

/// Returns the GPU architecture numbers supported by the loaded NVRTC library.
///
/// # Errors
///
/// Returns an error if NVRTC cannot report the supported architecture count or fill the architecture list.
pub fn supported_architectures() -> Result<Vec<i32>> {
    let mut count = 0;
    // SAFETY: `nvrtcGetNumSupportedArchs` receives valid output locations.
    unsafe {
        try_nvrtc!(sys::nvrtcGetNumSupportedArchs(&raw mut count))?;
    }

    let mut architectures = vec![0; count as usize];
    if architectures.is_empty() {
        return Ok(Vec::new());
    }

    // SAFETY: `nvrtcGetSupportedArchs` receives valid output locations.
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
    // SAFETY: `nvrtcGetPCHHeapSize` receives valid output locations.
    unsafe {
        try_nvrtc!(sys::nvrtcGetPCHHeapSize(&raw mut size))?;
    }
    to_usize(size, "NVRTC output byte count")
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
    // SAFETY: `nvrtcSetPCHHeapSize` receives only plain values.
    unsafe { try_nvrtc!(sys::nvrtcSetPCHHeapSize(size as _)) }
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

    // SAFETY: the payload is the cancellation flag registered with this callback, which outlives the compilation.
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
    // SAFETY: `nvrtcSetFlowCallback` receives only plain values.
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
        device::Device,
        error::{Error, Result},
        jit::JitOptions,
        memory::DeviceMemory,
        module::LaunchConfig,
        testing,
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

    fn compile_or_skip(result: Result<()>) -> bool {
        match result {
            Ok(()) => true,
            Err(Error::Nvrtc {
                code: Status::BuiltinOperationFailure,
                ..
            }) => {
                eprintln!(
                    "warning: skipping NVRTC compile-dependent test because NVRTC builtins are unavailable"
                );
                false
            }
            Err(error) => panic!("NVRTC compilation failed: {error:?}"),
        }
    }

    #[test]
    fn test_version_is_available() {
        let version = version().unwrap();
        assert_ne!(version.major, 0);
    }

    #[test]
    fn test_supported_architectures_are_sorted() {
        let architectures = supported_architectures().unwrap();
        assert!(!architectures.is_empty());
        assert!(
            architectures
                .windows(2)
                .all(|window| window[0] <= window[1])
        );
    }

    #[test]
    fn test_compile_options_build_expected_arguments() {
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
    fn test_compiles_to_ptx() {
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
        if !compile_or_skip(program.compile_with_options(&options)) {
            return;
        }

        let ptx = program.ptx_string().unwrap();
        assert!(ptx.contains(".visible .entry saxpy"));
    }

    #[test]
    fn test_compile_with_cancel_flag_succeeds_when_not_cancelled() {
        let (_lock, _ctx) = testing::bootstrap().unwrap();
        let cancel = AtomicBool::new(false);
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop.cu");
        let options = CompileOptions::new().gpu_architecture(GpuArchitecture::Compute80);

        if !compile_or_skip(program.compile_with_options_and_cancel_flag(&options, &cancel)) {
            return;
        }
        assert!(program.ptx_string().unwrap().contains("noop"));
    }

    #[test]
    fn test_clear_flow_callback_is_allowed_before_compilation() {
        let program =
            Program::from_source("extern \"C\" __global__ void noop() {}").with_name("noop.cu");
        clear_flow_callback(&program).unwrap();
    }

    #[test]
    fn test_cubin_artifact_loads_as_module() {
        let (_lock, context) = testing::bootstrap().unwrap();
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
        if !compile_or_skip(program.compile_with_options(&options)) {
            return;
        }

        let module = context
            .load_nvrtc_module(&program, OutputKind::Cubin)
            .unwrap();
        assert!(module.function("saxpy").is_ok());
    }

    #[test]
    fn test_cubin_artifact_loads_as_module_with_jit_options() {
        let (_lock, context) = testing::bootstrap().unwrap();
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_module_jit.cu");
        let architecture = current_device_sm_architecture().unwrap();
        let options = CompileOptions::new().gpu_architecture(architecture);
        if !compile_or_skip(program.compile_with_options(&options)) {
            return;
        }

        let mut info_log = [0u8; 1024];
        let mut error_log = [0u8; 1024];
        let jit_options = JitOptions::default()
            .with_generate_line_info(true)
            .with_info_log(&mut info_log)
            .with_error_log(&mut error_log);

        let module = context
            .load_nvrtc_module_with_options(&program, OutputKind::Cubin, jit_options)
            .unwrap();
        assert!(module.function("noop").is_ok());
    }

    #[test]
    fn test_compiles_loads_launches_and_reads_back_results() {
        let (_lock, context) = testing::bootstrap().unwrap();

        let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
        let mut output = vec![0.0f32; input.len()];
        let scalar = 2.5f32;
        let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
        let output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();
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
        if !compile_or_skip(program.compile_with_options(&compile_options)) {
            return;
        }

        let module = context
            .load_nvrtc_module(&program, OutputKind::Cubin)
            .unwrap();
        let function = module.function("scale_add").unwrap();

        let config = LaunchConfig::for_1d_grid(input.len(), 128);
        let input_ptr = input_device.as_ptr();
        let mut output_ptr = output_device.as_mut_ptr();

        // SAFETY: the compiled declaration matches these arguments, and both allocations
        // cover `length` elements and remain exclusively available until the readback completes.
        unsafe { function.launch(&config, (&input_ptr, &mut output_ptr, &scalar, &length)) }
            .unwrap();
        output_device.copy_to_host(&mut output).unwrap();

        let expected = input
            .iter()
            .map(|value| value * scalar + 1.0)
            .collect::<Vec<_>>();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_cubin_artifact_loads_as_library() {
        let (_lock, context) = testing::bootstrap().unwrap();
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_library.cu");
        let architecture = current_device_sm_architecture().unwrap();
        let options = CompileOptions::new().gpu_architecture(architecture);
        if !compile_or_skip(program.compile_with_options(&options)) {
            return;
        }

        let library = context
            .load_nvrtc_library(&program, OutputKind::Cubin)
            .unwrap();
        assert!(library.kernel_count().unwrap() >= 1);
    }

    #[test]
    fn test_lto_ir_artifact_is_available_when_requested() {
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_lto.cu");
        let options = CompileOptions::new().dlink_time_optimization(true);
        if !compile_or_skip(program.compile_with_options(&options)) {
            return;
        }

        let artifact = program.artifact(OutputKind::LtoIr).unwrap();
        assert!(!artifact.image().as_bytes().is_empty());
        let ptx = program.artifact(OutputKind::Ptx).unwrap();
        assert!(!ptx.image().as_bytes().is_empty());
    }

    #[test]
    fn test_optix_ir_artifact_is_available_when_requested() {
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_optix.cu");
        let options = CompileOptions::new().optix_ir(true);
        if !compile_or_skip(program.compile_with_options(&options)) {
            return;
        }

        let artifact = program.artifact(OutputKind::OptixIr).unwrap();
        assert!(!artifact.image().as_bytes().is_empty());
        let ptx = program.artifact(OutputKind::Ptx).unwrap();
        assert!(!ptx.image().as_bytes().is_empty());
    }

    #[test]
    fn test_cubin_artifact_requires_real_architecture() {
        let program = Program::from_source(
            r#"
            extern "C" __global__ void noop() {}
            "#,
        )
        .with_name("noop_cubin.cu");
        let options = CompileOptions::new().gpu_architecture(GpuArchitecture::Compute80);
        if !compile_or_skip(program.compile_with_options(&options)) {
            return;
        }

        assert!(program.artifact(OutputKind::Cubin).is_err());
    }
}
