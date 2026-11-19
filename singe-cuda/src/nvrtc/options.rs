use std::{
    fmt::Display,
    sync::{Arc, atomic::AtomicBool},
};

use singe_core::impl_enum_display;

use crate::architecture::GpuArchitecture;

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

/// Typed NVRTC compilation options.
/// Unset optional values and disabled flag switches leave the corresponding native option unspecified.
/// String lists preserve insertion order; `raw_options` are appended last.
/// Validation occurs when the program is materialized or compiled.
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
    /// Emits `--dopt=on` only for `Some(true)`.
    /// `None` and `Some(false)` leave the native default in effect.
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
    /// Flag another thread sets to true to cancel compilation; it must stay true until compilation returns.
    /// It is not rendered as an argument.
    pub cancel_flag: Option<Arc<AtomicBool>>,
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
            cancel_flag: None,
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

    /// Requests `--dopt=on` when `value` is true.
    /// False leaves the native optimization default in effect.
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

    /// Sets the maximum number of parallel optimization threads.
    /// Zero uses the available system threads; one disables splitting.
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

    /// Sets the flag another thread sets to true to cancel compilation.
    #[must_use]
    pub fn cancel_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cancel_flag = Some(flag);
        self
    }

    /// Renders configured options as owned NVRTC command-line argument strings.
    /// This performs no native compilation or option validation.
    #[must_use]
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
            arguments.push(format!("--diag-error={}", join_display(&self.diag_error)));
        }

        if !self.diag_suppress.is_empty() {
            arguments.push(format!(
                "--diag-suppress={}",
                join_display(&self.diag_suppress)
            ));
        }

        if !self.diag_warn.is_empty() {
            arguments.push(format!("--diag-warn={}", join_display(&self.diag_warn)));
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
