//! CUDA driver JIT options and device-code linking.

#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use std::{marker::PhantomData, ptr};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_cuda_sys::{driver, runtime};

use singe_core::impl_enum_display;

/// CUDA driver JIT compilation and loading options with borrowed diagnostic outputs.
///
/// Unset fields omit the native option and leave its default to CUDA.
/// Builders do not reject incompatible option combinations; CUDA validates them when loading code.
#[derive(Debug, Default)]
pub struct JitOptions<'a> {
    /// Maximum registers per thread requested from the JIT compiler.
    pub max_registers: Option<u32>,
    /// Minimum thread-block size targeted by compilation.
    /// The native actual target is returned in the encoded value array; this input setting is not updated.
    /// Cannot be combined with `target`.
    pub threads_per_block: Option<u32>,
    /// Output for elapsed JIT compiler and linker time in milliseconds.
    /// The encoded artifact copies the native value slot here when it drops.
    pub wall_time: Option<&'a mut f32>,
    /// Borrowed output buffer for informational JIT diagnostics.
    /// Its capacity includes any NUL terminator and is capped at `u32::MAX` bytes when encoded.
    pub info_log_buffer: Option<&'a mut [u8]>,
    /// Borrowed output buffer for JIT error diagnostics.
    /// Its capacity includes any NUL terminator and is capped at `u32::MAX` bytes when encoded.
    pub error_log_buffer: Option<&'a mut [u8]>,
    /// Requested optimization level, clamped to `0..=4` during encoding.
    /// CUDA defaults to level four when the option is omitted.
    pub optimization_level: Option<u32>,
    /// Presence of `Some(())` emits the option selecting the current CUDA context's target.
    /// No native value is required.
    pub target_from_cuda_context: Option<()>,
    /// Explicit JIT target architecture.
    /// Cannot be combined with `threads_per_block`.
    pub target: Option<JitTarget>,
    /// Choice between PTX compilation and compatible binary fallback when no exact cubin match exists.
    pub fallback_strategy: Option<JitFallback>,
    /// Whether to emit device debugging information.
    /// `None` leaves the native default unchanged.
    pub generate_debug_info: Option<bool>,
    /// Whether to request verbose JIT diagnostics.
    /// `None` leaves the native default unchanged.
    pub log_verbose: Option<bool>,
    /// Whether to emit line-number information for the compiled device code.
    pub generate_line_info: Option<bool>,
    /// Explicit default cache mode for generated global-memory loads.
    pub cache_mode: Option<JitCacheMode>,
}

/// Parallel CUDA JIT option arrays borrowing their output buffers.
#[derive(Debug)]
pub struct JitOptionsArtifact<'a> {
    /// Native option identifiers, in the same order as `values`.
    pub names: Vec<driver::CUjit_option>,
    /// Native value slots.
    /// Integers and returned wall time occupy the slot bits; log-buffer entries point to borrowed output storage.
    pub values: Vec<*mut ()>,
    wall_time: Option<&'a mut f32>,
    _log_buffers: PhantomData<&'a mut [u8]>,
}

impl Drop for JitOptionsArtifact<'_> {
    fn drop(&mut self) {
        if let Some(output) = self.wall_time.as_deref_mut()
            && let Some(index) = self
                .names
                .iter()
                .position(|&name| name == JitOption::WallTime.into())
            && let Some(value) = self.values.get(index)
        {
            let bytes = value.addr().to_ne_bytes();
            *output = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
    }
}

impl<'a> JitOptions<'a> {
    /// Sets the requested maximum registers per thread.
    #[must_use]
    pub fn with_max_registers(mut self, value: u32) -> Self {
        self.max_registers = Some(value);
        self
    }

    /// Sets the minimum thread-block size targeted by compilation.
    /// See [`JitOption::ThreadsPerBlock`] for its resource and option restrictions.
    #[must_use]
    pub fn with_threads_per_block(mut self, value: u32) -> Self {
        self.threads_per_block = Some(value);
        self
    }

    /// Selects borrowed storage for elapsed JIT time in milliseconds.
    /// The encoded artifact copies the native output here when it drops.
    #[must_use]
    pub fn with_wall_time(mut self, value: &'a mut f32) -> Self {
        self.wall_time = Some(value);
        self
    }

    /// Selects a borrowed informational diagnostic buffer.
    /// The encoded capacity includes the terminating NUL and is capped at `u32::MAX` bytes.
    #[must_use]
    pub fn with_info_log(mut self, buffer: &'a mut [u8]) -> Self {
        self.info_log_buffer = Some(buffer);
        self
    }

    /// Selects a borrowed error diagnostic buffer.
    /// The encoded capacity includes the terminating NUL and is capped at `u32::MAX` bytes.
    #[must_use]
    pub fn with_error_log(mut self, buffer: &'a mut [u8]) -> Self {
        self.error_log_buffer = Some(buffer);
        self
    }

    /// Sets the optimization level, clamping values above four to four.
    #[must_use]
    pub fn with_optimization_level(mut self, level: u32) -> Self {
        // Clamp to 0-4 as per docs.
        self.optimization_level = Some(level.min(4));
        self
    }

    /// Emits the option that selects the current CUDA context's target architecture.
    #[must_use]
    pub const fn with_target_from_cuda_context(mut self) -> Self {
        self.target_from_cuda_context = Some(());
        self
    }

    /// Sets an explicit JIT target architecture.
    /// CUDA rejects combining this option with the thread-block compilation target.
    #[must_use]
    pub fn with_target(mut self, target: JitTarget) -> Self {
        self.target = Some(target);
        self
    }

    /// Selects PTX compilation or compatible binary fallback when no exact cubin match exists.
    #[must_use]
    pub fn with_fallback_strategy(mut self, strategy: JitFallback) -> Self {
        self.fallback_strategy = Some(strategy);
        self
    }

    /// Enables or disables device debugging information in the generated code.
    #[must_use]
    pub fn with_generate_debug_info(mut self, enable: bool) -> Self {
        self.generate_debug_info = Some(enable);
        self
    }

    /// Enables or disables verbose JIT diagnostic messages.
    #[must_use]
    pub fn with_log_verbose(mut self, enable: bool) -> Self {
        self.log_verbose = Some(enable);
        self
    }

    /// Enables or disables line-number information in the generated code.
    #[must_use]
    pub fn with_generate_line_info(mut self, enable: bool) -> Self {
        self.generate_line_info = Some(enable);
        self
    }

    /// Sets the default cache mode for generated global-memory loads.
    #[must_use]
    pub fn with_cache_mode(mut self, mode: JitCacheMode) -> Self {
        self.cache_mode = Some(mode);
        self
    }

    /// Prepares parallel native option and value arrays.
    ///
    /// Scalar options occupy the pointer-sized value slots directly.
    /// Log-buffer entries point to borrowed output storage.
    /// CUDA returns elapsed time in its value slot, which is copied to `wall_time` when the artifact drops.
    /// Moving the artifact preserves its encoded values and pointers.
    /// Keep the arrays and their borrowed storage live for the native operation.
    /// Output options passed to a native link state must remain live until that link state is destroyed.
    pub fn build(&mut self) -> JitOptionsArtifact<'_> {
        let mut artifact = JitOptionsArtifact {
            names: Vec::new(),
            values: Vec::new(),
            wall_time: self.wall_time.as_deref_mut(),
            _log_buffers: PhantomData,
        };
        let mut push = |option: JitOption, value: *mut ()| {
            artifact.names.push(option.into());
            artifact.values.push(value);
        };

        if let Some(value) = self.max_registers {
            push(
                JitOption::MaxRegisters,
                ptr::without_provenance_mut(value as usize),
            );
        }

        if let Some(value) = self.threads_per_block {
            push(
                JitOption::ThreadsPerBlock,
                ptr::without_provenance_mut(value as usize),
            );
        }

        if artifact.wall_time.is_some() {
            push(JitOption::WallTime, ptr::null_mut());
        }

        if let Some(buffer) = self.info_log_buffer.as_deref_mut() {
            push(JitOption::InfoLogBuffer, buffer.as_mut_ptr().cast());
            let size = u32::try_from(buffer.len()).unwrap_or(u32::MAX);
            push(
                JitOption::InfoLogBufferSizeBytes,
                ptr::without_provenance_mut(size as usize),
            );
        }

        if let Some(buffer) = self.error_log_buffer.as_deref_mut() {
            push(JitOption::ErrorLogBuffer, buffer.as_mut_ptr().cast());
            let size = u32::try_from(buffer.len()).unwrap_or(u32::MAX);
            push(
                JitOption::ErrorLogBufferSizeBytes,
                ptr::without_provenance_mut(size as usize),
            );
        }

        if let Some(value) = self.optimization_level {
            push(
                JitOption::OptimizationLevel,
                ptr::without_provenance_mut(value.min(4) as usize),
            );
        }

        if self.target_from_cuda_context.is_some() {
            push(JitOption::TargetFromCudaContext, ptr::null_mut());
        }

        if let Some(value) = self.target {
            push(
                JitOption::Target,
                ptr::without_provenance_mut(value as usize),
            );
        }

        if let Some(value) = self.fallback_strategy {
            push(
                JitOption::FallbackStrategy,
                ptr::without_provenance_mut(value as usize),
            );
        }

        if let Some(value) = self.generate_debug_info {
            push(
                JitOption::GenerateDebugInfo,
                ptr::without_provenance_mut(usize::from(value)),
            );
        }

        if let Some(value) = self.log_verbose {
            push(
                JitOption::LogVerbose,
                ptr::without_provenance_mut(usize::from(value)),
            );
        }

        if let Some(value) = self.generate_line_info {
            push(
                JitOption::GenerateLineInfo,
                ptr::without_provenance_mut(usize::from(value)),
            );
        }

        if let Some(value) = self.cache_mode {
            push(
                JitOption::CacheMode,
                ptr::without_provenance_mut(value as usize),
            );
        }
        artifact
    }
}

singe_core::native_enum! {
    /// CUDA JIT option selectors for native compilation, linking, and module loading.
    ///
    /// Each selector defines the type and direction of its corresponding native value slot.
    /// The selected native operation determines which options it accepts.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum JitOption: runtime::cudaJitOption {
        /// Max number of registers that a thread may use.
        ///
        /// Native value type: unsigned int
        ///
        /// Applies to: compiler only
        MaxRegisters = CU_JIT_MAX_REGISTERS,
        /// Input: Specifies minimum number of threads per block to target compilation for Output: Returns the number of threads the compiler actually targeted.
        /// This restricts the resource utilization of the compiler (e.g. max registers) such that a block with the given number of threads should be able to launch based on register limitations.
        /// Note, this option does not currently take into account any other resource limitations, such as shared memory utilization.
        /// Cannot be combined with [`Self::Target`].
        ///
        /// Native value type: unsigned int
        ///
        /// Applies to: compiler only
        ThreadsPerBlock = CU_JIT_THREADS_PER_BLOCK,
        /// Overwrites the option value with the total wall clock time, in milliseconds, spent in the compiler and linker
        ///
        /// Native value type: float
        ///
        /// Applies to: compiler and linker
        WallTime = CU_JIT_WALL_TIME,
        /// Pointer to a buffer in which to print any log messages that are informational in nature (the buffer size is specified via option [`Self::InfoLogBufferSizeBytes`])
        ///
        /// Native value type: char *
        ///
        /// Applies to: compiler and linker
        InfoLogBuffer = CU_JIT_INFO_LOG_BUFFER,
        /// Input: Log buffer size in bytes.
        /// Log messages will be capped at this size (including null terminator) Output: Amount of log buffer filled with messages
        ///
        /// Native value type: unsigned int
        ///
        /// Applies to: compiler and linker
        InfoLogBufferSizeBytes = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        /// Pointer to a buffer in which to print any log messages that reflect errors (the buffer size is specified via option [`Self::ErrorLogBufferSizeBytes`])
        ///
        /// Native value type: char *
        ///
        /// Applies to: compiler and linker
        ErrorLogBuffer = CU_JIT_ERROR_LOG_BUFFER,
        /// Input: Log buffer size in bytes.
        /// Log messages will be capped at this size (including null terminator) Output: Amount of log buffer filled with messages
        ///
        /// Native value type: unsigned int
        ///
        /// Applies to: compiler and linker
        ErrorLogBufferSizeBytes = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        /// Level of optimizations to apply to generated code (0 - 4), with 4 being the default and highest level of optimizations.
        ///
        /// Native value type: unsigned int
        ///
        /// Applies to: compiler only
        OptimizationLevel = CU_JIT_OPTIMIZATION_LEVEL,
        /// No option value required.
        /// Determines the target based on the current attached context (default)
        ///
        /// Native value type: No option value needed
        ///
        /// Applies to: compiler and linker
        TargetFromCudaContext = CU_JIT_TARGET_FROM_CUCONTEXT,
        /// Target is chosen based on supplied [`JitTarget`].
        /// Cannot be combined with [`Self::ThreadsPerBlock`].
        ///
        /// Native value type: unsigned int for enumerated type [`JitTarget`]
        ///
        /// Applies to: compiler and linker
        Target = CU_JIT_TARGET,
        /// Specifies choice of fallback strategy if matching cubin is not found.
        /// Choice is based on supplied [`JitFallback`].
        /// This option cannot be used with native linker APIs as the linker requires exact matches.
        ///
        /// Native value type: unsigned int for enumerated type [`JitFallback`]
        ///
        /// Applies to: compiler only
        FallbackStrategy = CU_JIT_FALLBACK_STRATEGY,
        /// Specifies whether to create debug information in output (-g) (0: false, default)
        ///
        /// Native value type: int
        ///
        /// Applies to: compiler and linker
        GenerateDebugInfo = CU_JIT_GENERATE_DEBUG_INFO,
        /// Generate verbose log messages (0: false, default)
        ///
        /// Native value type: int
        ///
        /// Applies to: compiler and linker
        LogVerbose = CU_JIT_LOG_VERBOSE,
        /// Generate line number information (-lineinfo) (0: false, default)
        ///
        /// Native value type: int
        ///
        /// Applies to: compiler only
        GenerateLineInfo = CU_JIT_GENERATE_LINE_INFO,
        /// Specifies whether to enable caching explicitly (-dlcm) Choice is based on supplied [`JitCacheMode`].
        ///
        /// Native value type: unsigned int for enumerated type [`JitCacheMode`]
        ///
        /// Applies to: compiler only
        CacheMode = CU_JIT_CACHE_MODE,
        /// Deprecated: This jit option is deprecated and should not be used.
        #[deprecated]
        NewSm3xOpt = CU_JIT_NEW_SM3X_OPT,
        /// This jit option is used for internal purpose only.
        FastCompile = CU_JIT_FAST_COMPILE,
        /// Array of device symbol names that will be relocated to the corresponding host addresses stored in [`Self::GlobalSymbolAddresses`].
        /// Must contain [`Self::GlobalSymbolCount`] entries.
        /// When loading a device module, driver will relocate all encountered unresolved symbols to the host addresses.
        /// It is only allowed to register symbols that correspond to unresolved global variables.
        /// It is illegal to register the same device symbol at multiple addresses.
        ///
        /// Native value type: const char **
        ///
        /// Applies to: dynamic linker only
        GlobalSymbolNames = CU_JIT_GLOBAL_SYMBOL_NAMES,
        /// Array of host addresses that will be used to relocate corresponding device symbols stored in [`Self::GlobalSymbolNames`].
        /// Must contain [`Self::GlobalSymbolCount`] entries.
        ///
        /// Native value type: void **
        ///
        /// Applies to: dynamic linker only
        GlobalSymbolAddresses = CU_JIT_GLOBAL_SYMBOL_ADDRESSES,
        /// Number of entries in [`Self::GlobalSymbolNames`] and [`Self::GlobalSymbolAddresses`] arrays.
        ///
        /// Native value type: unsigned int
        ///
        /// Applies to: dynamic linker only
        GlobalSymbolCount = CU_JIT_GLOBAL_SYMBOL_COUNT,
        /// Deprecated: Enable link-time optimization (-dlto) for device code (Disabled by default).
        /// This option is not supported on 32-bit platforms.
        ///
        /// Native value type: int
        ///
        /// Applies to: compiler and linker
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        #[deprecated]
        Lto = CU_JIT_LTO,
        /// Deprecated: Control single-precision denormals (-ftz) support (0: false, default). 1 : flushes denormal values to zero 0 : preserves denormal values
        ///
        /// Native value type: int
        ///
        /// Applies to: link-time optimization specified with [`Self::Lto`]
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        #[deprecated]
        Ftz = CU_JIT_FTZ,
        /// Deprecated: Control single-precision floating-point division and reciprocals (-prec-div) support (1: true, default). 1 : Enables the IEEE round-to-nearest mode 0 : Enables the fast approximation mode
        ///
        /// Native value type: int
        ///
        /// Applies to: link-time optimization specified with [`Self::Lto`]
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        #[deprecated]
        PrecDiv = CU_JIT_PREC_DIV,
        /// Deprecated: Control single-precision floating-point square root (-prec-sqrt) support (1: true, default). 1 : Enables the IEEE round-to-nearest mode 0 : Enables the fast approximation mode
        ///
        /// Native value type: int
        ///
        /// Applies to: link-time optimization specified with [`Self::Lto`]
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        #[deprecated]
        PrecSqrt = CU_JIT_PREC_SQRT,
        /// Deprecated: Enable/Disable the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add (-fma) operations (1: Enable, default; 0: Disable).
        ///
        /// Native value type: int
        ///
        /// Applies to: link-time optimization specified with [`Self::Lto`]
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        #[deprecated]
        Fma = CU_JIT_FMA,
        /// Deprecated: Array of kernel names that should be preserved at link time while others can be removed.
        /// Must contain [`Self::ReferencedKernelCount`] entries.
        /// Note that kernel names can be mangled by the compiler in which case the mangled name needs to be specified.
        /// Wildcard "*" can be used to represent zero or more characters instead of specifying the full or mangled name.
        /// It is important to note that the wildcard "*" is also added implicitly.
        /// For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and thus preserve all kernels with those names.
        /// This can be avoided by providing a more specific name like "barfoobaz".
        ///
        /// Native value type: const char **
        ///
        /// Applies to: dynamic linker only
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        #[deprecated]
        ReferencedKernelNames = CU_JIT_REFERENCED_KERNEL_NAMES,
        /// Deprecated: Number of entries in [`Self::ReferencedKernelNames`] array.
        ///
        /// Native value type: unsigned int
        ///
        /// Applies to: dynamic linker only
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        #[deprecated]
        ReferencedKernelCount = CU_JIT_REFERENCED_KERNEL_COUNT,
        /// Deprecated: Array of variable names (__device__ and/or __constant__) that should be preserved at link time while others can be removed.
        /// Must contain [`Self::ReferencedVariableCount`] entries.
        /// Note that variable names can be mangled by the compiler in which case the mangled name needs to be specified.
        /// Wildcard "*" can be used to represent zero or more characters instead of specifying the full or mangled name.
        /// It is important to note that the wildcard "*" is also added implicitly.
        /// For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and thus preserve all variables with those names.
        /// This can be avoided by providing a more specific name like "barfoobaz".
        ///
        /// Native value type: const char **
        ///
        /// Applies to: link-time optimization specified with [`Self::Lto`]
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        #[deprecated]
        ReferencedVariableNames = CU_JIT_REFERENCED_VARIABLE_NAMES,
        /// Deprecated: Number of entries in [`Self::ReferencedVariableNames`] array.
        ///
        /// Native value type: unsigned int
        ///
        /// Applies to: link-time optimization specified with [`Self::Lto`]
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        #[deprecated]
        ReferencedVariableCount = CU_JIT_REFERENCED_VARIABLE_COUNT,
        /// Deprecated: This option serves as a hint to enable the JIT compiler/linker to remove constant (__constant__) and device (__device__) variables unreferenced in device code (Disabled by default).
        /// Note that host references to constant and device variables through APIs such as [`Module::global`](crate::module::Module::global) with this option specified may result in undefined behavior unless the variables are explicitly specified using [`Self::ReferencedVariableNames`].
        ///
        /// Native value type: int
        ///
        /// Applies to: link-time optimization specified with [`Self::Lto`]
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        #[deprecated]
        OptimizeUnusedDeviceVariables = CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES,
    }
}

impl_enum_display!(JitOption, {
    Self::MaxRegisters => "CU_JIT_MAX_REGISTERS",
    Self::ThreadsPerBlock => "CU_JIT_THREADS_PER_BLOCK",
    Self::WallTime => "CU_JIT_WALL_TIME",
    Self::InfoLogBuffer => "CU_JIT_INFO_LOG_BUFFER",
    Self::InfoLogBufferSizeBytes => "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES",
    Self::ErrorLogBuffer => "CU_JIT_ERROR_LOG_BUFFER",
    Self::ErrorLogBufferSizeBytes => "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES",
    Self::OptimizationLevel => "CU_JIT_OPTIMIZATION_LEVEL",
    Self::TargetFromCudaContext => "CU_JIT_TARGET_FROM_CUCONTEXT",
    Self::Target => "CU_JIT_TARGET",
    Self::FallbackStrategy => "CU_JIT_FALLBACK_STRATEGY",
    Self::GenerateDebugInfo => "CU_JIT_GENERATE_DEBUG_INFO",
    Self::LogVerbose => "CU_JIT_LOG_VERBOSE",
    Self::GenerateLineInfo => "CU_JIT_GENERATE_LINE_INFO",
    Self::CacheMode => "CU_JIT_CACHE_MODE",
    Self::NewSm3xOpt => "CU_JIT_NEW_SM3X_OPT",
    Self::FastCompile => "CU_JIT_FAST_COMPILE",
    Self::GlobalSymbolNames => "CU_JIT_GLOBAL_SYMBOL_NAMES",
    Self::GlobalSymbolAddresses => "CU_JIT_GLOBAL_SYMBOL_ADDRESSES",
    Self::GlobalSymbolCount => "CU_JIT_GLOBAL_SYMBOL_COUNT",
    Self::Lto => "CU_JIT_LTO",
    Self::Ftz => "CU_JIT_FTZ",
    Self::PrecDiv => "CU_JIT_PREC_DIV",
    Self::PrecSqrt => "CU_JIT_PREC_SQRT",
    Self::Fma => "CU_JIT_FMA",
    Self::ReferencedKernelNames => "CU_JIT_REFERENCED_KERNEL_NAMES",
    Self::ReferencedKernelCount => "CU_JIT_REFERENCED_KERNEL_COUNT",
    Self::ReferencedVariableNames => "CU_JIT_REFERENCED_VARIABLE_NAMES",
    Self::ReferencedVariableCount => "CU_JIT_REFERENCED_VARIABLE_COUNT",
    Self::OptimizeUnusedDeviceVariables => "CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES",
});

singe_core::native_enum! {
    /// Online compilation targets
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum JitTarget: driver::CUjit_target {
        /// Compute capability 12.1 target with architecture-family features.
        Compute121f = CU_TARGET_COMPUTE_121F,
        /// Compute capability 12.0 target with architecture-family features.
        Compute120f = CU_TARGET_COMPUTE_120F,
        /// Compute capability 10.7 target with architecture-family features.
        Compute107f = CU_TARGET_COMPUTE_107F,
        /// Compute capability 10.3 target with architecture-family features.
        Compute103f = CU_TARGET_COMPUTE_103F,
        /// Compute capability 11.0 target with architecture-family features.
        Compute110f = CU_TARGET_COMPUTE_110F,
        /// Compute capability 10.0 target with architecture-family features.
        Compute100f = CU_TARGET_COMPUTE_100F,
        /// Compute capability 12.1 target with architecture-specific accelerated features.
        Compute121a = CU_TARGET_COMPUTE_121A,
        /// Compute capability 12.0 target with architecture-specific accelerated features.
        Compute120a = CU_TARGET_COMPUTE_120A,
        /// Compute capability 10.7 target with architecture-specific accelerated features.
        Compute107a = CU_TARGET_COMPUTE_107A,
        /// Compute capability 10.3 target with architecture-specific accelerated features.
        Compute103a = CU_TARGET_COMPUTE_103A,
        /// Compute capability 11.0 target with architecture-specific accelerated features.
        Compute110a = CU_TARGET_COMPUTE_110A,
        /// Compute capability 10.0 target with architecture-specific accelerated features.
        Compute100a = CU_TARGET_COMPUTE_100A,
        /// Compute capability 9.0 target with architecture-specific accelerated features.
        Compute90a = CU_TARGET_COMPUTE_90A,
        /// Compute capability 12.1 target without architecture-specific or architecture-family extensions.
        Compute121 = CU_TARGET_COMPUTE_121,
        /// Compute device class 12.0.
        Compute120 = CU_TARGET_COMPUTE_120,
        /// Compute device class 10.7.
        Compute107 = CU_TARGET_COMPUTE_107,
        /// Compute device class 10.3.
        Compute103 = CU_TARGET_COMPUTE_103,
        /// Compute device class 11.0.
        Compute110 = CU_TARGET_COMPUTE_110,
        /// Compute device class 10.0.
        Compute100 = CU_TARGET_COMPUTE_100,
        /// Compute device class 3.0.
        Compute30 = CU_TARGET_COMPUTE_30,
        /// Compute device class 3.2.
        Compute32 = CU_TARGET_COMPUTE_32,
        /// Compute device class 3.5.
        Compute35 = CU_TARGET_COMPUTE_35,
        /// Compute device class 3.7.
        Compute37 = CU_TARGET_COMPUTE_37,
        /// Compute device class 5.0.
        Compute50 = CU_TARGET_COMPUTE_50,
        /// Compute device class 5.2.
        Compute52 = CU_TARGET_COMPUTE_52,
        /// Compute device class 5.3.
        Compute53 = CU_TARGET_COMPUTE_53,
        /// Compute device class 6.0.
        Compute60 = CU_TARGET_COMPUTE_60,
        /// Compute device class 6.1.
        Compute61 = CU_TARGET_COMPUTE_61,
        /// Compute device class 6.2.
        Compute62 = CU_TARGET_COMPUTE_62,
        /// Compute device class 7.0.
        Compute70 = CU_TARGET_COMPUTE_70,
        /// Compute device class 7.2.
        Compute72 = CU_TARGET_COMPUTE_72,
        /// Compute device class 7.5.
        Compute75 = CU_TARGET_COMPUTE_75,
        /// Compute device class 8.0.
        Compute80 = CU_TARGET_COMPUTE_80,
        /// Compute device class 8.6.
        Compute86 = CU_TARGET_COMPUTE_86,
        /// Compute device class 8.7.
        Compute87 = CU_TARGET_COMPUTE_87,
        /// Compute device class 8.9.
        Compute89 = CU_TARGET_COMPUTE_89,
        /// Compute device class 9.0.
        Compute90 = CU_TARGET_COMPUTE_90,
    }
}

impl_enum_display!(JitTarget, {
    Self::Compute121f => "CU_TARGET_COMPUTE_121F",
    Self::Compute120f => "CU_TARGET_COMPUTE_120F",
    Self::Compute107f => "CU_TARGET_COMPUTE_107F",
    Self::Compute103f => "CU_TARGET_COMPUTE_103F",
    Self::Compute110f => "CU_TARGET_COMPUTE_110F",
    Self::Compute100f => "CU_TARGET_COMPUTE_100F",
    Self::Compute121a => "CU_TARGET_COMPUTE_121A",
    Self::Compute120a => "CU_TARGET_COMPUTE_120A",
    Self::Compute107a => "CU_TARGET_COMPUTE_107A",
    Self::Compute103a => "CU_TARGET_COMPUTE_103A",
    Self::Compute110a => "CU_TARGET_COMPUTE_110A",
    Self::Compute100a => "CU_TARGET_COMPUTE_100A",
    Self::Compute90a => "CU_TARGET_COMPUTE_90A",
    Self::Compute121 => "CU_TARGET_COMPUTE_121",
    Self::Compute120 => "CU_TARGET_COMPUTE_120",
    Self::Compute107 => "CU_TARGET_COMPUTE_107",
    Self::Compute103 => "CU_TARGET_COMPUTE_103",
    Self::Compute110 => "CU_TARGET_COMPUTE_110",
    Self::Compute100 => "CU_TARGET_COMPUTE_100",
    Self::Compute30 => "CU_TARGET_COMPUTE_30",
    Self::Compute32 => "CU_TARGET_COMPUTE_32",
    Self::Compute35 => "CU_TARGET_COMPUTE_35",
    Self::Compute37 => "CU_TARGET_COMPUTE_37",
    Self::Compute50 => "CU_TARGET_COMPUTE_50",
    Self::Compute52 => "CU_TARGET_COMPUTE_52",
    Self::Compute53 => "CU_TARGET_COMPUTE_53",
    Self::Compute60 => "CU_TARGET_COMPUTE_60",
    Self::Compute61 => "CU_TARGET_COMPUTE_61",
    Self::Compute62 => "CU_TARGET_COMPUTE_62",
    Self::Compute70 => "CU_TARGET_COMPUTE_70",
    Self::Compute72 => "CU_TARGET_COMPUTE_72",
    Self::Compute75 => "CU_TARGET_COMPUTE_75",
    Self::Compute80 => "CU_TARGET_COMPUTE_80",
    Self::Compute86 => "CU_TARGET_COMPUTE_86",
    Self::Compute87 => "CU_TARGET_COMPUTE_87",
    Self::Compute89 => "CU_TARGET_COMPUTE_89",
    Self::Compute90 => "CU_TARGET_COMPUTE_90",
});

singe_core::native_enum! {
    /// Cubin matching fallback strategies
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum JitFallback: driver::CUjit_fallback {
        /// Prefer to compile ptx if exact binary match not found.
        PreferPtx = CU_PREFER_PTX,
        /// Prefer to fall back to compatible binary code if exact match not found.
        PreferBinary = CU_PREFER_BINARY,
    }
}

impl_enum_display!(JitFallback, {
    Self::PreferPtx => "CU_PREFER_PTX",
    Self::PreferBinary => "CU_PREFER_BINARY",
});

singe_core::native_enum! {
    /// Caching modes for dlcm
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum JitCacheMode: driver::CUjit_cacheMode {
        /// Compile with no -dlcm flag specified.
        OptionNone = CU_JIT_CACHE_OPTION_NONE,
        /// Compile with L1 cache disabled.
        OptionCg = CU_JIT_CACHE_OPTION_CG,
        /// Compile with L1 cache enabled.
        OptionCa = CU_JIT_CACHE_OPTION_CA,
    }
}

impl_enum_display!(JitCacheMode, {
    Self::OptionNone => "CU_JIT_CACHE_OPTION_NONE",
    Self::OptionCg => "CU_JIT_CACHE_OPTION_CG",
    Self::OptionCa => "CU_JIT_CACHE_OPTION_CA",
});

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(feature = "testing", feature = "nvrtc_13_4"))]
    use crate::error::Result;

    #[cfg(all(feature = "testing", feature = "nvrtc_13_4"))]
    #[test]
    fn test_native_ptx_load_with_scalar_options_and_output_buffers() -> Result<()> {
        use crate::{
            device::DeviceAttribute,
            module::LaunchConfig,
            nvrtc::{CompileOptions, OutputKind, Program},
            stream::StreamOptions,
            testing,
        };

        let test_device = testing::bootstrap()?;
        let context = test_device.context();
        let major = context
            .device()
            .attribute(DeviceAttribute::ComputeCapabilityMajor)?;
        let minor = context
            .device()
            .attribute(DeviceAttribute::ComputeCapabilityMinor)?;
        let target = format!("--gpu-architecture=compute_{major}{minor}");
        let program = Program::from_source("extern \"C\" __global__ void noop() {}");
        program.compile(&CompileOptions::new().raw_option(&target))?;

        let mut wall_time = -1.0_f32;
        let mut info_log = [0xa5_u8; 258];
        let mut error_log = [0xa5_u8; 258];
        let options = JitOptions::default()
            .with_max_registers(32)
            .with_threads_per_block(128)
            .with_optimization_level(2)
            .with_log_verbose(true)
            .with_generate_line_info(true)
            .with_wall_time(&mut wall_time)
            .with_info_log(&mut info_log[1..257])
            .with_error_log(&mut error_log[1..257]);
        let module = context.load_nvrtc_module(&program, OutputKind::Ptx, options)?;
        let function = module.function("noop")?;
        let config = LaunchConfig::try_for_1d_grid(128, 128)?;
        let stream = context.create_stream(StreamOptions::default())?;
        // SAFETY: this no-argument kernel accesses no storage and remains loaded through synchronization.
        unsafe { function.launch_async(&config, (), &stream)? };
        stream.synchronize()?;

        assert!(wall_time >= 0.0);
        assert_eq!([info_log[0], info_log[257]], [0xa5; 2]);
        assert_eq!([error_log[0], error_log[257]], [0xa5; 2]);
        Ok(())
    }

    #[test]
    fn test_scalar_options_use_native_values() {
        let mut options = JitOptions::default()
            .with_max_registers(32)
            .with_optimization_level(9)
            .with_target(JitTarget::Compute90)
            .with_fallback_strategy(JitFallback::PreferBinary)
            .with_generate_debug_info(false)
            .with_log_verbose(true)
            .with_generate_line_info(true)
            .with_cache_mode(JitCacheMode::OptionCa);
        let artifact = options.build();
        let value = |option: JitOption| {
            let index = artifact
                .names
                .iter()
                .position(|&name| name == option.into())
                .unwrap();
            artifact.values[index].addr()
        };

        assert_eq!(value(JitOption::MaxRegisters), 32);
        assert_eq!(value(JitOption::OptimizationLevel), 4);
        assert_eq!(value(JitOption::Target), JitTarget::Compute90 as usize);
        assert_eq!(
            value(JitOption::FallbackStrategy),
            JitFallback::PreferBinary as usize
        );
        assert_eq!(value(JitOption::GenerateDebugInfo), 0);
        assert_eq!(value(JitOption::LogVerbose), 1);
        assert_eq!(value(JitOption::GenerateLineInfo), 1);
        assert_eq!(value(JitOption::CacheMode), JitCacheMode::OptionCa as usize);
    }

    #[test]
    fn test_output_buffers_and_word_options_survive_artifact_moves() {
        let mut wall_time = 0.0_f32;
        let mut info = [0_u8; 16];
        let mut errors = [0_u8; 8];
        let info_pointer = info.as_mut_ptr().cast::<()>();
        let error_pointer = errors.as_mut_ptr().cast::<()>();
        let mut options = JitOptions::default()
            .with_threads_per_block(128)
            .with_wall_time(&mut wall_time)
            .with_info_log(&mut info)
            .with_error_log(&mut errors)
            .with_target_from_cuda_context();
        let mut artifact = Box::new(options.build());
        let value = |option: JitOption| {
            let index = artifact
                .names
                .iter()
                .position(|&name| name == option.into())
                .unwrap();
            artifact.values[index]
        };

        assert_eq!(value(JitOption::ThreadsPerBlock).addr(), 128);
        assert!(value(JitOption::WallTime).is_null());
        assert_eq!(value(JitOption::InfoLogBuffer), info_pointer);
        assert_eq!(value(JitOption::ErrorLogBuffer), error_pointer);
        assert_eq!(value(JitOption::InfoLogBufferSizeBytes).addr(), 16);
        assert_eq!(value(JitOption::ErrorLogBufferSizeBytes).addr(), 8);
        assert!(value(JitOption::TargetFromCudaContext).is_null());
        let index = artifact
            .names
            .iter()
            .position(|&name| name == JitOption::WallTime.into())
            .unwrap();
        artifact.values[index] = ptr::without_provenance_mut(2.5_f32.to_bits() as usize);
        drop(artifact);
        assert_eq!(wall_time, 2.5);
    }
}
