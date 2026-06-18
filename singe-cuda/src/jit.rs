#![allow(deprecated)]

use std::ptr;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_cuda_sys::{driver, runtime};

use singe_core::{impl_enum_conversion, impl_enum_display};

#[derive(Debug, Default)]
pub struct JitOptions<'a> {
    pub max_registers: Option<u32>,
    pub threads_per_block: Option<u32>,
    pub wall_time: Option<&'a mut f32>,
    pub info_log_buffer: Option<&'a mut [u8]>,
    pub error_log_buffer: Option<&'a mut [u8]>,
    pub optimization_level: Option<u32>,
    pub target_from_cuda_context: Option<()>,
    pub target: Option<JitTarget>,
    pub fallback_strategy: Option<JitFallback>,
    pub generate_debug_info: Option<bool>,
    pub log_verbose: Option<bool>,
    pub generate_line_info: Option<bool>,
    pub cache_mode: Option<JitCacheMode>,
}

pub struct JitOptionsArtifact {
    pub names: Vec<driver::CUjit_option>,
    pub values: Vec<*mut ()>,

    // Storage for values needing stable pointers for the FFI call.
    storage_target: Option<u32>,
    storage_fallback: Option<u32>,
    storage_debug_info: Option<i32>,
    storage_log_verbose: Option<i32>,
    storage_line_info: Option<i32>,
    storage_cache_mode: Option<u32>,

    // For buffers, the driver expects the buffer pointer directly plus mutable size storage.
    storage_info_log_ptr: Option<*mut u8>,
    storage_info_log_size: Option<u32>,
    storage_error_log_ptr: Option<*mut u8>,
    storage_error_log_size: Option<u32>,
    storage_max_registers: Option<u32>,
    storage_threads_per_block: Option<u32>,
    storage_optimization_level: Option<u32>,
}

impl<'a> JitOptions<'a> {
    pub fn with_max_registers(mut self, value: u32) -> Self {
        self.max_registers = Some(value);
        self
    }

    pub fn with_threads_per_block(mut self, value: u32) -> Self {
        self.threads_per_block = Some(value);
        self
    }

    pub fn with_wall_time(mut self, value: &'a mut f32) -> Self {
        self.wall_time = Some(value);
        self
    }

    pub fn with_info_log(mut self, buffer: &'a mut [u8]) -> Self {
        self.info_log_buffer = Some(buffer);
        self
    }

    pub fn with_error_log(mut self, buffer: &'a mut [u8]) -> Self {
        self.error_log_buffer = Some(buffer);
        self
    }

    pub fn with_optimization_level(mut self, level: u32) -> Self {
        // Clamp to 0-4 as per docs.
        self.optimization_level = Some(level.min(4));
        self
    }

    pub const fn with_target_from_cuda_context(mut self) -> Self {
        self.target_from_cuda_context = Some(());
        self
    }

    pub fn with_target(mut self, target: JitTarget) -> Self {
        self.target = Some(target);
        self
    }

    pub fn with_fallback_strategy(mut self, strategy: JitFallback) -> Self {
        self.fallback_strategy = Some(strategy);
        self
    }

    pub fn with_generate_debug_info(mut self, enable: bool) -> Self {
        self.generate_debug_info = Some(enable);
        self
    }

    pub fn with_log_verbose(mut self, enable: bool) -> Self {
        self.log_verbose = Some(enable);
        self
    }

    pub fn with_generate_line_info(mut self, enable: bool) -> Self {
        self.generate_line_info = Some(enable);
        self
    }

    pub fn with_cache_mode(mut self, mode: JitCacheMode) -> Self {
        self.cache_mode = Some(mode);
        self
    }

    /// Prepares the option and value arrays for the FFI call.
    /// Populates the internal storage fields to keep pointers stable.
    ///
    /// # Safety
    ///
    /// The caller must ensure that this [`JitOptions`] instance outlives the FFI call.
    pub fn build(&mut self) -> JitOptionsArtifact {
        let mut artifact = JitOptionsArtifact {
            names: Vec::new(),
            values: Vec::new(),
            storage_target: None,
            storage_fallback: None,
            storage_debug_info: None,
            storage_log_verbose: None,
            storage_line_info: None,
            storage_cache_mode: None,
            storage_info_log_ptr: None,
            storage_info_log_size: None,
            storage_error_log_ptr: None,
            storage_error_log_size: None,
            storage_max_registers: self.max_registers,
            storage_threads_per_block: self.threads_per_block,
            storage_optimization_level: self.optimization_level.map(|value| value.clamp(0, 4)),
        };

        // Populate internal storage.
        artifact.storage_target = self.target.map(Into::into);
        artifact.storage_fallback = self.fallback_strategy.map(Into::into);
        artifact.storage_cache_mode = self.cache_mode.map(Into::into);
        artifact.storage_debug_info = self.generate_debug_info.map(i32::from);
        artifact.storage_log_verbose = self.log_verbose.map(i32::from);
        artifact.storage_line_info = self.generate_line_info.map(i32::from);
        artifact.storage_info_log_ptr = self
            .info_log_buffer
            .as_mut()
            .map(|slice| slice.as_mut_ptr().cast::<u8>());
        artifact.storage_info_log_size = self
            .info_log_buffer
            .as_ref()
            .map(|buffer| buffer.len().min(u32::MAX as usize) as u32);
        artifact.storage_error_log_ptr = self
            .error_log_buffer
            .as_mut()
            .map(|slice| slice.as_mut_ptr().cast::<u8>());
        artifact.storage_error_log_size = self
            .error_log_buffer
            .as_ref()
            .map(|buffer| buffer.len().min(u32::MAX as usize) as u32);

        // Build the FFI arrays, pointing into `self`.
        if let Some(ref mut val) = artifact.storage_max_registers {
            artifact.names.push(JitOption::MaxRegisters.into());
            artifact.values.push(ptr::from_mut::<u32>(val).cast());
        }
        if let Some(ref mut val) = artifact.storage_threads_per_block {
            artifact.names.push(JitOption::ThreadsPerBlock.into());
            artifact.values.push(ptr::from_mut::<u32>(val).cast());
        }
        // OUT parameters provided by user need direct pointer.
        if let Some(ref mut val_ref) = self.wall_time {
            artifact.names.push(JitOption::WallTime.into());
            artifact.values.push(ptr::from_mut::<f32>(*val_ref).cast());
        }

        // Buffer handling: Add both pointer and size options if buffer is set.
        if let Some(info_log_ptr) = artifact.storage_info_log_ptr {
            artifact.names.push(JitOption::InfoLogBuffer.into());
            artifact.values.push(info_log_ptr.cast());

            if let Some(ref mut size_val) = artifact.storage_info_log_size {
                artifact
                    .names
                    .push(JitOption::InfoLogBufferSizeBytes.into());
                artifact.values.push(ptr::from_mut::<u32>(size_val).cast());
            }
        }
        if let Some(error_log_ptr) = artifact.storage_error_log_ptr {
            artifact.names.push(JitOption::ErrorLogBuffer.into());
            artifact.values.push(error_log_ptr.cast());

            if let Some(ref mut size_val) = artifact.storage_error_log_size {
                artifact
                    .names
                    .push(JitOption::ErrorLogBufferSizeBytes.into());
                artifact.values.push(ptr::from_mut::<u32>(size_val).cast());
            }
        }

        if let Some(ref mut value) = artifact.storage_optimization_level {
            artifact.names.push(JitOption::OptimizationLevel.into());
            artifact.values.push(ptr::from_mut::<u32>(value).cast());
        }
        if self.target_from_cuda_context.is_some() {
            artifact.names.push(JitOption::TargetFromCudaContext.into());
            artifact.values.push(ptr::null_mut()); // No value needed.
        }
        // Point to internal storage for enums/bools converted to primitive types.
        if let Some(ref mut val) = artifact.storage_target {
            artifact.names.push(JitOption::Target.into());
            artifact.values.push(ptr::from_mut::<u32>(val).cast());
        }
        if let Some(ref mut val) = artifact.storage_fallback {
            artifact.names.push(JitOption::FallbackStrategy.into());
            artifact.values.push(ptr::from_mut::<u32>(val).cast());
        }
        if let Some(ref mut val) = artifact.storage_debug_info {
            artifact.names.push(JitOption::GenerateDebugInfo.into());
            artifact.values.push(ptr::from_mut::<i32>(val).cast());
        }
        if let Some(ref mut val) = artifact.storage_log_verbose {
            artifact.names.push(JitOption::LogVerbose.into());
            artifact.values.push(ptr::from_mut::<i32>(val).cast());
        }
        if let Some(ref mut val) = artifact.storage_line_info {
            artifact.names.push(JitOption::GenerateLineInfo.into());
            artifact.values.push(ptr::from_mut::<i32>(val).cast());
        }
        if let Some(ref mut val) = artifact.storage_cache_mode {
            artifact.names.push(JitOption::CacheMode.into());
            artifact.values.push(ptr::from_mut::<u32>(val).cast());
        }

        artifact
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum JitOption {
    MaxRegisters = runtime::cudaJitOption::CU_JIT_MAX_REGISTERS as _,
    ThreadsPerBlock = runtime::cudaJitOption::CU_JIT_THREADS_PER_BLOCK as _,
    WallTime = runtime::cudaJitOption::CU_JIT_WALL_TIME as _,
    InfoLogBuffer = runtime::cudaJitOption::CU_JIT_INFO_LOG_BUFFER as _,
    InfoLogBufferSizeBytes = runtime::cudaJitOption::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES as _,
    ErrorLogBuffer = runtime::cudaJitOption::CU_JIT_ERROR_LOG_BUFFER as _,
    ErrorLogBufferSizeBytes = runtime::cudaJitOption::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES as _,
    OptimizationLevel = runtime::cudaJitOption::CU_JIT_OPTIMIZATION_LEVEL as _,
    TargetFromCudaContext = runtime::cudaJitOption::CU_JIT_TARGET_FROM_CUCONTEXT as _,
    Target = runtime::cudaJitOption::CU_JIT_TARGET as _,
    FallbackStrategy = runtime::cudaJitOption::CU_JIT_FALLBACK_STRATEGY as _,
    GenerateDebugInfo = runtime::cudaJitOption::CU_JIT_GENERATE_DEBUG_INFO as _,
    LogVerbose = runtime::cudaJitOption::CU_JIT_LOG_VERBOSE as _,
    GenerateLineInfo = runtime::cudaJitOption::CU_JIT_GENERATE_LINE_INFO as _,
    CacheMode = runtime::cudaJitOption::CU_JIT_CACHE_MODE as _,
    #[deprecated]
    NewSm3xOpt = runtime::cudaJitOption::CU_JIT_NEW_SM3X_OPT as _,
    FastCompile = runtime::cudaJitOption::CU_JIT_FAST_COMPILE as _,
    GlobalSymbolNames = runtime::cudaJitOption::CU_JIT_GLOBAL_SYMBOL_NAMES as _,
    GlobalSymbolAddresses = runtime::cudaJitOption::CU_JIT_GLOBAL_SYMBOL_ADDRESSES as _,
    GlobalSymbolCount = runtime::cudaJitOption::CU_JIT_GLOBAL_SYMBOL_COUNT as _,
    #[deprecated]
    Lto = runtime::cudaJitOption::CU_JIT_LTO as _,
    #[deprecated]
    Ftz = runtime::cudaJitOption::CU_JIT_FTZ as _,
    #[deprecated]
    PrecDiv = runtime::cudaJitOption::CU_JIT_PREC_DIV as _,
    #[deprecated]
    PrecSqrt = runtime::cudaJitOption::CU_JIT_PREC_SQRT as _,
    #[deprecated]
    Fma = runtime::cudaJitOption::CU_JIT_FMA as _,
    #[deprecated]
    ReferencedKernelNames = runtime::cudaJitOption::CU_JIT_REFERENCED_KERNEL_NAMES as _,
    #[deprecated]
    ReferencedKernelCount = runtime::cudaJitOption::CU_JIT_REFERENCED_KERNEL_COUNT as _,
    #[deprecated]
    ReferencedVariableNames = runtime::cudaJitOption::CU_JIT_REFERENCED_VARIABLE_NAMES as _,
    #[deprecated]
    ReferencedVariableCount = runtime::cudaJitOption::CU_JIT_REFERENCED_VARIABLE_COUNT as _,
    #[deprecated]
    OptimizeUnusedDeviceVariables =
        runtime::cudaJitOption::CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES as _,
    NumOptions = runtime::cudaJitOption::CU_JIT_NUM_OPTIONS as _,
}

impl_enum_conversion!(u32, runtime::cudaJitOption, JitOption);

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
    Self::NumOptions => "CU_JIT_NUM_OPTIONS",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum JitTarget {
    Compute30 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_30 as _,
    Compute32 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_32 as _,
    Compute35 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_35 as _,
    Compute37 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_37 as _,
    Compute50 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_50 as _,
    Compute52 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_52 as _,
    Compute53 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_53 as _,
    Compute60 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_60 as _,
    Compute61 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_61 as _,
    Compute62 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_62 as _,
    Compute70 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_70 as _,
    Compute72 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_72 as _,
    Compute75 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_75 as _,
    Compute80 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_80 as _,
    Compute86 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_86 as _,
    Compute87 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_87 as _,
    Compute89 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_89 as _,
    Compute90 = driver::CUjit_target_enum::CU_TARGET_COMPUTE_90 as _,
}

impl_enum_conversion!(u32, driver::CUjit_target, JitTarget);

impl_enum_display!(JitTarget, {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum JitFallback {
    PreferPtx = driver::CUjit_fallback_enum::CU_PREFER_PTX as _,
    PreferBinary = driver::CUjit_fallback_enum::CU_PREFER_BINARY as _,
}

impl_enum_conversion!(u32, driver::CUjit_fallback, JitFallback);

impl_enum_display!(JitFallback, {
    Self::PreferPtx => "CU_PREFER_PTX",
    Self::PreferBinary => "CU_PREFER_BINARY",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum JitCacheMode {
    OptionNone = driver::CUjit_cacheMode_enum::CU_JIT_CACHE_OPTION_NONE as _,
    OptionCg = driver::CUjit_cacheMode_enum::CU_JIT_CACHE_OPTION_CG as _,
    OptionCa = driver::CUjit_cacheMode_enum::CU_JIT_CACHE_OPTION_CA as _,
}

impl_enum_conversion!(u32, driver::CUjit_cacheMode, JitCacheMode);

impl_enum_display!(JitCacheMode, {
    Self::OptionNone => "CU_JIT_CACHE_OPTION_NONE",
    Self::OptionCg => "CU_JIT_CACHE_OPTION_CG",
    Self::OptionCa => "CU_JIT_CACHE_OPTION_CA",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum JitInputType {
    Cubin = driver::CUjitInputType_enum::CU_JIT_INPUT_CUBIN as _,
    Ptx = driver::CUjitInputType_enum::CU_JIT_INPUT_PTX as _,
    Fatbinary = driver::CUjitInputType_enum::CU_JIT_INPUT_FATBINARY as _,
    Object = driver::CUjitInputType_enum::CU_JIT_INPUT_OBJECT as _,
    Library = driver::CUjitInputType_enum::CU_JIT_INPUT_LIBRARY as _,
    #[deprecated]
    Nvvm = driver::CUjitInputType_enum::CU_JIT_INPUT_NVVM as _,
    NumInputTypes = driver::CUjitInputType_enum::CU_JIT_NUM_INPUT_TYPES as _,
}

impl_enum_conversion!(u32, driver::CUjitInputType, JitInputType);

impl_enum_display!(JitInputType, {
    Self::Cubin => "CU_JIT_INPUT_CUBIN",
    Self::Ptx => "CU_JIT_INPUT_PTX",
    Self::Fatbinary => "CU_JIT_INPUT_FATBINARY",
    Self::Object => "CU_JIT_INPUT_OBJECT",
    Self::Library => "CU_JIT_INPUT_LIBRARY",
    Self::Nvvm => "CU_JIT_INPUT_NVVM",
    Self::NumInputTypes => "CU_JIT_NUM_INPUT_TYPES",
});
