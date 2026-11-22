use std::{
    mem::{self, ManuallyDrop, size_of, size_of_val},
    ptr,
};

use singe_core::report_cleanup_failure;
use singe_cuda::device::Device;
use singe_cudss_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{
        ConfigParameter, DowncastFactorsMode, FactorizationAlgorithm, MatchingAlgorithm,
        MemoryOptimization, PivotEpsilonAlgorithm, PivotType, ReorderingAlgorithm, SolveAlgorithm,
        SolveMode,
    },
    utility::{ensure_exact_size, to_i32, to_size_t, to_usize},
};

/// Solver configuration settings for a cuDSS solve.
///
/// A configuration object stores algorithm choices and feature toggles read by [`Context::execute_async`](crate::context::Context::execute_async).
/// cuDSS allocates only lightweight host resources for this object.
#[derive(Debug)]
pub struct Config {
    raw: sys::cudssConfig_t,
}

// SAFETY: a cuDSS config is a host-side object that cuDSS does not tie to the creating thread, so it may be used from any thread one call at a time.
// It is not `Sync`, because cuDSS does not document concurrent calls on one config.
unsafe impl Send for Config {}

impl Config {
    /// Creates a solver configuration object.
    ///
    /// The configuration stores algorithm choices and feature toggles for a solve.
    /// cuDSS allocates only lightweight host resources for this object.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot create the configuration object or returns a null handle.
    pub fn create() -> Result<Self> {
        // SAFETY: `raw` is a live local that cuDSS writes the new handle into, and it is checked for null before the wrapper owns it.
        unsafe {
            let mut raw = ptr::null_mut();
            try_ffi!(sys::cudssConfigCreate(&raw mut raw))?;
            if raw.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self { raw })
        }
    }

    /// Takes ownership of an existing cuDSS config handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid cuDSS config handle created by cuDSS.
    /// Ownership is transferred to the returned wrapper, which destroys the handle on drop.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(raw: sys::cudssConfig_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { raw })
    }

    /// Sets a scalar configuration parameter.
    ///
    /// `T` must match the value type documented for `param`.
    ///
    /// # Safety
    ///
    /// T must match this parameter's native ABI, with every source byte initialized.
    /// Pointer-valued settings must retain valid pointees for every native use.
    ///
    /// # Errors
    ///
    /// Returns an error if the value size cannot be represented for cuDSS or if cuDSS rejects the parameter, value type, or value.
    pub unsafe fn set_raw<T: Copy>(&mut self, param: ConfigParameter, value: &T) -> Result<()> {
        // SAFETY: `self.raw` is the live config this wrapper owns, `value` is a borrowed `T` read for `size_of::<T>()` bytes, and the caller guarantees `T` matches the parameter's native ABI.
        unsafe {
            try_ffi!(sys::cudssConfigSet(
                self.raw,
                param.into(),
                ptr::from_ref(value).cast(),
                to_size_t(size_of::<T>(), "config parameter size")?,
            ))
        }
    }

    /// Sets an array-valued configuration parameter.
    ///
    /// `T` and the slice length must match the value type documented for `param`.
    ///
    /// # Safety
    ///
    /// T, slice length, byte initialization, and indirection must match this parameter's native array ABI.
    /// Retain referenced pointees for every native use.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice byte size cannot be represented for cuDSS or if cuDSS rejects the parameter, value type, or values.
    pub unsafe fn set_raw_slice<T: Copy>(
        &mut self,
        param: ConfigParameter,
        values: &[T],
    ) -> Result<()> {
        // SAFETY: `self.raw` is the live config this wrapper owns, `values` is read for its `size_of_val` bytes, and the caller guarantees the element type and length match the parameter's native array ABI.
        unsafe {
            try_ffi!(sys::cudssConfigSet(
                self.raw,
                param.into(),
                values.as_ptr().cast(),
                to_size_t(size_of_val(values), "config parameter size")?,
            ))
        }
    }

    /// Retrieves a scalar configuration parameter.
    ///
    /// `T` must match the value type documented for `param`.
    ///
    /// # Safety
    ///
    /// The parameter must be a synchronous host output that initializes all bytes as a valid T. Device outputs and parameters requiring an existing destination value cannot be read through this method.
    ///
    /// # Errors
    ///
    /// Returns an error if the value size cannot be represented for cuDSS, if cuDSS rejects the parameter or value type, or if cuDSS reports an internal byte-count mismatch.
    pub unsafe fn get_raw<T: Copy>(&self, param: ConfigParameter) -> Result<T> {
        // SAFETY: `self.raw` is the live config this wrapper owns and `value` has room for `size_of::<T>()` bytes.
        // `assume_init` runs only after cuDSS reported writing exactly that many bytes, and the caller guarantees they form a valid `T`.
        unsafe {
            let mut value = mem::MaybeUninit::<T>::uninit();
            let mut size_written = 0;
            try_ffi!(sys::cudssConfigGet(
                self.raw,
                param.into(),
                value.as_mut_ptr().cast(),
                to_size_t(size_of::<T>(), "config parameter size")?,
                &raw mut size_written,
            ))?;
            ensure_exact_size(
                to_usize(size_written, "config parameter size_written")?,
                size_of::<T>(),
                "config parameter",
            )?;
            Ok(value.assume_init())
        }
    }

    /// Sets the reordering algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_reordering_algorithm(&mut self, algorithm: ReorderingAlgorithm) -> Result<()> {
        // SAFETY: the value's enum is `#[repr(u32)]`, matching the parameter's native enum type.
        unsafe { self.set_raw(ConfigParameter::ReorderingAlgorithm, &algorithm) }
    }

    /// Returns the configured reordering algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the setting.
    pub fn reordering_algorithm(&self) -> Result<ReorderingAlgorithm> {
        // SAFETY: cuDSS writes the parameter as its native enum type, a `u32`, and every `u32` is a valid value.
        let raw = unsafe { self.get_raw::<u32>(ConfigParameter::ReorderingAlgorithm) }?;
        ReorderingAlgorithm::try_from(raw).map_err(|_| Error::UnknownEnumValue {
            name: "reordering_algorithm".into(),
            value: raw.into(),
        })
    }

    /// Sets the factorization algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_factorization_algorithm(&mut self, algorithm: FactorizationAlgorithm) -> Result<()> {
        // SAFETY: the value's enum is `#[repr(u32)]`, matching the parameter's native enum type.
        unsafe { self.set_raw(ConfigParameter::FactorizationAlgorithm, &algorithm) }
    }

    /// Sets the solve algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_solve_algorithm(&mut self, algorithm: SolveAlgorithm) -> Result<()> {
        // SAFETY: the value's enum is `#[repr(u32)]`, matching the parameter's native enum type.
        unsafe { self.set_raw(ConfigParameter::SolveAlgorithm, &algorithm) }
    }

    /// Sets the matching algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_matching_algorithm(&mut self, algorithm: MatchingAlgorithm) -> Result<()> {
        // SAFETY: the value's enum is `#[repr(u32)]`, matching the parameter's native enum type.
        unsafe { self.set_raw(ConfigParameter::MatchingAlgorithm, &algorithm) }
    }

    /// Returns the configured matching algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the setting.
    pub fn matching_algorithm(&self) -> Result<MatchingAlgorithm> {
        // SAFETY: cuDSS writes the parameter as its native enum type, a `u32`, and every `u32` is a valid value.
        let raw = unsafe { self.get_raw::<u32>(ConfigParameter::MatchingAlgorithm) }?;
        MatchingAlgorithm::try_from(raw).map_err(|_| Error::UnknownEnumValue {
            name: "matching_algorithm".into(),
            value: raw.into(),
        })
    }

    /// Forces nonsymmetric matching for matrix types other than [`MatrixType::General`](crate::types::MatrixType::General).
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_force_nonsymmetric_matching(&mut self, enabled: bool) -> Result<()> {
        self.set_bool(ConfigParameter::ForceNonsymmetricMatching, enabled)
    }

    /// Sets the matrix modifier used during the solve phase.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_solve_mode(&mut self, mode: SolveMode) -> Result<()> {
        // SAFETY: the value's enum is `#[repr(u32)]`, matching the parameter's native enum type.
        unsafe { self.set_raw(ConfigParameter::SolveMode, &mode) }
    }

    /// Returns the configured solve mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the setting.
    pub fn solve_mode(&self) -> Result<SolveMode> {
        // SAFETY: cuDSS writes the parameter as its native enum type, a `u32`, and every `u32` is a valid value.
        let raw = unsafe { self.get_raw::<u32>(ConfigParameter::SolveMode) }?;
        SolveMode::try_from(raw).map_err(|_| Error::UnknownEnumValue {
            name: "solve_mode".into(),
            value: raw.into(),
        })
    }

    /// Sets the mantissa bits floating-point emulation preserves during factorization, or `0` to disable emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_emulation_mantissa_bit_count(&mut self, bits: i32) -> Result<()> {
        // SAFETY: the parameter's native type is `int`, which `i32` matches.
        unsafe { self.set_raw(ConfigParameter::EmulationMantissaBitCount, &bits) }
    }

    /// Sets the precision cuDSS keeps the factors in after factorization.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_downcast_factors(&mut self, mode: DowncastFactorsMode) -> Result<()> {
        // SAFETY: the value's enum is `#[repr(u32)]`, matching the parameter's native enum type.
        unsafe { self.set_raw(ConfigParameter::DowncastFactors, &mode) }
    }

    /// Sets the memory optimizations cuDSS may apply.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_memory_optimization(&mut self, optimization: MemoryOptimization) -> Result<()> {
        // SAFETY: the flags are a `u32`, matching the parameter's native enum type.
        unsafe {
            self.set_raw(
                ConfigParameter::MemoryOptimizationSetting,
                &optimization.bits(),
            )
        }
    }

    /// Sets the pivot type.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_pivot_type(&mut self, pivot_type: PivotType) -> Result<()> {
        // SAFETY: the value's enum is `#[repr(u32)]`, matching the parameter's native enum type.
        unsafe { self.set_raw(ConfigParameter::PivotType, &pivot_type) }
    }

    /// Sets the pivot epsilon algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_pivot_epsilon_algorithm(&mut self, algorithm: PivotEpsilonAlgorithm) -> Result<()> {
        // SAFETY: the value's enum is `#[repr(u32)]`, matching the parameter's native enum type.
        unsafe { self.set_raw(ConfigParameter::PivotEpsilonAlgorithm, &algorithm) }
    }

    /// Sets the number of iterative refinement steps.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_iterative_refinement_steps(&mut self, steps: i32) -> Result<()> {
        // SAFETY: the parameter's native type is `int`, which `i32` matches.
        unsafe { self.set_raw(ConfigParameter::IterativeRefinementSteps, &steps) }
    }

    /// Sets the iterative refinement tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_iterative_refinement_tolerance(&mut self, tolerance: f64) -> Result<()> {
        // SAFETY: the parameter's native type is `double`, which `f64` matches.
        unsafe { self.set_raw(ConfigParameter::IterativeRefinementTolerance, &tolerance) }
    }

    /// Enables or disables hybrid memory mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_hybrid_memory_mode(&mut self, enabled: bool) -> Result<()> {
        self.set_bool(ConfigParameter::HybridMemoryMode, enabled)
    }

    /// Enables or disables hybrid execute mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_hybrid_execute_mode(&mut self, enabled: bool) -> Result<()> {
        self.set_bool(ConfigParameter::HybridExecuteMode, enabled)
    }

    /// Enables or disables CUDA host-memory registration in hybrid memory mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_use_cuda_register_memory(&mut self, enabled: bool) -> Result<()> {
        self.set_bool(ConfigParameter::UseCudaRegisterMemory, enabled)
    }

    /// Enables or disables the superpanel optimization.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_use_superpanels(&mut self, enabled: bool) -> Result<()> {
        self.set_bool(ConfigParameter::UseSuperpanels, enabled)
    }

    /// Sets multi-GPU device indices and matching device count.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects either setting.
    pub fn set_device_indices(&mut self, devices: &[Device]) -> Result<()> {
        let count = to_i32(devices.len(), "device count")?;
        let device_ids: Vec<i32> = devices.iter().map(|device| device.id()).collect();
        // SAFETY: the device count is an `int` and the device indices are an `int` array of that length.
        unsafe {
            self.set_raw(ConfigParameter::DeviceCount, &count)?;
            self.set_raw_slice(ConfigParameter::DeviceIndices, &device_ids)
        }
    }

    /// Sets uniform batch size.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_uniform_batch_size(&mut self, batch_size: i32) -> Result<()> {
        // SAFETY: the parameter's native type is `int`, which `i32` matches.
        unsafe { self.set_raw(ConfigParameter::UniformBatchSize, &batch_size) }
    }

    /// Sets the selected uniform batch index.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the setting.
    pub fn set_uniform_batch_index(&mut self, batch_index: i32) -> Result<()> {
        // SAFETY: the parameter's native type is `int`, which `i32` matches.
        unsafe { self.set_raw(ConfigParameter::UniformBatchIndex, &batch_index) }
    }

    fn set_bool(&mut self, param: ConfigParameter, enabled: bool) -> Result<()> {
        // SAFETY: every boolean configuration parameter is an `int` flag, which `i32` matches.
        unsafe { self.set_raw(param, &i32::from(enabled)) }
    }

    /// Returns the raw cuDSS config handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cudssConfig_t {
        self.raw
    }

    /// Consumes this wrapper and returns the owned raw cuDSS config handle.
    ///
    /// The caller becomes responsible for destroying the returned handle.
    #[must_use]
    pub fn into_raw(self) -> sys::cudssConfig_t {
        let config = ManuallyDrop::new(self);
        config.raw
    }
}

impl Drop for Config {
    fn drop(&mut self) {
        // SAFETY: `self.raw` is the live config this wrapper owns, and `into_raw` skips this drop when ownership leaves the wrapper.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudssConfigDestroy(self.raw)) {
                report_cleanup_failure("destroy", "cudss config", &error);
            }
        }
    }
}
