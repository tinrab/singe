use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ptr,
};

use crate::{
    context::{Context, ContextRef},
    error::{Error, Result},
    operation::{OperationDescriptor, OperationSignature},
    sys, try_ffi,
    types::{
        Algorithm, AutotuneMode, CacheMode, JitMode, PlanAttribute, PlanPreferenceAttribute,
        WorkspacePreference,
    },
    utility::{to_i32, to_usize},
};

/// Owns kernel-selection and tuning preferences for plans in one cuTENSOR context.
#[derive(Debug)]
pub struct PlanPreference {
    handle: sys::cutensorPlanPreference_t,
    context: ContextRef,
}

/// Transferred native ownership with its original metadata and retained library context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`PlanPreference::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawPlanPreference {
    /// Transferred native plan resource; destroy it before releasing its retained context.
    pub handle: sys::cutensorPlanPreference_t,
    context: ContextRef,
}

impl PlanPreference {
    /// Creates preferences using the default algorithm and JIT mode.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding or native preference creation fails, the defaults are unsupported, or the native handle is null.
    pub fn create_default(context: &Context) -> Result<Self> {
        Self::create(context, Algorithm::Default, JitMode::Default)
    }

    /// Creates a plan preference that limits the applicable kernels for a plan or operation.
    ///
    /// The preference is tied to `context` and frees its cuTENSOR handle when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the context is not initialized, cuTENSOR rejects the
    /// requested algorithm or JIT mode, or cuTENSOR does not return a valid handle.
    pub fn create(context: &Context, algorithm: Algorithm, jit_mode: JitMode) -> Result<Self> {
        context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cutensorCreatePlanPreference` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cutensorCreatePlanPreference(
                context.as_raw(),
                &raw mut handle,
                algorithm.into(),
                jit_mode.into(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
        })
    }

    /// Wraps an existing cuTENSOR plan-preference handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSOR plan-preference handle associated with
    /// `context`. Ownership of `handle` is transferred to the returned preference,
    /// and the handle must not be destroyed elsewhere after calling this
    /// function.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
    pub unsafe fn from_raw_parts(
        handle: sys::cutensorPlanPreference_t,
        context: &Context,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
        })
    }

    fn set_attribute<T>(&mut self, attr: PlanPreferenceAttribute, value: &T) -> Result<()> {
        validate_plan_preference_attribute_size(attr, size_of::<T>())?;

        self.context.bind()?;
        // SAFETY: `cutensorPlanPreferenceSetAttribute` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cutensorPlanPreferenceSetAttribute(
                self.context.as_raw(),
                self.handle,
                attr.into(),
                ptr::from_ref(value).cast(),
                size_of::<T>() as _,
            ))?;
        }
        Ok(())
    }

    fn attribute<T>(&self, attr: PlanPreferenceAttribute) -> Result<T> {
        validate_plan_preference_attribute_size(attr, size_of::<T>())?;

        self.context.bind()?;
        let mut value = MaybeUninit::<T>::uninit();
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        unsafe {
            try_ffi!(sys::cutensorPlanPreferenceGetAttribute(
                self.context.as_raw(),
                self.handle,
                attr.into(),
                value.as_mut_ptr().cast(),
                size_of::<T>() as _,
            ))?;
            Ok(value.assume_init())
        }
    }

    /// Sets the autotuning mode used across recurrent plan creation and execution.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or cuTENSOR rejects the setting.
    pub fn set_autotune_mode(&mut self, mode: AutotuneMode) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::AutotuneMode, &mode)
    }

    /// Sets the plan-cache lookup and insertion policy.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or cuTENSOR rejects the setting.
    pub fn set_cache_mode(&mut self, mode: CacheMode) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::CacheMode, &mode)
    }

    /// Sets the maximum number of kernel candidates tested by incremental autotuning.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or cuTENSOR rejects the setting.
    pub fn set_incremental_count(&mut self, count: i32) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::IncrementalCount, &count)
    }

    /// Sets the algorithm family selected by the native performance model or an explicit choice.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or cuTENSOR rejects the setting.
    pub fn set_algorithm(&mut self, algorithm: Algorithm) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::Algorithm, &algorithm)
    }

    /// Sets the kernel variant rank in the algorithm's native performance ordering.
    ///
    /// Rank zero selects the best-ranked variant; rank one selects the second-ranked variant, and so on.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or cuTENSOR rejects the setting.
    pub fn set_kernel_rank(&mut self, rank: i32) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::KernelRank, &rank)
    }

    /// Sets the just-in-time compilation mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or cuTENSOR rejects the setting.
    pub fn set_jit_mode(&mut self, mode: JitMode) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::JitMode, &mode)
    }

    /// Returns the autotuning mode used across recurrent plan creation and execution.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn autotune_mode(&self) -> Result<AutotuneMode> {
        self.attribute(PlanPreferenceAttribute::AutotuneMode)
    }

    /// Returns the plan-cache lookup and insertion policy.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn cache_mode(&self) -> Result<CacheMode> {
        self.attribute(PlanPreferenceAttribute::CacheMode)
    }

    /// Returns the maximum number of kernel candidates tested by incremental autotuning.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn incremental_count(&self) -> Result<i32> {
        let value = self.attribute(PlanPreferenceAttribute::IncrementalCount)?;
        Ok(value)
    }

    /// Returns the algorithm family selected by the native performance model or an explicit choice.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn algorithm(&self) -> Result<Algorithm> {
        self.attribute(PlanPreferenceAttribute::Algorithm)
    }

    /// Returns the kernel variant rank in the algorithm's native performance ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn kernel_rank(&self) -> Result<i32> {
        let value = self.attribute(PlanPreferenceAttribute::KernelRank)?;
        Ok(value)
    }

    /// Returns the just-in-time compilation mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn jit_mode(&self) -> Result<JitMode> {
        self.attribute(PlanPreferenceAttribute::JitMode)
    }

    /// Returns the target GPU architecture encoded as `10 * SM.major + SM.minor`.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn gpu_arch(&self) -> Result<u32> {
        let value: u32 = self.attribute(PlanPreferenceAttribute::GpuArch)?;
        Ok(value)
    }

    /// Plans for a specific GPU architecture instead of the architecture associated with the context.
    /// The value encodes the SM version as `10 * SM.major + SM.minor`.
    /// Currently only SM versions 80, 90 and 100 are supported.
    ///
    /// # Errors
    ///
    /// Returns an error if `gpu_arch` cannot be represented for cuTENSOR or if
    /// cuTENSOR rejects the architecture value.
    pub fn set_gpu_arch(&mut self, gpu_arch: u32) -> Result<()> {
        let gpu_arch = to_i32(gpu_arch, "gpu_arch")?;
        self.set_attribute(PlanPreferenceAttribute::GpuArch, &gpu_arch)
    }

    /// Returns the borrowed native preference handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorPlanPreference_t {
        self.handle
    }

    /// Consumes the preference and returns the raw cuTENSOR handle without
    /// destroying it.
    ///
    /// The returned record retains the native library owner and descriptor metadata.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawPlanPreference {
        let owner = ManuallyDrop::new(self);
        // SAFETY: move each field exactly once while suppressing native destruction.
        unsafe {
            RawPlanPreference {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
            }
        }
    }

    /// Reclaims a transferred resource with its original metadata and library owner.
    ///
    /// # Safety
    ///
    /// `handle` must remain valid and uniquely owned with its original descriptor metadata unchanged.
    /// Complete all borrowed queued accesses before replacing or releasing their resources.
    #[must_use]
    pub unsafe fn from_raw(raw: RawPlanPreference) -> Self {
        Self {
            handle: raw.handle,
            context: raw.context,
        }
    }
}

impl Drop for PlanPreference {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cutensor context before destroying plan preference: {error}");
        }

        // SAFETY: `cutensorDestroyPlanPreference` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorDestroyPlanPreference(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensor plan preference: {error}");
            }
        }
    }
}

/// Owns a selected native execution plan, its operation signature, and context.
#[derive(Debug)]
pub struct Plan {
    handle: sys::cutensorPlan_t,
    context: ContextRef,
    required_workspace_size: u64,
    signature: OperationSignature,
}

/// Transferred native ownership with its original metadata and retained library context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`Plan::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawPlan {
    /// Transferred native plan resource; destroy it before releasing its retained context.
    pub handle: sys::cutensorPlan_t,
    context: ContextRef,
    required_workspace_size: u64,
    signature: OperationSignature,
}

// SAFETY: a plan exclusively owns its native handle, and its retained library context
// can be released from any thread.
unsafe impl Send for Plan {}

impl Plan {
    /// Determines the required workspace size for the given operation encoded by `operation`.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, `operation` and
    /// `preference` do not belong to a compatible context, cuTENSOR rejects the
    /// workspace preference, or cuTENSOR cannot estimate the workspace size.
    pub fn estimate_workspace_size(
        context: &Context,
        operation: &OperationDescriptor,
        preference: &PlanPreference,
        workspace_preference: WorkspacePreference,
    ) -> Result<u64> {
        context.bind()?;

        let mut workspace_size = 0;
        // SAFETY: `cutensorEstimateWorkspaceSize` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cutensorEstimateWorkspaceSize(
                context.as_raw(),
                operation.as_raw(),
                preference.as_raw(),
                workspace_preference.into(),
                &raw mut workspace_size,
            ))?;
        }
        Ok(workspace_size)
    }

    /// Selects a kernel and prepares an execution plan under a workspace limit.
    ///
    /// `workspace_size_limit` is measured in bytes. The actual requirement is stored with the plan and can be queried with [`Self::required_workspace_size`]. The plan targets the active CUDA device; creation must not be captured in a CUDA graph when JIT compilation is enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, cuTENSOR rejects the operation, preferences, or workspace limit, native plan creation or workspace querying fails, or the returned handle is null.
    pub fn create(
        context: &Context,
        operation: &OperationDescriptor,
        preference: &PlanPreference,
        workspace_size_limit: u64,
    ) -> Result<Self> {
        context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cutensorCreatePlan` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cutensorCreatePlan(
                context.as_raw(),
                &raw mut handle,
                operation.as_raw(),
                preference.as_raw(),
                workspace_size_limit,
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let mut required_workspace_size = 0_u64;
        // SAFETY: `cutensorPlanGetAttribute` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cutensorPlanGetAttribute(
                context.as_raw(),
                handle,
                PlanAttribute::RequiredWorkspace.into(),
                (&raw mut required_workspace_size).cast(),
                size_of::<u64>() as _,
            ))?;
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            required_workspace_size,
            signature: operation.signature(),
        })
    }

    /// Wraps an existing cuTENSOR plan handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSOR plan handle associated with `context` and
    /// compatible with `signature`. `required_workspace_size` must match the
    /// plan's required workspace size. Ownership of `handle` is transferred to
    /// the returned plan, and the handle must not be destroyed elsewhere after
    /// calling this function.
    pub unsafe fn from_raw_parts(
        handle: sys::cutensorPlan_t,
        context: &Context,
        required_workspace_size: u64,
        signature: OperationSignature,
    ) -> Self {
        Self {
            handle,
            context: context.as_context_ref(),
            required_workspace_size,
            signature,
        }
    }

    /// Returns the plan's required workspace capacity in bytes.
    #[must_use]
    pub fn required_workspace_size(&self) -> u64 {
        self.required_workspace_size
    }

    /// Returns the required workspace byte count as `usize`.
    ///
    /// # Errors
    ///
    /// Returns an error if the native `u64` byte count does not fit `usize`.
    pub fn required_workspace_size_bytes(&self) -> Result<usize> {
        to_usize(self.required_workspace_size, "required workspace size")
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

    pub(crate) fn signature(&self) -> OperationSignature {
        self.signature
    }

    /// Returns the borrowed native execution plan handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorPlan_t {
        self.handle
    }

    /// Consumes the plan and returns the raw cuTENSOR plan handle without
    /// destroying it.
    ///
    /// The returned record retains the native library owner and descriptor metadata.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawPlan {
        let owner = ManuallyDrop::new(self);
        // SAFETY: move each field exactly once while suppressing native destruction.
        unsafe {
            RawPlan {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
                required_workspace_size: owner.required_workspace_size,
                signature: owner.signature,
            }
        }
    }

    /// Reclaims a transferred resource with its original metadata and library owner.
    ///
    /// # Safety
    ///
    /// `handle` must remain valid and uniquely owned with its original descriptor metadata unchanged.
    /// Complete all borrowed queued accesses before replacing or releasing their resources.
    #[must_use]
    pub unsafe fn from_raw(raw: RawPlan) -> Self {
        Self {
            handle: raw.handle,
            context: raw.context,
            required_workspace_size: raw.required_workspace_size,
            signature: raw.signature,
        }
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cutensor context before destroying plan: {error}");
        }

        // SAFETY: `cutensorDestroyPlan` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorDestroyPlan(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensor plan: {error}");
            }
        }
    }
}

fn validate_plan_preference_attribute_size(
    attr: PlanPreferenceAttribute,
    actual: usize,
) -> Result<()> {
    let expected = match attr {
        PlanPreferenceAttribute::AutotuneMode => size_of::<sys::cutensorAutotuneMode_t>(),
        PlanPreferenceAttribute::CacheMode => size_of::<sys::cutensorCacheMode_t>(),
        PlanPreferenceAttribute::IncrementalCount
        | PlanPreferenceAttribute::KernelRank
        | PlanPreferenceAttribute::GpuArch => size_of::<i32>(),
        PlanPreferenceAttribute::Algorithm => size_of::<sys::cutensorAlgo_t>(),
        PlanPreferenceAttribute::JitMode => size_of::<sys::cutensorJitMode_t>(),
    };

    if actual != expected {
        return Err(Error::PlanPreferenceInvalidAttributeSize {
            attr,
            expected,
            actual,
        });
    }

    Ok(())
}
