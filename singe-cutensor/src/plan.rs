use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ptr,
};

use singe_core::report_cleanup_failure;

use crate::{
    context::{Context, ContextRef, validate_same_context},
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

// SAFETY: the preference exclusively owns its native handle, which has no thread affinity, and its retained library context can be released from any thread.
unsafe impl Send for PlanPreference {}

/// Transferred native ownership with its original metadata and retained library context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`PlanPreference::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawPlanPreference {
    /// Transferred native plan resource; destroy it before releasing its retained context.
    pub handle: sys::cutensorPlanPreference_t,
    /// Library context the preference was created in.
    pub context: ContextRef,
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
    /// Returns an error if the context is not initialized, cuTENSOR rejects the requested algorithm or JIT mode, or cuTENSOR does not return a valid handle.
    pub fn create(context: &Context, algorithm: Algorithm, jit_mode: JitMode) -> Result<Self> {
        context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `context` keeps the library handle live, the algorithm and JIT mode are valid enum values, and `handle` is a local output.
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

    fn set_attribute<T>(&mut self, attr: PlanPreferenceAttribute, value: &T) -> Result<()> {
        validate_plan_preference_attribute_size(attr, size_of::<T>())?;

        self.context.bind()?;
        // SAFETY: `self.handle` and `self.context` are live, and `value` points to `size_of::<T>()` readable bytes, the size checked above for `attr`.
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
        // SAFETY: `self.handle` is live, `value` has room for `size_of::<T>()` bytes, the size checked above for `attr`, and a successful call fills it.
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
        self.set_attribute(
            PlanPreferenceAttribute::AutotuneMode,
            &sys::cutensorAutotuneMode_t::from(mode),
        )
    }

    /// Sets the plan-cache lookup and insertion policy.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or cuTENSOR rejects the setting.
    pub fn set_cache_mode(&mut self, mode: CacheMode) -> Result<()> {
        self.set_attribute(
            PlanPreferenceAttribute::CacheMode,
            &sys::cutensorCacheMode_t::from(mode),
        )
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
        self.set_attribute(
            PlanPreferenceAttribute::Algorithm,
            &sys::cutensorAlgo_t::from(algorithm),
        )
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
        self.set_attribute(
            PlanPreferenceAttribute::JitMode,
            &sys::cutensorJitMode_t::from(mode),
        )
    }

    /// Returns the autotuning mode used across recurrent plan creation and execution.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn autotune_mode(&self) -> Result<AutotuneMode> {
        let raw: sys::cutensorAutotuneMode_t =
            self.attribute(PlanPreferenceAttribute::AutotuneMode)?;
        Ok(AutotuneMode::try_from(raw)?)
    }

    /// Returns the plan-cache lookup and insertion policy.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute has an unexpected native byte size, the CUDA context cannot be bound, or the native attribute query fails.
    pub fn cache_mode(&self) -> Result<CacheMode> {
        let raw: sys::cutensorCacheMode_t = self.attribute(PlanPreferenceAttribute::CacheMode)?;
        Ok(CacheMode::try_from(raw)?)
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
        let raw: sys::cutensorAlgo_t = self.attribute(PlanPreferenceAttribute::Algorithm)?;
        Ok(Algorithm::try_from(raw)?)
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
        let raw: sys::cutensorJitMode_t = self.attribute(PlanPreferenceAttribute::JitMode)?;
        Ok(JitMode::try_from(raw)?)
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
    /// Returns an error if `gpu_arch` cannot be represented for cuTENSOR or if cuTENSOR rejects the architecture value.
    pub fn set_gpu_arch(&mut self, gpu_arch: u32) -> Result<()> {
        let gpu_arch = to_i32(gpu_arch, "gpu_arch")?;
        self.set_attribute(PlanPreferenceAttribute::GpuArch, &gpu_arch)
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

    /// Returns the borrowed native preference handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorPlanPreference_t {
        self.handle
    }

    /// Consumes the preference and returns the raw cuTENSOR handle without destroying it.
    ///
    /// The returned record retains the library context and plan metadata.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawPlanPreference {
        let owner = ManuallyDrop::new(self);
        // SAFETY: `owner` is never dropped, so each field is read out exactly once.
        unsafe {
            RawPlanPreference {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
            }
        }
    }

    /// Takes ownership of a native plan-preference handle and its library context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid plan preference created in `raw.context` and uniquely owned.
    /// The handle must not be destroyed elsewhere afterward.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawPlanPreference) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: raw.handle,
            context: raw.context,
        })
    }
}

impl Drop for PlanPreference {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cutensor plan preference",
                &error,
            );
            return;
        }

        // SAFETY: the preference exclusively owns its non-null handle, and `self.context` keeps the library handle it was created in live.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorDestroyPlanPreference(self.handle)) {
                report_cleanup_failure("destroy", "cutensor plan preference", &error);
            }
        }
    }
}

/// Owns a selected native execution plan, its operation signature, and context.
#[derive(Debug)]
pub struct Plan {
    handle: sys::cutensorPlan_t,
    context: ContextRef,
    workspace_size: usize,
    signature: OperationSignature,
    tensor_spans: Vec<u64>,
    block_spans: Vec<u64>,
}

/// Transferred native ownership with its original metadata and retained library context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`Plan::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawPlan {
    /// Transferred native plan resource; destroy it before releasing its retained context.
    pub handle: sys::cutensorPlan_t,
    /// Library context the plan was created in.
    pub context: ContextRef,
    /// Required workspace capacity in bytes.
    pub workspace_size: usize,
    /// Operation signature the plan executes.
    pub signature: OperationSignature,
    // The operand and block spans are wrapper state, so only `into_raw` can produce this record.
    tensor_spans: Vec<u64>,
    block_spans: Vec<u64>,
}

// SAFETY: a plan exclusively owns its native handle, and its retained library context
// can be released from any thread.
unsafe impl Send for Plan {}

impl Plan {
    /// Determines the required workspace size for the given operation encoded by `operation`.
    ///
    /// # Errors
    ///
    /// Returns an error if `operation` or `preference` belongs to another context, the context cannot be bound, cuTENSOR rejects the workspace preference, or cuTENSOR cannot estimate the workspace size.
    pub fn estimate_workspace_size(
        context: &Context,
        operation: &OperationDescriptor,
        preference: &PlanPreference,
        workspace_preference: WorkspacePreference,
    ) -> Result<usize> {
        validate_plan_inputs(context, operation, preference)?;
        context.bind()?;

        let mut workspace_size = 0;
        // SAFETY: `context`, `operation`, and `preference` each keep their native handles live, and `workspace_size` is a local output.
        unsafe {
            try_ffi!(sys::cutensorEstimateWorkspaceSize(
                context.as_raw(),
                operation.as_raw(),
                preference.as_raw(),
                workspace_preference.into(),
                &raw mut workspace_size,
            ))?;
        }
        to_usize(workspace_size, "workspace size")
    }

    /// Selects a kernel and prepares an execution plan under a workspace limit.
    ///
    /// `workspace_size_limit` is measured in bytes.
    /// The actual requirement is stored with the plan and can be queried with [`Self::workspace_size`].
    /// The plan targets the active CUDA device; creation must not be captured in a CUDA graph when JIT compilation is enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if `operation` or `preference` belongs to another context, the CUDA context cannot be bound, cuTENSOR rejects the operation, preferences, or workspace limit, native plan creation or workspace querying fails, or the returned handle is null.
    pub fn create(
        context: &Context,
        operation: &OperationDescriptor,
        preference: &PlanPreference,
        workspace_size_limit: usize,
    ) -> Result<Self> {
        let workspace_size_limit = u64::try_from(workspace_size_limit)
            .map_err(|_| Error::out_of_range("workspace_size_limit"))?;
        validate_plan_inputs(context, operation, preference)?;
        context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `context`, `operation`, and `preference` each keep their native handles live, and `handle` is a local output.
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

        // The plan owns the handle from here, so an error below destroys it.
        let mut plan = Self {
            handle,
            context: context.as_context_ref(),
            workspace_size: 0,
            signature: operation.signature(),
            tensor_spans: operation.tensor_spans(),
            block_spans: operation.block_spans(),
        };
        let mut required_workspace_size = 0_u64;
        // SAFETY: `handle` is the plan just created in `context`, and `required_workspace_size` is a local `u64`, the size this attribute writes.
        unsafe {
            try_ffi!(sys::cutensorPlanGetAttribute(
                context.as_raw(),
                handle,
                PlanAttribute::RequiredWorkspace.into(),
                (&raw mut required_workspace_size).cast(),
                size_of::<u64>() as _,
            ))?;
        }

        plan.workspace_size = to_usize(required_workspace_size, "workspace size")?;
        Ok(plan)
    }

    /// Returns the plan's required workspace capacity in bytes.
    #[must_use]
    pub fn workspace_size(&self) -> usize {
        self.workspace_size
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

    pub(crate) fn signature(&self) -> OperationSignature {
        self.signature
    }

    pub(crate) fn tensor_spans(&self) -> &[u64] {
        &self.tensor_spans
    }

    pub(crate) fn block_spans(&self) -> &[u64] {
        &self.block_spans
    }

    /// Returns the borrowed native execution plan handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorPlan_t {
        self.handle
    }

    /// Consumes the plan and returns the raw cuTENSOR plan handle without destroying it.
    ///
    /// The returned record retains the library context and plan metadata.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawPlan {
        let owner = ManuallyDrop::new(self);
        // SAFETY: `owner` is never dropped, so each field is read out exactly once.
        unsafe {
            RawPlan {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
                workspace_size: owner.workspace_size,
                signature: owner.signature,
                tensor_spans: ptr::read(&raw const owner.tensor_spans),
                block_spans: ptr::read(&raw const owner.block_spans),
            }
        }
    }

    /// Takes ownership of a native plan with its workspace requirement, operation signature, and library context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid plan created in `raw.context`, uniquely owned, compatible with `raw.signature`, and requiring `raw.workspace_size` bytes of workspace.
    /// The handle must not be destroyed elsewhere afterward.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawPlan) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: raw.handle,
            context: raw.context,
            workspace_size: raw.workspace_size,
            signature: raw.signature,
            tensor_spans: raw.tensor_spans,
            block_spans: raw.block_spans,
        })
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            report_cleanup_failure("bind the cuda context to destroy", "cutensor plan", &error);
            return;
        }

        // SAFETY: the plan exclusively owns its non-null handle, and `self.context` keeps the library handle it was created in live.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorDestroyPlan(self.handle)) {
                report_cleanup_failure("destroy", "cutensor plan", &error);
            }
        }
    }
}

fn validate_plan_inputs(
    context: &Context,
    operation: &OperationDescriptor,
    preference: &PlanPreference,
) -> Result<()> {
    let context = context.as_context_ref();
    validate_same_context(&context, operation.context(), "operation")?;
    validate_same_context(&context, preference.context(), "preference")
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
        return Err(Error::SizeMismatch {
            name: attr.to_string().into(),
            expected,
            actual,
        });
    }

    Ok(())
}
