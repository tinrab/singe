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

#[derive(Debug)]
pub struct PlanPreference {
    handle: sys::cutensorPlanPreference_t,
    context: ContextRef,
}

impl PlanPreference {
    pub fn create_default(ctx: &Context) -> Result<Self> {
        Self::create(ctx, Algorithm::Default, JitMode::Default)
    }

    /// Creates a plan preference that limits the applicable kernels for a plan or operation.
    ///
    /// The preference is tied to `ctx` and frees its cuTENSOR handle when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the context is not initialized, cuTENSOR rejects the
    /// requested algorithm or JIT mode, or cuTENSOR does not return a valid handle.
    pub fn create(ctx: &Context, algorithm: Algorithm, jit_mode: JitMode) -> Result<Self> {
        ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cutensorCreatePlanPreference(
                ctx.as_raw(),
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
            context: ctx.as_context_ref(),
        })
    }

    /// Wraps an existing cuTENSOR plan-preference handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSOR plan-preference handle associated with
    /// `ctx`. Ownership of `handle` is transferred to the returned preference,
    /// and the handle must not be destroyed elsewhere after calling this
    /// function.
    pub unsafe fn from_raw(handle: sys::cutensorPlanPreference_t, ctx: &Context) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: ctx.as_context_ref(),
        })
    }

    fn set_attribute<T>(&mut self, attr: PlanPreferenceAttribute, value: &T) -> Result<()> {
        validate_plan_preference_attribute_size(attr, size_of::<T>())?;

        self.context.bind()?;
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

    pub fn set_autotune_mode(&mut self, mode: AutotuneMode) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::AutotuneMode, &mode)
    }

    pub fn set_cache_mode(&mut self, mode: CacheMode) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::CacheMode, &mode)
    }

    pub fn set_incremental_count(&mut self, count: i32) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::IncrementalCount, &count)
    }

    pub fn set_algorithm(&mut self, algorithm: Algorithm) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::Algorithm, &algorithm)
    }

    pub fn set_kernel_rank(&mut self, rank: i32) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::KernelRank, &rank)
    }

    pub fn set_jit_mode(&mut self, mode: JitMode) -> Result<()> {
        self.set_attribute(PlanPreferenceAttribute::JitMode, &mode)
    }

    pub fn autotune_mode(&self) -> Result<AutotuneMode> {
        self.attribute(PlanPreferenceAttribute::AutotuneMode)
    }

    pub fn cache_mode(&self) -> Result<CacheMode> {
        self.attribute(PlanPreferenceAttribute::CacheMode)
    }

    pub fn incremental_count(&self) -> Result<i32> {
        let value = self.attribute(PlanPreferenceAttribute::IncrementalCount)?;
        Ok(value)
    }

    pub fn algorithm(&self) -> Result<Algorithm> {
        self.attribute(PlanPreferenceAttribute::Algorithm)
    }

    pub fn kernel_rank(&self) -> Result<i32> {
        let value = self.attribute(PlanPreferenceAttribute::KernelRank)?;
        Ok(value)
    }

    pub fn jit_mode(&self) -> Result<JitMode> {
        self.attribute(PlanPreferenceAttribute::JitMode)
    }

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

    pub const fn as_raw(&self) -> sys::cutensorPlanPreference_t {
        self.handle
    }

    /// Consumes the preference and returns the raw cuTENSOR handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with cuTENSOR.
    pub fn into_raw(self) -> sys::cutensorPlanPreference_t {
        let preference = ManuallyDrop::new(self);
        preference.handle
    }
}

impl Drop for PlanPreference {
    fn drop(&mut self) {
        if let Err(err) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cutensor context before destroying plan preference: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorDestroyPlanPreference(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensor plan preference: {err}");
            }
        }
    }
}

#[derive(Debug)]
pub struct Plan {
    handle: sys::cutensorPlan_t,
    context: ContextRef,
    required_workspace_size: u64,
    signature: OperationSignature,
}

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
        ctx: &Context,
        operation: &OperationDescriptor,
        preference: &PlanPreference,
        workspace_preference: WorkspacePreference,
    ) -> Result<u64> {
        ctx.bind()?;

        let mut workspace_size = 0;
        unsafe {
            try_ffi!(sys::cutensorEstimateWorkspaceSize(
                ctx.as_raw(),
                operation.as_raw(),
                preference.as_raw(),
                workspace_preference.into(),
                &raw mut workspace_size,
            ))?;
        }
        Ok(workspace_size)
    }

    pub fn create(
        ctx: &Context,
        operation: &OperationDescriptor,
        preference: &PlanPreference,
        workspace_size_limit: u64,
    ) -> Result<Self> {
        ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cutensorCreatePlan(
                ctx.as_raw(),
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
        unsafe {
            try_ffi!(sys::cutensorPlanGetAttribute(
                ctx.as_raw(),
                handle,
                PlanAttribute::RequiredWorkspace.into(),
                (&raw mut required_workspace_size).cast(),
                size_of::<u64>() as _,
            ))?;
        }

        Ok(Self {
            handle,
            context: ctx.as_context_ref(),
            required_workspace_size,
            signature: operation.signature(),
        })
    }

    /// Wraps an existing cuTENSOR plan handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSOR plan handle associated with `ctx` and
    /// compatible with `signature`. `required_workspace_size` must match the
    /// plan's required workspace size. Ownership of `handle` is transferred to
    /// the returned plan, and the handle must not be destroyed elsewhere after
    /// calling this function.
    pub unsafe fn from_raw(
        handle: sys::cutensorPlan_t,
        ctx: &Context,
        required_workspace_size: u64,
        signature: OperationSignature,
    ) -> Self {
        Self {
            handle,
            context: ctx.as_context_ref(),
            required_workspace_size,
            signature,
        }
    }

    pub fn required_workspace_size(&self) -> u64 {
        self.required_workspace_size
    }

    pub fn required_workspace_size_bytes(&self) -> Result<usize> {
        to_usize(self.required_workspace_size, "required workspace size")
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

    pub(crate) fn signature(&self) -> OperationSignature {
        self.signature
    }

    pub const fn as_raw(&self) -> sys::cutensorPlan_t {
        self.handle
    }

    /// Consumes the plan and returns the raw cuTENSOR plan handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with cuTENSOR.
    pub fn into_raw(self) -> sys::cutensorPlan_t {
        let plan = ManuallyDrop::new(self);
        plan.handle
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        if let Err(err) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cutensor context before destroying plan: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorDestroyPlan(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensor plan: {err}");
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
