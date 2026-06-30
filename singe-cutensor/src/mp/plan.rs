use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ptr,
};

use singe_cuda::memory::DeviceMemory;
use singe_cutensor_sys as sys;

use crate::{
    error::{Error, Result},
    mp::{
        context::{Context, ContextRef, validate_same_context},
        memory::WorkspaceMemory,
        tensor::TensorDescriptor,
        types::{Algorithm, PlanAttribute},
    },
    operation::ComputeDescriptor,
    try_ffi,
    types::{Mode, Operator},
    utility::modes_to_i32_vec,
};

#[derive(Debug)]
pub struct OperationDescriptor<'comm> {
    handle: sys::cutensorMpOperationDescriptor_t,
    context: ContextRef<'comm>,
}

#[derive(Debug)]
pub struct PlanPreference<'comm> {
    handle: sys::cutensorMpPlanPreference_t,
    context: ContextRef<'comm>,
    algorithm: Algorithm,
    device_workspace_limit: u64,
    host_workspace_limit: u64,
}

#[derive(Debug)]
pub struct Plan<'comm> {
    handle: sys::cutensorMpPlan_t,
    context: ContextRef<'comm>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workspace {
    device_size: u64,
    host_size: u64,
}

impl<'comm> OperationDescriptor<'comm> {
    pub fn create_contraction(
        context: &Context<'comm>,
        a: &TensorDescriptor<'comm>,
        modes_a: &[Mode],
        op_a: Operator,
        b: &TensorDescriptor<'comm>,
        modes_b: &[Mode],
        op_b: Operator,
        c: &TensorDescriptor<'comm>,
        modes_c: &[Mode],
        op_c: Operator,
        d: &TensorDescriptor<'comm>,
        modes_d: &[Mode],
        compute: ComputeDescriptor,
    ) -> Result<Self> {
        let context_ref = context.as_context_ref();
        validate_same_context(&context_ref, a.context(), "a")?;
        validate_same_context(&context_ref, b.context(), "b")?;
        validate_same_context(&context_ref, c.context(), "c")?;
        validate_same_context(&context_ref, d.context(), "d")?;
        validate_modes(a, modes_a)?;
        validate_modes(b, modes_b)?;
        validate_modes(c, modes_c)?;
        validate_modes(d, modes_d)?;

        let modes_a = modes_to_i32_vec(modes_a);
        let modes_b = modes_to_i32_vec(modes_b);
        let modes_c = modes_to_i32_vec(modes_c);
        let modes_d = modes_to_i32_vec(modes_d);
        let mut handle = ptr::null_mut();
        context.bind()?;
        unsafe {
            try_ffi!(sys::cutensorMpCreateContraction(
                context.as_raw(),
                &raw mut handle,
                a.as_raw(),
                modes_a.as_ptr(),
                op_a.into(),
                b.as_raw(),
                modes_b.as_ptr(),
                op_b.into(),
                c.as_raw(),
                modes_c.as_ptr(),
                op_c.into(),
                d.as_raw(),
                modes_d.as_ptr(),
                compute.as_raw(),
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

    /// Wraps an existing cuTENSORMp operation descriptor.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMp operation descriptor associated with
    /// `context`. The returned value takes ownership of `handle` and destroys
    /// it on drop.
    pub unsafe fn from_raw(
        handle: sys::cutensorMpOperationDescriptor_t,
        context: &Context<'comm>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
        })
    }

    pub(crate) fn context(&self) -> &ContextRef<'comm> {
        &self.context
    }

    pub const fn as_raw(&self) -> sys::cutensorMpOperationDescriptor_t {
        self.handle
    }

    /// Consumes this descriptor and returns the owned raw cuTENSORMp operation descriptor.
    ///
    /// The caller becomes responsible for destroying the descriptor.
    pub fn into_raw(self) -> sys::cutensorMpOperationDescriptor_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl<'comm> PlanPreference<'comm> {
    pub fn create(
        context: &Context<'comm>,
        algorithm: Algorithm,
        device_workspace_limit: u64,
        host_workspace_limit: u64,
    ) -> Result<Self> {
        let mut handle = ptr::null_mut();
        context.bind()?;
        unsafe {
            try_ffi!(sys::cutensorMpCreatePlanPreference(
                context.as_raw(),
                &raw mut handle,
                algorithm.into(),
                device_workspace_limit,
                host_workspace_limit,
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            algorithm,
            device_workspace_limit,
            host_workspace_limit,
        })
    }

    /// Wraps an existing cuTENSORMp plan preference.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMp plan preference associated with
    /// `context`. Metadata arguments must describe the raw preference. The
    /// returned value takes ownership of `handle` and destroys it on drop.
    pub unsafe fn from_raw(
        handle: sys::cutensorMpPlanPreference_t,
        context: &Context<'comm>,
        algorithm: Algorithm,
        device_workspace_limit: u64,
        host_workspace_limit: u64,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            algorithm,
            device_workspace_limit,
            host_workspace_limit,
        })
    }

    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    pub fn device_workspace_limit(&self) -> u64 {
        self.device_workspace_limit
    }

    pub fn host_workspace_limit(&self) -> u64 {
        self.host_workspace_limit
    }

    pub(crate) fn context(&self) -> &ContextRef<'comm> {
        &self.context
    }

    pub const fn as_raw(&self) -> sys::cutensorMpPlanPreference_t {
        self.handle
    }

    /// Consumes this preference and returns the owned raw cuTENSORMp plan preference.
    ///
    /// The caller becomes responsible for destroying the preference.
    pub fn into_raw(self) -> sys::cutensorMpPlanPreference_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl<'comm> Plan<'comm> {
    pub fn create(
        context: &Context<'comm>,
        desc: &OperationDescriptor<'comm>,
        preference: Option<&PlanPreference<'comm>>,
    ) -> Result<Self> {
        let context_ref = context.as_context_ref();
        validate_same_context(&context_ref, desc.context(), "desc")?;
        if let Some(preference) = preference {
            validate_same_context(&context_ref, preference.context(), "preference")?;
        }

        let mut handle = ptr::null_mut();
        context.bind()?;
        unsafe {
            try_ffi!(sys::cutensorMpCreatePlan(
                context.as_raw(),
                &raw mut handle,
                desc.as_raw(),
                preference.map_or(ptr::null_mut(), |preference| preference.as_raw()),
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

    /// Wraps an existing cuTENSORMp plan.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMp plan associated with `context`. The
    /// returned value takes ownership of `handle` and destroys it on drop.
    pub unsafe fn from_raw(
        handle: sys::cutensorMpPlan_t,
        context: &Context<'comm>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
        })
    }

    pub fn workspace(&self) -> Result<Workspace> {
        Ok(Workspace {
            device_size: self.attribute(PlanAttribute::RequiredWorkspaceDevice)?,
            host_size: self.attribute(PlanAttribute::RequiredWorkspaceHost)?,
        })
    }

    pub fn create_workspace_memory(&self) -> Result<WorkspaceMemory> {
        WorkspaceMemory::create(self.workspace()?)
    }

    pub fn contract<TA, TB, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &DeviceMemory<TA>,
        b: &DeviceMemory<TB>,
        beta: &TScalar,
        c: &DeviceMemory<TC>,
        d: &mut DeviceMemory<TD>,
        workspace: &mut WorkspaceMemory,
    ) -> Result<()> {
        unsafe {
            self.contract_raw(
                ptr::from_ref(alpha).cast(),
                a.as_ptr().cast(),
                b.as_ptr().cast(),
                ptr::from_ref(beta).cast(),
                c.as_ptr().cast(),
                d.as_mut_ptr().cast(),
                workspace.device_mut_ptr(),
                workspace.host_mut_ptr(),
            )
        }
    }

    pub fn contract_in_place<TA, TB, TC, TScalar>(
        &self,
        alpha: &TScalar,
        a: &DeviceMemory<TA>,
        b: &DeviceMemory<TB>,
        beta: &TScalar,
        c_and_d: &mut DeviceMemory<TC>,
        workspace: &mut WorkspaceMemory,
    ) -> Result<()> {
        unsafe {
            self.contract_raw(
                ptr::from_ref(alpha).cast(),
                a.as_ptr().cast(),
                b.as_ptr().cast(),
                ptr::from_ref(beta).cast(),
                c_and_d.as_ptr().cast(),
                c_and_d.as_mut_ptr().cast(),
                workspace.device_mut_ptr(),
                workspace.host_mut_ptr(),
            )
        }
    }

    /// Executes this distributed contraction with raw local tensor pointers.
    ///
    /// # Safety
    ///
    /// Pointers must refer to the local rank's tensor shards and workspaces
    /// matching the descriptors and plan. Scalars must have the type expected by
    /// cuTENSORMp for the data-type combination. All pointers must remain valid
    /// for the duration of the launched operation, and `d`, `device_workspace`,
    /// and `host_workspace` must be writable according to the plan's workspace
    /// requirements.
    pub unsafe fn contract_raw(
        &self,
        alpha: *const (),
        a: *const (),
        b: *const (),
        beta: *const (),
        c: *const (),
        d: *mut (),
        device_workspace: *mut (),
        host_workspace: *mut (),
    ) -> Result<()> {
        self.context.bind()?;
        unsafe {
            try_ffi!(sys::cutensorMpContract(
                self.context.as_raw(),
                self.handle,
                alpha.cast(),
                a.cast(),
                b.cast(),
                beta.cast(),
                c.cast(),
                d.cast(),
                device_workspace.cast(),
                host_workspace.cast(),
            ))?;
        }
        Ok(())
    }

    pub fn as_raw(&self) -> sys::cutensorMpPlan_t {
        self.handle
    }

    /// Consumes this plan and returns the owned raw cuTENSORMp plan.
    ///
    /// The caller becomes responsible for destroying the plan.
    pub fn into_raw(self) -> sys::cutensorMpPlan_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }

    fn attribute<T: Copy>(&self, attr: PlanAttribute) -> Result<T> {
        let mut value = MaybeUninit::<T>::uninit();
        self.context.bind()?;
        unsafe {
            try_ffi!(sys::cutensorMpPlanGetAttribute(
                self.context.as_raw(),
                self.handle,
                attr.into(),
                value.as_mut_ptr().cast(),
                size_of::<T>() as u64,
            ))?;
            Ok(value.assume_init())
        }
    }
}

impl Workspace {
    pub fn device_size(&self) -> u64 {
        self.device_size
    }

    pub fn host_size(&self) -> u64 {
        self.host_size
    }
}

impl Drop for OperationDescriptor<'_> {
    fn drop(&mut self) {
        if let Err(err) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!(
                "failed to bind cutensormp context before destroying operation descriptor: {err}"
            );
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorMpDestroyOperationDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormp operation descriptor: {err}");
            }
        }
    }
}

impl Drop for PlanPreference<'_> {
    fn drop(&mut self) {
        if let Err(err) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cutensormp context before destroying plan preference: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorMpDestroyPlanPreference(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormp plan preference: {err}");
            }
        }
    }
}

impl Drop for Plan<'_> {
    fn drop(&mut self) {
        if let Err(err) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cutensormp context before destroying plan: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorMpDestroyPlan(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormp plan: {err}");
            }
        }
    }
}

fn validate_modes(desc: &TensorDescriptor<'_>, modes: &[Mode]) -> Result<()> {
    if desc.rank() as usize != modes.len() {
        return Err(Error::TensorModeMismatch {
            rank: desc.rank(),
            mode_length: modes.len(),
        });
    }
    Ok(())
}
