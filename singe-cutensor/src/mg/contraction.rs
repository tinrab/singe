use std::{mem::ManuallyDrop, ptr};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_conversion;
use singe_cuda::stream::StreamBinding;
use singe_cutensor_sys as sys;

use crate::{
    error::{Error, Result},
    mg::{
        context::{Context, ContextRef, validate_same_context},
        copy::validate_count,
        memory::{DistributedDeviceMemory, WorkspaceMemory},
        tensor::TensorDescriptor,
        types::Algorithm,
        workspace::Workspace,
    },
    try_ffi,
    types::{ComputeType, Mode, WorkspacePreference},
    utility::modes_to_i32_vec,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ContractionFindAttribute {
    Max = sys::cutensorMgContractionFindAttribute_t::CUTENSORMG_CONTRACTION_FIND_ATTRIBUTE_MAX as _,
}

impl_enum_conversion!(
    sys::cutensorMgContractionFindAttribute_t,
    ContractionFindAttribute
);

#[derive(Debug)]
pub struct ContractionFind {
    handle: sys::cutensorMgContractionFind_t,
    context: ContextRef,
    algorithm: Algorithm,
}

#[derive(Debug)]
pub struct ContractionDescriptor {
    handle: sys::cutensorMgContractionDescriptor_t,
    context: ContextRef,
    a_pointer_count: usize,
    b_pointer_count: usize,
    c_pointer_count: usize,
    d_pointer_count: usize,
}

#[derive(Debug)]
pub struct ContractionPlan {
    handle: sys::cutensorMgContractionPlan_t,
    context: ContextRef,
    a_pointer_count: usize,
    b_pointer_count: usize,
    c_pointer_count: usize,
    d_pointer_count: usize,
    workspace: Workspace,
}

impl ContractionFind {
    pub fn create(context: &Context, algorithm: Algorithm) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cutensorMgCreateContractionFind(
                context.as_raw(),
                &raw mut handle,
                algorithm.into(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            algorithm,
        })
    }

    pub fn create_default(context: &Context) -> Result<Self> {
        Self::create(context, Algorithm::Default)
    }

    /// Wraps an existing cuTENSORMg contraction find handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMg contraction find handle associated
    /// with `context` and `algorithm`. The returned value takes ownership of
    /// `handle` and destroys it on drop.
    pub unsafe fn from_raw(
        handle: sys::cutensorMgContractionFind_t,
        context: &Context,
        algorithm: Algorithm,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            algorithm,
        })
    }

    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    pub fn set_attribute<T>(&mut self, attr: ContractionFindAttribute, value: &T) -> Result<()> {
        unsafe {
            try_ffi!(sys::cutensorMgContractionFindSetAttribute(
                self.context.as_raw(),
                self.handle,
                attr.into(),
                ptr::from_ref(value).cast(),
                size_of::<T>() as i64,
            ))?;
        }
        Ok(())
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

    pub const fn as_raw(&self) -> sys::cutensorMgContractionFind_t {
        self.handle
    }

    /// Consumes this find handle and returns the owned raw cuTENSORMg handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::cutensorMgContractionFind_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl ContractionDescriptor {
    pub fn create(
        context: &Context,
        a: &TensorDescriptor,
        modes_a: &[Mode],
        b: &TensorDescriptor,
        modes_b: &[Mode],
        c: &TensorDescriptor,
        modes_c: &[Mode],
        d: &TensorDescriptor,
        modes_d: &[Mode],
        compute: ComputeType,
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
        unsafe {
            try_ffi!(sys::cutensorMgCreateContractionDescriptor(
                context.as_raw(),
                &raw mut handle,
                a.as_raw(),
                modes_a.as_ptr(),
                b.as_raw(),
                modes_b.as_ptr(),
                c.as_raw(),
                modes_c.as_ptr(),
                d.as_raw(),
                modes_d.as_ptr(),
                compute.into(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            a_pointer_count: a.devices().len(),
            b_pointer_count: b.devices().len(),
            c_pointer_count: c.devices().len(),
            d_pointer_count: d.devices().len(),
        })
    }

    /// Wraps an existing cuTENSORMg contraction descriptor.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMg contraction descriptor associated
    /// with `context`. Pointer counts must match the tensors used to create the
    /// descriptor. The returned value takes ownership of `handle` and destroys
    /// it on drop.
    pub unsafe fn from_raw(
        handle: sys::cutensorMgContractionDescriptor_t,
        context: &Context,
        a_pointer_count: usize,
        b_pointer_count: usize,
        c_pointer_count: usize,
        d_pointer_count: usize,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            a_pointer_count,
            b_pointer_count,
            c_pointer_count,
            d_pointer_count,
        })
    }

    pub fn workspace(
        &self,
        find: &ContractionFind,
        preference: WorkspacePreference,
    ) -> Result<Workspace> {
        validate_same_context(&self.context, find.context(), "find")?;
        let mut device_workspace_size = vec![0; self.context.device_count()];
        let mut host_workspace_size = 0;
        unsafe {
            try_ffi!(sys::cutensorMgContractionGetWorkspace(
                self.context.as_raw(),
                self.handle,
                find.as_raw(),
                preference.into(),
                device_workspace_size.as_mut_ptr(),
                &raw mut host_workspace_size,
            ))?;
        }
        Workspace::from_raw(device_workspace_size, host_workspace_size)
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

    pub const fn as_raw(&self) -> sys::cutensorMgContractionDescriptor_t {
        self.handle
    }

    /// Consumes this descriptor and returns the owned raw cuTENSORMg descriptor.
    ///
    /// The caller becomes responsible for destroying the descriptor.
    pub fn into_raw(self) -> sys::cutensorMgContractionDescriptor_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl ContractionPlan {
    pub fn create(
        context: &Context,
        desc: &ContractionDescriptor,
        find: &ContractionFind,
        workspace: Workspace,
    ) -> Result<Self> {
        validate_same_context(&context.as_context_ref(), desc.context(), "desc")?;
        validate_same_context(desc.context(), find.context(), "find")?;
        workspace.validate_for_context(desc.context())?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cutensorMgCreateContractionPlan(
                context.as_raw(),
                &raw mut handle,
                desc.as_raw(),
                find.as_raw(),
                workspace.device_sizes_ptr(),
                workspace.host_size(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            a_pointer_count: desc.a_pointer_count,
            b_pointer_count: desc.b_pointer_count,
            c_pointer_count: desc.c_pointer_count,
            d_pointer_count: desc.d_pointer_count,
            workspace,
        })
    }

    /// Wraps an existing cuTENSORMg contraction plan.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMg contraction plan associated with
    /// `context` and `workspace`. Pointer counts must match the descriptors
    /// used to create the plan. The returned value takes ownership of `handle`
    /// and destroys it on drop.
    pub unsafe fn from_raw(
        handle: sys::cutensorMgContractionPlan_t,
        context: &Context,
        a_pointer_count: usize,
        b_pointer_count: usize,
        c_pointer_count: usize,
        d_pointer_count: usize,
        workspace: Workspace,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            a_pointer_count,
            b_pointer_count,
            c_pointer_count,
            d_pointer_count,
            workspace,
        })
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    pub fn create_workspace_memory(&self) -> Result<WorkspaceMemory> {
        WorkspaceMemory::create(&self.context, &self.workspace)
    }

    pub fn contract<TA, TB, TC, TD, TScalar>(
        &self,
        alpha: &TScalar,
        a: &DistributedDeviceMemory<TA>,
        b: &DistributedDeviceMemory<TB>,
        beta: &TScalar,
        c: &DistributedDeviceMemory<TC>,
        d: &mut DistributedDeviceMemory<TD>,
        workspace: &mut WorkspaceMemory,
        streams: &[StreamBinding],
    ) -> Result<()> {
        self.validate_execution_slices(
            a.pointer_count(),
            b.pointer_count(),
            c.pointer_count(),
            d.pointer_count(),
            workspace.device_pointer_count(),
            streams.len(),
        )?;

        let a = a.const_ptrs();
        let b = b.const_ptrs();
        let c = c.const_ptrs();
        let mut d = d.mut_ptrs();
        let mut device_workspace = workspace.device_mut_ptrs();

        unsafe {
            self.contract_raw(
                ptr::from_ref(alpha).cast(),
                &a,
                &b,
                ptr::from_ref(beta).cast(),
                &c,
                &mut d,
                &mut device_workspace,
                workspace.host_mut_ptr(),
                streams,
            )
        }
    }

    pub fn contract_in_place<TA, TB, TC, TScalar>(
        &self,
        alpha: &TScalar,
        a: &DistributedDeviceMemory<TA>,
        b: &DistributedDeviceMemory<TB>,
        beta: &TScalar,
        c_and_d: &mut DistributedDeviceMemory<TC>,
        workspace: &mut WorkspaceMemory,
        streams: &[StreamBinding],
    ) -> Result<()> {
        self.validate_execution_slices(
            a.pointer_count(),
            b.pointer_count(),
            c_and_d.pointer_count(),
            c_and_d.pointer_count(),
            workspace.device_pointer_count(),
            streams.len(),
        )?;

        let a = a.const_ptrs();
        let b = b.const_ptrs();
        let c = c_and_d.const_ptrs();
        let mut d = c_and_d.mut_ptrs();
        let mut device_workspace = workspace.device_mut_ptrs();

        unsafe {
            self.contract_raw(
                ptr::from_ref(alpha).cast(),
                &a,
                &b,
                ptr::from_ref(beta).cast(),
                &c,
                &mut d,
                &mut device_workspace,
                workspace.host_mut_ptr(),
                streams,
            )
        }
    }

    pub fn contract_in_place_on_default_streams<TA, TB, TC, TScalar>(
        &self,
        alpha: &TScalar,
        a: &DistributedDeviceMemory<TA>,
        b: &DistributedDeviceMemory<TB>,
        beta: &TScalar,
        c_and_d: &mut DistributedDeviceMemory<TC>,
        workspace: &mut WorkspaceMemory,
    ) -> Result<()> {
        self.validate_execution_slices(
            a.pointer_count(),
            b.pointer_count(),
            c_and_d.pointer_count(),
            c_and_d.pointer_count(),
            self.context.device_count(),
            self.context.device_count(),
        )?;

        let a = a.const_ptrs();
        let b = b.const_ptrs();
        let c = c_and_d.const_ptrs();
        let mut d = c_and_d.mut_ptrs();
        let mut device_workspace = workspace.device_mut_ptrs();

        unsafe {
            self.contract_raw_on_default_streams(
                ptr::from_ref(alpha).cast(),
                &a,
                &b,
                ptr::from_ref(beta).cast(),
                &c,
                &mut d,
                &mut device_workspace,
                workspace.host_mut_ptr(),
            )
        }
    }

    /// Executes this contraction plan with raw distributed tensor pointers.
    ///
    /// # Safety
    ///
    /// Pointer arrays must match the descriptors used for plan creation. Scalars must
    /// have the type expected by the cuTENSORMg data-type combination.
    pub unsafe fn contract_raw(
        &self,
        alpha: *const (),
        a: &[*const ()],
        b: &[*const ()],
        beta: *const (),
        c: &[*const ()],
        d: &mut [*mut ()],
        device_workspace: &mut [*mut ()],
        host_workspace: *mut (),
        streams: &[StreamBinding],
    ) -> Result<()> {
        self.validate_execution_slices(
            a.len(),
            b.len(),
            c.len(),
            d.len(),
            device_workspace.len(),
            streams.len(),
        )?;
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        let mut c = c.to_vec();
        let mut streams: Vec<_> = streams.iter().map(StreamBinding::as_raw).collect();
        unsafe {
            try_ffi!(sys::cutensorMgContraction(
                self.context.as_raw(),
                self.handle,
                alpha as _,
                a.as_mut_ptr() as _,
                b.as_mut_ptr() as _,
                beta as _,
                c.as_mut_ptr() as _,
                d.as_mut_ptr() as _,
                device_workspace.as_mut_ptr() as _,
                host_workspace as _,
                streams.as_mut_ptr(),
            ))?;
        }
        Ok(())
    }

    /// Executes this contraction plan on each device's default stream.
    ///
    /// # Safety
    ///
    /// Pointer arrays must match the descriptors used for plan creation. Scalars must
    /// have the type expected by the cuTENSORMg data-type combination.
    pub unsafe fn contract_raw_on_default_streams(
        &self,
        alpha: *const (),
        a: &[*const ()],
        b: &[*const ()],
        beta: *const (),
        c: &[*const ()],
        d: &mut [*mut ()],
        device_workspace: &mut [*mut ()],
        host_workspace: *mut (),
    ) -> Result<()> {
        self.validate_execution_slices(
            a.len(),
            b.len(),
            c.len(),
            d.len(),
            device_workspace.len(),
            self.context.device_count(),
        )?;
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        let mut c = c.to_vec();
        let mut streams = vec![ptr::null_mut(); self.context.device_count()];
        unsafe {
            try_ffi!(sys::cutensorMgContraction(
                self.context.as_raw(),
                self.handle,
                alpha as _,
                a.as_mut_ptr() as _,
                b.as_mut_ptr() as _,
                beta as _,
                c.as_mut_ptr() as _,
                d.as_mut_ptr() as _,
                device_workspace.as_mut_ptr() as _,
                host_workspace as _,
                streams.as_mut_ptr(),
            ))?;
        }
        Ok(())
    }

    fn validate_execution_slices(
        &self,
        a_count: usize,
        b_count: usize,
        c_count: usize,
        d_count: usize,
        device_workspace_count: usize,
        stream_count: usize,
    ) -> Result<()> {
        validate_count("a", self.a_pointer_count, a_count)?;
        validate_count("b", self.b_pointer_count, b_count)?;
        validate_count("c", self.c_pointer_count, c_count)?;
        validate_count("d", self.d_pointer_count, d_count)?;
        validate_count(
            "device_workspace",
            self.context.device_count(),
            device_workspace_count,
        )?;
        validate_count("streams", self.context.device_count(), stream_count)
    }

    pub const fn as_raw(&self) -> sys::cutensorMgContractionPlan_t {
        self.handle
    }

    /// Consumes this plan and returns the owned raw cuTENSORMg contraction plan.
    ///
    /// The caller becomes responsible for destroying the plan.
    pub fn into_raw(self) -> sys::cutensorMgContractionPlan_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl Drop for ContractionFind {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorMgDestroyContractionFind(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormg contraction find: {err}");
            }
        }
    }
}

impl Drop for ContractionDescriptor {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorMgDestroyContractionDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormg contraction descriptor: {err}");
            }
        }
    }
}

impl Drop for ContractionPlan {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorMgDestroyContractionPlan(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormg contraction plan: {err}");
            }
        }
    }
}

fn validate_modes(desc: &TensorDescriptor, modes: &[Mode]) -> Result<()> {
    if desc.rank() as usize != modes.len() {
        return Err(Error::TensorModeMismatch {
            rank: desc.rank(),
            mode_length: modes.len(),
        });
    }
    Ok(())
}
