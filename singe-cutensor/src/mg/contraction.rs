use std::{mem::ManuallyDrop, ptr};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
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

/// Represents various attributes for the contraction find
///
///
///
/// While empty for now, future attributes will allow for a more fine-grained tuning of the distributed tensor contraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ContractionFindAttribute {
    ///< Not an actual attribute, but the maximum value that the enum takes on
    Max = sys::cutensorMgContractionFindAttribute_t::CUTENSORMG_CONTRACTION_FIND_ATTRIBUTE_MAX as _,
}

impl_enum_conversion!(
    sys::cutensorMgContractionFindAttribute_t,
    ContractionFindAttribute
);

/// Owns the algorithm selection used to plan a multi-GPU contraction.
#[derive(Debug)]
pub struct ContractionFind {
    handle: sys::cutensorMgContractionFind_t,
    context: ContextRef,
    algorithm: Algorithm,
}

/// Owns a distributed contraction description and operand pointer counts.
#[derive(Debug)]
pub struct ContractionDescriptor {
    handle: sys::cutensorMgContractionDescriptor_t,
    context: ContextRef,
    a_pointer_count: usize,
    b_pointer_count: usize,
    c_pointer_count: usize,
    d_pointer_count: usize,
}

/// Owns a distributed contraction plan, workspace requirements, and its library context.
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
    /// Creates contraction kernel-search settings for an algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if native search-setting creation fails or returns a null handle.
    pub fn create(context: &Context, algorithm: Algorithm) -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `cutensorMgCreateContractionFind` receives live handles owned by the wrappers and valid output locations.
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

    /// Creates contraction search settings using the native default algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if native search-setting creation fails or returns a null handle.
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
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
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

    /// Returns the algorithm recorded when this search handle was created.
    #[must_use]
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Passes an attribute value and its Rust byte size to native search-setting configuration.
    ///
    /// The current attribute enum exposes only a sentinel, rather than a configurable native attribute.
    ///
    /// # Errors
    ///
    /// Returns an error if cuTENSOR rejects the attribute or supplied value representation.
    ///
    /// # Panics
    ///
    /// Panics if `size_of::<T>()` does not fit the native signed 64-bit byte count.
    pub fn set_attribute<T>(&mut self, attr: ContractionFindAttribute, value: &T) -> Result<()> {
        // SAFETY: `cutensorMgContractionFindSetAttribute` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cutensorMgContractionFindSetAttribute(
                self.context.as_raw(),
                self.handle,
                attr.into(),
                ptr::from_ref(value).cast(),
                i64::try_from(size_of::<T>()).expect("tensor element size fits native i64"),
            ))?;
        }
        Ok(())
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

    /// Returns the borrowed native Contraction find handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorMgContractionFind_t {
        self.handle
    }

    /// Consumes this find handle and returns the owned raw cuTENSORMg handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> sys::cutensorMgContractionFind_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl ContractionDescriptor {
    /// Describes `D = alpha * A * B + beta * C` using symbolic tensor modes.
    ///
    /// # Errors
    ///
    /// Returns an error if an operand belongs to another multi-GPU context, a mode count differs from its tensor rank, native descriptor creation fails, or the returned handle is null.
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
        // SAFETY: `cutensorMgCreateContractionDescriptor` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
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

    /// Estimates per-context-device and host workspace bytes for the selected algorithm and workspace preference.
    ///
    /// # Errors
    ///
    /// Returns an error if `find` belongs to another multi-GPU context, the native query fails, or a returned byte count is negative.
    pub fn workspace(
        &self,
        find: &ContractionFind,
        preference: WorkspacePreference,
    ) -> Result<Workspace> {
        validate_same_context(&self.context, find.context(), "find")?;
        let mut device_workspace_size = vec![0; self.context.device_count()];
        let mut host_workspace_size = 0;
        // SAFETY: `cutensorMgContractionGetWorkspace` receives live handles owned by the wrappers and valid output locations.
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

    /// Returns the borrowed native Contraction descriptor handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorMgContractionDescriptor_t {
        self.handle
    }

    /// Consumes this descriptor and returns the owned raw cuTENSORMg descriptor.
    ///
    /// The caller becomes responsible for destroying the descriptor.
    #[must_use]
    pub fn into_raw(self) -> sys::cutensorMgContractionDescriptor_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl ContractionPlan {
    /// Selects a contraction plan under the supplied per-device and host workspace byte limits.
    ///
    /// # Errors
    ///
    /// Returns an error if descriptors or search settings belong to another multi-GPU context, device workspace count is invalid, native plan creation fails, or the returned handle is null.
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
        // SAFETY: `cutensorMgCreateContractionPlan` receives live handles owned by the wrappers and valid output locations.
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
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
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

    /// Returns the workspace byte limits retained by this plan.
    #[must_use]
    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// Allocates pending device and pinned host workspace for this plan.
    ///
    /// # Errors
    ///
    /// Returns an error if workspace allocation or byte-count conversion fails, or the context has no participating device.
    pub fn create_workspace_memory(&self) -> Result<WorkspaceMemory> {
        WorkspaceMemory::create(&self.context, &self.workspace)
    }

    /// Submits `D = alpha * A * B + beta * C` using distributed device allocations.
    ///
    /// Element and scalar representations, partition capacities, and layouts must match the descriptors used to create the plan. The wrapper validates pointer counts, contexts, and workspace capacity. Tensor pointer arrays follow their respective descriptor device order. Device workspace pointers and streams follow context device order. Native submission can leave CUDA work pending; retain tensor storage, workspace, plans, and streams and preserve producer ordering and exclusive output access through completion on every path.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor pointer counts do not match the plan, workspace or stream counts do not match context devices, a tensor requests unsupported host storage or a nonparticipating device, allocation or stream contexts differ, workspace byte counts cannot fit `usize`, workspace capacity is insufficient, or native contraction submission fails.
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

        a.validate_context(&self.context)?;
        b.validate_context(&self.context)?;
        workspace.validate(&self.context, &self.workspace)?;
        let a = a.const_ptrs();
        let b = b.const_ptrs();
        c.validate_context(&self.context)?;
        d.validate_context(&self.context)?;
        let c = c.const_ptrs();
        let mut d = d.mut_ptrs();
        let mut device_workspace = workspace.device_mut_ptrs();

        // SAFETY: the tensors, scalars and workspace were validated against this plan context above and stay borrowed for the call.
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

    /// Submits a contraction with the accumulated C tensor and output D sharing storage.
    ///
    /// Element and scalar representations, partition capacities, and layouts must match the descriptors used to create the plan. The wrapper validates pointer counts, contexts, and workspace capacity. Tensor pointer arrays follow their respective descriptor device order. Device workspace pointers and streams follow context device order. Native submission can leave CUDA work pending; retain tensor storage, workspace, plans, and streams and preserve producer ordering and exclusive output access through completion on every path.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor pointer counts do not match the plan, workspace or stream counts do not match context devices, a tensor requests unsupported host storage or a nonparticipating device, allocation or stream contexts differ, workspace byte counts cannot fit `usize`, workspace capacity is insufficient, or native contraction submission fails.
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

        a.validate_context(&self.context)?;
        b.validate_context(&self.context)?;
        workspace.validate(&self.context, &self.workspace)?;
        let a = a.const_ptrs();
        let b = b.const_ptrs();
        c_and_d.validate_context(&self.context)?;
        let c = c_and_d.const_ptrs();
        let mut d = c_and_d.mut_ptrs();
        let mut device_workspace = workspace.device_mut_ptrs();

        // SAFETY: the tensors, scalars and workspace were validated against this plan context above and stay borrowed for the call.
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

    /// Submits an in-place contraction on every context device's default stream.
    ///
    /// Element and scalar representations, partition capacities, and layouts must match the descriptors used to create the plan. The wrapper validates pointer counts, contexts, and workspace capacity. Tensor pointer arrays follow their respective descriptor device order. Device workspace pointers and streams follow context device order. Native submission can leave CUDA work pending; retain tensor storage, workspace, plans, and streams and preserve producer ordering and exclusive output access through completion on every path.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor pointer counts do not match the plan, workspace or stream counts do not match context devices, a tensor requests unsupported host storage or a nonparticipating device, allocation or stream contexts differ, workspace byte counts cannot fit `usize`, workspace capacity is insufficient, or native contraction submission fails.
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

        a.validate_context(&self.context)?;
        b.validate_context(&self.context)?;
        workspace.validate(&self.context, &self.workspace)?;
        let a = a.const_ptrs();
        let b = b.const_ptrs();
        c_and_d.validate_context(&self.context)?;
        let c = c_and_d.const_ptrs();
        let mut d = c_and_d.mut_ptrs();
        let mut device_workspace = workspace.device_mut_ptrs();

        // SAFETY: the tensors, scalars and workspace were validated against this plan context above and stay borrowed for the call.
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
    ///
    /// Tensor pointer arrays follow their respective descriptor device order. Device workspace pointers and streams follow context device order. Native submission can leave CUDA work pending; retain tensor storage, workspace, plans, and streams and preserve producer ordering and exclusive output access through completion on every path.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor pointer counts differ from the plan, workspace or stream counts differ from the context device count, a stream belongs to the wrong device context, or native contraction submission fails.
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
        for (stream, &device) in streams.iter().zip(self.context.devices()) {
            if stream.context() != self.context.cuda_context(device)? {
                return Err(Error::StreamContextMismatch);
            }
        }
        let mut streams: Vec<_> = streams.iter().map(StreamBinding::as_raw).collect();
        // SAFETY: `cutensorMgContraction` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cutensorMgContraction(
                self.context.as_raw(),
                self.handle,
                alpha.cast(),
                a.as_mut_ptr().cast(),
                b.as_mut_ptr().cast(),
                beta.cast(),
                c.as_mut_ptr().cast(),
                d.as_mut_ptr().cast(),
                device_workspace.as_mut_ptr().cast(),
                host_workspace.cast(),
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
    ///
    /// Tensor pointer arrays follow their respective descriptor device order. Device workspace pointers and streams follow context device order. Native submission can leave CUDA work pending; retain tensor storage, workspace, plans, and streams and preserve producer ordering and exclusive output access through completion on every path.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor pointer counts differ from the plan, workspace or stream counts differ from the context device count, or native contraction submission fails.
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
        // SAFETY: `cutensorMgContraction` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cutensorMgContraction(
                self.context.as_raw(),
                self.handle,
                alpha.cast(),
                a.as_mut_ptr().cast(),
                b.as_mut_ptr().cast(),
                beta.cast(),
                c.as_mut_ptr().cast(),
                d.as_mut_ptr().cast(),
                device_workspace.as_mut_ptr().cast(),
                host_workspace.cast(),
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

    /// Returns the borrowed native Contraction plan handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorMgContractionPlan_t {
        self.handle
    }

    /// Consumes this plan and returns the owned raw cuTENSORMg contraction plan.
    ///
    /// The caller becomes responsible for destroying the plan.
    #[must_use]
    pub fn into_raw(self) -> sys::cutensorMgContractionPlan_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl Drop for ContractionFind {
    fn drop(&mut self) {
        // SAFETY: `cutensorMgDestroyContractionFind` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroyContractionFind(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormg contraction find: {error}");
            }
        }
    }
}

impl Drop for ContractionDescriptor {
    fn drop(&mut self) {
        // SAFETY: `cutensorMgDestroyContractionDescriptor` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroyContractionDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormg contraction descriptor: {error}");
            }
        }
    }
}

impl Drop for ContractionPlan {
    fn drop(&mut self) {
        // SAFETY: `cutensorMgDestroyContractionPlan` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroyContractionPlan(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormg contraction plan: {error}");
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

impl_enum_display!(ContractionFindAttribute, {
    ContractionFindAttribute::Max => "CUTENSORMG_CONTRACTION_FIND_ATTRIBUTE_MAX",
});
