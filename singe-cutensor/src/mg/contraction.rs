use std::{mem::ManuallyDrop, ptr};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_display, report_cleanup_failure};
use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    stream::{AsStream, StreamBinding},
};
use singe_cutensor_sys as sys;

use crate::{
    error::{Error, Result},
    mg::{
        context::{Context, ContextRef, validate_same_context},
        copy::{validate_count, validate_streams},
        memory::{DistributedDeviceMemory, WorkspaceMemory},
        tensor::{TensorDescriptor, TensorDevice},
        types::Algorithm,
        workspace::Workspace,
    },
    streams::{join_producers, sync_streams},
    try_ffi,
    types::{ComputeType, Mode, WorkspacePreference},
    utility::{modes_to_i32_vec, to_usize},
};

singe_core::native_enum! {
    /// Represents various attributes for the contraction find
    ///
    ///
    ///
    /// While empty for now, future attributes will allow for a more fine-grained tuning of the distributed tensor contraction.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ContractionFindAttribute: sys::cutensorMgContractionFindAttribute_t {
        ///< Not an actual attribute, but the maximum value that the enum takes on
        Max = CUTENSORMG_CONTRACTION_FIND_ATTRIBUTE_MAX,
    }
}

/// Owns the algorithm selection used to plan a multi-GPU contraction.
#[derive(Debug)]
pub struct ContractionFind {
    handle: sys::cutensorMgContractionFind_t,
    context: ContextRef,
    algorithm: Algorithm,
}

/// Transferred native contraction search settings with their retained multi-GPU context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`ContractionFind::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawContractionFind {
    /// Transferred native search-settings handle; destroy it before releasing the retained context.
    pub handle: sys::cutensorMgContractionFind_t,
    /// Multi-GPU context the search settings were created in.
    pub context: ContextRef,
    /// Algorithm the search settings were created for.
    pub algorithm: Algorithm,
}

/// Owns a distributed contraction description and the operand types and storage spans checked at execution.
#[derive(Debug)]
pub struct ContractionDescriptor {
    handle: sys::cutensorMgContractionDescriptor_t,
    context: ContextRef,
    signature: Signature,
}

/// Transferred native contraction descriptor with its operand metadata and retained multi-GPU context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`ContractionDescriptor::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawContractionDescriptor {
    /// Transferred native descriptor handle; destroy it before releasing the retained context.
    pub handle: sys::cutensorMgContractionDescriptor_t,
    /// Multi-GPU context the descriptor was created in.
    pub context: ContextRef,
    // Operand metadata is wrapper state, so only `into_raw` can produce this record.
    signature: Signature,
}

/// Owns a distributed contraction plan, workspace requirements, and its library context.
#[derive(Debug)]
pub struct ContractionPlan {
    handle: sys::cutensorMgContractionPlan_t,
    context: ContextRef,
    signature: Signature,
    workspace: Workspace,
}

/// Transferred native contraction plan with its operand metadata, workspace requirements, and retained multi-GPU context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`ContractionPlan::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawContractionPlan {
    /// Transferred native plan handle; destroy it before releasing the retained context.
    pub handle: sys::cutensorMgContractionPlan_t,
    /// Multi-GPU context the plan was created in.
    pub context: ContextRef,
    // Operand metadata is wrapper state, so only `into_raw` can produce this record.
    signature: Signature,
    /// Workspace byte limits the plan was created under.
    pub workspace: Workspace,
}

#[derive(Debug, Clone)]
struct Signature {
    a: Operand,
    b: Operand,
    c: Operand,
    d: Operand,
    // `None` when cuTENSOR documents no scalar type for the output type.
    scalar_type: Option<DataType>,
}

#[derive(Debug, Clone)]
pub(crate) struct Operand {
    data_type: DataType,
    devices: Vec<TensorDevice>,
    // `None` when the wrapper cannot bound the layout's storage span.
    partition_bytes: Option<usize>,
}

impl ContractionFind {
    /// Creates contraction kernel-search settings for an algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if native search-setting creation fails or returns a null handle.
    pub fn create(context: &Context, algorithm: Algorithm) -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `context` keeps the multi-GPU handle live, `algorithm` is a valid enum value, and `handle` is a local output.
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
        // SAFETY: `self.handle` and `self.context` are live, and `value` points to `size_of::<T>()` readable bytes, the size passed.
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

    /// Consumes these search settings and returns the native handle without destroying it.
    ///
    /// The returned record retains the multi-GPU context and the algorithm.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawContractionFind {
        let owner = ManuallyDrop::new(self);
        // SAFETY: `owner` is never dropped, so each field is read out exactly once.
        unsafe {
            RawContractionFind {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
                algorithm: owner.algorithm,
            }
        }
    }

    /// Takes ownership of native contraction search settings with their algorithm and multi-GPU context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid contraction find handle created in `raw.context` for `raw.algorithm`, and uniquely owned.
    /// The handle must not be destroyed elsewhere afterward.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawContractionFind) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: raw.handle,
            context: raw.context,
            algorithm: raw.algorithm,
        })
    }
}

impl ContractionDescriptor {
    /// Describes `D = alpha * A * B + beta * C` using symbolic tensor modes.
    ///
    /// The descriptor records each operand's data type, storage devices, and partition span for checked execution.
    ///
    /// # Errors
    ///
    /// Returns an error if an operand belongs to another multi-GPU context, a mode count differs from its tensor rank, an operand's block size or device count is zero or negative, an operand's partition span overflows, native descriptor creation fails, or the returned handle is null.
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
        let signature = Signature::create(a, b, c, d)?;

        let modes_a = modes_to_i32_vec(modes_a);
        let modes_b = modes_to_i32_vec(modes_b);
        let modes_c = modes_to_i32_vec(modes_c);
        let modes_d = modes_to_i32_vec(modes_d);
        let mut handle = ptr::null_mut();
        // SAFETY: `context` and the operand descriptors (checked to share it above) are live, each mode array has its descriptor's rank (`validate_modes`), and `handle` is a local output.
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
            signature,
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
        // SAFETY: `self.handle` and `find` (checked to share `self.context`) are live, `device_workspace_size` has one slot per context device, and `host_workspace_size` is a local output.
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

    /// Consumes this descriptor and returns the native handle without destroying it.
    ///
    /// The returned record retains the multi-GPU context and operand metadata.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawContractionDescriptor {
        let owner = ManuallyDrop::new(self);
        // SAFETY: `owner` is never dropped, so each field is read out exactly once.
        unsafe {
            RawContractionDescriptor {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
                signature: ptr::read(&raw const owner.signature),
            }
        }
    }

    /// Reclaims a transferred native descriptor with its operand metadata and multi-GPU context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must remain valid and uniquely owned.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawContractionDescriptor) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: raw.handle,
            context: raw.context,
            signature: raw.signature,
        })
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
        // SAFETY: `context`, `desc`, and `find` (checked to share one handle) are live, `workspace` has one size per context device (`validate_for_context`), and `handle` is a local output.
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
            signature: desc.signature.clone(),
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

    /// Enqueues `D = alpha * A * B + beta * C` over distributed device allocations.
    ///
    /// Each operand's element type must be the data type of its tensor descriptor, and `TScalar` must be the scalar type cuTENSOR assigns to D's data type.
    /// Each operand's partitions must lie on its descriptor's devices, in descriptor order, and each partition must cover the storage span the descriptor's layout reaches on one device.
    /// The wrapper bounds that span as if every device held the most blocks any device holds along each mode.
    /// Device workspace and `streams` follow context device order; [`Context::default_streams`] gives the legacy default streams.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution contract](crate#raw-execution), for every stream in `streams`.
    ///
    /// # Errors
    ///
    /// Returns an error if an element type differs from its descriptor's data type, `TScalar` differs from the scalar type or D's data type has no documented scalar type, an operand's partitions lie on other devices than its descriptor or in another context, a partition is shorter than its descriptor's span, a layout sets only one of its element and block strides where the other matters, workspace or stream counts do not match context devices, a stream belongs to another context, workspace capacity is insufficient, or native contraction submission fails.
    pub unsafe fn contract_async<
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    >(
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
        self.signature.validate_scalar::<TScalar>()?;
        self.signature.a.validate(a, "a", &self.context)?;
        self.signature.b.validate(b, "b", &self.context)?;
        self.signature.c.validate(c, "c", &self.context)?;
        self.signature.d.validate(d, "d", &self.context)?;
        workspace.validate(&self.context, &self.workspace)?;
        self.validate_execution_slices(
            a.pointer_count(),
            b.pointer_count(),
            c.pointer_count(),
            d.pointer_count(),
            workspace.device_pointer_count(),
            streams,
        )?;
        join_producers(
            streams,
            &[
                a.regions(),
                b.regions(),
                c.regions(),
                d.regions(),
                workspace.device_regions(),
            ]
            .concat(),
        )?;

        let a = a.const_ptrs();
        let b = b.const_ptrs();
        let c = c.const_ptrs();
        let mut d = d.mut_ptrs();
        let mut device_workspace = workspace.device_mut_ptrs();
        // SAFETY: the operands, scalar type, workspace, and streams were checked against the plan above; the caller keeps them alive through completion.
        unsafe {
            self.enqueue(
                ptr::from_ref(alpha).cast(),
                a,
                b,
                ptr::from_ref(beta).cast(),
                c,
                &mut d,
                &mut device_workspace,
                workspace.host_mut_ptr(),
                streams,
            )
        }
    }

    /// Computes `D = alpha * A * B + beta * C` over distributed device allocations and waits for every stream.
    ///
    /// The operands follow the rules of [`Self::contract_async`].
    ///
    /// # Errors
    ///
    /// Returns an error if a stream is capturing, the checks of [`Self::contract_async`] fail, native contraction submission fails, or a stream reports an error.
    pub fn contract<
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    >(
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
        sync_streams(streams, || {
            // SAFETY: every operand and the workspace stay borrowed until `sync_streams` has drained every stream, and the output borrows are exclusive.
            unsafe { self.contract_async(alpha, a, b, beta, c, d, workspace, streams) }
        })
    }

    /// Enqueues a contraction with the accumulated C tensor and output D sharing storage.
    ///
    /// `c_and_d` must satisfy both the C and the D descriptor, with the requirements of [`Self::contract_async`].
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution contract](crate#raw-execution), for every stream in `streams`.
    ///
    /// # Errors
    ///
    /// Returns an error under the conditions of [`Self::contract_async`].
    pub unsafe fn contract_in_place_async<
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TScalar: DataTypeLike,
    >(
        &self,
        alpha: &TScalar,
        a: &DistributedDeviceMemory<TA>,
        b: &DistributedDeviceMemory<TB>,
        beta: &TScalar,
        c_and_d: &mut DistributedDeviceMemory<TC>,
        workspace: &mut WorkspaceMemory,
        streams: &[StreamBinding],
    ) -> Result<()> {
        self.signature.validate_scalar::<TScalar>()?;
        self.signature.a.validate(a, "a", &self.context)?;
        self.signature.b.validate(b, "b", &self.context)?;
        self.signature.c.validate(c_and_d, "c", &self.context)?;
        self.signature.d.validate(c_and_d, "d", &self.context)?;
        workspace.validate(&self.context, &self.workspace)?;
        self.validate_execution_slices(
            a.pointer_count(),
            b.pointer_count(),
            c_and_d.pointer_count(),
            c_and_d.pointer_count(),
            workspace.device_pointer_count(),
            streams,
        )?;
        join_producers(
            streams,
            &[
                a.regions(),
                b.regions(),
                c_and_d.regions(),
                workspace.device_regions(),
            ]
            .concat(),
        )?;

        let a = a.const_ptrs();
        let b = b.const_ptrs();
        let c = c_and_d.const_ptrs();
        let mut d = c_and_d.mut_ptrs();
        let mut device_workspace = workspace.device_mut_ptrs();
        // SAFETY: the operands, scalar type, workspace, and streams were checked against the plan above; the caller keeps them alive through completion.
        unsafe {
            self.enqueue(
                ptr::from_ref(alpha).cast(),
                a,
                b,
                ptr::from_ref(beta).cast(),
                c,
                &mut d,
                &mut device_workspace,
                workspace.host_mut_ptr(),
                streams,
            )
        }
    }

    /// Computes a contraction with C and D sharing storage and waits for every stream.
    ///
    /// The operands follow the rules of [`Self::contract_in_place_async`].
    ///
    /// # Errors
    ///
    /// Returns an error if a stream is capturing, the checks of [`Self::contract_async`] fail, native contraction submission fails, or a stream reports an error.
    pub fn contract_in_place<
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TScalar: DataTypeLike,
    >(
        &self,
        alpha: &TScalar,
        a: &DistributedDeviceMemory<TA>,
        b: &DistributedDeviceMemory<TB>,
        beta: &TScalar,
        c_and_d: &mut DistributedDeviceMemory<TC>,
        workspace: &mut WorkspaceMemory,
        streams: &[StreamBinding],
    ) -> Result<()> {
        sync_streams(streams, || {
            // SAFETY: every operand and the workspace stay borrowed until `sync_streams` has drained every stream, and the output borrows are exclusive.
            unsafe { self.contract_in_place_async(alpha, a, b, beta, c_and_d, workspace, streams) }
        })
    }

    // Submits the native contraction over checked pointer tables.
    unsafe fn enqueue(
        &self,
        alpha: *const (),
        mut a: Vec<*const ()>,
        mut b: Vec<*const ()>,
        beta: *const (),
        mut c: Vec<*const ()>,
        d: &mut [*mut ()],
        device_workspace: &mut [*mut ()],
        host_workspace: *mut (),
        streams: &[StreamBinding],
    ) -> Result<()> {
        let mut streams: Vec<_> = streams.iter().map(AsStream::as_raw).collect();
        // SAFETY: the caller passes pointer tables, scalars, workspace, and streams checked against this plan.
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
            ))
        }
    }

    fn validate_execution_slices(
        &self,
        a_count: usize,
        b_count: usize,
        c_count: usize,
        d_count: usize,
        device_workspace_count: usize,
        streams: &[StreamBinding],
    ) -> Result<()> {
        validate_count("a", self.signature.a.devices.len(), a_count)?;
        validate_count("b", self.signature.b.devices.len(), b_count)?;
        validate_count("c", self.signature.c.devices.len(), c_count)?;
        validate_count("d", self.signature.d.devices.len(), d_count)?;
        validate_count(
            "device_workspace",
            self.context.device_count(),
            device_workspace_count,
        )?;
        validate_streams(&self.context, streams)
    }

    /// Returns the borrowed native Contraction plan handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorMgContractionPlan_t {
        self.handle
    }

    /// Consumes this plan and returns the native handle without destroying it.
    ///
    /// The returned record retains the multi-GPU context, operand metadata, and workspace requirements.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawContractionPlan {
        let owner = ManuallyDrop::new(self);
        // SAFETY: `owner` is never dropped, so each field is read out exactly once.
        unsafe {
            RawContractionPlan {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
                signature: ptr::read(&raw const owner.signature),
                workspace: ptr::read(&raw const owner.workspace),
            }
        }
    }

    /// Reclaims a transferred native plan with its operand metadata, workspace requirements, and multi-GPU context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must remain valid and uniquely owned, created under `raw.workspace`.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null or the device workspace count differs from the context device count.
    pub unsafe fn from_raw(raw: RawContractionPlan) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        raw.workspace.validate_for_context(&raw.context)?;
        Ok(Self {
            handle: raw.handle,
            context: raw.context,
            signature: raw.signature,
            workspace: raw.workspace,
        })
    }
}

impl Signature {
    fn create(
        a: &TensorDescriptor,
        b: &TensorDescriptor,
        c: &TensorDescriptor,
        d: &TensorDescriptor,
    ) -> Result<Self> {
        Ok(Self {
            a: Operand::create(a)?,
            b: Operand::create(b)?,
            c: Operand::create(c)?,
            d: Operand::create(d)?,
            scalar_type: scalar_type(d.data_type()),
        })
    }

    fn validate_scalar<T: DataTypeLike>(&self) -> Result<()> {
        let actual = T::data_type();
        match self.scalar_type {
            Some(expected) if expected == actual => Ok(()),
            Some(expected) => Err(Error::ScalarDataTypeMismatch {
                descriptor: expected,
                memory: actual,
            }),
            None => Err(Error::UnsupportedScalarDataType {
                name: "alpha".into(),
                data_type: actual,
            }),
        }
    }
}

impl Operand {
    pub(crate) fn create(tensor: &TensorDescriptor) -> Result<Self> {
        let partition_bytes = tensor
            .partition_len()?
            .map(|len| {
                to_usize(len, "tensor partition length")?
                    .checked_mul(tensor.data_type().size_of())
                    .ok_or_else(|| Error::out_of_range("tensor partition bytes"))
            })
            .transpose()?;
        Ok(Self {
            data_type: tensor.data_type(),
            devices: tensor.devices().to_vec(),
            partition_bytes,
        })
    }

    pub(crate) fn validate<T: DataTypeLike>(
        &self,
        memory: &DistributedDeviceMemory<T>,
        name: &'static str,
        context: &ContextRef,
    ) -> Result<()> {
        if T::data_type() != self.data_type {
            return Err(Error::TensorMemoryDataTypeMismatch {
                descriptor: self.data_type,
                memory: T::data_type(),
            });
        }

        if memory.devices() != self.devices.as_slice() {
            return Err(Error::MgDeviceMismatch { name: name.into() });
        }

        memory.validate_context(context)?;
        let required = self
            .partition_bytes
            .ok_or_else(|| Error::MgUncheckedLayout { name: name.into() })?;
        for buffer in memory.buffers() {
            if buffer.byte_len() < required {
                return Err(Error::BufferTooSmall {
                    name: name.into(),
                    required,
                    actual: buffer.byte_len(),
                });
            }
        }
        Ok(())
    }
}

// The scaling factors' type, from the scalar type table in the cuTENSOR user guide.
fn scalar_type(output: DataType) -> Option<DataType> {
    match output {
        DataType::F16 | DataType::Bf16 | DataType::F32 => Some(DataType::F32),
        DataType::F64 => Some(DataType::F64),
        DataType::ComplexF16 | DataType::ComplexBf16 | DataType::ComplexF32 => {
            Some(DataType::ComplexF32)
        }
        DataType::ComplexF64 => Some(DataType::ComplexF64),
        _ => None,
    }
}

impl Drop for ContractionFind {
    fn drop(&mut self) {
        // SAFETY: the search settings exclusively own their non-null handle, and `self.context` keeps the multi-GPU handle live.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroyContractionFind(self.handle)) {
                report_cleanup_failure("destroy", "cutensormg contraction find", &error);
            }
        }
    }
}

impl Drop for ContractionDescriptor {
    fn drop(&mut self) {
        // SAFETY: the descriptor exclusively owns its non-null handle, and `self.context` keeps the multi-GPU handle live.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroyContractionDescriptor(self.handle)) {
                report_cleanup_failure("destroy", "cutensormg contraction descriptor", &error);
            }
        }
    }
}

impl Drop for ContractionPlan {
    fn drop(&mut self) {
        // SAFETY: the plan exclusively owns its non-null handle, and `self.context` keeps the multi-GPU handle live.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroyContractionPlan(self.handle)) {
                report_cleanup_failure("destroy", "cutensormg contraction plan", &error);
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
