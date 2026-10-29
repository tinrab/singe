use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ptr,
};

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    memory::DeviceMemory,
    stream::Stream,
};
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

/// Owns a distributed operation and optional checked local-shard signature.
#[derive(Debug)]
pub struct OperationDescriptor<'comm> {
    signature: Option<Signature>,
    handle: sys::cutensorMpOperationDescriptor_t,
    context: ContextRef<'comm>,
}

/// Owns algorithm selection and rank-local device and host workspace byte limits.
#[derive(Debug)]
pub struct PlanPreference<'comm> {
    handle: sys::cutensorMpPlanPreference_t,
    context: ContextRef<'comm>,
    algorithm: Algorithm,
    device_workspace_limit: u64,
    host_workspace_limit: u64,
}

/// Owns a distributed execution plan, optional checked signature, and retained execution resources.
#[derive(Debug)]
pub struct Plan<'comm> {
    signature: Option<Signature>,
    stream: Option<Stream>,
    handle: sys::cutensorMpPlan_t,
    context: ContextRef<'comm>,
}

/// Required device and host workspace capacities in bytes for the calling rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workspace {
    device_size: u64,
    host_size: u64,
}

impl<'comm> OperationDescriptor<'comm> {
    /// Describes `D = alpha * op_a(A) * op_b(B) + beta * op_c(C)` across communicator ranks.
    ///
    /// The wrapper records each local shard's native sample type and required storage span for checked execution. Half and bfloat16 accumulated tensors use single-precision scaling factors.
    ///
    /// # Errors
    ///
    /// Returns an error if operands belong to another multi-process context, mode counts differ from tensor ranks, local-layout queries fail, CUDA context binding or native descriptor creation fails, or the returned handle is null.
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
        let signature = Some(Signature::Contraction {
            a: Shard::create(a)?,
            b: Shard::create(b)?,
            c: Shard::create(c)?,
            d: Shard::create(d)?,
            scalar_type: match c.data_type() {
                DataType::F16 | DataType::Bf16 => DataType::F32,
                other => other,
            },
        });
        let mut handle = ptr::null_mut();
        context.bind()?;
        // SAFETY: `cutensorMpCreateContraction` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
            signature,
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
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
    pub unsafe fn from_raw(
        handle: sys::cutensorMpOperationDescriptor_t,
        context: &Context<'comm>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            signature: None,
            handle,
            context: context.as_context_ref(),
        })
    }

    pub(crate) fn context(&self) -> &ContextRef<'comm> {
        &self.context
    }

    /// Returns the borrowed native operation descriptor handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorMpOperationDescriptor_t {
        self.handle
    }

    /// Consumes this descriptor and returns the owned raw cuTENSORMp operation descriptor.
    ///
    /// The caller becomes responsible for destroying the descriptor.
    #[must_use]
    pub fn into_raw(self) -> sys::cutensorMpOperationDescriptor_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl<'comm> PlanPreference<'comm> {
    /// Creates algorithm selection and local workspace limits measured in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding or native preference creation fails, the selection is unsupported, or the returned handle is null.
    pub fn create(
        context: &Context<'comm>,
        algorithm: Algorithm,
        device_workspace_limit: u64,
        host_workspace_limit: u64,
    ) -> Result<Self> {
        let mut handle = ptr::null_mut();
        context.bind()?;
        // SAFETY: `cutensorMpCreatePlanPreference` receives live handles owned by the wrappers and valid output locations.
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
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
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

    /// Returns the algorithm recorded by these preferences.
    #[must_use]
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Returns this rank's device workspace byte limit.
    #[must_use]
    pub fn device_workspace_limit(&self) -> u64 {
        self.device_workspace_limit
    }

    /// Returns this rank's host workspace byte limit.
    #[must_use]
    pub fn host_workspace_limit(&self) -> u64 {
        self.host_workspace_limit
    }

    pub(crate) fn context(&self) -> &ContextRef<'comm> {
        &self.context
    }

    /// Returns the borrowed native plan preference handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorMpPlanPreference_t {
        self.handle
    }

    /// Consumes this preference and returns the owned raw cuTENSORMp plan preference.
    ///
    /// The caller becomes responsible for destroying the preference.
    #[must_use]
    pub fn into_raw(self) -> sys::cutensorMpPlanPreference_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl<'comm> Plan<'comm> {
    /// Collectively plans a distributed operation with optional algorithm and local workspace preferences.
    ///
    /// All communicator ranks must provide matching operation descriptions and algorithm choices. Workspace limits are local resource budgets and may differ by rank. Plans created from raw descriptors without tracked signatures require explicit raw execution.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation or preferences belong to another multi-process context, CUDA context binding or native planning fails, or the returned handle is null.
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
        // SAFETY: `cutensorMpCreatePlan` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cutensorMpCreatePlan(
                context.as_raw(),
                &raw mut handle,
                desc.as_raw(),
                preference.map_or(ptr::null_mut(), PlanPreference::as_raw),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            signature: desc.signature.clone(),
            stream: None,
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
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
    pub unsafe fn from_raw(
        handle: sys::cutensorMpPlan_t,
        context: &Context<'comm>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            signature: None,
            stream: None,
            handle,
            context: context.as_context_ref(),
        })
    }

    /// Queries this rank's required device and host workspace capacities in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding or either native workspace-attribute query fails.
    pub fn workspace(&self) -> Result<Workspace> {
        Ok(Workspace {
            device_size: self.attribute(PlanAttribute::RequiredWorkspaceDevice)?,
            host_size: self.attribute(PlanAttribute::RequiredWorkspaceHost)?,
        })
    }

    /// Allocates pending device and pinned host workspace in this plan's actual CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if workspace queries fail, a byte count cannot fit `usize`, or device or pinned host allocation fails.
    pub fn create_workspace_memory(&self) -> Result<WorkspaceMemory<'comm>> {
        WorkspaceMemory::create(self.context.cuda_context(), self.workspace()?)
    }

    /// Submits a distributed contraction using checked local device allocations.
    ///
    /// Execution uses the plan's configured stream or the handle stream and may leave CUDA work pending. All participating communicator ranks must issue matching operations. Preserve input readiness and retain allocations, plans, and workspace through host and device completion on every path.
    ///
    /// # Errors
    ///
    /// Returns an error if no tracked contraction signature is available, a tensor sample type differs from the signature or its storage span is too small, scalar types differ, workspace belongs to another context or has insufficient capacity, a workspace byte count cannot fit `usize`, CUDA context binding or workspace querying fails, or native contraction submission fails.
    pub fn contract<
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TD: DataTypeLike,
        TScalar: DataTypeLike,
    >(
        &self,
        alpha: &TScalar,
        a: &DeviceMemory<TA>,
        b: &DeviceMemory<TB>,
        beta: &TScalar,
        c: &DeviceMemory<TC>,
        d: &mut DeviceMemory<TD>,
        workspace: &mut WorkspaceMemory<'comm>,
    ) -> Result<()> {
        let Some(Signature::Contraction {
            a: sa,
            b: sb,
            c: sc,
            d: sd,
            scalar_type,
        }) = &self.signature
        else {
            return Err(operation_mismatch("contraction"));
        };
        sa.validate(a, "a")?;
        sb.validate(b, "b")?;
        sc.validate(c, "c")?;
        sd.validate(d, "d")?;
        if TScalar::data_type() != *scalar_type {
            return Err(Error::ScalarDataTypeMismatch {
                descriptor: *scalar_type,
                memory: TScalar::data_type(),
            });
        }
        workspace.prepare(&self.context, self.workspace()?)?;
        // SAFETY: the tensors, scalars and workspace were validated against this plan context above and stay borrowed for the call.
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

    /// Submits a distributed contraction with C and D sharing the same local storage.
    ///
    /// Execution uses the plan's configured stream or the handle stream and may leave CUDA work pending. All participating communicator ranks must issue matching operations. Preserve input readiness and retain allocations, plans, and workspace through host and device completion on every path.
    ///
    /// # Errors
    ///
    /// Returns an error if no tracked contraction signature is available, a tensor sample type differs from the signature or its storage span is too small, scalar types differ, workspace belongs to another context or has insufficient capacity, a workspace byte count cannot fit `usize`, CUDA context binding or workspace querying fails, or native contraction submission fails.
    pub fn contract_in_place<
        TA: DataTypeLike,
        TB: DataTypeLike,
        TC: DataTypeLike,
        TScalar: DataTypeLike,
    >(
        &self,
        alpha: &TScalar,
        a: &DeviceMemory<TA>,
        b: &DeviceMemory<TB>,
        beta: &TScalar,
        c_and_d: &mut DeviceMemory<TC>,
        workspace: &mut WorkspaceMemory<'comm>,
    ) -> Result<()> {
        let Some(Signature::Contraction {
            a: sa,
            b: sb,
            c: sc,
            d: sd,
            scalar_type,
        }) = &self.signature
        else {
            return Err(operation_mismatch("contraction"));
        };
        sa.validate(a, "a")?;
        sb.validate(b, "b")?;
        sc.validate(c_and_d, "c")?;
        sd.validate(c_and_d, "d")?;
        if TScalar::data_type() != *scalar_type {
            return Err(Error::ScalarDataTypeMismatch {
                descriptor: *scalar_type,
                memory: TScalar::data_type(),
            });
        }
        workspace.prepare(&self.context, self.workspace()?)?;
        // SAFETY: the tensors, scalars and workspace were validated against this plan context above and stay borrowed for the call.
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
    ///
    /// Execution uses the plan's configured stream or the handle stream and may leave CUDA work pending. All participating communicator ranks must issue matching operations. Preserve input readiness and retain allocations, plans, and workspace through host and device completion on every path.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding or native contraction submission fails.
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
        let _lock = self.context.lock_execution();
        self.context.bind()?;
        // SAFETY: `cutensorMpContract` receives live handles owned by the wrappers.
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

    /// Returns the borrowed native execution plan handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::cutensorMpPlan_t {
        self.handle
    }

    /// Consumes this plan and returns the owned raw cuTENSORMp plan.
    ///
    /// The caller becomes responsible for destroying the plan.
    #[must_use]
    pub fn into_raw(self) -> sys::cutensorMpPlan_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }

    fn attribute<T: Copy>(&self, attr: PlanAttribute) -> Result<T> {
        let mut value = MaybeUninit::<T>::uninit();
        self.context.bind()?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
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
    /// Returns this rank's required device workspace bytes.
    #[must_use]
    pub fn device_size(&self) -> u64 {
        self.device_size
    }

    /// Returns this rank's required host workspace bytes.
    #[must_use]
    pub fn host_size(&self) -> u64 {
        self.host_size
    }
}

impl Drop for OperationDescriptor<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!(
                "failed to bind cutensormp context before destroying operation descriptor: {error}"
            );
        }

        // SAFETY: `cutensorMpDestroyOperationDescriptor` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMpDestroyOperationDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormp operation descriptor: {error}");
            }
        }
    }
}

impl Drop for PlanPreference<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!(
                "failed to bind cutensormp context before destroying plan preference: {error}"
            );
        }

        // SAFETY: `cutensorMpDestroyPlanPreference` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMpDestroyPlanPreference(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormp plan preference: {error}");
            }
        }
    }
}

impl Drop for Plan<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cutensormp context before destroying plan: {error}");
        }

        // SAFETY: `cutensorMpDestroyPlan` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMpDestroyPlan(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormp plan: {error}");
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

#[derive(Debug, Clone)]
struct Shard {
    data_type: DataType,
    bytes: u64,
}

impl Shard {
    fn create(tensor: &TensorDescriptor<'_>) -> Result<Self> {
        Ok(Self {
            data_type: tensor.data_type(),
            bytes: tensor.local_layout()?.storage_bytes,
        })
    }

    fn validate<T: DataTypeLike>(&self, memory: &DeviceMemory<T>, _name: &str) -> Result<()> {
        if self.data_type != T::data_type() {
            return Err(Error::TensorMemoryDataTypeMismatch {
                descriptor: self.data_type,
                memory: T::data_type(),
            });
        }
        if (memory.byte_len() as u64) < self.bytes {
            return Err(Error::InsufficientWorkspaceSize {
                required: self.bytes,
                actual: memory.byte_len() as u64,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransferKind {
    Redistribute,
    Unfold,
    Fold,
}

#[derive(Debug, Clone)]
enum Signature {
    Contraction {
        a: Shard,
        b: Shard,
        c: Shard,
        d: Shard,
        scalar_type: DataType,
    },
    Transfer {
        kind: TransferKind,
        source: Shard,
        destination: Shard,
    },
}

fn operation_mismatch(expected: &str) -> Error {
    Error::PlanOperationMismatch {
        expected: expected.into(),
        actual: "different operation or unknown raw metadata".into(),
    }
}

/// Ordered tensor-mode groups forming the row and column axes of a matrix.
#[derive(Debug, Clone, Copy)]
pub struct ModeGroups<'a> {
    /// Positional tensor-mode indices forming the first matrix axis.
    pub rows: &'a [usize],
    /// Positional tensor-mode indices forming the second matrix axis.
    pub columns: &'a [usize],
}

impl<'comm> OperationDescriptor<'comm> {
    /// Creates an operation descriptor that encodes a distributed tensor redistribution.
    /// Source and destination must describe the same logical tensor by mode labels.
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor belongs to another multi-process context, mode counts differ from ranks, local-layout queries or CUDA context binding fail, native redistribution setup fails, or the returned handle is null.
    pub fn create_redistribution(
        context: &Context<'comm>,
        source: &TensorDescriptor<'comm>,
        source_modes: &[Mode],
        destination: &TensorDescriptor<'comm>,
        destination_modes: &[Mode],
    ) -> Result<Self> {
        validate_same_context(&context.as_context_ref(), source.context(), "source")?;
        validate_same_context(
            &context.as_context_ref(),
            destination.context(),
            "destination",
        )?;
        validate_modes(source, source_modes)?;
        validate_modes(destination, destination_modes)?;
        let source_modes = modes_to_i32_vec(source_modes);
        let destination_modes = modes_to_i32_vec(destination_modes);
        let signature = Some(Signature::Transfer {
            kind: TransferKind::Redistribute,
            source: Shard::create(source)?,
            destination: Shard::create(destination)?,
        });
        let mut handle = ptr::null_mut();
        context.bind()?;
        // SAFETY: `cutensorMpCreateRedistribution` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorMpCreateRedistribution(
                context.as_raw(),
                &raw mut handle,
                destination.as_raw(),
                destination_modes.as_ptr(),
                source.as_raw(),
                source_modes.as_ptr()
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

    /// Creates an operation descriptor for a distributed tensor unfolding (tensor -> matrix).
    /// The ordered mode groups form matrix axes by multiplying their extents.
    /// Matrix block sizes must be multiples of 32 for F32, or 16 otherwise.
    /// F16 and BF16 unfolding are currently unsupported.
    ///
    /// # Errors
    ///
    /// Returns an error if tensors belong to another context, the matrix rank is not two, mode groups do not partition all tensor modes exactly once, grouped extent arithmetic overflows or disagrees with matrix extents, an index cannot fit `i32`, local-layout queries or context binding fail, native operation setup fails, or the returned handle is null.
    pub fn create_unfolding(
        context: &Context<'comm>,
        source: &TensorDescriptor<'comm>,
        destination: &TensorDescriptor<'comm>,
        groups: ModeGroups<'_>,
    ) -> Result<Self> {
        Self::create_reshape(context, source, destination, groups, TransferKind::Unfold)
    }

    /// Creates an operation descriptor for a distributed tensor folding (matrix -> tensor).
    /// The groups describe the destination tensor. Replication factors may differ;
    /// the planner broadcasts data or drops redundant replicas as needed.
    ///
    /// # Errors
    ///
    /// Returns an error if tensors belong to another context, the matrix rank is not two, mode groups do not partition all tensor modes exactly once, grouped extent arithmetic overflows or disagrees with matrix extents, an index cannot fit `i32`, local-layout queries or context binding fail, native operation setup fails, or the returned handle is null.
    pub fn create_folding(
        context: &Context<'comm>,
        source: &TensorDescriptor<'comm>,
        destination: &TensorDescriptor<'comm>,
        groups: ModeGroups<'_>,
    ) -> Result<Self> {
        Self::create_reshape(context, source, destination, groups, TransferKind::Fold)
    }

    fn create_reshape(
        context: &Context<'comm>,
        source: &TensorDescriptor<'comm>,
        destination: &TensorDescriptor<'comm>,
        groups: ModeGroups<'_>,
        kind: TransferKind,
    ) -> Result<Self> {
        validate_same_context(&context.as_context_ref(), source.context(), "source")?;
        validate_same_context(
            &context.as_context_ref(),
            destination.context(),
            "destination",
        )?;
        let (tensor, matrix) = if kind == TransferKind::Unfold {
            (source, destination)
        } else {
            (destination, source)
        };
        let indices = groups
            .rows
            .iter()
            .chain(groups.columns)
            .copied()
            .collect::<Vec<_>>();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        if matrix.rank() != 2 || sorted != (0..tensor.rank() as usize).collect::<Vec<_>>() {
            return Err(Error::out_of_range("mode groups"));
        }
        for (axis, group) in [groups.rows, groups.columns].into_iter().enumerate() {
            let extent = group.iter().try_fold(1u64, |a, &index| {
                a.checked_mul(tensor.shape()[index])
                    .ok_or_else(|| Error::out_of_range("group extent"))
            })?;
            if matrix.shape()[axis] != extent {
                return Err(Error::out_of_range("matrix extent"));
            }
        }
        let indices = indices
            .iter()
            .map(|&i| i32::try_from(i).map_err(|_| Error::out_of_range("mode index")))
            .collect::<Result<Vec<_>>>()?;
        let sizes = [
            i64::try_from(groups.rows.len()).expect("matrix group rank fits native i64"),
            i64::try_from(groups.columns.len()).expect("matrix group rank fits native i64"),
        ];
        let signature = Some(Signature::Transfer {
            kind,
            source: Shard::create(source)?,
            destination: Shard::create(destination)?,
        });
        let mut handle = ptr::null_mut();
        context.bind()?;
        // SAFETY: `cutensorMpCreateUnfolding`, `cutensorMpCreateFolding` receive live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            match kind {
                TransferKind::Unfold => try_ffi!(sys::cutensorMpCreateUnfolding(
                    context.as_raw(),
                    &raw mut handle,
                    source.as_raw(),
                    2,
                    indices.as_ptr(),
                    sizes.as_ptr(),
                    destination.as_raw()
                ))?,
                TransferKind::Fold => try_ffi!(sys::cutensorMpCreateFolding(
                    context.as_raw(),
                    &raw mut handle,
                    source.as_raw(),
                    destination.as_raw(),
                    2,
                    indices.as_ptr(),
                    sizes.as_ptr()
                ))?,
                TransferKind::Redistribute => unreachable!(),
            }
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
}

impl<'comm> Plan<'comm> {
    /// Overrides the CUDA stream that the given plan executes on.
    /// Synchronizes previous work before switching; subsequent executions use this stream.
    /// This is a rank-local operation and performs no communication.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream belongs to another CUDA context, CUDA binding or synchronization fails, or the native stream update fails.
    pub fn set_stream(&mut self, stream: &Stream) -> Result<()> {
        if self.context.cuda_context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }
        let _lock = self.context.lock_execution();
        self.context.bind()?;
        self.context.cuda_context().synchronize()?;
        // SAFETY: `cutensorMpSetStream` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cutensorMpSetStream(
                self.context.as_raw(),
                self.handle,
                stream.as_raw().cast()
            ))?;
        }
        self.stream = Some(stream.clone());
        Ok(())
    }

    /// Performs a distributed tensor redistribution across multiple processes.
    ///
    /// Execution uses the plan's configured stream or the handle stream and may leave CUDA work pending. All participating communicator ranks must issue matching operations. Preserve input readiness and retain allocations, plans, and workspace through host and device completion on every path.
    ///
    /// # Errors
    ///
    /// Returns an error if the tracked plan operation does not match this transfer, source or destination sample type or local storage capacity is incompatible, workspace context or capacity is invalid, workspace byte conversion or queries fail, CUDA context binding fails, or native transfer submission fails.
    pub fn redistribute<T: DataTypeLike>(
        &self,
        source: &DeviceMemory<T>,
        destination: &mut DeviceMemory<T>,
        workspace: &mut WorkspaceMemory<'comm>,
    ) -> Result<()> {
        self.transfer(TransferKind::Redistribute, source, destination, workspace)
    }

    /// Performs a distributed tensor unfolding (tensor -> matrix) across multiple processes.
    ///
    /// Execution uses the plan's configured stream or the handle stream and may leave CUDA work pending. All participating communicator ranks must issue matching operations. Preserve input readiness and retain allocations, plans, and workspace through host and device completion on every path.
    ///
    /// # Errors
    ///
    /// Returns an error if the tracked plan operation does not match this transfer, source or destination sample type or local storage capacity is incompatible, workspace context or capacity is invalid, workspace byte conversion or queries fail, CUDA context binding fails, or native transfer submission fails.
    pub fn unfold<T: DataTypeLike>(
        &self,
        source: &DeviceMemory<T>,
        destination: &mut DeviceMemory<T>,
        workspace: &mut WorkspaceMemory<'comm>,
    ) -> Result<()> {
        self.transfer(TransferKind::Unfold, source, destination, workspace)
    }

    /// Performs a distributed tensor folding (matrix -> tensor) across multiple processes.
    ///
    /// Execution uses the plan's configured stream or the handle stream and may leave CUDA work pending. All participating communicator ranks must issue matching operations. Preserve input readiness and retain allocations, plans, and workspace through host and device completion on every path.
    ///
    /// # Errors
    ///
    /// Returns an error if the tracked plan operation does not match this transfer, source or destination sample type or local storage capacity is incompatible, workspace context or capacity is invalid, workspace byte conversion or queries fail, CUDA context binding fails, or native transfer submission fails.
    pub fn fold<T: DataTypeLike>(
        &self,
        source: &DeviceMemory<T>,
        destination: &mut DeviceMemory<T>,
        workspace: &mut WorkspaceMemory<'comm>,
    ) -> Result<()> {
        self.transfer(TransferKind::Fold, source, destination, workspace)
    }

    fn transfer<T: DataTypeLike>(
        &self,
        kind: TransferKind,
        source: &DeviceMemory<T>,
        destination: &mut DeviceMemory<T>,
        workspace: &mut WorkspaceMemory<'comm>,
    ) -> Result<()> {
        let Some(Signature::Transfer {
            kind: expected,
            source: src,
            destination: dst,
        }) = &self.signature
        else {
            return Err(operation_mismatch("transfer"));
        };
        if *expected != kind {
            return Err(operation_mismatch("matching transfer"));
        }
        src.validate(source, "source")?;
        dst.validate(destination, "destination")?;
        workspace.prepare(&self.context, self.workspace()?)?;
        let _lock = self.context.lock_execution();
        self.context.bind()?;
        // SAFETY: `cutensorMpRedistribute`, `cutensorMpUnfold`, `cutensorMpFold` receive live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            match kind {
                TransferKind::Redistribute => try_ffi!(sys::cutensorMpRedistribute(
                    self.context.as_raw(),
                    self.handle,
                    destination.as_mut_ptr().cast(),
                    source.as_ptr().cast(),
                    workspace.device_mut_ptr().cast(),
                    workspace.host_mut_ptr().cast()
                ))?,
                TransferKind::Unfold => try_ffi!(sys::cutensorMpUnfold(
                    self.context.as_raw(),
                    self.handle,
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    workspace.device_mut_ptr().cast(),
                    workspace.host_mut_ptr().cast()
                ))?,
                TransferKind::Fold => try_ffi!(sys::cutensorMpFold(
                    self.context.as_raw(),
                    self.handle,
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    workspace.device_mut_ptr().cast(),
                    workspace.host_mut_ptr().cast()
                ))?,
            }
        }
        Ok(())
    }
}
