use std::{mem::ManuallyDrop, ptr};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    data_type::DataTypeLike,
    stream::{AsStream, StreamBinding},
};

use crate::{
    error::{Error, Result},
    mg::{
        context::{Context, ContextRef, validate_same_context},
        contraction::Operand,
        memory::{DistributedDeviceMemory, WorkspaceMemory},
        tensor::TensorDescriptor,
        workspace::Workspace,
    },
    streams::{join_producers, sync_streams},
    sys, try_ffi,
    types::Mode,
    utility::modes_to_i32_vec,
};

/// Owns a multi-GPU tensor-copy description and the operand types and storage spans checked at execution.
#[derive(Debug)]
pub struct CopyDescriptor {
    handle: sys::cutensorMgCopyDescriptor_t,
    context: ContextRef,
    signature: Signature,
}

/// Owns a multi-GPU copy plan, workspace requirements, and its library context.
#[derive(Debug)]
pub struct CopyPlan {
    handle: sys::cutensorMgCopyPlan_t,
    context: ContextRef,
    signature: Signature,
    workspace: Workspace,
}

/// Transferred native copy descriptor with its operand metadata and retained multi-GPU context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`CopyDescriptor::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawCopyDescriptor {
    /// Transferred native descriptor handle; destroy it before releasing the retained context.
    pub handle: sys::cutensorMgCopyDescriptor_t,
    /// Multi-GPU context the descriptor was created in.
    pub context: ContextRef,
    // Operand metadata is wrapper state, so only `into_raw` can produce this record.
    signature: Signature,
}

/// Transferred native copy plan with its operand metadata, workspace requirements, and retained multi-GPU context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`CopyPlan::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawCopyPlan {
    /// Transferred native plan handle; destroy it before releasing the retained context.
    pub handle: sys::cutensorMgCopyPlan_t,
    /// Multi-GPU context the plan was created in.
    pub context: ContextRef,
    // Operand metadata is wrapper state, so only `into_raw` can produce this record.
    signature: Signature,
    /// Workspace byte limits the plan was created under.
    pub workspace: Workspace,
}

#[derive(Debug, Clone)]
struct Signature {
    dst: Operand,
    src: Operand,
}

impl CopyDescriptor {
    /// Describes a copy or redistribution between tensors identified by symbolic mode labels.
    ///
    /// The descriptor records each operand's data type, storage devices, and partition span for checked execution.
    ///
    /// # Errors
    ///
    /// Returns an error if descriptors belong to another multi-GPU context, mode counts differ from tensor ranks, a partition span overflows, native descriptor creation fails, or the returned handle is null.
    pub fn create(
        context: &Context,
        dst: &TensorDescriptor,
        modes_dst: &[Mode],
        src: &TensorDescriptor,
        modes_src: &[Mode],
    ) -> Result<Self> {
        validate_same_context(&context.as_context_ref(), dst.context(), "dst")?;
        validate_same_context(&context.as_context_ref(), src.context(), "src")?;
        validate_modes(dst, modes_dst)?;
        validate_modes(src, modes_src)?;
        let signature = Signature {
            dst: Operand::create(dst)?,
            src: Operand::create(src)?,
        };

        let modes_dst = modes_to_i32_vec(modes_dst);
        let modes_src = modes_to_i32_vec(modes_src);
        let mut handle = ptr::null_mut();
        // SAFETY: `context` and both descriptors (checked to share it above) are live, each mode array has its descriptor's rank (`validate_modes`), and `handle` is a local output.
        unsafe {
            try_ffi!(sys::cutensorMgCreateCopyDescriptor(
                context.as_raw(),
                &raw mut handle,
                dst.as_raw(),
                modes_dst.as_ptr(),
                src.as_raw(),
                modes_src.as_ptr(),
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

    /// Queries per-context-device and host workspace requirements in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the native workspace query fails or reports a negative byte count.
    pub fn workspace(&self) -> Result<Workspace> {
        let mut device_workspace_size = vec![0; self.context.device_count()];
        let mut host_workspace_size = 0;
        // SAFETY: `self.handle` and `self.context` are live, `device_workspace_size` has one slot per context device, and `host_workspace_size` is a local output.
        unsafe {
            try_ffi!(sys::cutensorMgCopyGetWorkspace(
                self.context.as_raw(),
                self.handle,
                device_workspace_size.as_mut_ptr(),
                &raw mut host_workspace_size,
            ))?;
        }
        Workspace::from_raw(device_workspace_size, host_workspace_size)
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

    /// Returns the borrowed native Copy descriptor handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorMgCopyDescriptor_t {
        self.handle
    }

    /// Consumes this descriptor and returns the native handle without destroying it.
    ///
    /// The returned record retains the multi-GPU context and operand metadata.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawCopyDescriptor {
        let owner = ManuallyDrop::new(self);
        // SAFETY: `owner` is never dropped, so each field is read out exactly once.
        unsafe {
            RawCopyDescriptor {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
                signature: ptr::read(&raw const owner.signature),
            }
        }
    }

    /// Takes ownership of a native cuTENSORMg copy descriptor with its operand metadata and multi-GPU context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid copy descriptor created in `raw.context` for the record's operand metadata, and uniquely owned.
    /// The handle must not be destroyed elsewhere afterward.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawCopyDescriptor) -> Result<Self> {
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

impl CopyPlan {
    /// Creates a copy plan under the supplied workspace byte limits.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor belongs to another multi-GPU context, the device workspace count is invalid, native plan creation fails, or the returned handle is null.
    pub fn create(context: &Context, desc: &CopyDescriptor, workspace: Workspace) -> Result<Self> {
        validate_same_context(&context.as_context_ref(), desc.context(), "desc")?;
        workspace.validate_for_context(desc.context())?;

        let mut handle = ptr::null_mut();
        // SAFETY: `context` and `desc` (checked to share it above) are live, `workspace` has one size per context device (`validate_for_context`), and `handle` is a local output.
        unsafe {
            try_ffi!(sys::cutensorMgCreateCopyPlan(
                context.as_raw(),
                &raw mut handle,
                desc.as_raw(),
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

    /// Returns the workspace byte limits retained when creating this plan.
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

    /// Enqueues a copy from `src` into `dst` over distributed device allocations.
    ///
    /// Each operand's element type must be the data type of its tensor descriptor.
    /// Each operand's partitions must lie on its descriptor's devices, in descriptor order, and each partition must cover the storage span the descriptor's layout reaches on one device.
    /// Device workspace and `streams` follow context device order; [`Context::default_streams`] gives the legacy default streams.
    ///
    /// # Safety
    ///
    /// Follow the crate's [raw execution contract](crate#raw-execution), for every stream in `streams`.
    ///
    /// # Errors
    ///
    /// Returns an error if an element type differs from its descriptor's data type, an operand's partitions lie on other devices than its descriptor or in another context, a partition is shorter than its descriptor's span, a layout sets only one of its element and block strides where the other matters, workspace or stream counts do not match context devices, a stream belongs to another context, workspace capacity is insufficient, or the native copy fails.
    pub unsafe fn copy_async<TD: DataTypeLike, TS: DataTypeLike>(
        &self,
        dst: &mut DistributedDeviceMemory<TD>,
        src: &DistributedDeviceMemory<TS>,
        workspace: &mut WorkspaceMemory,
        streams: &[StreamBinding],
    ) -> Result<()> {
        self.signature.dst.validate(dst, "dst", &self.context)?;
        self.signature.src.validate(src, "src", &self.context)?;
        workspace.validate(&self.context, &self.workspace)?;
        validate_count(
            "device_workspace",
            self.context.device_count(),
            workspace.device_pointer_count(),
        )?;
        validate_streams(&self.context, streams)?;
        join_producers(
            streams,
            &[dst.regions(), src.regions(), workspace.device_regions()].concat(),
        )?;

        let mut dst = dst.mut_ptrs();
        let mut src = src.const_ptrs();
        let mut device_workspace = workspace.device_mut_ptrs();
        let host_workspace = workspace.host_mut_ptr();
        let mut streams: Vec<_> = streams.iter().map(AsStream::as_raw).collect();
        // SAFETY: the operands, workspace, and streams were checked against the plan above, and the caller keeps them alive through completion.
        unsafe {
            try_ffi!(sys::cutensorMgCopy(
                self.context.as_raw(),
                self.handle,
                dst.as_mut_ptr().cast(),
                src.as_mut_ptr().cast(),
                device_workspace.as_mut_ptr().cast(),
                host_workspace.cast(),
                streams.as_mut_ptr().cast(),
            ))
        }
    }

    /// Copies `src` into `dst` over distributed device allocations and waits for every stream.
    ///
    /// The operands follow the rules of [`Self::copy_async`].
    ///
    /// # Errors
    ///
    /// Returns an error if a stream is capturing, the checks of [`Self::copy_async`] fail, the native copy fails, or a stream reports an error.
    pub fn copy<TD: DataTypeLike, TS: DataTypeLike>(
        &self,
        dst: &mut DistributedDeviceMemory<TD>,
        src: &DistributedDeviceMemory<TS>,
        workspace: &mut WorkspaceMemory,
        streams: &[StreamBinding],
    ) -> Result<()> {
        sync_streams(streams, || {
            // SAFETY: both operands and the workspace stay borrowed until `sync_streams` has drained every stream, and the output borrow is exclusive.
            unsafe { self.copy_async(dst, src, workspace, streams) }
        })
    }

    /// Returns the borrowed native Copy plan handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorMgCopyPlan_t {
        self.handle
    }

    /// Consumes this plan and returns the native handle without destroying it.
    ///
    /// The returned record retains the multi-GPU context, operand metadata, and workspace requirements.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawCopyPlan {
        let owner = ManuallyDrop::new(self);
        // SAFETY: `owner` is never dropped, so each field is read out exactly once.
        unsafe {
            RawCopyPlan {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
                signature: ptr::read(&raw const owner.signature),
                workspace: ptr::read(&raw const owner.workspace),
            }
        }
    }

    /// Takes ownership of a native cuTENSORMg copy plan with its operand metadata, workspace limits, and multi-GPU context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid copy plan created in `raw.context` under `raw.workspace`, for the record's operand metadata, and uniquely owned.
    /// The handle must not be destroyed elsewhere afterward.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null or the device workspace count differs from the context device count.
    pub unsafe fn from_raw(raw: RawCopyPlan) -> Result<Self> {
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

impl Drop for CopyDescriptor {
    fn drop(&mut self) {
        // SAFETY: the descriptor exclusively owns its non-null handle, and `self.context` keeps the multi-GPU handle live.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroyCopyDescriptor(self.handle)) {
                report_cleanup_failure("destroy", "cutensormg copy descriptor", &error);
            }
        }
    }
}

impl Drop for CopyPlan {
    fn drop(&mut self) {
        // SAFETY: the plan exclusively owns its non-null handle, and `self.context` keeps the multi-GPU handle live.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroyCopyPlan(self.handle)) {
                report_cleanup_failure("destroy", "cutensormg copy plan", &error);
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

pub(crate) fn validate_count(name: &str, expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(Error::LengthMismatch {
            name: name.to_owned().into(),
            expected,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn validate_streams(context: &ContextRef, streams: &[StreamBinding]) -> Result<()> {
    validate_count("streams", context.device_count(), streams.len())?;
    for (stream, &device) in streams.iter().zip(context.devices()) {
        if stream.context() != context.cuda_context(device)? {
            return Err(Error::StreamContextMismatch);
        }
    }
    Ok(())
}
