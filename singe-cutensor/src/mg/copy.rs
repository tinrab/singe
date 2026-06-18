use std::{mem::ManuallyDrop, ptr};

use singe_cuda::stream::StreamBinding;

use crate::{
    error::{Error, Result},
    mg::{
        context::{Context, ContextRef, validate_same_context},
        tensor::TensorDescriptor,
        workspace::Workspace,
    },
    sys, try_ffi,
    types::Mode,
    utility::modes_to_i32_vec,
};

#[derive(Debug)]
pub struct CopyDescriptor {
    handle: sys::cutensorMgCopyDescriptor_t,
    context: ContextRef,
    dst_pointer_count: usize,
    src_pointer_count: usize,
}

#[derive(Debug)]
pub struct CopyPlan {
    handle: sys::cutensorMgCopyPlan_t,
    context: ContextRef,
    dst_pointer_count: usize,
    src_pointer_count: usize,
    workspace: Workspace,
}

impl CopyDescriptor {
    pub fn create(
        context: &Context,
        dst: &TensorDescriptor,
        modes_dst: &[Mode],
        src: &TensorDescriptor,
        modes_src: &[Mode],
    ) -> Result<Self> {
        validate_same_context(&context.as_context_ref(), dst.context(), "dst")?;
        validate_same_context(&context.as_context_ref(), src.context(), "src")?;
        validate_modes(dst, modes_dst, "modes_dst")?;
        validate_modes(src, modes_src, "modes_src")?;

        let modes_dst = modes_to_i32_vec(modes_dst);
        let modes_src = modes_to_i32_vec(modes_src);
        let mut handle = ptr::null_mut();
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
            dst_pointer_count: dst.devices().len(),
            src_pointer_count: src.devices().len(),
        })
    }

    /// Wraps an existing cuTENSORMg copy descriptor.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMg copy descriptor associated with
    /// `context` and created for `dst_pointer_count` destination pointers and
    /// `src_pointer_count` source pointers. The returned value takes ownership
    /// of `handle` and destroys it on drop.
    pub unsafe fn from_raw(
        handle: sys::cutensorMgCopyDescriptor_t,
        context: &Context,
        dst_pointer_count: usize,
        src_pointer_count: usize,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            dst_pointer_count,
            src_pointer_count,
        })
    }

    pub fn workspace(&self) -> Result<Workspace> {
        let mut device_workspace_size = vec![0; self.context.device_count()];
        let mut host_workspace_size = 0;
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

    pub const fn as_raw(&self) -> sys::cutensorMgCopyDescriptor_t {
        self.handle
    }

    /// Consumes this descriptor and returns the owned raw cuTENSORMg copy descriptor.
    ///
    /// The caller becomes responsible for destroying the descriptor.
    pub fn into_raw(self) -> sys::cutensorMgCopyDescriptor_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl CopyPlan {
    pub fn create(context: &Context, desc: &CopyDescriptor, workspace: Workspace) -> Result<Self> {
        validate_same_context(&context.as_context_ref(), desc.context(), "desc")?;
        workspace.validate_for_context(desc.context())?;

        let mut handle = ptr::null_mut();
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
            dst_pointer_count: desc.dst_pointer_count,
            src_pointer_count: desc.src_pointer_count,
            workspace,
        })
    }

    /// Wraps an existing cuTENSORMg copy plan.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMg copy plan associated with `context`
    /// and `workspace`. Pointer counts must match the descriptors used to
    /// create the plan. The returned value takes ownership of `handle` and
    /// destroys it on drop.
    pub unsafe fn from_raw(
        handle: sys::cutensorMgCopyPlan_t,
        context: &Context,
        dst_pointer_count: usize,
        src_pointer_count: usize,
        workspace: Workspace,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            dst_pointer_count,
            src_pointer_count,
            workspace,
        })
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// Executes this copy plan with raw distributed tensor pointers.
    ///
    /// # Safety
    ///
    /// The pointer arrays must match the tensor descriptors used to create the plan.
    /// Device workspace pointers and streams must be ordered like the MG context devices.
    pub unsafe fn copy_raw(
        &self,
        dst: &mut [*mut ()],
        src: &[*const ()],
        device_workspace: &mut [*mut ()],
        host_workspace: *mut (),
        streams: &[StreamBinding],
    ) -> Result<()> {
        self.validate_execution_slices(
            dst.len(),
            src.len(),
            device_workspace.len(),
            streams.len(),
        )?;
        let mut src = src.to_vec();
        let mut streams: Vec<_> = streams.iter().map(StreamBinding::as_raw).collect();
        unsafe {
            try_ffi!(sys::cutensorMgCopy(
                self.context.as_raw(),
                self.handle,
                dst.as_mut_ptr() as _,
                src.as_mut_ptr() as _,
                device_workspace.as_mut_ptr() as _,
                host_workspace as _,
                streams.as_mut_ptr() as _,
            ))?;
        }
        Ok(())
    }

    fn validate_execution_slices(
        &self,
        dst_count: usize,
        src_count: usize,
        device_workspace_count: usize,
        stream_count: usize,
    ) -> Result<()> {
        validate_count("dst", self.dst_pointer_count, dst_count)?;
        validate_count("src", self.src_pointer_count, src_count)?;
        validate_count(
            "device_workspace",
            self.context.device_count(),
            device_workspace_count,
        )?;
        validate_count("streams", self.context.device_count(), stream_count)
    }

    pub const fn as_raw(&self) -> sys::cutensorMgCopyPlan_t {
        self.handle
    }

    /// Consumes this plan and returns the owned raw cuTENSORMg copy plan.
    ///
    /// The caller becomes responsible for destroying the plan.
    pub fn into_raw(self) -> sys::cutensorMgCopyPlan_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl Drop for CopyDescriptor {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorMgDestroyCopyDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormg copy descriptor: {err}");
            }
        }
    }
}

impl Drop for CopyPlan {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorMgDestroyCopyPlan(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormg copy plan: {err}");
            }
        }
    }
}

fn validate_modes(desc: &TensorDescriptor, modes: &[Mode], _name: &str) -> Result<()> {
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
        return Err(Error::PointerCountMismatch {
            name: name.into(),
            expected,
            actual,
        });
    }
    Ok(())
}
