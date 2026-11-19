mod workspace;

use std::{ptr, ptr::null_mut, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext, error::Error as CudaError, types::DevicePtr,
    view::DeviceOutputView,
};

pub(crate) use self::workspace::WorkspaceInitialization;
use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::BackendTensorUid,
    },
    error::{Error, Result},
    tensor::Shape,
};

/// Owns a finalized native binding descriptor, runtime-layout metadata, and retained allocation context.
#[derive(Debug)]
pub struct VariantPack {
    descriptor: BackendDescriptor,
    storage: VariantPackStorage,
    cuda_context: Arc<CudaContext>,
    workspace_capacity: usize,
    workspace_prefix_bytes: usize,
    workspace_base: *mut u8,
    initializations: Vec<WorkspaceInitialization>,
}

/// Checked execution-time layout for one native tensor.
/// The plan still determines supported ranks, bounds and storage requirements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantPackOverride {
    /// Native identity of the tensor whose execution-time layout is replaced.
    pub id: BackendTensorUid,
    /// Checked dimensions and element strides used for this execution.
    pub shape: Shape,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct StdVectorHeader<T> {
    begin: *mut T,
    end: *mut T,
    end_cap: *mut T,
}

impl<T> StdVectorHeader<T> {
    /// Borrows read-only attribute data without transferring ownership to C++.
    fn from_slice(values: &[T]) -> Self {
        let begin = values.as_ptr().cast_mut();
        let end = begin.wrapping_add(values.len());
        Self {
            begin,
            end,
            end_cap: end,
        }
    }
}

#[derive(Debug, Default)]
struct VariantPackStorage {
    unique_ids: Vec<BackendTensorUid>,
    data_pointers: Vec<*mut ()>,
    overrides: Vec<VariantPackOverride>,
    override_shape_headers: Vec<StdVectorHeader<i64>>,
    override_stride_headers: Vec<StdVectorHeader<i64>>,
    override_shapes_root: Option<Box<StdVectorHeader<StdVectorHeader<i64>>>>,
    override_strides_root: Option<Box<StdVectorHeader<StdVectorHeader<i64>>>>,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the variant pack retains the exclusive native pointer storage and its allocation owners"
)]
// SAFETY: the pack exclusively owns its descriptor, which has no thread affinity, and its boxed override metadata, whose addresses stay stable when moved.
// The raw data pointers are CUDA addresses with no thread affinity; the type is not `Sync`.
unsafe impl Send for VariantPack {}

impl VariantPack {
    /// Creates a cuDNN variant pack from raw device pointers.
    ///
    /// # Safety
    ///
    /// Every pointer in `bindings` and the optional `workspace` allocation must remain valid and CUDA-accessible for every execution, CUDA graph population, or CUDA graph update that uses the returned variant pack.
    /// Pointers must match the tensor descriptors and aliases expected by the execution plan and belong to `allocation_context`.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation context cannot be bound, workspace belongs to another context, or native descriptor creation, attribute assignment, or finalization fails.
    /// Runtime-override construction also reports missing internal override metadata.
    pub unsafe fn create(
        allocation_context: &Arc<CudaContext>,
        bindings: &[(BackendTensorUid, DevicePtr)],
        workspace: Option<DeviceOutputView<'_, u8>>,
    ) -> Result<Self> {
        // SAFETY: the caller keeps `bindings` and `workspace` valid for every use of the pack, which is what `create_with_overrides` requires without overrides.
        unsafe { Self::create_with_overrides(allocation_context, bindings, workspace, &[]) }
    }

    /// Creates a cuDNN variant pack with runtime shape and stride overrides.
    ///
    /// # Safety
    ///
    /// Every pointer in `bindings` and the optional `workspace` allocation must remain valid and CUDA-accessible for every execution, CUDA graph population, or CUDA graph update that uses the returned variant pack.
    /// Pointers and overrides must match the tensor descriptors expected by the execution plan.
    /// Device pointers must belong to `allocation_context`.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation context cannot be bound, workspace belongs to another context, or native descriptor creation, attribute assignment, or finalization fails.
    /// Runtime-override construction also reports missing internal override metadata.
    pub unsafe fn create_with_overrides(
        allocation_context: &Arc<CudaContext>,
        bindings: &[(BackendTensorUid, DevicePtr)],
        workspace: Option<DeviceOutputView<'_, u8>>,
        overrides: &[VariantPackOverride],
    ) -> Result<Self> {
        // SAFETY: the caller keeps `bindings` and `workspace` valid for every use of the pack and matches `overrides` to the plan, as `create_with_workspace_prefix` requires; a zero prefix reserves nothing.
        unsafe {
            Self::create_with_workspace_prefix(
                allocation_context,
                bindings,
                workspace,
                0,
                overrides,
            )
        }
    }

    /// Reserves a frontend scratch prefix while exposing the remaining bytes to cuDNN.
    /// The raw pointer lifetime and access contracts of `create_with_overrides` apply.
    pub(crate) unsafe fn create_with_workspace_prefix(
        allocation_context: &Arc<CudaContext>,
        bindings: &[(BackendTensorUid, DevicePtr)],
        mut workspace: Option<DeviceOutputView<'_, u8>>,
        workspace_prefix_bytes: usize,
        overrides: &[VariantPackOverride],
    ) -> Result<Self> {
        allocation_context.bind()?;
        if let Some(workspace) = workspace.as_ref()
            && workspace.context().as_ref() != allocation_context.as_ref()
        {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        let total_capacity = workspace.as_ref().map_or(0, DeviceOutputView::len);
        if total_capacity < workspace_prefix_bytes {
            return Err(Error::BufferTooSmall {
                name: "workspace".into(),
                required: workspace_prefix_bytes,
                actual: total_capacity,
            });
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::VariantPack)?;
        let mut storage = VariantPackStorage {
            unique_ids: bindings.iter().map(|(id, _)| *id).collect(),
            data_pointers: bindings
                .iter()
                .map(|(_, ptr)| ptr.as_ptr().cast::<()>())
                .collect(),
            overrides: overrides.to_vec(),
            ..VariantPackStorage::default()
        };
        let workspace_capacity = total_capacity - workspace_prefix_bytes;
        let workspace_base = workspace
            .as_mut()
            .map_or(null_mut(), DeviceOutputView::as_mut_ptr);
        let workspace_ptr = if workspace_base.is_null() {
            null_mut()
        } else {
            workspace_base.wrapping_add(workspace_prefix_bytes).cast()
        };

        // SAFETY: `workspace_ptr` is null or lies `workspace_prefix_bytes` into the checked workspace allocation.
        // The caller keeps that allocation valid for every use of the pack, per `# Safety`.
        unsafe {
            descriptor.set_attribute_void_ptr(
                BackendAttributeName::VariantPackWorkspace,
                workspace_ptr,
            )?;
        }

        if !bindings.is_empty() {
            // SAFETY: cuDNN copies the pointer array; the caller keeps every bound device pointer valid for every use of the pack, per `# Safety`.
            unsafe {
                descriptor.set_attribute_void_ptr_slice(
                    BackendAttributeName::VariantPackDataPointers,
                    &storage.data_pointers,
                )?;
            }

            let unique_ids: Vec<i64> = storage.unique_ids.iter().copied().map(Into::into).collect();
            descriptor
                .set_attribute_i64_slice(BackendAttributeName::VariantPackUniqueIds, &unique_ids)?;
        }

        if !overrides.is_empty() {
            storage.override_shape_headers = storage
                .overrides
                .iter()
                .map(|override_| StdVectorHeader::from_slice(override_.shape.dimensions()))
                .collect();
            storage.override_stride_headers = storage
                .overrides
                .iter()
                .map(|override_| StdVectorHeader::from_slice(override_.shape.strides()))
                .collect();

            let override_ids: Vec<i64> = storage
                .overrides
                .iter()
                .map(|override_| override_.id.as_i64())
                .collect();
            descriptor.set_attribute_i64_slice(
                BackendAttributeName::VariantPackOverrideUniqueIds,
                &override_ids,
            )?;

            // cuDNN expects the same payload shape as the upstream C++ frontend helper:
            // a pointer to `std::vector<std::vector<int64_t>>`, not a flat array of data pointers.
            let override_shapes_root =
                storage
                    .override_shapes_root
                    .insert(Box::new(StdVectorHeader::from_slice(
                        &storage.override_shape_headers,
                    )));
            let override_shapes_root_ptr =
                ptr::from_mut::<StdVectorHeader<StdVectorHeader<i64>>>(override_shapes_root);
            let override_strides_root =
                storage
                    .override_strides_root
                    .insert(Box::new(StdVectorHeader::from_slice(
                        &storage.override_stride_headers,
                    )));
            let override_strides_root_ptr =
                ptr::from_mut::<StdVectorHeader<StdVectorHeader<i64>>>(override_strides_root);

            // SAFETY: the root points into `storage`, a boxed vector-of-vectors header whose rows the returned pack owns for its whole life.
            unsafe {
                descriptor.set_attribute_void_ptr_payload(
                    BackendAttributeName::VariantPackOverrideShapes,
                    override_shapes_root_ptr.cast(),
                )
            }?;
            // SAFETY: the root points into `storage`, a boxed vector-of-vectors header whose rows the returned pack owns for its whole life.
            unsafe {
                descriptor.set_attribute_void_ptr_payload(
                    BackendAttributeName::VariantPackOverrideStrides,
                    override_strides_root_ptr.cast(),
                )
            }?;
        }

        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            storage,
            cuda_context: Arc::clone(allocation_context),
            workspace_capacity,
            workspace_prefix_bytes,
            workspace_base,
            initializations: Vec::new(),
        })
    }

    /// Returns the CUDA context associated with this variant pack.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }

    /// Workspace bytes available to the native backend after the frontend scratch prefix.
    #[must_use]
    pub fn workspace_capacity(&self) -> usize {
        self.workspace_capacity
    }

    /// Opaque native workspace pointer after the frontend scratch prefix, if an allocation was supplied.
    /// Access must follow the plan's workspace capacity, ordering and lifetime contracts.
    #[must_use]
    pub fn workspace_ptr(&self) -> Option<DevicePtr> {
        if self.workspace_base.is_null() {
            return None;
        }
        // SAFETY: creation checked that the prefix fits the workspace allocation, so the address is inside it; the value grants no access by itself.
        Some(unsafe {
            DevicePtr::from_raw(
                self.workspace_base
                    .wrapping_add(self.workspace_prefix_bytes)
                    .cast(),
            )
        })
    }

    /// Total caller-supplied capacity, including the frontend scratch prefix.
    #[must_use]
    pub fn total_workspace_capacity(&self) -> usize {
        self.workspace_capacity + self.workspace_prefix_bytes
    }

    pub(crate) fn workspace_prefix_bytes(&self) -> usize {
        self.workspace_prefix_bytes
    }

    /// Returns whether the descriptor retains any execution-time shape or stride overrides.
    #[must_use]
    pub fn has_overrides(&self) -> bool {
        !self.storage.overrides.is_empty()
    }

    /// Native tensor identities and opaque pointers retained by this variant pack.
    /// Inspecting these addresses does not grant permission to access their pointees.
    pub fn bindings(&self) -> impl ExactSizeIterator<Item = (BackendTensorUid, DevicePtr)> + '_ {
        self.storage
            .unique_ids
            .iter()
            .copied()
            .zip(&self.storage.data_pointers)
            // SAFETY: each address came from a `DevicePtr` passed to the unsafe constructor, and the value grants no access by itself.
            .map(|(id, &pointer)| (id, unsafe { DevicePtr::from_raw(pointer) }))
    }

    /// Checked runtime layout for one native tensor, when overridden.
    #[must_use]
    pub fn override_for(&self, id: BackendTensorUid) -> Option<&Shape> {
        self.storage
            .overrides
            .iter()
            .find(|override_| override_.id == id)
            .map(|override_| &override_.shape)
    }

    /// Borrows the finalized native variant-pack descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
