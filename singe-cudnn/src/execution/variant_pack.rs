use singe_cuda::{memory::DeviceMemory, types::DevicePtr};

use crate::{
    attribute::BackendAttributeName,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::{Error, Result},
    tensor::TensorId,
    version,
};

#[derive(Debug)]
pub struct VariantPack {
    descriptor: BackendDescriptor,
    _storage: VariantPackStorage,
}

#[derive(Debug, Clone)]
pub struct VariantPackOverride {
    pub id: TensorId,
    pub dimensions: Vec<i64>,
    pub strides: Vec<i64>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct StdVectorHeader<T> {
    begin: *mut T,
    end: *mut T,
    end_cap: *mut T,
}

impl<T> StdVectorHeader<T> {
    fn from_vec(values: &mut Vec<T>) -> Self {
        let begin = values.as_mut_ptr();
        let end = begin.wrapping_add(values.len());
        let end_cap = begin.wrapping_add(values.capacity());
        Self {
            begin,
            end,
            end_cap,
        }
    }
}

#[derive(Debug, Default)]
struct VariantPackStorage {
    unique_ids: Vec<TensorId>,
    data_pointers: Vec<*mut ()>,
    workspace_fallback: Option<DeviceMemory<u8>>,
    override_ids: Vec<TensorId>,
    override_dimensions: Vec<Vec<i64>>,
    override_strides: Vec<Vec<i64>>,
    override_shape_headers: Vec<StdVectorHeader<i64>>,
    override_stride_headers: Vec<StdVectorHeader<i64>>,
    override_shapes_root: Option<StdVectorHeader<StdVectorHeader<i64>>>,
    override_strides_root: Option<StdVectorHeader<StdVectorHeader<i64>>>,
}

impl VariantPack {
    /// Creates a cuDNN variant pack from raw device pointers.
    ///
    /// # Safety
    ///
    /// Every pointer in `bindings` and the optional `workspace` allocation must
    /// remain valid and CUDA-accessible for every execution, CUDA graph
    /// population, or CUDA graph update that uses the returned variant pack.
    /// Pointers must match the tensor descriptors and aliases expected by the
    /// execution plan.
    pub unsafe fn create(
        bindings: &[(TensorId, DevicePtr)],
        workspace: Option<&mut DeviceMemory<u8>>,
    ) -> Result<Self> {
        unsafe { Self::create_with_overrides(bindings, workspace, &[]) }
    }

    /// Creates a cuDNN variant pack with runtime shape and stride overrides.
    ///
    /// # Safety
    ///
    /// Every pointer in `bindings` and the optional `workspace` allocation must
    /// remain valid and CUDA-accessible for every execution, CUDA graph
    /// population, or CUDA graph update that uses the returned variant pack.
    /// Pointers and overrides must match the tensor descriptors expected by the
    /// execution plan.
    pub unsafe fn create_with_overrides(
        bindings: &[(TensorId, DevicePtr)],
        workspace: Option<&mut DeviceMemory<u8>>,
        overrides: &[VariantPackOverride],
    ) -> Result<Self> {
        Self::create_with_overrides_for_version(bindings, workspace, overrides, version()?.raw())
    }

    fn create_with_overrides_for_version(
        bindings: &[(TensorId, DevicePtr)],
        workspace: Option<&mut DeviceMemory<u8>>,
        overrides: &[VariantPackOverride],
        cudnn_version: u64,
    ) -> Result<Self> {
        if bindings.is_empty() {
            return Err(Error::EmptyList {
                name: "bindings".into(),
            });
        }
        if !overrides.is_empty() && cudnn_version < 92100 {
            return Err(Error::FrontendFeatureRequiresVersion {
                feature: "override shape".into(),
                min_version: "9.21".into(),
                actual_version: cudnn_version,
            });
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::VariantPack)?;
        let mut storage = VariantPackStorage {
            unique_ids: bindings.iter().map(|(id, _)| *id).collect(),
            data_pointers: bindings
                .iter()
                .map(|(_, ptr)| ptr.as_ptr().cast::<()>())
                .collect(),
            ..VariantPackStorage::default()
        };
        let workspace_ptr = match workspace {
            Some(workspace) if !workspace.as_mut_ptr().is_null() => workspace.as_mut_ptr().cast(),
            _ => {
                let fallback = DeviceMemory::<u8>::create(1)?;
                let ptr = fallback.as_mut_ptr().cast();
                storage.workspace_fallback = Some(fallback);
                ptr
            }
        };

        descriptor
            .set_attribute_void_ptr(BackendAttributeName::VariantPackWorkspace, workspace_ptr)?;
        descriptor.set_attribute_void_ptr_slice(
            BackendAttributeName::VariantPackDataPointers,
            &storage.data_pointers,
        )?;

        let unique_ids: Vec<i64> = storage.unique_ids.iter().copied().map(Into::into).collect();
        descriptor
            .set_attribute_i64_slice(BackendAttributeName::VariantPackUniqueIds, &unique_ids)?;
        if !overrides.is_empty() {
            storage.override_ids = overrides.iter().map(|override_| override_.id).collect();
            storage.override_dimensions = overrides
                .iter()
                .map(|override_| override_.dimensions.clone())
                .collect();
            storage.override_strides = overrides
                .iter()
                .map(|override_| override_.strides.clone())
                .collect();
            storage.override_shape_headers = storage
                .override_dimensions
                .iter_mut()
                .map(StdVectorHeader::from_vec)
                .collect();
            storage.override_stride_headers = storage
                .override_strides
                .iter_mut()
                .map(StdVectorHeader::from_vec)
                .collect();

            let override_ids: Vec<i64> = storage
                .override_ids
                .iter()
                .copied()
                .map(Into::into)
                .collect();
            descriptor.set_attribute_i64_slice(
                BackendAttributeName::VariantPackOverrideUniqueIds,
                &override_ids,
            )?;
            storage.override_shapes_root = Some(StdVectorHeader::from_vec(
                &mut storage.override_shape_headers,
            ));
            storage.override_strides_root = Some(StdVectorHeader::from_vec(
                &mut storage.override_stride_headers,
            ));

            // cuDNN expects the same payload shape as the upstream C++ frontend helper:
            // a pointer to `std::vector<std::vector<int64_t>>`, not a flat array of data pointers.
            let override_shapes_root_ptr = storage.override_shapes_root.as_mut().ok_or(
                Error::FrontendInternalStateMissing {
                    name: "override_shapes_root",
                },
            )?
                as *mut StdVectorHeader<StdVectorHeader<i64>>;
            let override_strides_root_ptr = storage.override_strides_root.as_mut().ok_or(
                Error::FrontendInternalStateMissing {
                    name: "override_strides_root",
                },
            )?
                as *mut StdVectorHeader<StdVectorHeader<i64>>;

            descriptor.set_attribute_void_ptr_payload(
                BackendAttributeName::VariantPackOverrideShapes,
                override_shapes_root_ptr.cast(),
            )?;
            descriptor.set_attribute_void_ptr_payload(
                BackendAttributeName::VariantPackOverrideStrides,
                override_strides_root_ptr.cast(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            _storage: storage,
        })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
