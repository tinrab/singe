use std::collections::{BTreeMap, HashSet};

use singe_cuda::types::DevicePtr;

use crate::{
    backend::{
        execution::VariantPackOverride,
        tensor::{TensorId, TensorPlacement, WorkspaceInitializer},
    },
    error::{Error, Result},
    frontend::graph::TensorRecord,
    utility::to_usize,
};

const BACKEND_WORKSPACE_ALIGNMENT: usize = 256;

/// A scratch tensor's byte range inside the frontend workspace prefix.
/// Address resolution adjusts offsets when the allocation needs alignment padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceTensorBinding {
    /// Frontend identity of the scratch tensor assigned to this range.
    pub tensor: TensorId,
    /// Byte offset from the frontend workspace allocation base, after any address-resolution padding.
    pub byte_offset: usize,
    /// Reserved physical capacity of the scratch tensor, in bytes.
    pub byte_len: usize,
    /// Required byte alignment of the scratch tensor address.
    pub byte_alignment: usize,
    /// Optional initialization applied before execution, independently of native engine workspace.
    pub initializer: Option<WorkspaceInitializer>,
}

/// Aligned frontend scratch storage preceding the native engine's workspace.
/// The caller supplies one allocation covering both regions.
#[derive(Debug, Clone, Default)]
pub struct WorkspaceLayout {
    bindings: Vec<WorkspaceTensorBinding>,
    byte_len: usize,
}

impl WorkspaceLayout {
    /// Reuses prepared scratch geometry unless concrete runtime layouts change it.
    /// The returned layout still needs alignment resolution against the actual allocation address.
    pub(crate) fn with_runtime_overrides(
        &self,
        tensors: &BTreeMap<TensorId, TensorRecord>,
        native_ids: &[TensorId],
        overrides: &[VariantPackOverride],
    ) -> Result<Self> {
        if overrides.is_empty() {
            Ok(self.clone())
        } else {
            Self::create(tensors, native_ids, overrides)
        }
    }

    pub(crate) fn create(
        tensors: &BTreeMap<TensorId, TensorRecord>,
        native_ids: &[TensorId],
        overrides: &[VariantPackOverride],
    ) -> Result<Self> {
        let native_ids = native_ids.iter().copied().collect::<HashSet<_>>();
        let mut layout = Self::default();
        let mut maximum_alignment = 1;
        for record in tensors
            .values()
            .filter(|record| native_ids.contains(&record.id))
        {
            let TensorPlacement::Workspace { byte_capacity } = record.tensor.placement else {
                continue;
            };
            let mut tensor = record.tensor.clone();
            if let Some(shape) = overrides
                .iter()
                .find(|shape| shape.id.as_i64() == record.id.as_i64())
            {
                tensor.shape = shape.shape.clone();
            }
            let required = tensor.storage_byte_len(byte_capacity)?;
            if let Some(WorkspaceInitializer::Copy(data)) = &tensor.workspace_initializer
                && data.as_bytes().len() != required
            {
                return Err(Error::LengthMismatch {
                    name: "workspace tensor data".into(),
                    expected: required,
                    actual: data.as_bytes().len(),
                });
            }
            let byte_len = required.max(byte_capacity.unwrap_or(0));
            let alignment = to_usize(tensor.byte_alignment, "workspace tensor alignment")?;
            if !alignment.is_power_of_two() {
                return Err(Error::out_of_range("workspace tensor alignment"));
            }
            maximum_alignment = maximum_alignment.max(alignment);
            let offset = layout
                .byte_len
                .checked_next_multiple_of(alignment)
                .ok_or_else(|| Error::out_of_range("workspace bytes"))?;
            layout.byte_len = offset
                .checked_add(byte_len)
                .ok_or_else(|| Error::out_of_range("workspace bytes"))?;
            layout.bindings.push(WorkspaceTensorBinding {
                tensor: record.id,
                byte_offset: offset,
                byte_len,
                byte_alignment: alignment,
                initializer: tensor.workspace_initializer.clone(),
            });
        }
        if !layout.bindings.is_empty() {
            // Ordinary CUDA allocations cover the backend alignment, but stronger tensor
            // alignments require padding whose size depends on the allocation's address.
            let padding = if maximum_alignment > BACKEND_WORKSPACE_ALIGNMENT {
                maximum_alignment - 1
            } else {
                0
            };
            layout.byte_len = layout
                .byte_len
                .checked_add(padding)
                .ok_or_else(|| Error::out_of_range("workspace bytes"))?;
            layout.byte_len = layout
                .byte_len
                .checked_next_multiple_of(BACKEND_WORKSPACE_ALIGNMENT)
                .ok_or_else(|| Error::out_of_range("workspace bytes"))?;
        }
        Ok(layout)
    }

    /// Frontend prefix allocation budget, including potential alignment padding.
    /// Address-resolved layouts report the actual prefix before the native workspace.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Scratch byte ranges in tensor-identity order, preserved during address resolution.
    #[must_use]
    pub fn bindings(&self) -> &[WorkspaceTensorBinding] {
        &self.bindings
    }

    pub(crate) fn binding_pointer(
        &self,
        tensor: TensorId,
        base: DevicePtr,
    ) -> Result<Option<DevicePtr>> {
        let Ok(index) = self
            .bindings
            .binary_search_by_key(&tensor, |binding| binding.tensor)
        else {
            return Ok(None);
        };
        let byte_offset = self.bindings[index].byte_offset;
        base.checked_byte_add(byte_offset)
            .map(Some)
            .ok_or_else(|| Error::out_of_range("workspace bytes"))
    }

    /// Resolves tensor offsets and the native workspace boundary for one allocation.
    pub(crate) fn resolve_address(mut self, base_address: usize) -> Result<Self> {
        let alignment = self.bindings.iter().fold(1_usize, |alignment, binding| {
            alignment.max(binding.byte_alignment)
        });
        let padding = base_address.wrapping_neg() & (alignment - 1);
        for binding in &mut self.bindings {
            binding.byte_offset = binding
                .byte_offset
                .checked_add(padding)
                .ok_or_else(|| Error::out_of_range("workspace bytes"))?;
        }
        let end = self.bindings.last().map_or(Ok(0_usize), |binding| {
            binding
                .byte_offset
                .checked_add(binding.byte_len)
                .ok_or_else(|| Error::out_of_range("workspace bytes"))
        })?;
        self.byte_len = end
            .checked_next_multiple_of(BACKEND_WORKSPACE_ALIGNMENT)
            .ok_or_else(|| Error::out_of_range("workspace bytes"))?;
        Ok(self)
    }

    pub(crate) fn total_byte_len(&self, native_bytes: usize) -> Result<usize> {
        self.byte_len
            .checked_add(native_bytes)
            .ok_or_else(|| Error::out_of_range("workspace bytes"))
    }
}
