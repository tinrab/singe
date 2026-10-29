use singe_cuda::types::DevicePtr;

use crate::{
    backend::tensor::TensorId,
    error::Result,
    frontend::{
        bindings::{Bindings, adjusted_storage_view_ptr, resolve_binding_source},
        error::Error as FrontendError,
        plan::BindingReplacement,
        workspace::WorkspaceLayout,
    },
};

/// Root allocation and accumulated byte offset for one prepared native binding.
#[derive(Debug, Clone, Copy)]
pub struct BindingSource {
    pub tensor: TensorId,
    pub byte_offset: i64,
}

/// Immutable native binding order and corresponding resolved storage sources.
/// Construction resolves alias chains once; pointer preparation preserves this order.
#[derive(Debug)]
pub struct BindingTemplate {
    ids: Vec<TensorId>,
    sources: Vec<BindingSource>,
}

impl BindingTemplate {
    pub(crate) fn create(
        mut ids: Vec<TensorId>,
        replacements: &[BindingReplacement],
    ) -> Result<Self> {
        ids.sort_unstable();
        let sources = ids
            .iter()
            .map(|&id| {
                let (tensor, byte_offset) = resolve_binding_source(replacements, id)?;
                Ok(BindingSource {
                    tensor,
                    byte_offset,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { ids, sources })
    }

    pub(crate) fn ids(&self) -> &[TensorId] {
        &self.ids
    }

    /// Resolved sources in exactly the same order as identities and materialized pointers.
    pub(crate) fn sources(&self) -> impl ExactSizeIterator<Item = BindingSource> + '_ {
        self.sources.iter().copied()
    }

    pub(crate) fn materialize(
        &self,
        bindings: &Bindings<'_>,
        workspace: Option<(DevicePtr, &WorkspaceLayout)>,
    ) -> Result<Vec<(TensorId, DevicePtr)>> {
        self.ids
            .iter()
            .zip(&self.sources)
            .map(|(&id, source)| {
                let automatic = match workspace {
                    Some((base, layout)) => layout.binding_pointer(source.tensor, base)?,
                    None => None,
                };
                let pointer = automatic
                    .or_else(|| bindings.map.get(&source.tensor).copied())
                    .ok_or_else(|| {
                        if source.tensor == id {
                            FrontendError::TensorNotBound(id)
                        } else {
                            FrontendError::BindingReplacementSourceNotBound {
                                source_id: source.tensor,
                                target_id: id,
                                byte_offset: source.byte_offset,
                            }
                        }
                    })?;
                Ok((id, adjusted_storage_view_ptr(pointer, source.byte_offset)?))
            })
            .collect()
    }
}
