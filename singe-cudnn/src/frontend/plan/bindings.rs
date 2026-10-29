use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    backend::tensor::{TensorId, TensorSpec},
    error::{Error, Result},
    frontend::{
        bindings::resolve_binding_source, error::Error as FrontendError, graph::TensorRecord,
        operation::TensorAccess,
    },
};

pub fn required_tensor_bindings(
    tensors: &BTreeMap<TensorId, TensorRecord>,
    binding_replacements: &[BindingReplacement],
    binding_ids: &[TensorId],
) -> Result<Vec<RequiredTensor>> {
    let mut required = BTreeMap::<TensorId, RequiredTensor>::new();
    for &id in binding_ids {
        let operand = tensors.get(&id).ok_or(FrontendError::TensorNotFound(id))?;
        if operand.tensor.is_virtual()
            || operand.tensor.is_by_value()
            || operand.tensor.is_workspace()
        {
            continue;
        }
        let access = operand.access.ok_or(Error::DescriptorMismatch {
            name: "execution binding access".into(),
        })?;
        let (source_id, byte_offset) = resolve_binding_source(binding_replacements, id)?;
        let source = tensors
            .get(&source_id)
            .ok_or(FrontendError::TensorNotFound(source_id))?;
        if source.tensor.is_virtual() || source.tensor.is_by_value() || source.tensor.is_workspace()
        {
            return Err(Error::DescriptorMismatch {
                name: "caller binding storage".into(),
            });
        }
        let binding = required.entry(source_id).or_insert_with(|| RequiredTensor {
            id: source_id,
            bindings: Vec::new(),
            access,
            reads_initial_state: false,
            indexes_memory: false,
            spec: source.tensor.clone(),
        });
        binding.access = binding.access.merge(access);
        binding.reads_initial_state |= operand.reads_initial_state;
        binding.indexes_memory |= operand.indexes_memory;
        binding.bindings.push(RequiredTensorBinding {
            tensor_id: id,
            byte_offset,
            access,
            reads_initial_state: operand.reads_initial_state,
            indexes_memory: operand.indexes_memory,
        });
    }
    Ok(required.into_values().collect())
}

pub fn alias_tensor_bindings(
    tensors: &BTreeMap<TensorId, TensorRecord>,
    binding_replacements: &[BindingReplacement],
) -> Vec<AliasTensorBinding> {
    binding_replacements
        .iter()
        .map(|replacement| AliasTensorBinding {
            kind: replacement.kind,
            source_id: replacement.source_id,
            target_id: replacement.target_id,
            target_name: tensors
                .get(&replacement.target_id)
                .and_then(|record| record.tensor.name.clone()),
            byte_offset: replacement.byte_offset,
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BindingReplacement {
    pub(crate) source_id: TensorId,
    pub(crate) target_id: TensorId,
    pub(crate) byte_offset: i64,
    pub(crate) kind: AliasTensorKind,
}

/// Whether an alias has its own layout or follows a captured tensor's runtime layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AliasTensorKind {
    /// An explicit storage view keeps its declared layout unless overridden directly.
    StorageView,
    /// A score-graph capture inherits its source layout, including native index rank promotion.
    CapturedTensor,
}

/// A non-virtual tensor that must be bound by the caller before execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredTensor {
    /// Frontend graph tensor ID accepted by [`crate::frontend::bindings::Bindings::set`].
    pub id: TensorId,
    /// Execution operands reached through this caller binding, including storage views.
    pub bindings: Vec<RequiredTensorBinding>,
    /// Combined access of all execution uses of the caller's allocation.
    pub access: TensorAccess,
    /// Whether any execution use may consume values supplied before graph execution.
    /// Conditional writers and independent alias layouts are treated conservatively.
    pub reads_initial_state: bool,
    /// Whether any execution use supplies offsets, lengths, page tables or other addressing metadata.
    pub indexes_memory: bool,
    /// Complete caller-root declaration, including encoding, checked layout and storage constraints.
    /// Individual execution operands retain their own layouts and offsets through graph introspection.
    pub spec: TensorSpec,
}

/// One execution operand supplied by a caller binding or one of its storage views.
/// Native tensor identities are available through [`crate::frontend::plan::TensorInfo`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredTensorBinding {
    /// Frontend identity of the execution operand reached through this caller binding.
    pub tensor_id: TensorId,
    /// Byte offset from the required caller binding, including the complete storage-view chain.
    pub byte_offset: i64,
    /// Read/write access required by this individual execution operand.
    pub access: TensorAccess,
    /// Whether this operand may read values that the graph has not already produced.
    pub reads_initial_state: bool,
    /// Whether operand values control addresses or memory extents, such as offsets or sequence lengths.
    pub indexes_memory: bool,
}

/// One declared storage-view edge, including views used only as storage for another view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliasTensorBinding {
    /// Whether runtime layouts follow the source or belong to an independent storage view.
    pub kind: AliasTensorKind,
    /// Immediate frontend source, which may itself be a storage view.
    pub source_id: TensorId,
    /// Frontend alias tensor ID.
    pub target_id: TensorId,
    /// Optional alias tensor name.
    pub target_name: Option<String>,
    /// Byte offset added to the source pointer for the alias.
    pub byte_offset: i64,
}
