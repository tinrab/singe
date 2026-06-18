use std::collections::{BTreeMap, HashSet};

use crate::{
    frontend::{
        graph::TensorRecord,
        plan::{AliasTensorBinding, BindingReplacement, RequiredTensor},
    },
    tensor::TensorId,
};

pub(super) fn required_tensor_bindings(
    tensors: &BTreeMap<TensorId, TensorRecord>,
    binding_replacements: &[BindingReplacement],
) -> Vec<RequiredTensor> {
    let alias_targets = binding_replacements
        .iter()
        .map(|replacement| replacement.target_id)
        .collect::<HashSet<_>>();
    tensors
        .iter()
        .filter(|(_, record)| {
            !record.tensor.is_virtual
                && !record.tensor.is_by_value
                && !alias_targets.contains(&record.id)
        })
        .map(|(tensor_id, record)| RequiredTensor {
            id: *tensor_id,
            backend_uid: record.id.into(),
            name: record.tensor.name.clone(),
            data_type: record.tensor.data_type,
            dimensions: record.tensor.shape.dimensions().to_vec(),
            strides: record.tensor.shape.strides().to_vec(),
        })
        .collect()
}

pub(super) fn alias_tensor_bindings(
    tensors: &BTreeMap<TensorId, TensorRecord>,
    binding_replacements: &[BindingReplacement],
) -> Vec<AliasTensorBinding> {
    binding_replacements
        .iter()
        .map(|replacement| {
            let source = tensors
                .iter()
                .find(|(_, record)| record.id == replacement.source_id);
            let target = tensors
                .iter()
                .find(|(_, record)| record.id == replacement.target_id);
            AliasTensorBinding {
                source_id: source
                    .map(|(tensor_id, _)| *tensor_id)
                    .unwrap_or(replacement.source_id),
                source_backend_uid: replacement.source_id.into(),
                target_id: target
                    .map(|(tensor_id, _)| *tensor_id)
                    .unwrap_or(replacement.target_id),
                target_backend_uid: replacement.target_id.into(),
                target_name: target.and_then(|(_, record)| record.tensor.name.clone()),
                byte_offset: replacement.byte_offset,
            }
        })
        .collect()
}
