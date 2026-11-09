use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use crate::{
    error::{Error, Result},
    graph::{
        fingerprint::{
            shape::{hash_device, hash_layout_with_symbols, hash_shape, hash_shape_with_symbols},
            stable::StableHasher,
        },
        ir::{Value, ValueBinding, ValueBindingKind, ValueId},
        symbolic::SymbolicDimension,
    },
    tensor::Tensor,
};

pub fn hash_ir_value(
    value: &Value,
    symbols: Option<&[Option<SymbolicDimension>]>,
    hasher: &mut StableHasher,
) {
    value.name().hash(hasher);
    value.data_type().hash(hasher);
    hash_shape_with_symbols(value.shape(), symbols, hasher);
    hash_device(value.device(), hasher);
    hash_layout_with_symbols(value.layout(), symbols, hasher);
    hash_value_binding(value.binding(), hasher);
}

pub fn hash_value_id(
    id: ValueId,
    local_ids: &HashMap<ValueId, usize>,
    hasher: &mut impl Hasher,
) -> Result<()> {
    local_ids
        .get(&id)
        .copied()
        .ok_or_else(|| Error::missing_node("hash value id"))?
        .hash(hasher);
    Ok(())
}

pub fn hash_value_ids(
    ids: &[ValueId],
    local_ids: &HashMap<ValueId, usize>,
    hasher: &mut impl Hasher,
) -> Result<()> {
    ids.len().hash(hasher);
    for id in ids {
        hash_value_id(*id, local_ids, hasher)?;
    }
    Ok(())
}

fn hash_value_binding(binding: &ValueBinding, hasher: &mut StableHasher) {
    match binding.kind() {
        ValueBindingKind::RuntimeInput => {
            "runtime_input".hash(hasher);
            binding.input_name().hash(hasher);
        }
        ValueBindingKind::InitializerInput => {
            "initializer_input".hash(hasher);
            binding.input_name().hash(hasher);
            binding.runtime_overridable().hash(hasher);
        }
        ValueBindingKind::StaticInput => {
            "static_input".hash(hasher);
            binding.input_name().hash(hasher);
        }
        ValueBindingKind::Constant => "constant".hash(hasher),
        ValueBindingKind::OperationOutput => {
            "operation_output".hash(hasher);
            let (operation, output_index) = binding
                .operation_output()
                .expect("operation output binding must carry producer metadata");
            operation.index().hash(hasher);
            output_index.hash(hasher);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ConstantFingerprint {
    HostContent { digest: u64 },
    GraphLocal { local_node: usize },
}

pub fn hash_static_initializer(initializer: &Tensor, hasher: &mut impl Hasher) {
    initializer.storage().storage_identity().hash(hasher);
    initializer.storage().data_pointer_identity().hash(hasher);
    initializer.storage_offset().hash(hasher);
    initializer.storage().byte_size().hash(hasher);
    initializer.version().hash(hasher);
}

pub fn hash_constant_tensor(
    local_node: usize,
    tensor: &Tensor,
    hasher: &mut impl Hasher,
) -> Result<bool> {
    let fingerprint = constant_fingerprint(local_node, tensor)?;
    let cacheable = matches!(fingerprint, ConstantFingerprint::HostContent { .. });
    fingerprint.hash(hasher);
    Ok(cacheable)
}

fn constant_fingerprint(local_node: usize, tensor: &Tensor) -> Result<ConstantFingerprint> {
    let byte_size = tensor.byte_size()?;
    if tensor.device().is_cpu()
        && tensor.is_contiguous()
        && tensor.storage_offset() == 0
        && tensor.storage().byte_size() == byte_size
    {
        let mut bytes = vec![0_u8; byte_size];
        tensor.storage().copy_to_host(&mut bytes)?;
        let mut hasher = StableHasher::new();
        tensor.data_type().hash(&mut hasher);
        hash_shape(tensor.shape(), &mut hasher);
        bytes.hash(&mut hasher);
        return Ok(ConstantFingerprint::HostContent {
            digest: hasher.finish(),
        });
    }

    Ok(ConstantFingerprint::GraphLocal { local_node })
}
