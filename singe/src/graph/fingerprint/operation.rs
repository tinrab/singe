use std::{collections::HashMap, hash::Hash};

use crate::{
    backend::registry::{BitHash, OperationAttributeKind},
    error::{Error, Result},
    graph::{
        fingerprint::{
            stable::StableHasher,
            values::{hash_value_id, hash_value_ids},
        },
        ir::{OperationNode, ValueId},
    },
};

pub fn hash_ir_operation(
    operation: &OperationNode,
    local_values: &HashMap<ValueId, usize>,
    hasher: &mut StableHasher,
) -> Result<()> {
    if operation.opcode().is_some() {
        let schema = operation
            .schema()
            .ok_or_else(|| Error::missing_node("hash ir operation"))?;
        schema.name().hash(hasher);
    } else {
        match operation.attributes().kind() {
            OperationAttributeKind::Input => "input".hash(hasher),
            OperationAttributeKind::Constant => "constant".hash(hasher),
            _ => return Err(Error::missing_node("hash ir operation")),
        }
    }

    hash_value_ids(operation.inputs(), local_values, hasher)?;
    operation.attributes().bit_hash(hasher);
    hash_operation_results(operation, local_values, hasher)?;
    Ok(())
}

fn hash_operation_results(
    operation: &OperationNode,
    local_values: &HashMap<ValueId, usize>,
    hasher: &mut StableHasher,
) -> Result<()> {
    operation.results().len().hash(hasher);
    for result in operation.results() {
        result.index().hash(hasher);
        result.role().hash(hasher);
        hash_value_id(result.value(), local_values, hasher)?;
    }
    Ok(())
}
