use std::collections::{HashMap, HashSet};

use crate::{
    backend::{cuda::graph_value::CudaGraphValue, registry::Operation as RegistryOperation},
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
};

pub struct CudnnSubgraphMatcher<'a> {
    ir: &'a Graph,
    positions: &'a HashMap<ValueId, usize>,
    users: &'a HashMap<ValueId, Vec<ValueId>>,
    outputs: &'a HashSet<ValueId>,
    cache: &'a HashMap<ValueId, CudaGraphValue>,
}

impl<'a> CudnnSubgraphMatcher<'a> {
    pub(crate) fn new(
        ir: &'a Graph,
        positions: &'a HashMap<ValueId, usize>,
        users: &'a HashMap<ValueId, Vec<ValueId>>,
        outputs: &'a HashSet<ValueId>,
        cache: &'a HashMap<ValueId, CudaGraphValue>,
    ) -> Self {
        Self {
            ir,
            positions,
            users,
            outputs,
            cache,
        }
    }

    pub(crate) fn is_output(&self, node: ValueId) -> Result<bool> {
        let operation = self.ir.required_operation_for_value(node)?;
        Ok(operation
            .results()
            .iter()
            .any(|result| self.outputs.contains(&result.value())))
    }

    pub(crate) fn is_ready(&self, node: ValueId) -> bool {
        self.cache.contains_key(&node)
    }

    pub(crate) fn all_ready(&self, nodes: &[ValueId]) -> bool {
        nodes.iter().all(|node| self.is_ready(*node))
    }
}

impl CudnnSubgraphMatcher<'_> {
    fn single_user(&self, node: ValueId) -> Option<ValueId> {
        let [user] = self.users.get(&node)?.as_slice() else {
            return None;
        };
        Some(*user)
    }

    fn position(&self, node: ValueId) -> Option<usize> {
        self.positions.get(&node).copied()
    }

    pub(crate) fn ordered_after(&self, before: ValueId, after: ValueId) -> Option<usize> {
        let before_index = self.position(before)?;
        let after_index = self.position(after)?;
        (before_index < after_index).then_some(after_index)
    }

    pub(crate) fn add_with_operand(&self, producer: ValueId) -> Result<Option<(ValueId, ValueId)>> {
        let Some(add) = self.single_user(producer) else {
            return Ok(None);
        };
        let operation = self.ir.required_operation_for_value(add)?;
        if operation.opcode() != Some(RegistryOperation::Add) {
            return Ok(None);
        }

        let Some(other_value) = binary_operation_other_input_value(operation.inputs(), producer)?
        else {
            return Ok(None);
        };
        Ok(Some((add, other_value)))
    }

    pub(crate) fn relu_after(&self, producer: ValueId) -> Result<Option<ValueId>> {
        self.unary_after(producer, RegistryOperation::Relu)
    }

    pub(crate) fn gelu_after(&self, producer: ValueId) -> Result<Option<ValueId>> {
        self.unary_after(producer, RegistryOperation::Gelu)
    }

    fn unary_after(
        &self,
        producer: ValueId,
        expected: RegistryOperation,
    ) -> Result<Option<ValueId>> {
        let Some(relu) = self.single_user(producer) else {
            return Ok(None);
        };
        let operation = self.ir.required_operation_for_value(relu)?;
        if operation.opcode() != Some(expected) {
            return Ok(None);
        }

        let [_input] = operation.inputs() else {
            return Err(Error::missing_input("unary after"));
        };
        let producer_value = producer;
        Ok(operation.inputs().contains(&producer_value).then_some(relu))
    }
}

fn binary_operation_other_input_value(
    inputs: &[ValueId],
    producer: ValueId,
) -> Result<Option<ValueId>> {
    let [lhs, rhs] = inputs else {
        return Err(Error::missing_input("binary operation other input value"));
    };
    if *lhs == producer {
        Ok(Some(*rhs))
    } else if *rhs == producer {
        Ok(Some(*lhs))
    } else {
        Ok(None)
    }
}
