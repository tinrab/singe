use std::{
    collections::{HashMap, HashSet},
    mem,
};

use crate::{
    error::{Error, Result},
    graph::ir::{
        Block, Graph, OperationId, OperationNode, OperationResult, Partition, Value, ValueId,
    },
};

/// The new id of each value an IR rewrite kept, indexed by the value's old id.
#[derive(Debug, Clone)]
pub struct ValueRemap(Vec<Option<ValueId>>);

impl ValueRemap {
    /// Returns the new id of `value`, or `None` when the rewrite removed it.
    #[must_use]
    pub fn get(&self, value: ValueId) -> Option<ValueId> {
        self.0.get(value.index()).copied().flatten()
    }
}

impl Graph {
    /// Removes the operations no graph output depends on, with their values.
    ///
    /// Kept operations are renumbered densely in execution order, and kept values in the order their operations produce them.
    ///
    /// # Errors
    ///
    /// Returns an error if an output or operand has no producing operation.
    pub(crate) fn prune_dead_operations(&mut self) -> Result<ValueRemap> {
        let mut live = vec![false; self.operations.len()];
        let mut pending = self.output_values().to_vec();
        pending.extend(self.shape_program.constraints.keys().copied());
        while let Some(value) = pending.pop() {
            let (operation, _) = self
                .value_results
                .get(value.index())
                .copied()
                .flatten()
                .ok_or_else(|| Error::missing_node("prune dead operations"))?;
            if !mem::replace(&mut live[operation.index()], true) {
                let operation = self.required_operation(operation)?;
                pending.extend_from_slice(operation.inputs());
                for result in operation.result_values() {
                    pending.extend(
                        self.shape_program
                            .references(result)
                            .into_iter()
                            .map(ValueId::new),
                    );
                }
            }
        }
        self.retain_operations(&live)
    }

    /// Renumbers operations densely in execution order, and values in the order their operations produce them.
    ///
    /// # Errors
    ///
    /// Returns an error if an operand has no producing operation.
    pub(crate) fn renumber_in_execution_order(&mut self) -> Result<ValueRemap> {
        self.retain_operations(&vec![true; self.operations.len()])
    }

    fn retain_operations(&mut self, live: &[bool]) -> Result<ValueRemap> {
        let order = self
            .blocks
            .first()
            .map_or_else(Vec::new, |block| block.operations().to_vec());
        let mut operation_map = HashMap::new();
        let mut value_map = vec![None; self.values.len()];
        let mut next = 0;
        for &old in order.iter().filter(|old| live[old.index()]) {
            operation_map.insert(old, OperationId::new(operation_map.len()));
            for value in self.required_operation(old)?.result_values() {
                value_map[value.index()] = Some(ValueId::new(next));
                next += 1;
            }
        }

        let remap = ValueRemap(value_map);
        let map_value = |value: ValueId| {
            remap
                .get(value)
                .ok_or_else(|| Error::missing_input("retain operations"))
        };

        let mut values = Vec::new();
        let mut constant_tensors = Vec::new();
        let mut value_results = Vec::new();
        let mut operations = Vec::with_capacity(operation_map.len());
        for &old in order.iter().filter(|old| live[old.index()]) {
            let operation = self.required_operation(old)?;
            let id = operation_map[&old];
            let mut results = Vec::with_capacity(operation.results().len());
            for result in operation.results() {
                let old_value = self.required_value(result.value())?;
                let value = map_value(result.value())?;
                let alias = old_value
                    .alias()
                    .map(|alias| Ok::<_, Error>(alias.with_source(map_value(alias.source())?)))
                    .transpose()?;
                let mut renumbered = Value::new(
                    value,
                    old_value.name().map(ToOwned::to_owned),
                    old_value.data_type(),
                    old_value.layout().clone(),
                    old_value.device(),
                    old_value.binding().clone(),
                    alias,
                );
                if old_value.binding().operation_output().is_some() {
                    renumbered.set_producer(id, result.index());
                }

                values.push(renumbered);
                constant_tensors.push(self.constant_tensors[result.value().index()].clone());
                value_results.push(Some((id, result.index())));
                results.push(OperationResult::new(result.index(), value, result.role()));
            }

            operations.push(
                OperationNode::new(
                    id,
                    operation.opcode(),
                    operation.schema().cloned(),
                    operation.attributes().clone(),
                    operation
                        .inputs()
                        .iter()
                        .map(|&input| map_value(input))
                        .collect::<Result<_>>()?,
                    operation.input_roles().to_vec(),
                    results,
                )
                .with_source_location(operation.source_location().map(ToOwned::to_owned)),
            );
        }

        let map_operations = |operations: &[OperationId]| {
            operations
                .iter()
                .filter_map(|operation| operation_map.get(operation).copied())
                .collect::<Vec<_>>()
        };
        self.blocks = self
            .blocks
            .iter()
            .map(|block| {
                Ok(Block::new(
                    block.id(),
                    map_operations(block.operations()),
                    block
                        .outputs()
                        .iter()
                        .map(|&value| map_value(value))
                        .collect::<Result<_>>()?,
                ))
            })
            .collect::<Result<_>>()?;
        self.partitions = self
            .partitions
            .iter()
            .map(|partition| {
                Partition::new(
                    partition.id(),
                    map_operations(partition.operations()),
                    partition.provider(),
                )
            })
            .collect();
        self.values = values;
        self.operations = operations;
        self.constant_tensors = constant_tensors;
        self.value_results = value_results;
        let inputs = self
            .input_values()
            .map(|(_, name)| name.to_owned())
            .collect::<HashSet<_>>();
        self.initializers.retain(|name, _| inputs.contains(name));
        self.static_initializers
            .retain(|name| inputs.contains(name));
        self.state_kinds.retain(|name, _| inputs.contains(name));
        self.input_dimension_guards
            .retain(|(name, _), _| inputs.contains(name));
        self.shape_program
            .retain(|value| remap.get(value).is_some());
        self.shape_program = self.shape_program.remap(map_value, &|value| {
            map_value(ValueId::new(value)).map(ValueId::index)
        })?;
        Ok(remap)
    }
}
