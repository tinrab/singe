use std::collections::{HashMap, HashSet};

use crate::{
    backend::registry::{
        Operation, OperationAttributes, OutputRole, TopKAttributes, operation_schema,
    },
    data_type::DataType,
    error::{Error, Result},
    graph::ir::{
        Block, Graph, OperationId, OperationNode, OperationResult, Partition, Value, ValueBinding,
        ValueId,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TopKKey {
    input: ValueId,
    k: i64,
    dim: i64,
    largest: bool,
    sorted: bool,
}

#[derive(Default)]
struct TopKResults {
    values: [Option<ValueId>; 2],
    operation: Option<OperationId>,
}

impl Graph {
    /// Construction still permits individual result projections.
    /// Lower them into one schema invocation before any execution or provider analysis.
    #[expect(
        clippy::too_many_lines,
        reason = "the normalization pass keeps its ordered graph rewrites in one traversal"
    )]
    pub(crate) fn normalize_results(&mut self) -> Result<()> {
        let mut groups = Vec::<TopKResults>::new();
        let mut candidates = HashMap::<TopKKey, Vec<usize>>::new();
        let mut operation_groups = HashMap::new();
        for operation in &self.operations {
            let slot = match operation.opcode() {
                Some(Operation::TopKValues) => 0,
                Some(Operation::TopKIndices) => 1,
                _ => continue,
            };
            let OperationAttributes::TopK(TopKAttributes {
                k,
                dim,
                largest,
                sorted,
            }) = operation.attributes()
            else {
                return Err(Error::missing_node("normalize results"));
            };
            let key = TopKKey {
                input: *operation
                    .inputs()
                    .first()
                    .ok_or_else(|| Error::missing_input("normalize results"))?,
                k: *k,
                dim: *dim,
                largest: *largest,
                sorted: *sorted,
            };
            let candidates = candidates.entry(key).or_default();
            let group = candidates
                .iter()
                .copied()
                .find(|&group| groups[group].values[slot].is_none())
                .unwrap_or_else(|| {
                    let group = groups.len();
                    groups.push(TopKResults::default());
                    candidates.push(group);
                    group
                });
            groups[group].values[slot] = operation.result_values().next();
            operation_groups.insert(operation.id(), group);
        }

        if groups.is_empty() {
            return Ok(());
        }

        let old_operations = self.operations.clone();
        let mut operations = Vec::new();
        let mut remapped = HashMap::new();
        for old in &old_operations {
            let id = OperationId::new(operations.len());
            let operation = if let Some(&group) = operation_groups.get(&old.id()) {
                let group = &mut groups[group];
                if let Some(existing) = group.operation {
                    remapped.insert(old.id(), existing);
                    continue;
                }

                group.operation = Some(id);
                let template = self
                    .required_value(
                        old.result_values()
                            .next()
                            .ok_or_else(|| Error::missing_output("normalize results"))?,
                    )?
                    .clone();
                let input_dtype = self.required_value_data_type(
                    *old.inputs()
                        .first()
                        .ok_or_else(|| Error::missing_input("normalize results"))?,
                )?;
                for (index, slot) in group.values.iter_mut().enumerate() {
                    if slot.is_some() {
                        continue;
                    }

                    let value = ValueId::new(self.values.len());
                    self.values.push(Value::new(
                        value,
                        None,
                        if index == 0 {
                            input_dtype
                        } else {
                            DataType::I64
                        },
                        template.layout().clone(),
                        template.device(),
                        ValueBinding::produced_by_operation(id, index),
                        None,
                    ));
                    self.constant_tensors.push(None);
                    self.value_results.push(Some((id, index)));
                    *slot = Some(value);
                }

                let schema = operation_schema(Operation::TopK);
                OperationNode::new(
                    id,
                    Some(Operation::TopK),
                    Some(schema),
                    old.attributes().clone(),
                    old.inputs().to_vec(),
                    old.input_roles().to_vec(),
                    vec![
                        OperationResult::new(
                            0,
                            group.values[0]
                                .ok_or_else(|| Error::missing_output("normalize results"))?,
                            OutputRole::Output,
                        ),
                        OperationResult::new(
                            1,
                            group.values[1]
                                .ok_or_else(|| Error::missing_output("normalize results"))?,
                            OutputRole::Indices,
                        ),
                    ],
                )
            } else {
                OperationNode::new(
                    id,
                    old.opcode(),
                    old.schema().cloned(),
                    old.attributes().clone(),
                    old.inputs().to_vec(),
                    old.input_roles().to_vec(),
                    old.results().to_vec(),
                )
            };
            let operation =
                operation.with_source_location(old.source_location().map(ToOwned::to_owned));
            for result in operation.results() {
                self.value_results[result.value().index()] = Some((id, result.index()));
                let value = self
                    .values
                    .get_mut(result.value().index())
                    .ok_or_else(|| Error::missing_output("normalize results"))?;
                if value.binding().operation_output().is_some() {
                    value.set_producer(id, result.index());
                }
            }

            remapped.insert(old.id(), id);
            operations.push(operation);
        }

        self.operations = operations;
        for block in &mut self.blocks {
            *block = Block::new(
                block.id(),
                remap_order(block.operations(), &remapped)?,
                block.outputs().to_vec(),
            );
        }

        for partition in &mut self.partitions {
            *partition = Partition::new(
                partition.id(),
                remap_order(partition.operations(), &remapped)?,
                None,
            );
        }
        Ok(())
    }
}

fn remap_order(
    order: &[OperationId],
    remapped: &HashMap<OperationId, OperationId>,
) -> Result<Vec<OperationId>> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for old in order {
        let operation = *remapped
            .get(old)
            .ok_or_else(|| Error::missing_node("remap order"))?;
        if seen.insert(operation) {
            result.push(operation);
        }
    }
    Ok(result)
}
