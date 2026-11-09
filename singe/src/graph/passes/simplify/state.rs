use std::collections::HashMap;

use crate::{
    error::{Error, Result},
    graph::{
        Graph,
        ir::ValueId,
        node::{Node, NodeRef},
        passes::{
            fold::{ConstantKey, GeneratedConstantKey, constant_key, generated_constant_key},
            simplify::{operation_key::OperationKey, redundant::redundant_operation_input},
        },
    },
    tensor::Tensor,
};

pub struct SimplifyState {
    id_map: HashMap<ValueId, ValueId>,
    constant_map: HashMap<ConstantKey, ValueId>,
    generated_constant_map: HashMap<GeneratedConstantKey, ValueId>,
    operation_map: HashMap<OperationKey, ValueId>,
    nodes: Vec<Node>,
}

impl SimplifyState {
    pub(crate) fn create(graph: &Graph) -> Self {
        Self {
            id_map: HashMap::new(),
            constant_map: HashMap::new(),
            generated_constant_map: HashMap::new(),
            operation_map: HashMap::new(),
            nodes: Vec::with_capacity(graph.node_count()),
        }
    }

    pub(crate) fn id_map(&self) -> &HashMap<ValueId, ValueId> {
        &self.id_map
    }

    pub(crate) fn redundant_replacement(
        &self,
        graph: &Graph,
        node: &NodeRef<'_>,
    ) -> Result<Option<ValueId>> {
        let Some(input) = redundant_operation_input(graph, node)? else {
            return Ok(None);
        };
        self.id_map
            .get(&input)
            .copied()
            .ok_or_else(|| Error::missing_node("redundant replacement"))
            .map(Some)
    }

    pub(crate) fn existing_input_constant(&self, node: &NodeRef<'_>) -> Option<ValueId> {
        let tensor = node.constant_tensor(0)?;
        let key = constant_key(tensor);
        self.constant_map.get(&key).copied()
    }

    pub(crate) fn existing_remapped_node(
        &self,
        original: &NodeRef<'_>,
        remapped: &NodeRef<'_>,
        operation_key: Option<&OperationKey>,
    ) -> Result<Option<ValueId>> {
        if let Some(key) = operation_key
            && let Some(existing) = self.operation_map.get(key).copied()
        {
            return Ok(Some(existing));
        }

        let Some(tensor) = remapped.constant_tensor(0) else {
            return Ok(None);
        };
        let key = constant_key(tensor);
        if let Some(existing) = self.constant_map.get(&key).copied() {
            return Ok(Some(existing));
        }

        if deduplicates_by_value(original, tensor)
            && let Some(key) = generated_constant_key(tensor)?
            && let Some(existing) = self.generated_constant_map.get(&key).copied()
        {
            return Ok(Some(existing));
        }
        Ok(None)
    }

    pub(crate) fn retain(
        &mut self,
        old_id: ValueId,
        original: &NodeRef<'_>,
        remapped: Node,
        operation_key: Option<OperationKey>,
    ) -> Result<()> {
        let new_id = ValueId::new(self.nodes.len());
        if let Some(tensor) = remapped.view().constant_tensor(0) {
            self.constant_map.insert(constant_key(tensor), new_id);
            if deduplicates_by_value(original, tensor)
                && let Some(key) = generated_constant_key(tensor)?
            {
                self.generated_constant_map.insert(key, new_id);
            }
        }

        self.nodes.push(remapped);
        self.id_map.insert(old_id, new_id);
        if let Some(key) = operation_key {
            self.operation_map.insert(key, new_id);
        }
        Ok(())
    }

    pub(crate) fn replace(&mut self, old_id: ValueId, replacement: ValueId) {
        self.id_map.insert(old_id, replacement);
    }

    pub(crate) fn finish(self, graph: &Graph) -> Result<(Graph, Vec<(ValueId, ValueId)>)> {
        let remapped_nodes = graph.remapped_node_pairs(&self.id_map)?;
        let graph = Graph::from_retained_nodes(graph, &self.id_map, self.nodes)?;
        Ok((graph, remapped_nodes))
    }
}

/// Returns whether a constant merges with equal-valued constants: folded results, and authored scalars such as quantization parameters.
fn deduplicates_by_value(original: &NodeRef<'_>, tensor: &Tensor) -> bool {
    !original.is_constant() || tensor.shape().rank() == 0
}
