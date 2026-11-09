#[cfg(test)]
use std::cell::Cell;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    backend::registry::{
        OperationAttributes, OperationRequest, OutputRole, operation_schema_for_attributes,
    },
    error::{Error, Result},
    graph::{
        ir::{
            Block, BlockId, Graph, OperationId, OperationNode, OperationResult, Partition,
            PartitionId, Value, ValueAlias, ValueBinding, ValueId,
        },
        node::{Node, NodeRef, NodeRefKind},
        shape_program::ShapeProgram,
    },
    layout::Layout,
    tensor::Tensor,
};

#[cfg(test)]
thread_local! {
    static LOWERINGS: Cell<usize> = const { Cell::new(0) };
}

impl Graph {
    /// Creates an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            operations: Vec::new(),
            blocks: vec![Block::new(BlockId::new(0), Vec::new(), Vec::new())],
            partitions: vec![Partition::new(PartitionId::new(0), Vec::new(), None)],
            constant_tensors: Vec::new(),
            value_results: Vec::new(),
            output_names: Vec::new(),
            initializers: HashMap::new(),
            static_initializers: HashSet::new(),
            state_kinds: HashMap::new(),
            input_dimension_guards: HashMap::new(),
            shape_program: ShapeProgram::default(),
            input_writes: Vec::new(),
            counted_outputs: Vec::new(),
            generators: Vec::new(),
            capture_info: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn lowering_count_for_test() -> usize {
        LOWERINGS.with(Cell::get)
    }

    /// Returns the execution form of this graph: operations in dependency order, with `TopK` result projections reunited.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph has a missing dependency or cycle.
    pub fn to_ir(&self) -> Result<Self> {
        #[cfg(test)]
        LOWERINGS.with(|count| count.set(count.get() + 1));
        let order = self
            .topological_sort()?
            .into_iter()
            .map(|node| OperationId::new(node.index()))
            .collect::<Vec<_>>();
        let mut ir = self.clone();
        ir.refresh_input_bindings();
        ir.blocks = vec![Block::new(
            BlockId::new(0),
            order.clone(),
            self.output_values().to_vec(),
        )];
        ir.partitions = vec![Partition::new(PartitionId::new(0), order, None)];
        ir.normalize_results()?;
        Ok(ir)
    }

    /// Appends `node` as one operation producing one value, without validating it.
    pub(crate) fn push_node(&mut self, node: Node) -> ValueId {
        let id = ValueId::new(self.values.len());
        let operation = OperationId::new(self.operations.len());
        let (operation_node, value, constant) = self.node_parts(id, operation, node);
        self.values.push(value);
        self.operations.push(operation_node);
        self.constant_tensors.push(constant);
        self.value_results.push(Some((operation, 0)));
        if let Some(block) = self.blocks.first_mut() {
            block.push_operation(operation);
        }

        if let Some(partition) = self.partitions.first_mut() {
            partition.push_operation(operation);
        }
        id
    }

    /// Replaces node `id` with `node`, keeping its id and position.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` is not a node of this graph.
    pub(crate) fn replace_node(&mut self, id: ValueId, node: Node) -> Result<()> {
        let operation = self.node_operation(id)?;
        let (operation_node, value, constant) = self.node_parts(id, operation, node);
        self.values[id.index()] = value;
        self.operations[operation.index()] = operation_node;
        self.constant_tensors[id.index()] = constant;
        Ok(())
    }

    /// Applies `update` to node `id` and stores the result.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` is not a node of this graph.
    pub(crate) fn update_node<R>(
        &mut self,
        id: ValueId,
        update: impl FnOnce(&mut Node) -> R,
    ) -> Result<R> {
        let mut node = self.required_node(id)?.to_node();
        let result = update(&mut node);
        self.replace_node(id, node)?;
        Ok(result)
    }

    /// Returns result `index` of node `id`.
    ///
    /// # Errors
    ///
    /// Returns an error if `id` is not a node or has no result `index`.
    pub(crate) fn node_result(&self, id: ValueId, index: usize) -> Result<ValueId> {
        self.ensure_node(id)?;
        self.operations[id.index()]
            .results()
            .get(index)
            .map(|result| result.value())
            .ok_or_else(|| Error::missing_output("node result"))
    }

    /// Sets the graph outputs and their public names, without validating them.
    pub(crate) fn set_output_values_with_names(
        &mut self,
        values: Vec<ValueId>,
        names: Vec<Option<String>>,
    ) {
        self.output_names = names;
        if let Some(block) = self.blocks.first_mut() {
            block.set_outputs(values);
        }
    }

    /// Sets the import provenance of every node from `start` on.
    #[cfg(feature = "onnx")]
    pub(crate) fn set_source_location_from(&mut self, start: usize, source: &str) {
        for operation in self.operations.iter_mut().skip(start) {
            operation.set_source_location(Some(source.to_owned()));
        }
    }

    /// Returns the number of values node `id` produces.
    pub(crate) fn node_output_count(&self, id: ValueId) -> Result<usize> {
        self.ensure_node(id)?;
        Ok(self.operations[id.index()].results().len())
    }

    /// Rewrites every node through `rewrite`, in node order.
    ///
    /// # Errors
    ///
    /// Returns the first error `rewrite` returns.
    pub(crate) fn rewrite_nodes(
        &mut self,
        mut rewrite: impl FnMut(&Self, Node) -> Result<Node>,
    ) -> Result<()> {
        for id in self.node_ids() {
            let node = rewrite(self, self.required_node(id)?.to_node())?;
            self.replace_node(id, node)?;
        }
        Ok(())
    }

    /// Returns node `id`, or `None` when `id` is not a node of this graph.
    pub(crate) fn node(&self, id: ValueId) -> Option<NodeRef<'_>> {
        let operation = self.node_operation(id).ok()?;
        let operation = &self.operations[operation.index()];
        let value = &self.values[id.index()];
        let kind = match operation.attributes() {
            OperationAttributes::Input(name) => NodeRefKind::Input(name),
            OperationAttributes::Constant => NodeRefKind::Constant(self.constant_tensor(id)?),
            attributes => NodeRefKind::Operation {
                operation: operation.opcode()?,
                attributes,
                inputs: operation.inputs(),
            },
        };
        Some(NodeRef::new(
            kind,
            value.name(),
            operation.source_location(),
            value.data_type(),
            value.layout(),
            value.device(),
        ))
    }

    /// Returns node `id`.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` is not a node of this graph.
    pub(crate) fn required_node(&self, id: ValueId) -> Result<NodeRef<'_>> {
        self.node(id)
            .ok_or_else(|| Error::missing_node("required node"))
    }

    pub(crate) fn ensure_node(&self, id: ValueId) -> Result<()> {
        self.node_operation(id).map(|_| ())
    }

    /// Returns the number of nodes in the graph.
    pub(crate) fn node_count(&self) -> usize {
        self.values.len()
    }

    /// Iterates over all node ids in insertion order.
    pub(crate) fn node_ids(&self) -> impl Iterator<Item = ValueId> + use<> {
        (0..self.node_count()).map(ValueId::new)
    }

    /// Iterates over graph node ids and nodes in insertion order.
    pub(crate) fn node_entries(&self) -> impl Iterator<Item = (ValueId, NodeRef<'_>)> + '_ {
        self.node_ids()
            .filter_map(|id| self.node(id).map(|node| (id, node)))
    }

    /// Returns the operation producing node `id`, which construction keeps at the same index.
    fn node_operation(&self, id: ValueId) -> Result<OperationId> {
        match self.value_results.get(id.index()) {
            Some(Some((operation, 0))) if operation.index() == id.index() => Ok(*operation),
            _ => Err(Error::missing_node("node operation")),
        }
    }

    fn node_parts(
        &self,
        id: ValueId,
        operation: OperationId,
        node: Node,
    ) -> (OperationNode, Value, Option<Tensor>) {
        let view = node.view();
        let attributes = view.attributes();
        let opcode = view.operation();
        let schema = opcode.map(|opcode| operation_schema_for_attributes(opcode, &attributes));
        let inputs = view.inputs().to_vec();
        let input_roles = schema.as_ref().map_or_else(Vec::new, |schema| {
            schema.operands().roles_for_count(inputs.len())
        });
        let role = schema
            .as_ref()
            .and_then(|schema| schema.output_role(0))
            .unwrap_or(OutputRole::Output);
        let binding = if let Some((name, ..)) = view.input_metadata() {
            self.input_binding(name)
        } else if view.is_constant() {
            ValueBinding::constant()
        } else {
            ValueBinding::produced_by_operation(operation, 0)
        };
        let alias = view
            .alias_input()
            .map(|source| ValueAlias::new(source, view.is_view()));
        let data_type = role.data_type(view.output_data_type());
        let device = view.output_device();
        let (name, source_location, layout, constant) = node.into_parts();
        let value = Value::new(id, name, data_type, layout, device, binding, alias);
        let operation_node = OperationNode::new(
            operation,
            opcode,
            schema,
            attributes,
            inputs,
            input_roles,
            vec![OperationResult::new(0, id, role)],
        )
        .with_source_location(source_location);
        (operation_node, value, constant)
    }

    fn input_binding(&self, name: &str) -> ValueBinding {
        if self.static_initializers.contains(name) {
            ValueBinding::static_input(name.to_owned())
        } else if self.initializers.contains_key(name) {
            ValueBinding::initializer_input(name.to_owned(), true)
        } else {
            ValueBinding::runtime_input(name.to_owned())
        }
    }

    /// Recomputes input bindings after initializers change.
    pub(crate) fn refresh_input_bindings(&mut self) {
        for index in 0..self.values.len() {
            let Some(name) = self.values[index].binding().input_name() else {
                continue;
            };
            let binding = self.input_binding(name);
            self.values[index].set_binding(binding);
        }
    }

    /// Returns all graph nodes in dependency order.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph contains a missing dependency or cycle.
    pub(crate) fn topological_sort(&self) -> Result<Vec<ValueId>> {
        let node_count = self.node_count();
        let mut in_degree = vec![0usize; node_count];
        let mut adj: Vec<Vec<ValueId>> = vec![Vec::new(); node_count];
        for operation in &self.operations {
            let id = ValueId::new(operation.id().index());
            for &dep in operation.inputs() {
                self.ensure_node(dep)?;
                adj[dep.index()].push(id);
                in_degree[id.index()] += 1;
            }
        }

        let mut queue: VecDeque<ValueId> = in_degree
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d == 0)
            .map(|(index, _)| ValueId::new(index))
            .collect();
        let mut order = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &next in &adj[node.index()] {
                in_degree[next.index()] -= 1;
                if in_degree[next.index()] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() != node_count {
            return Err(Error::GraphCycleDetected);
        }
        Ok(order)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Adds a named operation node with declared output layout metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the node references a missing dependency node or its declared metadata contradicts operation-intrinsic metadata.
    pub(crate) fn add_named_node_with_layout(&mut self, mut node: Node) -> Result<ValueId> {
        let view = node.view();
        for &input in view.inputs() {
            self.ensure_node(input)?;
        }

        let id = ValueId::new(self.node_count());
        view.validate_declared_metadata(format!("graph add_node node {}", id.index()))?;
        self.validate_operation_request(&view)?;
        let metadata = if let Some(input) = view.alias_input() {
            let source = &self.required_node(input)?;
            let layout =
                if view.view_output_device(source.output_device()) == source.output_device() {
                    view.view_output_layout(source.output_layout())?
                } else {
                    Layout::from_shape(view.output_shape())
                };
            Some((view.output_data_type(), layout, view.output_device()))
        } else {
            None
        };
        if let Some((data_type, layout, device)) = metadata {
            node.replace_output_metadata(data_type, layout, device);
        }
        Ok(self.push_node(node))
    }

    fn validate_operation_request(&self, node: &NodeRef<'_>) -> Result<()> {
        let Some(operation) = node.operation() else {
            return Ok(());
        };
        let inputs = node
            .inputs()
            .iter()
            .map(|&id| self.required_node(id))
            .collect::<Result<Vec<_>>>()?;
        let attributes = node.attributes();
        let checked = OperationRequest::new(
            operation,
            attributes.clone(),
            inputs.iter().map(NodeRef::output_spec).collect(),
        )
        .check()?;
        let output = checked
            .inference()
            .outputs()
            .first()
            .ok_or_else(|| Error::missing_output("validate operation request"))?;
        if output.data_type() != node.output_data_type() {
            return Err(Error::data_type_mismatch(
                "graph operation output",
                output.data_type(),
                node.output_data_type(),
            ));
        }

        if output.shape() != node.output_shape() {
            return Err(Error::shape_mismatch(
                "graph operation output",
                output.shape().dimensions().to_vec(),
                node.output_shape().dimensions().to_vec(),
            ));
        }

        let first = inputs
            .first()
            .ok_or_else(|| Error::missing_input("validate operation request"))?;
        for input in &inputs {
            if input.output_device() != first.output_device() {
                return Err(Error::device_mismatch(
                    "graph operation operands",
                    first.output_device(),
                    input.output_device(),
                ));
            }
        }

        let device = match attributes {
            OperationAttributes::CopyTo(device) => device,
            _ => first.output_device(),
        };
        if device != node.output_device() {
            return Err(Error::device_mismatch(
                "graph operation output",
                device,
                node.output_device(),
            ));
        }
        Ok(())
    }
}

impl Graph {
    pub(crate) fn validate_node_contracts_with_schema(
        &self,
        id: ValueId,
        node: &NodeRef<'_>,
        name: &str,
    ) -> Result<()> {
        let dependencies = node.inputs();
        let schema = node.schema();

        if let Some(schema) = &schema
            && !schema.operands().accepts_count(dependencies.len())
        {
            return Err(Error::unsupported_operation_with_details(
                "graph pass validate_node_contracts",
                format!(
                    "node {} operation {} has {} operands, expected schema {:?}",
                    id.index(),
                    schema.name(),
                    dependencies.len(),
                    schema.operands()
                ),
            ));
        }

        for &dependency in dependencies {
            if self.ensure_node(dependency).is_err() {
                return Err(Error::unsupported_operation_with_details(
                    "graph pass validate_node_contracts",
                    format!(
                        "node {} references missing dependency {}",
                        id.index(),
                        dependency.index()
                    ),
                ));
            }
        }

        node.validate_terminal_declared_metadata(name)?;
        let Some(schema) = schema else {
            return Ok(());
        };

        node.validate_schema_declared_metadata_with_schema(name, &schema)?;
        self.validate_operation_request(node)
    }
}
