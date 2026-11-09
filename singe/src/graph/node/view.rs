use crate::{
    backend::registry::{
        AliasBehavior, Operation, OperationAttributes, OperationSchema, operation_schema,
        operation_schema_for_attributes,
    },
    data_type::DataType,
    device::Device,
    graph::{
        GraphOperationInputs,
        ir::ValueId,
        node::{Node, NodeKind},
    },
    layout::Layout,
    shape::Shape,
    tensor::Tensor,
    value::ValueSpec,
};

/// What a borrowed node is: an input, a constant, or an operation over earlier values.
#[derive(Debug, Clone, Copy)]
pub enum NodeRefKind<'a> {
    Input(&'a str),
    Constant(&'a Tensor),
    Operation {
        operation: Operation,
        attributes: &'a OperationAttributes,
        inputs: &'a [ValueId],
    },
}

/// A node borrowed from a graph or from an owned `Node`: one operation and the value it produces.
#[derive(Debug, Clone, Copy)]
pub struct NodeRef<'a> {
    kind: NodeRefKind<'a>,
    name: Option<&'a str>,
    source_location: Option<&'a str>,
    data_type: DataType,
    layout: &'a Layout,
    device: Device,
}

impl<'a> NodeRef<'a> {
    pub(crate) fn new(
        kind: NodeRefKind<'a>,
        name: Option<&'a str>,
        source_location: Option<&'a str>,
        data_type: DataType,
        layout: &'a Layout,
        device: Device,
    ) -> Self {
        Self {
            kind,
            name,
            source_location,
            data_type,
            layout,
            device,
        }
    }

    /// Returns an owned copy of this node.
    pub(crate) fn to_node(self) -> Node {
        self.with_kind(match self.kind {
            NodeRefKind::Input(name) => NodeKind::input(name.to_owned()),
            NodeRefKind::Constant(tensor) => NodeKind::constant(tensor.clone()),
            NodeRefKind::Operation {
                operation,
                attributes,
                inputs,
            } => NodeKind::Operation(GraphOperationInputs::new(
                operation,
                attributes.clone(),
                inputs.to_vec(),
            )),
        })
    }

    /// Returns an owned node of `kind` with this node's name, source location, and output metadata.
    pub(crate) fn with_kind(self, kind: NodeKind) -> Node {
        Node::create(
            kind,
            self.name.map(ToOwned::to_owned),
            self.data_type,
            self.layout.clone(),
            self.device,
        )
        .with_source_location(self.source_location.map(ToOwned::to_owned))
    }

    /// Returns the registry operation represented by this node.
    #[must_use]
    pub fn operation(&self) -> Option<Operation> {
        match self.kind {
            NodeRefKind::Operation { operation, .. } => Some(operation),
            NodeRefKind::Input(_) | NodeRefKind::Constant(_) => None,
        }
    }

    /// Returns graph input dependencies in operand order.
    #[must_use]
    pub fn inputs(&self) -> &'a [ValueId] {
        match self.kind {
            NodeRefKind::Operation { inputs, .. } => inputs,
            NodeRefKind::Input(_) | NodeRefKind::Constant(_) => &[],
        }
    }

    /// Returns the canonical attributes of this node.
    #[must_use]
    pub fn attributes(&self) -> OperationAttributes {
        match self.kind {
            NodeRefKind::Input(name) => OperationAttributes::Input(name.to_owned()),
            NodeRefKind::Constant(_) => OperationAttributes::Constant,
            NodeRefKind::Operation { attributes, .. } => attributes.clone(),
        }
    }

    /// Returns the source node when this node is a metadata-only view.
    #[must_use]
    pub fn view_input(&self) -> Option<ValueId> {
        let operation = self.operation()?;
        (operation_schema(operation).alias_behavior() == AliasBehavior::View)
            .then(|| self.inputs().first().copied())
            .flatten()
    }

    /// Returns the source node when this node aliases another tensor value.
    #[must_use]
    pub fn alias_input(&self) -> Option<ValueId> {
        let operation = self.operation()?;
        (operation_schema(operation).alias_behavior() != AliasBehavior::None)
            .then(|| self.inputs().first().copied())
            .flatten()
    }

    /// Returns the schema-declared alias behavior for this node.
    #[must_use]
    pub fn alias_behavior(&self) -> AliasBehavior {
        self.schema()
            .map_or(AliasBehavior::None, |schema| schema.alias_behavior())
    }

    /// Returns the operation schema for this node.
    #[must_use]
    pub fn schema(&self) -> Option<OperationSchema> {
        let NodeRefKind::Operation {
            operation,
            attributes,
            ..
        } = self.kind
        else {
            return None;
        };
        Some(operation_schema_for_attributes(operation, attributes))
    }

    /// Returns whether this node represents a metadata-only view.
    #[must_use]
    pub fn is_view(&self) -> bool {
        self.view_input().is_some()
    }

    /// Returns declared input binding metadata when this node is a graph input.
    #[must_use]
    pub fn input_metadata(&self) -> Option<(&'a str, DataType, &'a Shape, Device)> {
        let NodeRefKind::Input(name) = self.kind else {
            return None;
        };
        Some((name, self.data_type, self.layout.shape(), self.device))
    }

    /// Returns the embedded tensor payload for an output when it is constant-backed.
    #[must_use]
    pub fn constant_tensor(&self, output_index: usize) -> Option<&'a Tensor> {
        match (self.kind, output_index) {
            (NodeRefKind::Constant(tensor), 0) => Some(tensor),
            _ => None,
        }
    }

    /// Returns whether this node is backed by an embedded constant tensor.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        self.constant_tensor(0).is_some()
    }

    /// Returns this node's optional graph-local name.
    #[must_use]
    pub fn name(&self) -> Option<&'a str> {
        self.name
    }

    pub(crate) fn source_location(&self) -> Option<&'a str> {
        self.source_location
    }

    /// Returns this node's declared output data type.
    #[must_use]
    pub fn output_data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns this node's declared output shape.
    #[must_use]
    pub fn output_shape(&self) -> &'a Shape {
        self.layout.shape()
    }

    /// Returns this node's declared output layout.
    #[must_use]
    pub fn output_layout(&self) -> &'a Layout {
        self.layout
    }

    /// Returns this node's declared output device.
    #[must_use]
    pub fn output_device(&self) -> Device {
        self.device
    }

    /// Returns logical result metadata independently of its concrete layout.
    #[must_use]
    pub fn output_spec(&self) -> ValueSpec {
        ValueSpec::new(self.data_type, self.output_shape().clone())
    }

    /// Returns whether both nodes declare the same output metadata.
    #[must_use]
    pub fn has_same_value_metadata(&self, other: &Self) -> bool {
        self.has_same_output_device(other)
            && self.has_same_output_data_type(other)
            && self.has_same_output_layout(other)
    }

    /// Returns whether both nodes declare the same output device.
    #[must_use]
    pub fn has_same_output_device(&self, other: &Self) -> bool {
        self.device == other.device
    }

    /// Returns whether both nodes declare the same output data type.
    #[must_use]
    pub fn has_same_output_data_type(&self, other: &Self) -> bool {
        self.data_type == other.data_type
    }

    /// Returns whether both nodes declare the same output layout.
    #[must_use]
    pub fn has_same_output_layout(&self, other: &Self) -> bool {
        self.layout == other.layout
    }
}

impl Node {
    /// Borrows this node.
    pub(crate) fn view(&self) -> NodeRef<'_> {
        let kind = match &self.kind {
            NodeKind::Input { name } => NodeRefKind::Input(name),
            NodeKind::Constant { tensor } => NodeRefKind::Constant(tensor),
            NodeKind::Operation(invocation) => NodeRefKind::Operation {
                operation: invocation.operation,
                attributes: &invocation.attributes,
                inputs: &invocation.inputs,
            },
        };
        NodeRef::new(
            kind,
            self.name.as_deref(),
            self.source_location.as_deref(),
            self.output_data_type,
            &self.output_layout,
            self.output_device,
        )
    }
}
