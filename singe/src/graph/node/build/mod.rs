use crate::{
    data_type::DataType,
    device::Device,
    graph::node::{Node, kind::NodeKind},
    layout::Layout,
    shape::Shape,
    tensor::Tensor,
};

impl Node {
    pub(crate) fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub(crate) fn with_source_location(mut self, source: Option<String>) -> Self {
        self.source_location = source;
        self
    }

    /// Creates a graph node from raw operation kind and declared output metadata.
    pub(crate) fn create(
        kind: NodeKind,
        name: Option<String>,
        output_data_type: DataType,
        output_layout: Layout,
        output_device: Device,
    ) -> Self {
        Self {
            kind,
            name,
            source_location: None,
            output_data_type,
            output_layout,
            output_device,
        }
    }

    /// Creates a graph node with compact shape metadata.
    pub(crate) fn create_with_shape(
        kind: NodeKind,
        name: Option<String>,
        output_data_type: DataType,
        output_shape: Shape,
        output_device: Device,
    ) -> Self {
        Self::create(
            kind,
            name,
            output_data_type,
            Layout::new(output_shape, 0),
            output_device,
        )
    }

    /// Creates a graph input node.
    pub(crate) fn input_node(
        name: String,
        data_type: DataType,
        layout: Layout,
        device: Device,
    ) -> Self {
        Self::create(
            NodeKind::input(name.clone()),
            Some(name),
            data_type,
            layout,
            device,
        )
    }

    /// Creates an embedded constant node.
    #[must_use]
    pub fn constant_node(tensor: Tensor) -> Self {
        let data_type = tensor.data_type();
        let layout = Layout::from_tensor(&tensor);
        let device = tensor.device();
        Self::create(NodeKind::constant(tensor), None, data_type, layout, device)
    }

    /// Splits this node into its name, source location, output layout, and constant tensor.
    pub(crate) fn into_parts(self) -> (Option<String>, Option<String>, Layout, Option<Tensor>) {
        let constant = match self.kind {
            NodeKind::Constant { tensor } => Some(tensor),
            NodeKind::Input { .. } | NodeKind::Operation(_) => None,
        };
        (
            self.name,
            self.source_location,
            self.output_layout,
            constant,
        )
    }
}
