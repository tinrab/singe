use crate::{
    device::Device,
    error::{Error, Result},
    graph::{
        ir::ValueId,
        node::{Node, NodeRef, kind::NodeKind},
    },
    layout::Layout,
    shape::Shape,
    tensor::Tensor,
};

impl NodeRef<'_> {
    /// Remaps canonical operands without rebuilding operation attributes.
    pub(crate) fn map_inputs(
        &self,
        mut map: impl FnMut(ValueId) -> Result<ValueId>,
    ) -> Result<Node> {
        let mut node = self.to_node();
        if let NodeKind::Operation(operation) = &mut node.kind {
            for input in &mut operation.inputs {
                *input = map(*input)?;
            }
        }
        Ok(node)
    }

    pub(crate) fn placed_on(&self, device: Device) -> Result<Node> {
        if self.output_device() == device {
            return Ok(self.to_node());
        }

        if self.copy_to().is_some() {
            return Err(Error::unsupported_operation_with_details(
                "graph placement",
                "explicit copy-to operations require their declared placement",
            ));
        }

        let mut node = if let Some(tensor) = self.constant_tensor(0) {
            let mut node = self.to_node();
            node.replace_with_constant(tensor.to_device(device)?);
            return Ok(node);
        } else if let Some((name, _, _, _)) = self.input_metadata() {
            self.with_input(name.to_owned())
        } else {
            self.to_node()
        };
        node.replace_output_metadata(
            self.output_data_type(),
            self.output_layout().clone(),
            device,
        );
        Ok(node)
    }

    /// Returns a copy with a replaced raw node kind while preserving declared output metadata.
    fn with_replaced_kind(&self, kind: NodeKind) -> Node {
        self.with_kind(kind)
    }
}

impl Node {
    /// Replaces the raw node kind while preserving declared output metadata.
    #[cfg(test)]
    pub(crate) fn replace_kind(&mut self, kind: NodeKind) {
        self.kind = kind;
    }

    /// Replaces this node with an embedded constant and updates output metadata.
    pub(crate) fn replace_with_constant(&mut self, tensor: Tensor) {
        self.replace_output_metadata(
            tensor.data_type(),
            Layout::from_tensor(&tensor),
            tensor.device(),
        );
        self.kind = NodeKind::constant(tensor);
    }
}

impl NodeRef<'_> {
    pub(crate) fn with_reshape_operation(&self, input: ValueId, shape: Shape) -> Node {
        self.with_replaced_kind(NodeKind::reshape(input, shape))
    }
}

impl NodeRef<'_> {}

impl NodeRef<'_> {
    pub(crate) fn with_input(&self, name: String) -> Node {
        self.with_replaced_kind(NodeKind::input(name))
    }
}
