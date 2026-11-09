use crate::graph::{
    node::{Node, NodeKind},
    operation::GraphOperationRequest,
};

impl GraphOperationRequest {
    pub(crate) fn into_node(self) -> Node {
        Node::create_with_shape(
            NodeKind::Operation(self.inputs),
            self.name,
            self.output.data_type,
            self.output.shape,
            self.output.device,
        )
    }
}
