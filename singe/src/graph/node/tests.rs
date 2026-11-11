use crate::{
    graph::{
        ir::ValueId,
        node::{Node, NodeKind},
    },
    layout::Layout,
    shape::Shape,
    tensor::Tensor,
};

impl Node {
    pub(crate) fn reshape_with_declared_output_for_test(
        input: ValueId,
        shape: Shape,
        output: &Tensor,
    ) -> Self {
        Self::create(
            NodeKind::reshape(input, shape),
            None,
            output.data_type(),
            Layout::from_tensor(output),
            output.device(),
        )
    }
}
