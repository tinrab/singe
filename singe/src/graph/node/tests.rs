use crate::{
    data_type::DataType,
    device::Device,
    graph::{
        ir::ValueId,
        node::{Node, NodeKind},
    },
    layout::Layout,
    shape::Shape,
    tensor::Tensor,
};

impl Node {
    pub(crate) fn copy_to_with_declared_output_for_test(
        input: ValueId,
        device: Device,
        output: &Tensor,
    ) -> Self {
        Self::create(
            NodeKind::copy_to(input, device),
            None,
            output.data_type(),
            Layout::from_tensor(output),
            output.device(),
        )
    }

    pub(crate) fn cast_with_declared_output_for_test(
        input: ValueId,
        data_type: DataType,
        output: &Tensor,
    ) -> Self {
        Self::create(
            NodeKind::cast(input, data_type),
            None,
            output.data_type(),
            Layout::from_tensor(output),
            output.device(),
        )
    }

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
