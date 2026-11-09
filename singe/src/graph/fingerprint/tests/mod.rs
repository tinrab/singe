mod pruning;
mod quantization;
mod structural;
mod values;

use crate::{backend::registry::Operation as RegistryOperation, graph::test_support::*};

fn add_quantization_operation(
    graph: &mut Graph,
    operation: RegistryOperation,
    input: ValueId,
    scale: f32,
    zero_point: i64,
    data_type: DataType,
    output: &Tensor,
) -> ValueId {
    graph
        .add_named_quantization_operation(
            None,
            operation,
            input,
            scale,
            zero_point,
            data_type,
            output.shape().clone(),
            output.device(),
        )
        .unwrap()
}
