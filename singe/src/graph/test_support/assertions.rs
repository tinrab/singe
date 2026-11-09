use super::*;

pub fn assert_value_metadata(
    graph: &Graph,
    value: ValueId,
    data_type: DataType,
    shape: &[i64],
    layout: &[i64],
    device: Device,
) {
    let ir = graph.to_ir().unwrap();
    assert_eq!(ir.required_value_data_type(value).unwrap(), data_type);
    assert_eq!(ir.required_value_shape(value).unwrap().dimensions(), shape);
    assert_eq!(
        ir.required_value_layout(value).unwrap().dimensions(),
        layout
    );
    assert_eq!(ir.required_value(value).unwrap().device(), device);
}

pub fn assert_value_shape(graph: &Graph, value: ValueId, shape: &[i64]) {
    let ir = graph.to_ir().unwrap();
    assert_eq!(ir.required_value_shape(value).unwrap().dimensions(), shape);
}

pub fn assert_value_layout(
    graph: &Graph,
    value: ValueId,
    dimensions: &[i64],
    strides: &[i64],
    offset: usize,
) {
    let ir = graph.to_ir().unwrap();
    let layout = ir.required_value_layout(value).unwrap();
    assert_eq!(layout.dimensions(), dimensions);
    assert_eq!(layout.strides(), strides);
    assert_eq!(layout.storage_offset(), offset);
}

pub fn assert_operation(
    graph: &Graph,
    value: ValueId,
    operation: Operation,
    inputs: &[ValueId],
    attributes: &OperationAttributes,
) {
    let ir = graph.to_ir().unwrap();
    let operation_id = ir.result_for_value(value).unwrap().0;
    let actual = ir.required_operation(operation_id).unwrap();
    let input_values = inputs.to_vec();
    assert_eq!(actual.opcode(), Some(operation));
    assert_eq!(actual.inputs(), input_values);
    assert_eq!(actual.attributes(), attributes);
}

pub fn folded_constant_tensor(graph: &Graph, value: ValueId) -> Tensor {
    let ir = graph.to_ir().unwrap();
    ir.constant_tensor(value)
        .or_else(|| ir.initializer_tensor(value))
        .unwrap()
        .clone()
}
