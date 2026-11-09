use super::*;
use crate::backend::registry::TransposeAttributes;

#[test]
fn test_graph_capture_records_tensor_views_and_indexing() {
    let shape = Shape::create(vec![2, 3]).unwrap();
    let index_shape = Shape::create(vec![2]).unwrap();
    let input =
        Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let indices = Tensor::from_vec(Device::Cpu, &index_shape, vec![2_i64, 0]).unwrap();

    let (captured, captured_outputs) =
        Graph::capture([("x", input), ("indices", indices)], |inputs| {
            let reshaped = inputs[0].reshape([3, 2])?;
            let transposed = inputs[0].transpose(0, 1)?;
            let narrowed = inputs[0].narrow(1, 1, 2)?;
            let selected = inputs[0].index_select(1, &inputs[1])?;
            Ok(vec![reshaped, transposed, narrowed, selected])
        })
        .unwrap();

    assert_close!(
        &captured_outputs[0].to_vec::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1e-6,
    );
    assert_eq!(captured_outputs[1].shape().dimensions(), &[3, 2]);
    assert_close!(
        &captured_outputs[2].to_vec::<f32>().unwrap(),
        &[2.0, 3.0, 5.0, 6.0],
        1e-6,
    );
    assert_close!(
        &captured_outputs[3].to_vec::<f32>().unwrap(),
        &[3.0, 1.0, 6.0, 4.0],
        1e-6,
    );

    let ir = captured.to_ir().unwrap();
    let reshape = ir.operations().iter().find_map(|operation| {
        if let OperationAttributes::Shape(shape) = operation.attributes()
            && operation.opcode() == Some(Operation::Reshape)
        {
            Some(shape.dimensions().to_vec())
        } else {
            None
        }
    });
    assert_eq!(reshape, Some(vec![3, 2]));

    let transpose = ir.operations().iter().find_map(|operation| {
        if let OperationAttributes::Transpose(TransposeAttributes { dim0, dim1 }) =
            operation.attributes()
        {
            Some((*dim0, *dim1))
        } else {
            None
        }
    });
    assert_eq!(transpose, Some((0, 1)));

    let slice = ir.operations().iter().find_map(|operation| {
        if let OperationAttributes::Slice(SliceAttributes {
            dim, start, length, ..
        }) = operation.attributes()
        {
            Some((*dim, *start, *length))
        } else {
            None
        }
    });
    assert_eq!(slice, Some((1, 1, 2)));

    let index_select = ir.operations().iter().find_map(|operation| {
        if let OperationAttributes::IndexSelect(dim) = operation.attributes() {
            Some(*dim)
        } else {
            None
        }
    });
    assert_eq!(index_select, Some(1));
}
