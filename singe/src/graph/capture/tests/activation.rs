use super::*;

#[test]
fn test_graph_capture_records_tensor_activation_math_and_comparison_methods() {
    let shape = Shape::create(vec![2]).unwrap();
    let input = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap();
    let other = Tensor::from_vec(Device::Cpu, &shape, vec![3.0_f32, 1.0]).unwrap();

    let (captured, captured_outputs) = Graph::capture([("x", input), ("other", other)], |inputs| {
        let softmax = inputs[0].softmax(0)?;
        let relu = inputs[0].relu()?;
        let sqrt = inputs[0].sqrt()?;
        let maximum = inputs[0].maximum(&inputs[1])?;
        let equal = inputs[0].equal(&inputs[1])?;
        let lerp = inputs[0].lerp(&inputs[1], 0.25)?;
        let expanded = inputs[0].expand(&Shape::create(vec![2, 2])?)?;
        Ok(vec![softmax, relu, sqrt, maximum, equal, lerp, expanded])
    })
    .unwrap();

    assert_close!(
        &captured_outputs[0].to_vec::<f32>().unwrap(),
        &[0.268_941_43, 0.731_058_6],
        1e-6,
    );
    assert_close!(
        &captured_outputs[3].to_vec::<f32>().unwrap(),
        &[3.0, 2.0],
        1e-6,
    );
    assert_eq!(
        captured_outputs[4].to_vec::<bool>().unwrap(),
        [false, false]
    );
    assert_close!(
        &captured_outputs[5].to_vec::<f32>().unwrap(),
        &[1.5, 1.75],
        1e-6,
    );
    assert_close!(
        &captured_outputs[6].to_vec::<f32>().unwrap(),
        &[1.0, 2.0, 1.0, 2.0],
        1e-6,
    );

    let ir = captured.to_ir().unwrap();
    let softmax_dim = ir.operations().iter().find_map(|operation| {
        (operation.opcode() == Some(Operation::Softmax))
            .then(|| operation.attributes().dimension())
            .flatten()
    });
    assert_eq!(softmax_dim, Some(0));
    assert!(
        ir.operations()
            .iter()
            .any(|operation| operation.opcode() == Some(Operation::Relu))
    );
    assert!(
        ir.operations()
            .iter()
            .any(|operation| operation.opcode() == Some(Operation::Sqrt))
    );
    assert!(
        ir.operations()
            .iter()
            .any(|operation| operation.opcode() == Some(Operation::Maximum))
    );
    assert!(
        ir.operations()
            .iter()
            .any(|operation| operation.opcode() == Some(Operation::Equal))
    );

    let lerp = ir.operations().iter().find_map(|operation| {
        (operation.opcode() == Some(Operation::Lerp))
            .then(|| operation.attributes().lerp_t())
            .flatten()
    });
    assert_eq!(lerp, Some(0.25));

    let expand_shape = ir.operations().iter().find_map(|operation| {
        if operation.opcode() == Some(Operation::Expand) {
            operation
                .attributes()
                .shape()
                .map(|shape| shape.dimensions().to_vec())
        } else {
            None
        }
    });
    assert_eq!(expand_shape, Some(vec![2, 2]));
}
