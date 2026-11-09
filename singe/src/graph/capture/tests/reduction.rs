use super::*;

#[test]
fn test_graph_capture_records_tensor_reduction_and_topk() {
    let shape = Shape::create(vec![2, 2]).unwrap();
    let input = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let (captured, captured_outputs) = Graph::capture([("x", input)], |inputs| {
        let sum = inputs[0].sum(&[1], false)?;
        let (values, indices) = inputs[0].topk(1, 1, true, true)?;
        Ok(vec![sum, values, indices])
    })
    .unwrap();

    assert_close!(
        &captured_outputs[0].to_vec::<f32>().unwrap(),
        &[3.0, 7.0],
        1e-6,
    );
    assert_close!(
        &captured_outputs[1].to_vec::<f32>().unwrap(),
        &[2.0, 4.0],
        1e-6,
    );
    assert_eq!(captured_outputs[2].to_vec::<i64>().unwrap(), [1, 1]);

    let ir = captured.to_ir().unwrap();
    let reduction = ir.operations().iter().find_map(|operation| {
        if let OperationAttributes::Reduction(ReductionAttributes {
            dimensions,
            keepdim,
        }) = operation.attributes()
        {
            Some((dimensions.clone(), *keepdim))
        } else {
            None
        }
    });
    assert_eq!(reduction, Some((vec![1], false)));

    let topk = ir.operations().iter().find_map(|operation| {
        if let OperationAttributes::TopK(TopKAttributes {
            k,
            dim,
            largest,
            sorted,
        }) = operation.attributes()
        {
            Some((*k, *dim, *largest, *sorted))
        } else {
            None
        }
    });
    assert_eq!(topk, Some((1, 1, true, true)));
}
