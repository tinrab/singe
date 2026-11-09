use super::*;

#[test]
fn test_graph_capture_records_tensor_concat() {
    let shape = Shape::create(vec![2]).unwrap();
    let a = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap();
    let b = Tensor::from_vec(Device::Cpu, &shape, vec![3.0_f32, 4.0]).unwrap();

    let (captured, captured_outputs) = Graph::capture([("a", a), ("b", b)], |inputs| {
        Tensor::concat(&[inputs[0].clone(), inputs[1].clone()], 0).map(|output| vec![output])
    })
    .unwrap();

    assert_close!(
        &captured_outputs[0].to_vec::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
        1e-6,
    );
    let ir = captured.to_ir().unwrap();
    let concat_dim = ir.operations().iter().find_map(|operation| {
        if let OperationAttributes::Concat(dim) = operation.attributes() {
            Some(*dim)
        } else {
            None
        }
    });
    assert_eq!(concat_dim, Some(0));
}

#[test]
fn test_graph_capture_records_tensor_contraction() {
    let a_shape = Shape::create([2, 3]).unwrap();
    let b_shape = Shape::create([3, 2]).unwrap();
    let c_shape = Shape::create([2, 2]).unwrap();
    let a = Tensor::from_vec(
        Device::Cpu,
        &a_shape,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let b = Tensor::from_vec(
        Device::Cpu,
        &b_shape,
        vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();
    let c = Tensor::zeroed_on(Device::Cpu, DataType::F32, &c_shape).unwrap();

    let (captured, captured_outputs) = Graph::capture([("a", a), ("b", b), ("c", c)], |inputs| {
        Ok(vec![inputs[0].contraction(
            &[0, 2],
            &inputs[1],
            &[2, 1],
            &inputs[2],
            &[0, 1],
            &[0, 1],
            1.0,
            0.0,
        )?])
    })
    .unwrap();

    assert_close!(
        &captured_outputs[0].to_vec::<f32>().unwrap(),
        &[58.0, 64.0, 139.0, 154.0],
        1e-5,
    );
    let ir = captured.to_ir().unwrap();
    let contraction = ir.operations().iter().find_map(|operation| {
        if let OperationAttributes::Contraction(ContractionAttributes {
            a_mode,
            b_mode,
            c_mode,
            d_mode,
            alpha,
            beta,
        }) = &operation.attributes()
        {
            Some((a_mode, b_mode, c_mode, d_mode, alpha, beta))
        } else {
            None
        }
    });
    assert_eq!(
        contraction,
        Some((
            &vec![0, 2],
            &vec![2, 1],
            &vec![0, 1],
            &vec![0, 1],
            &1.0,
            &0.0,
        ))
    );
}

#[test]
fn test_graph_capture_records_eager_tensor_ops() {
    let shape = Shape::create(vec![2]).unwrap();
    let a = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, -6.0]).unwrap();
    let b = Tensor::from_vec(Device::Cpu, &shape, vec![3.0_f32, 2.0]).unwrap();

    let (captured, captured_outputs) =
        Graph::capture([("a", a), ("b", b)], |inputs| -> Result<Vec<Tensor>> {
            let sum = inputs[0].add(&inputs[1])?;
            let scaled = sum.mul_scalar(0.5)?;
            let output = scaled.relu()?;
            Ok(vec![output])
        })
        .unwrap();

    let ir = captured.to_ir().unwrap();
    assert_eq!(ir.values().len(), 5);
    assert_eq!(ir.output_values().len(), 1);
    assert_eq!(captured_outputs.len(), 1);

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![2.0_f32, -2.0]).unwrap(),
    );
    inputs.insert(
        "b".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![4.0_f32, 1.0]).unwrap(),
    );

    let outputs = captured.run(&inputs).unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[3.0, 0.0], 1e-6);
}

#[test]
fn test_graph_fingerprint_matches_capture_and_manual_graph() {
    let shape = Shape::create(vec![2]).unwrap();
    let captured_input = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap();
    let (captured, _) = Graph::capture([("x", captured_input)], |inputs| -> Result<Vec<Tensor>> {
        Ok(vec![inputs[0].mul_scalar(2.0)?])
    })
    .unwrap();

    let mut manual = Graph::new();
    let input = manual.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let scaled = add_scalar_operation(&mut manual, RegistryOperation::Scale, input, 2.0, &output);
    manual.set_outputs(vec![scaled]).unwrap();

    assert_eq!(
        captured.fingerprint().unwrap(),
        manual.fingerprint().unwrap()
    );
}
