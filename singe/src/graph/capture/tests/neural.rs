use super::*;

#[test]
fn test_graph_capture_records_tensor_scaled_dot_product_attention() {
    let query_shape = Shape::create([1, 1, 2, 2]).unwrap();
    let key_shape = Shape::create([1, 1, 3, 2]).unwrap();
    let query = Tensor::from_vec(Device::Cpu, &query_shape, vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap();
    let key = Tensor::from_vec(
        Device::Cpu,
        &key_shape,
        vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let value = Tensor::from_vec(
        Device::Cpu,
        &key_shape,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let (captured, captured_outputs) = Graph::capture(
        [("query", query), ("key", key), ("value", value)],
        |inputs| {
            Ok(vec![inputs[0].scaled_dot_product_attention(
                &inputs[1],
                &inputs[2],
                Some(0.5),
                true,
            )?])
        },
    )
    .unwrap();

    assert_close!(
        &captured_outputs[0].to_vec::<f32>().unwrap(),
        &[1.0, 2.0, 2.244_918_6, 3.244_918_6],
        1e-5,
    );

    let attention = captured
        .to_ir()
        .unwrap()
        .operations()
        .iter()
        .find_map(|operation| {
            (operation.opcode() == Some(Operation::ScaledDotProductAttention))
                .then(|| {
                    operation
                        .attributes()
                        .scaled_dot_product_attention()
                        .copied()
                })
                .flatten()
        });
    assert_eq!(
        attention.map(|attributes| (attributes.scale, attributes.causal)),
        Some((Some(0.5), true)),
    );
}

#[test]
fn test_graph_capture_records_tensor_embedding() {
    let weight = Tensor::from_vec(
        Device::Cpu,
        Shape::create(vec![2, 2]).unwrap(),
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let indices =
        Tensor::from_vec(Device::Cpu, Shape::create(vec![2]).unwrap(), vec![1_i64, 0]).unwrap();

    let (captured, captured_outputs) = Graph::capture(
        [("embedding_weight", weight), ("embedding_indices", indices)],
        |inputs| Ok(vec![inputs[0].embedding(&inputs[1])?]),
    )
    .unwrap();

    assert_close!(
        &captured_outputs[0].to_vec::<f32>().unwrap(),
        &[3.0, 4.0, 1.0, 2.0],
        1e-6,
    );

    let ir = captured.to_ir().unwrap();
    assert!(
        ir.operations()
            .iter()
            .any(|operation| operation.opcode() == Some(Operation::Embedding))
    );
}

#[test]
fn test_graph_capture_records_tensor_batch_norm() {
    let image = Tensor::from_vec(
        Device::Cpu,
        Shape::create(vec![1, 1, 2, 2]).unwrap(),
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let weight =
        Tensor::from_vec(Device::Cpu, Shape::create(vec![1]).unwrap(), vec![1.5_f32]).unwrap();
    let bias =
        Tensor::from_vec(Device::Cpu, Shape::create(vec![1]).unwrap(), vec![0.25_f32]).unwrap();
    let running_mean =
        Tensor::from_vec(Device::Cpu, Shape::create(vec![1]).unwrap(), vec![1.0_f32]).unwrap();
    let running_var =
        Tensor::from_vec(Device::Cpu, Shape::create(vec![1]).unwrap(), vec![4.0_f32]).unwrap();

    let (captured, _) = Graph::capture(
        [
            ("image", image),
            ("weight", weight),
            ("bias", bias),
            ("running_mean", running_mean),
            ("running_var", running_var),
        ],
        |inputs| {
            Ok(vec![inputs[0].batch_norm(
                Some(&inputs[1]),
                Some(&inputs[2]),
                Some(&inputs[3]),
                Some(&inputs[4]),
                false,
                0.1,
                1e-5,
            )?])
        },
    )
    .unwrap();

    let batch_norm = captured
        .to_ir()
        .unwrap()
        .operations()
        .iter()
        .find_map(|operation| {
            (operation.opcode() == Some(Operation::BatchNorm))
                .then(|| operation.attributes().batch_normalization().copied())
                .flatten()
        });
    assert_eq!(
        batch_norm.map(|attributes| {
            (
                attributes.has_weight,
                attributes.has_bias,
                attributes.has_running_mean,
                attributes.has_running_var,
                attributes.training,
            )
        }),
        Some((true, true, true, true, false)),
    );
}

#[test]
fn test_graph_capture_records_tensor_spatial_neural_ops() {
    let image = Tensor::from_vec(
        Device::Cpu,
        Shape::create(vec![1, 1, 2, 2]).unwrap(),
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let conv_weight = Tensor::from_vec(
        Device::Cpu,
        Shape::create(vec![1, 1, 1, 1]).unwrap(),
        vec![2.0_f32],
    )
    .unwrap();
    let conv_bias =
        Tensor::from_vec(Device::Cpu, Shape::create(vec![1]).unwrap(), vec![1.0_f32]).unwrap();

    let (captured, captured_outputs) = Graph::capture(
        [
            ("image", image),
            ("conv_weight", conv_weight),
            ("conv_bias", conv_bias),
        ],
        |inputs| {
            let conv = inputs[0].convolution2d(
                &inputs[1],
                Some(&inputs[2]),
                (1, 1),
                (0, 0, 0, 0),
                (1, 1),
                1,
            )?;
            let pool = inputs[0].max_pool2d((2, 2), (1, 1), (0, 0, 0, 0), (1, 1), false)?;
            Ok(vec![conv, pool])
        },
    )
    .unwrap();

    assert_close!(
        &captured_outputs[0].to_vec::<f32>().unwrap(),
        &[3.0, 5.0, 7.0, 9.0],
        1e-6,
    );
    assert_close!(&captured_outputs[1].to_vec::<f32>().unwrap(), &[4.0], 1e-6);

    let ir = captured.to_ir().unwrap();
    let conv = ir.operations().iter().find_map(|operation| {
        (operation.opcode() == Some(Operation::Conv2d))
            .then(|| operation.attributes().conv2d().copied())
            .flatten()
    });
    assert_eq!(conv.map(|attributes| attributes.has_bias), Some(true));

    let pool = ir.operations().iter().find_map(|operation| {
        (operation.opcode() == Some(Operation::MaxPool2d))
            .then(|| operation.attributes().pool2d().copied())
            .flatten()
    });
    assert_eq!(
        pool.map(|attributes| {
            (
                attributes.kernel_size,
                attributes.stride,
                attributes.count_include_pad,
            )
        }),
        Some(((2, 2), (1, 1), None)),
    );
}
