use crate::{
    autograd::{checkpoint, no_grad},
    backend::registry::{
        LayerNormalizationAttributes, Operation, OperationAttributes, Pool2dAttributes,
    },
    data_type::DataType,
    device::Device,
    dispatch::call_one,
    error::Result,
    shape::Shape,
    tensor::{AttentionOptions, Tensor},
};

fn tensor(dimensions: &[i64], values: &[f64]) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create(dimensions.to_vec()).unwrap(),
        values.to_vec(),
    )
    .unwrap()
}

fn leaf(dimensions: &[i64], values: &[f64]) -> Tensor {
    let leaf = tensor(dimensions, values);
    leaf.set_requires_grad(true).unwrap();
    leaf
}

fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.cast(DataType::F64).unwrap().to_vec::<f64>().unwrap()
}

fn tensor32(dimensions: &[i64], values: &[f32]) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create(dimensions.to_vec()).unwrap(),
        values.to_vec(),
    )
    .unwrap()
}

fn leaf32(dimensions: &[i64], values: &[f32]) -> Tensor {
    let leaf = tensor32(dimensions, values);
    leaf.set_requires_grad(true).unwrap();
    leaf
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
}

/// Returns the central-difference gradient of `function` at `point`.
fn numeric_gradient(
    point: &[f64],
    dimensions: &[i64],
    function: impl Fn(&Tensor) -> f64,
) -> Vec<f64> {
    let step = 1e-6;
    (0..point.len())
        .map(|index| {
            let mut plus = point.to_vec();
            let mut minus = point.to_vec();
            plus[index] += step;
            minus[index] -= step;
            (function(&tensor(dimensions, &plus)) - function(&tensor(dimensions, &minus)))
                / (2.0 * step)
        })
        .collect()
}

#[test]
fn test_gradients_of_broadcast_arithmetic_match_their_derivatives() {
    let x = leaf32(&[2, 3], &[1., -2., 3., -4., 5., -6.]);
    let b = leaf32(&[3], &[0.5, 0.5, 0.5]);
    // loss = sum(relu(x * 2 + b))
    let loss = x
        .mul_scalar(2.0)
        .unwrap()
        .add(&b)
        .unwrap()
        .relu()
        .unwrap()
        .sum(&[], false)
        .unwrap();
    assert!(loss.requires_grad());
    assert_eq!(loss.grad_fn_name(), Some("sum"));
    loss.backward().unwrap();
    assert_close(&values(&x.grad().unwrap()), &[2., 0., 2., 0., 2., 0.]);
    // Each column's bias reaches two rows, of which the positive ones pass gradient.
    assert_close(&values(&b.grad().unwrap()), &[1., 1., 1.]);

    // A second backward accumulates, as PyTorch does.
    let again = x.mul(&x).unwrap().sum(&[], false).unwrap();
    x.zero_grad();
    again.backward().unwrap();
    assert_close(&values(&x.grad().unwrap()), &[2., -4., 6., -8., 10., -12.]);
}

#[test]
fn test_matmul_and_division_gradients_match_finite_differences() {
    let point = [0.3, -1.2, 0.7, 2.0, -0.5, 1.1];
    let w = tensor(&[3, 2], &[0.2, -0.4, 1.5, 0.3, -0.7, 0.9]);
    let function = |x: &Tensor| -> Tensor {
        x.matmul(&w)
            .unwrap()
            .tanh()
            .unwrap()
            .div(
                &x.sum(&[1], true)
                    .unwrap()
                    .square()
                    .unwrap()
                    .add_scalar(1.0)
                    .unwrap(),
            )
            .unwrap()
            .exp()
            .unwrap()
            .mean(&[], false)
            .unwrap()
    };
    let x = leaf(&[2, 3], &point);
    function(&x).backward().unwrap();
    let expected = numeric_gradient(&point, &[2, 3], |x| values(&function(x))[0]);
    assert_close(&values(&x.grad().unwrap()), &expected);
}

#[test]
fn test_cross_entropy_gradient_is_softmax_minus_targets() {
    let logits = leaf32(&[2, 3], &[1., 2., 3., 0., -1., 2.]);
    let targets = tensor32(&[2, 3], &[0., 0., 1., 1., 0., 0.]);
    let loss = logits
        .log_softmax(1)
        .unwrap()
        .mul(&targets)
        .unwrap()
        .sum(&[], false)
        .unwrap()
        .neg()
        .unwrap();
    loss.backward().unwrap();
    let softmax = values(&logits.detach().softmax(1).unwrap());
    let expected = softmax
        .iter()
        .zip(values(&targets))
        .map(|(probability, target)| probability - target)
        .collect::<Vec<_>>();
    assert_close(&values(&logits.grad().unwrap()), &expected);
}

#[test]
fn test_writes_to_saved_tensors_and_to_leaves_are_rejected() {
    let x = leaf(&[2], &[1., 2.]);
    let scale = tensor(&[2], &[3., 4.]);
    let product = x.mul(&scale).unwrap().sum(&[], false).unwrap();
    scale.assign(&tensor(&[2], &[0., 0.])).unwrap();
    assert!(product.backward().is_err());
    assert!(x.assign(&tensor(&[2], &[0., 0.])).is_err());
}

#[test]
fn test_no_grad_and_detach_stop_recording() {
    let x = leaf(&[2], &[1., 2.]);
    {
        let _no_grad = no_grad();
        assert!(!x.exp().unwrap().requires_grad());
    }
    assert!(!x.detach().exp().unwrap().requires_grad());
    assert!(x.exp().unwrap().requires_grad());
    let integers =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1_i64, 2]).unwrap();
    assert!(integers.set_requires_grad(true).is_err());
}

#[test]
fn test_checkpoint_recomputes_and_matches_ordinary_gradients() {
    let block = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
        Ok(vec![inputs[0].tanh()?.mul(&inputs[1])?.sigmoid()?])
    };
    let point = [0.4_f32, -0.8, 1.6];
    let weights = [2.0_f32, -1.0, 0.5];

    let x = leaf32(&[3], &point);
    let w = leaf32(&[3], &weights);
    block(&[x.clone(), w.clone()]).unwrap()[0]
        .sum(&[], false)
        .unwrap()
        .backward()
        .unwrap();

    let checkpointed_x = leaf32(&[3], &point);
    let checkpointed_w = leaf32(&[3], &weights);
    let outputs = checkpoint(&[checkpointed_x.clone(), checkpointed_w.clone()], block).unwrap();
    assert_eq!(outputs[0].grad_fn_name(), Some("checkpoint"));
    outputs[0].sum(&[], false).unwrap().backward().unwrap();
    assert_close(
        &values(&checkpointed_x.grad().unwrap()),
        &values(&x.grad().unwrap()),
    );
    assert_close(
        &values(&checkpointed_w.grad().unwrap()),
        &values(&w.grad().unwrap()),
    );
}

// Checks the gradient of the weighted sum of `operation`'s output against central differences, in each operand that `checked` names.
fn check_operands(
    operands: &[(&[i64], Vec<f64>)],
    checked: &[usize],
    operation: impl Fn(&[Tensor]) -> Tensor,
) {
    let tensors = |point: &[f64], index: usize| -> Vec<Tensor> {
        operands
            .iter()
            .enumerate()
            .map(|(position, (dimensions, values))| {
                if position == index {
                    tensor(dimensions, point)
                } else {
                    tensor(dimensions, values)
                }
            })
            .collect()
    };
    let loss = |output: &Tensor| -> Tensor {
        // Every output point weighs differently, so no gradient cancels by symmetry.
        let count = i64::try_from(output.element_count()).unwrap();
        let weights = Tensor::from_vec(
            Device::Cpu,
            output.shape().clone(),
            (0..count)
                .map(|index| ((index * 7) % 11) as f64 / 5.0 - 1.0)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        output.mul(&weights).unwrap().sum(&[], false).unwrap()
    };
    for &index in checked {
        let (dimensions, point) = &operands[index];
        let mut inputs = tensors(point, index);
        inputs[index] = leaf(dimensions, point);
        loss(&operation(&inputs)).backward().unwrap();
        let expected = numeric_gradient(point, dimensions, |candidate| {
            let mut inputs = tensors(point, index);
            inputs[index] = candidate.clone();
            let _no_grad = no_grad();
            values(&loss(&operation(&inputs)))[0]
        });
        let actual = values(&inputs[index].grad().unwrap());
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "operand {index}: {actual} != {expected}"
            );
        }
    }
}

fn samples(count: usize, seed: usize) -> Vec<f64> {
    (0..count)
        .map(|index| (((index + seed) * 37 + seed * 11) % 53) as f64 / 26.0 - 0.97)
        .collect()
}

fn normalization(operation: Operation, inputs: &[Tensor], size: i64) -> Tensor {
    call_one(
        operation,
        OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
            normalized_shape: vec![size],
            has_weight: true,
            has_bias: inputs.len() > 2,
            eps: 1e-5,
        }),
        &inputs.iter().collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn test_normalization_gradients_match_finite_differences() {
    check_operands(
        &[(&[2, 4], samples(8, 1)), (&[4], samples(4, 2))],
        &[0, 1],
        |inputs| normalization(Operation::RmsNorm, inputs, 4),
    );
    check_operands(
        &[
            (&[2, 4], samples(8, 1)),
            (&[4], samples(4, 2)),
            (&[4], samples(4, 3)),
        ],
        &[0, 1, 2],
        |inputs| normalization(Operation::LayerNorm, inputs, 4),
    );
    let batch = [
        (&[3_i64, 2, 2][..], samples(12, 1)),
        (&[2][..], samples(2, 2)),
        (&[2][..], samples(2, 3)),
        (&[2][..], samples(2, 4)),
        (&[2][..], vec![0.7, 1.3]),
    ];
    check_operands(&batch, &[0, 1, 2, 3, 4], |inputs| {
        inputs[0]
            .batch_norm(
                Some(&inputs[1]),
                Some(&inputs[2]),
                Some(&inputs[3]),
                Some(&inputs[4]),
                false,
                0.1,
                1e-5,
            )
            .unwrap()
    });
    check_operands(
        &[(&[3, 2, 2], samples(12, 1)), (&[2], samples(2, 2))],
        &[0, 1],
        |inputs| {
            inputs[0]
                .batch_norm(Some(&inputs[1]), None, None, None, true, 0.1, 1e-5)
                .unwrap()
        },
    );
}

#[test]
fn test_attention_gradients_match_finite_differences() {
    for (key_heads, causal) in [(2_i64, true), (1, false)] {
        check_operands(
            &[
                (&[1, 2, 2, 4], samples(16, 1)),
                (
                    &[1, key_heads, 3, 4],
                    samples(usize::try_from(key_heads * 12).unwrap(), 2),
                ),
                (
                    &[1, key_heads, 3, 4],
                    samples(usize::try_from(key_heads * 12).unwrap(), 3),
                ),
            ],
            &[0, 1, 2],
            |inputs| {
                inputs[0]
                    .attend(
                        &inputs[1],
                        &inputs[2],
                        None,
                        None,
                        AttentionOptions {
                            causal,
                            ..AttentionOptions::default()
                        },
                    )
                    .unwrap()
            },
        );
    }
}

#[test]
fn test_convolution_gradients_match_finite_differences() {
    for (stride, padding, dilation, groups) in [
        ((1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((2, 1), (1, 0, 1, 2), (1, 2), 1),
        ((1, 2), (1, 1, 0, 0), (1, 1), 2),
    ] {
        check_operands(
            &[
                (&[1, 2, 4, 5], samples(40, 1)),
                (
                    &[4, 2 / groups, 2, 2],
                    samples(usize::try_from(16 * (2 / groups)).unwrap(), 2),
                ),
                (&[4], samples(4, 3)),
            ],
            &[0, 1, 2],
            |inputs| {
                inputs[0]
                    .convolution2d(
                        &inputs[1],
                        Some(&inputs[2]),
                        stride,
                        padding,
                        dilation,
                        groups,
                    )
                    .unwrap()
            },
        );
    }
    check_operands(
        &[
            (&[1, 1, 3, 3, 3], samples(27, 1)),
            (&[2, 1, 2, 2, 2], samples(16, 2)),
        ],
        &[0, 1],
        |inputs| {
            inputs[0]
                .convolution3d(&inputs[1], None, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1)
                .unwrap()
        },
    );
}

#[test]
fn test_pooling_gradients_match_finite_differences() {
    let pool = |operation: Operation, attributes: Pool2dAttributes, input: &Tensor| {
        call_one(operation, OperationAttributes::Pool2d(attributes), &[input]).unwrap()
    };
    let window = |padding, ceil_mode, count_include_pad| Pool2dAttributes {
        kernel_size: (2, 3),
        stride: (2, 2),
        padding,
        dilation: (1, 1),
        ceil_mode,
        count_include_pad,
    };
    // Values with no ties in any window, so the maximum's gradient is one point's.
    let distinct = (0..30)
        .map(|index| f64::from((index * 17) % 31) / 7.0 - 2.0)
        .collect::<Vec<_>>();
    for attributes in [
        window((0, 0, 0, 0), false, Some(true)),
        window((1, 1, 1, 1), false, Some(true)),
        window((1, 1, 1, 1), false, Some(false)),
        window((0, 0, 0, 0), true, Some(true)),
        window((0, 0, 1, 1), true, Some(false)),
    ] {
        check_operands(&[(&[1, 2, 3, 5], distinct.clone())], &[0], |inputs| {
            pool(Operation::AvgPool2d, attributes, &inputs[0])
        });
    }
    for attributes in [
        window((0, 0, 0, 0), false, None),
        window((1, 1, 1, 1), false, None),
        window((0, 0, 0, 0), true, None),
    ] {
        check_operands(&[(&[1, 2, 3, 5], distinct.clone())], &[0], |inputs| {
            pool(Operation::MaxPool2d, attributes, &inputs[0])
        });
    }
}
