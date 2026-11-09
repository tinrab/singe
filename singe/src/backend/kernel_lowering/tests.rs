use std::slice;

use singe_codegen::interpret;

use crate::{
    backend::{
        kernel_lowering::{
            ElementGroup, ElementWiseKernel, EpilogueNode, EpilogueValue, FusedOperand, FusedStep,
            Lhs, LoweredKernel, RowEpilogue, STATUS_INDEX_OUT_OF_BOUNDS,
            STATUS_INVALID_QUANTIZATION_SCALE, Selection, Weight, batch_norm, contraction, conv2d,
            gather, normalization, normalization_with_epilogue, pool2d, quantization, reduction,
            scan, scatter, softmax,
        },
        registry::{
            BackendExecution, BatchNormalizationAttributes, ClampAttributes, Conv2dAttributes,
            LayerNormalizationAttributes, Operation, OperationAttributes, OperationRequest,
            PackedFormat, Pool2dAttributes, QuantizationAttributes, ReductionAttributes,
            operation_support,
        },
    },
    data_type::DataType,
    device::Device,
    dispatch::call_one,
    layout::Layout,
    shape::Shape,
    tensor::Tensor,
    value::ValueSpec,
};

fn sample(dimensions: &[i64], seed: usize, data_type: DataType) -> Tensor {
    let count = usize::try_from(dimensions.iter().product::<i64>()).unwrap();
    let values = (0..count)
        .map(|index| 0.05 + ((index * 37 + seed * 11) % 97) as f64 / 50.0)
        .collect::<Vec<_>>();
    let tensor = Tensor::from_vec(Device::Cpu, Shape::create(dimensions).unwrap(), values).unwrap();
    if data_type == DataType::Bool {
        tensor
            .greater(&Tensor::scalar(Device::Cpu, 1.0_f64).unwrap())
            .unwrap()
    } else {
        tensor.cast(data_type).unwrap()
    }
}

/// Lowers `operation` on `inputs`, runs the kernel on the interpreter, and asserts that it matches the CPU backend within `tolerance`.
fn assert_matches_cpu(
    operation: Operation,
    attributes: &OperationAttributes,
    inputs: &[Tensor],
    tolerance: f64,
) {
    let name = format!("{} {:?}", operation.name(), inputs[0].data_type());
    let reference_type = match inputs[0].data_type() {
        DataType::Bf16 => DataType::F32,
        other => other,
    };
    if !operation_support(operation)
        .supports_input_data_type(BackendExecution::CpuEager, reference_type)
    {
        return;
    }
    // The CPU backend computes half-precision values in F32, as the kernels do, and rounds to the inferred output type.
    let output = OperationRequest::new(
        operation,
        attributes.clone(),
        inputs
            .iter()
            .map(|input| ValueSpec::new(input.data_type(), input.shape().clone()))
            .collect(),
    )
    .infer()
    .unwrap()
    .outputs()[0]
        .data_type();
    let references = inputs
        .iter()
        .map(|input| {
            if input.data_type() == DataType::Bf16 {
                input.cast(DataType::F32).unwrap()
            } else {
                input.clone()
            }
        })
        .collect::<Vec<_>>();
    let expected = call_one(
        operation,
        attributes.clone(),
        &references.iter().collect::<Vec<_>>(),
    )
    .unwrap()
    .cast(output)
    .unwrap()
    .contiguous()
    .unwrap();
    let layouts = inputs
        .iter()
        .map(|input| (input.data_type(), input.layout()))
        .collect::<Vec<_>>();
    let (mut kernel, values) =
        ElementWiseKernel::create("singe_test", expected.shape(), &layouts).unwrap();
    let value = kernel.apply(operation, attributes, &values).unwrap();
    kernel.store(value, expected.data_type()).unwrap();
    let (kernel, bindings, _) = kernel.finish();
    // The only strided inputs here are transposes of contiguous tensors, whose storage is the transpose undone.
    let mut buffers = inputs
        .iter()
        .map(|input| {
            if input.layout().is_contiguous() {
                input.to_bytes().unwrap()
            } else {
                input.transpose(0, 1).unwrap().to_bytes().unwrap()
            }
        })
        .collect::<Vec<_>>();
    buffers.push(vec![0; expected.to_bytes().unwrap().len()]);
    interpret::run(&kernel, &bindings, &mut buffers).unwrap();
    let actual = Tensor::from_bytes(
        Device::Cpu,
        expected.data_type(),
        expected.shape().clone(),
        buffers.last().unwrap(),
    )
    .unwrap();
    let [actual, expected] = [&actual, &expected]
        .map(|tensor| tensor.cast(DataType::F64).unwrap().to_vec::<f64>().unwrap());
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        let close = actual == expected
            || (actual.is_nan() && expected.is_nan())
            || (actual - expected).abs() <= tolerance * expected.abs().max(1.0);
        assert!(
            close,
            "{name}: element {index} is {actual}, the CPU gives {expected}"
        );
    }
}

fn tolerance(data_type: DataType) -> f64 {
    match data_type {
        DataType::F64 => 1e-12,
        DataType::Bf16 => 1e-2,
        _ => 1e-6,
    }
}

#[test]
fn test_lowered_element_wise_operations_match_the_cpu() {
    use Operation as Op;
    let unary = [
        Op::Neg,
        Op::Abs,
        Op::Sqrt,
        Op::Exp,
        Op::Log,
        Op::Sin,
        Op::Cos,
        Op::Tan,
        Op::Sinh,
        Op::Cosh,
        Op::Tanh,
        Op::Asin,
        Op::Acos,
        Op::Atan,
        Op::Ceil,
        Op::Floor,
        Op::Truncate,
        Op::Asinh,
        Op::Acosh,
        Op::Atanh,
        Op::Erf,
        Op::Reciprocal,
        Op::Square,
        Op::Rsqrt,
        Op::Relu,
        Op::GeluExact,
        Op::Gelu,
        Op::Silu,
        Op::Sigmoid,
    ];
    let binary = [
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Modulo,
        Op::Remainder,
        Op::Atan2,
        Op::Maximum,
        Op::Minimum,
        Op::SwiGlu,
        Op::Equal,
        Op::NotEqual,
        Op::Less,
        Op::LessEqual,
        Op::Greater,
        Op::GreaterEqual,
    ];
    let scalar = [
        Op::AddScalar,
        Op::SubScalar,
        Op::RSubScalar,
        Op::Scale,
        Op::RDivScalar,
    ];
    for data_type in [DataType::F32, DataType::Bf16, DataType::F64] {
        let tolerance = tolerance(data_type);
        let matrix = |seed| sample(&[3, 4], seed, data_type);
        for operation in unary {
            assert_matches_cpu(
                operation,
                &OperationAttributes::None,
                &[matrix(1)],
                tolerance,
            );
        }
        for operation in binary {
            // The second operand broadcasts along the rows.
            let inputs = [matrix(1), sample(&[4], 2, data_type)];
            assert_matches_cpu(operation, &OperationAttributes::None, &inputs, tolerance);
        }
        for operation in scalar {
            let attributes = OperationAttributes::Scalar(0.75_f64.into());
            assert_matches_cpu(operation, &attributes, &[matrix(1)], tolerance);
        }
        let clamp = OperationAttributes::Clamp(ClampAttributes { min: 0.4, max: 1.3 });
        assert_matches_cpu(Op::Clamp, &clamp, &[matrix(1)], tolerance);
        let lerp = OperationAttributes::Lerp(0.25);
        assert_matches_cpu(Op::Lerp, &lerp, &[matrix(1), matrix(2)], tolerance);
        let condition = sample(&[3, 4], 3, DataType::Bool);
        assert_matches_cpu(
            Op::Where,
            &OperationAttributes::None,
            &[condition, matrix(1), matrix(2)],
            tolerance,
        );
        for target in [DataType::F64, DataType::F32, DataType::Bool] {
            assert_matches_cpu(
                Op::Cast,
                &OperationAttributes::Cast(target),
                &[matrix(1)],
                tolerance,
            );
        }
        // A transposed operand reads through its strides.
        let transposed = matrix(4).transpose(0, 1).unwrap();
        assert_matches_cpu(
            Op::Add,
            &OperationAttributes::None,
            &[transposed, sample(&[4, 3], 5, data_type)],
            tolerance,
        );
    }
    let flags = |seed| sample(&[3, 4], seed, DataType::Bool);
    for operation in [Op::LogicalAnd, Op::LogicalOr] {
        assert_matches_cpu(
            operation,
            &OperationAttributes::None,
            &[flags(1), flags(2)],
            0.0,
        );
    }
    assert_matches_cpu(Op::LogicalNot, &OperationAttributes::None, &[flags(1)], 0.0);
}

#[test]
fn test_contiguous_dimensions_merge_into_one_axis_whose_extent_is_a_launch_argument() {
    let matrix = Layout::contiguous([6, 8]).unwrap();
    let row = Layout::contiguous([8]).unwrap();
    let (kernel, _) = ElementWiseKernel::create(
        "singe_test",
        &Shape::create([6, 8]).unwrap(),
        &[(DataType::F32, &matrix), (DataType::F32, &matrix)],
    )
    .unwrap();
    let (kernel, bindings, labels) = kernel.finish();
    assert_eq!(kernel.axes.len(), 1);
    assert_eq!(bindings.sizes, [48]);
    assert_eq!(labels.sizes, ["outermost extent"]);

    // A broadcast row keeps the row axis separate.
    let (kernel, _) = ElementWiseKernel::create(
        "singe_test",
        &Shape::create([6, 8]).unwrap(),
        &[(DataType::F32, &matrix), (DataType::F32, &row)],
    )
    .unwrap();
    let (kernel, bindings, labels) = kernel.finish();
    assert_eq!(kernel.axes.len(), 2);
    assert_eq!(bindings.sizes, [6]);
    assert_eq!(labels.sizes, ["outermost extent"]);
}

/// Runs `lowered` on the interpreter with the storage of `inputs` and asserts that it matches `expected` within `tolerance`.
fn assert_lowered_matches(
    lowered: &LoweredKernel,
    inputs: &[Tensor],
    expected: &Tensor,
    tolerance: f64,
) {
    let name = lowered.kernel.name.clone();
    let mut buffers = inputs
        .iter()
        .map(|input| input.to_bytes().unwrap())
        .collect::<Vec<_>>();
    buffers.push(vec![0; expected.to_bytes().unwrap().len()]);
    interpret::run(&lowered.kernel, &lowered.bindings, &mut buffers).unwrap();
    assert_eq!(
        lowered.outputs[0].shape(),
        expected.shape(),
        "{name}: shape"
    );
    let actual = Tensor::from_bytes(
        Device::Cpu,
        expected.data_type(),
        expected.shape().clone(),
        buffers.last().unwrap(),
    )
    .unwrap();
    let [actual, expected] = [&actual, expected]
        .map(|tensor| tensor.cast(DataType::F64).unwrap().to_vec::<f64>().unwrap());
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        let close = actual == expected
            || (actual.is_nan() && expected.is_nan())
            || (actual - expected).abs() <= tolerance * expected.abs().max(1.0);
        assert!(
            close,
            "{name}: element {index} is {actual}, the CPU gives {expected}"
        );
    }
}

#[test]
fn test_lowered_reductions_softmax_and_normalizations_match_the_cpu() {
    for data_type in [DataType::F32, DataType::F64] {
        let tolerance = tolerance(data_type) * 10.0;
        let input = sample(&[2, 3, 5], 1, data_type);
        for operation in [
            Operation::Sum,
            Operation::Mean,
            Operation::Max,
            Operation::Min,
            Operation::Product,
        ] {
            for (dimensions, keepdim) in [
                (vec![2], false),
                (vec![0, 2], true),
                (vec![], false),
                (vec![1], false),
            ] {
                let attributes = OperationAttributes::Reduction(ReductionAttributes {
                    dimensions: dimensions.clone(),
                    keepdim,
                });
                let expected = call_one(operation, attributes, &[&input]).unwrap();
                let group = ElementGroup::input(data_type, input.layout());
                let lowered = reduction(operation, &group, &dimensions, keepdim, None).unwrap();
                assert_lowered_matches(&lowered, slice::from_ref(&input), &expected, tolerance);
            }
        }
        let cpu_supports = |operation| {
            operation_support(operation)
                .supports_input_data_type(BackendExecution::CpuEager, data_type)
        };
        for operation in [Operation::Softmax, Operation::LogSoftmax] {
            if !cpu_supports(operation) {
                continue;
            }
            for dim in [-1, 1] {
                let expected =
                    call_one(operation, OperationAttributes::Dimension(dim), &[&input]).unwrap();
                let group = ElementGroup::input(data_type, input.layout());
                let lowered = softmax(operation, &group, dim, None).unwrap();
                assert_lowered_matches(&lowered, slice::from_ref(&input), &expected, tolerance);
            }
        }
        for (operation, has_bias) in [(Operation::LayerNorm, true), (Operation::RmsNorm, false)] {
            if !cpu_supports(operation) {
                continue;
            }
            let attributes = LayerNormalizationAttributes {
                normalized_shape: vec![3, 5],
                has_weight: true,
                has_bias,
                eps: 1e-5,
            };
            let mut inputs = vec![input.clone(), sample(&[3, 5], 2, data_type)];
            if has_bias {
                inputs.push(sample(&[3, 5], 3, data_type));
            }
            let expected = call_one(
                operation,
                OperationAttributes::LayerNormalization(attributes.clone()),
                &inputs.iter().collect::<Vec<_>>(),
            )
            .unwrap();
            let layouts = inputs
                .iter()
                .map(|input| (input.data_type(), input.layout()))
                .collect::<Vec<_>>();
            let group = ElementGroup::input(data_type, input.layout());
            let lowered = normalization(operation, &attributes, &group, &layouts[1..]).unwrap();
            assert_lowered_matches(&lowered, &inputs, &expected, tolerance);
        }
        if !cpu_supports(Operation::BatchNorm) {
            continue;
        }
        let attributes = BatchNormalizationAttributes {
            has_weight: true,
            has_bias: true,
            has_running_mean: true,
            has_running_var: true,
            training: false,
            momentum: 0.1,
            eps: 1e-5,
        };
        let inputs = [
            input.clone(),
            sample(&[3], 2, data_type),
            sample(&[3], 3, data_type),
            sample(&[3], 4, data_type),
            sample(&[3], 5, data_type),
        ];
        let expected = call_one(
            Operation::BatchNorm,
            OperationAttributes::BatchNormalization(attributes),
            &inputs.iter().collect::<Vec<_>>(),
        )
        .unwrap();
        let layouts = inputs
            .iter()
            .map(|input| (input.data_type(), input.layout()))
            .collect::<Vec<_>>();
        let lowered = batch_norm(&attributes, &layouts).unwrap();
        assert_lowered_matches(&lowered, &inputs, &expected, tolerance);
    }
}

#[test]
fn test_row_lowerings_compute_an_element_wise_prologue_in_the_kernel() {
    let data_type = DataType::F32;
    let x = sample(&[2, 3, 5], 1, data_type);
    let y = sample(&[2, 3, 5], 2, data_type);
    // The mask broadcasts over the leading dimension, as an attention mask does over heads.
    let mask = sample(&[3, 5], 3, data_type);
    let none = OperationAttributes::None;
    let steps = [
        FusedStep {
            operation: Operation::Mul,
            attributes: &none,
            operands: vec![FusedOperand::Input(0), FusedOperand::Input(1)],
        },
        FusedStep {
            operation: Operation::Add,
            attributes: &none,
            operands: vec![FusedOperand::Step(0), FusedOperand::Input(2)],
        },
    ];
    let group = ElementGroup {
        data_type,
        shape: x.shape().clone(),
        inputs: [&x, &y, &mask]
            .iter()
            .map(|input| (input.data_type(), input.layout()))
            .collect(),
        steps: &steps,
        stored: false,
    };
    let value = x.mul(&y).unwrap().add(&mask).unwrap();
    let inputs = [x.clone(), y.clone(), mask.clone()];
    let tolerance = tolerance(data_type) * 10.0;

    let expected = call_one(
        Operation::Softmax,
        OperationAttributes::Dimension(-1),
        &[&value],
    )
    .unwrap();
    let lowered = softmax(Operation::Softmax, &group, -1, None).unwrap();
    assert_eq!(lowered.kernel.name, "singe_fused_mul_add_softmax");
    assert_lowered_matches(&lowered, &inputs, &expected, tolerance);

    let attributes = LayerNormalizationAttributes {
        normalized_shape: vec![5],
        has_weight: true,
        has_bias: false,
        eps: 1e-5,
    };
    let weight = sample(&[5], 4, data_type);
    let expected = call_one(
        Operation::RmsNorm,
        OperationAttributes::LayerNormalization(attributes.clone()),
        &[&value, &weight],
    )
    .unwrap();
    let lowered = normalization(
        Operation::RmsNorm,
        &attributes,
        &group,
        &[(data_type, weight.layout())],
    )
    .unwrap();
    let mut inputs = inputs.to_vec();
    inputs.push(weight.clone());
    assert_lowered_matches(&lowered, &inputs, &expected, tolerance);

    // A stored group is the kernel's second output, after the normalization's.
    let stored = ElementGroup {
        stored: true,
        ..group.clone()
    };
    let lowered = normalization(
        Operation::RmsNorm,
        &attributes,
        &stored,
        &[(data_type, weight.layout())],
    )
    .unwrap();
    assert_eq!(lowered.outputs.len(), 2);
    let mut buffers = inputs
        .iter()
        .map(|input| input.to_bytes().unwrap())
        .collect::<Vec<_>>();
    buffers.extend([vec![0; 120], vec![0; 120]]);
    interpret::run(&lowered.kernel, &lowered.bindings, &mut buffers).unwrap();
    assert_eq!(buffers[4], expected.to_bytes().unwrap());
    assert_eq!(buffers[5], value.to_bytes().unwrap());

    let attributes = ReductionAttributes {
        dimensions: vec![1],
        keepdim: false,
    };
    let expected = call_one(
        Operation::Sum,
        OperationAttributes::Reduction(attributes.clone()),
        &[&value],
    )
    .unwrap();
    let lowered = reduction(Operation::Sum, &group, &attributes.dimensions, false, None).unwrap();
    assert_lowered_matches(&lowered, &inputs[..3], &expected, tolerance);
}

#[test]
fn test_lowered_gathers_and_scatters_check_their_indices() {
    let input = sample(&[3, 4, 2], 1, DataType::F32);
    let rows =
        Tensor::from_vec(Device::Cpu, Shape::create([3]).unwrap(), vec![2_i64, -3, 1]).unwrap();
    let columns = Tensor::from_vec(
        Device::Cpu,
        Shape::create([1, 3]).unwrap(),
        vec![0_i64, 3, -1],
    )
    .unwrap();
    let index_shape = Shape::create([1, 3]).unwrap();
    let indices = [
        (DataType::I64, rows.layout()),
        (DataType::I64, columns.layout()),
    ];
    let selection = Selection {
        axis: 0,
        extents: &[3, 4],
        index_shape: &index_shape,
        indices: &indices,
        wrap: true,
    };
    let lowered = gather("singe_index", DataType::F32, input.layout(), &selection).unwrap();
    let values = input.to_vec::<f32>().unwrap();
    let expected = Tensor::from_vec(
        Device::Cpu,
        Shape::create([1, 3, 2]).unwrap(),
        [(2, 0), (0, 3), (1, 3)]
            .iter()
            .flat_map(|&(row, column)| {
                [
                    values[row * 8 + column * 2],
                    values[row * 8 + column * 2 + 1],
                ]
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let inputs = [input.clone(), rows.clone(), columns.clone()];
    assert_lowered_matches(&lowered, &inputs, &expected, 0.0);

    // An index past its dimension reports the status code.
    let outside =
        Tensor::from_vec(Device::Cpu, Shape::create([3]).unwrap(), vec![0_i64, 3, 1]).unwrap();
    let indices = [(DataType::I64, outside.layout()), indices[1]];
    let selection = Selection {
        indices: &indices,
        ..selection
    };
    let lowered = gather("singe_index", DataType::F32, input.layout(), &selection).unwrap();
    let mut buffers = vec![
        input.to_bytes().unwrap(),
        outside.to_bytes().unwrap(),
        columns.to_bytes().unwrap(),
        vec![0; 24],
    ];
    assert_eq!(
        interpret::run(&lowered.kernel, &lowered.bindings, &mut buffers).unwrap(),
        STATUS_INDEX_OUT_OF_BOUNDS
    );

    // Accumulating scatters add every value that lands on one element.
    let repeated =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1_i64, 1]).unwrap();
    let index_shape = Shape::create([2]).unwrap();
    let indices = [(DataType::I64, repeated.layout())];
    let selection = Selection {
        axis: 0,
        extents: &[3],
        index_shape: &index_shape,
        indices: &indices,
        wrap: true,
    };
    let values = sample(&[2, 4, 2], 2, DataType::F32);
    let lowered = scatter(
        "singe_index_put",
        ValueSpec::new(DataType::F32, input.shape().clone()),
        values.layout(),
        &selection,
        true,
    )
    .unwrap();
    let mut buffers = vec![
        values.to_bytes().unwrap(),
        repeated.to_bytes().unwrap(),
        input.to_bytes().unwrap(),
    ];
    assert_eq!(
        interpret::run(&lowered.kernel, &lowered.bindings, &mut buffers).unwrap(),
        0
    );
    let actual = Tensor::from_bytes(
        Device::Cpu,
        DataType::F32,
        input.shape().clone(),
        &buffers[2],
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
    let (base, added) = (
        input.to_vec::<f32>().unwrap(),
        values.to_vec::<f32>().unwrap(),
    );
    for (index, actual) in actual.iter().enumerate() {
        let expected = if (8..16).contains(&index) {
            base[index] + added[index - 8] + added[index]
        } else {
            base[index]
        };
        assert_eq!(*actual, expected, "element {index}");
    }
}

#[test]
fn test_lowered_quantization_matches_the_cpu_and_reports_invalid_scales() {
    fn layout(tensor: &Tensor) -> (DataType, &Layout) {
        (tensor.data_type(), tensor.layout())
    }

    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([2, 3]).unwrap(),
        vec![0.3_f32, -1.7, 2.2, 4.1, -0.4, 9.9],
    )
    .unwrap();
    let scale = Tensor::from_vec(
        Device::Cpu,
        Shape::create([3]).unwrap(),
        vec![0.5_f32, 0.25, 2.0],
    )
    .unwrap();
    let zero_point =
        Tensor::from_vec(Device::Cpu, Shape::create([3]).unwrap(), vec![0_i64, 3, -2]).unwrap();
    let attributes = QuantizationAttributes {
        axis: Some(1),
        block_size: None,
        data_type: Some(DataType::I8),
        packed: None,
    };
    let lowered = quantization(
        Operation::Quantize,
        attributes,
        layout(&input),
        layout(&scale),
        layout(&zero_point),
    )
    .unwrap();
    let expected = call_one(
        Operation::Quantize,
        OperationAttributes::Quantization(attributes),
        &[&input, &scale, &zero_point],
    )
    .unwrap();
    let inputs = [input.clone(), scale, zero_point.clone()];
    assert_lowered_matches(&lowered, &inputs, &expected, 0.0);

    let invalid = Tensor::from_vec(
        Device::Cpu,
        Shape::create([3]).unwrap(),
        vec![0.5_f32, 0.0, 2.0],
    )
    .unwrap();
    let mut buffers = vec![
        input.to_bytes().unwrap(),
        invalid.to_bytes().unwrap(),
        zero_point.to_bytes().unwrap(),
        vec![0; 6],
    ];
    assert_eq!(
        interpret::run(&lowered.kernel, &lowered.bindings, &mut buffers).unwrap(),
        STATUS_INVALID_QUANTIZATION_SCALE
    );
}

#[test]
fn test_contractions_with_packed_weights_match_dequantizing_then_multiplying() {
    let (rows, inner, outputs) = (2_i64, 6_i64, 3_i64);
    let x = sample(&[rows, inner], 1, DataType::F32);
    let bytes = Tensor::from_vec(
        Device::Cpu,
        Shape::create([outputs, inner / 2]).unwrap(),
        (0..outputs * inner / 2)
            .map(|index| i8::try_from(index * 37 % 256 - 128).unwrap())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let scale = Tensor::from_vec(
        Device::Cpu,
        Shape::create([outputs]).unwrap(),
        vec![0.5_f32, 0.25, 2.0],
    )
    .unwrap();
    let zero_point = Tensor::from_vec(
        Device::Cpu,
        Shape::create([outputs]).unwrap(),
        vec![1_i64, 0, -2],
    )
    .unwrap();
    for format in [PackedFormat::Int4, PackedFormat::Float4E2M1] {
        let weight = bytes
            .dequantize_packed(&scale, &zero_point, Some(0), format)
            .unwrap();
        let expected = x.matmul(&weight.transpose(0, 1).unwrap()).unwrap();
        // The weight's bytes, transposed to `[inner / 2, outputs]`, with one scale and zero point per output column.
        let packed = Layout::contiguous([outputs, inner / 2])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let parameter = Layout::with_strides([inner, outputs], [0, 1]).unwrap();
        let lowered = contraction(
            DataType::F32,
            &Lhs::plain(DataType::F32, x.layout()),
            &Weight::Dequantized {
                quantized: (DataType::I8, &packed),
                scale: &parameter,
                zero_point: &parameter,
                rounded: None,
                packed: Some(format),
            },
            None,
        )
        .unwrap();
        assert_lowered_matches(
            &lowered,
            &[x.clone(), bytes.clone(), scale.clone(), zero_point.clone()],
            &expected,
            1e-5,
        );
    }
}

#[test]
fn test_normalizations_store_a_rotary_epilogue_read_through_slices_and_a_transpose() {
    // A head's RMS norm over `[batch, sequence, heads, width]`, transposed to heads before the sequence, then the rotary embedding of its halves.
    let (batch, sequence, heads, width) = (2_i64, 3, 2, 8);
    let half = width / 2;
    let x = sample(&[batch, sequence, heads, width], 1, DataType::F32);
    let weight = sample(&[width], 2, DataType::F32);
    let cos = sample(&[sequence, half], 3, DataType::F32);
    let sin = sample(&[sequence, half], 4, DataType::F32);
    let attributes = LayerNormalizationAttributes {
        normalized_shape: vec![width],
        has_weight: true,
        has_bias: false,
        eps: 1e-6,
    };
    let normed = call_one(
        Operation::RmsNorm,
        OperationAttributes::LayerNormalization(attributes.clone()),
        &[&x, &weight],
    )
    .unwrap()
    .transpose(1, 2)
    .unwrap();
    let first = normed.narrow(3, 0, half).unwrap();
    let second = normed.narrow(3, half, half).unwrap();
    let expected = Tensor::concat(
        &[
            first
                .mul(&cos)
                .unwrap()
                .sub(&second.mul(&sin).unwrap())
                .unwrap(),
            second
                .mul(&cos)
                .unwrap()
                .add(&first.mul(&sin).unwrap())
                .unwrap(),
        ],
        3,
    )
    .unwrap()
    .contiguous()
    .unwrap();
    let shape = |dimensions: &[i64]| Shape::create(dimensions).unwrap();
    let (rows, rotated, halves) = (
        shape(&[batch, sequence, heads, width]),
        shape(&[batch, heads, sequence, width]),
        shape(&[batch, heads, sequence, half]),
    );
    let attributes_none = OperationAttributes::None;
    let node = |value, shape: &Shape| EpilogueNode {
        value,
        shape: shape.clone(),
    };
    let step = |operation, operands: Vec<usize>| EpilogueValue::Step {
        operation,
        attributes: &attributes_none,
        operands,
    };
    let epilogue = RowEpilogue {
        inputs: vec![(DataType::F32, cos.layout()), (DataType::F32, sin.layout())],
        nodes: vec![
            node(EpilogueValue::Row, &rows),
            node(
                EpilogueValue::Transpose {
                    value: 0,
                    dims: [1, 2],
                },
                &rotated,
            ),
            node(EpilogueValue::Slice { value: 1, start: 0 }, &halves),
            node(
                EpilogueValue::Slice {
                    value: 1,
                    start: half,
                },
                &halves,
            ),
            node(EpilogueValue::Input(0), cos.shape()),
            node(EpilogueValue::Input(1), sin.shape()),
            node(step(Operation::Mul, vec![2, 4]), &halves),
            node(step(Operation::Mul, vec![3, 5]), &halves),
            node(step(Operation::Sub, vec![6, 7]), &halves),
            node(step(Operation::Mul, vec![3, 4]), &halves),
            node(step(Operation::Mul, vec![2, 5]), &halves),
            node(step(Operation::Add, vec![9, 10]), &halves),
            node(
                EpilogueValue::Concat {
                    values: vec![8, 11],
                },
                &rotated,
            ),
        ],
    };
    let group = ElementGroup::input(DataType::F32, x.layout());
    let lowered = normalization_with_epilogue(
        Operation::RmsNorm,
        &attributes,
        &group,
        &[(DataType::F32, weight.layout())],
        &epilogue,
    )
    .unwrap();
    assert_lowered_matches(&lowered, &[x, weight, cos, sin], &expected, 1e-5);
}

#[test]
fn test_reductions_store_epilogues_of_their_input_shape_along_the_reduced_axes() {
    // RMS norm over the channels of `[batch, channels, height, width]`: the group squares `x`, the mean over the channels keeps them, and the epilogue divides `x` by the root.
    let x = sample(&[2, 8, 3, 4], 1, DataType::F32);
    let reduced = Shape::create([2, 1, 3, 4]).unwrap();
    let expected = x
        .div(&x.mul(&x).unwrap().mean(&[1], true).unwrap().sqrt().unwrap())
        .unwrap();
    let attributes = OperationAttributes::None;
    let steps = [FusedStep {
        operation: Operation::Mul,
        attributes: &attributes,
        operands: vec![FusedOperand::Input(0), FusedOperand::Input(0)],
    }];
    let group = ElementGroup {
        steps: &steps,
        ..ElementGroup::input(DataType::F32, x.layout())
    };
    let node = |value, shape: &Shape| EpilogueNode {
        value,
        shape: shape.clone(),
    };
    let epilogue = RowEpilogue {
        inputs: Vec::new(),
        nodes: vec![
            node(EpilogueValue::Row, &reduced),
            node(
                EpilogueValue::Step {
                    operation: Operation::Sqrt,
                    attributes: &attributes,
                    operands: vec![0],
                },
                &reduced,
            ),
            node(
                EpilogueValue::GroupInput {
                    slot: 0,
                    data_type: DataType::F32,
                },
                x.shape(),
            ),
            node(
                EpilogueValue::Step {
                    operation: Operation::Div,
                    attributes: &attributes,
                    operands: vec![2, 1],
                },
                x.shape(),
            ),
        ],
    };
    let lowered = reduction(Operation::Mean, &group, &[1], true, Some(&epilogue)).unwrap();
    assert_lowered_matches(&lowered, &[x], &expected, 1e-5);
}

#[test]
fn test_window_reductions_match_the_cpu_pools_and_depthwise_convolutions() {
    let pools = [
        // Plain windows, padded and dilated windows, ceil-mode windows that overhang the padding, and a `[C, H, W]` image.
        (
            vec![2_i64, 3, 9, 8],
            (3, 2),
            (2, 2),
            (0, 0, 0, 0),
            (1, 1),
            false,
        ),
        (
            vec![1, 2, 9, 8],
            (3, 3),
            (2, 1),
            (1, 1, 1, 0),
            (2, 1),
            false,
        ),
        (vec![1, 2, 7, 8], (3, 2), (2, 3), (1, 0, 0, 1), (1, 1), true),
        (vec![3, 6, 5], (2, 2), (1, 2), (1, 1, 0, 0), (1, 1), true),
    ];
    for (dimensions, kernel_size, stride, padding, dilation, ceil_mode) in pools {
        let input = sample(&dimensions, 1, DataType::F32)
            .sub_scalar(1.0)
            .unwrap();
        let mut cases = vec![(Operation::MaxPool2d, None)];
        if dilation == (1, 1) {
            cases.push((Operation::AvgPool2d, Some(true)));
            cases.push((Operation::AvgPool2d, Some(false)));
        }
        for (operation, count_include_pad) in cases {
            let attributes = Pool2dAttributes {
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
                count_include_pad,
            };
            let expected = call_one(
                operation,
                OperationAttributes::Pool2d(attributes),
                &[&input],
            )
            .unwrap();
            let lowered = pool2d(operation, &attributes, DataType::F32, input.layout()).unwrap();
            assert_lowered_matches(&lowered, slice::from_ref(&input), &expected, 1e-5);
        }
    }
    // Depthwise convolutions with one and two output channels per input channel, padded, strided, and dilated, with and without a bias.
    let convolutions = [
        (
            3_i64,
            1_i64,
            (3_i64, 3_i64),
            (1_i64, 1_i64),
            (1, 1, 1, 1),
            (1_i64, 1_i64),
            true,
        ),
        (2, 2, (2, 3), (2, 1), (0, 1, 2, 0), (2, 1), false),
    ];
    for (channels, multiplier, (kernel_height, kernel_width), stride, padding, dilation, biased) in
        convolutions
    {
        let input = sample(&[2, channels, 7, 6], 2, DataType::F32);
        let weight = sample(
            &[channels * multiplier, 1, kernel_height, kernel_width],
            3,
            DataType::F32,
        )
        .sub_scalar(1.0)
        .unwrap();
        let bias = sample(&[channels * multiplier], 4, DataType::F32);
        let attributes = Conv2dAttributes {
            has_bias: biased,
            stride,
            padding,
            dilation,
            groups: channels,
        };
        let mut inputs = vec![input.clone(), weight.clone()];
        if biased {
            inputs.push(bias.clone());
        }
        let expected = call_one(
            Operation::Conv2d,
            OperationAttributes::Conv2d(attributes),
            &inputs.iter().collect::<Vec<_>>(),
        )
        .unwrap();
        let lowered = conv2d(
            false,
            &attributes,
            DataType::F32,
            input.layout(),
            (DataType::F32, weight.layout()),
            biased.then(|| (DataType::F32, bias.layout())),
        )
        .unwrap();
        assert_lowered_matches(&lowered, &inputs, &expected, 1e-5);
    }
}

#[test]
fn test_argument_reductions_match_the_cpu_with_ties_and_nans() {
    // Values repeat every 97 points, so rows hold ties, and one row of each holds two NaNs, of which the first wins.
    for (dimensions, dim) in [
        (vec![4_i64, 50], 1_i64),
        (vec![30, 6], 0),
        (vec![3, 20, 5], -2),
    ] {
        let count = usize::try_from(dimensions.iter().product::<i64>()).unwrap();
        let mut values = (0..count)
            .map(|index| ((index * 37) % 97) as f32 / 25.0 - 1.9)
            .collect::<Vec<_>>();
        values[count / 2] = f32::NAN;
        values[count / 2 + 1] = f32::NAN;
        let input = Tensor::from_vec(
            Device::Cpu,
            Shape::create(dimensions.clone()).unwrap(),
            values,
        )
        .unwrap();
        for operation in [Operation::ArgMax, Operation::ArgMin] {
            let expected =
                call_one(operation, OperationAttributes::Dimension(dim), &[&input]).unwrap();
            let lowered = reduction(
                operation,
                &ElementGroup::input(DataType::F32, input.layout()),
                &[dim],
                false,
                None,
            )
            .unwrap();
            assert_lowered_matches(&lowered, slice::from_ref(&input), &expected, 0.0);
        }
    }
}

#[test]
fn test_scans_match_the_cpu_along_inner_and_outer_axes() {
    for (dimensions, dim) in [
        (vec![4_i64, 50], 1_i64),
        (vec![30, 6], 0),
        (vec![3, 20, 5], -2),
        (vec![], 0),
    ] {
        for data_type in [DataType::F32, DataType::I32] {
            let input = sample(&dimensions, 1, data_type);
            for operation in [Operation::Cumsum, Operation::Cumprod] {
                let expected =
                    call_one(operation, OperationAttributes::Dimension(dim), &[&input]).unwrap();
                let lowered = scan(
                    operation,
                    &ElementGroup::input(data_type, input.layout()),
                    dim,
                )
                .unwrap();
                assert_lowered_matches(&lowered, slice::from_ref(&input), &expected, 1e-5);
            }
        }
    }
}
