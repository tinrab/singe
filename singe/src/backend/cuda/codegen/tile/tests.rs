use half::{bf16, f16};
use singe_codegen::{
    abi::Bindings,
    ir::{
        Access, AxisId, AxisKind, BinaryOp, ElementType, Expr, Kernel, Literal, NodeId, OperandId,
        OperandRole, Size, SizeId, UnaryOp,
    },
    schedule::{Schedule, space::operand_lengths},
};

use crate::{
    backend::{
        cuda::{
            codegen::{
                KernelCode, KernelDescription, KernelLanguage, PlanningContext, plan_kernel,
                test_support::{assert_matches, bf16_buffer, f32_buffer, rows_kernel},
            },
            kernel_plan::{self, KernelInput, KernelPlanner},
            runtime::runtime_for_device,
        },
        kernel_lowering::matmul,
        registry::{Operation, OperationAttributes, QuantizationAttributes},
    },
    data_type::DataType,
    device::Device,
    dispatch::call_one,
    error::Error,
    idx,
    layout::Layout,
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};

/// Returns an access that walks `axis` with a stride of `one`.
fn contiguous(operand: OperandId, axis: AxisId, one: SizeId) -> Access {
    Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: vec![(axis, one)],
    }
}

#[test]
#[expect(
    clippy::many_single_char_names,
    reason = "the operands and axes take the names of the formula"
)]
fn test_tile_element_kernels_match_the_interpreter_exactly() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if skip_if_stub(runtime_for_device(Device::Cuda(0))).is_none() {
        return;
    }
    // out[i, j] = x[i, j] + y[j], over an argument row count.
    let mut add = Kernel::new("singe_tile_add");
    let x = add.add_operand(ElementType::F32, OperandRole::Input);
    let y = add.add_operand(ElementType::F32, OperandRole::Input);
    let out = add.add_operand(ElementType::F32, OperandRole::Output);
    let rows = add.add_size(Size::Argument);
    let columns = add.add_size(Size::Static(5));
    let one = add.add_size(Size::Static(1));
    let i = add.add_axis(rows, AxisKind::Parallel);
    let j = add.add_axis(columns, AxisKind::Parallel);
    let matrix = |operand| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, columns), (j, one)],
    };
    let lhs = add.load(matrix(x));
    let rhs = add.load(contiguous(y, j, one));
    let sum = add.binary(BinaryOp::Add, lhs, rhs);
    add.store(matrix(out), sum);
    for rows in [0_usize, 1, 7, 130] {
        let bindings = Bindings {
            sizes: vec![rows as u64],
            scalars: Vec::new(),
        };
        let buffers = vec![
            f32_buffer(rows * 5, 1),
            f32_buffer(5, 2),
            vec![0; rows * 20],
        ];
        for size in [64, 1024] {
            assert_matches(&add, Schedule::Tiles { size }, &bindings, &buffers, 0.0);
        }
    }

    // A transposed BF16 copy.
    let mut copy = Kernel::new("singe_tile_copy");
    let x = copy.add_operand(ElementType::Bf16, OperandRole::Input);
    let out = copy.add_operand(ElementType::Bf16, OperandRole::Output);
    let three = copy.add_size(Size::Static(3));
    let five = copy.add_size(Size::Static(5));
    let one = copy.add_size(Size::Static(1));
    let i = copy.add_axis(five, AxisKind::Parallel);
    let j = copy.add_axis(three, AxisKind::Parallel);
    let value = copy.load(Access {
        operand: x,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, one), (j, five)],
    });
    copy.store(
        Access {
            operand: out,
            base: None,
            indirect: Vec::new(),
            strides: vec![(i, three), (j, one)],
        },
        value,
    );
    assert_matches(
        &copy,
        Schedule::Tiles { size: 16 },
        &Bindings::default(),
        &[bf16_buffer(15, 3), vec![0; 30]],
        0.0,
    );

    // A threshold with a scalar, a cast, and a select, at an argument offset into the input.
    let mut threshold = Kernel::new("singe_tile_threshold");
    let x = threshold.add_operand(ElementType::F32, OperandRole::Input);
    let out = threshold.add_operand(ElementType::I64, OperandRole::Output);
    let offset = threshold.add_size(Size::Argument);
    let length = threshold.add_size(Size::Static(40));
    let one = threshold.add_size(Size::Static(1));
    let limit = threshold.add_scalar(ElementType::F32);
    let i = threshold.add_axis(length, AxisKind::Parallel);
    let value = threshold.load(Access {
        operand: x,
        base: Some(offset),
        indirect: Vec::new(),
        strides: vec![(i, one)],
    });
    let bound = threshold.add_node(ElementType::F32, Expr::Scalar(limit));
    let above = threshold.binary(BinaryOp::Greater, value, bound);
    let scaled = threshold.binary(BinaryOp::Mul, value, value);
    let whole = threshold.cast(ElementType::I64, scaled);
    let fallback = threshold.literal(ElementType::I64, Literal::Int(-1));
    let result = threshold.select(above, whole, fallback);
    threshold.store(contiguous(out, i, one), result);
    assert_matches(
        &threshold,
        Schedule::Tiles { size: 32 },
        &Bindings {
            sizes: vec![3],
            scalars: vec![Literal::Float(0.25)],
        },
        &[f32_buffer(43, 4), vec![0; 320]],
        0.0,
    );

    // Integer arithmetic and `bool` storage: out[i] = a[i] % 3 == 0 ? a[i] / 2 : -a[i], and flags[i] = !(a[i] > 4).
    let mut integers = Kernel::new("singe_tile_integers");
    let a = integers.add_operand(ElementType::I32, OperandRole::Input);
    let out = integers.add_operand(ElementType::I32, OperandRole::Output);
    let flags = integers.add_operand(ElementType::Bool, OperandRole::Output);
    let length = integers.add_size(Size::Argument);
    let one = integers.add_size(Size::Static(1));
    let i = integers.add_axis(length, AxisKind::Parallel);
    let value = integers.load(contiguous(a, i, one));
    let three = integers.literal(ElementType::I32, Literal::Int(3));
    let two = integers.literal(ElementType::I32, Literal::Int(2));
    let zero = integers.literal(ElementType::I32, Literal::Int(0));
    let four = integers.literal(ElementType::I32, Literal::Int(4));
    let remainder = integers.binary(BinaryOp::Rem, value, three);
    let divisible = integers.binary(BinaryOp::Equal, remainder, zero);
    let half = integers.binary(BinaryOp::Div, value, two);
    let negated = integers.unary(UnaryOp::Neg, value);
    let result = integers.select(divisible, half, negated);
    integers.store(contiguous(out, i, one), result);
    let large = integers.binary(BinaryOp::Greater, value, four);
    let small = integers.unary(UnaryOp::Not, large);
    integers.store(contiguous(flags, i, one), small);
    let count = 37_usize;
    let input = (0..count)
        .flat_map(|index| (i32::try_from(index).unwrap() - 18).to_le_bytes())
        .collect::<Vec<_>>();
    assert_matches(
        &integers,
        Schedule::Tiles { size: 16 },
        &Bindings {
            sizes: vec![count as u64],
            scalars: Vec::new(),
        },
        &[input, vec![0; count * 4], vec![0; count]],
        0.0,
    );

    // A gather of table rows through an index tensor: out[t, j] = table[index[t], j].
    let mut gather = Kernel::new("singe_tile_gather");
    let table = gather.add_operand(ElementType::F32, OperandRole::Input);
    let index = gather.add_operand(ElementType::I64, OperandRole::Input);
    let out = gather.add_operand(ElementType::F32, OperandRole::Output);
    let tokens = gather.add_size(Size::Argument);
    let width = gather.add_size(Size::Static(6));
    let one = gather.add_size(Size::Static(1));
    let t = gather.add_axis(tokens, AxisKind::Parallel);
    let j = gather.add_axis(width, AxisKind::Parallel);
    let row = gather.load(contiguous(index, t, one));
    let value = gather.load(Access {
        operand: table,
        base: None,
        indirect: vec![(row, width)],
        strides: vec![(j, one)],
    });
    gather.store(
        Access {
            operand: out,
            base: None,
            indirect: Vec::new(),
            strides: vec![(t, width), (j, one)],
        },
        value,
    );
    let indices = [3_i64, 0, 4, 4, 1]
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect::<Vec<_>>();
    assert_matches(
        &gather,
        Schedule::Tiles { size: 8 },
        &Bindings {
            sizes: vec![5],
            scalars: Vec::new(),
        },
        &[f32_buffer(30, 5), indices, vec![0; 5 * 6 * 4]],
        0.0,
    );
}

#[test]
fn test_tile_row_kernels_match_the_interpreter() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if skip_if_stub(runtime_for_device(Device::Cuda(0))).is_none() {
        return;
    }
    for element in [ElementType::F32, ElementType::Bf16] {
        // `reduce` combines a row in its own order, so results agree with the interpreter's in-order sums to rounding.
        let tolerance = if element == ElementType::Bf16 {
            1e-2
        } else {
            1e-5
        };
        let buffer = |count, seed| match element {
            ElementType::F32 => f32_buffer(count, seed),
            _ => bf16_buffer(count, seed),
        };
        let size = element.size_bytes();
        for (weighted, baked) in [(false, false), (true, false), (false, true), (true, true)] {
            for (rows, width) in [(3_usize, 1_usize), (5, 33), (2, 513)] {
                let (mut kernel, _) = rows_kernel("singe_tile_rows", element, weighted);
                let mut sizes = vec![rows as u64, width as u64];
                if baked {
                    kernel.sizes[1] = Size::Static(width as u64);
                    sizes.pop();
                }
                let mut buffers = vec![buffer(rows * width, 1)];
                if weighted {
                    buffers.push(buffer(width, 2));
                }
                buffers.push(vec![0; rows * width * size]);
                let bindings = Bindings {
                    sizes,
                    scalars: if weighted {
                        vec![Literal::Float(width as f64), Literal::Float(1e-6)]
                    } else {
                        Vec::new()
                    },
                };
                let width = u32::try_from(width).unwrap().next_power_of_two();
                for width in [width, width * 2] {
                    assert_matches(
                        &kernel,
                        Schedule::TileRows { width },
                        &bindings,
                        &buffers,
                        tolerance,
                    );
                }
            }
        }
    }
}

#[test]
fn test_the_tile_target_emits_tile_ir_and_keeps_cuda_c_for_what_it_cannot_express() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let mut kernel = Kernel::new("singe_tile_target");
    let x = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let length = kernel.add_size(Size::Argument);
    let one = kernel.add_size(Size::Static(1));
    let i = kernel.add_axis(length, AxisKind::Parallel);
    let value = kernel.load(contiguous(x, i, one));
    let exponential = kernel.unary(UnaryOp::Exp, value);
    kernel.store(contiguous(out, i, one), exponential);
    let mut erf = kernel.clone();
    erf.nodes[1].expr = Expr::Unary(UnaryOp::Erf, NodeId(0));
    let schedule = Schedule::Elements {
        threads: 128,
        vector: 1,
    };
    let bindings = Bindings {
        sizes: vec![100],
        scalars: Vec::new(),
    };
    let tile = PlanningContext {
        language: KernelLanguage::TileIr,
        ..PlanningContext::default()
    };
    let description = KernelDescription::default();
    let tiled = plan_kernel(&kernel, schedule, &bindings, &description, &tile);
    let fallback = plan_kernel(&erf, schedule, &bindings, &description, &tile);
    let tiled = tiled.unwrap();
    assert!(matches!(tiled.code, KernelCode::TileIr(_)));
    // One tile of 1024 points covers the hundred elements, as one thread the tile target sizes.
    assert_eq!((tiled.launch.blocks, tiled.launch.threads), (1, 1));
    assert!(matches!(fallback.unwrap().code, KernelCode::Source(_)));
    assert!(matches!(
        plan_kernel(
            &kernel,
            schedule,
            &bindings,
            &description,
            &PlanningContext::default()
        )
        .unwrap()
        .code,
        KernelCode::Source(_)
    ));
}

#[test]
fn test_tile_kernels_report_status_codes_and_add_atomically() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let planner = KernelPlanner::new(PlanningContext {
        language: KernelLanguage::TileIr,
        ..PlanningContext::default()
    });
    let device = Device::Cuda(0);
    let Some(table) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([3, 2]).unwrap(),
        vec![1_f32, 2., 3., 4., 5., 6.],
    )) else {
        return;
    };
    let indices = |values: Vec<i64>| {
        Tensor::from_vec(
            device,
            Shape::create([i64::try_from(values.len()).unwrap()]).unwrap(),
            values,
        )
        .unwrap()
    };
    let gather = |index: &Tensor| {
        kernel_plan::execute_with(
            &planner,
            Operation::IndexSelect,
            &OperationAttributes::IndexSelect(0),
            &[&table, index],
        )
    };
    let zero_scale = Tensor::from_vec(device, Shape::scalar(), vec![0_f32]).unwrap();
    let zero_point = Tensor::from_vec(device, Shape::scalar(), vec![0_i64]).unwrap();
    let quantized =
        Tensor::from_vec(device, Shape::create([4]).unwrap(), vec![1_i8, -2, 3, 4]).unwrap();
    let dequantize = || {
        kernel_plan::execute_with(
            &planner,
            Operation::Dequantize,
            &OperationAttributes::Quantization(QuantizationAttributes {
                axis: None,
                block_size: None,
                data_type: None,
                packed: None,
            }),
            &[&quantized, &zero_scale, &zero_point],
        )
    };
    let updates = Tensor::from_vec(
        device,
        Shape::create([4, 2]).unwrap(),
        vec![0.5_f32, 1., 1.5, 2., 2.5, 3., 3.5, 4.],
    )
    .unwrap();
    // Repeated rows add every update that lands on them.
    let repeated = indices(vec![1, 1, 0, 1]);
    let expected = table
        .to_device(Device::Cpu)
        .unwrap()
        .index_put(
            &idx![&repeated.to_device(Device::Cpu).unwrap()],
            &updates.to_device(Device::Cpu).unwrap(),
            true,
        )
        .unwrap();

    // Quantizing rounds ties to even: 0.5, 1.5, 2.5, and -2.5 at a scale of one.
    let unit_scale = Tensor::from_vec(device, Shape::scalar(), vec![1_f32]).unwrap();
    let values = Tensor::from_vec(
        device,
        Shape::create([7]).unwrap(),
        vec![0.5_f32, 1.5, 2.5, -2.5, 3.7, -0.2, 300.0],
    )
    .unwrap();
    let quantize_attributes = OperationAttributes::Quantization(QuantizationAttributes {
        axis: None,
        block_size: None,
        data_type: Some(DataType::I8),
        packed: None,
    });
    let quantize = |inputs: [&Tensor; 3]| {
        kernel_plan::execute_with(&planner, Operation::Quantize, &quantize_attributes, &inputs)
    };
    let cpu = |tensor: &Tensor| tensor.to_device(Device::Cpu).unwrap();
    let expected_quantized = call_one(
        Operation::Quantize,
        quantize_attributes.clone(),
        &[&cpu(&values), &cpu(&unit_scale), &cpu(&zero_point)],
    )
    .unwrap();

    let quantized_values = quantize([&values, &unit_scale, &zero_point]);
    let in_range = gather(&indices(vec![2, 0, 1]));
    let out_of_range = gather(&indices(vec![0, 3]));
    let invalid_scale = dequantize();
    let accumulated = planner
        .index_put(
            &KernelInput::from_source(&table),
            &KernelInput::from_source(&updates),
            &[KernelInput::from_source(&repeated)],
            0,
            true,
        )
        .and_then(|plan| {
            let output = kernel_plan::execute_with(
                &planner,
                Operation::Copy,
                &OperationAttributes::None,
                &[&table],
            )?;
            kernel_plan::run_into(&plan, &[&updates, &repeated], &output)?;
            Ok(output)
        });

    assert_eq!(
        in_range.unwrap().to_vec::<f32>().unwrap(),
        vec![5., 6., 1., 2., 3., 4.]
    );
    assert!(matches!(out_of_range, Err(Error::IndexOutOfBounds)));
    assert_eq!(
        quantized_values.unwrap().to_vec::<i8>().unwrap(),
        expected_quantized.to_vec::<i8>().unwrap()
    );
    assert!(invalid_scale.is_err());
    assert_eq!(
        accumulated.unwrap().to_vec::<f32>().unwrap(),
        expected.to_vec::<f32>().unwrap()
    );
}

#[test]
fn test_tensor_core_matmuls_match_the_interpreter() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if skip_if_stub(runtime_for_device(Device::Cuda(0))).is_none() {
        return;
    }
    // Tails along every dimension, a batch, a transposed right operand, and `F32` operands.
    let contiguous = |dimensions: &[i64]| Layout::contiguous(dimensions).unwrap();
    let cases = [
        (DataType::Bf16, contiguous(&[37, 70]), contiguous(&[70, 45])),
        (
            DataType::Bf16,
            contiguous(&[3, 20, 64]),
            contiguous(&[3, 33, 64]).transpose(1, 2).unwrap(),
        ),
        (
            DataType::F16,
            contiguous(&[64, 128]),
            contiguous(&[128, 64]),
        ),
        (DataType::F32, contiguous(&[33, 17]), contiguous(&[17, 9])),
    ];
    for (data_type, lhs, rhs) in cases {
        let lowered = matmul(data_type, &lhs, &rhs).unwrap();
        let lengths = operand_lengths(&lowered.kernel, &lowered.bindings.sizes).unwrap();
        let buffers = lowered
            .kernel
            .operands
            .iter()
            .zip(&lengths)
            .enumerate()
            .map(|(index, (operand, &length))| {
                let count = usize::try_from(length).unwrap();
                match (operand.role, operand.element) {
                    (OperandRole::Output, element) => vec![0; count * element.size_bytes()],
                    (_, ElementType::F32) => f32_buffer(count, index),
                    // `F16` and `BF16` values of the same small integers over 25 are exact in both.
                    (_, _) => bf16_buffer(count, index),
                }
            })
            .collect::<Vec<_>>();
        let buffers = if data_type == DataType::F16 {
            buffers
                .into_iter()
                .zip(&lowered.kernel.operands)
                .map(|(buffer, operand)| {
                    if operand.role == OperandRole::Output {
                        return buffer;
                    }
                    buffer
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .flat_map(|&bits| {
                            let value = f32::from(bf16::from_le_bytes(bits));
                            f16::from_f32(value).to_le_bytes()
                        })
                        .collect()
                })
                .collect()
        } else {
            buffers
        };
        for schedule in [
            Schedule::TileMatmul {
                m: 16,
                n: 16,
                k: 16,
            },
            Schedule::TileMatmul {
                m: 64,
                n: 32,
                k: 32,
            },
        ] {
            assert_matches(&lowered.kernel, schedule, &lowered.bindings, &buffers, 2e-2);
        }
    }
}
