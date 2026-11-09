use singe_codegen::{
    abi::Bindings,
    ir::{
        Access, AxisKind, BinaryOp, Combiner, ElementType, Expr, Kernel, Literal, OperandRole, Size,
    },
    schedule::{Schedule, space::candidates},
};

use crate::{
    backend::{
        cuda::{
            codegen::test_support::{assert_matches, bf16_buffer, f32_buffer, rows_kernel},
            runtime::runtime_for_device,
        },
        kernel_lowering::{ElementGroup, reduction, scan},
        registry::Operation,
    },
    data_type::DataType,
    device::Device,
    layout::Layout,
    testing::{maybe_lock_device, skip_if_stub},
};

#[test]
fn test_emitted_element_kernels_match_the_interpreter_exactly() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if skip_if_stub(runtime_for_device(Device::Cuda(0))).is_none() {
        return;
    }
    // out[i, j] = x[i, j] + y[j], over an argument row count.
    let mut add = Kernel::new("singe_test_add");
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
    let rhs = add.load(Access {
        operand: y,
        base: None,
        indirect: Vec::new(),
        strides: vec![(j, one)],
    });
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
        // Rows of five leave a vector tail in every row and misalign every other row.
        for (threads, vector) in [(64, 1), (128, 1), (64, 2), (128, 4)] {
            assert_matches(
                &add,
                Schedule::Elements { threads, vector },
                &bindings,
                &buffers,
                0.0,
            );
        }
    }

    // A transposed BF16 copy, and a threshold with a cast and a select.
    let mut copy = Kernel::new("singe_test_copy");
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
        Schedule::Elements {
            threads: 32,
            vector: 1,
        },
        &Bindings::default(),
        &[bf16_buffer(15, 3), vec![0; 30]],
        0.0,
    );

    let mut threshold = Kernel::new("singe_test_threshold");
    let x = threshold.add_operand(ElementType::F32, OperandRole::Input);
    let out = threshold.add_operand(ElementType::I64, OperandRole::Output);
    let length = threshold.add_size(Size::Static(40));
    let one = threshold.add_size(Size::Static(1));
    let limit = threshold.add_scalar(ElementType::F32);
    let i = threshold.add_axis(length, AxisKind::Parallel);
    let vector = |operand| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, one)],
    };
    let value = threshold.load(vector(x));
    let bound = threshold.add_node(ElementType::F32, Expr::Scalar(limit));
    let above = threshold.binary(BinaryOp::Greater, value, bound);
    let scaled = threshold.binary(BinaryOp::Mul, value, value);
    let whole = threshold.cast(ElementType::I64, scaled);
    let fallback = threshold.literal(ElementType::I64, Literal::Int(-1));
    let result = threshold.select(above, whole, fallback);
    threshold.store(vector(out), result);
    // A vector of eight `I64` outputs is wider than one access, so it is written lane by lane.
    for vector in [1, 2, 8] {
        assert_matches(
            &threshold,
            Schedule::Elements {
                threads: 128,
                vector,
            },
            &Bindings {
                sizes: Vec::new(),
                scalars: vec![Literal::Float(0.25)],
            },
            &[f32_buffer(40, 4), vec![0; 320]],
            0.0,
        );
    }

    // out[i] = x[offset + i] × 2 over BF16 views at an argument offset, which misaligns the input's vectors.
    let mut doubled = Kernel::new("singe_test_doubled");
    let x = doubled.add_operand(ElementType::Bf16, OperandRole::Input);
    let out = doubled.add_operand(ElementType::Bf16, OperandRole::Output);
    let offset = doubled.add_size(Size::Argument);
    let length = doubled.add_size(Size::Argument);
    let one = doubled.add_size(Size::Static(1));
    let i = doubled.add_axis(length, AxisKind::Parallel);
    let value = doubled.load(Access {
        operand: x,
        base: Some(offset),
        indirect: Vec::new(),
        strides: vec![(i, one)],
    });
    let two = doubled.literal(ElementType::F32, Literal::Float(2.0));
    let product = doubled.binary(BinaryOp::Mul, value, two);
    doubled.store(
        Access {
            operand: out,
            base: None,
            indirect: Vec::new(),
            strides: vec![(i, one)],
        },
        product,
    );
    for (start, count) in [(0_usize, 64_usize), (3, 64), (1, 13), (0, 7)] {
        for vector in [1, 4, 8] {
            assert_matches(
                &doubled,
                Schedule::Elements {
                    threads: 32,
                    vector,
                },
                &Bindings {
                    sizes: vec![start as u64, count as u64],
                    scalars: Vec::new(),
                },
                &[bf16_buffer(start + count, 5), vec![0; count * 2]],
                0.0,
            );
        }
    }
}

#[test]
fn test_emitted_row_kernels_match_the_interpreter() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if skip_if_stub(runtime_for_device(Device::Cuda(0))).is_none() {
        return;
    }
    for element in [ElementType::F32, ElementType::Bf16] {
        // Element schedules combine in the interpreter's order; row schedules split the sum across threads.
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
        // A static width lets row schedules keep each row in registers.
        for (weighted, baked) in [(false, false), (true, false), (false, true), (true, true)] {
            for (rows, width) in [(3_usize, 1_usize), (5, 33), (2, 513)] {
                let (mut kernel, _) = rows_kernel("singe_test_rows", element, weighted);
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
                for schedule in [
                    Schedule::Elements {
                        threads: 64,
                        vector: 1,
                    },
                    Schedule::Rows {
                        threads: 32,
                        vector: 1,
                        tiles: [1, 1],
                    },
                    Schedule::Rows {
                        threads: 256,
                        vector: 4,
                        tiles: [1, 1],
                    },
                    Schedule::Warps {
                        threads: 32,
                        vector: 1,
                        lanes: 32,
                        tiles: [1, 1],
                    },
                    Schedule::Warps {
                        threads: 256,
                        vector: 8,
                        lanes: 32,
                        tiles: [1, 1],
                    },
                    Schedule::Warps {
                        threads: 64,
                        vector: 1,
                        lanes: 8,
                        tiles: [1, 1],
                    },
                    Schedule::Warps {
                        threads: 256,
                        vector: 4,
                        lanes: 16,
                        tiles: [1, 1],
                    },
                ] {
                    assert_matches(&kernel, schedule, &bindings, &buffers, tolerance);
                }
            }
        }
    }
}

#[test]
#[expect(
    clippy::many_single_char_names,
    reason = "the operands and axes take the names of the formula"
)]
fn test_vectors_wider_than_the_innermost_reduction_axis_cover_every_point() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if skip_if_stub(runtime_for_device(Device::Cuda(0))).is_none() {
        return;
    }
    // out[i] = sum over j and k of x[i, j, k] · (w[j] + k), where w, like a packed byte, serves both points of k.
    // Rows of 128 let row schedules keep them in registers, and rows of 8192 do not.
    for pairs in [64_u64, 4096] {
        let mut kernel = Kernel::new("singe_test_pairs");
        let x = kernel.add_operand(ElementType::F32, OperandRole::Input);
        let w = kernel.add_operand(ElementType::F32, OperandRole::Input);
        let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
        let rows = kernel.add_size(Size::Argument);
        let row = kernel.add_size(Size::Static(2 * pairs));
        let extent = kernel.add_size(Size::Static(pairs));
        let two = kernel.add_size(Size::Static(2));
        let one = kernel.add_size(Size::Static(1));
        let i = kernel.add_axis(rows, AxisKind::Parallel);
        let j = kernel.add_axis(extent, AxisKind::Reduction);
        let k = kernel.add_axis(two, AxisKind::Reduction);
        let value = kernel.load(Access {
            operand: x,
            base: None,
            indirect: Vec::new(),
            strides: vec![(i, row), (j, two), (k, one)],
        });
        let weight = kernel.load(Access {
            operand: w,
            base: None,
            indirect: Vec::new(),
            strides: vec![(j, one)],
        });
        let index = kernel.add_node(ElementType::I32, Expr::AxisIndex(k));
        let index = kernel.cast(ElementType::F32, index);
        let weight = kernel.binary(BinaryOp::Add, weight, index);
        let product = kernel.binary(BinaryOp::Mul, value, weight);
        let sum = kernel.reduce(Combiner::Sum, vec![j, k], product);
        kernel.store(
            Access {
                operand: out,
                base: None,
                indirect: Vec::new(),
                strides: vec![(i, one)],
            },
            sum,
        );
        let bindings = Bindings {
            sizes: vec![3],
            scalars: Vec::new(),
        };
        let pairs = usize::try_from(pairs).unwrap();
        let buffers = vec![
            f32_buffer(3 * 2 * pairs, 1),
            f32_buffer(pairs, 2),
            vec![0; 3 * 4],
        ];
        for schedule in [
            Schedule::Rows {
                threads: 32,
                vector: 4,
                tiles: [1, 1],
            },
            Schedule::Warps {
                threads: 128,
                vector: 8,
                lanes: 32,
                tiles: [1, 1],
            },
            Schedule::Warps {
                threads: 128,
                vector: 16,
                lanes: 8,
                tiles: [1, 1],
            },
        ] {
            assert_matches(&kernel, schedule, &bindings, &buffers, 1e-4);
        }
    }
}

#[test]
#[expect(
    clippy::many_single_char_names,
    reason = "the operands and axes take the names of the attention formula"
)]
fn test_softmax_sums_over_nested_reductions_match_the_interpreter() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if skip_if_stub(runtime_for_device(Device::Cuda(0))).is_none() {
        return;
    }
    // Attention for each query row i: logit[j] = sum over d of q[i, d] · k[j, d] / 4, out[i, e] = the softmax-weighted sum of v[j, e], and lse[i] = log sum over j of exp(logit[j]).
    let (keys, width, values) = (37_u64, 16_u64, 8_u64);
    let mut kernel = Kernel::new("singe_test_attend");
    let q = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let k = kernel.add_operand(ElementType::Bf16, OperandRole::Input);
    let v = kernel.add_operand(ElementType::Bf16, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let lse = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let key_count = kernel.add_size(Size::Static(keys));
    let key_width = kernel.add_size(Size::Static(width));
    let value_width = kernel.add_size(Size::Static(values));
    let one = kernel.add_size(Size::Static(1));
    let i = kernel.add_axis(rows, AxisKind::Parallel);
    let e = kernel.add_axis(value_width, AxisKind::Parallel);
    let j = kernel.add_axis(key_count, AxisKind::Reduction);
    let d = kernel.add_axis(key_width, AxisKind::Reduction);
    let access = |operand, strides| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides,
    };
    let query = kernel.load(access(q, vec![(i, key_width), (d, one)]));
    let key = kernel.load(access(k, vec![(j, key_width), (d, one)]));
    let key = kernel.cast(ElementType::F32, key);
    let product = kernel.binary(BinaryOp::Mul, query, key);
    let dot = kernel.reduce(Combiner::Sum, vec![d], product);
    let quarter = kernel.add_node(ElementType::F32, Expr::Literal(Literal::Float(0.25)));
    let logit = kernel.binary(BinaryOp::Mul, dot, quarter);
    let value = kernel.load(access(v, vec![(j, value_width), (e, one)]));
    let value = kernel.cast(ElementType::F32, value);
    let weighted = kernel.softmax_sum(vec![j], logit, value);
    let normalizer = kernel.reduce(Combiner::LogSumExp, vec![j], logit);
    kernel.store(access(out, vec![(i, value_width), (e, one)]), weighted);
    kernel.store(access(lse, vec![(i, one)]), normalizer);
    kernel.verify().unwrap();
    let rows = 5_usize;
    let [keys, width, values] = [keys, width, values].map(|size| usize::try_from(size).unwrap());
    let bindings = Bindings {
        sizes: vec![rows as u64],
        scalars: Vec::new(),
    };
    let buffers = vec![
        f32_buffer(rows * width, 1),
        bf16_buffer(keys * width, 2),
        bf16_buffer(keys * values, 3),
        vec![0; rows * values * 4],
        vec![0; rows * 4],
    ];
    for schedule in [
        Schedule::Elements {
            threads: 64,
            vector: 1,
        },
        Schedule::Rows {
            threads: 32,
            vector: 1,
            tiles: [1, 1],
        },
        Schedule::Rows {
            threads: 64,
            vector: 4,
            tiles: [1, 1],
        },
        Schedule::Warps {
            threads: 128,
            vector: 1,
            lanes: 32,
            tiles: [1, 1],
        },
        Schedule::Warps {
            threads: 128,
            vector: 2,
            lanes: 8,
            tiles: [1, 4],
        },
    ] {
        assert_matches(&kernel, schedule, &bindings, &buffers, 1e-4);
    }
}

#[test]
fn test_column_schedules_match_the_interpreter_on_outer_reductions() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if skip_if_stub(runtime_for_device(Device::Cuda(0))).is_none() {
        return;
    }
    let cases: [(&[i64], &[i64], Operation); 3] = [
        (&[300, 70], &[0], Operation::Sum),
        (&[5, 260, 33], &[1], Operation::Max),
        (&[129, 40], &[0], Operation::Mean),
    ];
    for (shape, dimensions, operation) in cases {
        let layout = Layout::contiguous(shape).unwrap();
        let lowered = reduction(
            operation,
            &ElementGroup::input(DataType::F32, &layout),
            dimensions,
            false,
            None,
        )
        .unwrap();
        let columns = candidates(&lowered.kernel, &lowered.bindings.sizes, false)
            .into_iter()
            .filter(|schedule| matches!(schedule, Schedule::Columns { .. }))
            .collect::<Vec<_>>();
        assert!(!columns.is_empty(), "{operation:?} over {dimensions:?}");
        let count = usize::try_from(shape.iter().product::<i64>()).unwrap();
        let reduced = usize::try_from(
            dimensions
                .iter()
                .map(|&dim| shape[usize::try_from(dim).unwrap()])
                .product::<i64>(),
        )
        .unwrap();
        let buffers = [f32_buffer(count, 1), vec![0; count / reduced * 4]];
        for schedule in columns {
            assert_matches(&lowered.kernel, schedule, &lowered.bindings, &buffers, 1e-4);
        }
    }
}

#[test]
fn test_argument_reductions_match_the_interpreter_under_every_schedule() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if skip_if_stub(runtime_for_device(Device::Cuda(0))).is_none() {
        return;
    }
    // Rows along the inner axis and columns along an outer one, short and long, with values that repeat every 97 points and a few NaNs.
    let cases: [(&[i64], i64); 4] = [
        (&[6, 40], 1),
        (&[3, 700], 1),
        (&[300, 7], 0),
        (&[4, 150, 9], 1),
    ];
    for (shape, dim) in cases {
        let count = usize::try_from(shape.iter().product::<i64>()).unwrap();
        let reduced = usize::try_from(shape[usize::try_from(dim).unwrap()]).unwrap();
        let mut input = f32_buffer(count, 1);
        for nan in [count / 3, count / 3 + 5] {
            input[nan * 4..nan * 4 + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        }
        let buffers = [input, vec![0; count / reduced * 8]];
        let layout = Layout::contiguous(shape).unwrap();
        for operation in [Operation::ArgMax, Operation::ArgMin] {
            let lowered = reduction(
                operation,
                &ElementGroup::input(DataType::F32, &layout),
                &[dim],
                false,
                None,
            )
            .unwrap();
            let schedules = candidates(&lowered.kernel, &lowered.bindings.sizes, false);
            assert!(!schedules.is_empty());
            for schedule in schedules {
                assert_matches(&lowered.kernel, schedule, &lowered.bindings, &buffers, 0.0);
            }
        }
    }
}

#[test]
fn test_scans_match_the_interpreter_under_element_and_row_schedules() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if skip_if_stub(runtime_for_device(Device::Cuda(0))).is_none() {
        return;
    }
    // Rows along the inner axis and down an outer one, short and longer than a block's threads, which take several points each.
    let cases: [(&[i64], i64); 4] = [
        (&[6, 40], 1),
        (&[3, 700], 1),
        (&[300, 7], 0),
        (&[4, 150, 9], 1),
    ];
    for (shape, dim) in cases {
        let count = usize::try_from(shape.iter().product::<i64>()).unwrap();
        let buffers = [f32_buffer(count, 1), vec![0; count * 4]];
        let layout = Layout::contiguous(shape).unwrap();
        for operation in [Operation::Cumsum, Operation::Cumprod] {
            let lowered =
                scan(operation, &ElementGroup::input(DataType::F32, &layout), dim).unwrap();
            for schedule in [
                Schedule::Elements {
                    threads: 64,
                    vector: 1,
                },
                Schedule::Rows {
                    threads: 32,
                    vector: 1,
                    tiles: [1, 1],
                },
                Schedule::Rows {
                    threads: 256,
                    vector: 1,
                    tiles: [1, 1],
                },
            ] {
                // Row schedules add each chunk's points in another order than the interpreter's walk.
                assert_matches(&lowered.kernel, schedule, &lowered.bindings, &buffers, 1e-4);
            }
        }
    }
}
