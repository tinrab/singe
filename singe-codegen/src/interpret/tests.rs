use half::bf16;

use crate::{
    abi::Bindings,
    interpret::{
        InterpretError,
        minifloat::{E4M3, E5M2},
        run,
    },
    ir::{
        Access, AxisKind, BinaryOp, Combiner, ElementType, Expr, Kernel, Literal, OperandRole,
        Size, UnaryOp,
    },
};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

#[test]
fn test_broadcast_addition_reads_argument_sizes_from_the_bindings() {
    let mut kernel = Kernel::new("singe_add");
    let x = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let y = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let columns = kernel.add_size(Size::Static(3));
    let one = kernel.add_size(Size::Static(1));
    let i = kernel.add_axis(rows, AxisKind::Parallel);
    let j = kernel.add_axis(columns, AxisKind::Parallel);
    let matrix = |operand| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, columns), (j, one)],
    };
    let lhs = kernel.load(matrix(x));
    let rhs = kernel.load(Access {
        operand: y,
        base: None,
        indirect: Vec::new(),
        strides: vec![(j, one)],
    });
    let sum = kernel.binary(BinaryOp::Add, lhs, rhs);
    kernel.store(matrix(out), sum);

    let mut buffers = vec![
        f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        f32_bytes(&[10.0, 20.0, 30.0]),
        vec![0; 24],
    ];
    let bindings = Bindings {
        sizes: vec![2],
        scalars: Vec::new(),
    };
    run(&kernel, &bindings, &mut buffers).unwrap();
    assert_eq!(
        f32_values(&buffers[2]),
        [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );

    let error = run(&kernel, &Bindings::default(), &mut buffers).unwrap_err();
    assert_eq!(
        error,
        InterpretError::LengthMismatch {
            what: "argument size",
            expected: 1,
            actual: 0
        }
    );
    buffers[0].truncate(20);
    assert!(matches!(
        run(&kernel, &bindings, &mut buffers),
        Err(InterpretError::BufferTooSmall { operand: 0, .. })
    ));
}

#[test]
fn test_transposed_copies_of_bf16_are_exact() {
    let mut kernel = Kernel::new("singe_copy");
    let x = kernel.add_operand(ElementType::Bf16, OperandRole::Input);
    let out = kernel.add_operand(ElementType::Bf16, OperandRole::Output);
    let two = kernel.add_size(Size::Static(2));
    let three = kernel.add_size(Size::Static(3));
    let one = kernel.add_size(Size::Static(1));
    let i = kernel.add_axis(three, AxisKind::Parallel);
    let j = kernel.add_axis(two, AxisKind::Parallel);
    // `x` is 2 × 3; the output is its 3 × 2 transpose.
    let value = kernel.load(Access {
        operand: x,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, one), (j, three)],
    });
    kernel.store(
        Access {
            operand: out,
            base: None,
            indirect: Vec::new(),
            strides: vec![(i, two), (j, one)],
        },
        value,
    );
    let input = [1.5_f32, -2.25, 3.0, 4.0, 5.5, -6.0]
        .iter()
        .flat_map(|&value| bf16::from_f32(value).to_le_bytes())
        .collect::<Vec<_>>();
    let mut buffers = vec![input, vec![0; 12]];
    run(&kernel, &Bindings::default(), &mut buffers).unwrap();
    let output = buffers[1]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16::from_le_bytes(*chunk).to_f32())
        .collect::<Vec<_>>();
    assert_eq!(output, [1.5, 4.0, -2.25, 5.5, 3.0, -6.0]);
}

#[test]
fn test_softmax_rows_combine_two_reductions_in_one_kernel() {
    let mut kernel = Kernel::new("singe_softmax");
    let x = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let rows = kernel.add_size(Size::Static(2));
    let width = kernel.add_size(Size::Argument);
    let one = kernel.add_size(Size::Static(1));
    let row = kernel.add_axis(rows, AxisKind::Parallel);
    let column = kernel.add_axis(width, AxisKind::Reduction);
    let matrix = |operand| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: vec![(row, width), (column, one)],
    };
    let value = kernel.load(matrix(x));
    let maximum = kernel.reduce(Combiner::Max, vec![column], value);
    let shifted = kernel.binary(BinaryOp::Sub, value, maximum);
    let exponential = kernel.unary(UnaryOp::Exp, shifted);
    let sum = kernel.reduce(Combiner::Sum, vec![column], exponential);
    let probability = kernel.binary(BinaryOp::Div, exponential, sum);
    kernel.store(matrix(out), probability);

    let input = [1.0_f32, 2.0, 3.0, -1.0, 0.0, 4.0];
    let mut buffers = vec![f32_bytes(&input), vec![0; 24]];
    let bindings = Bindings {
        sizes: vec![3],
        scalars: Vec::new(),
    };
    run(&kernel, &bindings, &mut buffers).unwrap();
    let mut expected = Vec::new();
    for row in input.chunks(3) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials = row
            .iter()
            .map(|&value| (value - maximum).exp())
            .collect::<Vec<_>>();
        let sum = exponentials.iter().sum::<f32>();
        expected.extend(exponentials.iter().map(|value| value / sum));
    }
    for (actual, expected) in f32_values(&buffers[1]).iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
    }
}

#[test]
fn test_scalars_casts_and_selects_follow_their_types() {
    // out[i] = x[i] > c0 ? i64(x[i]) : -1, with NaN taking the else branch.
    let mut kernel = Kernel::new("singe_threshold");
    let x = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let out = kernel.add_operand(ElementType::I64, OperandRole::Output);
    let length = kernel.add_size(Size::Static(4));
    let one = kernel.add_size(Size::Static(1));
    let threshold = kernel.add_scalar(ElementType::F32);
    let i = kernel.add_axis(length, AxisKind::Parallel);
    let vector = |operand| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, one)],
    };
    let value = kernel.load(vector(x));
    let limit = kernel.add_node(ElementType::F32, Expr::Scalar(threshold));
    let above = kernel.binary(BinaryOp::Greater, value, limit);
    let whole = kernel.cast(ElementType::I64, value);
    let fallback = kernel.literal(ElementType::I64, Literal::Int(-1));
    let result = kernel.select(above, whole, fallback);
    kernel.store(vector(out), result);

    let mut buffers = vec![f32_bytes(&[2.75, 0.5, f32::NAN, -3.5]), vec![0; 32]];
    let bindings = Bindings {
        sizes: Vec::new(),
        scalars: vec![Literal::Float(1.0)],
    };
    run(&kernel, &bindings, &mut buffers).unwrap();
    let output = buffers[1]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| i64::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    assert_eq!(output, [2, -1, -1, -1]);
}

#[test]
fn test_softmax_sums_and_log_sum_exps_read_a_nested_reduction() {
    // logit[j] = sum over d of k[j, d]; out = softmax-weighted sum of v[j], and lse = log sum over j of exp(logit[j]).
    let mut kernel = Kernel::new("singe_attend");
    let k = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let v = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let lse = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let keys = kernel.add_size(Size::Static(3));
    let width = kernel.add_size(Size::Static(2));
    let one = kernel.add_size(Size::Static(1));
    let j = kernel.add_axis(keys, AxisKind::Reduction);
    let d = kernel.add_axis(width, AxisKind::Reduction);
    let key = kernel.load(Access {
        operand: k,
        base: None,
        indirect: Vec::new(),
        strides: vec![(j, width), (d, one)],
    });
    let logit = kernel.reduce(Combiner::Sum, vec![d], key);
    let value = kernel.load(Access {
        operand: v,
        base: None,
        indirect: Vec::new(),
        strides: vec![(j, one)],
    });
    let weighted = kernel.softmax_sum(vec![j], logit, value);
    let normalizer = kernel.reduce(Combiner::LogSumExp, vec![j], logit);
    let scalar = |operand| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: Vec::new(),
    };
    kernel.store(scalar(out), weighted);
    kernel.store(scalar(lse), normalizer);
    kernel.verify().unwrap();

    let mut buffers = vec![
        f32_bytes(&[1.0, 0.0, 0.5, 1.5, -1.0, -2.0]),
        f32_bytes(&[10.0, 20.0, 30.0]),
        vec![0; 4],
        vec![0; 4],
    ];
    run(&kernel, &Bindings::default(), &mut buffers).unwrap();
    let logits = [1.0_f64, 2.0, -3.0];
    let total = logits.iter().map(|logit| logit.exp()).sum::<f64>();
    let expected = logits
        .iter()
        .zip([10.0, 20.0, 30.0])
        .map(|(logit, value)| logit.exp() * value)
        .sum::<f64>()
        / total;
    assert!((f64::from(f32_values(&buffers[2])[0]) - expected).abs() < 1e-5);
    assert!((f64::from(f32_values(&buffers[3])[0]) - total.ln()).abs() < 1e-5);
}

#[test]
fn test_minifloats_round_to_nearest_even_and_overflow_as_pytorch_converts() {
    // The values and bits `torch.tensor(values).to(float8_e4m3fn)` and `float8_e5m2` give.
    let values = [
        0.0,
        -0.0,
        1.0,
        448.0,
        449.0,
        464.0,
        480.0,
        1000.0,
        f64::INFINITY,
        f64::NAN,
        1e-9,
        0.0019,
        57344.0,
        61440.0,
        70000.0,
    ];
    let e4m3 = [
        0x00, 0x80, 0x38, 0x7e, 0x7e, 0x7e, 0x7f, 0x7f, 0x7f, 0x7f, 0x00, 0x01, 0x7f, 0x7f, 0x7f,
    ];
    let e5m2 = [
        0x00, 0x80, 0x3c, 0x5f, 0x5f, 0x5f, 0x60, 0x64, 0x7c, 0x7f, 0x00, 0x18, 0x7b, 0x7c, 0x7c,
    ];
    for ((&value, &e4m3), &e5m2) in values.iter().zip(&e4m3).zip(&e5m2) {
        assert_eq!(E4M3.encode(value), e4m3, "e4m3 {value}");
        assert_eq!(E5M2.encode(value), e5m2, "e5m2 {value}");
    }
    // Every code that is not NaN decodes to a value that encodes back to it.
    for format in [E4M3, E5M2] {
        for bits in 0..=u8::MAX {
            let value = format.decode(bits);
            if !value.is_nan() {
                assert_eq!(format.encode(value), bits, "{format:?} {bits:#x}");
            }
        }
    }
}
