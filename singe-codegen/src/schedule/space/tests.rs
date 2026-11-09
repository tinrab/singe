use crate::{
    ir::{
        Access, AxisKind, BinaryOp, Combiner, ElementType, Expr, Kernel, Literal, OperandRole, Size,
    },
    schedule::{
        Schedule,
        contraction::Contraction,
        space::{candidates, operand_lengths},
    },
};

/// Builds `out[i, j] = x[i, j] + y[j]` for F32 rows of eight.
fn addition() -> Kernel {
    let mut kernel = Kernel::new("singe_space_add");
    let x = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let y = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let width = kernel.add_size(Size::Static(8));
    let one = kernel.add_size(Size::Static(1));
    let i = kernel.add_axis(rows, AxisKind::Parallel);
    let j = kernel.add_axis(width, AxisKind::Parallel);
    let matrix = |operand| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, width), (j, one)],
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
    kernel
}

/// Builds the row sums of BF16 rows of an argument width.
fn row_sums() -> Kernel {
    let mut kernel = Kernel::new("singe_space_sum");
    let x = kernel.add_operand(ElementType::Bf16, OperandRole::Input);
    let out = kernel.add_operand(ElementType::Bf16, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let width = kernel.add_size(Size::Argument);
    let one = kernel.add_size(Size::Static(1));
    let row = kernel.add_axis(rows, AxisKind::Parallel);
    let column = kernel.add_axis(width, AxisKind::Reduction);
    let value = kernel.load(Access {
        operand: x,
        base: None,
        indirect: Vec::new(),
        strides: vec![(row, width), (column, one)],
    });
    let sum = kernel.reduce(Combiner::Sum, vec![column], value);
    kernel.store(
        Access {
            operand: out,
            base: None,
            indirect: Vec::new(),
            strides: vec![(row, one)],
        },
        sum,
    );
    kernel
}

#[test]
fn test_spaces_follow_the_kernel_structure() {
    let elements = candidates(&addition(), &[100], true);
    assert!(elements.contains(&Schedule::Elements {
        threads: 256,
        vector: 4
    }));
    assert!(!elements.iter().any(|schedule| matches!(
        schedule,
        Schedule::Elements { vector: 8, .. } | Schedule::Rows { .. } | Schedule::TileRows { .. }
    )));
    assert!(elements.contains(&Schedule::Tiles { size: 1024 }));
    assert!(
        !candidates(&addition(), &[100], false)
            .iter()
            .any(|schedule| matches!(schedule, Schedule::Tiles { .. }))
    );

    let rows = candidates(&row_sums(), &[512, 128], true);
    assert!(rows.contains(&Schedule::Warps {
        threads: 256,
        vector: 8,
        lanes: 16,
        tiles: [1, 1]
    }));
    assert!(rows.contains(&Schedule::TileRows { width: 128 }));
    // Rows of 128 never take more threads than points.
    assert!(!rows.contains(&Schedule::Rows {
        threads: 256,
        vector: 1,
        tiles: [1, 1]
    }));
    assert!(
        !rows
            .iter()
            .any(|schedule| matches!(schedule, Schedule::Elements { .. }))
    );
}

#[test]
fn test_operand_lengths_cover_the_largest_offset_of_each_operand() {
    assert_eq!(operand_lengths(&addition(), &[3]), Some(vec![24, 8, 24]));
    assert_eq!(operand_lengths(&addition(), &[0]), Some(vec![0, 8, 0]));
    assert_eq!(operand_lengths(&row_sums(), &[4, 5]), Some(vec![20, 4]));
    assert_eq!(operand_lengths(&row_sums(), &[4]), None);
}

#[test]
fn test_operand_lengths_bound_computed_indices_and_reject_loaded_ones() {
    // out[i] = x[min(2 · i + 1, 6) · 3]: the clamp bounds the index, unlike one loaded from memory.
    let mut kernel = Kernel::new("singe_space_clamp");
    let x = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let indices = kernel.add_operand(ElementType::I64, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let rows = kernel.add_size(Size::Static(5));
    let one = kernel.add_size(Size::Static(1));
    let three = kernel.add_size(Size::Static(3));
    let i = kernel.add_axis(rows, AxisKind::Parallel);
    let index = kernel.add_node(ElementType::I64, Expr::AxisIndex(i));
    let two = kernel.literal(ElementType::I64, Literal::Int(2));
    let doubled = kernel.binary(BinaryOp::Mul, index, two);
    let unit = kernel.literal(ElementType::I64, Literal::Int(1));
    let position = kernel.binary(BinaryOp::Add, doubled, unit);
    let last = kernel.literal(ElementType::I64, Literal::Int(6));
    let clamped = kernel.binary(BinaryOp::Min, position, last);
    let value = kernel.load(Access {
        operand: x,
        base: None,
        indirect: vec![(clamped, three)],
        strides: Vec::new(),
    });
    let output = Access {
        operand: out,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, one)],
    };
    kernel.store(output.clone(), value);
    assert_eq!(operand_lengths(&kernel, &[]), Some(vec![19, 0, 5]));
    let loaded = kernel.load(Access {
        operand: indices,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, one)],
    });
    let gathered = kernel.load(Access {
        operand: x,
        base: None,
        indirect: vec![(loaded, one)],
        strides: Vec::new(),
    });
    kernel.store(output, gathered);
    assert_eq!(operand_lengths(&kernel, &[]), None);
}

#[test]
#[expect(
    clippy::many_single_char_names,
    reason = "the operands and axes take the names of the contraction formula"
)]
fn test_plain_contractions_get_tensor_core_tiles_over_their_rows_and_columns() {
    // out[i, j] = Σ_k x[i, k] · y[k, j], with 37 rows, 45 columns, and 70 points of contraction.
    let mut kernel = Kernel::new("singe_space_matmul");
    let x = kernel.add_operand(ElementType::Bf16, OperandRole::Input);
    let y = kernel.add_operand(ElementType::Bf16, OperandRole::Input);
    let out = kernel.add_operand(ElementType::Bf16, OperandRole::Output);
    let rows = kernel.add_size(Size::Static(37));
    let columns = kernel.add_size(Size::Static(45));
    let depth = kernel.add_size(Size::Static(70));
    let one = kernel.add_size(Size::Static(1));
    let i = kernel.add_axis(rows, AxisKind::Parallel);
    let j = kernel.add_axis(columns, AxisKind::Parallel);
    let k = kernel.add_axis(depth, AxisKind::Reduction);
    let access = |operand, strides| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides,
    };
    let lhs = kernel.load(access(x, vec![(i, depth), (k, one)]));
    let rhs = kernel.load(access(y, vec![(k, columns), (j, one)]));
    let product = kernel.binary(BinaryOp::Mul, lhs, rhs);
    let sum = kernel.reduce(Combiner::Sum, vec![k], product);
    kernel.store(access(out, vec![(i, columns), (j, one)]), sum);
    let contraction = Contraction::of(&kernel).unwrap();
    assert_eq!(
        (
            contraction.lhs,
            contraction.rhs,
            contraction.rows,
            contraction.columns
        ),
        (lhs, rhs, i, j)
    );
    assert!(contraction.batch.is_empty());
    let tensor_cores = |tile| {
        candidates(&kernel, &[], tile)
            .into_iter()
            .any(|schedule| matches!(schedule, Schedule::TileMatmul { .. }))
    };
    assert!(tensor_cores(true));
    assert!(!tensor_cores(false));
    let schedule = Schedule::TileMatmul {
        m: 16,
        n: 32,
        k: 16,
    };
    assert_eq!(schedule.launch(&kernel, &[]).unwrap().blocks, 3 * 2);
}
