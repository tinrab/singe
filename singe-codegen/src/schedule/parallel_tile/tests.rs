use crate::{
    abi::Bindings,
    interpret::run,
    ir::{
        Access, AxisId, AxisKind, BinaryOp, Combiner, ElementType, Expr, Kernel, OperandRole, Size,
    },
    schedule::{
        Schedule, ScheduleError,
        parallel_tile::{tile_parallel_axis, tiled, tiled_axis},
    },
};

/// Builds `out[m, n] = Σ_k x[m, k] · w[n, k] + m` for `F32` operands, with `m` an argument and `n` and `k` static.
fn matmul(columns: u64, inner: u64) -> Kernel {
    let mut kernel = Kernel::new("singe_test");
    let x = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let w = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let n = kernel.add_size(Size::Static(columns));
    let k = kernel.add_size(Size::Static(inner));
    let one = kernel.add_size(Size::Static(1));
    let m_axis = kernel.add_axis(rows, AxisKind::Parallel);
    let n_axis = kernel.add_axis(n, AxisKind::Parallel);
    let k_axis = kernel.add_axis(k, AxisKind::Reduction);
    let load = |kernel: &mut Kernel, operand, strides| {
        kernel.load(Access {
            operand,
            base: None,
            indirect: Vec::new(),
            strides,
        })
    };
    let lhs = load(&mut kernel, x, vec![(m_axis, k), (k_axis, one)]);
    let rhs = load(&mut kernel, w, vec![(n_axis, k), (k_axis, one)]);
    let product = kernel.binary(BinaryOp::Mul, lhs, rhs);
    let sum = kernel.reduce(Combiner::Sum, vec![k_axis], product);
    let index = kernel.add_node(ElementType::U64, Expr::AxisIndex(m_axis));
    let row = kernel.cast(ElementType::F32, index);
    let total = kernel.binary(BinaryOp::Add, sum, row);
    kernel.store(
        Access {
            operand: out,
            base: None,
            indirect: Vec::new(),
            strides: vec![(m_axis, n), (n_axis, one)],
        },
        total,
    );
    kernel
}

/// Returns the output of `kernel` with argument sizes `sizes` on inputs of `rows × inner` and `columns × inner` values.
fn output(kernel: &Kernel, sizes: Vec<u64>, rows: u64, columns: u64, inner: u64) -> Vec<f32> {
    let bytes = |count: u64, seed: u64| {
        (0..count)
            .flat_map(|index| ((((index * 7 + seed) % 11) as f32) - 5.0).to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let mut buffers = vec![
        bytes(rows * inner, 1),
        bytes(columns * inner, 3),
        vec![0; usize::try_from(rows * columns * 4).unwrap()],
    ];
    let bindings = Bindings {
        sizes,
        scalars: Vec::new(),
    };
    run(kernel, &bindings, &mut buffers).unwrap();
    buffers[2]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

#[test]
fn test_tiled_kernels_compute_what_their_source_computes() {
    let (rows, columns, inner) = (8, 6, 5);
    let kernel = matmul(columns, inner);
    let expected = output(&kernel, vec![rows], rows, columns, inner);
    // The weight, 30 elements, is shared along the rows; the 40 inputs along the columns.
    assert_eq!(tiled_axis(&kernel, &[rows]), Some(AxisId(1)));
    for (axis, factor) in [
        (AxisId(0), 2),
        (AxisId(0), 4),
        (AxisId(0), 8),
        (AxisId(1), 2),
        (AxisId(1), 3),
    ] {
        let (tiled_kernel, sizes) = tile_parallel_axis(&kernel, &[rows], axis, factor).unwrap();
        tiled_kernel.verify().unwrap();
        assert_eq!(
            output(&tiled_kernel, sizes, rows, columns, inner),
            expected,
            "axis {axis:?} by {factor}"
        );
    }
}

#[test]
fn test_schedules_tile_only_axes_their_factor_divides() {
    let kernel = matmul(6, 5);
    let schedule = |tile| Schedule::Warps {
        threads: 128,
        vector: 1,
        lanes: 32,
        tiles: [tile, 1],
    };
    assert_eq!(tiled(&kernel, schedule(1), &[8]).unwrap(), None);
    let (tiled_kernel, sizes) = tiled(&kernel, schedule(2), &[8]).unwrap().unwrap();
    // The columns are static, so the arguments stay; tiling the rows adds their new extent as an argument.
    assert_eq!(sizes, vec![8]);
    assert_eq!(
        tile_parallel_axis(&kernel, &[8], AxisId(0), 2).unwrap().1,
        vec![8, 4]
    );
    assert_eq!(tiled_kernel.stores.len(), 2);
    assert_eq!(
        tiled(&kernel, schedule(4), &[8]),
        Err(ScheduleError::TileFactor {
            factor: 4,
            extent: 6
        })
    );
    // A tiled launch covers the tiled kernel's points.
    assert_eq!(schedule(2).launch(&kernel, &[8]).unwrap().blocks, 6);
    assert_eq!(schedule(1).launch(&kernel, &[8]).unwrap().blocks, 12);
}

#[test]
fn test_two_tiles_cover_both_parallel_axes() {
    let (rows, columns, inner) = (8, 6, 5);
    let kernel = matmul(columns, inner);
    let expected = output(&kernel, vec![rows], rows, columns, inner);
    let schedule = Schedule::Rows {
        threads: 64,
        vector: 1,
        tiles: [2, 4],
    };
    let (tiled_kernel, sizes) = tiled(&kernel, schedule, &[rows]).unwrap().unwrap();
    // Two columns and four rows per point: the rows axis takes its new extent as an argument.
    assert_eq!(sizes, vec![rows, 2]);
    assert_eq!(tiled_kernel.stores.len(), 8);
    assert_eq!(output(&tiled_kernel, sizes, rows, columns, inner), expected);
    assert_eq!(schedule.launch(&kernel, &[rows]).unwrap().blocks, 6);
}
