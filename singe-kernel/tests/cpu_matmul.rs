use singe_kernel::{
    cpu::matmul::{matmul_f32, matmul_f64, matmul_strided_f32, matmul_strided_f64},
    error::Error,
};

#[test]
fn test_portable_gemm_matches_reference_order_across_tile_edges_and_transposes() {
    for (rows, columns, reduction) in [(0, 3, 4), (3, 0, 4), (3, 4, 0), (3, 7, 5), (17, 65, 33)] {
        for transpose_lhs in [false, true] {
            for transpose_rhs in [false, true] {
                let lhs = (0..rows * reduction)
                    .map(|i| (i % 19) as f32 / 7. - 1.)
                    .collect::<Vec<_>>();
                let rhs = (0..reduction * columns)
                    .map(|i| (i % 13) as f32 / 3. - 2.)
                    .collect::<Vec<_>>();
                let lhs_row = if transpose_lhs { rows } else { reduction };
                let rhs_row = if transpose_rhs { reduction } else { columns };
                let lhs_strides = if transpose_lhs {
                    [1, lhs_row]
                } else {
                    [lhs_row, 1]
                };
                let rhs_strides = if transpose_rhs {
                    [1, rhs_row]
                } else {
                    [rhs_row, 1]
                };
                let mut actual = vec![42_f32; rows * columns];
                matmul_strided_f32(
                    &lhs,
                    &rhs,
                    &mut actual,
                    rows,
                    columns,
                    reduction,
                    lhs_strides,
                    rhs_strides,
                )
                .unwrap();
                assert_eq!(
                    actual,
                    matmul_f32(
                        &lhs,
                        &rhs,
                        rows,
                        columns,
                        reduction,
                        lhs_row,
                        rhs_row,
                        transpose_lhs,
                        transpose_rhs
                    )
                );
                let lhs = lhs.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
                let mut actual = vec![42_f64; rows * columns];
                matmul_strided_f64(
                    &lhs,
                    &rhs,
                    &mut actual,
                    rows,
                    columns,
                    reduction,
                    lhs_strides,
                    rhs_strides,
                )
                .unwrap();
                assert_eq!(
                    actual,
                    matmul_f64(
                        &lhs,
                        &rhs,
                        rows,
                        columns,
                        reduction,
                        lhs_row,
                        rhs_row,
                        transpose_lhs,
                        transpose_rhs
                    )
                );
            }
        }
    }
}

#[test]
fn test_portable_gemm_checks_spans_and_supports_broadcast_strides() {
    let mut output = [0_f32; 4];
    matmul_strided_f32(
        &[2., 3.],
        &[4., 5., 6., 7.],
        &mut output,
        2,
        2,
        2,
        [0, 1],
        [2, 1],
    )
    .unwrap();
    assert_eq!(output, [26., 31., 26., 31.]);
    assert!(matches!(
        matmul_strided_f32(&[1.], &[1.], &mut output, 2, 2, 2, [2, 1], [2, 1]),
        Err(Error::LengthTooShort { .. })
    ));
    assert!(matches!(
        matmul_strided_f32(&[], &[], &mut [], usize::MAX, 2, 2, [2, 1], [2, 1]),
        Err(Error::SizeOverflow)
    ));
    assert!(matches!(
        matmul_strided_f32(&[1.], &[1.], &mut output, 2, 2, 2, [usize::MAX, 1], [2, 1]),
        Err(Error::SizeOverflow)
    ));
}
