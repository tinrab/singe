use crate::{
    backend::spec,
    shape::{
        Shape,
        spatial::{Padding2d, Spatial2d},
    },
};

#[test]
fn test_conv2d_output_shape_matches_nchw_oihw() {
    let input = Shape::create([2, 4, 7, 9]).unwrap();
    let weight = Shape::create([6, 2, 3, 5]).unwrap();
    let output = spec::conv2d_output_shape(
        &input,
        &weight,
        Spatial2d::new(2, 1),
        Padding2d::symmetric(1, 2),
        Spatial2d::square(1),
        2,
    )
    .unwrap();
    assert_eq!(output.dimensions(), &[2, 6, 4, 9]);
}

#[test]
fn test_pool2d_output_shape_preserves_optional_batch_axis() {
    let batched = Shape::create([2, 4, 7, 9]).unwrap();
    let unbatched = Shape::create([4, 7, 9]).unwrap();
    assert_eq!(
        spec::pool2d_output_shape(
            &batched,
            Spatial2d::square(3),
            Spatial2d::square(2),
            Padding2d::symmetric(1, 1),
            Spatial2d::square(1),
            false
        )
        .unwrap()
        .dimensions(),
        &[2, 4, 4, 5],
    );
    assert_eq!(
        spec::pool2d_output_shape(
            &unbatched,
            Spatial2d::square(3),
            Spatial2d::square(2),
            Padding2d::symmetric(1, 1),
            Spatial2d::square(1),
            false,
        )
        .unwrap()
        .dimensions(),
        &[4, 4, 5],
    );
}

#[test]
fn test_matmul_gemm_and_concat_output_shapes_match_expected_values() {
    let lhs = Shape::create([2, 3, 4]).unwrap();
    let rhs = Shape::create([2, 4, 5]).unwrap();
    assert_eq!(
        spec::matmul_output_shape(&lhs, &rhs).unwrap().dimensions(),
        &[2, 3, 5],
    );

    let a = Shape::create([4, 3]).unwrap();
    let b = Shape::create([4, 5]).unwrap();
    assert_eq!(
        spec::gemm_output_shape(&a, &b, true, false)
            .unwrap()
            .dimensions(),
        &[3, 5],
    );

    let first = Shape::create([2, 3, 4]).unwrap();
    let second = Shape::create([2, 5, 4]).unwrap();
    assert_eq!(
        spec::concat_output_shape(&[first, second], 1)
            .unwrap()
            .dimensions(),
        &[2, 8, 4],
    );
}
