use super::*;

#[test]
fn test_cuda_eager_kernels_handle_offsets_layouts_broadcasts_and_integers() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([129_i64]).unwrap();
    let values = (0..129usize).map(|value| value as f32).collect::<Vec<_>>();
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };

    let i32_values = (0..129_i32).collect::<Vec<_>>();
    let i32_input = Tensor::from_vec(Device::Cuda(0), &shape, i32_values.clone()).unwrap();
    assert_eq!(
        i32_input.neg().unwrap().to_vec::<i32>().unwrap(),
        i32_values.iter().map(|value| -value).collect::<Vec<_>>()
    );

    let matrix_shape = Shape::create([3_i64, 43]).unwrap();
    let matrix_values = (0..129usize).map(|value| value as f32).collect::<Vec<_>>();
    let matrix = Tensor::from_vec(Device::Cuda(0), &matrix_shape, matrix_values).unwrap();
    let transposed = matrix.transpose(0, 1).unwrap();
    assert!(!transposed.is_contiguous());
    let transposed_values = (0..43usize)
        .flat_map(|col| (0..3usize).map(move |row| (row * 43 + col) as f32))
        .collect::<Vec<_>>();
    let neg_output = transposed.neg().unwrap();
    let expected_neg = transposed_values
        .iter()
        .map(|value| -*value)
        .collect::<Vec<_>>();
    assert_eq!(neg_output.shape().dimensions(), &[43_i64, 3]);
    assert!(neg_output.is_contiguous());
    assert_close!(&neg_output.to_vec::<f32>().unwrap(), &expected_neg);

    let scalar_output = transposed.add_scalar(1.5).unwrap();
    let expected_scalar = transposed_values
        .iter()
        .map(|value| *value + 1.5)
        .collect::<Vec<_>>();
    assert_close!(&scalar_output.to_vec::<f32>().unwrap(), &expected_scalar);

    let clamp_output = transposed.clamp(17.0, 73.0).unwrap();
    let expected_clamp = transposed_values
        .iter()
        .map(|value| value.clamp(17.0, 73.0))
        .collect::<Vec<_>>();
    assert_close!(&clamp_output.to_vec::<f32>().unwrap(), &expected_clamp);

    let rhs_values = (0..129usize)
        .map(|value| (value as f32).mul_add(0.25, -3.0))
        .collect::<Vec<_>>();
    let rhs = Tensor::from_vec(Device::Cuda(0), &matrix_shape, rhs_values).unwrap();
    let rhs = rhs.transpose(0, 1).unwrap();
    let rhs_transposed_values = (0..43usize)
        .flat_map(|col| (0..3usize).map(move |row| ((row * 43 + col) as f32).mul_add(0.25, -3.0)))
        .collect::<Vec<_>>();
    let add_output = transposed.add(&rhs).unwrap();
    let expected_add = transposed_values
        .iter()
        .zip(&rhs_transposed_values)
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    assert_close!(&add_output.to_vec::<f32>().unwrap(), &expected_add);

    let lerp_output = transposed.lerp(&rhs, 0.25).unwrap();
    let expected_lerp = transposed_values
        .iter()
        .zip(&rhs_transposed_values)
        .map(|(lhs, rhs)| lhs + (rhs - lhs) * 0.25)
        .collect::<Vec<_>>();
    assert_close!(&lerp_output.to_vec::<f32>().unwrap(), &expected_lerp);

    let offset_shape = Shape::create([130_i64]).unwrap();
    let offset_values = (0..130usize).map(|value| value as f32).collect::<Vec<_>>();
    let offset_input = Tensor::from_vec(Device::Cuda(0), &offset_shape, offset_values).unwrap();
    let offset_view = offset_input.narrow(0, 1, 129).unwrap();
    assert_eq!(offset_view.storage_offset(), 1);
    let offset_output = offset_view.neg().unwrap();
    let expected_offset_output = (1..130usize)
        .map(|value| -(value as f32))
        .collect::<Vec<_>>();
    assert_close!(
        &offset_output.to_vec::<f32>().unwrap(),
        &expected_offset_output
    );

    let scalar_shape = Shape::create([1_i64]).unwrap();
    let scalar = Tensor::from_vec(Device::Cuda(0), &scalar_shape, vec![1.0_f32]).unwrap();
    let output = scalar.add(&input).unwrap();
    assert_eq!(
        output.to_vec::<f32>().unwrap(),
        (1..130).map(|value| value as f32).collect::<Vec<_>>()
    );
}
