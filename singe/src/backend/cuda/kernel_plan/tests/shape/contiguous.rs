use super::*;

#[test]
fn test_cuda_eager_contiguous_f32_materializes_rank1_to_rank4_strided_views() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let rows = 3usize;
    let cols = 43usize;
    let shape =
        Shape::create([i64::try_from(rows).unwrap(), i64::try_from(cols).unwrap()]).unwrap();
    let values = (0..rows * cols)
        .map(|value| (value as f32).mul_add(0.25, -7.0))
        .collect::<Vec<_>>();
    let mut expected_transposed = Vec::with_capacity(rows * cols);
    for row in 0..cols {
        for col in 0..rows {
            expected_transposed.push(values[col * cols + row]);
        }
    }
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };
    let transposed = input.transpose(0, 1).unwrap();
    assert!(!transposed.is_contiguous());

    let output = transposed.contiguous().unwrap();

    assert!(output.is_contiguous());
    assert_eq!(output.storage_offset(), 0);
    assert_eq!(
        output.shape().dimensions(),
        &[i64::try_from(cols).unwrap(), i64::try_from(rows).unwrap()]
    );
    assert_close!(
        &output.to_vec::<f32>().unwrap(),
        &expected_transposed,
        1.0e-5
    );

    let offset_shape = Shape::create([130_i64]).unwrap();
    let offset_values = (0..130usize).map(|value| value as f32).collect::<Vec<_>>();
    let offset_input = Tensor::from_vec(Device::Cuda(0), &offset_shape, offset_values).unwrap();
    let offset_view = offset_input.narrow(0, 1, 129).unwrap();
    assert!(offset_view.is_contiguous());
    assert_eq!(offset_view.storage_offset(), 1);

    let offset_output = offset_view.contiguous().unwrap();

    let expected_offset = (1..130usize).map(|value| value as f32).collect::<Vec<_>>();
    assert!(offset_output.is_contiguous());
    assert_eq!(offset_output.storage_offset(), 0);
    assert_close!(&offset_output.to_vec::<f32>().unwrap(), &expected_offset);

    let rank3_shape = Shape::create([2_i64, 3, 43]).unwrap();
    let rank3_values = (0..2 * 3 * 43)
        .map(|value| (value as f32).mul_add(0.125, -5.0))
        .collect::<Vec<_>>();
    let mut expected_rank3 = Vec::with_capacity(rank3_values.len());
    for i in 0..43usize {
        for j in 0..2usize {
            for k in 0..3usize {
                expected_rank3.push(rank3_values[j * 3 * 43 + k * 43 + i]);
            }
        }
    }
    let rank3_input = Tensor::from_vec(Device::Cuda(0), &rank3_shape, rank3_values).unwrap();
    let rank3_permuted = rank3_input.permute(&[2, 0, 1]).unwrap();
    assert!(!rank3_permuted.is_contiguous());

    let rank3_output = rank3_permuted.contiguous().unwrap();

    assert!(rank3_output.is_contiguous());
    assert_eq!(rank3_output.storage_offset(), 0);
    assert_eq!(rank3_output.shape().dimensions(), &[43_i64, 2, 3]);
    assert_close!(
        &rank3_output.to_vec::<f32>().unwrap(),
        &expected_rank3,
        1.0e-5
    );
}

#[test]
fn test_cuda_eager_contiguous_f32_materializes_rank5_strided_view() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2_i64, 2, 2, 2, 2]).unwrap();
    let values = (0..32usize).map(|value| value as f32).collect::<Vec<_>>();
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };
    let permuted = input.permute(&[4, 0, 1, 2, 3]).unwrap();
    assert!(!permuted.is_contiguous());

    let output = permuted.contiguous().unwrap();
    let expected = (0..2)
        .flat_map(|last| (0..16).map(move |prefix| (prefix * 2 + last) as f32))
        .collect::<Vec<_>>();
    assert!(output.is_contiguous());
    assert_eq!(output.storage_offset(), 0);
    assert_eq!(output.to_vec::<f32>().unwrap(), expected);
}
