use super::*;

#[test]
fn test_cuda_eager_rotary_embedding_f32_materializes_tail_output() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let input_shape = Shape::create([2_i64, 3, 5, 7]).unwrap();
    let trig_shape = Shape::create([5_i64, 3]).unwrap();
    let input_values = rotary_input_values(2, 3, 5, 7);
    let cos_values = rotary_trig_values(5, 3, true);
    let sin_values = rotary_trig_values(5, 3, false);
    let expected =
        expected_rotary_embedding_f32(&input_values, &cos_values, &sin_values, 2, 3, 5, 7, 3);
    let Some(input) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &input_shape,
        input_values,
    )) else {
        return;
    };
    let cos = Tensor::from_vec(Device::Cuda(0), &trig_shape, cos_values).unwrap();
    let sin = Tensor::from_vec(Device::Cuda(0), &trig_shape, sin_values).unwrap();

    let output = input.rotary_embedding(&cos, &sin).unwrap();

    assert_eq!(output.shape().dimensions(), &[2_i64, 3, 5, 7]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-5);
}

#[test]
fn test_cuda_eager_rotary_embedding_f32_materializes_strided_input() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let input_shape = Shape::create([2_i64, 5, 3, 7]).unwrap();
    let trig_shape = Shape::create([5_i64, 2]).unwrap();
    let input_values = (0..210usize)
        .map(|value| ((value % 23) as f32 - 11.0) * 0.03125)
        .collect::<Vec<_>>();
    let cos_values = rotary_trig_values(5, 2, true);
    let sin_values = rotary_trig_values(5, 2, false);
    let Some(input) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &input_shape,
        input_values,
    )) else {
        return;
    };
    let input = input.permute(&[0, 2, 1, 3]).unwrap();
    let cos = Tensor::from_vec(Device::Cuda(0), &trig_shape, cos_values).unwrap();
    let sin = Tensor::from_vec(Device::Cuda(0), &trig_shape, sin_values).unwrap();
    assert!(!input.is_contiguous());
    assert_eq!(
        input
            .rotary_embedding(&cos, &sin)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        input
            .contiguous()
            .unwrap()
            .rotary_embedding(&cos, &sin)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    );
}

#[test]
fn test_cuda_eager_rotary_embedding_f32_handles_zero_rotary_pairs() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let input_shape = Shape::create([1_i64, 2, 3, 5]).unwrap();
    let trig_shape = Shape::create([3_i64, 0]).unwrap();
    let values = (0..30usize)
        .map(|value| (value as f32).mul_add(0.125, -2.0))
        .collect::<Vec<_>>();
    let Some(input) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &input_shape,
        values.clone(),
    )) else {
        return;
    };
    let cos = Tensor::from_vec(Device::Cuda(0), &trig_shape, Vec::<f32>::new()).unwrap();
    let sin = Tensor::from_vec(Device::Cuda(0), &trig_shape, Vec::<f32>::new()).unwrap();

    let output = input.rotary_embedding(&cos, &sin).unwrap();

    assert_eq!(output.shape().dimensions(), &[1_i64, 2, 3, 5]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &values);
}
