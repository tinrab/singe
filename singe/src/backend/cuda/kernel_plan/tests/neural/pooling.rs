use super::*;

#[test]
fn test_cuda_eager_avg_pool2d_f32_materializes_strided_input_and_padding_modes() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2_i64, 3, 5, 7]).unwrap();
    let values = (0..210usize)
        .map(|value| ((value % 53) as f32 - 26.0) * 0.0625)
        .collect::<Vec<_>>();
    let cpu_input = Tensor::from_vec(Device::Cpu, &shape, values.clone())
        .unwrap()
        .transpose(2, 3)
        .unwrap();
    let Some(cuda_input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };
    let cuda_input = cuda_input.transpose(2, 3).unwrap();
    assert!(!cuda_input.is_contiguous());

    for count_include_pad in [false, true] {
        let expected = cpu_input
            .avg_pool2d(
                (2, 3),
                (1, 2),
                (1, 1, 0, 1),
                (1, 1),
                true,
                count_include_pad,
            )
            .unwrap();

        let output = cuda_input
            .avg_pool2d(
                (2, 3),
                (1, 2),
                (1, 1, 0, 1),
                (1, 1),
                true,
                count_include_pad,
            )
            .unwrap();

        assert_eq!(output.shape(), expected.shape());
        assert!(output.is_contiguous());
        assert_close!(
            &output.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap()
        );
    }
}

#[test]
fn test_cuda_eager_avg_pool2d_f32_rejects_dilation() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([1_i64, 1, 4, 4]).unwrap();
    let values = (0..16usize).map(|value| value as f32).collect::<Vec<_>>();
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };

    assert!(matches!(
        input.avg_pool2d((2, 2), (1, 1), (0, 0, 0, 0), (2, 1), false, false),
        Err(Error::InvalidArgument { ref name }) if name == "operation pool2d avg pool dilation"
    ));
}

#[test]
fn test_cuda_eager_max_pool2d_f32_materializes_tail_output() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2_i64, 3, 6, 7]).unwrap();
    let values = (0..252usize)
        .map(|value| ((value % 37) as f32 - 18.0) * 0.125)
        .collect::<Vec<_>>();
    let cpu_input = Tensor::from_vec(Device::Cpu, &shape, values.clone()).unwrap();
    let expected = cpu_input
        .max_pool2d((3, 2), (2, 1), (1, 0, 1, 1), (2, 1), true)
        .unwrap();
    let Some(cuda_input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };

    let output = cuda_input
        .max_pool2d((3, 2), (2, 1), (1, 0, 1, 1), (2, 1), true)
        .unwrap();

    assert_eq!(output.shape().dimensions(), expected.shape().dimensions());
    assert!(output.is_contiguous());
    assert_close!(
        &output.to_vec::<f32>().unwrap(),
        &expected.to_vec::<f32>().unwrap()
    );
}

#[test]
fn test_cuda_eager_max_pool2d_f32_preserves_negative_infinity() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([1_i64, 1, 2, 2]).unwrap();
    let values = vec![f32::NEG_INFINITY; 4];
    let cpu_input = Tensor::from_vec(Device::Cpu, &shape, values.clone()).unwrap();
    let expected = cpu_input
        .max_pool2d((2, 2), (1, 1), (0, 0, 0, 0), (1, 1), true)
        .unwrap();
    let Some(cuda_input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };

    let output = cuda_input
        .max_pool2d((2, 2), (1, 1), (0, 0, 0, 0), (1, 1), true)
        .unwrap();
    let actual = output.to_vec::<f32>().unwrap();
    let expected = expected.to_vec::<f32>().unwrap();

    assert_eq!(output.shape().dimensions(), &[1_i64, 1, 1, 1]);
    assert_eq!(actual.len(), 1);
    assert_eq!(expected.len(), 1);
    assert!(actual[0].is_infinite() && actual[0].is_sign_negative());
    assert!(expected[0].is_infinite() && expected[0].is_sign_negative());
}
