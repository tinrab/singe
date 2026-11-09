use super::*;

#[test]
fn test_cuda_eager_dequantize_rejects_invalid_scale_and_materializes_strided_byte_input() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2_i64, 65]).unwrap();
    let values = (0..130usize)
        .map(|value| u8::try_from((value * 13) % 256).unwrap())
        .collect::<Vec<_>>();
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };
    assert!(matches!(
        input.dequantize(0.0, 0),
        Err(Error::InvalidArgument { ref name }) if name == "quantization scale"
    ));

    let transposed = input.transpose(0, 1).unwrap();
    assert!(!transposed.is_contiguous());
    assert_eq!(
        transposed
            .dequantize(0.5, 0)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        transposed
            .contiguous()
            .unwrap()
            .dequantize(0.5, 0)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    );
}

#[test]
fn test_cuda_eager_cast_f32_bool_materializes_strided_input_and_rejects_other_targets() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([3_i64, 43]).unwrap();
    let values = (0..129usize)
        .map(|value| {
            if value % 7 == 0 {
                0.0
            } else {
                value as f32 - 64.0
            }
        })
        .collect::<Vec<_>>();
    let cpu_input = Tensor::from_vec(Device::Cpu, &shape, values.clone())
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let expected = cpu_input
        .contiguous()
        .unwrap()
        .cast(DataType::Bool)
        .unwrap();
    let Some(cuda_input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };
    let cuda_input = cuda_input.transpose(0, 1).unwrap();
    assert!(!cuda_input.is_contiguous());

    let output = cuda_input.cast(DataType::Bool).unwrap();

    assert_eq!(output.shape().dimensions(), expected.shape().dimensions());
    assert!(output.is_contiguous());
    assert_eq!(
        output.to_vec::<bool>().unwrap(),
        expected.to_vec::<bool>().unwrap()
    );
    // Kernels hold no packed FP4 values, so the cast has no kernel.
    assert_unsupported_target(
        cuda_input.cast(DataType::F4E2M1),
        UnsupportedTarget::DataType,
    );
}

#[test]
fn test_cuda_eager_where_rejects_f32_conditions() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    for len in [0usize, 1, 127, 128, 129, 257] {
        let shape = Shape::create([i64::try_from(len).unwrap()]).unwrap();
        let Some(condition) = skip_if_stub(Tensor::full(Device::Cuda(0), &shape, 1.0_f32)) else {
            return;
        };
        let values = Tensor::full(Device::Cuda(0), &shape, 2.0_f32).unwrap();
        assert_unsupported_target(
            cuda_kernels::elementwise(
                Operation::Where,
                &OperationAttributes::None,
                &[&condition, &values, &values],
            ),
            UnsupportedTarget::DataType,
        );
    }
}
