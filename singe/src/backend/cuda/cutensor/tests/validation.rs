use super::*;

#[test]
fn test_cuda_cutensor_contraction_rejects_invalid_inputs() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let a_shape = Shape::create([2, 3]).unwrap();
    let b_shape = Shape::create([3, 2]).unwrap();
    let c_shape = Shape::create([2, 2]).unwrap();
    let Some(a_f64) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &a_shape,
        vec![1.0_f64; 6],
    )) else {
        return;
    };
    let a = Tensor::from_vec(Device::Cuda(0), &a_shape, vec![1.0_f32; 6]).unwrap();
    let b = Tensor::from_vec(Device::Cuda(0), &b_shape, vec![1.0_f32; 6]).unwrap();
    let c = Tensor::from_vec(Device::Cuda(0), &c_shape, vec![0.0_f32; 4]).unwrap();

    let error =
        contraction(&a_f64, &[0, 2], &b, &[2, 1], &c, &[0, 1], &[0, 1], 1.0, 0.0).unwrap_err();
    assert!(
        matches!(error, Error::Unsupported { .. }),
        "expected detailed cuda eager contraction dtype error, got {error:?}"
    );
    let Error::Unsupported {
        operation, details, ..
    } = error
    else {
        return;
    };
    assert_eq!(operation, operation_name(Operation::Contraction));
    assert!(details.contains("providers [cutensor]"));
    assert!(details.contains("dtype F64"));
    assert!(details.contains("layout any strided"));
    assert!(details.contains("shape [2, 3]"));
    assert!(details.contains("input dtypes [F64, F32, F32]"));
    assert_shape_mismatch(contraction(
        &a,
        &[0],
        &b,
        &[2, 1],
        &c,
        &[0, 1],
        &[0, 1],
        1.0,
        0.0,
    ));
}
