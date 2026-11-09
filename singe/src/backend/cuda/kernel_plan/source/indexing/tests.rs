use crate::{
    Device, Error, Shape, Tensor,
    backend::{
        cuda::kernel_plan,
        registry::{Operation, OperationAttributes},
    },
};

#[test]
fn test_gather_kernel_reports_out_of_range_indices_without_host_preflight() {
    let input = Tensor::from_vec(
        Device::Cuda(0),
        Shape::create([3]).unwrap(),
        vec![1_f32, 2., 3.],
    )
    .unwrap();
    let indices = Tensor::from_vec(
        Device::Cuda(0),
        Shape::create([4]).unwrap(),
        vec![i64::MIN, -4, 3, i64::MAX],
    )
    .unwrap();
    // The kernel itself reports the out-of-range indices, without a host preflight, and writes zero instead of addressing memory outside the input.
    let error = kernel_plan::execute(
        Operation::Take,
        &OperationAttributes::IndexSelect(0),
        &[&input, &indices],
    )
    .unwrap_err();
    assert!(matches!(error, Error::IndexOutOfBounds));
}
