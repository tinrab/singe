use singe_cublas::{
    blas::level3::gemm_batched_ex,
    context::Context,
    error::Result,
    types::{ComputeType, GemmAlgorithm, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
    types::DevicePtr,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a0 = DeviceMemory::from_slice(&[1.0_f64, 3.0, 2.0, 4.0])?;
    let a1 = DeviceMemory::from_slice(&[5.0_f64, 7.0, 6.0, 8.0])?;
    let b0 = DeviceMemory::from_slice(&[5.0_f64, 7.0, 6.0, 8.0])?;
    let b1 = DeviceMemory::from_slice(&[9.0_f64, 11.0, 10.0, 12.0])?;
    let c0 = DeviceMemory::<f64>::zeroes(4)?;
    let c1 = DeviceMemory::<f64>::zeroes(4)?;

    let a = [
        unsafe { DevicePtr::from_raw(a0.as_ptr().cast_mut().cast::<()>()) },
        unsafe { DevicePtr::from_raw(a1.as_ptr().cast_mut().cast::<()>()) },
    ];
    let b = [
        unsafe { DevicePtr::from_raw(b0.as_ptr().cast_mut().cast::<()>()) },
        unsafe { DevicePtr::from_raw(b1.as_ptr().cast_mut().cast::<()>()) },
    ];
    let mut c = [
        unsafe { DevicePtr::from_raw(c0.as_ptr().cast_mut().cast::<()>()) },
        unsafe { DevicePtr::from_raw(c1.as_ptr().cast_mut().cast::<()>()) },
    ];

    gemm_batched_ex(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        &1.0,
        &a,
        DataType::F64,
        2,
        &b,
        DataType::F64,
        2,
        &0.0,
        &mut c,
        DataType::F64,
        2,
        ComputeType::F64,
        GemmAlgorithm::Default,
    )?;

    assert_eq!(c0.copy_to_host_vec()?, vec![19.0, 43.0, 22.0, 50.0]);
    assert_eq!(c1.copy_to_host_vec()?, vec![111.0, 151.0, 122.0, 166.0]);

    println!("gemm_batched c0: {:?}", c0.copy_to_host_vec()?);
    println!("gemm_batched c1: {:?}", c1.copy_to_host_vec()?);
    Ok(())
}
