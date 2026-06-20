use singe_cublas::{
    blas::level3::gemm_grouped_batched_ex,
    context::Context,
    error::Result,
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
    types::DevicePtr,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a0 = DeviceMemory::from_slice(&[1.0_f32, 3.0, 2.0, 4.0])?;
    let a1 = DeviceMemory::from_slice(&[5.0_f32, 7.0, 6.0, 8.0])?;
    let a2 = DeviceMemory::from_slice(&[1.0_f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])?;
    let b0 = DeviceMemory::from_slice(&[5.0_f32, 7.0, 6.0, 8.0])?;
    let b1 = DeviceMemory::from_slice(&[9.0_f32, 11.0, 10.0, 12.0])?;
    let b2 = DeviceMemory::from_slice(&[4.0_f32, 7.0, 10.0, 5.0, 8.0, 11.0, 6.0, 9.0, 12.0])?;
    let c0 = DeviceMemory::<f32>::zeroes(4)?;
    let c1 = DeviceMemory::<f32>::zeroes(4)?;
    let c2 = DeviceMemory::<f32>::zeroes(9)?;

    let a = [
        unsafe { DevicePtr::from_raw(a0.as_ptr().cast_mut().cast::<()>()) },
        unsafe { DevicePtr::from_raw(a1.as_ptr().cast_mut().cast::<()>()) },
        unsafe { DevicePtr::from_raw(a2.as_ptr().cast_mut().cast::<()>()) },
    ];
    let b = [
        unsafe { DevicePtr::from_raw(b0.as_ptr().cast_mut().cast::<()>()) },
        unsafe { DevicePtr::from_raw(b1.as_ptr().cast_mut().cast::<()>()) },
        unsafe { DevicePtr::from_raw(b2.as_ptr().cast_mut().cast::<()>()) },
    ];
    let mut c = [
        unsafe { DevicePtr::from_raw(c0.as_ptr().cast_mut().cast::<()>()) },
        unsafe { DevicePtr::from_raw(c1.as_ptr().cast_mut().cast::<()>()) },
        unsafe { DevicePtr::from_raw(c2.as_ptr().cast_mut().cast::<()>()) },
    ];

    gemm_grouped_batched_ex(
        &cublas,
        &[Operation::NonTranspose, Operation::NonTranspose],
        &[Operation::NonTranspose, Operation::NonTranspose],
        &[2, 3],
        &[2, 3],
        &[2, 3],
        &[1.0_f32, 1.0_f32],
        &a,
        DataType::F32,
        &[2, 3],
        &b,
        DataType::F32,
        &[2, 3],
        &[0.0_f32, 0.0_f32],
        &mut c,
        DataType::F32,
        &[2, 3],
        &[2, 1],
        ComputeType::F32,
    )?;

    assert_eq!(c0.copy_to_host_vec()?, vec![19.0, 43.0, 22.0, 50.0]);
    assert_eq!(c1.copy_to_host_vec()?, vec![111.0, 151.0, 122.0, 166.0]);
    assert_eq!(
        c2.copy_to_host_vec()?,
        vec![48.0, 111.0, 174.0, 54.0, 126.0, 198.0, 60.0, 141.0, 222.0]
    );

    println!("gemm_grouped_batched_ex c0: {:?}", c0.copy_to_host_vec()?);
    println!("gemm_grouped_batched_ex c1: {:?}", c1.copy_to_host_vec()?);
    println!("gemm_grouped_batched_ex c2: {:?}", c2.copy_to_host_vec()?);
    Ok(())
}
