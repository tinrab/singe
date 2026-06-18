use std::{error::Error, mem::size_of};

use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, matmul},
    },
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext,
    data_type::DataType,
    device::Device,
    memory::DeviceMemory,
    types::{Complex32, f16},
};

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[f16::from_f32(1.0), f16::from_f32(2.0)])?;
    let b = DeviceMemory::from_slice(&[f16::from_f32(3.0), f16::from_f32(4.0)])?;
    let c = DeviceMemory::<f16>::zeroes(2)?;
    let mut d = DeviceMemory::<f16>::zeroes(2)?;

    let plane_offset = size_of::<f16>() as i64;

    let mut a_layout = MatrixLayout::create(DataType::ComplexF16, 1, 1, 1)?;
    a_layout.set_plane_offset(plane_offset)?;

    let mut b_layout = MatrixLayout::create(DataType::ComplexF16, 1, 1, 1)?;
    b_layout.set_plane_offset(plane_offset)?;

    let mut c_layout = MatrixLayout::create(DataType::ComplexF16, 1, 1, 1)?;
    c_layout.set_plane_offset(plane_offset)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::ComplexF32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;

    let alpha = Complex32::new(1.0, 0.0);
    let beta = Complex32::new(0.0, 0.0);
    // `matrix_transform` is not involved here: the plane offsets make cuBLASLt
    // interpret the single allocation as planar complex storage.
    matmul(
        &lt, &desc, &alpha, &a, &a_layout, &b, &b_layout, &beta, &c, &c_layout, &mut d, &c_layout,
        None, None, None,
    )?;

    let result = d.copy_to_host_vec()?;
    assert_eq!(result, vec![f16::from_f32(-5.0), f16::from_f32(10.0)]);
    println!("lt_planar_complex result: {result:?}");
    Ok(())
}
