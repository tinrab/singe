//! Port of the cuSOLVER `getrf` sample: LU-factorize a 3x3 matrix with partial
//! pivoting and solve `A * x = b`.

use std::{error::Error, mem::MaybeUninit};

use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, view::DeviceOutputView,
};
use singe_cusolver::{
    context::Context,
    dense::{Getrf, GetrfArgs, Getrs, GetrsArgs},
    layout::{ByteWorkspaceMut, MatrixLayout, MatrixMut},
    params::Params,
    types::Operation,
};

const N: usize = 3;

fn main() -> Result<(), Box<dyn Error>> {
    let cuda_context = CudaContext::create_for_device(Device::new(0))?;
    let context = Context::create(&cuda_context)?;
    let params = Params::create()?;

    // Column-major storage of
    //     | 1 2 3 |
    // A = | 4 5 6 |
    //     | 7 8 10|
    let a = [1.0_f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0];
    let b = [1.0_f64, 2.0, 3.0];

    let mut a_device = DeviceMemory::from_slice(&cuda_context, &a)?;
    let mut b_device = DeviceMemory::from_slice(&cuda_context, &b)?;
    let mut pivots = DeviceMemory::<i64>::zeroed(&cuda_context, N)?;
    let mut info = DeviceMemory::<i32>::zeroed(&cuda_context, 1)?;

    let getrf = Getrf::<f64>::new(N, N);
    let sizes = getrf.workspace_size(&context, &params, MatrixLayout::new(N))?;
    let mut device_workspace = DeviceMemory::<u8>::zeroed(&cuda_context, sizes.device_bytes)?;
    let mut host_workspace = vec![MaybeUninit::<u8>::uninit(); sizes.host_bytes];

    getrf.run(
        &context,
        &params,
        GetrfArgs {
            a: MatrixMut::new(&mut a_device, N),
            pivots: Some(DeviceOutputView::from_output(&mut pivots)),
            workspace: ByteWorkspaceMut::new(&mut device_workspace, &mut host_workspace),
            dev_info: DeviceOutputView::from_output(&mut info),
        },
    )?;
    let factor_info = info.copy_to_host_vec()?[0];
    assert_eq!(factor_info, 0, "getrf reported info = {factor_info}");

    let factor = MatrixMut::new(&mut a_device, N);
    Getrs::<f64>::new(Operation::NonTranspose, N, 1).run(
        &context,
        &params,
        GetrsArgs {
            factor: factor.as_ref(),
            pivots: pivots.view(),
            rhs: MatrixMut::new(&mut b_device, N),
            dev_info: DeviceOutputView::from_output(&mut info),
        },
    )?;
    let solve_info = info.copy_to_host_vec()?[0];
    assert_eq!(solve_info, 0, "getrs reported info = {solve_info}");

    let x = b_device.copy_to_host_vec()?;
    println!("x = {x:?}");

    // Check the residual of the original system on the host.
    for row in 0..N {
        let lhs = (0..N).map(|col| a[col * N + row] * x[col]).sum::<f64>();
        assert!(
            (lhs - b[row]).abs() < 1.0e-9,
            "row {row}: {lhs} != {}",
            b[row]
        );
    }
    Ok(())
}
