//! Port of the cuSPARSE `spmv_csr` sample: `y = alpha * A * x + beta * y` with a
//! 4x4 CSR matrix.

use std::error::Error;

use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, view::DeviceOutputView,
};
use singe_cusparse::{
    context::Context,
    matrix::SparseMatrixDescriptor,
    scalar::Scalar,
    spmv::{spmv, spmv_buffer_size},
    types::{IndexBase, Operation, SpMvAlgorithm},
    vector::DenseVectorDescriptor,
};

const ROWS: usize = 4;
const COLS: usize = 4;

fn main() -> Result<(), Box<dyn Error>> {
    let cuda_context = CudaContext::create_for_device(Device::new(0))?;
    let context = Context::create(&cuda_context)?;

    //     | 1 0 2 3 |
    // A = | 0 4 0 0 |
    //     | 5 0 6 7 |
    //     | 0 8 0 9 |
    let row_offsets = [0_i32, 3, 4, 7, 9];
    let column_indices = [0_i32, 2, 3, 1, 0, 2, 3, 1, 3];
    let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let x = [1.0_f32, 2.0, 3.0, 4.0];
    let expected = [19.0_f32, 8.0, 51.0, 52.0];

    let mut row_offsets_device = DeviceMemory::from_slice(&cuda_context, &row_offsets)?;
    let mut column_indices_device = DeviceMemory::from_slice(&cuda_context, &column_indices)?;
    let mut values_device = DeviceMemory::from_slice(&cuda_context, &values)?;
    let mut x_device = DeviceMemory::from_slice(&cuda_context, &x)?;
    let mut y_device = DeviceMemory::<f32>::zeroed(&cuda_context, ROWS)?;

    let matrix = SparseMatrixDescriptor::create_csr(
        &context,
        ROWS,
        COLS,
        values.len(),
        Some(DeviceOutputView::from_output(&mut row_offsets_device)),
        Some(DeviceOutputView::from_output(&mut column_indices_device)),
        Some(DeviceOutputView::from_output(&mut values_device)),
        IndexBase::Zero,
    )?;
    let x_vector = DenseVectorDescriptor::create(
        &context,
        COLS,
        DeviceOutputView::from_output(&mut x_device),
    )?;
    let mut y_vector = DenseVectorDescriptor::create(
        &context,
        ROWS,
        DeviceOutputView::from_output(&mut y_device),
    )?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    // SAFETY: the descriptors describe the complete device buffers above.
    let buffer_size = unsafe {
        spmv_buffer_size::<f32>(
            &context,
            Operation::NonTranspose,
            Scalar::host(&alpha),
            &matrix,
            &x_vector,
            Scalar::host(&beta),
            &mut y_vector,
            SpMvAlgorithm::Default,
        )?
    };
    let mut buffer = DeviceMemory::<u8>::zeroed(&cuda_context, buffer_size.max(1))?;
    context.stream()?.sync_scope(|| {
        // SAFETY: the descriptors cover the complete device buffers above, the CSR indices
        // are in bounds, and every buffer stays borrowed until the scope drains.
        unsafe {
            spmv::<f32>(
                &context,
                Operation::NonTranspose,
                Scalar::host(&alpha),
                &matrix,
                &x_vector,
                Scalar::host(&beta),
                &mut y_vector,
                SpMvAlgorithm::Default,
                Some(DeviceOutputView::from_output(&mut buffer)),
            )
        }
    })?;
    drop((matrix, x_vector, y_vector));

    let y = y_device.copy_to_host_vec()?;
    println!("y = {y:?}");
    assert_eq!(y, expected);
    Ok(())
}
