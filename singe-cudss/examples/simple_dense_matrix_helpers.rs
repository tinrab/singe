mod common;

use singe_cuda::{memory::DeviceMemory, types::DevicePtr};
use singe_cudss::{
    error::Result,
    matrix::{DenseMatrixDescriptor, Matrix},
    types::{DataType, Layout, MatrixFormat},
};

fn main() -> Result<()> {
    println!("cuDSS example: dense matrix helper APIs");
    println!("cuDSS version: {}", singe_cudss::version()?);

    let rows = 6i64;
    let cols = rows / 2;
    let leading_dim = 2 * rows;
    let len = (leading_dim * cols) as usize;

    let host_values: Vec<_> = (1..=len).map(|value| value as f64).collect();
    let values = DeviceMemory::from_slice(&host_values)?;
    let replacement_values = DeviceMemory::from_slice(&{
        let mut values = vec![0.0f64; len];
        values[1] = -4.0;
        values
    })?;

    let mut matrix = Matrix::create_dense(
        DenseMatrixDescriptor {
            rows,
            columns: cols,
            leading_dim,
            layout: Layout::ColumnMajor,
        },
        &values,
    )?;

    let info = matrix.dense_info()?;
    assert_eq!(info.rows, rows);
    assert_eq!(info.columns, cols);
    assert_eq!(info.leading_dim, leading_dim);
    assert_eq!(info.values, unsafe {
        DevicePtr::from_raw(values.as_mut_ptr().cast())
    });
    assert_eq!(info.value_type, DataType::F64);
    assert_eq!(info.layout, Layout::ColumnMajor);
    println!("dense_info matched the original descriptor");

    matrix.set_values(&replacement_values)?;
    let info = matrix.dense_info()?;
    assert_eq!(info.values, unsafe {
        DevicePtr::from_raw(replacement_values.as_mut_ptr().cast())
    });

    let copied = replacement_values.copy_to_host_vec()?;
    assert_eq!(copied[1], -4.0);
    println!("set_values replaced the dense value buffer");

    assert!(matrix.format()?.contains(MatrixFormat::DENSE));
    println!("format reports a dense matrix");
    println!("Example PASSED");
    Ok(())
}
