use std::result::Result as StdResult;

use singe_cudss::types::ConfigParameter;

mod common;

use std::{error::Error, slice, str::FromStr};

use singe_cuda::memory::DeviceMemory;
use singe_cudss::{
    config::Config,
    matrix::{CsrMatrixDescriptor, Matrix},
    types::{IndexBase, MatrixType, MatrixViewType, ReorderingAlgorithm},
};

use crate::common::{create_context, dense_matrix, dense_output, solve};

type Result<T> = StdResult<T, Box<dyn Error>>;

fn main() -> Result<()> {
    let (cuda, stream, context) = create_context()?;
    println!("cuDSS example: solving a Matrix Market SPD system");

    let matrix = parse_coordinate_matrix(include_str!("./data/matrix.mtx"))?;
    let rhs = parse_array_vector(include_str!("./data/rhs.mtx"))?;
    assert_eq!(matrix.rows, matrix.cols);
    assert_eq!(matrix.rows as usize, rhs.len());

    let row_offsets = DeviceMemory::from_slice(&cuda, &matrix.row_offsets)?;
    let col_indices = DeviceMemory::from_slice(&cuda, &matrix.col_indices)?;
    let a_values = DeviceMemory::from_slice(&cuda, &matrix.values)?;
    let b_values = DeviceMemory::from_slice(&cuda, &rhs)?;
    let mut x_values = DeviceMemory::<f64>::create_uninit(&cuda, rhs.len())?;

    let mut config = Config::create()?;
    let reordering = ReorderingAlgorithm::Default;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    unsafe { config.set_raw(ConfigParameter::ReorderingAlgorithm, &reordering) }?;
    let mut data = context.create_data()?;

    let b = dense_matrix(matrix.rows, 1, matrix.rows, &b_values)?;
    let mut x = dense_output(matrix.cols, 1, matrix.cols, x_values.output())?;
    let a = Matrix::create_csr(
        CsrMatrixDescriptor {
            rows: matrix.rows,
            columns: matrix.cols,
            nnz: matrix.values.len() as i64,
            matrix_type: MatrixType::Spd,
            view_type: MatrixViewType::Upper,
            index_base: IndexBase::Zero,
        },
        row_offsets.view(),
        None,
        col_indices.view(),
        a_values.view(),
    )?;

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        solve(
            slice::from_ref(&stream),
            &context,
            &config,
            &mut data,
            &a,
            &mut x,
            &b,
        )
    }?;
    cuda.synchronize()?;

    drop(x);
    // SAFETY: checked completed production initialized this compact output.
    let x_values = unsafe { x_values.assume_init() };
    let x = x_values.copy_to_host_vec()?;
    let residual = residual_norm(&matrix, &x, &rhs);
    println!("solution: {x:?}");
    println!("residual L2 norm: {residual:.3e}");
    assert!(residual <= 1.0e-12);

    println!("Example PASSED");
    Ok(())
}

#[derive(Debug)]
struct Csr {
    rows: i64,
    cols: i64,
    row_offsets: Vec<i32>,
    col_indices: Vec<i32>,
    values: Vec<f64>,
}

fn parse_coordinate_matrix(input: &str) -> Result<Csr> {
    let mut lines = input
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('%'));

    let shape = lines.next().ok_or("missing Matrix Market shape")?;
    let [rows, cols, declared_nnz] = parse_array::<usize, 3>(shape)?;
    let mut entries = Vec::with_capacity(declared_nnz);

    for line in lines {
        let mut parts = line.split_whitespace();
        let row = usize::from_str(parts.next().ok_or("missing row")?)? - 1;
        let col = usize::from_str(parts.next().ok_or("missing column")?)? - 1;
        let value = f64::from_str(parts.next().ok_or("missing value")?)?;
        entries.push((row, col, value));
    }

    if entries.len() != declared_nnz {
        return Err(format!("declared {declared_nnz} nonzeros, parsed {}", entries.len()).into());
    }

    entries.sort_by_key(|(row, col, _)| (*row, *col));
    let mut row_offsets = vec![0i32; rows + 1];
    let mut col_indices = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());

    for (row, col, value) in entries {
        if row >= rows || col >= cols {
            return Err("matrix entry is out of bounds".into());
        }
        if col < row {
            return Err("sample matrix is expected to contain upper-triangle entries".into());
        }
        row_offsets[row + 1] += 1;
        col_indices.push(col as i32);
        values.push(value);
    }

    for row in 0..rows {
        row_offsets[row + 1] += row_offsets[row];
    }

    Ok(Csr {
        rows: rows as i64,
        cols: cols as i64,
        row_offsets,
        col_indices,
        values,
    })
}

fn parse_array_vector(input: &str) -> Result<Vec<f64>> {
    let mut lines = input
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('%'));
    let shape = lines.next().ok_or("missing rhs shape")?;
    let [rows, cols] = parse_array::<usize, 2>(shape)?;
    if cols != 1 {
        return Err("only one right-hand side is supported".into());
    }

    let values: Vec<_> = lines.map(f64::from_str).collect::<StdResult<_, _>>()?;
    if values.len() != rows {
        return Err(format!("expected {rows} rhs values, parsed {}", values.len()).into());
    }
    Ok(values)
}

fn parse_array<T, const N: usize>(line: &str) -> Result<[T; N]>
where
    T: FromStr,
    T::Err: Error + 'static,
{
    let values: Vec<_> = line
        .split_whitespace()
        .map(T::from_str)
        .collect::<StdResult<_, _>>()?;
    values
        .try_into()
        .map_err(|values: Vec<T>| format!("expected {N} values, got {}", values.len()).into())
}

fn residual_norm(matrix: &Csr, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; matrix.rows as usize];
    for row in 0..matrix.rows as usize {
        for offset in matrix.row_offsets[row] as usize..matrix.row_offsets[row + 1] as usize {
            let col = matrix.col_indices[offset] as usize;
            let value = matrix.values[offset];
            ax[row] += value * x[col];
            if col != row {
                ax[col] += value * x[row];
            }
        }
    }

    ax.iter()
        .zip(b)
        .map(|(actual, expected)| {
            let diff = actual - expected;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}
