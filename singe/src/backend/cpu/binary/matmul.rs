use singe_kernel::cpu::matmul::{matmul_strided_f32, matmul_strided_f64};

use crate::{
    backend::{
        cpu::iteration::typed_values,
        spec::{grouped_matmul_shape, matmul_problem},
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    layout::Layout,
    parallel::{self, Parallelism},
    shape::Shape,
    tensor::{Tensor, TensorDataType},
    utility::to_usize,
};

pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.device() != Device::Cpu || rhs.device() != Device::Cpu {
        return Err(Error::unsupported_operation("cpu matmul device"));
    }

    if lhs.data_type() != rhs.data_type() {
        return Err(Error::data_type_mismatch(
            "matmul operands",
            lhs.data_type(),
            rhs.data_type(),
        ));
    }
    match lhs.data_type() {
        DataType::F32 => multiply::<f32>(lhs, rhs),
        DataType::F64 => multiply::<f64>(lhs, rhs),
        DataType::Bool | DataType::F4E2M1 => Err(Error::unsupported_data_type("cpu matmul")),
        // Narrower floats multiply in `F32` and round once, as PyTorch's do.
        data_type if data_type.is_float() => {
            multiply::<f32>(&lhs.cast(DataType::F32)?, &rhs.cast(DataType::F32)?)?.cast(data_type)
        }
        // Integers multiply in `I64`, wrapping; narrowing back keeps the low bits a narrower product wraps to.
        data_type => {
            multiply::<i64>(&lhs.cast(DataType::I64)?, &rhs.cast(DataType::I64)?)?.cast(data_type)
        }
    }
}

/// Multiply-adds per batch below which matmul stays on one thread.
const MATMUL_GRAIN_FLOPS: usize = 1 << 18;

trait MatmulElement: TensorDataType + Copy + Send + Sync {
    const ZERO: Self;

    fn multiply(
        lhs: &[Self],
        rhs: &[Self],
        output: &mut [Self],
        rows: usize,
        columns: usize,
        reduction: usize,
        lhs_strides: [usize; 2],
        rhs_strides: [usize; 2],
    ) -> Result<()>;
}

macro_rules! matmul_element {
    ($ty:ty, $kernel:ident) => {
        impl MatmulElement for $ty {
            const ZERO: Self = 0.0;

            fn multiply(
                lhs: &[Self],
                rhs: &[Self],
                output: &mut [Self],
                rows: usize,
                columns: usize,
                reduction: usize,
                lhs_strides: [usize; 2],
                rhs_strides: [usize; 2],
            ) -> Result<()> {
                $kernel(
                    lhs,
                    rhs,
                    output,
                    rows,
                    columns,
                    reduction,
                    lhs_strides,
                    rhs_strides,
                )?;
                Ok(())
            }
        }
    };
}

matmul_element!(f32, matmul_strided_f32);

matmul_element!(f64, matmul_strided_f64);

impl MatmulElement for i64 {
    const ZERO: Self = 0;

    fn multiply(
        lhs: &[Self],
        rhs: &[Self],
        output: &mut [Self],
        rows: usize,
        columns: usize,
        reduction: usize,
        lhs_strides: [usize; 2],
        rhs_strides: [usize; 2],
    ) -> Result<()> {
        for row in 0..rows {
            for column in 0..columns {
                output[row * columns + column] = (0..reduction).fold(0_i64, |sum, inner| {
                    sum.wrapping_add(
                        lhs[row * lhs_strides[0] + inner * lhs_strides[1]]
                            .wrapping_mul(rhs[inner * rhs_strides[0] + column * rhs_strides[1]]),
                    )
                });
            }
        }
        Ok(())
    }
}

fn multiply<T: MatmulElement>(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let problem = matmul_problem(lhs.shape(), rhs.shape(), "matmul")?;
    let lhs_layout = if lhs.shape().rank() == 1 {
        lhs.layout().unsqueeze(0)?
    } else {
        lhs.layout().clone()
    };
    let rhs_layout = if rhs.shape().rank() == 1 {
        rhs.layout().unsqueeze(-1)?
    } else {
        rhs.layout().clone()
    };
    let mut lhs_dimensions = problem.batch_dimensions.clone();
    lhs_dimensions.extend([problem.lhs_rows, problem.inner]);
    let mut rhs_dimensions = problem.batch_dimensions.clone();
    rhs_dimensions.extend([problem.inner, problem.rhs_cols]);
    let lhs_layout = lhs_layout.expand(&Shape::create(lhs_dimensions)?)?;
    let rhs_layout = rhs_layout.expand(&Shape::create(rhs_dimensions)?)?;
    let lhs_values = typed_values::<T>(lhs)?;
    let rhs_values = typed_values::<T>(rhs)?;
    let batches = Shape::create(&problem.batch_dimensions)?.element_count();
    let mut output = vec![T::ZERO; problem.output_shape.element_count()];
    if output.is_empty() {
        return Tensor::from_vec(Device::Cpu, problem.output_shape, output);
    }

    let rows = usize::try_from(problem.lhs_rows).expect("validated matrix rows fit the host size");
    let columns =
        usize::try_from(problem.rhs_cols).expect("validated matrix columns fit the host size");
    let reduction =
        usize::try_from(problem.inner).expect("validated reduction dimension fits the host size");
    let matrix_len = rows
        .checked_mul(columns)
        .ok_or_else(|| Error::out_of_range("cpu matmul output elements"))?;
    let lhs_strides = matrix_strides(&lhs_layout)?;
    let rhs_strides = matrix_strides(&rhs_layout)?;
    // Split each batch's rows across threads when the work outweighs spawning them.
    let work = rows.saturating_mul(columns).saturating_mul(reduction);
    let threads = parallel::num_threads()
        .min(rows)
        .min(work / MATMUL_GRAIN_FLOPS)
        .max(1);
    let chunk_rows = rows.div_ceil(threads);
    for batch in 0..batches {
        let lhs_start = batch_offset(&lhs_layout, batch)?;
        let rhs_start = batch_offset(&rhs_layout, batch)?;
        if matrix_len == 0 || reduction == 0 {
            continue;
        }

        parallel::for_each_chunk_mut(
            &mut output[batch * matrix_len..(batch + 1) * matrix_len],
            chunk_rows * columns,
            if threads > 1 {
                Parallelism::Pool
            } else {
                Parallelism::Serial
            },
            |chunk, values| {
                T::multiply(
                    &lhs_values[lhs_start + chunk * chunk_rows * lhs_strides[0]..],
                    &rhs_values[rhs_start..],
                    values,
                    values.len() / columns,
                    columns,
                    reduction,
                    lhs_strides,
                    rhs_strides,
                )
            },
        )?;
    }
    Tensor::from_vec(Device::Cpu, problem.output_shape, output)
}

fn matrix_strides(layout: &Layout) -> Result<[usize; 2]> {
    let Some((_, &[rows, columns])) = layout.strides().split_last_chunk::<2>() else {
        return Err(Error::invalid_argument("cpu matmul operand rank"));
    };
    Ok([
        to_usize(rows, "cpu matmul row stride")?,
        to_usize(columns, "cpu matmul column stride")?,
    ])
}

fn batch_offset(layout: &Layout, mut linear: usize) -> Result<usize> {
    let (Some((batch, _)), Some((batch_strides, _))) = (
        layout.dimensions().split_last_chunk::<2>(),
        layout.strides().split_last_chunk::<2>(),
    ) else {
        return Err(Error::invalid_argument("cpu matmul operand rank"));
    };
    let mut offset = layout.storage_offset();
    for (&dimension, &stride) in batch.iter().zip(batch_strides).rev() {
        let dimension = to_usize(dimension, "cpu matmul batch dimension")?;
        offset += linear % dimension * to_usize(stride, "cpu matmul batch stride")?;
        linear /= dimension;
    }
    Ok(offset)
}

// Rows sorted by group times their group's weight, in `F64`, rounded once to the rows' type.
pub fn grouped_matmul(input: &Tensor, weights: &Tensor, offsets: &Tensor) -> Result<Tensor> {
    if [input, weights, offsets]
        .iter()
        .any(|tensor| tensor.device() != Device::Cpu)
    {
        return Err(Error::unsupported_operation("cpu grouped matmul device"));
    }

    let shape = grouped_matmul_shape(
        input.shape(),
        weights.shape(),
        (offsets.shape(), offsets.data_type()),
        "cpu grouped matmul",
    )?;
    let data_type = input.data_type();
    if !data_type.is_float() || weights.data_type() != data_type {
        return Err(Error::unsupported_data_type("cpu grouped matmul"));
    }

    let &[rows, inner] = input.shape().dimensions() else {
        return Err(Error::invalid_argument("cpu grouped matmul rows"));
    };
    let &[_, columns] = shape.dimensions() else {
        return Err(Error::invalid_argument("cpu grouped matmul columns"));
    };
    let extent = |value: i64| to_usize(value, "cpu grouped matmul extent");
    let (rows, inner, columns) = (extent(rows)?, extent(inner)?, extent(columns)?);
    let values = |tensor: &Tensor| tensor.cast(DataType::F64)?.contiguous()?.to_vec::<f64>();
    let (input, weights) = (values(input)?, values(weights)?);
    let offsets = offsets
        .cast(DataType::I64)?
        .to_vec::<i64>()?
        .into_iter()
        .map(extent)
        .collect::<Result<Vec<_>>>()?;
    let mut output = vec![0.0; rows * columns];
    for (group, bounds) in offsets.windows(2).enumerate() {
        if bounds[0] > bounds[1] || bounds[1] > rows {
            return Err(Error::invalid_argument(
                "cpu grouped matmul offsets must rise within the rows",
            ));
        }

        let weight = &weights[group * inner * columns..][..inner * columns];
        for row in bounds[0]..bounds[1] {
            let lhs = &input[row * inner..][..inner];
            for (column, out) in output[row * columns..][..columns].iter_mut().enumerate() {
                *out = lhs
                    .iter()
                    .enumerate()
                    .map(|(k, value)| value * weight[k * columns + column])
                    .sum();
            }
        }
    }
    Tensor::from_vec(Device::Cpu, shape, output)?.cast(data_type)
}
