use crate::error::{Error, Result};

fn validate_span(values: usize, rows: usize, columns: usize, strides: [usize; 2]) -> Result<()> {
    if rows == 0 || columns == 0 {
        return Ok(());
    }
    let span = (rows - 1)
        .checked_mul(strides[0])
        .and_then(|row| {
            (columns - 1)
                .checked_mul(strides[1])
                .and_then(|column| row.checked_add(column))
        })
        .and_then(|last| last.checked_add(1))
        .ok_or(Error::SizeOverflow)?;
    if span > values {
        return Err(Error::LengthTooShort {
            name: "matmul operand".into(),
            minimum: span,
            actual: values,
        });
    }
    Ok(())
}

macro_rules! blocked_matmul {
    ($name:ident, $ty:ty) => {
        /// Portable GEMM over checked allocation-relative strides. Inputs may be
        /// transposed, broadcast, or padded. Output is contiguous and overwritten.
        /// Accumulation visits reduction indices in ascending order without
        /// reassociation or reduced precision. No input packing is required.
        #[expect(clippy::suboptimal_flops, reason = "the CPU reference keeps separate multiply and add rounding for reproducible comparisons")]
        pub fn $name(
            lhs: &[$ty],
            rhs: &[$ty],
            output: &mut [$ty],
            rows: usize,
            columns: usize,
            reduction: usize,
            lhs_strides: [usize; 2],
            rhs_strides: [usize; 2],
        ) -> Result<()> {
            let count = rows.checked_mul(columns).ok_or(Error::SizeOverflow)?;
            if output.len() != count {
                return Err(Error::LengthMismatch);
            }
            if count == 0 {
                return Ok(());
            }
            validate_span(lhs.len(), rows, reduction, lhs_strides)?;
            validate_span(rhs.len(), reduction, columns, rhs_strides)?;
            output.fill(0.0);
            if rhs_strides[1] != 1 {
                // Transposed weights already expose contiguous dot products.
                for row in 0..rows {
                    for column in 0..columns {
                        let mut sum = 0.0;
                        for k in 0..reduction {
                            sum += lhs[row * lhs_strides[0] + k * lhs_strides[1]]
                                * rhs[k * rhs_strides[0] + column * rhs_strides[1]];
                        }
                        output[row * columns + column] = sum;
                    }
                }
                return Ok(());
            }
            // Keep a small output tile hot while vectorizing across independent
            // columns. Each output retains the reference reduction order.
            for row_start in (0..rows).step_by(16) {
                for column_start in (0..columns).step_by(64) {
                    let column_end = columns.min(column_start + 64);
                    for k_start in (0..reduction).step_by(32) {
                        for row in row_start..rows.min(row_start + 16) {
                            let out = &mut output
                                [row * columns + column_start..row * columns + column_end];
                            for k in k_start..reduction.min(k_start + 32) {
                                let a = lhs[row * lhs_strides[0] + k * lhs_strides[1]];
                                let b = &rhs[k * rhs_strides[0] + column_start
                                    ..k * rhs_strides[0] + column_end];
                                for (out, &b) in out.iter_mut().zip(b) {
                                    *out += a * b;
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

blocked_matmul!(matmul_strided_f32, f32);

blocked_matmul!(matmul_strided_f64, f64);
