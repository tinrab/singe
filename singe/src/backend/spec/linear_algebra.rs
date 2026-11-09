use std::collections::{HashMap, HashSet};

#[cfg(any(test, feature = "onnx"))]
use crate::backend::spec::{validate_non_negative, validate_rank};
use crate::{
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
    utility::to_usize,
};

/// Checked output shape for a mode-labelled scaled tensor contraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionProblem {
    /// Output dimensions in the requested output mode order, checked against C's shape.
    pub output_shape: Shape,
}

/// Checks operand mode ranks and shared extents, then infers the contraction output shape.
/// Output modes must have the same set as C's modes, and the resulting ordered shape must equal C's shape.
///
/// # Errors
///
/// Returns an error if a mode count differs from its operand rank, shared mode extents differ, an output mode is unknown, output and C mode sets or shapes differ, or the output shape cannot be represented.
pub fn contraction_problem(
    a_shape: &Shape,
    a_mode: &[i64],
    b_shape: &Shape,
    b_mode: &[i64],
    c_shape: &Shape,
    c_mode: &[i64],
    d_mode: &[i64],
) -> Result<ContractionProblem> {
    validate_mode_rank(a_shape, a_mode)?;
    validate_mode_rank(b_shape, b_mode)?;
    validate_mode_rank(c_shape, c_mode)?;

    let mut dims_by_mode = HashMap::new();
    insert_mode_dims(&mut dims_by_mode, a_shape, a_mode)?;
    insert_mode_dims(&mut dims_by_mode, b_shape, b_mode)?;
    insert_mode_dims(&mut dims_by_mode, c_shape, c_mode)?;

    let d_mode_set = d_mode.iter().copied().collect::<HashSet<_>>();
    let c_mode_set = c_mode.iter().copied().collect::<HashSet<_>>();

    for &mode in &d_mode_set {
        if !dims_by_mode.contains_key(&mode) {
            return Err(Error::out_of_range(format!(
                "contraction output mode {mode}"
            )));
        }
    }

    if d_mode_set != c_mode_set {
        return Err(Error::shape_mismatch(
            "contraction output modes",
            c_mode_set.iter().copied().collect::<Vec<_>>(),
            d_mode_set.iter().copied().collect::<Vec<_>>(),
        ));
    }

    let output_dims = d_mode
        .iter()
        .map(|mode| {
            dims_by_mode
                .get(mode)
                .copied()
                .ok_or_else(|| Error::out_of_range(format!("contraction output mode {mode}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let output_shape = Shape::create(output_dims)?;

    if &output_shape != c_shape {
        return Err(Error::shape_mismatch(
            "contraction output shape",
            c_shape.dimensions().to_vec(),
            output_shape.dimensions().to_vec(),
        ));
    }

    Ok(ContractionProblem { output_shape })
}

fn validate_mode_rank(shape: &Shape, modes: &[i64]) -> Result<()> {
    let rank = shape.dimensions().len();
    if rank != modes.len() {
        return Err(Error::length_mismatch(
            "contraction mode rank",
            rank,
            modes.len(),
        ));
    }
    Ok(())
}

fn insert_mode_dims(
    dims_by_mode: &mut HashMap<i64, i64>,
    shape: &Shape,
    modes: &[i64],
) -> Result<()> {
    for (&mode, &dim) in modes.iter().zip(shape.dimensions()) {
        match dims_by_mode.get(&mode) {
            Some(existing) if *existing != dim => {
                return Err(Error::length_mismatch(
                    format!("contraction mode {mode} dimension"),
                    to_usize(*existing, "backend spec dimension")?,
                    to_usize(dim, "backend spec dimension")?,
                ));
            }
            Some(_) => {}
            None => {
                dims_by_mode.insert(mode, dim);
            }
        }
    }
    Ok(())
}

#[cfg(any(test, feature = "onnx"))]
/// Infers vector, matrix, and broadcast-batch matmul output dimensions.
///
/// # Errors
///
/// Returns an error if either input is scalar, batch dimensions cannot broadcast, contraction dimensions differ, or a required shape cannot be represented.
pub fn matmul_output_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    Ok(matmul_problem(lhs, rhs, "matmul")?.output_shape)
}

#[cfg(any(test, feature = "onnx"))]
/// Computes a checked 2-D GEMM output shape, applying boundary transposition.
///
/// # Errors
///
/// Returns an error if either operand is not rank two, an effective dimension is negative, contraction dimensions differ, or the output shape cannot be represented.
pub fn gemm_output_shape(a: &Shape, b: &Shape, trans_a: bool, trans_b: bool) -> Result<Shape> {
    let a_dims = a.dimensions();
    let b_dims = b.dimensions();
    validate_rank("gemm a", "input", a_dims, 2, "MK")?;
    validate_rank("gemm b", "input", b_dims, 2, "KN")?;
    let m = if trans_a { a_dims[1] } else { a_dims[0] };
    let k_a = if trans_a { a_dims[0] } else { a_dims[1] };
    let k_b = if trans_b { b_dims[1] } else { b_dims[0] };
    let n = if trans_b { b_dims[0] } else { b_dims[1] };
    validate_non_negative("gemm rows", m)?;
    validate_non_negative("gemm columns", n)?;
    validate_non_negative("gemm lhs inner dimension", k_a)?;
    validate_non_negative("gemm rhs inner dimension", k_b)?;
    if k_a != k_b {
        return Err(Error::length_mismatch(
            "gemm inner dimension",
            to_usize(k_a, "backend spec dimension")?,
            to_usize(k_b, "backend spec dimension")?,
        ));
    }
    Shape::create([m, n])
}

/// Matrix multiplication dimensions after vector promotion and batch broadcasting.
/// The logical output shape omits axes introduced only to promote vector operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulProblem {
    /// Logical result shape after removing promoted vector axes.
    pub output_shape: Shape,
    /// Broadcast dimensions shared by the matrix batches.
    pub batch_dimensions: Vec<i64>,
    /// Left matrix row count, or one for a promoted left vector.
    pub lhs_rows: i64,
    /// Contraction dimension shared by the promoted matrices.
    pub inner: i64,
    /// Right matrix column count, or one for a promoted right vector.
    pub rhs_cols: i64,
}

impl MatmulProblem {
    /// Returns whether the output has no elements or the contraction dimension is zero.
    #[must_use]
    pub fn is_zero_work(&self) -> bool {
        self.output_shape.element_count() == 0 || self.inner == 0
    }
}

/// Resolves vector promotion, batch broadcasting, and contraction dimensions for matrix multiplication.
/// `diagnostic_prefix` identifies the operation in shape error messages.
///
/// # Errors
///
/// Returns an error if either input is scalar, batch dimensions cannot broadcast, contraction dimensions differ, or a required shape cannot be represented.
pub fn matmul_problem(
    lhs: &Shape,
    rhs: &Shape,
    diagnostic_prefix: &'static str,
) -> Result<MatmulProblem> {
    let lhs_vector = lhs.rank() == 1;
    let rhs_vector = rhs.rank() == 1;
    let mut lhs_dimensions = lhs.dimensions().to_vec();
    let mut rhs_dimensions = rhs.dimensions().to_vec();
    if lhs_vector {
        lhs_dimensions.insert(0, 1);
    }

    if rhs_vector {
        rhs_dimensions.push(1);
    }

    for (name, dimensions) in [("lhs", &lhs_dimensions), ("rhs", &rhs_dimensions)] {
        if dimensions.len() < 2 {
            return Err(Error::length_mismatch(
                format!("{diagnostic_prefix} {name} minimum rank"),
                2,
                dimensions.len(),
            ));
        }
    }

    let lhs_matrix = lhs_dimensions.len() - 2;
    let rhs_matrix = rhs_dimensions.len() - 2;
    let batch_dimensions = Shape::create(&lhs_dimensions[..lhs_matrix])?
        .broadcast_with(&Shape::create(&rhs_dimensions[..rhs_matrix])?)
        .map_err(|error| match error {
            Error::BroadcastMismatch { left, right, .. } => Error::broadcast_mismatch(
                format!("{diagnostic_prefix} batch dimensions"),
                left,
                right,
            ),
            other => other,
        })?
        .dimensions()
        .to_vec();
    let lhs_rows = lhs_dimensions[lhs_matrix];
    let inner = lhs_dimensions[lhs_matrix + 1];
    let rhs_rows = rhs_dimensions[rhs_matrix];
    let rhs_cols = rhs_dimensions[rhs_matrix + 1];
    if inner != rhs_rows {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} inner dimension"),
            to_usize(inner, "backend spec dimension")?,
            to_usize(rhs_rows, "backend spec dimension")?,
        ));
    }

    let mut output_dimensions = batch_dimensions.clone();
    if !lhs_vector {
        output_dimensions.push(lhs_rows);
    }

    if !rhs_vector {
        output_dimensions.push(rhs_cols);
    }
    Ok(MatmulProblem {
        output_shape: Shape::create(output_dimensions)?,
        batch_dimensions,
        lhs_rows,
        inner,
        rhs_cols,
    })
}

// How `scaled_mm` scales its product: by one value per operand, or by one per row of the left operand and one per column of the right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulScaling {
    Tensorwise,
    Rowwise,
}

// The checked shape of `torch._scaled_mm`'s product of FP8 `lhs [M, K]` and `rhs [K, N]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScaledMatmulProblem {
    pub rows: i64,
    pub depth: i64,
    pub columns: i64,
    pub scaling: MatmulScaling,
    pub output_shape: Shape,
}

// Checks `scaled_mm`'s operands: FP8 matrices `[M, K]` and `[K, N]`, not both `E5M2`, with `F32` scales of one value each, or `[M, 1]` and `[1, N]`.
pub fn scaled_matmul_problem(
    lhs: (&Shape, DataType),
    rhs: (&Shape, DataType),
    lhs_scale: (&Shape, DataType),
    rhs_scale: (&Shape, DataType),
    diagnostic_prefix: &'static str,
) -> Result<ScaledMatmulProblem> {
    let invalid = |what: &str| Error::invalid_argument(format!("{diagnostic_prefix} {what}"));
    let fp8 = |data_type: DataType| matches!(data_type, DataType::F8E4M3 | DataType::F8E5M2);
    if !fp8(lhs.1)
        || !fp8(rhs.1)
        || (lhs.1 == DataType::F8E5M2 && rhs.1 == DataType::F8E5M2)
        || lhs_scale.1 != DataType::F32
        || rhs_scale.1 != DataType::F32
    {
        return Err(Error::unsupported_data_type(diagnostic_prefix));
    }

    let (&[rows, depth], &[rhs_depth, columns]) = (lhs.0.dimensions(), rhs.0.dimensions()) else {
        return Err(invalid("operand rank"));
    };
    if depth != rhs_depth {
        return Err(invalid("contracted extent"));
    }

    let scaling = if lhs_scale.0.element_count() == 1 && rhs_scale.0.element_count() == 1 {
        MatmulScaling::Tensorwise
    } else if lhs_scale.0.dimensions() == [rows, 1] && rhs_scale.0.dimensions() == [1, columns] {
        MatmulScaling::Rowwise
    } else {
        return Err(invalid("scale shapes"));
    };
    Ok(ScaledMatmulProblem {
        rows,
        depth,
        columns,
        scaling,
        output_shape: Shape::create([rows, columns])?,
    })
}

// Whether cuBLASLt's FP8 tensor cores take `scaled_mm`'s operand layouts: a row-major `lhs`, a column-major `rhs`, as `torch._scaled_mm` requires, extents that are multiples of 16, and origins 16 elements apart, so every pointer and leading dimension is 16-byte aligned.
#[must_use]
pub fn takes_scaled_matmul(lhs: &Layout, rhs: &Layout, problem: &ScaledMatmulProblem) -> bool {
    let aligned = |value: i64| value > 0 && value % 16 == 0;
    let origin = |layout: &Layout| layout.storage_offset().is_multiple_of(16);
    aligned(problem.rows)
        && aligned(problem.depth)
        && aligned(problem.columns)
        && lhs.strides() == [problem.depth, 1]
        && rhs.strides() == [1, problem.depth]
        && origin(lhs)
        && origin(rhs)
}

// The output shape of a grouped matmul: rows `[M, K]`, sorted by group, whose group `g` is rows `offsets[g]..offsets[g + 1]`, each times its group's weight of `weights` `[G, K, N]`, into `[M, N]`; the offsets are `G + 1` `I32` or `I64` values.
pub fn grouped_matmul_shape(
    input: &Shape,
    weights: &Shape,
    offsets: (&Shape, DataType),
    diagnostic_prefix: &'static str,
) -> Result<Shape> {
    let invalid = |what: &str| Error::invalid_argument(format!("{diagnostic_prefix} {what}"));
    let (&[rows, inner], &[groups, weight_inner, columns], &[bounds]) = (
        input.dimensions(),
        weights.dimensions(),
        offsets.0.dimensions(),
    ) else {
        return Err(invalid(
            "needs [M, K] rows, [G, K, N] weights, and G + 1 offsets",
        ));
    };
    if !matches!(offsets.1, DataType::I32 | DataType::I64) {
        return Err(Error::unsupported_data_type(diagnostic_prefix));
    }

    if weight_inner != inner || bounds != groups + 1 {
        return Err(invalid("operand shapes"));
    }
    Shape::create([rows, columns])
}
