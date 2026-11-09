use crate::{
    backend::registry::{Operation, TopKAttributes},
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    scalar::Scalar,
    shape::{Shape, normalize_dim_usize},
    utility::to_usize,
};

// The most values a selection top-k picks; more pick faster from a sorted row.
const SELECTED: i64 = 32;

// Whether CUDA runs `topk` as a selection, which picks few values from contiguous rows without sorting them, in eager calls and compiled graphs alike.
#[must_use]
pub fn selects_topk(data_type: DataType, layout: &Layout, topk: &TopKAttributes) -> bool {
    let dimensions = layout.shape().dimensions();
    !matches!(
        data_type,
        DataType::F16 | DataType::Bf16 | DataType::F8E4M3 | DataType::F8E5M2 | DataType::F4E2M1
    ) && layout.is_contiguous()
        && !dimensions.is_empty()
        && dimensions.iter().all(|&extent| extent > 0)
        && (1..=SELECTED).contains(&topk.k)
}

/// Returns the scalar fill required for an empty reduction, or `None` when normal reduction work remains.
///
/// An empty reduced axis uses zero for sum, one for product, and NaN for mean.
/// If only the output shape is empty, returns zero.
/// The specification must describe valid axes for `input`.
///
/// # Errors
///
/// Returns an error for minimum or maximum over an empty reduced axis, or for an unsupported operation when a reduced axis is empty.
///
/// # Panics
///
/// Panics if a reduction axis is outside the input rank.
pub fn empty_reduction_fill(
    input: &Shape,
    spec: &ReductionProblem,
    operation: Operation,
) -> Result<Option<Scalar>> {
    if spec
        .dimensions
        .iter()
        .any(|&axis| input.dimensions()[axis] == 0)
    {
        return Ok(Some(match operation {
            Operation::Sum => Scalar::Float(0.0),
            Operation::Product => Scalar::Float(1.0),
            Operation::Mean => Scalar::Float(f64::NAN),
            Operation::LogSumExp => Scalar::Float(f64::NEG_INFINITY),
            Operation::Max | Operation::Min => {
                return Err(Error::invalid_argument(
                    "min/max reduction over an empty axis",
                ));
            }
            _ => return Err(Error::unsupported_operation("empty reduction")),
        }));
    }
    Ok((spec.output_shape.element_count() == 0).then_some(Scalar::Float(0.0)))
}

/// Sorted reduction axes and the resulting output shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReductionProblem {
    /// Sorted, unique, normalized zero-based axes; empty for scalar input.
    pub dimensions: Vec<usize>,
    /// Shape after removing reduced axes or retaining them with extent one.
    pub output_shape: Shape,
}

/// Normalizes reduction axes and infers the output shape.
///
/// An empty axis list reduces every axis.
/// Negative axes count from the end.
/// `keepdim` retains reduced axes with extent one.
/// Scalars accept an empty list or one axis equal to zero or minus one and remain scalar.
///
/// # Errors
///
/// Returns an error if an axis is out of range, normalized axes repeat, or constructing the output shape fails.
pub fn reduction_problem(
    shape: &Shape,
    dimensions: &[i64],
    keepdim: bool,
) -> Result<ReductionProblem> {
    let rank = shape.rank();
    if rank == 0 {
        if dimensions.iter().any(|&axis| axis != 0 && axis != -1) {
            return Err(Error::DimensionOutOfBounds);
        }

        if dimensions.len() > 1 {
            return Err(Error::invalid_argument("duplicate reduction axes"));
        }
        return Ok(ReductionProblem {
            dimensions: vec![],
            output_shape: Shape::scalar(),
        });
    }

    let axes = if dimensions.is_empty() {
        (0..rank).collect::<Vec<_>>()
    } else {
        dimensions.to_vec()
    };
    let mut normalized = axes
        .iter()
        .map(|&dim| normalize_dim_usize(dim, rank))
        .collect::<Result<Vec<_>>>()?;
    normalized.sort_unstable();
    if normalized.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(Error::invalid_argument("duplicate reduction axes"));
    }

    let mut output_dims = shape.dimensions().to_vec();
    for &dim in normalized.iter().rev() {
        if keepdim {
            output_dims[dim] = 1;
        } else {
            output_dims.remove(dim);
        }
    }

    Ok(ReductionProblem {
        dimensions: normalized,
        output_shape: Shape::create(output_dims)?,
    })
}

/// Shape-preserving softmax metadata for one normalized axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftmaxProblem {
    /// Output shape, identical to the input shape.
    pub output_shape: Shape,
    /// Normalized zero-based softmax axis.
    pub axis: usize,
    /// Extent of the softmax axis.
    pub axis_dimension: i64,
    /// Total input and output element count.
    pub element_count: usize,
}

impl SoftmaxProblem {}

/// Normalizes `dim` and records shape-preserving softmax metadata.
///
/// Negative dimensions count from the end.
/// Empty tensors are retained as zero-work problems.
///
/// # Errors
///
/// Returns an error if `dim` is outside the input rank.
pub fn softmax_problem(shape: &Shape, dim: i64) -> Result<SoftmaxProblem> {
    let axis = normalize_dim_usize(dim, shape.rank())?;
    Ok(SoftmaxProblem {
        output_shape: shape.clone(),
        axis,
        axis_dimension: shape.dimensions()[axis],
        element_count: shape.element_count(),
    })
}

/// Selected-axis dimensions and output shape for top-k values and indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopKProblem {
    /// Input shape with the selected extent replaced by `k`.
    pub output_shape: Shape,
    /// Normalized zero-based selection axis.
    pub axis: usize,
    /// Number of values selected along the axis; zero is allowed.
    pub k: i64,
    /// Original input extent along the selection axis.
    pub axis_dimension: i64,
}

/// Checks `k` against a normalized selection axis and infers the output shape.
///
/// Negative dimensions count from the end.
/// `diagnostic_prefix` identifies operation-specific errors.
///
/// # Errors
///
/// Returns an error if `k` is negative, the axis is out of range, `k` exceeds its extent, a reported extent cannot fit the host size, or constructing the output shape fails.
pub fn topk_problem(
    shape: &Shape,
    k: i64,
    dim: i64,
    diagnostic_prefix: &'static str,
) -> Result<TopKProblem> {
    if k < 0 {
        return Err(Error::invalid_argument(format!("{diagnostic_prefix} k")));
    }

    let axis = normalize_dim_usize(dim, shape.rank())?;
    let axis_dimension = shape.dimensions()[axis];
    if axis_dimension < 0 || k > axis_dimension {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} k"),
            to_usize(axis_dimension, "backend spec dimension")?,
            to_usize(k, "backend spec dimension")?,
        ));
    }

    let mut output_dimensions = shape.dimensions().to_vec();
    output_dimensions[axis] = k;
    Ok(TopKProblem {
        output_shape: Shape::create(output_dimensions)?,
        axis,
        k,
        axis_dimension,
    })
}
