use crate::{
    backend::{
        registry::{Operation, OperationAttributes},
        spec::{self, IndexingProblem},
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
    utility::to_i64,
};

#[derive(Debug, Clone, Copy)]
pub enum IndexBounds {
    Nonnegative,
    WrapNegative,
}

pub fn indexing_problem(
    operation: Operation,
    attributes: &OperationAttributes,
    input: &Shape,
    indices: &Shape,
) -> Result<(IndexingProblem, IndexBounds)> {
    let dim = || {
        attributes
            .index_select_dimension()
            .ok_or_else(|| Error::invalid_argument("indexing axis"))
    };
    match operation {
        Operation::Take => Ok((
            spec::gather_problem(input, indices, dim()?, "gather")?,
            IndexBounds::WrapNegative,
        )),
        Operation::IndexSelect => Ok((
            spec::index_select_problem(input, indices, dim()?, "index select")?,
            IndexBounds::Nonnegative,
        )),
        Operation::Embedding => {
            spec::embedding_problem(input, indices, "embedding")?;
            Ok((
                spec::gather_problem(input, indices, 0, "embedding")?,
                IndexBounds::Nonnegative,
            ))
        }
        _ => Err(Error::invalid_argument("indexing operation")),
    }
}

/// Reads logical index values at the explicit bounds-validation boundary.
/// CUDA callers must invoke this before recording/enqueueing an indexing kernel.
pub fn normalized_indices(
    indices: &Tensor,
    dimension: usize,
    bounds: IndexBounds,
) -> Result<Vec<usize>> {
    let values = match indices.data_type() {
        DataType::I64 => indices.to_vec::<i64>()?,
        DataType::I32 => indices
            .to_vec::<i32>()?
            .into_iter()
            .map(i64::from)
            .collect(),
        _ => {
            return Err(Error::unsupported_data_type(
                "index tensor must be i32 or i64",
            ));
        }
    };
    let dimension = to_i64(dimension, "indexing dimension")?;
    values
        .into_iter()
        .map(|index| {
            let index = if index < 0 && matches!(bounds, IndexBounds::WrapNegative) {
                index + dimension
            } else {
                index
            };
            if index < 0 || index >= dimension {
                return Err(Error::IndexOutOfBounds);
            }
            usize::try_from(index).map_err(|_| Error::IndexOutOfBounds)
        })
        .collect()
}
