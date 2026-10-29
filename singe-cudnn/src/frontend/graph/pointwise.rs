use std::{borrow::Cow, iter};

use crate::{
    backend::tensor::TensorId,
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, TensorOutputConfig},
        operation::{Operation, PointwiseOperation},
    },
    utility::to_i64,
};

impl Graph {
    /// Adds a pointwise operation to the graph.
    ///
    /// Pointwise operations apply elementwise unary, binary, or ternary modes selected by cuDNN's pointwise mode.
    ///
    /// # Errors
    /// Returns an error for a mode with the wrong tensor ports.
    pub fn pointwise_into(&mut self, operation: PointwiseOperation) -> Result<()> {
        operation.validate_tensor_ports()?;
        self.operations.push(Operation::Pointwise(operation));
        Ok(())
    }

    /// Infers a unary, binary, ternary or activation-gradient result and records its operation.
    /// Output placement selects I/O or intermediate precision unless its encoding is explicit.
    /// Without a selected policy, the first operand supplies the result encoding.
    /// Broadcasting preserves the first original operand layout matching the complete output dimensions.
    /// Otherwise, inferred strides follow the first operand with the output rank.
    /// Vector lanes and their axis follow the first vectorized operand, with rank promotion applied to the axis.
    /// During preparation, scalar operands adopt the highest non-scalar rank among all inputs and the result.
    /// Activation parameters, input scaling, NaN behavior, compute precision and index axes reach lowering unchanged.
    /// A configured operation scope names the result `scope::OUT_0`.
    ///
    /// # Errors
    /// Returns an error for unknown operands, incompatible broadcasting or a mode with the wrong tensor ports.
    /// Construction errors leave the graph unchanged.
    pub fn pointwise(
        &mut self,
        operation: PointwiseOperation<TensorOutputConfig>,
    ) -> Result<TensorId> {
        operation.validate_tensor_ports()?;
        let (first, others) = operation.input_ids();
        let first = self.tensor_spec(first)?;
        let [second, third] = others.map(|id| id.map(|id| self.tensor_spec(id)).transpose());
        let others = [second?, third?];
        let mut vector_source = first.vectorization()?.map(|lanes| (first, lanes));
        let mut dimensions = Cow::Borrowed(first.shape.dimensions());
        let mut format = first;
        for tensor in others.iter().flatten() {
            let tensor = *tensor;
            let lanes = tensor.vectorization()?;
            if vector_source.is_none() {
                vector_source = lanes.map(|lanes| (tensor, lanes));
            }
            if tensor.shape.rank() > format.shape.rank() {
                format = tensor;
            }
            dimensions = Cow::Owned(infer_binary_pointwise_dimensions(
                &dimensions,
                tensor.shape.dimensions(),
            )?);
        }
        let shape = match iter::once(first)
            .chain(others.into_iter().flatten())
            .find(|tensor| tensor.shape.dimensions() == dimensions.as_ref())
        {
            Some(tensor) => tensor.shape.clone(),
            None => format
                .shape
                .with_dimensions_preserving_stride_order(dimensions.into_owned())?,
        };
        let mut output =
            operation
                .output()
                .into_spec(self.config.data_type_policy, first.data_type, shape);
        if let Some((source, (count, axis))) = vector_source {
            let axis = axis + to_i64(output.shape.rank(), "output rank")?
                - to_i64(source.shape.rank(), "source rank")?;
            output = output.with_vectorization(count, axis)?;
        }
        let output = self.inferred_tensor(output, operation.name(), "OUT_0");
        self.pointwise_into(operation.map_output(|_| output))?;
        Ok(output)
    }
}

pub fn infer_binary_pointwise_dimensions(lhs: &[i64], rhs: &[i64]) -> Result<Vec<i64>> {
    let rank = lhs.len().max(rhs.len());
    let lhs_offset = rank - lhs.len();
    let rhs_offset = rank - rhs.len();

    let mut dimensions = Vec::with_capacity(rank);
    for index in 0..rank {
        let lhs_dim = if index < lhs_offset {
            1
        } else {
            lhs[index - lhs_offset]
        };
        let rhs_dim = if index < rhs_offset {
            1
        } else {
            rhs[index - rhs_offset]
        };

        let dimension = match (lhs_dim, rhs_dim) {
            (left, right) if left == right => left,
            (1, right) => right,
            (left, 1) => left,
            _ => {
                return Err(FrontendError::PointwiseBroadcastShapeMismatch {
                    lhs: lhs.to_vec(),
                    rhs: rhs.to_vec(),
                }
                .into());
            }
        };
        dimensions.push(dimension);
    }
    Ok(dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcasts_leading_and_unit_dimensions() -> Result<()> {
        assert_eq!(
            infer_binary_pointwise_dimensions(&[4, 1, 8], &[3, 1])?,
            [4, 3, 8]
        );
        assert_eq!(infer_binary_pointwise_dimensions(&[1], &[2, 5])?, [2, 5]);
        Ok(())
    }

    #[test]
    fn test_rejects_incompatible_dimensions() {
        assert!(infer_binary_pointwise_dimensions(&[4, 3], &[4, 2]).is_err());
    }
}
