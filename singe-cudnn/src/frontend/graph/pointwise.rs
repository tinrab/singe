use std::collections::HashMap;

use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        infer::infer_binary_pointwise_output,
        operation::{Operation, PointwiseOperation},
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    tensor::{Shape, TensorId, TensorSpec},
};

impl Graph {
    /// Adds a pointwise operation to the graph.
    ///
    /// Pointwise operations apply elementwise unary, binary, or ternary modes selected by cuDNN's pointwise mode.
    pub fn pointwise(&mut self, op: PointwiseOperation) {
        self.operations.push(Operation::Pointwise(op));
    }

    /// Adds a binary pointwise operation and creates the output tensor.
    pub fn pointwise_binary_infer(
        &mut self,
        lhs: TensorId,
        rhs: TensorId,
        mode: PointwiseMode,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let lhs_tensor = self.tensor_config(lhs)?.clone();
        let rhs_tensor = self.tensor_config(rhs)?.clone();
        let output_shape = infer_binary_pointwise_output(&lhs_tensor.shape, &rhs_tensor.shape)?;
        let output_data_type = self.effective_io_data_type(lhs_tensor.data_type);
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        self.pointwise(PointwiseOperation::Binary {
            mode,
            lhs,
            rhs,
            output,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });
        Ok(output)
    }

    /// Adds a binary pointwise operation with an explicit output data type.
    pub fn pointwise_binary_infer_as(
        &mut self,
        lhs: TensorId,
        rhs: TensorId,
        mode: PointwiseMode,
        compute_type: DataType,
        output_data_type: DataType,
    ) -> Result<TensorId> {
        let lhs_tensor = self.tensor_config(lhs)?.clone();
        let rhs_tensor = self.tensor_config(rhs)?.clone();
        let output_shape = infer_binary_pointwise_output(&lhs_tensor.shape, &rhs_tensor.shape)?;
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        self.pointwise(PointwiseOperation::Binary {
            mode,
            lhs,
            rhs,
            output,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });
        Ok(output)
    }

    pub(crate) fn pointwise_identity_infer(
        &mut self,
        input: TensorId,
        output_data_type: DataType,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let input_tensor = self.tensor_config(input)?.clone();
        let output = self
            .tensor(TensorSpec::new(output_data_type, input_tensor.shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Identity,
            input,
            output,
            compute_type,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            axis: None,
        });
        Ok(output)
    }

    pub(crate) fn validate_pointwise_operations(&self) -> Result<()> {
        for operation in &self.operations {
            match operation {
                Operation::Pointwise(PointwiseOperation::Unary { input, output, .. }) => {
                    let expected = self.shape(*input)?.dimensions().to_vec();
                    self.validate_pointwise_output_shape(
                        "unary",
                        expected,
                        self.shape(*output)?.dimensions(),
                    )?;
                }
                Operation::Pointwise(PointwiseOperation::ReluForward { input, output, .. }) => {
                    let expected = self.shape(*input)?.dimensions().to_vec();
                    self.validate_pointwise_output_shape(
                        "relu forward",
                        expected,
                        self.shape(*output)?.dimensions(),
                    )?;
                }
                Operation::Pointwise(PointwiseOperation::Binary {
                    lhs, rhs, output, ..
                }) => {
                    let expected =
                        infer_binary_pointwise_output(self.shape(*lhs)?, self.shape(*rhs)?)?
                            .dimensions()
                            .to_vec();
                    self.validate_pointwise_output_shape(
                        "binary",
                        expected,
                        self.shape(*output)?.dimensions(),
                    )?;
                }
                Operation::Pointwise(PointwiseOperation::Ternary {
                    x, b, t, output, ..
                }) => {
                    let expected = infer_binary_pointwise_output(self.shape(*x)?, self.shape(*b)?)?;
                    let expected = infer_binary_pointwise_output(&expected, self.shape(*t)?)?
                        .dimensions()
                        .to_vec();
                    self.validate_pointwise_output_shape(
                        "ternary",
                        expected,
                        self.shape(*output)?.dimensions(),
                    )?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn validate_pointwise_output_shape(
        &self,
        operation: &'static str,
        expected: Vec<i64>,
        actual: &[i64],
    ) -> Result<()> {
        if expected.as_slice() != actual {
            return Err(Error::FrontendPointwiseOutputShapeMismatch {
                operation: operation.to_string(),
                expected,
                actual: actual.to_vec(),
            });
        }

        Ok(())
    }

    pub(crate) fn expand_fused_scalar_shapes_for_pointwise(&mut self) -> Result<()> {
        let mut scalar_target_ranks = HashMap::<TensorId, i64>::new();

        for operation in &self.operations {
            let (inputs, output, alpha2) = match operation {
                Operation::Pointwise(PointwiseOperation::Binary {
                    lhs,
                    rhs,
                    output,
                    alpha2,
                    ..
                }) => (
                    [Some(*lhs), Some(*rhs), Some(*output)],
                    Some(*output),
                    Some(*alpha2),
                ),
                Operation::Pointwise(PointwiseOperation::Ternary {
                    x, b, t, output, ..
                }) => ([Some(*x), Some(*b), Some(*t)], Some(*output), None),
                _ => continue,
            };

            if let Some(alpha2) = alpha2
                && alpha2 != 1.0
            {
                return Err(Error::FrontendPointwiseBinaryAlpha2Unsupported { alpha2 });
            }

            let mut target_rank = None;
            for tensor in inputs.into_iter().flatten().chain(output) {
                let tensor = self.tensor_config(tensor)?;
                if !tensor.is_by_value {
                    target_rank = Some(tensor.shape.rank());
                    break;
                }
            }

            let Some(target_rank) = target_rank else {
                continue;
            };

            for tensor_id in inputs.into_iter().flatten() {
                let tensor = self.tensor_config(tensor_id)?;
                if !tensor.is_by_value {
                    continue;
                }

                match scalar_target_ranks.insert(tensor_id, target_rank) {
                    Some(existing_rank) if existing_rank != target_rank => {
                        return Err(Error::FrontendFusedScalarRankConflict {
                            tensor_id,
                            first_rank: existing_rank,
                            second_rank: target_rank,
                        });
                    }
                    _ => {}
                }
            }
        }

        for (tensor_index, target_rank) in scalar_target_ranks {
            let tensor = self.tensor_config(tensor_index)?.clone();
            if tensor.shape.rank() == target_rank {
                continue;
            }

            let expanded_shape = Shape::contiguous(vec![1; target_rank as usize])?
                .with_strides(vec![1; target_rank as usize])?;
            self.replace_tensor(tensor_index, tensor.with_shape(expanded_shape))?;
        }

        Ok(())
    }
}
