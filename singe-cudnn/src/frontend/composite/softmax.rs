use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        infer::infer_softmax_shapes,
        operation::{PointwiseOperation, ReductionOperation, SoftmaxConfig},
        support::{SOFTMAX_OUTPUT_DATA_TYPE, SOFTMAX_RANK, SOFTMAX_REDUCTION_TENSORS},
    },
    pointwise::PointwiseMode,
    reduction::ReduceTensorOperator,
    tensor::{TensorId, TensorSpec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxCompositeOutputs {
    pub stats: TensorId,
    pub max: TensorId,
    pub sum: TensorId,
    pub output: TensorId,
}

impl SoftmaxCompositeOutputs {
    pub fn new(stats: TensorId, max: TensorId, sum: TensorId, output: TensorId) -> Self {
        Self {
            stats,
            max,
            sum,
            output,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxCompositeTensors {
    pub input: TensorId,
    pub stats: TensorId,
    pub max: TensorId,
    pub sum: TensorId,
    pub output: TensorId,
}

impl SoftmaxCompositeTensors {
    pub fn new(
        input: TensorId,
        stats: TensorId,
        max: TensorId,
        sum: TensorId,
        output: TensorId,
    ) -> Self {
        Self {
            input,
            stats,
            max,
            sum,
            output,
        }
    }
}

impl Graph {
    /// Adds a composite softmax subgraph with explicit intermediate outputs.
    ///
    /// This expands softmax into pointwise and reduction operations, matching
    /// the BMM-softmax-BMM decomposition described for cuDNN frontend attention
    /// when a direct SDPA operation is not used.
    pub fn softmax_composite(
        &mut self,
        tensors: SoftmaxCompositeTensors,
        config: SoftmaxConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(tensors.input)?.clone();
        let max_tensor = self.validate_softmax_composite_tensors(
            tensors.input,
            tensors.stats,
            tensors.max,
            tensors.sum,
            tensors.output,
            config.compute_type(),
        )?;

        let shifted = self.tensor(
            TensorSpec::new(input_tensor.data_type, input_tensor.shape.clone()).virtual_tensor(),
        );
        let exp = self.tensor(
            TensorSpec::new(input_tensor.data_type, input_tensor.shape.clone()).virtual_tensor(),
        );

        self.reduction(ReductionOperation::Reduce {
            op: ReduceTensorOperator::Max,
            input: tensors.input,
            output: tensors.max,
            compute_type: config.compute_type(),
            is_deterministic: false,
        });
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Sub,
            lhs: tensors.input,
            rhs: tensors.max,
            output: shifted,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            alpha2: 1.0,
        });
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Exp,
            input: shifted,
            output: exp,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            axis: None,
        });
        self.reduction(ReductionOperation::Reduce {
            op: ReduceTensorOperator::Add,
            input: exp,
            output: tensors.sum,
            compute_type: config.compute_type(),
            is_deterministic: false,
        });
        let log_sum = self.tensor(
            TensorSpec::new(max_tensor.data_type, max_tensor.shape.clone()).virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Log,
            input: tensors.sum,
            output: log_sum,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            axis: None,
        });
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Add,
            lhs: tensors.max,
            rhs: log_sum,
            output: tensors.stats,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            alpha2: 1.0,
        });
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Div,
            lhs: exp,
            rhs: tensors.sum,
            output: tensors.output,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            alpha2: 1.0,
        });

        Ok(())
    }

    fn validate_softmax_composite_tensors(
        &self,
        input: TensorId,
        stats: TensorId,
        max: TensorId,
        sum: TensorId,
        output: TensorId,
        compute_type: DataType,
    ) -> Result<TensorSpec> {
        let input_tensor = self.tensor_config(input)?;
        let stats_tensor = self.tensor_config(stats)?;
        let max_tensor = self.tensor_config(max)?.clone();
        let sum_tensor = self.tensor_config(sum)?;
        let output_tensor = self.tensor_config(output)?;

        if input_tensor.data_type != output_tensor.data_type {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: output,
                operation: SOFTMAX_OUTPUT_DATA_TYPE.into(),
                expected: input_tensor.data_type,
                actual: output_tensor.data_type,
            });
        }
        for (tensor_id, tensor) in [(stats, stats_tensor), (max, &max_tensor), (sum, sum_tensor)] {
            if tensor.data_type != compute_type {
                return Err(Error::FrontendTensorDataTypeMismatch {
                    tensor_id,
                    operation: SOFTMAX_REDUCTION_TENSORS.into(),
                    expected: compute_type,
                    actual: tensor.data_type,
                });
            }
        }
        for (tensor_id, tensor) in [(stats, stats_tensor), (sum, sum_tensor)] {
            if tensor.shape.dimensions() != max_tensor.shape.dimensions() {
                return Err(Error::FrontendTensorDimensionsMismatch {
                    tensor_id,
                    operation: SOFTMAX_REDUCTION_TENSORS.into(),
                    expected: max_tensor.shape.dimensions().to_vec(),
                    actual: tensor.shape.dimensions().to_vec(),
                });
            }
        }
        for (tensor_id, tensor) in [(max, &max_tensor), (sum, sum_tensor)] {
            if input_tensor.shape.dimensions().len() != tensor.shape.dimensions().len() {
                return Err(Error::FrontendTensorRankMismatch {
                    tensor_id,
                    operation: SOFTMAX_RANK.into(),
                    expected: input_tensor.shape.dimensions().len().to_string(),
                    actual: tensor.shape.dimensions().len(),
                });
            }
        }

        Ok(max_tensor)
    }

    /// Adds a composite softmax subgraph and creates stats/intermediate outputs.
    pub fn softmax_composite_infer(
        &mut self,
        input: TensorId,
        config: SoftmaxConfig,
    ) -> Result<SoftmaxCompositeOutputs> {
        let input_tensor = self.tensor_config(input)?.clone();
        let (max_shape, sum_shape) = infer_softmax_shapes(&input_tensor.shape)?;

        let max = self.tensor(TensorSpec::new(config.compute_type(), max_shape));
        let stats_shape = sum_shape.clone();
        let stats = self.tensor(TensorSpec::new(config.compute_type(), stats_shape));
        let sum = self.tensor(TensorSpec::new(config.compute_type(), sum_shape));
        let output = self.tensor(TensorSpec::new(
            input_tensor.data_type,
            input_tensor.shape.clone(),
        ));

        self.softmax_composite(
            SoftmaxCompositeTensors::new(input, stats, max, sum, output),
            config,
        )?;

        Ok(SoftmaxCompositeOutputs::new(stats, max, sum, output))
    }
}
