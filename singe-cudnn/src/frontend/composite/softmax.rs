use crate::{
    error::{Error, Result},
    frontend::{
        graph::Graph,
        infer::infer_softmax_shapes,
        operation::{PointwiseOperation, ReductionOperation, SoftmaxConfig},
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

impl Graph {
    /// Adds a composite softmax subgraph with explicit intermediate outputs.
    ///
    /// This expands softmax into pointwise and reduction operations, matching
    /// the BMM-softmax-BMM decomposition described for cuDNN frontend attention
    /// when a direct SDPA operation is not used.
    pub fn softmax_composite(
        &mut self,
        input: TensorId,
        max: TensorId,
        sum: TensorId,
        output: TensorId,
        config: SoftmaxConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        let max_tensor = self.tensor_config(max)?.clone();
        let sum_tensor = self.tensor_config(sum)?.clone();
        let output_tensor = self.tensor_config(output)?.clone();

        if input_tensor.data_type != output_tensor.data_type {
            return Err(Error::DescriptorMismatch {
                name: "softmax output data type".into(),
            });
        }
        if max_tensor.data_type != config.compute_type()
            || sum_tensor.data_type != config.compute_type()
        {
            return Err(Error::DescriptorMismatch {
                name: "softmax reduction tensors".into(),
            });
        }
        if max_tensor.shape.dimensions() != sum_tensor.shape.dimensions() {
            return Err(Error::DescriptorMismatch {
                name: "softmax reduction tensors".into(),
            });
        }
        if input_tensor.shape.dimensions().len() != max_tensor.shape.dimensions().len()
            || input_tensor.shape.dimensions().len() != sum_tensor.shape.dimensions().len()
        {
            return Err(Error::DescriptorMismatch {
                name: "softmax rank".into(),
            });
        }

        let shifted = self.tensor(
            TensorSpec::new(input_tensor.data_type, input_tensor.shape.clone()).virtual_tensor(),
        );
        let exp = self.tensor(
            TensorSpec::new(input_tensor.data_type, input_tensor.shape.clone()).virtual_tensor(),
        );

        self.reduction(ReductionOperation::Reduce {
            op: ReduceTensorOperator::Max,
            input,
            output: max,
            compute_type: config.compute_type(),
            is_deterministic: false,
        });
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Sub,
            lhs: input,
            rhs: max,
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
            output: sum,
            compute_type: config.compute_type(),
            is_deterministic: false,
        });
        let log_sum = self.tensor(
            TensorSpec::new(max_tensor.data_type, max_tensor.shape.clone()).virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Log,
            input: sum,
            output: log_sum,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            axis: None,
        });
        let stats = self.tensor(
            TensorSpec::new(max_tensor.data_type, max_tensor.shape.clone()).virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Add,
            lhs: max,
            rhs: log_sum,
            output: stats,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            alpha2: 1.0,
        });
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Div,
            lhs: exp,
            rhs: sum,
            output,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            alpha2: 1.0,
        });

        Ok(())
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
        let stats = self.tensor(TensorSpec::new(config.compute_type(), sum_shape.clone()));
        let sum = self.tensor(TensorSpec::new(config.compute_type(), sum_shape));
        let output = self.tensor(TensorSpec::new(
            input_tensor.data_type,
            input_tensor.shape.clone(),
        ));

        self.softmax_composite(input, max, sum, output, config)?;

        let log_sum = self.tensor(
            TensorSpec::new(
                config.compute_type(),
                self.tensor_config(sum)?.shape.clone(),
            )
            .virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Log,
            input: sum,
            output: log_sum,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            axis: None,
        });
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Add,
            lhs: max,
            rhs: log_sum,
            output: stats,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            alpha2: 1.0,
        });

        Ok(SoftmaxCompositeOutputs::new(stats, max, sum, output))
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use crate::{
        Status, data_type::DataType, frontend::operation::HeuristicMode, tensor::Shape,
        testing::setup_context,
    };

    use super::*;

    fn is_expected_compile_status(error: &Error) -> bool {
        match error {
            Error::NoAvailableEngines => true,
            Error::Cudnn { code, .. } => {
                *code == Status::NotSupported
                    || *code == Status::NotSupportedRuntimePrerequisiteMissing
                    || *code == Status::InternalErrorUnexpectedValue
            }
            _ => false,
        }
    }

    #[test]
    fn test_softmax_composite_validates_reduction_rank() -> Result<()> {
        let mut graph = Graph::new();
        let input =
            graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 4])?).with_id(1));
        let max = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?).with_id(2));
        let sum = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2])?).with_id(3));
        let output =
            graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 4])?).with_id(4));

        let err = graph
            .softmax_composite(input, max, sum, output, SoftmaxConfig::new(DataType::F32))
            .unwrap_err();
        assert!(matches!(
            err,
            Error::DescriptorMismatch { name } if name == "softmax rank"
        ));

        Ok(())
    }

    #[test]
    fn test_softmax_composite_accepts_noncontiguous_reduction_tensor_strides() -> Result<()> {
        let mut graph = Graph::new();
        let input = graph.tensor(TensorSpec::new(
            DataType::F32,
            Shape::contiguous([2, 3, 4])?,
        ));
        let max = graph.tensor(TensorSpec::new(
            DataType::F32,
            Shape::contiguous([2, 3, 1])?.with_strides([12, 4, 1])?,
        ));
        let sum = graph.tensor(TensorSpec::new(
            DataType::F32,
            Shape::contiguous([2, 3, 1])?.with_strides([24, 8, 1])?,
        ));
        let output = graph.tensor(TensorSpec::new(
            DataType::F32,
            Shape::contiguous([2, 3, 4])?,
        ));

        graph.softmax_composite(input, max, sum, output, SoftmaxConfig::new(DataType::F32))?;

        Ok(())
    }

    #[test]
    fn test_softmax_infer_creates_expected_shapes() -> Result<()> {
        let mut graph = Graph::new();
        let input =
            graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([2, 3, 4])?).with_id(1));

        let outputs = graph.softmax_composite_infer(input, SoftmaxConfig::new(DataType::F32))?;

        assert_eq!(graph.tensor_config(outputs.stats)?.data_type, DataType::F32);
        assert_eq!(graph.tensor_config(outputs.max)?.data_type, DataType::F32);
        assert_eq!(graph.tensor_config(outputs.sum)?.data_type, DataType::F32);
        assert_eq!(graph.shape(outputs.max)?.dimensions(), &[2, 3, 1]);
        assert_eq!(graph.shape(outputs.sum)?.dimensions(), &[2, 3, 1]);
        assert_eq!(graph.shape(outputs.output)?.dimensions(), &[2, 3, 4]);

        Ok(())
    }

    #[test]
    fn test_softmax_infer_compiles_when_plan_is_available() -> Result<()> {
        let context = setup_context()?;

        let mut graph = Graph::new();
        let input =
            graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3, 4])?).with_id(11));
        let outputs = graph.softmax_composite_infer(input, SoftmaxConfig::new(DataType::F32))?;

        match graph.compile(&context, &[HeuristicMode::Instant, HeuristicMode::Fallback]) {
            Ok(compiled) => {
                // TODO: is it somehow possible that TensorId are created with <0? If so, make that part of the type system.
                assert!(outputs.output.as_i64() > 0);
                let _ = compiled.workspace_size()?;
            }
            Err(error) if is_expected_compile_status(&error) => {}
            Err(error) => return Err(error),
        }

        Ok(())
    }
}
