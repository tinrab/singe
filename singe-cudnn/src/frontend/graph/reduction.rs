use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        infer::infer_reduction_output,
        operation::{
            Operation, PointwiseOperation, ReductionAxes, ReductionConfig, ReductionOperation,
            SoftmaxOperationConfig,
        },
        support,
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    reduction::ReduceTensorOperator,
    tensor::{Shape, TensorId, TensorSpec},
    version,
};

impl Graph {
    /// Adds a softmax operation with optional reduction side outputs.
    ///
    /// cuDNN frontend models fused flash attention as BMM-softmax-BMM with
    /// optional statistics and masking features. This helper records a standalone
    /// frontend softmax node for the graph.
    pub fn softmax(
        &mut self,
        x: TensorId,
        y: TensorId,
        config: SoftmaxOperationConfig,
    ) -> Result<()> {
        let x_tensor = self.tensor_config(x)?.clone();
        let y_tensor = self.tensor_config(y)?.clone();
        if x_tensor.data_type != y_tensor.data_type {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: y,
                operation: "softmax output".into(),
                expected: x_tensor.data_type,
                actual: y_tensor.data_type,
            });
        }
        if x_tensor.shape.dimensions() != y_tensor.shape.dimensions() {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: y,
                operation: "softmax output".into(),
                expected: x_tensor.shape.dimensions().to_vec(),
                actual: y_tensor.shape.dimensions().to_vec(),
            });
        }

        let reduction_shape = infer_reduction_output(&x_tensor.shape, ReductionAxes::Last)?;
        let reduction_data_type = self.effective_compute_data_type(DataType::F32);
        if let Some(stats) = config.stats() {
            self.validate_softmax_reduction_tensor(
                stats,
                &reduction_shape,
                reduction_data_type,
                "softmax stats",
            )?;
        }
        if let Some(max) = config.max() {
            self.validate_softmax_reduction_tensor(
                max,
                &reduction_shape,
                reduction_data_type,
                "softmax max",
            )?;
        }
        if let Some(sum_exp) = config.sum_exp() {
            self.validate_softmax_reduction_tensor(
                sum_exp,
                &reduction_shape,
                reduction_data_type,
                "softmax sum exp",
            )?;
        }
        if let Some(sink) = config.sink() {
            self.validate_softmax_sink_tensor(sink, x, &x_tensor.shape, reduction_data_type)?;
        }

        self.operations.push(Operation::Softmax {
            x,
            y,
            stats: config.stats(),
            max: config.max(),
            sum_exp: config.sum_exp(),
            sink: config.sink(),
        });
        Ok(())
    }

    pub(crate) fn absolute_max_infer(
        &mut self,
        input: TensorId,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let abs = self.tensor(
            TensorSpec::new(compute_type, self.tensor_config(input)?.shape.clone())
                .virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Abs,
            input,
            output: abs,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: None,
        });
        self.reduction_infer(
            abs,
            ReductionConfig::new(ReduceTensorOperator::AbsoluteMax, compute_type)
                .with_axes(ReductionAxes::All),
        )
    }

    /// Adds a reduction operation to the graph.
    ///
    /// Reduction modes are controlled by [`ReduceTensorOperator`], with reduced
    /// dimensions inferred from the input and output tensor shapes.
    pub fn reduction(&mut self, op: ReductionOperation) {
        self.operations.push(Operation::Reduction(op));
    }

    /// Adds a reduction operation and creates the output tensor.
    pub fn reduction_infer(
        &mut self,
        input: TensorId,
        config: ReductionConfig,
    ) -> Result<TensorId> {
        self.validate_reduction_support_surface_for_version(version()?.raw(), config)?;
        let input_tensor = self.tensor_config(input)?.clone();
        let output_shape = infer_reduction_output(&input_tensor.shape, config.axes())?;
        let output_shape = Self::default_nhwc_shape(output_shape.dimensions().to_vec())?;
        let output_data_type = self.effective_io_data_type(input_tensor.data_type);
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        self.reduction(ReductionOperation::Reduce {
            op: config.op(),
            input,
            output,
            compute_type: config.compute_type(),
            is_deterministic: config.is_deterministic(),
        });
        Ok(output)
    }

    pub(crate) fn expand_softmax_for_legacy_runtime(
        &mut self,
        x: TensorId,
        y: TensorId,
        stats: Option<TensorId>,
        max: Option<TensorId>,
        sum_exp: Option<TensorId>,
        sink: Option<TensorId>,
    ) -> Result<()> {
        let has_legacy_supported_outputs = matches!(
            (stats, max, sum_exp),
            (None, None, None) | (Some(_), None, None) | (None, Some(_), Some(_))
        );
        if !has_legacy_supported_outputs {
            return Err(Error::DescriptorMismatch {
                name: "softmax legacy outputs".into(),
            });
        }

        let x_tensor = self.tensor_config(x)?.clone();
        let reduction_shape = infer_reduction_output(&x_tensor.shape, ReductionAxes::Last)?;
        let compute_type = self.effective_compute_data_type(x_tensor.data_type);

        let max_output = match max {
            Some(max) => max,
            None => self.tensor(
                TensorSpec::new(x_tensor.data_type, reduction_shape.clone()).virtual_tensor(),
            ),
        };
        let sum_output = match sum_exp {
            Some(sum_exp) => sum_exp,
            None => {
                self.tensor(TensorSpec::new(x_tensor.data_type, reduction_shape).virtual_tensor())
            }
        };

        if let Some(sink) = sink {
            let reduced_max = self.tensor(
                TensorSpec::new(
                    x_tensor.data_type,
                    self.tensor_config(max_output)?.shape.clone(),
                )
                .virtual_tensor(),
            );
            self.reduction(ReductionOperation::Reduce {
                op: ReduceTensorOperator::Max,
                input: x,
                output: reduced_max,
                compute_type,
                is_deterministic: false,
            });
            self.pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Max,
                lhs: reduced_max,
                rhs: sink,
                output: max_output,
                compute_type,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
        } else {
            self.reduction(ReductionOperation::Reduce {
                op: ReduceTensorOperator::Max,
                input: x,
                output: max_output,
                compute_type,
                is_deterministic: false,
            });
        }

        let shifted = self
            .tensor(TensorSpec::new(x_tensor.data_type, x_tensor.shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Sub,
            lhs: x,
            rhs: max_output,
            output: shifted,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });
        let exp = self
            .tensor(TensorSpec::new(x_tensor.data_type, x_tensor.shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Exp,
            input: shifted,
            output: exp,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: None,
        });

        if let Some(sink) = sink {
            let sub_sink = self.tensor(
                TensorSpec::new(
                    x_tensor.data_type,
                    self.tensor_config(max_output)?.shape.clone(),
                )
                .virtual_tensor(),
            );
            self.pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Sub,
                lhs: sink,
                rhs: max_output,
                output: sub_sink,
                compute_type,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
            let exp_sink = self.tensor(
                TensorSpec::new(
                    x_tensor.data_type,
                    self.tensor_config(max_output)?.shape.clone(),
                )
                .virtual_tensor(),
            );
            self.pointwise(PointwiseOperation::Unary {
                mode: PointwiseMode::Exp,
                input: sub_sink,
                output: exp_sink,
                compute_type,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                axis: None,
            });
            let reduced_sum = self.tensor(
                TensorSpec::new(
                    x_tensor.data_type,
                    self.tensor_config(sum_output)?.shape.clone(),
                )
                .virtual_tensor(),
            );
            self.reduction(ReductionOperation::Reduce {
                op: ReduceTensorOperator::Add,
                input: exp,
                output: reduced_sum,
                compute_type,
                is_deterministic: false,
            });
            self.pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Add,
                lhs: reduced_sum,
                rhs: exp_sink,
                output: sum_output,
                compute_type,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
        } else {
            self.reduction(ReductionOperation::Reduce {
                op: ReduceTensorOperator::Add,
                input: exp,
                output: sum_output,
                compute_type,
                is_deterministic: false,
            });
        }

        if let Some(stats) = stats {
            let log_sum = self.tensor(
                TensorSpec::new(
                    x_tensor.data_type,
                    self.tensor_config(sum_output)?.shape.clone(),
                )
                .virtual_tensor(),
            );
            self.pointwise(PointwiseOperation::Unary {
                mode: PointwiseMode::Log,
                input: sum_output,
                output: log_sum,
                compute_type,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                axis: None,
            });
            self.pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Add,
                lhs: max_output,
                rhs: log_sum,
                output: stats,
                compute_type,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
        }

        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::Div,
            lhs: exp,
            rhs: sum_output,
            output: y,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });
        Ok(())
    }

    fn validate_softmax_reduction_tensor(
        &self,
        tensor_ref: TensorId,
        expected_shape: &Shape,
        expected_data_type: DataType,
        name: &str,
    ) -> Result<()> {
        let tensor = self.tensor_config(tensor_ref)?.clone();
        if tensor.data_type != expected_data_type {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: tensor_ref,
                operation: name.into(),
                expected: expected_data_type,
                actual: tensor.data_type,
            });
        }
        if tensor.shape.dimensions() != expected_shape.dimensions() {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: tensor_ref,
                operation: name.into(),
                expected: expected_shape.dimensions().to_vec(),
                actual: tensor.shape.dimensions().to_vec(),
            });
        }
        Ok(())
    }

    fn validate_softmax_sink_tensor(
        &self,
        tensor_ref: TensorId,
        input: TensorId,
        input_shape: &Shape,
        expected_data_type: DataType,
    ) -> Result<()> {
        let tensor = self.tensor_config(tensor_ref)?.clone();
        let input_dims = input_shape.dimensions();
        self.validate_tensor_rank(input, 4, "softmax sink input")?;
        let expected_shape = Shape::contiguous([1, input_dims[1], 1, 1])?;
        if tensor.data_type != expected_data_type {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: tensor_ref,
                operation: "softmax sink".into(),
                expected: expected_data_type,
                actual: tensor.data_type,
            });
        }
        if tensor.shape.dimensions() != expected_shape.dimensions() {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: tensor_ref,
                operation: "softmax sink".into(),
                expected: expected_shape.dimensions().to_vec(),
                actual: tensor.shape.dimensions().to_vec(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_reduction_support_surface_for_version(
        &self,
        cudnn_version: u64,
        config: ReductionConfig,
    ) -> Result<()> {
        support::require_reduction_support(cudnn_version, config)
    }
}
