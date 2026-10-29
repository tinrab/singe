use std::f64::consts::LOG2_E;

use crate::{
    backend::{
        execution::attention::SoftmaxStatsBase,
        pointwise::PointwiseMode,
        reduction::ReduceTensorOperator,
        tensor::{TensorId, TensorSpec},
    },
    error::Result,
    frontend::{
        graph::Graph,
        operation::{PointwiseOperation, ReductionConfig, SoftmaxConfig},
    },
};

/// Complete tensor set after the public constructor resolves private intermediates.
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxTensors {
    pub(crate) input: TensorId,
    pub(crate) output: TensorId,
    pub(crate) stats: Option<TensorId>,
    pub(crate) max: TensorId,
    pub(crate) sum_exp: TensorId,
}

impl Graph {
    #[expect(
        clippy::too_many_lines,
        reason = "the normalization graph keeps its coupled shape and execution contracts together"
    )]
    pub(crate) fn expand_softmax(
        &mut self,
        tensors: SoftmaxTensors,
        config: &SoftmaxConfig,
    ) -> Result<()> {
        let input_shape = self.shape(tensors.input)?.clone();
        let reduction_shape = self.shape(tensors.max)?.clone();
        let compute_type = self.resolve_compute_data_type(config.compute_type());
        let intermediate_type = self.effective_intermediate_data_type(compute_type);
        let shifted = self.inferred_tensor(
            TensorSpec::new(intermediate_type, input_shape.clone()).virtual_tensor(),
            config.name(),
            "shifted",
        );
        let exp = self.inferred_tensor(
            TensorSpec::new(intermediate_type, input_shape).virtual_tensor(),
            config.name(),
            "exponentials",
        );

        let score_max = if config.sink().is_some() {
            self.inferred_tensor(
                TensorSpec::new(intermediate_type, reduction_shape.clone()).virtual_tensor(),
                config.name(),
                "score_max",
            )
        } else {
            tensors.max
        };
        self.reduction_into(
            tensors.input,
            score_max,
            &ReductionConfig::new(ReduceTensorOperator::Max)
                .with_compute_type(config.compute_type()),
        )?;
        if let Some(sink) = config.sink() {
            self.pointwise_into(PointwiseOperation::Binary {
                name: None,
                mode: PointwiseMode::Max,
                lhs: score_max,
                rhs: sink,
                output: tensors.max,
                compute_type: config.compute_type(),
                nan_propagation: config.nan_propagation(),
                alpha1: 1.0,
                alpha2: 1.0,
            })?;
        }
        self.pointwise_into(PointwiseOperation::Binary {
            name: None,
            mode: PointwiseMode::Sub,
            lhs: tensors.input,
            rhs: tensors.max,
            output: shifted,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            alpha2: 1.0,
        })?;
        self.pointwise_into(PointwiseOperation::Unary {
            name: None,
            mode: PointwiseMode::Exp,
            input: shifted,
            output: exp,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            axis: None,
        })?;
        let score_sum = if config.sink().is_some() {
            self.inferred_tensor(
                TensorSpec::new(intermediate_type, reduction_shape.clone()).virtual_tensor(),
                config.name(),
                "score_sum",
            )
        } else {
            tensors.sum_exp
        };
        self.reduction_into(
            exp,
            score_sum,
            &ReductionConfig::new(ReduceTensorOperator::Add)
                .with_compute_type(config.compute_type()),
        )?;
        if let Some(sink) = config.sink() {
            let shifted_sink = self.inferred_tensor(
                TensorSpec::new(intermediate_type, reduction_shape.clone()).virtual_tensor(),
                config.name(),
                "shifted_sink",
            );
            self.pointwise_into(PointwiseOperation::Binary {
                name: None,
                mode: PointwiseMode::Sub,
                lhs: sink,
                rhs: tensors.max,
                output: shifted_sink,
                compute_type: config.compute_type(),
                nan_propagation: config.nan_propagation(),
                alpha1: 1.0,
                alpha2: 1.0,
            })?;
            let exp_sink = self.inferred_tensor(
                TensorSpec::new(intermediate_type, reduction_shape.clone()).virtual_tensor(),
                config.name(),
                "sink_exponentials",
            );
            self.pointwise_into(PointwiseOperation::Unary {
                name: None,
                mode: PointwiseMode::Exp,
                input: shifted_sink,
                output: exp_sink,
                compute_type: config.compute_type(),
                nan_propagation: config.nan_propagation(),
                alpha1: 1.0,
                axis: None,
            })?;
            self.pointwise_into(PointwiseOperation::Binary {
                name: None,
                mode: PointwiseMode::Add,
                lhs: score_sum,
                rhs: exp_sink,
                output: tensors.sum_exp,
                compute_type: config.compute_type(),
                nan_propagation: config.nan_propagation(),
                alpha1: 1.0,
                alpha2: 1.0,
            })?;
        }
        if let Some(stats) = tensors.stats {
            let log_sum = self.inferred_tensor(
                TensorSpec::new(intermediate_type, reduction_shape).virtual_tensor(),
                config.name(),
                "log_sum_exp",
            );
            self.pointwise_into(PointwiseOperation::Unary {
                name: None,
                mode: PointwiseMode::Log,
                input: tensors.sum_exp,
                output: log_sum,
                compute_type: config.compute_type(),
                nan_propagation: config.nan_propagation(),
                alpha1: 1.0,
                axis: None,
            })?;
            let stats_scale = match config.stats_base() {
                SoftmaxStatsBase::NaturalLog => 1.0,
                SoftmaxStatsBase::Log2 => LOG2_E,
            };
            self.pointwise_into(PointwiseOperation::Binary {
                name: None,
                mode: PointwiseMode::Add,
                lhs: tensors.max,
                rhs: log_sum,
                output: stats,
                compute_type: config.compute_type(),
                nan_propagation: config.nan_propagation(),
                alpha1: stats_scale,
                alpha2: stats_scale,
            })?;
        }
        self.pointwise_into(PointwiseOperation::Binary {
            name: None,
            mode: PointwiseMode::Div,
            lhs: exp,
            rhs: tensors.sum_exp,
            output: tensors.output,
            compute_type: config.compute_type(),
            nan_propagation: config.nan_propagation(),
            alpha1: 1.0,
            alpha2: 1.0,
        })?;

        Ok(())
    }
}
