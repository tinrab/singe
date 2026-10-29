use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        execution::reduction::{
            ReducerAccumulator, ReductionDescriptor, ReductionDescriptorConfig,
            ReductionOperation as BackendReductionOperation,
        },
        reduction::ReduceTensorOperator,
        tensor::{Shape, TensorId},
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, ValidatedGraph},
        lower::{LoweredOperation, LoweringContext, optional_tensor_at, tensor_at},
    },
};

/// Frontend reduction operation.
///
/// cuDNN reductions reduce an input tensor using a mode selected by
/// [`ReduceTensorOperator`]. The reduced dimensions are inferred from the input and
/// output tensor shapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReductionOperation {
    /// Reduces inputs using an explicit reducer subgraph and accumulator identities.
    Custom {
        /// Input identities in the enclosing graph, paired with outputs and accumulators.
        inputs: Vec<TensorId>,
        /// Output identities in the enclosing graph.
        outputs: Vec<TensorId>,
        /// Ascending, distinct zero-based axes reduced to unit extents.
        dimensions: Vec<i64>,
        /// Frontend graph combining accumulator and current-element values.
        body: Box<Graph>,
        /// Body-local accumulator/current/output identities and initial values, one per output.
        accumulators: Vec<ReducerAccumulator<TensorId>>,
    },
    /// Ordinary reduction whose input and output shapes select the reduced dimensions.
    Reduce {
        /// Optional diagnostic scope, independent of an explicit output tensor's name.
        name: Option<String>,
        /// Native reduction operator.
        op: ReduceTensorOperator,
        /// Input tensor identity.
        input: TensorId,
        /// Reduced-output tensor identity.
        output: TensorId,
        /// Inherit graph compute precision when omitted.
        compute_type: Option<DataType>,
        /// Requests deterministic native reduction selection.
        is_deterministic: bool,
        /// Optional group-offset tensor identity.
        group_offset: Option<TensorId>,
    },
}

impl ReductionOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        match self {
            Self::Reduce {
                input,
                output,
                group_offset,
                ..
            } => {
                graph.tensor_spec(*input)?;
                graph.tensor_spec(*output)?;
                if let Some(group_offset) = group_offset {
                    graph.tensor_spec(*group_offset)?;
                }
            }
            Self::Custom {
                inputs,
                outputs,
                dimensions,
                body,
                accumulators,
            } => {
                if inputs.len() != outputs.len()
                    || inputs.is_empty()
                    || outputs.len() != accumulators.len()
                    || dimensions.is_empty()
                    || dimensions.windows(2).any(|pair| pair[0] >= pair[1])
                {
                    return Err(Error::InvalidDataShape);
                }
                for ((input, output), accumulator) in inputs.iter().zip(outputs).zip(accumulators) {
                    let input = graph.tensor_spec(*input)?;
                    let output_id = *output;
                    let output = graph.tensor_spec(output_id)?;
                    let identity_type = accumulator.initial_value.data_type();
                    if output.data_type != identity_type {
                        return Err(FrontendError::TensorDataTypeMismatch {
                            tensor_id: output_id,
                            operation: "custom reduction identity",
                            expected: output.data_type,
                            actual: identity_type,
                        }
                        .into());
                    }
                    for id in [
                        accumulator.accumulator,
                        accumulator.current,
                        accumulator.output,
                    ] {
                        body.tensor_spec(id)?;
                    }
                    let mut dimensions_after_reduction = input.shape.dimensions().to_vec();
                    for &axis in dimensions {
                        let axis = usize::try_from(axis).map_err(|_| Error::InvalidDataShape)?;
                        *dimensions_after_reduction
                            .get_mut(axis)
                            .ok_or(Error::InvalidDataShape)? = 1;
                    }
                    graph.validate_tensor_dimensions(
                        output_id,
                        &dimensions_after_reduction,
                        "custom reduction output",
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn lower(
        &self,
        context: &LoweringContext<'_>,
        reducer: Option<ValidatedGraph>,
    ) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        match self {
            Self::Custom {
                inputs,
                outputs,
                dimensions,
                accumulators,
                ..
            } => {
                let validated = reducer.ok_or(FrontendError::InternalStateMissing {
                    name: "validated reducer body",
                })?;
                let prepared = Box::new(validated.prepare(context.cudnn_context())?);
                let accumulators = accumulators
                    .iter()
                    .copied()
                    .map(|accumulator| {
                        accumulator.try_map(|id| {
                            Ok(tensor_at(&prepared.lowered.backend_tensors, id)?.id())
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let inputs = inputs
                    .iter()
                    .map(|id| tensor_at(tensors, *id))
                    .collect::<Result<Vec<_>>>()?;
                let outputs = outputs
                    .iter()
                    .map(|id| tensor_at(tensors, *id))
                    .collect::<Result<Vec<_>>>()?;
                let operation = BackendReductionOperation::create_custom(
                    &inputs,
                    &outputs,
                    dimensions,
                    prepared.operation_graph(),
                    &accumulators,
                )?;
                Ok(LoweredOperation::CustomReduction {
                    operation,
                    _body: prepared,
                })
            }
            Self::Reduce {
                op,
                input,
                output,
                compute_type,
                is_deterministic,
                group_offset,
                ..
            } => {
                let descriptor = ReductionDescriptor::create(ReductionDescriptorConfig {
                    op: *op,
                    compute_type: context.compute_data_type(*compute_type),
                    is_deterministic: *is_deterministic,
                })?;
                Ok(LoweredOperation::Reduction(
                    BackendReductionOperation::create(
                        &descriptor,
                        tensor_at(tensors, *input)?,
                        tensor_at(tensors, *output)?,
                        optional_tensor_at(tensors, *group_offset)?,
                    )?,
                ))
            }
        }
    }
}

/// Reduction axes selected by this Rust frontend helper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReductionAxes {
    /// Distinct input dimension indices in any order.
    Dimensions(Vec<usize>),
    /// Reduce the last dimension.
    Last,
    /// Reduce all dimensions.
    All,
}

/// Output geometry for ordinary or grouped reduction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionOutput {
    /// Preserve input dimensions and infer NHWC strides, matching C++ when no result geometry is supplied.
    InputDimensions,
    /// Infers reduced dimensions by replacing selected axes with unit extents.
    Axes(ReductionAxes),
    /// Uses the supplied output dimensions and strides without inferring geometry.
    Shape(Shape),
}

/// Attributes for building a frontend reduction operation.
///
/// This maps to the C++ frontend `Reduction_attributes` mode and compute data
/// type setters, with Rust-side axis and determinism options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReductionConfig {
    #[serde(default)]
    name: Option<String>,
    op: ReduceTensorOperator,
    compute_type: Option<DataType>,
    output: ReductionOutput,
    is_deterministic: bool,
    group_offset: Option<TensorId>,
}

impl ReductionConfig {
    /// Uses the reduction mode and inherits graph compute precision, defaulting to `F32`.
    #[must_use]
    pub fn new(op: ReduceTensorOperator) -> Self {
        Self {
            name: None,
            op,
            compute_type: None,
            output: ReductionOutput::InputDimensions,
            is_deterministic: false,
            group_offset: None,
        }
    }

    /// Names inferred results without replacing explicit output tensor names.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty result-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Selects reduction axes, or restores input dimensions and inferred NHWC strides with `None`.
    /// Replaces any previously selected axes or explicit output shape.
    #[must_use]
    pub fn with_axes(mut self, axes: impl Into<Option<ReductionAxes>>) -> Self {
        self.output = axes
            .into()
            .map_or(ReductionOutput::InputDimensions, ReductionOutput::Axes);
        self
    }

    /// Sets compute precision, or inherits graph policy when the value is `None`.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Enables or disables deterministic reduction selection without rebuilding the configuration.
    #[must_use]
    pub fn with_deterministic(mut self, is_deterministic: bool) -> Self {
        self.is_deterministic = is_deterministic;
        self
    }

    /// Returns the native reduction operator.
    #[must_use]
    pub fn op(&self) -> ReduceTensorOperator {
        self.op
    }

    /// Returns the compute-precision override, or `None` to inherit graph policy.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }

    /// Borrows the selected output-geometry policy.
    #[must_use]
    pub fn output(&self) -> &ReductionOutput {
        &self.output
    }

    /// Infers result dimensions without calculating destination strides or creating descriptors.
    /// Omitted geometry preserves input dimensions; explicit shapes supply their own dimensions.
    /// Ordinary axis selection is independent of axis order and preserves the recorded request.
    ///
    /// # Errors
    /// Returns an error for empty, repeated or out-of-range explicit axes.
    pub fn infer_output_dimensions(&self, input: &Shape) -> Result<Vec<i64>> {
        let axes = match &self.output {
            ReductionOutput::InputDimensions => return Ok(input.dimensions().to_vec()),
            ReductionOutput::Shape(shape) => return Ok(shape.dimensions().to_vec()),
            ReductionOutput::Axes(axes) => axes,
        };
        let mut dimensions = input.dimensions().to_vec();
        match axes {
            ReductionAxes::Dimensions(axes) => {
                let mut axes = axes.clone();
                axes.sort_unstable();
                if axes.is_empty() || axes.windows(2).any(|pair| pair[0] == pair[1]) {
                    return Err(Error::InvalidDataShape);
                }
                for axis in axes {
                    *dimensions.get_mut(axis).ok_or(Error::InvalidDataShape)? = 1;
                }
            }
            ReductionAxes::Last => dimensions[input.rank() - 1] = 1,
            ReductionAxes::All => dimensions.fill(1),
        }
        Ok(dimensions)
    }

    /// Infers NHWC result strides, matching C++, or preserves the selected explicit shape.
    /// Does not query a graph, device or native descriptor, and leaves encoding selection independent.
    ///
    /// # Errors
    /// Returns an error for invalid explicit axes or an unrepresentable inferred stride.
    pub fn infer_output_shape(&self, input: &Shape) -> Result<Shape> {
        match &self.output {
            ReductionOutput::Shape(shape) => Ok(shape.clone()),
            _ => Shape::channels_last(self.infer_output_dimensions(input)?),
        }
    }

    /// Selects explicit inferred-result geometry, or restores default geometry with `None`.
    /// Clearing the shape also clears any previously selected reduction axes.
    #[must_use]
    pub fn with_output_shape(mut self, shape: impl Into<Option<Shape>>) -> Self {
        self.output = shape
            .into()
            .map_or(ReductionOutput::InputDimensions, ReductionOutput::Shape);
        self
    }

    /// Sets or clears the group-offset operand while retaining the other reduction attributes.
    #[must_use]
    pub fn with_group_offset(mut self, group_offset: impl Into<Option<TensorId>>) -> Self {
        self.group_offset = group_offset.into();
        self
    }

    /// Returns the optional group-offset tensor identity.
    #[must_use]
    pub fn group_offset(&self) -> Option<TensorId> {
        self.group_offset
    }

    /// Returns whether deterministic reduction selection is requested.
    #[must_use]
    pub fn is_deterministic(&self) -> bool {
        self.is_deterministic
    }
}
