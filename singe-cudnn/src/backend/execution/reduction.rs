use std::result::Result as StdResult;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        execution::OperationGraph,
        reduction::ReduceTensorOperator,
        tensor::{BackendTensorUid, Tensor},
    },
    data_type::DataType,
    error::{Error, Result},
    scalar::{ScalarStorage, ScalarValue},
};

/// Resolved native reduction attributes, independent of input and result encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReductionDescriptorConfig {
    /// Native reduction operator.
    pub op: ReduceTensorOperator,
    /// Native compute precision, independent of input and output storage encodings.
    pub compute_type: DataType,
    /// Requests deterministic native reduction selection when true.
    pub is_deterministic: bool,
}

impl ReductionDescriptorConfig {
    /// Selects explicit native compute precision and defaults to ordinary reduction selection.
    #[must_use]
    pub fn new(op: ReduceTensorOperator, compute_type: DataType) -> Self {
        Self {
            op,
            compute_type,
            is_deterministic: false,
        }
    }

    /// Sets the native determinism request while retaining operator and precision.
    #[must_use]
    pub fn with_deterministic(mut self, is_deterministic: bool) -> Self {
        self.is_deterministic = is_deterministic;
        self
    }
}

/// Finalized native reduction descriptor with retained configuration.
#[derive(Debug)]
pub struct ReductionDescriptor {
    descriptor: BackendDescriptor,
    config: ReductionDescriptorConfig,
}

impl ReductionDescriptor {
    /// Creates a reduction descriptor with explicit mode, compute precision and determinism.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects descriptor creation, attributes or finalization.
    pub fn create(config: ReductionDescriptorConfig) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Reduction)?;
        descriptor.set_attribute_enum(BackendAttributeName::ReductionOperator, config.op)?;
        descriptor
            .set_attribute_enum(BackendAttributeName::ReductionCompType, config.compute_type)?;
        descriptor.set_attribute_bool(
            BackendAttributeName::ReductionIsDeterministic,
            config.is_deterministic,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor, config })
    }

    /// Retained attributes without native queries or allocation.
    #[must_use]
    pub fn config(&self) -> ReductionDescriptorConfig {
        self.config
    }

    /// Borrows the finalized native reduction descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

/// Native reduction operands and requested settings retained after finalization.
/// Custom reductions retain parent operands and the accumulator/current/output identities in their body.
#[derive(Debug, Clone)]
pub enum ReductionOperationMetadata {
    /// Ordinary native reduction with optional group offsets.
    Standard {
        /// Retained reduction operator, math precision, and determinism request.
        descriptor: ReductionDescriptorConfig,
        /// Native input tensor identity.
        input: BackendTensorUid,
        /// Native reduced-output tensor identity.
        output: BackendTensorUid,
        /// Optional native group-offset tensor identity.
        group_offset: Option<BackendTensorUid>,
    },
    /// User-defined reduction retaining parent operands, dimensions, and body accumulator mappings.
    Custom {
        /// Native identities of the enclosing reduction inputs.
        inputs: Vec<BackendTensorUid>,
        /// Native identities of the enclosing reduction outputs.
        outputs: Vec<BackendTensorUid>,
        /// Ascending, distinct reduction-axis indices.
        dimensions: Vec<i64>,
        /// Requested identity values and compute precision; `None` inherits the corresponding output encoding.
        accumulators: Vec<ReducerAccumulator<BackendTensorUid>>,
    },
}

/// Finalized native ordinary or user-defined reduction with retained metadata and owned accumulator identity storage.
#[derive(Debug)]
pub struct ReductionOperation {
    descriptor: BackendDescriptor,
    metadata: ReductionOperationMetadata,
    _initial_values: Vec<ScalarStorage>,
}

impl ReductionOperation {
    /// Creates an ordinary reduction with shape-selected axes and an optional group-offset tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if native descriptor creation, attribute assignment, or finalization fails.
    pub fn create(
        reduction: &ReductionDescriptor,
        x: &Tensor,
        y: &Tensor,
        group_offset: Option<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationReduction)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationReductionDesc,
            reduction.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationReductionXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationReductionYDesc,
            y.descriptor(),
        )?;
        if let Some(offset) = group_offset {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationReductionGroupOffsetDesc,
                offset.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: ReductionOperationMetadata::Standard {
                descriptor: reduction.config(),
                input: x.id(),
                output: y.id(),
                group_offset: group_offset.map(Tensor::id),
            },
            _initial_values: Vec::new(),
        })
    }

    /// Retained operands, attributes and custom reducer slot mappings without native queries or allocation.
    /// The snapshot owns its values independently of the original tensor and configuration wrappers.
    #[must_use]
    pub fn metadata(&self) -> &ReductionOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native reduction descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

/// A single accumulator of a user-defined reducer body.
/// Declarative graphs use frontend tensor IDs; native construction requires backend tensor UIDs.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ReducerAccumulator<T> {
    /// Identity of the body tensor holding the accumulator value.
    pub accumulator: T,
    /// Identity of the body tensor holding the current input element.
    pub current: T,
    /// Identity of the body tensor holding the combined output value.
    pub output: T,
    /// Accumulator identity value, with the output element type.
    pub initial_value: ScalarValue,
    /// Compute type the reducer body evaluates in; defaults to the output element type.
    pub compute_type: Option<DataType>,
}

impl<T> ReducerAccumulator<T> {
    /// Maps body identities while preserving the identity value and compute precision.
    ///
    /// # Errors
    ///
    /// Returns the first mapping error while processing accumulator, current-element, and combined-output identities in order.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<ReducerAccumulator<U>, E> {
        Ok(ReducerAccumulator {
            accumulator: map(self.accumulator)?,
            current: map(self.current)?,
            output: map(self.output)?,
            initial_value: self.initial_value,
            compute_type: self.compute_type,
        })
    }
}

impl ReductionOperation {
    /// Creates a reduction whose finalized subgraph combines accumulator and current values.
    /// Dimensions must be ascending and distinct. Each output has an identity value and
    /// corresponding accumulator, current-element and output native UIDs in the subgraph.
    ///
    /// Returns the native `NotSupported` status when the loaded cuDNN runtime does
    /// not implement the user-defined reducer attributes declared by its headers.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or unequal input/output/accumulator counts, empty, negative, repeated, or unordered dimensions, an identity encoding that differs from its output, or native descriptor creation, attribute assignment, or finalization failure.
    pub fn create_custom(
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        dimensions: &[i64],
        body: &OperationGraph,
        accumulators: &[ReducerAccumulator<BackendTensorUid>],
    ) -> Result<Self> {
        if inputs.is_empty()
            || outputs.is_empty()
            || accumulators.len() != outputs.len()
            || inputs.len() != outputs.len()
            || dimensions.is_empty()
            || dimensions.iter().any(|&axis| axis < 0)
            || dimensions.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::InvalidDataShape);
        }
        for (output, accumulator) in outputs.iter().zip(accumulators) {
            if output.data_type() != accumulator.initial_value.data_type() {
                return Err(Error::UnsupportedDataType);
            }
        }
        let initial_values = accumulators
            .iter()
            .map(|a| ScalarStorage::new(a.initial_value))
            .collect::<Vec<_>>();
        let pointers = initial_values
            .iter()
            .map(|value| value.as_ptr().cast_mut().cast())
            .collect::<Vec<_>>();
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationReduction)?;
        descriptor.set_attribute_descriptor_slice(
            BackendAttributeName::OperationReductionXDesc,
            &inputs.iter().map(|t| t.descriptor()).collect::<Vec<_>>(),
        )?;
        descriptor.set_attribute_descriptor_slice(
            BackendAttributeName::OperationReductionYDesc,
            &outputs.iter().map(|t| t.descriptor()).collect::<Vec<_>>(),
        )?;
        descriptor
            .set_attribute_i64_slice(BackendAttributeName::OperationReductionDims, dimensions)?;
        // SAFETY: initial_values retains aligned native bytes for every correctly typed identity value.
        unsafe {
            descriptor.set_attribute_void_ptr_slice(
                BackendAttributeName::OperationReductionInitValues,
                &pointers,
            )
        }?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationReductionSubgraph,
            body.descriptor(),
        )?;
        descriptor.set_attribute_i64_slice(
            BackendAttributeName::OperationReductionSubgraphAccumulatorUids,
            &accumulators
                .iter()
                .map(|a| i64::from(a.accumulator))
                .collect::<Vec<_>>(),
        )?;
        descriptor.set_attribute_i64_slice(
            BackendAttributeName::OperationReductionSubgraphCurrentUids,
            &accumulators
                .iter()
                .map(|a| i64::from(a.current))
                .collect::<Vec<_>>(),
        )?;
        descriptor.set_attribute_i64_slice(
            BackendAttributeName::OperationReductionSubgraphOutputUids,
            &accumulators
                .iter()
                .map(|a| i64::from(a.output))
                .collect::<Vec<_>>(),
        )?;
        descriptor.set_attribute_enum_slice(
            BackendAttributeName::OperationReductionSubgraphComputeTypes,
            &accumulators
                .iter()
                .zip(outputs)
                .map(|(a, output)| a.compute_type.unwrap_or_else(|| output.data_type()))
                .collect::<Vec<_>>(),
        )?;
        descriptor.finalize()?;
        Ok(Self {
            descriptor,
            metadata: ReductionOperationMetadata::Custom {
                inputs: inputs.iter().map(|tensor| tensor.id()).collect(),
                outputs: outputs.iter().map(|tensor| tensor.id()).collect(),
                dimensions: dimensions.to_vec(),
                accumulators: accumulators.to_vec(),
            },
            _initial_values: initial_values,
        })
    }
}
