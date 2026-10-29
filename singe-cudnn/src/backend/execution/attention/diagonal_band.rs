use std::result::Result as StdResult;

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        execution::attention::{AttentionSequenceInputs, SequenceLength},
        pointwise::PointwiseMode,
        tensor::{BackendTensorUid, Tensor, TensorId},
    },
    error::Result,
};

/// One side of a diagonal-band mask, with an optional shift for a right bound.
/// An unshifted right bound compares against the diagonal itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DiagonalBandMaskBound<T> {
    /// A tensor-backed left diagonal-band bound.
    Left(T),
    /// A right diagonal-band bound with an optional tensor-backed shift.
    Right {
        /// Optional shift from the diagonal; `None` compares against the diagonal itself.
        shift: Option<T>,
    },
}

impl<T> Default for DiagonalBandMaskBound<T> {
    fn default() -> Self {
        Self::Right { shift: None }
    }
}

impl<T> DiagonalBandMaskBound<T> {
    /// Returns the left-bound or right-shift tensor; an unshifted right bound returns `None`.
    pub fn tensor(self) -> Option<T> {
        match self {
            Self::Left(tensor) => Some(tensor),
            Self::Right { shift } => shift,
        }
    }

    /// Borrows the bound operand while preserving an unshifted right bound.
    pub fn as_ref(&self) -> DiagonalBandMaskBound<&T> {
        match self {
            Self::Left(tensor) => DiagonalBandMaskBound::Left(tensor),
            Self::Right { shift } => DiagonalBandMaskBound::Right {
                shift: shift.as_ref(),
            },
        }
    }

    /// Maps the bound tensor while retaining its side and any absent right shift.
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> DiagonalBandMaskBound<U> {
        match self {
            Self::Left(tensor) => DiagonalBandMaskBound::Left(map(tensor)),
            Self::Right { shift } => DiagonalBandMaskBound::Right {
                shift: shift.map(map),
            },
        }
    }

    /// Maps the bound operand without changing its side or optional shift.
    ///
    /// # Errors
    ///
    /// Returns the supplied mapping error if a bound tensor is present and mapping fails.
    pub fn try_map<U, E>(
        self,
        map: impl FnOnce(T) -> StdResult<U, E>,
    ) -> StdResult<DiagonalBandMaskBound<U>, E> {
        Ok(match self {
            Self::Left(tensor) => DiagonalBandMaskBound::Left(map(tensor)?),
            Self::Right { shift } => DiagonalBandMaskBound::Right {
                shift: shift.map(map).transpose()?,
            },
        })
    }
}

/// Comparison, scope, optional sequence lengths and one tensor-backed diagonal bound.
/// Operands can be frontend identities, borrowed native tensors or retained native UIDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagonalBandMaskConfig<T = TensorId> {
    #[serde(default)]
    name: Option<String>,
    comparison_mode: PointwiseMode,
    sequence: AttentionSequenceInputs<T>,
    bound: DiagonalBandMaskBound<T>,
}

impl<T> DiagonalBandMaskConfig<T> {
    /// Creates an unnamed diagonal-band request with the selected comparison, no sequence lengths, and an unshifted right bound.
    #[must_use]
    pub fn new(comparison_mode: PointwiseMode) -> Self {
        Self {
            name: None,
            comparison_mode,
            sequence: AttentionSequenceInputs::default(),
            bound: DiagonalBandMaskBound::default(),
        }
    }

    /// Names inferred results `name::Y` without renaming explicit output declarations.
    /// An empty scope leaves inferred results unnamed.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty result-name scope, or `None` when unnamed.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Replaces optional query and key/value lengths while retaining comparison and bound settings.
    #[must_use]
    pub fn with_sequence_inputs(mut self, sequence: AttentionSequenceInputs<T>) -> Self {
        self.sequence = sequence;
        self
    }

    /// Replaces the complete side/shift choice, clearing any previous bound operand.
    #[must_use]
    pub fn with_bound(mut self, bound: DiagonalBandMaskBound<T>) -> Self {
        self.bound = bound;
        self
    }

    /// Returns the native pointwise comparison used to form the mask.
    pub fn comparison_mode(&self) -> PointwiseMode {
        self.comparison_mode
    }

    /// Maps operands without changing the scope, comparison or mask geometry.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> DiagonalBandMaskConfig<U> {
        DiagonalBandMaskConfig {
            name: self.name,
            comparison_mode: self.comparison_mode,
            sequence: self.sequence.map(&mut map),
            bound: self.bound.map(map),
        }
    }

    /// Maps every sequence and bound operand while preserving scope and comparison settings.
    ///
    /// # Errors
    ///
    /// Returns the first tensor-mapping error, processing query and key/value lengths before the bound operand.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<DiagonalBandMaskConfig<U>, E> {
        Ok(DiagonalBandMaskConfig {
            name: self.name,
            comparison_mode: self.comparison_mode,
            sequence: self.sequence.try_map(&mut map)?,
            bound: self.bound.try_map(map)?,
        })
    }
}

impl<T: Copy> DiagonalBandMaskConfig<T> {
    /// Returns the query and key/value sequence-length operands.
    pub fn sequence_inputs(&self) -> AttentionSequenceInputs<T> {
        self.sequence
    }

    /// Returns the selected mask side and optional tensor-backed shift.
    pub fn bound(&self) -> DiagonalBandMaskBound<T> {
        self.bound
    }
}

/// Complete native mask request, independent of the original tensor wrappers.
#[derive(Debug, Clone)]
pub struct DiagonalBandMaskOperationMetadata {
    /// Native identity of the input tensor whose positions are masked.
    pub input: BackendTensorUid,
    /// Values selected at positions excluded by the mask.
    pub replacement: BackendTensorUid,
    /// Native identity of the masked output tensor.
    pub output: BackendTensorUid,
    /// Retained comparison, name scope, sequence operands, and diagonal-bound identity.
    pub config: DiagonalBandMaskConfig<BackendTensorUid>,
}

/// Finalized native diagonal-band mask operation with retained operand identities and mask settings.
#[derive(Debug)]
pub struct DiagonalBandMaskOperation {
    descriptor: BackendDescriptor,
    metadata: DiagonalBandMaskOperationMetadata,
}

impl DiagonalBandMaskOperation {
    /// Creates a diagonal-band mask selecting input `x` or replacement `b` into output `y`.
    ///
    /// # Errors
    ///
    /// Returns an error if native descriptor creation, attribute assignment, or finalization fails. cuDNN validates comparison, bound, sequence, and tensor compatibility.
    pub fn create(
        x: &Tensor,
        b: &Tensor,
        y: &Tensor,
        config: DiagonalBandMaskConfig<&Tensor>,
    ) -> Result<Self> {
        let sequence = config.sequence_inputs();
        let bound = config.bound();
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationDiagonalBandMask)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationDiagonalBandMaskXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationDiagonalBandMaskBDesc,
            b.descriptor(),
        )?;
        if let Some(lengths) = sequence.query {
            let attribute = match lengths {
                SequenceLength::PerBatch(_) => {
                    BackendAttributeName::OperationDiagonalBandMaskSeqLenQDesc
                }
                SequenceLength::Cumulative(_) => {
                    BackendAttributeName::OperationDiagonalBandMaskCuSeqLenQDesc
                }
            };
            descriptor.set_attribute_descriptor(attribute, lengths.tensor().descriptor())?;
        }
        if let Some(lengths) = sequence.key_value {
            let attribute = match lengths {
                SequenceLength::PerBatch(_) => {
                    BackendAttributeName::OperationDiagonalBandMaskSeqLenKvDesc
                }
                SequenceLength::Cumulative(_) => {
                    BackendAttributeName::OperationDiagonalBandMaskCuSeqLenKvDesc
                }
            };
            descriptor.set_attribute_descriptor(attribute, lengths.tensor().descriptor())?;
        }
        let (attribute, tensor) = match bound {
            DiagonalBandMaskBound::Left(tensor) => (
                BackendAttributeName::OperationDiagonalBandMaskLeftBoundDesc,
                Some(tensor),
            ),
            DiagonalBandMaskBound::Right { shift } => (
                BackendAttributeName::OperationDiagonalBandMaskShiftRightBoundDesc,
                shift,
            ),
        };
        if let Some(tensor) = tensor {
            descriptor.set_attribute_descriptor(attribute, tensor.descriptor())?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationDiagonalBandMaskYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationDiagonalBandMaskComparisonMode,
            config.comparison_mode(),
        )?;
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: DiagonalBandMaskOperationMetadata {
                input: x.id(),
                replacement: b.id(),
                output: y.id(),
                config: config.map(Tensor::id),
            },
        })
    }

    /// Complete retained request, including X, B and Y, without native queries or allocation.
    #[must_use]
    pub fn metadata(&self) -> &DiagonalBandMaskOperationMetadata {
        &self.metadata
    }

    /// Retained comparison, scope, sequence representations and bound UIDs without native queries.
    #[must_use]
    pub fn config(&self) -> &DiagonalBandMaskConfig<BackendTensorUid> {
        &self.metadata.config
    }

    /// Borrows the finalized native diagonal-band mask descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
