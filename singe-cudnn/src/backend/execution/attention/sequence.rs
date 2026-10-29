use std::result::Result as StdResult;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Lengths for one side of an attention problem.
/// Cumulative lengths contain batch offsets and have one more entry than per-batch lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SequenceLength<T> {
    /// One sequence length per batch entry.
    PerBatch(T),
    /// Cumulative token offsets with `batch + 1` entries, beginning at zero.
    Cumulative(T),
}

impl<T> SequenceLength<T> {
    /// Consumes the length representation and returns its tensor operand.
    pub fn tensor(self) -> T {
        match self {
            Self::PerBatch(tensor) | Self::Cumulative(tensor) => tensor,
        }
    }

    /// Returns whether this operand holds cumulative token offsets.
    pub fn is_cumulative(&self) -> bool {
        matches!(self, Self::Cumulative(_))
    }

    /// Returns the tensor only for the per-batch representation; cumulative offsets yield `None`.
    pub fn per_batch(self) -> Option<T> {
        match self {
            Self::PerBatch(tensor) => Some(tensor),
            Self::Cumulative(_) => None,
        }
    }

    /// Borrows the tensor without changing its length representation.
    pub fn as_ref(&self) -> SequenceLength<&T> {
        match self {
            Self::PerBatch(tensor) => SequenceLength::PerBatch(tensor),
            Self::Cumulative(tensor) => SequenceLength::Cumulative(tensor),
        }
    }

    /// Maps the tensor operand while preserving the length representation.
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> SequenceLength<U> {
        match self {
            Self::PerBatch(tensor) => SequenceLength::PerBatch(map(tensor)),
            Self::Cumulative(tensor) => SequenceLength::Cumulative(map(tensor)),
        }
    }

    /// Maps the tensor operand while preserving the length representation.
    ///
    /// # Errors
    ///
    /// Returns the error produced by the supplied tensor mapping.
    pub fn try_map<U, E>(
        self,
        map: impl FnOnce(T) -> StdResult<U, E>,
    ) -> StdResult<SequenceLength<U>, E> {
        Ok(match self {
            Self::PerBatch(tensor) => SequenceLength::PerBatch(map(tensor)?),
            Self::Cumulative(tensor) => SequenceLength::Cumulative(map(tensor)?),
        })
    }
}

/// Independent query and key/value sequence-length representations.
/// A cumulative tensor contains `batch + 1` offsets, beginning with zero.
/// A per-batch tensor contains `batch` lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionSequenceInputs<T> {
    /// Optional query-side lengths in per-batch or cumulative form.
    pub query: Option<SequenceLength<T>>,
    /// Optional key/value-side lengths in per-batch or cumulative form.
    pub key_value: Option<SequenceLength<T>>,
}

impl<T> AttentionSequenceInputs<T> {
    /// Groups a complete pair, allowing each side to use a different representation.
    pub fn new(query: SequenceLength<T>, key_value: SequenceLength<T>) -> Self {
        Self {
            query: Some(query),
            key_value: Some(key_value),
        }
    }

    /// Groups query and key/value tensors containing one sequence length per batch entry.
    pub fn per_batch(query: T, key_value: T) -> Self {
        Self::new(
            SequenceLength::PerBatch(query),
            SequenceLength::PerBatch(key_value),
        )
    }

    /// Groups query and key/value tensors containing cumulative token offsets.
    pub fn cumulative(query: T, key_value: T) -> Self {
        Self::new(
            SequenceLength::Cumulative(query),
            SequenceLength::Cumulative(key_value),
        )
    }

    /// Returns whether both query and key/value lengths are absent.
    pub fn is_empty(&self) -> bool {
        self.query.is_none() && self.key_value.is_none()
    }

    /// Forward SDPA accepts no lengths or a complete pair, including mixed representations.
    /// Standalone masks and backward descriptors retain their own operand rules.
    pub(crate) fn validate_forward(&self) -> Result<()> {
        if self.query.is_some() != self.key_value.is_some() {
            return Err(Error::SdpaSequenceLengthsIncomplete);
        }
        Ok(())
    }

    /// Returns whether either side uses cumulative token offsets.
    pub fn has_cumulative(&self) -> bool {
        self.query
            .as_ref()
            .is_some_and(SequenceLength::is_cumulative)
            || self
                .key_value
                .as_ref()
                .is_some_and(SequenceLength::is_cumulative)
    }

    /// Consumes the group and iterates over present tensors in query then key/value order.
    pub fn tensors(self) -> impl Iterator<Item = T> {
        self.query
            .into_iter()
            .chain(self.key_value)
            .map(SequenceLength::tensor)
    }

    /// Returns both tensors only when both sides use per-batch lengths; otherwise returns `None`.
    pub fn per_batch_pair(self) -> Option<(T, T)> {
        Some((self.query?.per_batch()?, self.key_value?.per_batch()?))
    }

    /// Borrows independently optional length tensors, including non-Copy owners.
    pub fn as_ref(&self) -> AttentionSequenceInputs<&T> {
        AttentionSequenceInputs {
            query: self.query.as_ref().map(SequenceLength::as_ref),
            key_value: self.key_value.as_ref().map(SequenceLength::as_ref),
        }
    }

    /// Maps both sides without changing omissions or length representations.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> AttentionSequenceInputs<U> {
        AttentionSequenceInputs {
            query: self.query.map(|lengths| lengths.map(&mut map)),
            key_value: self.key_value.map(|lengths| lengths.map(&mut map)),
        }
    }

    /// Maps both sides while preserving per-batch and cumulative representations.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; query is mapped before key/value and absent operands are skipped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<AttentionSequenceInputs<U>, E> {
        Ok(AttentionSequenceInputs {
            query: self
                .query
                .map(|lengths| lengths.try_map(&mut map))
                .transpose()?,
            key_value: self
                .key_value
                .map(|lengths| lengths.try_map(&mut map))
                .transpose()?,
        })
    }
}

impl<T> Default for AttentionSequenceInputs<T> {
    fn default() -> Self {
        Self {
            query: None,
            key_value: None,
        }
    }
}
