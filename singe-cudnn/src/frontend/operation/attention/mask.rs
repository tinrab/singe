use serde::{Deserialize, Serialize};

use crate::backend::tensor::TensorId;

/// Diagonal used when comparing query and key positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionDiagonalAlignment {
    /// Aligns the query and key sequence starts, so equal positions lie on the diagonal.
    TopLeft,
    /// Aligns sequence ends by shifting the diagonal by key/value length minus query length.
    BottomRight,
}

/// Optional diagonal-band bounds and padding, matching the C++ attention attributes.
/// Positions satisfy `diagonal - left_window < key <= diagonal + right_window`.
/// Omitted bounds leave that side unrestricted. Bottom-right alignment shifts
/// the diagonal by key/value length minus query length for each batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionDiagonalBand {
    pub(crate) alignment: AttentionDiagonalAlignment,
    pub(crate) left_window: Option<i64>,
    pub(crate) right_window: Option<i64>,
    pub(crate) padding: bool,
}

impl AttentionDiagonalBand {
    /// Creates an unbounded diagonal band with the requested alignment and padding masking disabled.
    #[must_use]
    pub fn new(alignment: AttentionDiagonalAlignment) -> Self {
        Self {
            alignment,
            left_window: None,
            right_window: None,
            padding: false,
        }
    }

    /// Retains keys at or before the selected diagonal.
    #[must_use]
    pub fn causal(alignment: AttentionDiagonalAlignment) -> Self {
        Self::new(alignment).with_right_window(0)
    }

    /// Sets or clears the left bound, which counts visible tokens including self.
    #[must_use]
    pub fn with_left_window(mut self, window: impl Into<Option<i64>>) -> Self {
        self.left_window = window.into();
        self
    }

    /// Sets or clears the right bound, which counts future tokens.
    #[must_use]
    pub fn with_right_window(mut self, window: impl Into<Option<i64>>) -> Self {
        self.right_window = window.into();
        self
    }

    /// Enables or disables sequence-length padding masking while retaining alignment and window bounds.
    #[must_use]
    pub fn with_padding(mut self, padding: bool) -> Self {
        self.padding = padding;
        self
    }

    /// Replaces diagonal alignment while retaining window bounds and padding masking.
    #[must_use]
    pub fn with_alignment(mut self, alignment: AttentionDiagonalAlignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Returns the query/key diagonal alignment.
    #[must_use]
    pub fn alignment(self) -> AttentionDiagonalAlignment {
        self.alignment
    }

    /// Returns the optional left bound, counting visible tokens including the diagonal token.
    #[must_use]
    pub fn left_window(self) -> Option<i64> {
        self.left_window
    }

    /// Returns the optional right bound, counting future tokens beyond the diagonal.
    #[must_use]
    pub fn right_window(self) -> Option<i64> {
        self.right_window
    }

    /// Returns whether sequence-length padding masking is enabled.
    #[must_use]
    pub fn padding(self) -> bool {
        self.padding
    }

    /// Returns whether either left or right window bound is supplied.
    #[must_use]
    pub fn has_bounds(self) -> bool {
        self.left_window.is_some() || self.right_window.is_some()
    }
}

/// Structural attention masking with one optional diagonal alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttentionMaskMode {
    /// No structural attention mask.
    None,
    /// Masks positions outside supplied query and key/value sequence lengths.
    Padding,
    /// Uses the supplied diagonal alignment, optional window bounds, and padding flag.
    DiagonalBand(AttentionDiagonalBand),
    /// Uses the supplied block-mask tensor identity.
    Block(TensorId),
}

/// Fused attention causal-mask selection with sequence operands for bottom-right alignment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SdpaFusedMaskMode {
    /// No fused causal mask.
    #[default]
    None,
    /// Retains keys at or before the query position with sequence starts aligned.
    CausalTopLeft,
    /// Retains keys at or before the query diagonal with sequence ends aligned per batch.
    CausalBottomRight {
        /// Query sequence-length tensor identity used for the per-batch diagonal shift.
        sequence_length_query: TensorId,
        /// Key/value sequence-length tensor identity used for the per-batch diagonal shift.
        sequence_length_key_value: TensorId,
    },
}

impl SdpaFusedMaskMode {
    /// Returns whether the fused mask uses top-left causal alignment.
    #[must_use]
    pub fn is_causal_top_left(self) -> bool {
        matches!(self, Self::CausalTopLeft)
    }

    /// Returns whether the fused mask uses bottom-right causal alignment.
    #[must_use]
    pub fn is_causal_bottom_right(self) -> bool {
        matches!(self, Self::CausalBottomRight { .. })
    }

    /// Returns the query-length identity for bottom-right masking, or `None` for another mode.
    #[must_use]
    pub fn sequence_length_query(self) -> Option<TensorId> {
        match self {
            Self::CausalBottomRight {
                sequence_length_query,
                ..
            } => Some(sequence_length_query),
            _ => None,
        }
    }

    /// Returns the key/value-length identity for bottom-right masking, or `None` for another mode.
    #[must_use]
    pub fn sequence_length_key_value(self) -> Option<TensorId> {
        match self {
            Self::CausalBottomRight {
                sequence_length_key_value,
                ..
            } => Some(sequence_length_key_value),
            _ => None,
        }
    }
}
