use serde::{Deserialize, Serialize};

use crate::{backend::tensor::TensorId, frontend::operation::RandomNumberSeedSource};

/// Dropout configuration for scaled dot product attention.
///
/// cuDNN SDPA can apply dropout after softmax using Philox RNG seed/offset
/// inputs, or a custom mask and scale through score modifiers. This config
/// represents the Philox-style probability and seed/offset path.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AttentionDropoutConfig {
    #[serde(with = "crate::scalar::float::f32_bits")]
    pub(crate) probability: f32,
    pub(crate) seed: RandomNumberSeedSource<TensorId>,
}

impl AttentionDropoutConfig {
    /// Uses the shared RNG seed configuration, including the required offset for a device seed.
    #[must_use]
    pub fn new(probability: f32, seed: RandomNumberSeedSource<TensorId>) -> Self {
        Self { probability, seed }
    }

    /// Returns the requested probability of dropping a softmax value.
    #[must_use]
    pub fn probability(&self) -> f32 {
        self.probability
    }

    /// Bernoulli probability in native descriptor precision, without an intermediate FP32 subtraction.
    #[must_use]
    pub fn keep_probability(&self) -> f64 {
        1.0 - f64::from(self.probability)
    }

    /// Scale applied to surviving values, using the C++ frontend's FP32 arithmetic.
    #[must_use]
    pub fn scale(&self) -> f32 {
        1.0 / self.inverse_scale()
    }

    /// FP32 inverse scale used by attention backward's softmax correction.
    #[must_use]
    pub fn inverse_scale(&self) -> f32 {
        1.0 - self.probability
    }

    /// Returns the embedded host seed, or `None` for a tensor-backed seed.
    #[must_use]
    pub fn seed(&self) -> Option<i64> {
        self.seed.seed()
    }

    /// Returns the seed tensor identity, or `None` for an embedded host seed.
    #[must_use]
    pub fn seed_tensor(&self) -> Option<TensorId> {
        self.seed.seed_tensor()
    }

    /// Returns the complete seed representation and offset request.
    #[must_use]
    pub fn seed_source(&self) -> RandomNumberSeedSource<TensorId> {
        self.seed
    }

    /// Returns the configured Philox offset tensor identity, if supplied.
    #[must_use]
    pub fn offset(&self) -> Option<TensorId> {
        self.seed.offset()
    }
}

/// Caller-provided dropout mask and its scalar multiplier.
/// Both operands are required whenever custom dropout is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionDropoutMask {
    /// Caller-provided dropout mask tensor identity.
    pub mask: TensorId,
    /// Scalar multiplier tensor applied with the caller-provided mask.
    pub scale: TensorId,
}

impl AttentionDropoutMask {
    /// Records the required custom mask and scaling tensor identities.
    #[must_use]
    pub fn new(mask: TensorId, scale: TensorId) -> Self {
        Self { mask, scale }
    }
}

/// Dropout mode for high-level SDPA helpers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttentionDropoutMode {
    /// Disables dropout.
    None,
    /// Generates a dropout mask from a probability and Philox seed/offset request.
    Philox(AttentionDropoutConfig),
    /// Uses the caller-provided mask and scalar multiplier.
    CustomMask(AttentionDropoutMask),
}

/// Direct native SDPA dropout probability, seed, offset, and optional RNG-dump destination.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DirectSdpaDropout {
    #[serde(with = "crate::scalar::float::f32_bits")]
    pub(crate) probability: f32,
    pub(crate) seed: TensorId,
    pub(crate) offset: TensorId,
    pub(crate) random_number_generator_dump: Option<TensorId>,
}

impl DirectSdpaDropout {
    /// Configures native Philox dropout with its required seed and offset tensors.
    #[must_use]
    pub fn new(probability: f32, seed: TensorId, offset: TensorId) -> Self {
        Self {
            probability,
            seed,
            offset,
            random_number_generator_dump: None,
        }
    }

    /// Sets or clears the optional dump output while retaining probability, seed and offset.
    #[must_use]
    pub fn with_random_number_generator_dump(
        mut self,
        random_number_generator_dump: impl Into<Option<TensorId>>,
    ) -> Self {
        self.random_number_generator_dump = random_number_generator_dump.into();
        self
    }

    /// Returns the requested dropout probability.
    #[must_use]
    pub fn probability(self) -> f32 {
        self.probability
    }

    /// Returns the required native seed tensor identity.
    #[must_use]
    pub fn seed(self) -> TensorId {
        self.seed
    }

    /// Returns the required Philox offset tensor identity.
    #[must_use]
    pub fn offset(self) -> TensorId {
        self.offset
    }

    /// Returns the optional destination identity for generated random numbers.
    #[must_use]
    pub fn random_number_generator_dump(self) -> Option<TensorId> {
        self.random_number_generator_dump
    }
}
