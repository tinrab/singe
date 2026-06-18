use serde::{Deserialize, Serialize};

use crate::tensor::TensorId;

/// Random number distribution for a frontend RNG operation.
///
/// cuDNN frontend attention patterns use an RNG node for dropout masks; this
/// enum captures the supported Rust-side distributions for that node.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RandomNumberDistribution {
    Bernoulli { probability: f64 },
    Uniform { minimum: f64, maximum: f64 },
    Normal { mean: f64, standard_deviation: f64 },
}

/// Source for frontend RNG seed data.
///
/// Attention dropout docs describe seed and offset tensors for Philox RNG
/// dropout. This Rust enum also supports host seeds for helper paths that
/// materialize seed data internally.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RandomNumberSeedSource {
    Host(i64),
    Device(TensorId),
}

/// Attributes for a frontend RNG operation.
///
/// Used by composite attention/dropout graph patterns to generate random masks
/// from a distribution, seed, and optional offset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomNumberGeneratorConfig {
    distribution: RandomNumberDistribution,
    seed: RandomNumberSeedSource,
    offset: Option<TensorId>,
}

impl RandomNumberGeneratorConfig {
    pub fn bernoulli(probability: f64, seed: i64) -> Self {
        Self {
            distribution: RandomNumberDistribution::Bernoulli { probability },
            seed: RandomNumberSeedSource::Host(seed),
            offset: None,
        }
    }

    pub fn uniform(minimum: f64, maximum: f64, seed: i64) -> Self {
        Self {
            distribution: RandomNumberDistribution::Uniform { minimum, maximum },
            seed: RandomNumberSeedSource::Host(seed),
            offset: None,
        }
    }

    pub fn normal(mean: f64, standard_deviation: f64, seed: i64) -> Self {
        Self {
            distribution: RandomNumberDistribution::Normal {
                mean,
                standard_deviation,
            },
            seed: RandomNumberSeedSource::Host(seed),
            offset: None,
        }
    }

    pub fn with_offset(mut self, offset: TensorId) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn with_seed_tensor(mut self, seed: TensorId, offset: TensorId) -> Self {
        self.seed = RandomNumberSeedSource::Device(seed);
        self.offset = Some(offset);
        self
    }

    pub fn distribution(&self) -> &RandomNumberDistribution {
        &self.distribution
    }

    pub fn seed(&self) -> Option<i64> {
        match self.seed {
            RandomNumberSeedSource::Host(seed) => Some(seed),
            RandomNumberSeedSource::Device(_) => None,
        }
    }

    pub fn seed_tensor(&self) -> Option<TensorId> {
        match self.seed {
            RandomNumberSeedSource::Host(_) => None,
            RandomNumberSeedSource::Device(seed) => Some(seed),
        }
    }

    pub fn seed_source(&self) -> &RandomNumberSeedSource {
        &self.seed
    }

    pub fn offset(&self) -> Option<TensorId> {
        self.offset
    }
}
