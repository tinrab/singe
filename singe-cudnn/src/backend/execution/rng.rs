use std::result::Result as StdResult;

use num_enum::{IntoPrimitive, TryFromPrimitive};

use serde::{Deserialize, Serialize};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_cudnn_sys as sys;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::{BackendTensorUid, Tensor},
    },
    error::{Error, Result},
};

/// Distribution used by backend random number generator operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum RngDistribution {
    /// The Bernoulli distribution used for random number generation.
    /// Configure [`BackendAttributeName::RngBernoulliDistProbability`] to specify the probability of generating 1’s.
    Bernoulli = sys::cudnnRngDistribution_t::CUDNN_RNG_DISTRIBUTION_BERNOULLI as _,
    /// The uniform distribution used for random number generation.
    /// Configure [`BackendAttributeName::RngUniformDistMinimum`] and [`BackendAttributeName::RngUniformDistMaximum`] to specify the range for generated values.
    Uniform = sys::cudnnRngDistribution_t::CUDNN_RNG_DISTRIBUTION_UNIFORM as _,
    /// The normal distribution used for random number generation.
    /// Configure [`BackendAttributeName::RngNormalDistMean`] and [`BackendAttributeName::RngNormalDistStandardDeviation`] to specify the mean and standard deviation.
    Normal = sys::cudnnRngDistribution_t::CUDNN_RNG_DISTRIBUTION_NORMAL as _,
}

impl_enum_conversion!(sys::cudnnRngDistribution_t, RngDistribution);

impl_enum_display!(RngDistribution, {
    Self::Bernoulli => "CUDNN_RNG_DISTRIBUTION_BERNOULLI",
    Self::Uniform => "CUDNN_RNG_DISTRIBUTION_UNIFORM",
    Self::Normal => "CUDNN_RNG_DISTRIBUTION_NORMAL",
});

/// Active distribution and its parameters for frontend and native RNG construction.
/// Values are validated at graph recording and native descriptor creation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RandomNumberDistribution {
    /// Binary samples with a configured probability of producing one.
    Bernoulli {
        /// Probability of generating one; must lie in `[0, 1]`.
        #[serde(with = "crate::scalar::float::f64_bits")]
        probability: f64,
    },
    /// Uniform samples between configured native minimum and maximum bounds.
    Uniform {
        /// Finite lower distribution bound, no greater than `maximum`.
        #[serde(with = "crate::scalar::float::f64_bits")]
        minimum: f64,
        /// Finite upper distribution bound, no less than `minimum`.
        #[serde(with = "crate::scalar::float::f64_bits")]
        maximum: f64,
    },
    /// Normally distributed samples with configured mean and standard deviation.
    Normal {
        /// Finite distribution mean.
        #[serde(with = "crate::scalar::float::f64_bits")]
        mean: f64,
        /// Finite nonnegative standard deviation; zero is passed to native support selection.
        #[serde(with = "crate::scalar::float::f64_bits")]
        standard_deviation: f64,
    },
}

impl RandomNumberDistribution {
    /// Native distribution selector, derived from the active parameter variant.
    #[must_use]
    pub fn kind(self) -> RngDistribution {
        match self {
            Self::Bernoulli { .. } => RngDistribution::Bernoulli,
            Self::Uniform { .. } => RngDistribution::Uniform,
            Self::Normal { .. } => RngDistribution::Normal,
        }
    }

    /// Validates the active distribution without creating a native descriptor.
    ///
    /// # Errors
    /// Returns an error for probabilities outside `[0, 1]`, reversed or nonfinite
    /// uniform bounds, nonfinite means, or negative/nonfinite standard deviations.
    /// Equal uniform bounds and zero standard deviation reach native support selection.
    pub fn validate(self) -> Result<()> {
        match self {
            Self::Bernoulli { probability } => {
                if !(0.0..=1.0).contains(&probability) {
                    return Err(Error::out_of_range("probability"));
                }
            }
            Self::Uniform { minimum, maximum } => {
                if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
                    return Err(Error::DescriptorInvalidAttributeValue(
                        BackendAttributeName::RngUniformDistMaximum,
                        String::from(
                            "minimum and maximum must be finite with minimum no greater than maximum",
                        ),
                    ));
                }
            }
            Self::Normal {
                mean,
                standard_deviation,
            } => {
                if !mean.is_finite() {
                    return Err(Error::out_of_range("mean"));
                }
                if !standard_deviation.is_finite() || standard_deviation < 0.0 {
                    return Err(Error::out_of_range("standard_deviation"));
                }
            }
        }
        Ok(())
    }
}

/// Finalized native RNG descriptor retaining its complete active distribution.
#[derive(Debug)]
pub struct RngDescriptor {
    descriptor: BackendDescriptor,
    distribution: RandomNumberDistribution,
}

impl RngDescriptor {
    /// Creates a descriptor with the supplied distribution and double-precision parameters.
    ///
    /// # Errors
    /// Returns an error for invalid distribution parameters or native descriptor failure.
    pub fn create(distribution: RandomNumberDistribution) -> Result<Self> {
        distribution.validate()?;
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Rng)?;
        descriptor
            .set_attribute_enum(BackendAttributeName::RngDistribution, distribution.kind())?;
        match distribution {
            RandomNumberDistribution::Bernoulli { probability } => {
                descriptor.set_attribute_f64(
                    BackendAttributeName::RngBernoulliDistProbability,
                    probability,
                )?;
            }
            RandomNumberDistribution::Uniform { minimum, maximum } => {
                descriptor
                    .set_attribute_f64(BackendAttributeName::RngUniformDistMinimum, minimum)?;
                descriptor
                    .set_attribute_f64(BackendAttributeName::RngUniformDistMaximum, maximum)?;
            }
            RandomNumberDistribution::Normal {
                mean,
                standard_deviation,
            } => {
                descriptor.set_attribute_f64(BackendAttributeName::RngNormalDistMean, mean)?;
                descriptor.set_attribute_f64(
                    BackendAttributeName::RngNormalDistStandardDeviation,
                    standard_deviation,
                )?;
            }
        }
        descriptor.finalize()?;
        Ok(Self {
            descriptor,
            distribution,
        })
    }

    /// Retained parameters without native queries, including the exact uniform minimum and maximum.
    #[must_use]
    pub fn distribution(&self) -> RandomNumberDistribution {
        self.distribution
    }

    /// Borrows the finalized native random-number descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

/// Retained distribution, output identity and seed/offset configuration.
/// Does not borrow the original RNG descriptor or tensor wrappers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RngOperationMetadata {
    /// Retained active distribution and its parameters.
    pub distribution: RandomNumberDistribution,
    /// Native identity of the generated output tensor.
    pub output: BackendTensorUid,
    /// Embedded host seed or native seed/offset tensor identities.
    pub seed_source: RandomNumberSeedSource<BackendTensorUid>,
}

/// Finalized native random-number operation retaining distribution, output identity, and seed/offset configuration.
#[derive(Debug)]
pub struct RngOperation {
    descriptor: BackendDescriptor,
    metadata: RngOperationMetadata,
}

/// Seed and Philox offset shared by frontend identities, native tensor references and native UIDs.
/// A host seed is embedded in the operation descriptor and can have an offset tensor.
/// A device seed carries its required offset together with the seed tensor.
/// Tensor-based sources can use device or runtime by-value storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RandomNumberSeedSource<T> {
    /// An embedded host seed with an optional Philox offset tensor.
    Host {
        /// Seed value embedded in the native operation descriptor.
        value: i64,
        /// Optional Philox offset tensor.
        offset: Option<T>,
    },
    /// A tensor-backed seed paired with its required Philox offset tensor.
    Device {
        /// Seed tensor, using device or runtime by-value storage.
        tensor: T,
        /// Required Philox offset tensor paired with the seed.
        offset: T,
    },
}

impl<T> RandomNumberSeedSource<T> {
    /// Borrows tensor operands while preserving the host value and seed/offset pairing.
    pub fn as_ref(&self) -> RandomNumberSeedSource<&T> {
        match self {
            Self::Host { value, offset } => RandomNumberSeedSource::Host {
                value: *value,
                offset: offset.as_ref(),
            },
            Self::Device { tensor, offset } => RandomNumberSeedSource::Device { tensor, offset },
        }
    }

    /// Embedded host seed, without consuming owned tensor operands.
    pub fn seed(&self) -> Option<i64> {
        match self {
            Self::Host { value, .. } => Some(*value),
            Self::Device { .. } => None,
        }
    }

    /// Returns the seed tensor for a tensor-backed source, or `None` for an embedded host seed.
    pub fn seed_tensor(self) -> Option<T> {
        match self {
            Self::Host { .. } => None,
            Self::Device { tensor, .. } => Some(tensor),
        }
    }

    /// Returns the configured offset tensor, or `None` when a host source omits it.
    pub fn offset(self) -> Option<T> {
        match self {
            Self::Host { offset, .. } => offset,
            Self::Device { offset, .. } => Some(offset),
        }
    }

    /// Maps tensor operands while preserving the selected seed representation and offset pairing.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> RandomNumberSeedSource<U> {
        match self {
            Self::Host { value, offset } => RandomNumberSeedSource::Host {
                value,
                offset: offset.map(map),
            },
            Self::Device { tensor, offset } => RandomNumberSeedSource::Device {
                tensor: map(tensor),
                offset: map(offset),
            },
        }
    }

    /// Maps tensor identities while preserving the selected seed representation and offset pairing.
    ///
    /// # Errors
    ///
    /// Returns the first tensor-mapping error; tensor-backed seeds are mapped before their required offsets.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<RandomNumberSeedSource<U>, E> {
        Ok(match self {
            Self::Host { value, offset } => RandomNumberSeedSource::Host {
                value,
                offset: offset.map(&mut map).transpose()?,
            },
            Self::Device { tensor, offset } => RandomNumberSeedSource::Device {
                tensor: map(tensor)?,
                offset: map(offset)?,
            },
        })
    }
}

impl RngOperation {
    /// Creates an RNG operation with a host or device seed and optional Philox offset.
    /// An offset is honored with either seed representation.
    /// Device seeds require an offset tensor, as in the C++ graph API.
    /// Tensor seeds and offsets contain one scalar with unit dimensions and strides on every axis.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects the descriptors or their configuration.
    pub fn create(
        rng: &RngDescriptor,
        y: &Tensor,
        seed: RandomNumberSeedSource<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationRng)?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationRngDesc, rng.descriptor())?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationRngYDesc, y.descriptor())?;
        let offset = match seed {
            RandomNumberSeedSource::Host { value, offset } => {
                descriptor.set_attribute_i64(BackendAttributeName::OperationRngSeed, value)?;
                offset
            }
            RandomNumberSeedSource::Device { tensor, offset } => {
                descriptor.set_attribute_descriptor(
                    BackendAttributeName::OperationRngSeed,
                    tensor.descriptor(),
                )?;
                Some(offset)
            }
        };
        if let Some(offset) = offset {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationRngOffsetDesc,
                offset.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: RngOperationMetadata {
                distribution: rng.distribution(),
                output: y.id(),
                seed_source: seed.map(Tensor::id),
            },
        })
    }

    /// Complete retained native attributes and identities without queries or allocation.
    #[must_use]
    pub fn metadata(&self) -> &RngOperationMetadata {
        &self.metadata
    }

    /// Retained seed value and actual native operand UIDs, without querying cuDNN.
    #[must_use]
    pub fn seed_source(&self) -> RandomNumberSeedSource<BackendTensorUid> {
        self.metadata.seed_source
    }

    /// Borrows the finalized native random-number descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
