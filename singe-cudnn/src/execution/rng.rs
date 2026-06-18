use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;

use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::{Error, Result},
    tensor::Tensor,
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

#[derive(Debug)]
pub struct RngDescriptor {
    descriptor: BackendDescriptor,
    distribution: RngDistribution,
}

impl RngDescriptor {
    pub fn create_bernoulli(probability: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(Error::OutOfRange {
                name: "probability".into(),
            });
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Rng)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::RngDistribution,
            BackendAttributeType::RngDistribution,
            RngDistribution::Bernoulli,
        )?;
        descriptor.set_attribute_f64(
            BackendAttributeName::RngBernoulliDistProbability,
            probability,
        )?;
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            distribution: RngDistribution::Bernoulli,
        })
    }

    pub fn create_uniform(minimum: f64, maximum: f64) -> Result<Self> {
        if minimum >= maximum {
            return Err(Error::DescriptorInvalidAttributeValue(
                BackendAttributeName::RngUniformDistMaximum,
                String::from("maximum must be greater than minimum"),
            ));
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Rng)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::RngDistribution,
            BackendAttributeType::RngDistribution,
            RngDistribution::Uniform,
        )?;
        descriptor.set_attribute_f64(BackendAttributeName::RngUniformDistMinimum, minimum)?;
        descriptor.set_attribute_f64(BackendAttributeName::RngUniformDistMaximum, maximum)?;
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            distribution: RngDistribution::Uniform,
        })
    }

    pub fn create_normal(mean: f64, standard_deviation: f64) -> Result<Self> {
        if standard_deviation <= 0.0 {
            return Err(Error::OutOfRange {
                name: "standard_deviation".into(),
            });
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Rng)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::RngDistribution,
            BackendAttributeType::RngDistribution,
            RngDistribution::Normal,
        )?;
        descriptor.set_attribute_f64(BackendAttributeName::RngNormalDistMean, mean)?;
        descriptor.set_attribute_f64(
            BackendAttributeName::RngNormalDistStandardDeviation,
            standard_deviation,
        )?;
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            distribution: RngDistribution::Normal,
        })
    }

    pub fn distribution(&self) -> RngDistribution {
        self.distribution
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct RngOperation {
    descriptor: BackendDescriptor,
}

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum RngSeed<'a> {
    Host(i64),
    Device(&'a Tensor),
}

impl RngOperation {
    pub fn create(
        rng: &RngDescriptor,
        y: &Tensor,
        seed: RngSeed<'_>,
        offset: Option<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationRng)?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationRngDesc, rng.descriptor())?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationRngYDesc, y.descriptor())?;
        match seed {
            RngSeed::Host(seed) => {
                descriptor.set_attribute_i64(BackendAttributeName::OperationRngSeed, seed)?;
            }
            RngSeed::Device(seed) => {
                descriptor.set_attribute_descriptor(
                    BackendAttributeName::OperationRngSeed,
                    seed.descriptor(),
                )?;
                let offset = offset.ok_or(Error::DescriptorMismatch {
                    name: "rng offset".into(),
                })?;
                descriptor.set_attribute_descriptor(
                    BackendAttributeName::OperationRngOffsetDesc,
                    offset.descriptor(),
                )?;
            }
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
