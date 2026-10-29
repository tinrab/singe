use serde::{Deserialize, Serialize};

pub use crate::backend::execution::rng::{RandomNumberDistribution, RandomNumberSeedSource};
use crate::{
    backend::{
        execution::rng::{RngDescriptor, RngOperation},
        tensor::TensorId,
    },
    data_type::DataType,
    error::Result,
    frontend::{
        graph::Graph,
        lower::{LoweredOperation, LoweringContext, tensor_at},
    },
};

const RNG_SEED: &str = "rng seed";
const RNG_OFFSET: &str = "rng offset";

/// Frontend random-number-generator operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomNumberGeneratorOperation {
    /// Destination tensor identity for generated random values.
    pub output: TensorId,
    /// Active distribution, result scope, and seed/offset configuration.
    pub config: RandomNumberGeneratorConfig,
}

impl RandomNumberGeneratorOperation {
    /// Records the random-value destination and RNG configuration for later graph validation.
    #[must_use]
    pub fn new(output: TensorId, config: RandomNumberGeneratorConfig) -> Self {
        Self { output, config }
    }
}

impl RandomNumberGeneratorOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        self.config.distribution().validate()?;
        graph.tensor_spec(self.output)?;
        for (tensor, name) in [
            (self.config.seed_tensor(), RNG_SEED),
            (self.config.offset(), RNG_OFFSET),
        ] {
            if let Some(tensor) = tensor {
                graph.validate_tensor_data_type_supported(
                    tensor,
                    &[DataType::I32, DataType::I64],
                    name,
                )?;
                graph.validate_tensor_scalar_shape(tensor, name)?;
                let spec = graph.tensor_spec(tensor)?;
                if !spec.is_by_value() {
                    let strides = vec![1; spec.shape.strides().len()];
                    graph.validate_tensor_strides(tensor, &strides, name)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        let descriptor = RngDescriptor::create(self.config.distribution())?;
        let seed = self
            .config
            .seed_source()
            .try_map(|id| tensor_at(tensors, id))?;
        Ok(LoweredOperation::Rng(RngOperation::create(
            &descriptor,
            tensor_at(tensors, self.output)?,
            seed,
        )?))
    }
}

/// Attributes for a frontend RNG operation.
/// Used by composite attention/dropout graph patterns to generate random masks.
/// Tensor seeds and offsets use INT32 or INT64 and contain one logical scalar.
/// Native seed and offset descriptors require unit strides on every axis.
/// Preparation resets fused scalar strides to one, including declarations whose rank already matches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomNumberGeneratorConfig {
    name: Option<String>,
    distribution: RandomNumberDistribution,
    seed: RandomNumberSeedSource<TensorId>,
}

impl RandomNumberGeneratorConfig {
    /// Records an active distribution and complete seed source without validating parameters yet.
    #[must_use]
    pub fn new(
        distribution: RandomNumberDistribution,
        seed: RandomNumberSeedSource<TensorId>,
    ) -> Self {
        Self {
            name: None,
            distribution,
            seed,
        }
    }

    /// Records a Bernoulli distribution with an embedded host seed and no offset tensor.
    #[must_use]
    pub fn bernoulli(probability: f64, seed: i64) -> Self {
        Self::new(
            RandomNumberDistribution::Bernoulli { probability },
            RandomNumberSeedSource::Host {
                value: seed,
                offset: None,
            },
        )
    }

    /// Records uniform bounds with an embedded host seed and no offset tensor.
    #[must_use]
    pub fn uniform(minimum: f64, maximum: f64, seed: i64) -> Self {
        Self::new(
            RandomNumberDistribution::Uniform { minimum, maximum },
            RandomNumberSeedSource::Host {
                value: seed,
                offset: None,
            },
        )
    }

    /// Records normal-distribution parameters with an embedded host seed and no offset tensor.
    #[must_use]
    pub fn normal(mean: f64, standard_deviation: f64, seed: i64) -> Self {
        Self::new(
            RandomNumberDistribution::Normal {
                mean,
                standard_deviation,
            },
            RandomNumberSeedSource::Host {
                value: seed,
                offset: None,
            },
        )
    }

    /// Names inferred results `name::Y`, matching the C++ RNG output scope.
    /// Explicit output declarations retain their names; an empty scope disables naming.
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

    /// Sets the Philox offset for either a host or device seed.
    #[must_use]
    pub fn with_offset(mut self, offset: TensorId) -> Self {
        match &mut self.seed {
            RandomNumberSeedSource::Host { offset: slot, .. } => *slot = Some(offset),
            RandomNumberSeedSource::Device { offset: slot, .. } => *slot = offset,
        }
        self
    }

    /// Replaces the complete seed source while preserving the distribution.
    /// A host source can omit its offset; a device source always includes one.
    #[must_use]
    pub fn with_seed_source(mut self, seed: RandomNumberSeedSource<TensorId>) -> Self {
        self.seed = seed;
        self
    }

    /// Replaces the active distribution and all its parameters while retaining the seed, offset and scope.
    /// Parameters are validated when the operation is recorded, as with construction.
    #[must_use]
    pub fn with_distribution(mut self, distribution: RandomNumberDistribution) -> Self {
        self.distribution = distribution;
        self
    }

    /// Returns the active distribution and its parameters.
    #[must_use]
    pub fn distribution(&self) -> RandomNumberDistribution {
        self.distribution
    }

    /// Returns the embedded host seed, or `None` for a tensor-backed source.
    #[must_use]
    pub fn seed(&self) -> Option<i64> {
        self.seed.seed()
    }

    /// Returns the seed tensor identity, or `None` for an embedded host seed.
    #[must_use]
    pub fn seed_tensor(&self) -> Option<TensorId> {
        self.seed.seed_tensor()
    }

    /// Returns the complete host or tensor-backed seed and offset request.
    #[must_use]
    pub fn seed_source(&self) -> RandomNumberSeedSource<TensorId> {
        self.seed
    }

    /// Returns the Philox offset tensor identity, if supplied.
    #[must_use]
    pub fn offset(&self) -> Option<TensorId> {
        self.seed.offset()
    }
}
