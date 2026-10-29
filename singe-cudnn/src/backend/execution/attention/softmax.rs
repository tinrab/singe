use std::result::Result as StdResult;

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::{BackendTensorUid, Tensor, TensorId},
    },
    data_type::DataType,
    error::Result,
};

/// Encoding of softmax log-sum-exp statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoftmaxStatsBase {
    /// Natural-log statistics, required by SDPA backward.
    #[default]
    NaturalLog,
    /// Natural-log LSE multiplied by `log2(e)`.
    Log2,
}

/// Probability output and independently optional row summaries.
/// Frontend inference uses output configurations and tensor IDs; native construction uses borrowed tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxOutputs<T = TensorId> {
    /// Probability output tensor or its frontend inference policy.
    pub output: T,
    /// Log-sum-exp, using the configured statistics base.
    pub stats: Option<T>,
    /// Row maximum, including the sink when present.
    pub max: Option<T>,
    /// Sum of shifted exponentials, including the sink contribution.
    pub sum_exp: Option<T>,
}

impl<T> SoftmaxOutputs<T> {
    /// Records the probability output with all optional row summaries disabled.
    pub fn new(output: T) -> Self {
        Self {
            output,
            stats: None,
            max: None,
            sum_exp: None,
        }
    }

    /// Borrows probabilities and requested summaries without requiring copyable tensor owners.
    pub fn as_ref(&self) -> SoftmaxOutputs<&T> {
        SoftmaxOutputs {
            output: &self.output,
            stats: self.stats.as_ref(),
            max: self.max.as_ref(),
            sum_exp: self.sum_exp.as_ref(),
        }
    }

    /// Maps every requested result while preserving omitted summaries.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> SoftmaxOutputs<U> {
        SoftmaxOutputs {
            output: map(self.output),
            stats: self.stats.map(&mut map),
            max: self.max.map(&mut map),
            sum_exp: self.sum_exp.map(map),
        }
    }

    /// Maps every requested output without changing its role or requesting absent summaries.
    ///
    /// # Errors
    ///
    /// Returns the first output-mapping error, visiting probabilities, statistics, maximum, and exponential sum in order and skipping absent summaries.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<SoftmaxOutputs<U>, E> {
        Ok(SoftmaxOutputs {
            output: map(self.output)?,
            stats: self.stats.map(&mut map).transpose()?,
            max: self.max.map(&mut map).transpose()?,
            sum_exp: self.sum_exp.map(map).transpose()?,
        })
    }
}

/// Native softmax settings, independent of the requested outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxOperationConfig<T> {
    /// Logarithm base used for saved log-sum-exp statistics.
    pub stats_base: SoftmaxStatsBase,
    /// Optional logits broadcastable to the row summaries.
    pub sink: Option<T>,
}

impl<T> SoftmaxOperationConfig<T> {
    /// Borrows the optional sink while retaining the selected statistics base.
    pub fn as_ref(&self) -> SoftmaxOperationConfig<&T> {
        SoftmaxOperationConfig {
            stats_base: self.stats_base,
            sink: self.sink.as_ref(),
        }
    }

    /// Maps the optional sink operand while retaining the statistics encoding.
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> SoftmaxOperationConfig<U> {
        SoftmaxOperationConfig {
            stats_base: self.stats_base,
            sink: self.sink.map(map),
        }
    }

    /// Maps the optional sink operand while retaining the statistics encoding.
    ///
    /// # Errors
    ///
    /// Returns the supplied mapping error if the sink is present and mapping fails.
    pub fn try_map<U, E>(
        self,
        map: impl FnOnce(T) -> StdResult<U, E>,
    ) -> StdResult<SoftmaxOperationConfig<U>, E> {
        Ok(SoftmaxOperationConfig {
            stats_base: self.stats_base,
            sink: self.sink.map(map).transpose()?,
        })
    }
}

impl<T> Default for SoftmaxOperationConfig<T> {
    fn default() -> Self {
        Self {
            stats_base: SoftmaxStatsBase::default(),
            sink: None,
        }
    }
}

/// Finalized native softmax stage retaining input/output identities, optional row summaries, and sink settings.
#[derive(Debug)]
pub struct SoftmaxOperation {
    descriptor: BackendDescriptor,
    input: BackendTensorUid,
    outputs: SoftmaxOutputs<BackendTensorUid>,
    config: SoftmaxOperationConfig<BackendTensorUid>,
    pub(crate) stats_data_type: Option<DataType>,
}

impl SoftmaxOperation {
    /// Creates the native softmax stage used by SDPA with independently optional summaries.
    /// Retains native tensor identities and accepted settings for query-free introspection.
    ///
    /// # Errors
    ///
    /// Returns an error if native descriptor creation, attribute assignment, or finalization fails. cuDNN validates tensor layouts, requested row summaries, and sink compatibility.
    pub fn create(
        input: &Tensor,
        outputs: SoftmaxOutputs<&Tensor>,
        config: SoftmaxOperationConfig<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationSoftmax)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSoftmaxXDesc,
            input.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSoftmaxYDesc,
            outputs.output.descriptor(),
        )?;
        if let Some(stats) = outputs.stats {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSoftmaxStatsDesc,
                stats.descriptor(),
            )?;
        }
        if let Some(max) = outputs.max {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSoftmaxMaxDesc,
                max.descriptor(),
            )?;
        }
        if let Some(sum_exp) = outputs.sum_exp {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSoftmaxSumExpDesc,
                sum_exp.descriptor(),
            )?;
        }
        if let Some(sink) = config.sink {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSoftmaxSinkDesc,
                sink.descriptor(),
            )?;
        }
        if outputs.stats.is_some() && config.stats_base == SoftmaxStatsBase::Log2 {
            descriptor.set_attribute_bool(BackendAttributeName::OperationSoftmaxStatsLog2, true)?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            input: input.id(),
            outputs: outputs.map(Tensor::id),
            config: config.map(Tensor::id),
            stats_data_type: outputs.stats.map(Tensor::data_type),
        })
    }

    /// Returns the retained native identity of the softmax input logits.
    #[must_use]
    pub fn input(&self) -> BackendTensorUid {
        self.input
    }

    /// Returns retained native probability and optional row-summary identities without querying cuDNN.
    #[must_use]
    pub fn outputs(&self) -> SoftmaxOutputs<BackendTensorUid> {
        self.outputs
    }

    /// Returns retained statistics encoding and optional native sink identity without querying cuDNN.
    #[must_use]
    pub fn config(&self) -> SoftmaxOperationConfig<BackendTensorUid> {
        self.config
    }

    /// Borrows the finalized native softmax descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
