use std::{convert::Infallible, result::Result as StdResult, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        execution::attention::{
            AttentionSequenceInputs, SdpaSubgraph, SdpaSubgraphPorts, SequenceLength,
            SoftmaxOperation,
        },
        tensor::{BackendTensorUid, Tensor},
    },
    data_type::DataType,
    error::{Error, Result},
};

/// Optional FP8 scaling operands and pre-quantization absolute-maximum outputs.
/// Descale tensors may describe per-tensor or native block-scale layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaForwardQuantization<T> {
    /// Optional query dequantization tensor.
    pub descale_query: Option<T>,
    /// Optional key dequantization tensor.
    pub descale_key: Option<T>,
    /// Optional value dequantization tensor.
    pub descale_value: Option<T>,
    /// Optional dequantization tensor for softmax probabilities S.
    pub descale_scores: Option<T>,
    /// Optional quantization tensor for softmax probabilities S.
    pub scale_scores: Option<T>,
    /// Optional quantization tensor for the attention output.
    pub scale_output: Option<T>,
    /// FP32 maximum of softmax probabilities S before FP8 scaling, corresponding to `Amax_S`.
    pub absolute_max_probabilities: Option<T>,
    /// Optional FP32 absolute maximum of the attention output before quantization.
    pub absolute_max_output: Option<T>,
}

impl<T> Default for SdpaForwardQuantization<T> {
    fn default() -> Self {
        Self {
            descale_query: None,
            descale_key: None,
            descale_value: None,
            descale_scores: None,
            scale_scores: None,
            scale_output: None,
            absolute_max_probabilities: None,
            absolute_max_output: None,
        }
    }
}

impl<T> SdpaForwardQuantization<T> {
    /// Borrows only the scale and absolute-maximum operands that were supplied.
    pub fn as_ref(&self) -> SdpaForwardQuantization<&T> {
        SdpaForwardQuantization {
            descale_query: self.descale_query.as_ref(),
            descale_key: self.descale_key.as_ref(),
            descale_value: self.descale_value.as_ref(),
            descale_scores: self.descale_scores.as_ref(),
            scale_scores: self.scale_scores.as_ref(),
            scale_output: self.scale_output.as_ref(),
            absolute_max_probabilities: self.absolute_max_probabilities.as_ref(),
            absolute_max_output: self.absolute_max_output.as_ref(),
        }
    }

    /// Maps optional tensor operands without introducing omitted scales or outputs.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> SdpaForwardQuantization<U> {
        match self.try_map::<_, Infallible>(|tensor| Ok(map(tensor))) {
            Ok(quantization) => quantization,
            Err(never) => match never {},
        }
    }

    /// Maps tensor operands in declaration order while preserving their roles and optional omissions.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied mapping without mapping later operands.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<SdpaForwardQuantization<U>, E> {
        Ok(SdpaForwardQuantization {
            descale_query: self.descale_query.map(&mut map).transpose()?,
            descale_key: self.descale_key.map(&mut map).transpose()?,
            descale_value: self.descale_value.map(&mut map).transpose()?,
            descale_scores: self.descale_scores.map(&mut map).transpose()?,
            scale_scores: self.scale_scores.map(&mut map).transpose()?,
            scale_output: self.scale_output.map(&mut map).transpose()?,
            absolute_max_probabilities: self
                .absolute_max_probabilities
                .map(&mut map)
                .transpose()?,
            absolute_max_output: self.absolute_max_output.map(&mut map).transpose()?,
        })
    }
}

/// Required inputs and output of a native attention forward operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaForwardTensors<T> {
    /// Query tensor operand.
    pub query: T,
    /// Key tensor operand.
    pub key: T,
    /// Value tensor operand.
    pub value: T,
    /// Forward attention result tensor.
    pub output: T,
    /// Scalar attention-logit multiplier tensor.
    pub scale: T,
}

impl<T> SdpaForwardTensors<T> {
    /// Borrows the required tensor owners without changing their native operation roles.
    pub fn as_ref(&self) -> SdpaForwardTensors<&T> {
        SdpaForwardTensors {
            query: &self.query,
            key: &self.key,
            value: &self.value,
            output: &self.output,
            scale: &self.scale,
        }
    }

    /// Maps the complete required tensor group without changing operand roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> SdpaForwardTensors<U> {
        match self.try_map::<_, Infallible>(|tensor| Ok(map(tensor))) {
            Ok(tensors) => tensors,
            Err(never) => match never {},
        }
    }

    /// Maps tensor operands in declaration order while preserving their roles and optional omissions.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied mapping without mapping later operands.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<SdpaForwardTensors<U>, E> {
        Ok(SdpaForwardTensors {
            query: map(self.query)?,
            key: map(self.key)?,
            value: map(self.value)?,
            output: map(self.output)?,
            scale: map(self.scale)?,
        })
    }
}

/// Finalized native SDPA forward operation with retained tensor identities and descriptor settings.
#[derive(Debug)]
pub struct SdpaForwardOperation {
    descriptor: BackendDescriptor,
    metadata: SdpaForwardOperationMetadata,
}

/// Native Philox dropout operands, with an optional caller-visible generated mask.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdpaForwardDropout<T> {
    /// Native dropout probability; cuDNN validates its supported range.
    pub probability: f32,
    /// Philox random-number-generator seed tensor.
    pub seed: T,
    /// Philox random-number-generator offset tensor.
    pub offset: T,
    /// Optional destination for random numbers generated by dropout.
    pub random_number_generator_dump: Option<T>,
}

impl<T> SdpaForwardDropout<T> {
    /// Borrows seed, offset and optional mask owners while retaining the probability.
    pub fn as_ref(&self) -> SdpaForwardDropout<&T> {
        SdpaForwardDropout {
            probability: self.probability,
            seed: &self.seed,
            offset: &self.offset,
            random_number_generator_dump: self.random_number_generator_dump.as_ref(),
        }
    }

    /// Maps dropout operands without changing probability or mask-output omission.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> SdpaForwardDropout<U> {
        SdpaForwardDropout {
            probability: self.probability,
            seed: map(self.seed),
            offset: map(self.offset),
            random_number_generator_dump: self.random_number_generator_dump.map(map),
        }
    }
}

/// Native attention softmax selection and its associated saved-statistics request.
#[derive(Debug, Clone)]
pub enum SdpaForwardSoftmax<T> {
    /// Uses native default softmax with optional FP32 saved statistics.
    Default {
        /// Optional FP32 saved softmax log-sum-exp statistics tensor.
        stats: Option<T>,
    },
    /// Attaches an explicit native softmax stage, including all its requested outputs.
    Descriptor(Arc<SoftmaxOperation>),
}

impl<T> Default for SdpaForwardSoftmax<T> {
    fn default() -> Self {
        Self::Default { stats: None }
    }
}

impl<T> SdpaForwardSoftmax<T> {
    /// Maps direct statistics while preserving an explicit softmax stage and its owner.
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> SdpaForwardSoftmax<U> {
        match self {
            Self::Default { stats } => SdpaForwardSoftmax::Default {
                stats: stats.map(map),
            },
            Self::Descriptor(softmax) => SdpaForwardSoftmax::Descriptor(softmax),
        }
    }
}

/// Optional native forward attention operands, softmax selection, and execution attributes.
#[derive(Debug, Clone, Default)]
pub struct SdpaForwardConfig<'a> {
    /// Independent query and key/value sequence-length operands.
    pub sequence: AttentionSequenceInputs<&'a Tensor>,
    /// Optional page table for the key cache.
    pub page_table_k: Option<&'a Tensor>,
    /// Optional page table for the value cache.
    pub page_table_v: Option<&'a Tensor>,
    /// Optional block-mask tensor.
    pub block_mask: Option<&'a Tensor>,
    /// Native default softmax with optional statistics, or a retained explicit softmax descriptor.
    pub softmax: SdpaForwardSoftmax<&'a Tensor>,
    /// Optional score-modifier graph and its input/output tensor endpoints.
    pub subgraph: Option<SdpaSubgraph<'a>>,
    /// Optional native Philox dropout settings and tensor operands.
    pub dropout: Option<SdpaForwardDropout<&'a Tensor>>,
    /// Requests native unfused multiply-add operations when true.
    pub unfuse_fma: bool,
    /// Optional scaling operands and pre-quantization absolute-maximum outputs.
    pub quantization: Option<SdpaForwardQuantization<&'a Tensor>>,
}

/// Retained native operands and optional forward attention settings.
/// An explicit softmax stage retains its descriptor owner and exposes its own outputs and configuration.
#[derive(Debug, Clone)]
pub struct SdpaForwardOperationMetadata {
    /// Required attention activation and output tensor identities.
    pub tensors: SdpaForwardTensors<BackendTensorUid>,
    /// Independent query and key/value sequence-length operands.
    pub sequence: AttentionSequenceInputs<BackendTensorUid>,
    /// Optional page table for the key cache.
    pub page_table_k: Option<BackendTensorUid>,
    /// Optional page table for the value cache.
    pub page_table_v: Option<BackendTensorUid>,
    /// Optional block-mask tensor.
    pub block_mask: Option<BackendTensorUid>,
    /// Native default softmax with optional statistics, or a retained explicit softmax descriptor.
    pub softmax: SdpaForwardSoftmax<BackendTensorUid>,
    /// Optional native identities of the score-graph input and output ports.
    pub subgraph: Option<SdpaSubgraphPorts>,
    /// Optional native Philox dropout settings and tensor operands.
    pub dropout: Option<SdpaForwardDropout<BackendTensorUid>>,
    /// Requests native unfused multiply-add operations when true.
    pub unfuse_fma: bool,
    /// Optional scaling operands and pre-quantization absolute-maximum outputs.
    pub quantization: Option<SdpaForwardQuantization<BackendTensorUid>>,
}

impl SdpaForwardOperation {
    /// Creates a native SDPA descriptor with FP32 statistics storage.
    /// Required operands are named, and softmax selection owns its complete statistics request.
    ///
    /// # Errors
    /// Returns an error for unpaired sequence lengths, non-FP32 statistics, including statistics attached through a softmax descriptor, or native descriptor failures.
    #[expect(
        clippy::too_many_lines,
        reason = "the composite attention operation builds and validates its complete tensor graph"
    )]
    pub fn create(
        tensors: SdpaForwardTensors<&Tensor>,
        config: SdpaForwardConfig<'_>,
    ) -> Result<Self> {
        config.sequence.validate_forward()?;
        let SdpaForwardTensors {
            query: q,
            key: k,
            value: v,
            output: o,
            scale,
        } = tensors;
        let stats_data_type = match &config.softmax {
            SdpaForwardSoftmax::Descriptor(softmax) => softmax.stats_data_type,
            SdpaForwardSoftmax::Default { stats } => stats.map(Tensor::data_type),
        };
        if let Some(actual) = stats_data_type
            && actual != DataType::F32
        {
            return Err(Error::SdpaStatsDataType { actual });
        }
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationSdpaForward)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaFwdQDesc,
            q.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaFwdKDesc,
            k.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaFwdVDesc,
            v.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaFwdODesc,
            o.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaFwdScaleDesc,
            scale.descriptor(),
        )?;
        match &config.softmax {
            SdpaForwardSoftmax::Descriptor(softmax) => descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdSoftmaxDesc,
                softmax.descriptor(),
            )?,
            SdpaForwardSoftmax::Default { stats: Some(stats) } => descriptor
                .set_attribute_descriptor(
                    BackendAttributeName::OperationSdpaFwdStatsDesc,
                    stats.descriptor(),
                )?,
            SdpaForwardSoftmax::Default { stats: None } => {}
        }
        if let Some(lengths) = config.sequence.query {
            let attribute = match lengths {
                SequenceLength::PerBatch(_) => BackendAttributeName::OperationSdpaFwdSeqLenQDesc,
                SequenceLength::Cumulative(_) => {
                    BackendAttributeName::OperationSdpaFwdCuSeqLenQDesc
                }
            };
            descriptor.set_attribute_descriptor(attribute, lengths.tensor().descriptor())?;
        }
        if let Some(lengths) = config.sequence.key_value {
            let attribute = match lengths {
                SequenceLength::PerBatch(_) => BackendAttributeName::OperationSdpaFwdSeqLenKvDesc,
                SequenceLength::Cumulative(_) => {
                    BackendAttributeName::OperationSdpaFwdCuSeqLenKvDesc
                }
            };
            descriptor.set_attribute_descriptor(attribute, lengths.tensor().descriptor())?;
        }
        if let Some(page_table_k) = config.page_table_k {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdPageTableKDesc,
                page_table_k.descriptor(),
            )?;
        }
        if let Some(page_table_v) = config.page_table_v {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdPageTableVDesc,
                page_table_v.descriptor(),
            )?;
        }
        if let Some(block_mask) = config.block_mask {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdBlockMaskDesc,
                block_mask.descriptor(),
            )?;
        }
        if let Some(subgraph) = config.subgraph {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdSubgraph,
                subgraph.operation_graph.descriptor(),
            )?;
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationSdpaFwdSubgraphInputId,
                subgraph.input_id.as_i64(),
            )?;
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationSdpaFwdSubgraphOutputId,
                subgraph.output_id.as_i64(),
            )?;
        }
        if let Some(dropout) = config.dropout {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdDropoutSeedDesc,
                dropout.seed.descriptor(),
            )?;
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdDropoutOffsetDesc,
                dropout.offset.descriptor(),
            )?;
            if let Some(dump) = dropout.random_number_generator_dump {
                descriptor.set_attribute_descriptor(
                    BackendAttributeName::OperationSdpaFwdDropoutRngDumpDesc,
                    dump.descriptor(),
                )?;
            }
            descriptor.set_attribute_f32(
                BackendAttributeName::OperationSdpaFwdDropoutProbability,
                dropout.probability,
            )?;
        }
        if config.unfuse_fma {
            descriptor.set_attribute_bool(BackendAttributeName::OperationSdpaFwdUnfuseFma, true)?;
        }
        if let Some(quantization) = config.quantization {
            for (tensor, attribute) in [
                (
                    quantization.descale_query,
                    BackendAttributeName::OperationSdpaFwdDescaleQDesc,
                ),
                (
                    quantization.descale_key,
                    BackendAttributeName::OperationSdpaFwdDescaleKDesc,
                ),
                (
                    quantization.descale_value,
                    BackendAttributeName::OperationSdpaFwdDescaleVDesc,
                ),
                (
                    quantization.descale_scores,
                    BackendAttributeName::OperationSdpaFwdDescaleSDesc,
                ),
                (
                    quantization.scale_scores,
                    BackendAttributeName::OperationSdpaFwdScaleSDesc,
                ),
                (
                    quantization.scale_output,
                    BackendAttributeName::OperationSdpaFwdScaleODesc,
                ),
                (
                    quantization.absolute_max_probabilities,
                    BackendAttributeName::OperationSdpaFwdAmaxSDesc,
                ),
                (
                    quantization.absolute_max_output,
                    BackendAttributeName::OperationSdpaFwdAmaxODesc,
                ),
            ] {
                if let Some(tensor) = tensor {
                    descriptor.set_attribute_descriptor(attribute, tensor.descriptor())?;
                }
            }
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: SdpaForwardOperationMetadata {
                tensors: tensors.map(Tensor::id),
                sequence: config.sequence.map(Tensor::id),
                page_table_k: config.page_table_k.map(Tensor::id),
                page_table_v: config.page_table_v.map(Tensor::id),
                block_mask: config.block_mask.map(Tensor::id),
                softmax: config.softmax.map(Tensor::id),
                subgraph: config.subgraph.map(|subgraph| subgraph.ports()),
                dropout: config.dropout.map(|dropout| dropout.map(Tensor::id)),
                unfuse_fma: config.unfuse_fma,
                quantization: config
                    .quantization
                    .map(|quantization| quantization.map(Tensor::id)),
            },
        })
    }

    /// Borrows finalized operands and settings without querying cuDNN or allocating.
    /// Tensor identities and score-graph endpoints do not borrow the original wrappers.
    #[must_use]
    pub fn metadata(&self) -> &SdpaForwardOperationMetadata {
        &self.metadata
    }

    /// Retained native operand UIDs without querying descriptors or translating frontend identities.
    #[must_use]
    pub fn tensor_uids(&self) -> &SdpaForwardTensors<BackendTensorUid> {
        &self.metadata.tensors
    }

    /// Borrows the finalized native attention operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
