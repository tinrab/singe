use std::{convert::Infallible, result::Result as StdResult};

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        execution::attention::{
            AttentionSequenceInputs, SdpaSubgraph, SdpaSubgraphPorts, SequenceLength,
        },
        tensor::{BackendTensorUid, Tensor},
    },
    error::Result,
    frontend::error::{Error as FrontendError, SdpaFeature},
};

/// Required inputs and gradient outputs of a native attention backward operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaBackwardTensors<T> {
    /// Query tensor operand.
    pub query: T,
    /// Key tensor operand.
    pub key: T,
    /// Value tensor operand.
    pub value: T,
    /// Saved forward attention output tensor.
    pub output: T,
    /// Saved forward softmax log-sum-exp statistics tensor.
    pub stats: T,
    /// Scalar attention-logit multiplier tensor.
    pub scale: T,
    /// Gradient with respect to the forward output.
    pub output_gradient: T,
    /// Gradient with respect to the query tensor.
    pub query_gradient: T,
    /// Gradient with respect to the key tensor.
    pub key_gradient: T,
    /// Gradient with respect to the value tensor.
    pub value_gradient: T,
}

impl<T> SdpaBackwardTensors<T> {
    /// Borrows tensor owners without changing their activation or gradient roles.
    pub fn as_ref(&self) -> SdpaBackwardTensors<&T> {
        SdpaBackwardTensors {
            query: &self.query,
            key: &self.key,
            value: &self.value,
            output: &self.output,
            stats: &self.stats,
            scale: &self.scale,
            output_gradient: &self.output_gradient,
            query_gradient: &self.query_gradient,
            key_gradient: &self.key_gradient,
            value_gradient: &self.value_gradient,
        }
    }

    /// Maps every required tensor without changing its native operation role.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> SdpaBackwardTensors<U> {
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
    ) -> StdResult<SdpaBackwardTensors<U>, E> {
        Ok(SdpaBackwardTensors {
            query: map(self.query)?,
            key: map(self.key)?,
            value: map(self.value)?,
            output: map(self.output)?,
            stats: map(self.stats)?,
            scale: map(self.scale)?,
            output_gradient: map(self.output_gradient)?,
            query_gradient: map(self.query_gradient)?,
            key_gradient: map(self.key_gradient)?,
            value_gradient: map(self.value_gradient)?,
        })
    }
}

impl<T: Copy> SdpaBackwardTensors<T> {
    /// Activation and gradient tensors requiring attention's vector alignment.
    pub(crate) fn activations_and_gradients(&self) -> [T; 8] {
        [
            self.query,
            self.key,
            self.value,
            self.output,
            self.output_gradient,
            self.query_gradient,
            self.key_gradient,
            self.value_gradient,
        ]
    }
}

/// Sink logits with an independently optional gradient destination.
/// A sink gradient always has its corresponding forward token operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaBackwardSink<T> {
    /// Forward attention sink-logit tensor.
    pub token: T,
    /// Optional sink-gradient destination.
    pub gradient: Option<T>,
}

impl<T> SdpaBackwardSink<T> {
    /// Records sink logits without requesting a sink-gradient output.
    pub fn new(token: T) -> Self {
        Self {
            token,
            gradient: None,
        }
    }

    /// Selects or clears gradient output while retaining the token operand.
    #[must_use]
    pub fn with_gradient(mut self, gradient: impl Into<Option<T>>) -> Self {
        self.gradient = gradient.into();
        self
    }

    /// Borrows the token owner and only the gradient owner that was supplied.
    pub fn as_ref(&self) -> SdpaBackwardSink<&T> {
        SdpaBackwardSink {
            token: &self.token,
            gradient: self.gradient.as_ref(),
        }
    }

    /// Maps the token and optional gradient while preserving an omitted destination.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> SdpaBackwardSink<U> {
        SdpaBackwardSink {
            token: map(self.token),
            gradient: self.gradient.map(map),
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
    ) -> StdResult<SdpaBackwardSink<U>, E> {
        Ok(SdpaBackwardSink {
            token: map(self.token)?,
            gradient: self.gradient.map(map).transpose()?,
        })
    }
}

/// Finalized native SDPA backward operation with retained activation and gradient identities.
#[derive(Debug)]
pub struct SdpaBackwardOperation {
    descriptor: BackendDescriptor,
    metadata: SdpaBackwardOperationMetadata,
}

/// Optional native backward sequence lengths, score subgraph, sinks, and packed-token bounds.
#[derive(Debug, Clone, Copy, Default)]
pub struct SdpaBackwardConfig<'a> {
    /// Independent query and key/value sequence-length operands.
    pub sequence: AttentionSequenceInputs<&'a Tensor>,
    /// Optional score-modifier graph and its input/output tensor endpoints.
    pub subgraph: Option<SdpaSubgraph<'a>>,
    /// Optional sink logits and independently optional sink-gradient destination.
    pub sink: Option<SdpaBackwardSink<&'a Tensor>>,
    /// Optional native bound on total packed query tokens across batches.
    pub max_total_sequence_length_query: Option<i64>,
    /// Optional native bound on total packed key/value tokens across batches.
    pub max_total_sequence_length_key_value: Option<i64>,
}

/// Retained native operands and optional backward attention settings.
/// Score-graph metadata identifies its endpoints, independently of the original graph wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaBackwardOperationMetadata {
    /// Required saved activations, scale, and gradient tensor identities.
    pub tensors: SdpaBackwardTensors<BackendTensorUid>,
    /// Independent query and key/value sequence-length operands.
    pub sequence: AttentionSequenceInputs<BackendTensorUid>,
    /// Optional native identities of the score-graph input and output ports.
    pub subgraph: Option<SdpaSubgraphPorts>,
    /// Optional sink logits and independently optional sink-gradient destination.
    pub sink: Option<SdpaBackwardSink<BackendTensorUid>>,
    /// Optional native bound on total packed query tokens across batches.
    pub max_total_sequence_length_query: Option<i64>,
    /// Optional native bound on total packed key/value tokens across batches.
    pub max_total_sequence_length_key_value: Option<i64>,
}

impl SdpaBackwardOperation {
    /// Creates native attention backward from named operands and optional sequence, score and sink settings.
    /// A sink gradient belongs to its token request and cannot be supplied independently.
    ///
    /// # Errors
    /// Returns an error for unsupported cumulative lengths or rejected native descriptor attributes.
    #[expect(
        clippy::too_many_lines,
        reason = "the composite attention operation builds and validates its complete tensor graph"
    )]
    pub fn create(
        tensors: SdpaBackwardTensors<&Tensor>,
        config: SdpaBackwardConfig<'_>,
    ) -> Result<Self> {
        let SdpaBackwardTensors {
            query: q,
            key: k,
            value: v,
            output: o,
            stats,
            scale,
            output_gradient: d_o,
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
        } = tensors;
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationSdpaBackward)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdQDesc,
            q.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdKDesc,
            k.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdVDesc,
            v.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdODesc,
            o.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdStatsDesc,
            stats.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdScaleDesc,
            scale.descriptor(),
        )?;
        if let Some(lengths) = config.sequence.query {
            let SequenceLength::PerBatch(tensor) = lengths else {
                return Err(FrontendError::SdpaUnsupported {
                    operation: "sdpa backward",
                    feature: SdpaFeature::CumulativeSequenceLengths,
                    sm_version: None,
                }
                .into());
            };
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaBwdSeqLenQDesc,
                tensor.descriptor(),
            )?;
        }
        if let Some(lengths) = config.sequence.key_value {
            let SequenceLength::PerBatch(tensor) = lengths else {
                return Err(FrontendError::SdpaUnsupported {
                    operation: "sdpa backward",
                    feature: SdpaFeature::CumulativeSequenceLengths,
                    sm_version: None,
                }
                .into());
            };
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaBwdSeqLenKvDesc,
                tensor.descriptor(),
            )?;
        }
        if let Some(subgraph) = config.subgraph {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaBwdSubgraph,
                subgraph.operation_graph.descriptor(),
            )?;
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationSdpaBwdSubgraphInputId,
                subgraph.input_id.as_i64(),
            )?;
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationSdpaBwdSubgraphOutputId,
                subgraph.output_id.as_i64(),
            )?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdDqDesc,
            d_q.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdDkDesc,
            d_k.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdDvDesc,
            d_v.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdDoDesc,
            d_o.descriptor(),
        )?;
        if let Some(sink) = config.sink {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaBwdSinkDesc,
                sink.token.descriptor(),
            )?;
            if let Some(gradient) = sink.gradient {
                descriptor.set_attribute_descriptor(
                    BackendAttributeName::OperationSdpaBwdDsinkDesc,
                    gradient.descriptor(),
                )?;
            }
        }
        if let Some(max_total_sequence_length_query) = config.max_total_sequence_length_query {
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationSdpaBwdMaxTotalSeqLenQ,
                max_total_sequence_length_query,
            )?;
        }
        if let Some(max_total_sequence_length_key_value) =
            config.max_total_sequence_length_key_value
        {
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationSdpaBwdMaxTotalSeqLenKv,
                max_total_sequence_length_key_value,
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: SdpaBackwardOperationMetadata {
                tensors: tensors.map(Tensor::id),
                sequence: config.sequence.map(Tensor::id),
                subgraph: config.subgraph.map(|subgraph| subgraph.ports()),
                sink: config.sink.map(|sink| sink.map(Tensor::id)),
                max_total_sequence_length_query: config.max_total_sequence_length_query,
                max_total_sequence_length_key_value: config.max_total_sequence_length_key_value,
            },
        })
    }

    /// Borrows finalized operands and settings without querying cuDNN or allocating.
    /// Remains available after the original tensor and score-graph wrappers are released.
    #[must_use]
    pub fn metadata(&self) -> &SdpaBackwardOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native attention operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
