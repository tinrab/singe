mod sealed {
    use crate::{
        error::Result,
        frontend::{
            composite::sdpa::SdpaBackwardOperands,
            graph::Graph,
            operation::{AttentionBackwardConfig, AttentionBackwardRequest},
        },
    };

    #[expect(
        unnameable_types,
        reason = "seals `SdpaBackwardOperands`, so callers cannot name or implement it"
    )]
    pub trait Sealed: Sized {
        fn infer(
            self,
            graph: &mut Graph,
            config: AttentionBackwardConfig,
            outputs: <Self as SdpaBackwardOperands>::OutputConfig,
        ) -> Result<<Self as SdpaBackwardOperands>::Gradients>
        where
            Self: SdpaBackwardOperands;

        fn request(
            self,
            gradients: <Self as SdpaBackwardOperands>::Gradients,
            config: AttentionBackwardConfig,
        ) -> AttentionBackwardRequest
        where
            Self: SdpaBackwardOperands;
    }
}

use crate::{
    error::Result,
    frontend::{
        composite::sdpa::{
            SdpaBackwardGradientTensors, SdpaBackwardInputs, SdpaFp8BackwardGradientTensors,
            SdpaFp8BackwardInputs, SdpaFp8BackwardOutputConfig, SdpaGradientOutputConfig,
            SdpaMxfp8BackwardGradientTensors, SdpaMxfp8BackwardInputs,
            SdpaMxfp8BackwardOutputConfig,
        },
        graph::Graph,
        operation::{AttentionBackwardConfig, AttentionBackwardRequest},
    },
};

/// Saved forward operands whose type selects the precision of [`Graph::sdpa_backward`].
///
/// [`SdpaBackwardInputs`] selects floating attention, [`SdpaFp8BackwardInputs`] per-tensor FP8 attention, and [`SdpaMxfp8BackwardInputs`] block-scaled MXFP8 attention.
pub trait SdpaBackwardOperands: sealed::Sealed {
    /// Placement and encoding requests for the gradients and auxiliary outputs.
    type OutputConfig;

    /// Gradient and auxiliary-output tensors of the backward operation.
    type Gradients;
}

impl SdpaBackwardOperands for SdpaBackwardInputs {
    type OutputConfig = SdpaGradientOutputConfig;

    type Gradients = SdpaBackwardGradientTensors;
}

impl sealed::Sealed for SdpaBackwardInputs {
    fn infer(
        self,
        graph: &mut Graph,
        config: AttentionBackwardConfig,
        outputs: SdpaGradientOutputConfig,
    ) -> Result<SdpaBackwardGradientTensors> {
        graph.infer_floating_sdpa_backward(self, config, outputs)
    }

    fn request(
        self,
        gradients: SdpaBackwardGradientTensors,
        config: AttentionBackwardConfig,
    ) -> AttentionBackwardRequest {
        AttentionBackwardRequest::Floating {
            inputs: self,
            outputs: gradients,
            config,
        }
    }
}

impl SdpaBackwardOperands for SdpaFp8BackwardInputs {
    type OutputConfig = SdpaFp8BackwardOutputConfig;

    type Gradients = SdpaFp8BackwardGradientTensors;
}

impl sealed::Sealed for SdpaFp8BackwardInputs {
    fn infer(
        self,
        graph: &mut Graph,
        config: AttentionBackwardConfig,
        outputs: SdpaFp8BackwardOutputConfig,
    ) -> Result<SdpaFp8BackwardGradientTensors> {
        graph.infer_fp8_sdpa_backward(self, config, outputs)
    }

    fn request(
        self,
        gradients: SdpaFp8BackwardGradientTensors,
        config: AttentionBackwardConfig,
    ) -> AttentionBackwardRequest {
        AttentionBackwardRequest::Fp8 {
            inputs: self,
            outputs: gradients,
            config,
        }
    }
}

impl SdpaBackwardOperands for SdpaMxfp8BackwardInputs {
    type OutputConfig = SdpaMxfp8BackwardOutputConfig;

    type Gradients = SdpaMxfp8BackwardGradientTensors;
}

impl sealed::Sealed for SdpaMxfp8BackwardInputs {
    fn infer(
        self,
        graph: &mut Graph,
        config: AttentionBackwardConfig,
        outputs: SdpaMxfp8BackwardOutputConfig,
    ) -> Result<SdpaMxfp8BackwardGradientTensors> {
        graph.infer_mxfp8_sdpa_backward(self, config, outputs)
    }

    fn request(
        self,
        gradients: SdpaMxfp8BackwardGradientTensors,
        config: AttentionBackwardConfig,
    ) -> AttentionBackwardRequest {
        AttentionBackwardRequest::Mxfp8 {
            inputs: self,
            outputs: gradients,
            config,
        }
    }
}

impl Graph {
    /// Creates attention gradients and independently requested auxiliary outputs.
    /// The operand type selects floating, FP8 or MXFP8 attention, and with it the output configuration and gradient types.
    /// Each gradient selects its placement and encoding independently.
    /// Floating bias gradients follow the bias layout, or the score shape when no bias is supplied, and default to FP32 like RNG dumps.
    /// Quantized attention supports neither bias gradients nor RNG dumps, and each requested Amax describes its gradient before the output scale and conversion.
    /// Expansion occurs during graph preparation, after resolving the target architecture.
    ///
    /// # Errors
    ///
    /// Returns an error if a query, key, value, or bias reference is unknown, an attention input is not four-dimensional, the query/key or value geometry is incompatible, a bias gradient or RNG dump is requested for quantized attention, or an inferred score, statistic, or auxiliary-output layout is unrepresentable.
    pub fn sdpa_backward<I: SdpaBackwardOperands>(
        &mut self,
        inputs: I,
        config: AttentionBackwardConfig,
        outputs: I::OutputConfig,
    ) -> Result<I::Gradients> {
        sealed::Sealed::infer(inputs, self, config, outputs)
    }

    /// Records attention backward with caller-declared gradients, auxiliary outputs and layouts.
    /// The operand type selects floating, FP8 or MXFP8 attention.
    /// Preparation resolves support and engine selection for the compilation target.
    ///
    /// # Errors
    ///
    /// This recording step currently returns no errors.
    /// Graph preparation validates the deferred operands, layouts, and target-specific support.
    pub fn sdpa_backward_into<I: SdpaBackwardOperands>(
        &mut self,
        inputs: I,
        gradients: I::Gradients,
        config: AttentionBackwardConfig,
    ) -> Result<()> {
        let request = sealed::Sealed::request(inputs, gradients, config);
        self.record_attention_backward_request(request);
        Ok(())
    }
}
