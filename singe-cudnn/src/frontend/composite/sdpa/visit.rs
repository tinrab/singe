use crate::{
    error::Result,
    frontend::{
        composite::sdpa::{
            SdpaBackwardGradientTensors, SdpaBackwardInputs, SdpaFp8BackwardGradientTensors,
            SdpaFp8BackwardInputs, SdpaFp8Inputs, SdpaInputs, SdpaMxfp8BackwardGradientTensors,
            SdpaMxfp8BackwardInputs, SdpaMxfp8Inputs,
        },
        operation::{TensorRole, TensorVisitor},
    },
};

macro_rules! impl_tensor_visit {
    ($name:ident { required: [$($required:ident),* $(,)?], optional: [$($optional:ident),* $(,)?] $(,)? }) => {
        impl $name {
            /// Visits every tensor with the same role.
            pub(crate) fn visit_tensor_ids(&mut self, role: TensorRole, visit: &mut TensorVisitor<'_>) -> Result<()> {
                $(visit(&mut self.$required, role)?;)*
                $(if let Some(id) = &mut self.$optional {
                    visit(id, role)?;
                })*
                Ok(())
            }
        }
    };
}

impl_tensor_visit!(SdpaInputs {
    required: [query, key, value, scale],
    optional: [],
});

impl_tensor_visit!(SdpaFp8Inputs {
    required: [
        query,
        key,
        value,
        attention_scale,
        descale_query,
        descale_key,
        descale_value,
        descale_scores,
        scale_scores,
        scale_output
    ],
    optional: [],
});

impl_tensor_visit!(SdpaMxfp8Inputs {
    required: [
        query,
        key,
        value,
        scale_query,
        scale_key,
        scale_value,
        attention_scale
    ],
    optional: [],
});

impl_tensor_visit!(SdpaBackwardInputs {
    required: [query, key, value, output, output_gradient, stats, scale],
    optional: [],
});

impl_tensor_visit!(SdpaFp8BackwardInputs {
    required: [
        query,
        key,
        value,
        output,
        output_gradient,
        stats,
        attention_scale,
        descale_query,
        descale_key,
        descale_value,
        descale_output,
        descale_output_gradient,
        descale_scores,
        descale_probability_gradient,
        scale_scores,
        scale_query_gradient,
        scale_key_gradient,
        scale_value_gradient,
        scale_probability_gradient
    ],
    optional: [],
});

impl_tensor_visit!(SdpaMxfp8BackwardInputs {
    required: [
        query,
        sequence_scaled_query,
        key,
        sequence_scaled_key,
        value,
        output,
        output_gradient,
        quantized_output_gradient,
        sequence_scaled_output_gradient,
        stats,
        scale_query,
        sequence_scale_query,
        scale_key,
        sequence_scale_key,
        scale_value,
        scale_output_gradient,
        sequence_scale_output_gradient,
        attention_scale
    ],
    optional: [],
});

impl_tensor_visit!(SdpaBackwardGradientTensors {
    required: [query_gradient, key_gradient, value_gradient],
    optional: [bias_gradient, rng_dump, sink_token_gradient],
});

impl_tensor_visit!(SdpaFp8BackwardGradientTensors {
    required: [
        query_gradient,
        key_gradient,
        value_gradient,
        absolute_max_query_gradient,
        absolute_max_key_gradient,
        absolute_max_value_gradient,
        absolute_max_probability_gradient
    ],
    optional: [sink_token_gradient],
});

impl_tensor_visit!(SdpaMxfp8BackwardGradientTensors {
    required: [query_gradient, key_gradient, value_gradient],
    optional: [
        sink_token_gradient,
        absolute_max_query_gradient,
        absolute_max_key_gradient,
        absolute_max_value_gradient
    ],
});
