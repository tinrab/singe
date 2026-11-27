//! Reads a `transformers` `EfficientLoFTR` config and imports its checkpoint, whose names Singe keeps below the model's prefix.

use serde::Deserialize;

use crate::{
    error::{Error, Result},
    format::Rules,
    model::vision::efficientloftr::{EfficientLoftrConfig, RepVggBlock},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    stage_block_in_channels: Vec<Vec<i64>>,
    stage_block_out_channels: Vec<Vec<i64>>,
    stage_block_stride: Vec<Vec<i64>>,
    hidden_size: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    num_attention_layers: i64,
    intermediate_size: i64,
    q_aggregation_kernel_size: i64,
    q_aggregation_stride: i64,
    kv_aggregation_kernel_size: i64,
    kv_aggregation_stride: i64,
    rope_theta: f64,
    partial_rotary_factor: f64,
    attention_bias: bool,
    activation_function: String,
    mlp_activation_function: String,
    coarse_matching_temperature: f32,
    coarse_matching_threshold: f32,
    coarse_matching_border_removal: i64,
    coarse_matching_skip_softmax: bool,
    fine_fusion_dims: Vec<i64>,
    fine_kernel_size: i64,
    fine_matching_slice_dim: i64,
    fine_matching_regress_temperature: f32,
}

pub(crate) fn config(source: &Source) -> Result<EfficientLoftrConfig> {
    let fields: Fields = source.json("config.json")?;
    let aggregation = fields.q_aggregation_kernel_size;
    let fits = [
        fields.q_aggregation_stride,
        fields.kv_aggregation_kernel_size,
        fields.kv_aggregation_stride,
    ]
    .iter()
    .all(|&size| size == aggregation)
        && fields.num_key_value_heads == fields.num_attention_heads
        && !fields.attention_bias
        && fields.activation_function == "relu"
        && fields.mlp_activation_function == "leaky_relu"
        && !fields.coarse_matching_skip_softmax;
    if !fits {
        return Err(Error::unsupported(
            "an EfficientLoFTR other than one aggregation size, full heads without biases, ReLU blocks, leaky ReLU MLPs, and a dual softmax",
        ));
    }

    let stages = fields
        .stage_block_in_channels
        .iter()
        .zip(&fields.stage_block_out_channels)
        .zip(&fields.stage_block_stride)
        .map(|((inputs, outputs), strides)| {
            inputs
                .iter()
                .zip(outputs)
                .zip(strides)
                .map(|((&input, &output), &stride)| RepVggBlock {
                    input,
                    output,
                    stride,
                })
                .collect()
        })
        .collect();
    Ok(EfficientLoftrConfig {
        stages,
        width: fields.hidden_size,
        heads: fields.num_attention_heads,
        layers: fields.num_attention_layers,
        inner: fields.intermediate_size,
        aggregation,
        theta: fields.rope_theta,
        rotary_factor: fields.partial_rotary_factor,
        temperature: fields.coarse_matching_temperature,
        threshold: fields.coarse_matching_threshold,
        border: fields.coarse_matching_border_removal,
        fine_dims: fields.fine_fusion_dims,
        fine_kernel: fields.fine_kernel_size,
        slice: fields.fine_matching_slice_dim,
        regress_temperature: fields.fine_matching_regress_temperature,
    })
}

pub(crate) fn rules() -> Rules {
    Rules::new()
        // A batch norm's count of the batches it saw in training, which evaluation never reads.
        .skip("efficientloftr.backbone.stages.*.blocks.*.conv1.norm.num_batches_tracked")
        .skip("efficientloftr.backbone.stages.*.blocks.*.conv2.norm.num_batches_tracked")
        .skip("efficientloftr.backbone.stages.*.blocks.*.identity.num_batches_tracked")
        .skip("refinement_layer.out_conv_layers.*.batch_norm.num_batches_tracked")
        .rename("efficientloftr.backbone", "backbone")
        .rename("efficientloftr.local_feature_transformer", "transformer")
        .rename("refinement_layer", "refinement")
}
