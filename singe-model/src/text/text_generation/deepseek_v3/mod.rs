mod config;

use std::path::Path;

use singe::{
    DataType, Device, Result as SingeResult,
    nn::{
        CheckpointOptions, Decoder, DecoderFeedForward, GatedMlp, LatentKvCacheSpec, LatentQuery,
        MixtureOfExperts, Model, MoeLayers, MoeRouting, MultiHeadLatentAttention,
        MultiHeadLatentLayers, StatePiece,
    },
    tensor::AttentionOptions,
};

pub use self::config::DeepseekV3Config;
use crate::{
    assembly::{DecoderSpec, causal, causal_lm, gated_mlp, pre_norm_block_with},
    config::Snapshot,
    error::Result,
    schema::Schema,
};

// DeepSeek-V3's layers: multi-head latent attention, and a feed-forward layer that is dense in the first few and mixture-of-experts after them.
pub type DeepseekV3Layers = Decoder<MultiHeadLatentAttention, DecoderFeedForward>;

pub type DeepseekV3 = Model<DeepseekV3Layers>;

fn attention(
    schema: &mut Schema<'_>,
    config: &DeepseekV3Config,
    score_correction: f64,
) -> SingeResult<MultiHeadLatentAttention> {
    let width = config.decoder.hidden_size;
    let heads = config.decoder.num_attention_heads;
    let eps = config.decoder.rms_norm_eps;
    let query_width = heads * (config.qk_nope_head_dim + config.qk_rope_head_dim);
    let query = match config.q_lora_rank {
        Some(rank) => LatentQuery::LowRank {
            compress: schema.linear("q_a_proj", width, rank, false)?,
            norm: schema.rms_norm("q_a_layernorm", rank, eps)?,
            expand: schema.linear("q_b_proj", rank, query_width, false)?,
        },
        None => LatentQuery::Direct(schema.linear("q_proj", width, query_width, false)?),
    };
    let compress = schema.linear(
        "kv_a_proj_with_mqa",
        width,
        config.kv_lora_rank + config.qk_rope_head_dim,
        false,
    )?;
    let latent_norm = schema.rms_norm("kv_a_layernorm", config.kv_lora_rank, eps)?;
    let expand = schema.parameter(
        "kv_b_proj.weight",
        [
            heads * (config.qk_nope_head_dim + config.v_head_dim),
            config.kv_lora_rank,
        ],
    )?;
    let output = schema.linear("o_proj", heads * config.v_head_dim, width, false)?;
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "the scale is defined in F32, as the references compute it"
    )]
    let scale = (score_correction
        / ((config.qk_nope_head_dim + config.qk_rope_head_dim) as f64).sqrt())
        as f32;
    MultiHeadLatentAttention::create(
        MultiHeadLatentLayers {
            query,
            compress,
            latent_norm,
            expand,
            output,
            heads,
            key_dim: config.qk_nope_head_dim,
            rotary_dim: config.qk_rope_head_dim,
            value_dim: config.v_head_dim,
            latent_dim: config.kv_lora_rank,
        },
        AttentionOptions {
            scale: Some(scale),
            ..causal()
        },
    )
}

// Registers `mlp.gate`, `mlp.experts`, and `mlp.shared_experts`, as `transformers` holds them.
fn mixture_of_experts(
    schema: &mut Schema<'_>,
    config: &DeepseekV3Config,
    spec: &DecoderSpec,
) -> SingeResult<MixtureOfExperts> {
    let (width, inner) = (spec.width, config.moe_intermediate_size);
    let experts = config.n_routed_experts;
    let mut mlp = schema.scope("mlp");
    let router = mlp.parameter("gate.weight", [experts, width])?;
    let router_bias = mlp.parameter("gate.e_score_correction_bias", [experts])?;
    let gate_up = mlp.parameter("experts.gate_up_proj", [experts, 2 * inner, width])?;
    let down = mlp.parameter("experts.down_proj", [experts, width, inner])?;
    let shared_inner = inner * config.n_shared_experts;
    let mut shared = mlp.scope("shared_experts");
    let shared = GatedMlp::new(
        shared.linear("gate_proj", width, shared_inner, false)?,
        shared.linear("up_proj", width, shared_inner, false)?,
        shared.linear("down_proj", shared_inner, width, false)?,
        spec.activation,
    );
    Ok(MixtureOfExperts::new(
        MoeLayers {
            router,
            router_bias,
            gate_up,
            down,
            shared,
        },
        MoeRouting {
            groups: config.n_group,
            kept_groups: config.topk_group,
            top_k: config.num_experts_per_tok,
            scale: config.routed_scaling_factor,
        },
    ))
}

// Where a real checkpoint's per-expert tensor belongs in the packed state: `model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight`.
fn expert_piece(name: &str, inner: i64) -> Option<StatePiece> {
    let (prefix, rest) = name.split_once(".mlp.experts.")?;
    let (expert, projection) = rest.split_once('.')?;
    let expert = expert.parse::<i64>().ok()?;
    let (packed, row) = match projection {
        "gate_proj.weight" => ("gate_up_proj", 0),
        "up_proj.weight" => ("gate_up_proj", inner),
        "down_proj.weight" => ("down_proj", 0),
        _ => return None,
    };
    Some(StatePiece {
        state: format!("{prefix}.mlp.experts.{packed}"),
        origin: vec![expert, row, 0],
    })
}

fn schema(config: &DeepseekV3Config, data_type: DataType) -> Result<DeepseekV3> {
    let spec = config.decoder.spec(data_type, false)?;
    let rotary = config.rotary()?;
    let score_correction = rotary.frequencies.score_correction();
    let cache = LatentKvCacheSpec {
        latent_dim: config.kv_lora_rank,
        positional_dim: config.qk_rope_head_dim,
    };
    let (layers, state) = causal_lm(&spec, rotary, cache, None, |schema, index| {
        let moe = i64::try_from(index).is_ok_and(|index| index >= config.first_k_dense_replace);
        pre_norm_block_with(
            schema,
            &spec,
            |schema| attention(schema, config, score_correction),
            |schema| {
                if moe {
                    mixture_of_experts(schema, config, &spec).map(DecoderFeedForward::Moe)
                } else {
                    gated_mlp(schema, &spec).map(DecoderFeedForward::Dense)
                }
            },
        )
    })?;
    Ok(Model::new(layers, state))
}

// Loads a directory holding `config.json` and a checkpoint, converting the weights to `data_type`.
pub fn load(snapshot: impl AsRef<Path>, device: Device, data_type: DataType) -> Result<DeepseekV3> {
    let config: DeepseekV3Config = Snapshot::new(&snapshot).config()?;
    config.validate()?;
    let schema = schema(&config, data_type)?;
    let inner = config.moe_intermediate_size;
    let pieces = move |name: &str| expert_piece(name, inner);
    let state = schema.state().load_checkpoint(
        snapshot,
        &CheckpointOptions {
            device: Some(device),
            convert: true,
            names: None,
            pieces: Some(&pieces),
        },
    )?;
    Ok(schema.with_state(state)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_tensors_map_to_their_place_in_the_packed_state() {
        let piece = |name: &str| expert_piece(name, 32);
        assert_eq!(
            piece("model.layers.3.mlp.experts.5.up_proj.weight"),
            Some(StatePiece {
                state: "model.layers.3.mlp.experts.gate_up_proj".to_owned(),
                origin: vec![5, 32, 0],
            })
        );
        assert_eq!(
            piece("model.layers.3.mlp.experts.5.down_proj.weight"),
            Some(StatePiece {
                state: "model.layers.3.mlp.experts.down_proj".to_owned(),
                origin: vec![5, 0, 0],
            })
        );
        // The packed form, and everything else, stays whole.
        assert_eq!(piece("model.layers.3.mlp.experts.gate_up_proj"), None);
        assert_eq!(
            piece("model.layers.3.mlp.shared_experts.up_proj.weight"),
            None
        );
    }
}
