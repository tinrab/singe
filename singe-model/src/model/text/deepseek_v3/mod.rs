// DeepSeek-V3: multi-head latent attention, whose cache holds a compressed latent per token, and feed-forward layers that are dense in the first blocks and after them a mixture of grouped experts beside a shared expert.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Result as SingeResult,
    nn::{
        Decoder, DecoderFeedForward, GatedMlp, LatentKvCacheSpec, LatentQuery, MixtureOfExperts,
        Model, MoeLayers, MoeRouting, MultiHeadLatentAttention, MultiHeadLatentLayers,
        RotaryLayout, RotarySpec,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{CausalLmSpec, DecoderConfig, causal, causal_lm, gated_mlp, pre_norm_block_with},
    error::Result,
    format::Loader,
    head::HeadLoader,
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

// The sizes of multi-head latent attention, beside the decoder's heads and `rotary_dim`, the features of each query and key head that turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentAttentionConfig {
    // The rank queries are compressed to, or `None` for queries projected directly.
    pub query_rank: Option<i64>,
    // The width of the latent that each token's keys and values are compressed to, which the cache holds.
    pub latent_dim: i64,
    // The features of each query and key head that no rotation turns.
    pub key_dim: i64,
    pub value_dim: i64,
    // Whether rotated features pair with their neighbors instead of with the feature half a head away.
    pub interleaved: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExpertsConfig {
    pub count: i64,
    pub inner: i64,
    // The width of the shared expert every token also passes through.
    pub shared_inner: i64,
    // The experts each token goes through, chosen from the `kept_groups` of `groups` whose experts score highest.
    pub top_k: i64,
    pub groups: i64,
    pub kept_groups: i64,
    // What the chosen experts' weights are multiplied by.
    pub scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeepseekV3Config {
    pub decoder: DecoderConfig,
    pub attention: LatentAttentionConfig,
    // The leading blocks whose feed-forward layer is dense; the rest have experts.
    pub dense_blocks: i64,
    pub experts: ExpertsConfig,
}

impl DeepseekV3Config {
    #[must_use]
    pub const fn rotary(&self) -> RotarySpec {
        RotarySpec {
            head_dim: self.decoder.rotary_dim,
            frequencies: self.decoder.frequencies,
            layout: if self.attention.interleaved {
                RotaryLayout::Interleaved
            } else {
                RotaryLayout::HalfSplit
            },
        }
    }
}

// DeepSeek-V3's layers: multi-head latent attention, and a feed-forward layer that is dense in the first few and mixture-of-experts after them.
pub type DeepseekV3Layers = Decoder<MultiHeadLatentAttention, DecoderFeedForward>;

pub type DeepseekV3 = Model<DeepseekV3Layers>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "deepseek_v3",
    formats: &[(Format::Transformers, &["deepseek_v3"])],
    heads: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

fn attention(
    schema: &mut Schema<'_>,
    config: &DeepseekV3Config,
    score_correction: f64,
) -> SingeResult<MultiHeadLatentAttention> {
    let (width, heads, eps) = (
        config.decoder.width,
        config.decoder.heads,
        config.decoder.eps,
    );
    let rotary_dim = config.decoder.rotary_dim;
    let LatentAttentionConfig {
        query_rank,
        latent_dim,
        key_dim,
        value_dim,
        ..
    } = config.attention;
    let query_width = heads * (key_dim + rotary_dim);
    let query = match query_rank {
        Some(rank) => LatentQuery::LowRank {
            compress: schema.linear("query_compress", width, rank, false)?,
            norm: schema.rms_norm("query_latent_norm", rank, eps)?,
            expand: schema.linear("query_expand", rank, query_width, false)?,
        },
        None => LatentQuery::Direct(schema.linear("query", width, query_width, false)?),
    };
    let compress = schema.linear("latent_compress", width, latent_dim + rotary_dim, false)?;
    let latent_norm = schema.rms_norm("latent_norm", latent_dim, eps)?;
    let expand = schema.parameter(
        "latent_expand.weight",
        [heads * (key_dim + value_dim), latent_dim],
    )?;
    let output = schema.linear("output", heads * value_dim, width, false)?;
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "the scale is defined in F32, as the references compute it"
    )]
    let scale = (score_correction / ((key_dim + rotary_dim) as f64).sqrt()) as f32;
    MultiHeadLatentAttention::create(
        MultiHeadLatentLayers {
            query,
            compress,
            latent_norm,
            expand,
            output,
            heads,
            key_dim,
            rotary_dim,
            value_dim,
            latent_dim,
        },
        AttentionOptions {
            scale: Some(scale),
            ..causal()
        },
    )
}

// Registers `feed_forward`'s `router`, the packed `experts`, and the `shared` expert.
fn mixture_of_experts(
    schema: &mut Schema<'_>,
    config: &DeepseekV3Config,
) -> SingeResult<MixtureOfExperts> {
    let width = config.decoder.width;
    let ExpertsConfig {
        count,
        inner,
        shared_inner,
        ..
    } = config.experts;
    let mut feed_forward = schema.scope("feed_forward");
    let router = feed_forward.parameter("router.weight", [count, width])?;
    let router_bias = feed_forward.parameter("router.bias", [count])?;
    let gate_up = feed_forward.parameter("experts.gate_up", [count, 2 * inner, width])?;
    let down = feed_forward.parameter("experts.down", [count, width, inner])?;
    let mut shared = feed_forward.scope("shared");
    let shared = GatedMlp::new(
        shared.linear("gate", width, shared_inner, false)?,
        shared.linear("up", width, shared_inner, false)?,
        shared.linear("down", shared_inner, width, false)?,
        config.decoder.activation,
    );
    Ok(MixtureOfExperts::new(
        MoeLayers {
            router,
            gate_up,
            down,
            shared,
            shared_gate: None,
        },
        MoeRouting::Grouped {
            bias: router_bias,
            groups: config.experts.groups,
            kept_groups: config.experts.kept_groups,
            top_k: config.experts.top_k,
            scale: config.experts.scale,
        },
    ))
}

fn schema(config: &DeepseekV3Config, data_type: DataType) -> SingeResult<DeepseekV3> {
    let decoder = &config.decoder;
    let score_correction = decoder.frequencies.score_correction();
    let cache = LatentKvCacheSpec {
        latent_dim: config.attention.latent_dim,
        positional_dim: decoder.rotary_dim,
    };
    let (layers, state) = causal_lm(
        decoder,
        data_type,
        CausalLmSpec::new(config.rotary(), |_| cache),
        |schema, index| {
            let moe = i64::try_from(index).is_ok_and(|index| index >= config.dense_blocks);
            pre_norm_block_with(
                schema,
                decoder,
                |schema| attention(schema, config, score_correction),
                |schema| {
                    if moe {
                        mixture_of_experts(schema, config).map(DecoderFeedForward::Moe)
                    } else {
                        gated_mlp(schema, decoder).map(DecoderFeedForward::Dense)
                    }
                },
            )
        },
    )?;
    Ok(Model::new(layers, state))
}

pub fn load(loader: &mut Loader<'_>) -> Result<DeepseekV3> {
    let config: DeepseekV3Config = loader.config(transformers::config)?;
    let schema = schema(&config, loader.options().data_type)?;
    let model = loader.load("model", &schema, &transformers::rules(&config))?;
    loader.finish()?;
    Ok(model)
}
