// DeepSeek-V3: multi-head latent attention, whose cache holds a compressed latent per token, and feed-forward layers that are dense in the first blocks and after them a mixture of grouped experts beside a shared expert.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult,
    nn::{
        Decoder, DecoderFeedForward, LatentKvCacheSpec, LatentQuery, Model,
        MultiHeadLatentAttention, MultiHeadLatentLayers,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{
        CausalLmSpec, DecoderConfig, ExpertsConfig, causal, causal_lm, gated_mlp,
        mixture_of_experts, pre_norm_block_with,
    },
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
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeepseekV3Config {
    pub decoder: DecoderConfig,
    pub attention: LatentAttentionConfig,
    // The leading blocks whose feed-forward layer is dense; the rest have experts.
    pub dense_blocks: i64,
    pub experts: ExpertsConfig,
}

// DeepSeek-V3's layers: multi-head latent attention, and a feed-forward layer that is dense in the first few and mixture-of-experts after them.
pub type DeepseekV3Layers = Decoder<MultiHeadLatentAttention, DecoderFeedForward>;

pub type DeepseekV3 = Model<DeepseekV3Layers>;

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

fn decoder(schema: &mut Schema<'_>, config: &DeepseekV3Config) -> SingeResult<DeepseekV3Layers> {
    let decoder = &config.decoder;
    let score_correction = decoder.frequencies.score_correction();
    let cache = LatentKvCacheSpec {
        latent_dim: config.attention.latent_dim,
        positional_dim: decoder.rotary_dim,
    };
    causal_lm(
        schema,
        decoder,
        CausalLmSpec::new(decoder.rotary(), |_| cache),
        |schema, index| {
            let moe = i64::try_from(index).is_ok_and(|index| index >= config.dense_blocks);
            pre_norm_block_with(
                schema,
                decoder,
                |schema| attention(schema, config, score_correction),
                |schema| {
                    if moe {
                        mixture_of_experts(
                            schema,
                            &config.experts,
                            decoder.width,
                            decoder.activation,
                        )
                        .map(DecoderFeedForward::Moe)
                    } else {
                        gated_mlp(schema, decoder).map(DecoderFeedForward::Dense)
                    }
                },
            )
        },
    )
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "deepseek_v3",
    formats: &[(Format::Transformers, &["deepseek_v3"])],
    loaders: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<DeepseekV3> {
    let config: DeepseekV3Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        decoder(schema, &config)
    })?;
    loader.load_whole(&model, &transformers::rules(&config))
}
