// GPT-2: learned positions, `LayerNorm`, and attention whose projections all have a bias.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    nn::{
        Activation, AttentionInput, CacheSize, DecodeStep, DecoderModel, DecodingLayer, Embedding,
        GroupedQueryAttention, KvCache, KvCacheSpec, LayerNorm, Linear, Mlp, Model, ModelState,
        PreNormBlock,
    },
};

use crate::{
    assembly::{GroupedAttentionSpec, causal, grouped_attention_layers, kv_caches},
    error::Result,
    format::Loader,
    head::HeadLoader,
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Gpt2Config {
    pub vocab_size: i64,
    pub width: i64,
    pub heads: i64,
    pub layers: i64,
    // The positions the learned position table holds.
    pub max_positions: i64,
    pub inner: i64,
    pub eps: f32,
    pub activation: Activation,
}

type Block = PreNormBlock<GroupedQueryAttention, Mlp, LayerNorm>;

fn block(schema: &mut Schema<'_>, config: &Gpt2Config) -> SingeResult<Block> {
    let (width, eps) = (config.width, config.eps);
    let attention_norm = schema.layer_norm("attention_norm", width, eps)?;
    let attention = GroupedQueryAttention::create(
        grouped_attention_layers(
            &mut schema.scope("attention"),
            &GroupedAttentionSpec::multi_head(width, config.heads, true),
        )?,
        causal(),
    )?;
    let feed_forward_norm = schema.layer_norm("feed_forward_norm", width, eps)?;
    let feed_forward = schema.mlp(
        "feed_forward",
        [width, config.inner, width],
        config.activation,
        true,
    )?;
    Ok(PreNormBlock::new(
        attention_norm,
        attention,
        feed_forward_norm,
        feed_forward,
    ))
}

// GPT-2 shares no assembly with the Llama family, so its layer and its `forward` are written out here.
#[derive(Debug)]
pub struct Gpt2Layers {
    tokens: Embedding,
    positions: Embedding,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
    cache: KvCacheSpec,
}

impl Gpt2Layers {
    // The output head shares the token embedding's weight.
    pub fn create(schema: &mut Schema<'_>, config: &Gpt2Config) -> SingeResult<Self> {
        let width = config.width;
        let tokens = schema.parameter("embeddings.tokens.weight", [config.vocab_size, width])?;
        let positions =
            schema.parameter("embeddings.positions.weight", [config.max_positions, width])?;
        let blocks = (0..config.layers)
            .map(|layer| block(&mut schema.scope(format!("blocks.{layer}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = schema.layer_norm("norm", width, config.eps)?;
        let head = schema.tie("head.weight", &tokens)?;
        Ok(Self {
            tokens: Embedding::new(tokens),
            positions: Embedding::new(positions),
            blocks,
            norm,
            head: Linear::new(head, None),
            cache: KvCacheSpec {
                kv_heads: config.heads,
                head_dim: width / config.heads,
            },
        })
    }
}

impl DecoderModel for Gpt2Layers {
    type Cache = KvCache;

    // Positions are looked up in the state, so nothing depends on the capacity.
    type Tables = ();

    fn tables(&self, _: DataType, _: Device, _: i64) -> SingeResult<()> {
        Ok(())
    }

    fn caches(
        &self,
        data_type: DataType,
        device: Device,
        size: CacheSize,
    ) -> SingeResult<Vec<KvCache>> {
        kv_caches(self.blocks.len(), data_type, device, size, self.cache)
    }

    fn forward(
        &self,
        step: &DecodeStep,
        (): &(),
        caches: &[KvCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mut hidden = step
            .tokens()
            .apply(&self.tokens, state)?
            .add(&step.positions().apply(&self.positions, state)?)?;
        for (block, cache) in self.blocks.iter().zip(caches) {
            hidden = block.decode(
                &AttentionInput {
                    hidden: &hidden,
                    step,
                    angles: None,
                    cache,
                },
                state,
            )?;
        }
        hidden
            .apply(&self.norm, state)?
            .index_select(0, step.last_rows())?
            .apply(&self.head, state)
    }
}

pub type Gpt2 = Model<Gpt2Layers>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "gpt2",
    formats: &[(Format::Transformers, &["gpt2"])],
    loaders: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Gpt2> {
    let config: Gpt2Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Gpt2Layers::create(schema, &config)
    })?;
    loader.load_whole(&model, &transformers::rules())
}
