// GPT-2: learned positions, `LayerNorm`, and attention whose projections all have a bias.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    nn::{
        Activation, AttentionInput, CacheSize, DecodeStep, DecoderModel, DecodingLayer, Embedding,
        GroupedQueryAttention, GroupedQueryLayers, HeadProjection, KvCache, KvCacheSpec, LayerNorm,
        Linear, Mlp, Model, ModelState, PreNormBlock, StateBuilder,
    },
};

use crate::{
    assembly::{causal, kv_caches},
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
    pub positions: i64,
    pub inner: i64,
    pub eps: f32,
    pub activation: Activation,
}

type Block = PreNormBlock<GroupedQueryAttention, Mlp, LayerNorm>;

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

pub type Gpt2 = Model<Gpt2Layers>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "gpt2",
    formats: &[(Format::Transformers, &["gpt2"])],
    heads: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

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

fn attention(schema: &mut Schema<'_>, config: &Gpt2Config) -> SingeResult<GroupedQueryAttention> {
    let (width, heads) = (config.width, config.heads);
    let mut head = |name: &str| -> SingeResult<HeadProjection> {
        let linear = schema.linear(name, width, width, true)?;
        Ok(HeadProjection::new(linear, heads, width / heads))
    };
    let query = head("query")?;
    let key = head("key")?;
    let value = head("value")?;
    let output = schema.linear("output", width, width, true)?;
    GroupedQueryAttention::create(
        GroupedQueryLayers {
            query,
            key,
            value,
            output,
        },
        causal(),
    )
}

fn block(schema: &mut Schema<'_>, config: &Gpt2Config) -> SingeResult<Block> {
    let (width, inner, eps) = (config.width, config.inner, config.eps);
    let attention_norm = schema.layer_norm("attention_norm", width, eps)?;
    let attention = attention(&mut schema.scope("attention"), config)?;
    let feed_forward_norm = schema.layer_norm("feed_forward_norm", width, eps)?;
    let feed_forward = {
        let mut schema = schema.scope("feed_forward");
        Mlp::new(
            schema.linear("up", width, inner, true)?,
            schema.linear("down", inner, width, true)?,
            config.activation,
        )
    };
    Ok(PreNormBlock::new(
        attention_norm,
        attention,
        feed_forward_norm,
        feed_forward,
    ))
}

// The output head shares the token embedding's weight.
fn schema(config: &Gpt2Config, data_type: DataType) -> SingeResult<Gpt2> {
    let width = config.width;
    let mut builder = StateBuilder::new();
    let (tokens, positions, blocks, norm) = {
        let mut model = Schema::root(&mut builder, data_type);
        let tokens = model.parameter("embeddings.tokens.weight", [config.vocab_size, width])?;
        let positions =
            model.parameter("embeddings.positions.weight", [config.positions, width])?;
        let blocks = (0..config.layers)
            .map(|layer| block(&mut model.scope(format!("blocks.{layer}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = model.layer_norm("norm", width, config.eps)?;
        (tokens, positions, blocks, norm)
    };
    let head = builder.tie("head.weight", &tokens)?;
    Ok(Model::new(
        Gpt2Layers {
            tokens: Embedding::new(tokens),
            positions: Embedding::new(positions),
            blocks,
            norm,
            head: Linear::new(head, None),
            cache: KvCacheSpec {
                kv_heads: config.heads,
                head_dim: width / config.heads,
            },
        },
        builder.finish()?,
    ))
}

pub fn load(loader: &mut Loader<'_>) -> Result<Gpt2> {
    let config: Gpt2Config = loader.config(transformers::config)?;
    let schema = schema(&config, loader.options().data_type)?;
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(model)
}
