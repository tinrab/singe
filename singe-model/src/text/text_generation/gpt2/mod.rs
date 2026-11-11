use std::path::Path;

use serde::Deserialize;
use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    nn::{
        Activation, AttentionInput, CacheSize, DecodeStep, DecoderModel, DecodingLayer, Embedding,
        GroupedQueryAttention, GroupedQueryLayers, HeadProjection, KvCache, KvCacheSpec, LayerNorm,
        Linear, Mlp, Model, ModelState, PreNormBlock, StateBuilder,
    },
};

use crate::{
    assembly::{causal, kv_caches, load_weights},
    config::{Snapshot, unsupported},
    error::Result,
    schema::Schema,
};

const fn default_scale_attention() -> bool {
    true
}

// The fields of a Hugging Face `config.json` this definition reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gpt2Config {
    pub vocab_size: i64,
    pub n_embd: i64,
    pub n_head: i64,
    pub n_layer: i64,
    pub n_positions: i64,
    #[serde(default)]
    pub n_inner: Option<i64>,
    pub layer_norm_epsilon: f32,
    pub activation_function: Activation,
    #[serde(default = "default_scale_attention")]
    pub scale_attn_weights: bool,
    #[serde(default)]
    pub scale_attn_by_inverse_layer_idx: bool,
    #[serde(default)]
    pub reorder_and_upcast_attn: bool,
}

impl Gpt2Config {
    // A checkpoint using one of these would load and then compute something else.
    fn validate(&self) -> Result<()> {
        if !self.scale_attn_weights
            || self.scale_attn_by_inverse_layer_idx
            || self.reorder_and_upcast_attn
        {
            return Err(unsupported("non-default attention scaling"));
        }
        Ok(())
    }
}

type Block = PreNormBlock<GroupedQueryAttention, Mlp, LayerNorm>;

// GPT-2: learned positions, `LayerNorm`, and one weight holding each block's query, key, and value projections.
//
// It shares no assembly with the Llama family, so its layer and its `forward` are written out here.
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

fn block(schema: &mut Schema<'_>, config: &Gpt2Config) -> SingeResult<Block> {
    let width = config.n_embd;
    let heads = config.n_head;
    let head_dim = width / heads;
    let eps = config.layer_norm_epsilon;
    let inner = config.n_inner.unwrap_or(4 * width);
    let attention_norm = schema.layer_norm("ln_1", width, eps)?;
    let attention = {
        let mut schema = schema.scope("attn");
        let projections = schema.conv1d("c_attn", width, 3 * width)?;
        let head = |index: i64| {
            HeadProjection::new(
                projections.clone().with_outputs(index * width, width),
                heads,
                head_dim,
            )
        };
        GroupedQueryAttention::create(
            GroupedQueryLayers {
                query: head(0),
                key: head(1),
                value: head(2),
                output: schema.conv1d("c_proj", width, width)?,
            },
            causal(),
        )?
    };
    let feed_forward_norm = schema.layer_norm("ln_2", width, eps)?;
    let feed_forward = {
        let mut schema = schema.scope("mlp");
        Mlp::new(
            schema.conv1d("c_fc", width, inner)?,
            schema.conv1d("c_proj", inner, width)?,
            config.activation_function,
        )
    };
    Ok(PreNormBlock::new(
        attention_norm,
        attention,
        feed_forward_norm,
        feed_forward,
    ))
}

fn schema(config: &Gpt2Config, data_type: DataType) -> Result<Gpt2> {
    let width = config.n_embd;
    let eps = config.layer_norm_epsilon;
    let mut builder = StateBuilder::new();
    let (tokens, positions, blocks, norm) = {
        let mut model = Schema::new(builder.scope("transformer"), data_type);
        let tokens = model.parameter("wte.weight", [config.vocab_size, width])?;
        let positions = model.parameter("wpe.weight", [config.n_positions, width])?;
        let blocks = (0..config.n_layer)
            .map(|layer| block(&mut model.scope(format!("h.{layer}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = model.layer_norm("ln_f", width, eps)?;
        (tokens, positions, blocks, norm)
    };
    let head = builder.tie("lm_head.weight", &tokens)?;
    Ok(Model::new(
        Gpt2Layers {
            tokens: Embedding::new(tokens),
            positions: Embedding::new(positions),
            blocks,
            norm,
            head: Linear::new(head, None),
            cache: KvCacheSpec {
                kv_heads: config.n_head,
                head_dim: width / config.n_head,
            },
        },
        builder.finish()?,
    ))
}

// Older checkpoints store blocks without `transformer.`, along with attention masks this definition computes itself.
fn state_name(name: &str) -> Option<String> {
    if name.ends_with(".attn.bias") || name.ends_with(".attn.masked_bias") {
        return None;
    }

    if name.starts_with("transformer.") || name == "lm_head.weight" {
        return Some(name.to_owned());
    }
    Some(format!("transformer.{name}"))
}

// Loads a Hugging Face snapshot directory, converting the weights to `data_type`.
pub fn load(snapshot: impl AsRef<Path>, device: Device, data_type: DataType) -> Result<Gpt2> {
    let config: Gpt2Config = Snapshot::new(&snapshot).config()?;
    config.validate()?;
    load_weights(
        &schema(&config, data_type)?,
        snapshot,
        device,
        Some(&state_name),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_older_checkpoint_names_map_to_the_schema_and_masks_are_ignored() {
        assert_eq!(
            state_name("h.0.ln_1.weight").unwrap(),
            "transformer.h.0.ln_1.weight"
        );
        assert_eq!(
            state_name("transformer.wte.weight").unwrap(),
            "transformer.wte.weight"
        );
        assert_eq!(state_name("lm_head.weight").unwrap(), "lm_head.weight");
        assert_eq!(state_name("h.0.attn.bias"), None);
        assert_eq!(state_name("h.0.attn.masked_bias"), None);
        assert_eq!(
            state_name("h.0.attn.c_attn.bias").unwrap(),
            "transformer.h.0.attn.c_attn.bias"
        );
    }
}
