use std::fmt::{Debug, Formatter, Result as FmtResult};

use crate::{
    data_type::DataType,
    device::Device,
    error::Result,
    nn::{
        AttentionInput, CacheSize, DecodeCache, DecodeStep, DecoderModel, DecodingLayer, Embedding,
        GatedMlp, Layer, Linear, ModelState, PreNormBlock, RmsNorm, RotarySpec, RotaryTable,
    },
    tensor::Tensor,
};

// What a [`CausalLm`] is assembled from.
pub struct CausalLmParts<B: DecodingLayer, N = RmsNorm> {
    pub embedding: Embedding,
    // What each token's embedding is multiplied by, if anything.
    pub embedding_scale: Option<f32>,
    pub rotary: RotarySpec,
    pub blocks: Vec<B>,
    pub norm: N,
    pub head: Linear,
    pub cache: <B::Cache as DecodeCache>::Spec,
}

// A causal language model: an embedding, rotary blocks over one cache each, a final norm, and an output head on each sequence's last token.
pub struct CausalLm<B: DecodingLayer, N = RmsNorm> {
    parts: CausalLmParts<B, N>,
}

// The usual decoder: attention `A` and feed-forward `F` in pre-norm blocks, with norm `N`.
pub type Decoder<A, F = GatedMlp, N = RmsNorm> = CausalLm<PreNormBlock<A, F, N>, N>;

impl<B: DecodingLayer, N> CausalLm<B, N> {
    #[must_use]
    pub const fn new(parts: CausalLmParts<B, N>) -> Self {
        Self { parts }
    }
}

// By hand, since the cache's sizes are only known to be `Debug` through its trait.
impl<B: DecodingLayer, N: Debug> Debug for CausalLm<B, N> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        let parts = &self.parts;
        formatter
            .debug_struct("CausalLm")
            .field("embedding", &parts.embedding)
            .field("embedding_scale", &parts.embedding_scale)
            .field("rotary", &parts.rotary)
            .field("blocks", &parts.blocks)
            .field("norm", &parts.norm)
            .field("head", &parts.head)
            .field("cache", &parts.cache)
            .finish()
    }
}

impl<B, N> DecoderModel for CausalLm<B, N>
where
    B: DecodingLayer,
    N: Layer + 'static,
{
    type Cache = B::Cache;

    type Tables = RotaryTable;

    fn tables(&self, data_type: DataType, device: Device, capacity: i64) -> Result<RotaryTable> {
        RotaryTable::create(device, data_type, self.parts.rotary, capacity)
    }

    fn caches(
        &self,
        data_type: DataType,
        device: Device,
        size: CacheSize,
    ) -> Result<Vec<B::Cache>> {
        self.parts
            .blocks
            .iter()
            .map(|_| B::Cache::create(device, data_type, size, self.parts.cache))
            .collect()
    }

    fn forward(
        &self,
        step: &DecodeStep,
        tables: &RotaryTable,
        caches: &[B::Cache],
        state: &ModelState,
    ) -> Result<Tensor> {
        let parts = &self.parts;
        let angles = tables.angles(step.positions())?;
        let mut hidden = step.tokens().apply(&parts.embedding, state)?;
        if let Some(scale) = parts.embedding_scale {
            hidden = hidden.mul_scalar(scale)?;
        }

        for (block, cache) in parts.blocks.iter().zip(caches) {
            hidden = block.decode(
                &AttentionInput {
                    hidden: &hidden,
                    step,
                    angles: Some(&angles),
                    cache,
                },
                state,
            )?;
        }
        hidden
            .apply(&parts.norm, state)?
            .index_select(0, step.last_rows())?
            .apply(&parts.head, state)
    }
}
