use std::fmt::{Debug, Formatter, Result as FmtResult};

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    nn::{
        AttentionInput, CacheSize, DecodeCache, DecodeStep, DecoderModel, DecodingLayer, Embedding,
        GatedMlp, Layer, Linear, ModelState, PreNormBlock, RmsNorm, RotarySpec, RotaryTable,
        SourceCache, SourceCacheSpec, SourcedCache,
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
    // Each block's cache, in the blocks' order, since a hybrid model's layers hold caches of different kinds.
    pub caches: Vec<<B::Cache as DecodeCache>::Spec>,
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
            .field("caches", &parts.caches)
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
        if self.parts.caches.len() != self.parts.blocks.len() {
            return Err(Error::length_mismatch(
                "causal model cache specs",
                self.parts.blocks.len(),
                self.parts.caches.len(),
            ));
        }

        self.parts
            .caches
            .iter()
            .map(|&spec| B::Cache::create(device, data_type, size, spec))
            .collect()
    }

    fn forward(
        &self,
        step: &DecodeStep,
        tables: &RotaryTable,
        caches: &[B::Cache],
        state: &ModelState,
    ) -> Result<Tensor> {
        self.forward_embedded(step, self.embed(step, state)?, tables, caches, state)
    }
}

impl<B, N> CausalLm<B, N>
where
    B: DecodingLayer,
    N: Layer + 'static,
{
    // Embeds the step's tokens, `[T, W]`, scaled if the model scales them.
    pub fn embed(&self, step: &DecodeStep, state: &ModelState) -> Result<Tensor> {
        let parts = &self.parts;
        let hidden = step.tokens().apply(&parts.embedding, state)?;
        match parts.embedding_scale {
            Some(scale) => hidden.mul_scalar(scale),
            None => Ok(hidden),
        }
    }

    // Runs the step from its embeddings, `[T, W]`, as `forward` does from its tokens.
    pub fn forward_embedded(
        &self,
        step: &DecodeStep,
        mut hidden: Tensor,
        tables: &RotaryTable,
        caches: &[B::Cache],
        state: &ModelState,
    ) -> Result<Tensor> {
        let parts = &self.parts;
        let angles = tables.angles(step.positions())?;
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

// A causal language model whose prompts read sources in place of a placeholder token, as Qwen3-ASR's prompt reads audio features in place of its audio tokens.
//
// Its caches are the model's own, one per block, then a [`SourceCache`], which a sequence's sources are written to before its first step.
pub struct SourcedCausalLm<B: DecodingLayer, N = RmsNorm> {
    model: CausalLm<B, N>,
    placeholder: i64,
    width: i64,
}

impl<B: DecodingLayer, N> SourcedCausalLm<B, N> {
    // Reads sources of `width` features in place of the token `placeholder`.
    #[must_use]
    pub const fn new(model: CausalLm<B, N>, placeholder: i64, width: i64) -> Self {
        Self {
            model,
            placeholder,
            width,
        }
    }

    #[must_use]
    pub const fn placeholder(&self) -> i64 {
        self.placeholder
    }

    // Writes `rows`, `[N, W]`, as sequence `sequence`'s sources at positions `position` onward, in the source cache of `caches`.
    pub fn write_sources(
        caches: &[SourcedCache<B::Cache>],
        sequence: usize,
        position: i64,
        rows: &Tensor,
    ) -> Result<()> {
        match caches.last() {
            Some(SourcedCache::Sources(sources)) => sources.write(sequence, position, rows),
            _ => Err(Error::missing_input("a sourced decoder's source cache")),
        }
    }
}

impl<B: DecodingLayer, N: Debug> Debug for SourcedCausalLm<B, N> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("SourcedCausalLm")
            .field("model", &self.model)
            .field("placeholder", &self.placeholder)
            .field("width", &self.width)
            .finish()
    }
}

impl<B, N> DecoderModel for SourcedCausalLm<B, N>
where
    B: DecodingLayer,
    N: Layer + Debug + 'static,
{
    type Cache = SourcedCache<B::Cache>;

    type Tables = RotaryTable;

    fn tables(&self, data_type: DataType, device: Device, capacity: i64) -> Result<RotaryTable> {
        self.model.tables(data_type, device, capacity)
    }

    fn caches(
        &self,
        data_type: DataType,
        device: Device,
        size: CacheSize,
    ) -> Result<Vec<Self::Cache>> {
        let mut caches = self
            .model
            .caches(data_type, device, size)?
            .into_iter()
            .map(SourcedCache::Layer)
            .collect::<Vec<_>>();
        caches.push(SourcedCache::Sources(SourceCache::create(
            device,
            data_type,
            size,
            SourceCacheSpec { width: self.width },
        )?));
        Ok(caches)
    }

    fn forward(
        &self,
        step: &DecodeStep,
        tables: &RotaryTable,
        caches: &[Self::Cache],
        state: &ModelState,
    ) -> Result<Tensor> {
        let Some((SourcedCache::Sources(sources), layers)) = caches.split_last() else {
            return Err(Error::missing_input("a sourced decoder's source cache"));
        };
        let layers = layers
            .iter()
            .map(|cache| match cache {
                SourcedCache::Layer(cache) => Ok(cache.clone()),
                SourcedCache::Sources(_) => Err(Error::invalid_argument(
                    "a sourced decoder has one source cache, after its blocks' caches",
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let embedded = sources.embed(step, &self.model.embed(step, state)?, self.placeholder)?;
        self.model
            .forward_embedded(step, embedded, tables, &layers, state)
    }
}
