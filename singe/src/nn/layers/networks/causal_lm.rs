use std::fmt::{Debug, Formatter, Result as FmtResult};

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    generation::DraftingModel,
    nn::{
        AttentionInput, CacheSize, DecodeCache, DecodeStep, DecoderModel, DecodingLayer, Embedding,
        GatedMlp, Layer, Linear, ModelState, PositionCache, PreNormBlock, RmsNorm, RotaryAngles,
        RotarySpec, RotaryTable, SourceCache, SourceCacheSpec, SourcedCache,
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

pub struct MultiTokenPrediction<B: DecodingLayer, N = RmsNorm> {
    pub embedding_norm: N,
    pub hidden_norm: N,
    pub projection: Linear,
    pub block: B,
    pub cache: <B::Cache as DecodeCache>::Spec,
    pub norm: N,
    pub width: i64,
}

impl<B: DecodingLayer, N: Debug> Debug for MultiTokenPrediction<B, N> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("MultiTokenPrediction")
            .field("embedding_norm", &self.embedding_norm)
            .field("hidden_norm", &self.hidden_norm)
            .field("projection", &self.projection)
            .field("block", &self.block)
            .field("cache", &self.cache)
            .field("norm", &self.norm)
            .field("width", &self.width)
            .finish()
    }
}

// A causal language model: an embedding, rotary blocks over one cache each, a final norm, and an output head on each sequence's last token, with a layer that drafts the tokens after it if it has one.
pub struct CausalLm<B: DecodingLayer, N = RmsNorm> {
    parts: CausalLmParts<B, N>,
    draft: Option<MultiTokenPrediction<B, N>>,
}

// The usual decoder: attention `A` and feed-forward `F` in pre-norm blocks, with norm `N`.
pub type Decoder<A, F = GatedMlp, N = RmsNorm> = CausalLm<PreNormBlock<A, F, N>, N>;

impl<B: DecodingLayer, N> CausalLm<B, N> {
    #[must_use]
    pub const fn new(parts: CausalLmParts<B, N>) -> Self {
        Self { parts, draft: None }
    }

    #[must_use]
    pub fn with_draft_layer(mut self, draft: MultiTokenPrediction<B, N>) -> Self {
        self.draft = Some(draft);
        self
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
            .field("draft", &self.draft)
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
        hidden: Tensor,
        tables: &RotaryTable,
        caches: &[B::Cache],
        state: &ModelState,
    ) -> Result<Tensor> {
        let angles = tables.angles(step.positions())?;
        self.forward_rotated(step, hidden, &angles, caches, state)
    }

    // Runs the step from its embeddings, `[T, W]`, rotating by `angles` rather than by its positions.
    pub fn forward_rotated(
        &self,
        step: &DecodeStep,
        hidden: Tensor,
        angles: &RotaryAngles,
        caches: &[B::Cache],
        state: &ModelState,
    ) -> Result<Tensor> {
        self.hidden_rotated(step, hidden, angles, caches, state)?
            .index_select(0, step.last_rows())?
            .apply(&self.parts.head, state)
    }

    fn hidden_rotated(
        &self,
        step: &DecodeStep,
        mut hidden: Tensor,
        angles: &RotaryAngles,
        caches: &[B::Cache],
        state: &ModelState,
    ) -> Result<Tensor> {
        let parts = &self.parts;
        for (block, cache) in parts.blocks.iter().zip(caches) {
            hidden = block.decode(
                &AttentionInput {
                    hidden: &hidden,
                    step,
                    angles: Some(angles),
                    cache,
                },
                state,
            )?;
        }
        hidden.apply(&parts.norm, state)
    }

    fn draft_layer(&self) -> Result<&MultiTokenPrediction<B, N>> {
        self.draft
            .as_ref()
            .ok_or_else(|| Error::missing_input("a decoder's draft layer"))
    }
}

impl<B, N> DraftingModel for CausalLm<B, N>
where
    B: DecodingLayer,
    N: Layer + 'static,
{
    fn hidden_width(&self) -> i64 {
        self.draft.as_ref().map_or(0, |draft| draft.width)
    }

    fn forward_rows(
        &self,
        step: &DecodeStep,
        tables: &RotaryTable,
        caches: &[B::Cache],
        state: &ModelState,
    ) -> Result<(Tensor, Tensor)> {
        let angles = tables.angles(step.positions())?;
        let hidden = self.hidden_rotated(step, self.embed(step, state)?, &angles, caches, state)?;
        Ok((hidden.apply(&self.parts.head, state)?, hidden))
    }

    fn draft_cache(
        &self,
        data_type: DataType,
        device: Device,
        size: CacheSize,
    ) -> Result<B::Cache> {
        B::Cache::create(device, data_type, size, self.draft_layer()?.cache)
    }

    fn draft(
        &self,
        step: &DecodeStep,
        hidden: &Tensor,
        tables: &RotaryTable,
        cache: &B::Cache,
        state: &ModelState,
    ) -> Result<(Tensor, Tensor)> {
        let draft = self.draft_layer()?;
        let joined = Tensor::concat(
            &[
                self.embed(step, state)?
                    .apply(&draft.embedding_norm, state)?,
                hidden.apply(&draft.hidden_norm, state)?,
            ],
            1,
        )?
        .apply(&draft.projection, state)?;
        let angles = tables.angles(step.positions())?;
        let hidden = draft
            .block
            .decode(
                &AttentionInput {
                    hidden: &joined,
                    step,
                    angles: Some(&angles),
                    cache,
                },
                state,
            )?
            .apply(&draft.norm, state)?;
        let logits = match state.named_tensor(DRAFT_HEAD) {
            Some(head) => {
                self.parts
                    .head
                    .forward_with(&hidden.cast(head.data_type())?, head, state)?
            }
            None => hidden.apply(&self.parts.head, state)?,
        };
        Ok((logits, hidden))
    }

    fn draft_buffers(&self, state: &ModelState) -> Result<Vec<(String, Tensor)>> {
        // A draft only proposes, and the check decides in the weights' own type, so an F32 head drafts from a BF16 copy, which reads half as much.
        let head = state.tensor(self.parts.head.weight())?;
        if head.data_type() != DataType::F32 {
            return Ok(Vec::new());
        }

        Ok(vec![(DRAFT_HEAD.to_owned(), head.cast(DataType::Bf16)?)])
    }
}

const DRAFT_HEAD: &str = "draft_head";

// A causal language model whose prompts read sources in place of a placeholder token, as Qwen3-ASR's prompt reads audio features in place of its audio tokens.
//
// Its caches are the model's own, one per block, then a [`SourceCache`], which a sequence's sources are written to before its first step.
// With M-RoPE sections, as Qwen3.5 places an image's tokens on its grid, a [`PositionCache`] follows, which a sequence's rotary offsets are written to before its first step too.
pub struct SourcedCausalLm<B: DecodingLayer, N = RmsNorm> {
    model: CausalLm<B, N>,
    placeholder: i64,
    width: i64,
    sections: Option<[i64; 3]>,
}

impl<B: DecodingLayer, N> SourcedCausalLm<B, N> {
    // Reads sources of `width` features in place of the token `placeholder`.
    #[must_use]
    pub const fn new(model: CausalLm<B, N>, placeholder: i64, width: i64) -> Self {
        Self {
            model,
            placeholder,
            width,
            sections: None,
        }
    }

    // Rotates by three position streams, which the position cache gives, combined by M-RoPE's interleaved `sections` as `RotaryTable::angles_in_sections` does.
    #[must_use]
    pub const fn with_position_sections(mut self, sections: [i64; 3]) -> Self {
        self.sections = Some(sections);
        self
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
        sources_of(caches)?.write(sequence, position, rows)
    }

    // Writes `offsets`, `[capacity, 3]`, as sequence `sequence`'s rotary offsets, in the position cache of `caches`.
    pub fn write_positions(
        caches: &[SourcedCache<B::Cache>],
        sequence: usize,
        offsets: &Tensor,
    ) -> Result<()> {
        positions_of(caches)?.write(sequence, offsets)
    }

    // Writes `offsets`, `[N, 3]`, as sequence `sequence`'s rotary offsets at `positions`, `[N]` of `I64`, as a captured encode step writes them.
    pub fn write_positions_at(
        caches: &[SourcedCache<B::Cache>],
        sequence: usize,
        positions: &Tensor,
        offsets: &Tensor,
    ) -> Result<()> {
        positions_of(caches)?.write_at(sequence, positions, offsets)
    }

    // Writes `rows`, `[N, W]`, as sequence `sequence`'s sources at `positions`, `[N]` of `I64`, in the source cache of `caches`, as a captured encode step that reads where its sources go writes them.
    pub fn write_sources_at(
        caches: &[SourcedCache<B::Cache>],
        sequence: usize,
        positions: &Tensor,
        rows: &Tensor,
    ) -> Result<()> {
        sources_of(caches)?.write_at(sequence, positions, rows)
    }
}

// The position cache of a sourced decoder's `caches`, its last.
fn positions_of<C>(caches: &[SourcedCache<C>]) -> Result<&PositionCache> {
    match caches.last() {
        Some(SourcedCache::Positions(positions)) => Ok(positions),
        _ => Err(Error::missing_input("a sourced decoder's position cache")),
    }
}

// The source cache of a sourced decoder's `caches`, after its blocks' caches.
fn sources_of<C>(caches: &[SourcedCache<C>]) -> Result<&SourceCache> {
    caches
        .iter()
        .rev()
        .find_map(|cache| match cache {
            SourcedCache::Sources(sources) => Some(sources),
            _ => None,
        })
        .ok_or_else(|| Error::missing_input("a sourced decoder's source cache"))
}

impl<B: DecodingLayer, N: Debug> Debug for SourcedCausalLm<B, N> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("SourcedCausalLm")
            .field("model", &self.model)
            .field("placeholder", &self.placeholder)
            .field("width", &self.width)
            .field("sections", &self.sections)
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
        if self.sections.is_some() {
            caches.push(SourcedCache::Positions(PositionCache::create(
                device,
                data_type,
                size,
                (),
            )?));
        }
        Ok(caches)
    }

    fn forward(
        &self,
        step: &DecodeStep,
        tables: &RotaryTable,
        caches: &[Self::Cache],
        state: &ModelState,
    ) -> Result<Tensor> {
        let (mut layers, mut sources, mut positions) = (Vec::new(), None, None);
        for cache in caches {
            match cache {
                SourcedCache::Layer(cache) => layers.push(cache.clone()),
                SourcedCache::Sources(cache) => sources = Some(cache),
                SourcedCache::Positions(cache) => positions = Some(cache),
            }
        }

        let sources =
            sources.ok_or_else(|| Error::missing_input("a sourced decoder's source cache"))?;
        let embedded = sources.embed(step, &self.model.embed(step, state)?, self.placeholder)?;
        let angles = match (self.sections, positions) {
            (Some(sections), Some(positions)) => {
                tables.angles_in_sections(&positions.positions(step)?, sections)?
            }
            (None, _) => tables.angles(step.positions())?,
            (Some(_), None) => {
                return Err(Error::missing_input("a sourced decoder's position cache"));
            }
        };
        self.model
            .forward_rotated(step, embedded, &angles, &layers, state)
    }
}
