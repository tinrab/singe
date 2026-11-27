use std::f32::consts::LN_2;

use crate::{
    data_type::DataType,
    error::{Error, Result},
    graph::capture,
    nn::{DecodeStep, Layer, Linear, ModelState},
    shape::Shape,
    tensor::{AttentionOptions, Paging, Tensor},
};

// Attention from a decoder's tokens to an encoder's outputs: queries from the decoder, and keys and values the encoder's outputs project to once per sequence, without a mask.
// It is paged attention without the causal mask, reading each sequence's pages where they are, as many of their slots as its encoder gave it.
#[derive(Debug, Clone)]
pub struct CrossAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
    head_dim: i64,
    scale: Option<f32>,
}

// What cross-attention reads in a decode step: `hidden`, `[T, W]`, the step that says which sequence each token is of, and every sequence's keys and values, `[N · k, page, H, D]`, and source counts, `[N]` of `I64`, as `EncoderDecoderCache` holds them.
#[derive(Debug, Clone, Copy)]
pub struct CrossAttentionInput<'a> {
    pub hidden: &'a Tensor,
    pub step: &'a DecodeStep,
    pub keys: &'a Tensor,
    pub values: &'a Tensor,
    pub lengths: &'a Tensor,
}

impl CrossAttention {
    // Stores the projections of `heads` heads that split `inner` features.
    //
    // # Errors
    //
    // Returns an error if `heads` is not positive or does not divide a nonzero `inner`.
    pub fn create(
        query: Linear,
        key: Linear,
        value: Linear,
        output: Linear,
        heads: i64,
        inner: i64,
    ) -> Result<Self> {
        if heads <= 0 || inner <= 0 || inner % heads != 0 {
            return Err(Error::invalid_argument("attention heads and width"));
        }

        Ok(Self {
            query,
            key,
            value,
            output,
            heads,
            head_dim: inner / heads,
            scale: None,
        })
    }

    // Scales scores by `scale` rather than by the reciprocal square root of the head dimension, as T5's unscaled attention does.
    #[must_use]
    pub const fn with_scale(mut self, scale: f32) -> Self {
        self.scale = Some(scale);
        self
    }

    // Projects one sequence's encoder outputs, `[F, W]`, to the keys and values its tokens attend to, `[F, H, D]`.
    pub fn sources(&self, encoded: &Tensor, state: &ModelState) -> Result<(Tensor, Tensor)> {
        let &[frames, _] = encoded.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "cross-attention sources require [frames, width]",
            ));
        };
        let heads = |linear: &Linear| {
            encoded
                .apply(linear, state)?
                .reshape([frames, self.heads, self.head_dim])
        };
        Ok((heads(&self.key)?, heads(&self.value)?))
    }
}

impl Layer<CrossAttentionInput<'_>> for CrossAttention {
    fn forward(&self, input: &CrossAttentionInput<'_>, state: &ModelState) -> Result<Tensor> {
        let &[tokens, _] = input.hidden.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "cross-attention requires [tokens, width]",
            ));
        };
        let query =
            input
                .hidden
                .apply(&self.query, state)?
                .reshape([tokens, self.heads, self.head_dim])?;
        let paging = SourcePaging::create(input, tokens)?;
        let pages = paging.pages;
        let (partial, lse) = query
            .unsqueeze(0)?
            .expand(Shape::create([pages, tokens, self.heads, self.head_dim])?)?
            .contiguous()?
            .reshape([pages * tokens, self.heads, self.head_dim])?
            .paged_attention(
                input.keys,
                input.values,
                paging.paging(),
                None,
                AttentionOptions {
                    scale: self.scale,
                    ..AttentionOptions::default()
                },
            )?;
        // Each page's result weighs by its share of the softmax's total, from its base-two log-sum-exp.
        let lse = lse
            .reshape([pages, tokens, self.heads, 1])?
            .add(&paging.empty)?;
        let weights = lse
            .sub(&lse.max_reduce(&[0], true)?)?
            .mul_scalar(LN_2)?
            .exp()?;
        let attended = partial
            .reshape([pages, tokens, self.heads, self.head_dim])?
            .cast(DataType::F32)?
            .mul(&weights)?
            .sum(&[0], false)?
            .div(&weights.sum(&[0], false)?)?
            .cast(query.data_type())?;
        attended
            .reshape([tokens, self.heads * self.head_dim])?
            .apply(&self.output, state)
    }
}

// How many pages of `page` slots hold `sources` rows.
#[must_use]
pub fn pages_for(sources: i64, page: i64) -> i64 {
    (sources + page - 1) / page
}

// How a cross-attention cache, `[N · k, page, H, D]`, pages each of its `N` sequences' encoder outputs: sequence `s` owns the `k` pages from `s · k`.
// Both what writes a sequence's sources and what attends to them read the layout from here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePageLayout {
    pub page: i64,
    pub per_sequence: i64,
    pub sequences: i64,
}

impl SourcePageLayout {
    // The layout of `cache` for `sequences` sequences.
    //
    // # Errors
    //
    // Returns an error if `cache` is not `[pages, page, heads, dim]`, or its pages do not divide among `sequences` sequences.
    pub fn create(cache: &Tensor, sequences: i64) -> Result<Self> {
        let &[cache_pages, page, _, _] = cache.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "cross-attention caches are [pages, page, heads, dim]",
            ));
        };
        if sequences <= 0 || page <= 0 || cache_pages <= 0 || cache_pages % sequences != 0 {
            return Err(Error::invalid_argument(
                "a cross-attention cache holds whole sequences of pages",
            ));
        }

        Ok(Self {
            page,
            per_sequence: cache_pages / sequences,
            sequences,
        })
    }

    // The most sources a sequence holds.
    #[must_use]
    pub const fn capacity(&self) -> i64 {
        self.per_sequence * self.page
    }

    // The first page sequence `sequence` owns.
    //
    // # Errors
    //
    // Returns an error if the cache holds no such sequence.
    pub fn first(&self, sequence: i64) -> Result<i64> {
        if !(0..self.sequences).contains(&sequence) {
            return Err(Error::out_of_range("cross-attention sequence"));
        }

        Ok(sequence * self.per_sequence)
    }
}

// The amount that sends a page holding none of its sequence's sources far below every other page's log-sum-exp, so its weight is zero.
const EMPTY_PAGE: f64 = 1.0e30;

// How a decode step attends to a cross cache, `[N · k, page, H, D]`, in which each of `N` sequences owns `k` pages.
// Each page of each sequence runs as a sequence of its own, page-major, so a step's few queries spread over many blocks, as FlashDecoding splits a row's keys.
struct SourcePaging {
    // The pages each sequence owns, `k`.
    pages: i64,
    // Each page's queries: the step's, offset by the page's place.
    query_offsets: Tensor,
    page_offsets: Tensor,
    table: Tensor,
    // Each page's keys: its sequence's sources past the page's first slot, at least one, so every kernel sees a key, and at most a page.
    last_page_lengths: Tensor,
    // What each page adds to each of its tokens' log-sum-exps, `[k, T, 1, 1]`: nothing where the page holds sources, and `-EMPTY_PAGE` where it holds none.
    empty: Tensor,
}

impl SourcePaging {
    fn create(input: &CrossAttentionInput<'_>, tokens: i64) -> Result<Self> {
        let &[sequences] = input.lengths.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "cross-attention source lengths are [sequences]",
            ));
        };
        let SourcePageLayout {
            page,
            per_sequence: pages,
            ..
        } = SourcePageLayout::create(input.keys, sequences)?;
        let step_offsets = input.step.paging().query_offsets;
        if step_offsets.shape().dimensions() != [sequences + 1] {
            return Err(Error::invalid_argument(
                "cross-attention reads one source length per decoder sequence",
            ));
        }

        let literal = |values: Vec<i64>, dimensions: &[i64]| {
            capture::literal(input.keys.device(), |device| {
                Tensor::from_vec(device, Shape::create(dimensions.to_vec())?, values)
            })
        };
        let count = pages * sequences;
        let query_offsets = Tensor::concat(
            &[
                step_offsets
                    .narrow(0, 0, sequences)?
                    .unsqueeze(0)?
                    .add(&literal(
                        (0..pages).map(|index| index * tokens).collect(),
                        &[pages, 1],
                    )?)?
                    .reshape([count])?,
                literal(vec![pages * tokens], &[1])?,
            ],
            0,
        )?;
        // Each page's first slot in its sequence, `[k, 1]`.
        let starts = literal((0..pages).map(|index| index * page).collect(), &[pages, 1])?;
        let last_page_lengths = input
            .lengths
            .unsqueeze(0)?
            .sub(&starts)?
            .clamp(1.0, page as f32)?
            .cast(DataType::I64)?
            .reshape([count])?;
        let held = input
            .lengths
            .index_select(0, input.step.sequence_ids())?
            .unsqueeze(0)?
            .sub(&starts)?
            .clamp(0.0, 1.0)?
            .reshape([pages, tokens, 1, 1])?;
        Ok(Self {
            pages,
            query_offsets,
            page_offsets: literal((0..=count).collect(), &[count + 1])?,
            table: literal(
                (0..pages)
                    .flat_map(|index| (0..sequences).map(move |sequence| sequence * pages + index))
                    .collect(),
                &[count],
            )?,
            last_page_lengths,
            empty: held.sub_scalar(1.0)?.mul_scalar(EMPTY_PAGE)?,
        })
    }

    fn paging(&self) -> Paging<'_> {
        Paging {
            query_offsets: &self.query_offsets,
            page_offsets: &self.page_offsets,
            pages: &self.table,
            last_page_lengths: &self.last_page_lengths,
        }
    }
}
