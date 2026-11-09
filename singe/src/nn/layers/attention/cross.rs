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
// It is paged attention without the causal mask, reading each sequence's pages where they are.
#[derive(Debug, Clone)]
pub struct CrossAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
    head_dim: i64,
    // How many encoder outputs each sequence's tokens attend to.
    sources: i64,
}

// What cross-attention reads in a decode step: `hidden`, `[T, W]`, the step that says which sequence each token is of, and every sequence's keys and values, `[N · k, page, H, D]`, as `EncoderDecoderCache` pages them.
#[derive(Debug, Clone, Copy)]
pub struct CrossAttentionInput<'a> {
    pub hidden: &'a Tensor,
    pub step: &'a DecodeStep,
    pub keys: &'a Tensor,
    pub values: &'a Tensor,
}

impl CrossAttention {
    // Stores the projections of `heads` heads over a decoder of `width` features, whose tokens attend to `sources` encoder outputs.
    //
    // # Errors
    //
    // Returns an error if `heads` is not positive or does not divide a nonzero `width`, or `sources` is not positive.
    pub fn create(
        query: Linear,
        key: Linear,
        value: Linear,
        output: Linear,
        heads: i64,
        width: i64,
        sources: i64,
    ) -> Result<Self> {
        if heads <= 0 || width <= 0 || width % heads != 0 {
            return Err(Error::invalid_argument("attention heads and width"));
        }

        if sources <= 0 {
            return Err(Error::invalid_argument("cross-attention sources"));
        }
        Ok(Self {
            query,
            key,
            value,
            output,
            heads,
            head_dim: width / heads,
            sources,
        })
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

// The pages each chunk of a sequence's keys holds: a decode step has one query per sequence, so splitting its keys into chunks that run as sequences of their own gives the kernel blocks to spread over, as FlashDecoding does.
const CHUNK_PAGES: i64 = 4;

impl CrossAttention {
    // The paging of a cross cache of `[N · k, page, H, D]`, in which each of `N` sequences owns `k` pages, the last of which holds what `sources` leaves, split into chunks of at most `CHUNK_PAGES` pages that each run as a sequence.
    // Returns the chunks per sequence, and the page offsets, pages, and last page lengths of every chunk of every sequence, chunk-major.
    fn paging(&self, keys: &Tensor) -> Result<(i64, [Tensor; 3])> {
        let SourcePages {
            page,
            per_sequence: blocks,
            sequences,
            last,
        } = SourcePages::create(keys, self.sources)?;
        let per_chunk = blocks.min(CHUNK_PAGES);
        let chunks = pages_for(blocks, per_chunk);
        let (mut offsets, mut pages, mut lengths) = (vec![0_i64], Vec::new(), Vec::new());
        for chunk in 0..chunks {
            let first = chunk * per_chunk;
            let end = (first + per_chunk).min(blocks);
            for sequence in 0..sequences {
                pages.extend(sequence * blocks + first..sequence * blocks + end);
                offsets.push(i64::try_from(pages.len()).map_err(|_| Error::out_of_range("pages"))?);
                lengths.push(if end == blocks { last } else { page });
            }
        }

        let literal = |values: Vec<i64>| {
            let length = i64::try_from(values.len()).map_err(|_| Error::out_of_range("pages"))?;
            capture::literal(keys.device(), |device| {
                Tensor::from_vec(device, Shape::create([length])?, values)
            })
        };
        Ok((
            chunks,
            [literal(offsets)?, literal(pages)?, literal(lengths)?],
        ))
    }
}

// How many pages of `page` slots hold `sources` rows.
#[must_use]
pub fn pages_for(sources: i64, page: i64) -> i64 {
    (sources + page - 1) / page
}

// How a cross-attention cache, `[N · k, page, H, D]`, pages each of its `N` sequences' encoder outputs: sequence `s` owns the `k` pages from `s · k`, the last of which holds what the others leave.
// Both what writes a sequence's sources and what attends to them read the layout from here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePages {
    pub page: i64,
    pub per_sequence: i64,
    pub sequences: i64,
    // The rows the last page of a sequence holds.
    pub last: i64,
}

impl SourcePages {
    // The layout of `cache` for sequences of `sources` encoder outputs each.
    //
    // # Errors
    //
    // Returns an error if `cache` is not `[pages, page, heads, dim]`, `sources` is not positive, or the pages do not divide into whole sequences.
    pub fn create(cache: &Tensor, sources: i64) -> Result<Self> {
        let &[cache_pages, page, _, _] = cache.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "cross-attention caches are [pages, page, heads, dim]",
            ));
        };
        if sources <= 0 || page <= 0 {
            return Err(Error::invalid_argument("cross-attention sources and pages"));
        }

        let per_sequence = pages_for(sources, page);
        if cache_pages % per_sequence != 0 {
            return Err(Error::invalid_argument(
                "a cross-attention cache holds whole sequences of pages",
            ));
        }

        Ok(Self {
            page,
            per_sequence,
            sequences: cache_pages / per_sequence,
            last: sources - (per_sequence - 1) * page,
        })
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

impl Layer<CrossAttentionInput<'_>> for CrossAttention {
    fn forward(&self, input: &CrossAttentionInput<'_>, state: &ModelState) -> Result<Tensor> {
        let &[tokens, width] = input.hidden.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "cross-attention requires [tokens, width]",
            ));
        };
        let query =
            input
                .hidden
                .apply(&self.query, state)?
                .reshape([tokens, self.heads, self.head_dim])?;
        let (chunks, [page_offsets, pages, last_page_lengths]) = self.paging(input.keys)?;
        // Each chunk's queries are the step's, offset by the chunk's place.
        let offsets = input.step.paging().query_offsets;
        let sequences = offsets.shape().dimensions()[0] - 1;
        let shifts = capture::literal(offsets.device(), |device| {
            Tensor::from_vec(
                device,
                Shape::create([chunks, 1])?,
                (0..chunks).map(|chunk| chunk * tokens).collect(),
            )
        })?;
        let end = capture::literal(offsets.device(), |device| {
            Tensor::from_vec(device, Shape::create([1])?, vec![chunks * tokens])
        })?;
        let query_offsets = Tensor::concat(
            &[
                offsets
                    .narrow(0, 0, sequences)?
                    .unsqueeze(0)?
                    .add(&shifts)?
                    .reshape([chunks * sequences])?,
                end,
            ],
            0,
        )?;
        let copies = usize::try_from(chunks).map_err(|_| Error::out_of_range("chunks"))?;
        let (partial, lse) = Tensor::concat(&vec![query.clone(); copies], 0)?.paged_attention(
            input.keys,
            input.values,
            Paging {
                query_offsets: &query_offsets,
                page_offsets: &page_offsets,
                pages: &pages,
                last_page_lengths: &last_page_lengths,
            },
            AttentionOptions::default(),
        )?;
        // Each chunk's result weighs by its share of the softmax's total, from its base-two log-sum-exp.
        let lse = lse.reshape([chunks, tokens, self.heads, 1])?;
        let weights = lse
            .sub(&lse.max_reduce(&[0], true)?)?
            .mul_scalar(LN_2)?
            .exp()?;
        let attended = partial
            .reshape([chunks, tokens, self.heads, self.head_dim])?
            .cast(DataType::F32)?
            .mul(&weights)?
            .sum(&[0], false)?
            .div(&weights.sum(&[0], false)?)?
            .cast(query.data_type())?;
        attended
            .reshape([tokens, width])?
            .apply(&self.output, state)
    }
}
