//! T5: an encoder and a decoder of pre-norm blocks with RMS norms, whose attention scores are unscaled and whose self-attention scores are biased by buckets of each pair's relative position, learned once per stack.
//! A tied decoder's output reads the token embedding, after scaling the final states by `width^-0.5`.
//! Switch Transformers is T5 with a mixture of experts in place of every other block's feed-forward layer, and `LongT5` T5 whose encoder attends within a band and to one summed token per block.

pub(crate) mod transformers;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::{DimensionExpr, capture},
    nn::{
        Activation, AttentionInput, CrossAttention, CrossAttentionInput, DecodeStep, Embedding,
        EncoderDecoderCache, GatedMlp, GroupedQueryAttention, KvCacheSpec, Layer, Linear, Mlp,
        ModelState, RmsNorm, StateKey,
    },
    shape::spatial::Padding,
    tensor::AttentionOptions,
};

use crate::{
    assembly::{
        Attention, AttentionSpec, merge_heads, paged_cross_attention, paged_self_attention,
        split_heads,
    },
    error::Result,
    format::{Loader, transformers::seq2seq_generation},
    head::{Boxed, HeadLoader, Seq2SeqDecoding, Seq2SeqInput, Seq2SeqLayers, Seq2SeqLm},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct T5Config {
    pub vocab_size: i64,
    pub width: i64,
    /// Each head's width, which `heads` times need not be `width`.
    pub head_dim: i64,
    pub heads: i64,
    pub inner: i64,
    pub encoder_layers: i64,
    pub decoder_layers: i64,
    pub buckets: i64,
    pub max_distance: i64,
    pub eps: f32,
    pub activation: Activation,
    /// Whether the feed-forward layer gates its activation with a second projection, as T5 v1.1's does.
    pub gated: bool,
    /// Whether the output reads the token embedding, after scaling the decoder's final states by `width^-0.5`.
    pub tied: bool,
    /// Switch Transformers' mixtures of experts in place of some blocks' feed-forward layers.
    #[serde(default)]
    pub experts: Option<SwitchExperts>,
    /// `LongT5`'s transient-global encoder attention, in place of full attention.
    #[serde(default)]
    pub transient_global: Option<TransientGlobal>,
}

/// `LongT5`'s transient-global attention: each token reads the tokens within `radius` of it and one global token per block of `block` tokens, the sum of the block's normed states, normed again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransientGlobal {
    pub radius: i64,
    pub block: i64,
}

/// The bias of a score `LongT5` leaves out: a key outside the band, which `transformers` never puts in a query's blocks.
const OUTSIDE: f32 = -1e10;

/// Where Switch Transformers' mixtures of experts sit and how many tokens each expert takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwitchExperts {
    pub count: i64,
    /// Every `step`-th block of each stack, from the second, has experts.
    pub encoder_step: i64,
    pub decoder_step: i64,
    /// The most tokens of one sequence each expert takes, in order; later ones get no expert output.
    pub capacity: i64,
}

impl SwitchExperts {
    /// Returns whether block `index` of a stack whose step is `step` has experts, as `SwitchTransformersStack` places them.
    const fn is_sparse(step: i64, index: i64) -> bool {
        step > 0 && (step == 1 || index % step == 1)
    }
}

/// Which of T5's stacks a part belongs to: the encoder, whose position bias looks both ways, or the decoder, whose looks only back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StackRole {
    Encoder,
    Decoder,
}

/// Returns the bucket of the relative position `relative`, `key - query`, as `_relative_position_bucket` sorts them.
/// An encoder gives each sign half the buckets, and a decoder gives all of them to keys at or before the query.
/// Within those, distances below half the buckets have a bucket each, and longer ones share buckets logarithmically up to `max_distance`.
fn bucket(relative: i64, role: StackRole, buckets: i64, max_distance: i64) -> i64 {
    let (buckets, mut bucket, distance) = match role {
        StackRole::Encoder => {
            let half = buckets / 2;
            (half, if relative > 0 { half } else { 0 }, relative.abs())
        }
        StackRole::Decoder => (buckets, 0, (-relative).max(0)),
    };
    let exact = buckets / 2;
    if distance < exact {
        bucket += distance;
    } else {
        let large = exact
            + ((distance as f32 / exact as f32).ln() / (max_distance as f32 / exact as f32).ln()
                * (buckets - exact) as f32) as i64;
        bucket += large.min(buckets - 1);
    }
    bucket
}

/// The attention options of T5's unscaled scores, its initialization having folded the scale into the queries.
const UNSCALED: AttentionOptions = AttentionOptions {
    scale: Some(1.0),
    causal: false,
    window: None,
    softcap: None,
};

#[derive(Debug, Clone)]
enum FeedForward {
    Plain(Mlp),
    Gated(GatedMlp),
    Experts(Experts),
}

impl Layer for FeedForward {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        match self {
            Self::Plain(mlp) => mlp.forward(input, state),
            Self::Gated(mlp) => mlp.forward(input, state),
            Self::Experts(experts) => experts.forward(input, state),
        }
    }
}

/// A mixture of `ReLU` experts, as `SwitchTransformersSparseMLP` computes it: a router in `F32` picks each token's likeliest expert, whose output is scaled by that probability, and a token past its expert's capacity in its sequence gets no expert output.
/// Every expert runs over every token and the routing picks the outputs, which is exact and costs the experts' count in feed-forward layers.
#[derive(Debug, Clone)]
struct Experts {
    router: StateKey,
    networks: Vec<[Linear; 2]>,
    activation: Activation,
    capacity: i64,
}

impl Experts {
    fn create(
        schema: &mut Schema<'_>,
        config: &T5Config,
        spec: SwitchExperts,
    ) -> SingeResult<Self> {
        let (width, inner) = (config.width, config.inner);
        let networks = (0..spec.count)
            .map(|expert| {
                Ok([
                    schema.linear(&format!("experts.expert_{expert}.wi"), width, inner, false)?,
                    schema.linear(&format!("experts.expert_{expert}.wo"), inner, width, false)?,
                ])
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            router: schema.parameter("router.weight", [spec.count, width])?,
            networks,
            activation: config.activation,
            capacity: spec.capacity,
        })
    }

    /// Runs the experts over an encoder's `[B, L, W]`, whose sequences' tokens count toward capacity in order, or a decode step's `[T, W]`, each row its sequence's one token, which no capacity binds.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let rank = input.shape().dimensions().len();
        let width = input.dimension(rank - 1)?;
        let rows = input.reshape_with(&[DimensionExpr::constant(-1), width])?;
        let device = input.device();
        let count = i64::try_from(self.networks.len()).unwrap_or(i64::MAX);
        let probabilities = rows
            .cast(DataType::F32)?
            .matmul(
                &state
                    .tensor(&self.router)?
                    .cast(DataType::F32)?
                    .transpose(0, 1)?,
            )?
            .softmax(-1)?;
        let (best, chosen) = probabilities.topk(1, -1, true, true)?;
        let experts = Tensor::range(device, 0_i64, count)?.unsqueeze(0)?;
        let mut routed = chosen.equal(&experts)?;
        if rank == 3 {
            // Each expert takes its sequence's first `capacity` tokens.
            let shape = [input.dimension(0)?, input.dimension(1)?, count.into()];
            let order = routed
                .cast(DataType::I64)?
                .reshape_with(&shape)?
                .cumsum(1)?
                .reshape_with(&[DimensionExpr::constant(-1), count.into()])?;
            let capacity = capture::scalar(device, self.capacity)?;
            routed = routed.logical_and(&order.less_equal(&capacity)?)?;
        }

        let weights = routed
            .cast(DataType::F32)?
            .mul(&best)?
            .cast(input.data_type())?;
        let mut output: Option<Tensor> = None;
        for (index, [up, down]) in self.networks.iter().enumerate() {
            let expert = rows
                .apply(up, state)?
                .apply(&self.activation, state)?
                .apply(down, state)?
                .mul(&weights.narrow(1, i64::try_from(index).unwrap_or(i64::MAX), 1)?)?;
            output = Some(match output {
                Some(output) => output.add(&expert)?,
                None => expert,
            });
        }

        let output = output.ok_or_else(|| SingeError::invalid_argument("Switch experts"))?;
        let shape = (0..rank)
            .map(|axis| input.dimension(axis))
            .collect::<SingeResult<Vec<_>>>()?;
        output.reshape_with(&shape)
    }
}

/// The projections of each attention in a block.
fn attention_spec(config: &T5Config) -> AttentionSpec {
    AttentionSpec {
        width: config.width,
        context: config.width,
        inner: config.head_dim * config.heads,
        heads: config.heads,
        qkv_bias: false,
        output_bias: false,
    }
}

/// A decoder block's attention: causal self-attention over its paged cache, then the norm and cross-attention that read the encoder's states.
#[derive(Debug, Clone)]
struct DecoderAttention {
    attention: GroupedQueryAttention,
    cross_norm: RmsNorm,
    cross: CrossAttention,
}

impl DecoderAttention {
    fn create(schema: &mut Schema<'_>, config: &T5Config) -> SingeResult<Self> {
        let spec = attention_spec(config);
        Ok(Self {
            attention: paged_self_attention(&mut schema.scope("attention"), spec, UNSCALED)?,
            cross_norm: schema.rms_norm("cross_attention_norm", config.width, config.eps)?,
            cross: paged_cross_attention(&mut schema.scope("cross_attention"), spec)?
                .with_scale(1.0),
        })
    }
}

/// A block whose attention is `A`: an encoder's [`Attention`], or a decoder's [`DecoderAttention`].
#[derive(Debug, Clone)]
struct Block<A> {
    attention_norm: RmsNorm,
    attention: A,
    feed_forward_norm: RmsNorm,
    feed_forward: FeedForward,
}

impl<A> Block<A> {
    fn create(
        schema: &mut Schema<'_>,
        config: &T5Config,
        sparse: bool,
        attention: impl FnOnce(&mut Schema<'_>) -> SingeResult<A>,
    ) -> SingeResult<Self> {
        let (width, eps) = (config.width, config.eps);
        let sizes = [width, config.inner, width];
        let feed_forward = if let Some(spec) = config.experts.filter(|_| sparse) {
            FeedForward::Experts(Experts::create(
                &mut schema.scope("feed_forward"),
                config,
                spec,
            )?)
        } else if config.gated {
            FeedForward::Gated(schema.gated_mlp("feed_forward", sizes, config.activation, false)?)
        } else {
            FeedForward::Plain(schema.mlp("feed_forward", sizes, config.activation, false)?)
        };
        Ok(Self {
            attention_norm: schema.rms_norm("attention_norm", width, eps)?,
            attention: attention(schema)?,
            feed_forward_norm: schema.rms_norm("feed_forward_norm", width, eps)?,
            feed_forward,
        })
    }

    /// Returns `hidden` with the feed-forward layer's output on it, normed, added.
    fn add_feed_forward(&self, hidden: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let normed = hidden.apply(&self.feed_forward_norm, state)?;
        hidden.add(&normed.apply(&self.feed_forward, state)?)
    }
}

/// An encoder block's attention, and `LongT5`'s norm of the global tokens it adds to its keys.
#[derive(Debug, Clone)]
struct EncoderAttention {
    attention: Attention,
    global_norm: Option<RmsNorm>,
}

/// What an encoder's attention adds to its scores: T5's relative position bias over every key, `[1, heads, n, n]`, or `LongT5`'s blocked transient-global layout.
#[derive(Debug)]
enum EncoderBias {
    Full(Tensor),
    Transient(TransientLayout),
}

/// `LongT5`'s transient-global attention as `TransientGlobalSelfAttention` lays it out, in memory linear in the tokens: queries in blocks of `radius + 1`, each block reading its own and its two neighbours' keys, then every global token.
#[derive(Debug)]
struct TransientLayout {
    /// The scores' bias, `[1, heads · blocks, block, 3 · block + globals]`: the relative position bias inside the radius and over real keys, then each global token's bias by its distance from the query's global block.
    bias: Tensor,
    heads: i64,
    length: i64,
    block: i64,
    blocks: i64,
    globals: i64,
    global_block: i64,
}

impl TransientLayout {
    /// Returns the global tokens, `[B, G, W]`: each global block's sum of `normed`, `[B, n, W]`, the tokens past the last full block joining it.
    fn global_states(&self, normed: &Tensor) -> SingeResult<Tensor> {
        let &[batch, _, width] = normed.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a LongT5 encoder's states"));
        };
        let full = self.globals * self.global_block;
        let sums = normed
            .narrow(1, 0, full)?
            .reshape([batch, self.globals, self.global_block, width])?
            .sum(&[2], false)?;
        if full == self.length {
            return Ok(sums);
        }

        let tail = normed
            .narrow(1, full, self.length - full)?
            .sum(&[1], true)?;
        Tensor::concat(
            &[
                sums.narrow(1, 0, self.globals - 1)?,
                sums.narrow(1, self.globals - 1, 1)?.add(&tail)?,
            ],
            1,
        )
    }

    /// Attends from `normed`, `[B, n, W]`, to the keys of each query's block window and to `globals`, `[B, G, W]`, through `attention`'s projections.
    fn attend(
        &self,
        attention: &Attention,
        normed: &Tensor,
        globals: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let heads =
            |input: &Tensor, linear: &Linear| split_heads(&input.apply(linear, state)?, self.heads);
        let query = heads(normed, attention.query())?;
        let &[batch, count, _, head] = query.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a LongT5 encoder's queries"));
        };
        let (block, blocks) = (self.block, self.blocks);
        let tail = blocks * block - self.length;
        let padded = |input: &Tensor, before: i64, after: i64| {
            input.pad(
                &[Padding::ZERO, Padding::ZERO, Padding::new(before, after)],
                0.0,
            )
        };
        // Each block's window, its neighbours' keys on either side, then every global token's, `[B, H · blocks, 3 · block + G, d]`.
        let window = |linear: &Linear| -> SingeResult<Tensor> {
            let keys = padded(&heads(normed, linear)?, block, tail + block)?.reshape([
                batch,
                count,
                blocks + 2,
                block,
                head,
            ])?;
            let mut parts = (0..3)
                .map(|offset| keys.narrow(2, offset, blocks))
                .collect::<SingeResult<Vec<_>>>()?;
            if let Some(globals) = globals {
                parts.push(
                    heads(globals, linear)?
                        .unsqueeze(2)?
                        .expand(Shape::create([batch, count, blocks, self.globals, head])?)?,
                );
            }

            let keys = 3 * block + if globals.is_some() { self.globals } else { 0 };
            Tensor::concat(&parts, 3)?.reshape([batch, count * blocks, keys, head])
        };
        let attended = padded(&query, 0, tail)?
            .reshape([batch, count * blocks, block, head])?
            .attend(
                &window(attention.key())?,
                &window(attention.value())?,
                Some(&self.bias),
                None,
                UNSCALED,
            )?
            .reshape([batch, count, blocks * block, head])?
            .narrow(2, 0, self.length)?;
        merge_heads(&attended)?.apply(attention.output(), state)
    }
}

impl Block<EncoderAttention> {
    /// Runs an encoder's block over its whole input, `hidden`, with `bias` added to its attention's scores.
    fn forward(
        &self,
        hidden: &Tensor,
        bias: &EncoderBias,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let normed = hidden.apply(&self.attention_norm, state)?;
        let attention = &self.attention.attention;
        let attended = match (bias, &self.attention.global_norm) {
            (EncoderBias::Transient(layout), Some(norm)) => {
                let globals = (layout.globals > 0)
                    .then(|| layout.global_states(&normed)?.apply(norm, state))
                    .transpose()?;
                layout.attend(attention, &normed, globals.as_ref(), state)?
            }
            (EncoderBias::Full(bias), _) => {
                attention.attend(&normed, &normed, &normed, Some(bias), UNSCALED, state)?
            }
            (EncoderBias::Transient(_), None) => {
                return Err(SingeError::invalid_argument(
                    "a transient-global layout for a block without a global norm",
                ));
            }
        };
        self.add_feed_forward(&hidden.add(&attended)?, state)
    }
}

impl Block<DecoderAttention> {
    /// Runs a decoder's block over a decode step's rows, `hidden`, which read and extend `cache`, with `bias`, `[heads, distances]`, added to each self-attention score by its query's distance from its key.
    fn decode(
        &self,
        hidden: &Tensor,
        step: &DecodeStep,
        bias: &Tensor,
        cache: &EncoderDecoderCache,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let DecoderAttention {
            attention,
            cross_norm,
            cross,
        } = &self.attention;
        let attended = attention.decode_biased(
            &AttentionInput {
                hidden: &hidden.apply(&self.attention_norm, state)?,
                step,
                angles: None,
                cache: &cache.self_attention(),
            },
            bias,
            state,
        )?;
        let hidden = hidden.add(&attended)?;
        let crossed = cross.forward(
            &CrossAttentionInput {
                hidden: &hidden.apply(cross_norm, state)?,
                step,
                keys: cache.cross_keys(),
                values: cache.cross_values(),
                lengths: cache.source_lengths(),
            },
            state,
        )?;
        self.add_feed_forward(&hidden.add(&crossed)?, state)
    }
}

/// An encoder or decoder stack, whose blocks' attention is `A`.
#[derive(Debug, Clone)]
struct Stack<A> {
    blocks: Vec<Block<A>>,
    /// Each relative position bucket's bias for each head, `[buckets, heads]`.
    position_bias: StateKey,
    /// `LongT5`'s bias of each bucket of a global token's block relative to the query's, `[buckets, heads]`.
    global_bias: Option<StateKey>,
    norm: RmsNorm,
}

impl<A> Stack<A> {
    fn create(
        schema: &mut Schema<'_>,
        config: &T5Config,
        role: StackRole,
        attention: impl Fn(&mut Schema<'_>) -> SingeResult<A>,
    ) -> SingeResult<Self> {
        let (layers, step) = match role {
            StackRole::Encoder => (
                config.encoder_layers,
                config.experts.map_or(0, |experts| experts.encoder_step),
            ),
            StackRole::Decoder => (
                config.decoder_layers,
                config.experts.map_or(0, |experts| experts.decoder_step),
            ),
        };
        let blocks = (0..layers)
            .map(|index| {
                Block::create(
                    &mut schema.scope(format!("blocks.{index}")),
                    config,
                    SwitchExperts::is_sparse(step, index),
                    &attention,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let global_bias = (role == StackRole::Encoder && config.transient_global.is_some())
            .then(|| schema.parameter("global_bias", [config.buckets, config.heads]))
            .transpose()?;
        Ok(Self {
            blocks,
            position_bias: schema.parameter("position_bias", [config.buckets, config.heads])?,
            global_bias,
            norm: schema.rms_norm("norm", config.width, config.eps)?,
        })
    }
}

impl Stack<EncoderAttention> {
    /// Returns the bias of an encoder's attention among `length` tokens, `[1, heads, length, length]`.
    fn encoder_bias(
        &self,
        length: i64,
        config: &T5Config,
        device: Device,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let table = state.tensor(&self.position_bias)?;
        let buckets = (0..length)
            .flat_map(|query| {
                (0..length).map(move |key| {
                    bucket(
                        key - query,
                        StackRole::Encoder,
                        config.buckets,
                        config.max_distance,
                    )
                })
            })
            .collect::<Vec<_>>();
        let index = Tensor::from_vec(device, Shape::create([length * length, 1])?, buckets)?
            .expand(Shape::create([length * length, config.heads])?)?;
        table
            .gather(0, &index)?
            .reshape([length, length, config.heads])?
            .permute(&[2, 0, 1])?
            .unsqueeze(0)
    }

    /// Runs the stack as an encoder over `hidden`, its whole input.
    fn forward(
        &self,
        hidden: &Tensor,
        config: &T5Config,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let length = sequence_length(hidden)?;
        let device = hidden.device();
        let bias =
            match (config.transient_global, &self.global_bias) {
                (Some(transient), Some(global_bias)) => EncoderBias::Transient(
                    self.transient_layout(length, transient, global_bias, config, device, state)?,
                ),
                _ => EncoderBias::Full(self.encoder_bias(length, config, device, state)?),
            };
        let mut hidden = hidden.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden, &bias, state)?;
        }
        hidden.apply(&self.norm, state)
    }

    /// Returns `LongT5`'s blocked layout of `length` tokens, as `TransientGlobalSelfAttention` computes it for a sequence without padding.
    /// A token reads the tokens within the radius through the position table and each global token through the global table, by the distance from its own global block; tokens past the last full global block join it.
    fn transient_layout(
        &self,
        length: i64,
        transient: TransientGlobal,
        global_bias: &StateKey,
        config: &T5Config,
        device: Device,
        state: &ModelState,
    ) -> SingeResult<TransientLayout> {
        let heads = config.heads;
        let block = transient.radius + 1;
        let blocks = (length + block - 1) / block;
        let globals = length / transient.block;
        let window = 3 * block;
        let bucket_of = |relative: i64| {
            bucket(
                relative,
                StackRole::Encoder,
                config.buckets,
                config.max_distance,
            )
        };
        // Each bucket's row of `table`, `[count, heads]`.
        let rows = |table: &StateKey, buckets: Vec<i64>| -> SingeResult<Tensor> {
            let count =
                i64::try_from(buckets.len()).map_err(|_| SingeError::out_of_range("buckets"))?;
            let index = Tensor::from_vec(device, Shape::create([count])?, buckets)?;
            state.tensor(table)?.index_select(0, &index)
        };
        let host = |shape: [i64; 5], values: Vec<f32>, like: &Tensor| {
            Tensor::from_vec(device, Shape::create(shape)?, values)?.cast(like.data_type())
        };
        // A block's query `q` reads its window's key `k`, `block` before the block's start, at the relative position `k - block - q`.
        let local = rows(
            &self.position_bias,
            (0..block)
                .flat_map(|query| (0..window).map(move |key| bucket_of(key - block - query)))
                .collect(),
        )?
        .reshape([block, window, heads])?
        .permute(&[2, 0, 1])?
        .reshape([1, heads, 1, block, window])?;
        let inside = host(
            [1, 1, 1, block, window],
            (0..block)
                .flat_map(|query| {
                    (0..window).map(move |key| {
                        if (key - block - query).abs() <= transient.radius {
                            0.0
                        } else {
                            OUTSIDE
                        }
                    })
                })
                .collect(),
            &local,
        )?;
        let real = host(
            [1, 1, blocks, 1, window],
            (0..blocks)
                .flat_map(|index| {
                    (0..window).map(move |key| {
                        if (0..length).contains(&(index * block + key - block)) {
                            0.0
                        } else {
                            OUTSIDE
                        }
                    })
                })
                .collect(),
            &local,
        )?;
        let local = local.add(&inside)?.add(&real)?;
        let bias = if globals > 0 {
            // Each distance from the query's global block to a global token, from `-(G - 1)` to `G - 1`, and the row of it each query reads for each global token, `[n, G]`.
            let distances = rows(global_bias, (1 - globals..globals).map(bucket_of).collect())?;
            let offsets = Tensor::from_vec(
                device,
                Shape::create([length, 1])?,
                (0..length)
                    .map(|token| globals - 1 - (token / transient.block).min(globals - 1))
                    .collect(),
            )?;
            let index = Tensor::range(device, 0_i64, globals)?
                .unsqueeze(0)?
                .add(&offsets)?
                .reshape([length * globals])?;
            let side = distances
                .index_select(0, &index)?
                .reshape([length, globals, heads])?
                .permute(&[2, 0, 1])?
                .pad(
                    &[Padding::ZERO, Padding::new(0, blocks * block - length)],
                    0.0,
                )?
                .reshape([1, heads, blocks, block, globals])?;
            Tensor::concat(&[local, side], 4)?
        } else {
            local
        };
        let keys = window + globals;
        Ok(TransientLayout {
            bias: bias.reshape([1, heads * blocks, block, keys])?,
            heads,
            length,
            block,
            blocks,
            globals,
            global_block: transient.block,
        })
    }
}

impl Stack<DecoderAttention> {
    /// Returns each head's bias by a decoder query's distance from its key, `[heads, max_distance + 1]`.
    /// Every distance from `max_distance` on falls in the last bucket, so longer ones share the last column.
    fn distance_bias(
        &self,
        config: &T5Config,
        device: Device,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let distances = config.max_distance + 1;
        let buckets = capture::literal(device, |device| {
            Tensor::from_vec(
                device,
                Shape::create([distances])?,
                (0..distances)
                    .map(|distance| {
                        bucket(
                            -distance,
                            StackRole::Decoder,
                            config.buckets,
                            config.max_distance,
                        )
                    })
                    .collect(),
            )
        })?;
        state
            .tensor(&self.position_bias)?
            .index_select(0, &buckets)?
            .transpose(0, 1)
    }
}

fn sequence_length(hidden: &Tensor) -> SingeResult<i64> {
    let &[_, length, _] = hidden.shape().dimensions() else {
        return Err(SingeError::invalid_argument("a T5 stack's input"));
    };
    Ok(length)
}

/// T5's decoder and its output, embedding tokens with the embedding the encoder shares.
#[derive(Debug, Clone)]
struct T5Decoder {
    tokens: Embedding,
    stack: Stack<DecoderAttention>,
    head: Linear,
    config: T5Config,
}

impl Seq2SeqDecoding for T5Decoder {
    fn cross_attentions(&self) -> Vec<&CrossAttention> {
        self.stack
            .blocks
            .iter()
            .map(|block| &block.attention.cross)
            .collect()
    }

    fn cache_spec(&self) -> KvCacheSpec {
        KvCacheSpec {
            kv_heads: self.config.heads,
            head_dim: self.config.head_dim,
        }
    }

    fn forward(
        &self,
        step: &DecodeStep,
        caches: &[EncoderDecoderCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        if caches.len() != self.stack.blocks.len() {
            return Err(SingeError::missing_input("a decoder block's cache"));
        }

        let bias = self
            .stack
            .distance_bias(&self.config, step.tokens().device(), state)?;
        let mut hidden = step.tokens().apply(&self.tokens, state)?;
        for (block, cache) in self.stack.blocks.iter().zip(caches) {
            hidden = block.decode(&hidden, step, &bias, cache, state)?;
        }

        let hidden = hidden
            .apply(&self.stack.norm, state)?
            .index_select(0, step.last_rows())?;
        let hidden = if self.config.tied {
            hidden.mul_scalar((self.config.width as f64).powf(-0.5))?
        } else {
            hidden
        };
        hidden.apply(&self.head, state)
    }
}

/// T5's encoder alone, its token embeddings and encoder stack, which `MusicGen` reads its text with.
#[derive(Debug, Clone)]
pub struct T5Encoder {
    tokens: Embedding,
    stack: Stack<EncoderAttention>,
    config: T5Config,
}

impl T5Encoder {
    /// Registers the encoder stack as `encoder` in `schema`, reading `tokens`.
    pub fn create(
        schema: &mut Schema<'_>,
        config: &T5Config,
        tokens: Embedding,
    ) -> SingeResult<Self> {
        Ok(Self {
            tokens,
            stack: Stack::create(
                &mut schema.scope("encoder"),
                config,
                StackRole::Encoder,
                |schema| {
                    Ok(EncoderAttention {
                        attention: Attention::create(
                            &mut schema.scope("attention"),
                            attention_spec(config),
                        )?,
                        global_norm: config
                            .transient_global
                            .map(|_| schema.rms_norm("global_norm", config.width, config.eps))
                            .transpose()?,
                    })
                },
            )?,
            config: config.clone(),
        })
    }

    /// Returns each token's final state, `[1, T, W]`, for `tokens`, `[1, T]`.
    pub fn forward(&self, tokens: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = tokens.apply(&self.tokens, state)?;
        self.stack.forward(&hidden, &self.config, state)
    }
}

#[derive(Debug, Clone)]
pub struct T5Layers {
    encoder: T5Encoder,
    decoder: Arc<dyn Seq2SeqDecoding>,
}

impl T5Layers {
    pub fn create(schema: &mut Schema<'_>, config: &T5Config) -> SingeResult<Self> {
        let embedding = schema.parameter("tokens.weight", [config.vocab_size, config.width])?;
        let head = if config.tied {
            Linear::new(schema.tie("head.weight", &embedding)?, None)
        } else {
            schema.linear("head", config.width, config.vocab_size, false)?
        };
        let tokens = Embedding::new(embedding);
        Ok(Self {
            encoder: T5Encoder::create(schema, config, tokens.clone())?,
            decoder: Arc::new(T5Decoder {
                tokens,
                stack: Stack::create(
                    &mut schema.scope("decoder"),
                    config,
                    StackRole::Decoder,
                    |schema| DecoderAttention::create(schema, config),
                )?,
                head,
                config: config.clone(),
            }),
        })
    }
}

impl Seq2SeqLayers for T5Layers {
    fn encode(&self, input: Seq2SeqInput<'_>, state: &ModelState) -> SingeResult<Tensor> {
        let Seq2SeqInput::Tokens(batch) = input else {
            return Err(SingeError::invalid_argument("T5 reads tokens"));
        };
        self.encoder.forward(batch.tokens(), state)
    }

    fn decoder(&self) -> Arc<dyn Seq2SeqDecoding> {
        Arc::clone(&self.decoder)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "t5",
    formats: &[(
        Format::Transformers,
        &["t5", "switch_transformers", "longt5"],
    )],
    loaders: &[HeadLoader::Seq2SeqLm(load_seq2seq)],
};

pub fn load_seq2seq(loader: &mut Loader<'_>) -> Result<Seq2SeqLm> {
    let config: T5Config = loader.config(transformers::config)?;
    let generation = seq2seq_generation(loader.source())?;
    let model = Schema::model(loader.options().data_type, |schema| {
        T5Layers::create(schema, &config).map(|layers| Boxed::<dyn Seq2SeqLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(config.tied))?;
    Ok(Seq2SeqLm::new(model, generation))
}
