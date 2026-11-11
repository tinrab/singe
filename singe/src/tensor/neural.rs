use crate::{
    backend::{
        decomposition,
        registry::{
            BatchNormalizationAttributes, Conv2dAttributes, LayerNormalizationAttributes,
            Operation, OperationAttributes, Pool2dAttributes, ScaledDotProductAttentionAttributes,
        },
    },
    data_type::DataType,
    device::Device,
    dispatch::{call, call_one},
    error::{Error, Result},
    graph::capture,
    scalar::Scalar,
    shape::Shape,
    tensor::Tensor,
    utility::to_usize,
};

/// Returns the present tensors among `inputs`, in order.
fn present<'a>(inputs: &[Option<&'a Tensor>]) -> Vec<&'a Tensor> {
    inputs.iter().flatten().copied().collect()
}

// How `Tensor::attend` scores and masks keys: the scale, which defaults to `1 / sqrt(head_dim)`, a causal mask, a sliding window of the keys each query sees up to itself, and a cap on the scaled scores.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AttentionOptions {
    pub scale: Option<f32>,
    pub causal: bool,
    pub window: Option<i64>,
    pub softcap: Option<f32>,
}

impl Tensor {
    // The additive mask of attention from `query_len` queries to `key_len` keys on `device`: negative infinity where key `j` is hidden from query `i`, after it under a causal mask or `window` or more positions before it, and zero elsewhere.
    pub fn attention_mask(
        device: Device,
        data_type: DataType,
        query_len: i64,
        key_len: i64,
        causal: bool,
        window: Option<i64>,
    ) -> Result<Self> {
        let shape = Shape::create([query_len, key_len])?;
        let columns = to_usize(key_len, "attention key length")?;
        let window = window
            .map(|window| to_usize(window, "attention window"))
            .transpose()?;
        let values = (0..shape.element_count())
            .map(|index| {
                let (query, key) = (index / columns.max(1), index % columns.max(1));
                let hidden =
                    (causal && key > query) || window.is_some_and(|window| key + window <= query);
                if hidden { f32::NEG_INFINITY } else { 0.0 }
            })
            .collect::<Vec<_>>();
        let mask = Self::from_vec(device, &shape, values)?;
        if data_type == DataType::F32 {
            Ok(mask)
        } else {
            mask.cast(data_type)
        }
    }

    /// Applies rotary position embedding) to the first `2 * cos.shape()[1]` features of a `[batch, heads, sequence, head_dim]` tensor.
    ///
    /// `cos` and `sin` must be rank-2 F32 tensors shaped `[sequence, rotary_dim / 2]`.
    /// Features beyond `rotary_dim` are copied unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if shapes, data types, devices, or backend execution are incompatible.
    pub fn rotary_embedding(&self, cos: &Self, sin: &Self) -> Result<Self> {
        call_one(
            Operation::RotaryEmbedding,
            OperationAttributes::None,
            &[self, cos, sin],
        )
    }

    /// Applies scaled dot-product attention to rank-4 F32 tensors.
    ///
    /// `self` is the query tensor shaped `[batch, heads, query, head_dim]`, `key` is shaped `[batch, heads, key, head_dim]`, and `value` is shaped `[batch, heads, key, value_dim]`.
    /// The output shape is `[batch, heads, query, value_dim]`.
    ///
    /// If `scale` is `None`, `1 / sqrt(head_dim)` is used.
    /// When `causal` is true, query positions cannot attend to later key positions.
    ///
    /// # Errors
    ///
    /// Returns an error if shapes, data types, devices, or backend execution are incompatible.
    pub fn scaled_dot_product_attention(
        &self,
        key: &Self,
        value: &Self,
        scale: Option<f32>,
        causal: bool,
    ) -> Result<Self> {
        self.attend(
            key,
            value,
            None,
            None,
            AttentionOptions {
                scale,
                causal,
                ..AttentionOptions::default()
            },
        )
    }

    // Scaled dot-product attention of this query tensor over `key` and `value` under `options`.
    // A `mask` broadcast to the scores' `[B, H, L, S]` adds to them after any cap, as PyTorch's `attn_mask` does: a Boolean mask lets the keys where it holds take part, and hides the others.
    // `sinks`, one logit per query head, join each row's softmax without weighing a value, as GPT-OSS's attention sinks do.
    pub fn attend(
        &self,
        key: &Self,
        value: &Self,
        mask: Option<&Self>,
        sinks: Option<&Self>,
        options: AttentionOptions,
    ) -> Result<Self> {
        self.attend_all(key, value, mask, sinks, None, options)
    }

    // Scaled dot-product attention of a padded batch: each batch `b` of this `[B, H, L, D]` query attends to its first `key_lengths[b]` keys, `[B]` of `I32`, as a batch padded to one length would, and a batch of no keys gives zeros.
    // Libraries that take sequence lengths, as cuDNN's attention does, read them as they are, so no layer builds a padding mask.
    pub fn attend_padded(
        &self,
        key: &Self,
        value: &Self,
        key_lengths: &Self,
        options: AttentionOptions,
    ) -> Result<Self> {
        self.attend_all(key, value, None, None, Some(key_lengths), options)
    }

    fn attend_all(
        &self,
        key: &Self,
        value: &Self,
        mask: Option<&Self>,
        sinks: Option<&Self>,
        key_lengths: Option<&Self>,
        options: AttentionOptions,
    ) -> Result<Self> {
        let mask = mask
            .map(|mask| {
                if mask.data_type() == DataType::Bool {
                    let additive = |value: f64| {
                        capture::literal(mask.device(), |device| {
                            Scalar::Float(value).tensor(device, self.data_type())
                        })
                    };
                    Self::where_select(mask, &additive(0.0)?, &additive(f64::NEG_INFINITY)?)
                } else {
                    mask.cast(self.data_type())
                }
            })
            .transpose()?;
        let inputs = [self, key, value]
            .into_iter()
            .chain(mask.as_ref())
            .chain(sinks)
            .chain(key_lengths)
            .collect::<Vec<_>>();
        call_one(
            Operation::ScaledDotProductAttention,
            OperationAttributes::ScaledDotProductAttention(ScaledDotProductAttentionAttributes {
                scale: options.scale,
                causal: options.causal,
                window: options.window,
                softcap: options.softcap,
                masked: mask.is_some(),
                sinks: sinks.is_some(),
                key_lengths: key_lengths.is_some(),
            }),
            &inputs,
        )
    }
}

impl Tensor {
    /// Applies 2-D convolution using NCHW input and OIHW weights.
    ///
    /// This records a graph operation when called inside graph capture.
    /// CUDA accepts matching FP16, BF16, FP32 or FP64 inputs and weights, with an independently typed floating bias.
    /// Half and single-precision convolutions compute in FP32; FP64 convolutions compute in FP64.
    ///
    /// # Errors
    ///
    /// Returns an error if shapes, parameters, data types, devices, or backend capabilities are incompatible.
    pub fn convolution2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (i64, i64),
        padding: (i64, i64, i64, i64),
        dilation: (i64, i64),
        groups: i64,
    ) -> Result<Self> {
        call_one(
            Operation::Conv2d,
            OperationAttributes::Conv2d(Conv2dAttributes {
                has_bias: bias.is_some(),
                stride,
                padding,
                dilation,
                groups,
            }),
            &present(&[Some(self), Some(weight), bias]),
        )
    }
}

impl Tensor {
    /// Looks up rows from a rank-2 table using nonnegative I64 indices.
    /// The result appends the embedding width to the index tensor's shape.
    ///
    /// This records a graph operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if the table, indices, or backend capabilities are incompatible.
    pub fn embedding(&self, indices: &Self) -> Result<Self> {
        call_one(
            Operation::Embedding,
            OperationAttributes::None,
            &[self, indices],
        )
    }
}

macro_rules! tensor_standard_normalization_forwarders {
    ($($method:ident => $operation:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Applies `", stringify!($method), "` over `normalized_shape`.")]
            ///
            /// # Errors
            ///
            /// Returns an error if parameter tensors have incompatible shape, data type, or device, or if backend execution fails.
            pub fn $method(
                &self,
                normalized_shape: &[i64],
                weight: Option<&Tensor>,
                bias: Option<&Tensor>,
                eps: f32,
            ) -> Result<Self> {
                call_one(
                    Operation::$operation,
                    OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
                        normalized_shape: normalized_shape.to_vec(),
                        has_weight: weight.is_some(),
                        has_bias: bias.is_some(),
                        eps,
                    }),
                    &present(&[Some(self), weight, bias]),
                )
            }
        )+
    };
}

impl Tensor {
    /// Applies batch normalization.
    ///
    /// # Errors
    ///
    /// Returns an error if parameter tensors have incompatible shape, data type, or device, or if backend execution fails.
    pub fn batch_norm(
        &self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        running_mean: Option<&Self>,
        running_var: Option<&Self>,
        training: bool,
        momentum: f32,
        eps: f32,
    ) -> Result<Self> {
        // Training updates the running statistics in place, as recorded writes the operation itself never makes.
        if training && (running_mean.is_some() || running_var.is_some()) {
            return decomposition::batch_norm_training(
                self,
                weight,
                bias,
                running_mean,
                running_var,
                momentum,
                eps,
            );
        }
        call_one(
            Operation::BatchNorm,
            OperationAttributes::BatchNormalization(BatchNormalizationAttributes {
                has_weight: weight.is_some(),
                has_bias: bias.is_some(),
                has_running_mean: running_mean.is_some(),
                has_running_var: running_var.is_some(),
                training,
                momentum,
                eps,
            }),
            &present(&[Some(self), weight, bias, running_mean, running_var]),
        )
    }

    tensor_standard_normalization_forwarders!(
        layer_norm => LayerNorm,
        rms_norm => RmsNorm,
    );
}

macro_rules! tensor_pool2d_forwarders {
    ($($method:ident => $operation:ident [$($include_pad:ident)?]),+ $(,)?) => {
        $(
            #[doc = concat!("Applies `", stringify!($method), "`.")]
            pub fn $method(
                &self,
                kernel_size: (i64, i64),
                stride: (i64, i64),
                padding: (i64, i64, i64, i64),
                dilation: (i64, i64),
                ceil_mode: bool
                $(, $include_pad: bool)?
            ) -> Result<Self> {
                call_one(
                    Operation::$operation,
                    OperationAttributes::Pool2d(Pool2dAttributes {
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        ceil_mode,
                        count_include_pad: None $(.or(Some($include_pad)))?,
                    }),
                    &[self],
                )
            }
        )+
    };
}

impl Tensor {
    tensor_pool2d_forwarders!(
        max_pool2d => MaxPool2d [],
        avg_pool2d => AvgPool2d [count_include_pad],
    );
}

// Normalizations composed from layer normalization, so they run on its row kernel.
impl Tensor {
    // Normalizes each of `groups` groups of channels of an `[N, C, *]` tensor over its channels and spatial points, then applies the per-channel `weight` and `bias`.
    pub fn group_norm(
        &self,
        groups: i64,
        weight: Option<&Self>,
        bias: Option<&Self>,
        eps: f32,
    ) -> Result<Self> {
        let dimensions = self.shape().dimensions();
        let [batch, channels, ..] = *dimensions else {
            return Err(Error::invalid_argument(
                "group normalization needs batch and channel dimensions",
            ));
        };
        if groups <= 0 || channels % groups != 0 {
            return Err(Error::invalid_argument(
                "group normalization's groups must divide the channels",
            ));
        }

        let points = dimensions[2..].iter().product::<i64>() * (channels / groups);
        let mut normalized = self
            .reshape([batch, groups, points])?
            .layer_norm(&[points], None, None, eps)?
            .reshape(dimensions.to_vec())?;
        // The affine parameters hold one value per channel, broadcast over the spatial points.
        let mut channel = vec![1; dimensions.len() - 1];
        channel[0] = channels;
        if let Some(weight) = weight {
            normalized = normalized.mul(&weight.reshape(channel.clone())?)?;
        }

        if let Some(bias) = bias {
            normalized = normalized.add(&bias.reshape(channel)?)?;
        }
        Ok(normalized)
    }

    // Normalizes each channel of each sample over its spatial points.
    pub fn instance_norm(
        &self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        eps: f32,
    ) -> Result<Self> {
        let channels = *self
            .shape()
            .dimensions()
            .get(1)
            .ok_or_else(|| Error::invalid_argument("instance normalization needs channels"))?;
        self.group_norm(channels, weight, bias, eps)
    }
}

// Where ragged queries and their keys sit in a paged cache: sequence `b` holds queries `query_offsets[b]..query_offsets[b + 1]` and the keys of pages `pages[page_offsets[b]..page_offsets[b + 1]]`, all full but the last, which holds `last_page_lengths[b]`.
// The tensors are `I32` or `I64` vectors.
#[derive(Debug, Clone, Copy)]
pub struct Paging<'a> {
    pub query_offsets: &'a Tensor,
    pub page_offsets: &'a Tensor,
    pub pages: &'a Tensor,
    pub last_page_lengths: &'a Tensor,
}

impl Tensor {
    // Attends ragged queries `[T, H, D]` over paged key and value caches `[P, S, G, D]` of `P` pages of `S` slots, as FlashInfer batches decode and prefill.
    // Its queries are its last tokens, so a causal mask lets each see the keys up to its own position; a query that sees no key yields zeros.
    // Also returns each row's base-two log-sum-exp of its scores, `[T, H]` in `F32`, negative infinity for a row that sees no key.
    pub fn paged_attention(
        &self,
        key_cache: &Self,
        value_cache: &Self,
        paging: Paging<'_>,
        options: AttentionOptions,
    ) -> Result<(Self, Self)> {
        let mut outputs = call(
            Operation::PagedAttention,
            OperationAttributes::ScaledDotProductAttention(ScaledDotProductAttentionAttributes {
                scale: options.scale,
                causal: options.causal,
                window: options.window,
                softcap: options.softcap,
                masked: false,
                sinks: false,
                key_lengths: false,
            }),
            &[
                self,
                key_cache,
                value_cache,
                paging.query_offsets,
                paging.page_offsets,
                paging.pages,
                paging.last_page_lengths,
            ],
        )?;
        let lse = outputs
            .pop()
            .ok_or_else(|| Error::missing_output("paged attention"))?;
        let output = outputs
            .pop()
            .ok_or_else(|| Error::missing_output("paged attention"))?;
        Ok((output, lse))
    }
}

impl Tensor {
    // Attends DeepSeek's absorbed latent queries, this tensor `[T, H, C]` and its positional part `[T, H, R]`, over a paged latent cache `[P, S, C]` and positional cache `[P, S, R]` that every head shares, paged as `paged_attention` pages its caches.
    // A head's scores are its latent query against the latent cache plus its positional query against the positional cache, and its output weighs the latent cache; the scale defaults to `1 / sqrt(C + R)`.
    // Returns the output `[T, H, C]` and each row's base-two log-sum-exp of its scaled scores `[T, H]` in `F32`, as FlashInfer's MLA kernels return them.
    pub fn paged_mla_attention(
        &self,
        query_positions: &Self,
        latent_cache: &Self,
        position_cache: &Self,
        paging: Paging<'_>,
        options: AttentionOptions,
    ) -> Result<(Self, Self)> {
        let mut outputs = call(
            Operation::PagedMlaAttention,
            OperationAttributes::ScaledDotProductAttention(ScaledDotProductAttentionAttributes {
                scale: options.scale,
                causal: options.causal,
                window: options.window,
                softcap: options.softcap,
                masked: false,
                sinks: false,
                key_lengths: false,
            }),
            &[
                self,
                query_positions,
                latent_cache,
                position_cache,
                paging.query_offsets,
                paging.page_offsets,
                paging.pages,
                paging.last_page_lengths,
            ],
        )?;
        let lse = outputs
            .pop()
            .ok_or_else(|| Error::missing_output("paged mla attention"))?;
        let output = outputs
            .pop()
            .ok_or_else(|| Error::missing_output("paged mla attention"))?;
        Ok((output, lse))
    }
}

// Where each sequence's rows of a step sit, for operations that carry a state per sequence: sequence `b` holds rows `query_offsets[b]..query_offsets[b + 1]`, of which the first `real_tokens[b]` are its own, and `positions` gives each row's position, so a sequence whose first row is at position zero starts from a zero state.
// The tensors are `I64` vectors.
#[derive(Debug, Clone, Copy)]
pub struct SequenceSpans<'a> {
    pub query_offsets: &'a Tensor,
    pub real_tokens: &'a Tensor,
    pub positions: &'a Tensor,
}

impl Tensor {
    // Convolves each sequence's real rows of `self`, `[T, C]`, with per-channel taps `weight`, `[C, K]`, after the `K − 1` inputs `state` holds for it, `[S, C, K − 1]`, or zeros for a sequence starting at position zero; rows that are not a sequence's own are zero.
    pub fn causal_conv1d(
        &self,
        weight: &Self,
        state: &Self,
        spans: SequenceSpans<'_>,
    ) -> Result<Self> {
        call_one(
            Operation::CausalConv1d,
            OperationAttributes::None,
            &[
                self,
                weight,
                state,
                spans.query_offsets,
                spans.real_tokens,
                spans.positions,
            ],
        )
    }

    // Returns `state` as `causal_conv1d` leaves it: each sequence's last `K − 1` inputs, after its real rows of `self`.
    pub fn causal_conv1d_state(&self, state: &Self, spans: SequenceSpans<'_>) -> Result<Self> {
        call_one(
            Operation::CausalConv1dState,
            OperationAttributes::None,
            &[
                self,
                state,
                spans.query_offsets,
                spans.real_tokens,
                spans.positions,
            ],
        )
    }

    // Runs the gated delta rule over each sequence's real rows, from `state`, `[S, H, Dk, Dv]` in `F32`, or zeros for a sequence starting at position zero.
    // For each row in order, with `self` the queries `[T, H, Dk]`, the state decays by `exp(decay)`, takes `strength · (value − read)` at the key, where `read` is what it held there, and is read at the query; the result is `[T, H, Dv]`, and rows that are not a sequence's own are zero.
    // Queries and keys arrive as the caller wants them read, so any normalization and scale come first.
    pub fn gated_delta_rule(
        &self,
        key: &Self,
        value: &Self,
        decay: &Self,
        strength: &Self,
        state: &Self,
        spans: SequenceSpans<'_>,
    ) -> Result<Self> {
        call_one(
            Operation::GatedDeltaRule,
            OperationAttributes::None,
            &[
                self,
                key,
                value,
                decay,
                strength,
                state,
                spans.query_offsets,
                spans.real_tokens,
                spans.positions,
            ],
        )
    }

    // Returns `state` as `gated_delta_rule` leaves it.
    pub fn gated_delta_rule_state(
        &self,
        key: &Self,
        value: &Self,
        decay: &Self,
        strength: &Self,
        state: &Self,
        spans: SequenceSpans<'_>,
    ) -> Result<Self> {
        call_one(
            Operation::GatedDeltaRuleState,
            OperationAttributes::None,
            &[
                self,
                key,
                value,
                decay,
                strength,
                state,
                spans.query_offsets,
                spans.real_tokens,
                spans.positions,
            ],
        )
    }
}

impl Tensor {
    // Multiplies rows `[M, K]` sorted by group by their group's weight of `weights` `[G, K, N]`, into `[M, N]`: group `g` is rows `offsets[g]..offsets[g + 1]` of the `G + 1` `I32` or `I64` offsets, and rows outside every group are zero.
    pub fn grouped_matmul(&self, weights: &Self, offsets: &Self) -> Result<Self> {
        call_one(
            Operation::GroupedMatMul,
            OperationAttributes::None,
            &[self, weights, offsets],
        )
    }

    // A mixture-of-experts layer over tokens `[T, H]`, each routed to the `k` experts `expert_ids` `[T, k]` names with the weights `expert_weights` `[T, k]`, as the router chose them.
    // Expert `e` multiplies a token by `gate_up[e]` `[2I, H]`, takes the SiLU of the first half times the second, multiplies that by `down[e]` `[H, I]`, and each token sums its experts' outputs by their weights.
    // Tokens are sorted by expert, so each expert's weights are read once for all its tokens, in two grouped matmuls.
    pub fn moe(
        &self,
        expert_ids: &Self,
        expert_weights: &Self,
        gate_up: &Self,
        down: &Self,
    ) -> Result<Self> {
        let (&[tokens, hidden], &[_, chosen], &[experts, doubled, _], &[_, _, intermediate]) = (
            self.shape().dimensions(),
            expert_ids.shape().dimensions(),
            gate_up.shape().dimensions(),
            down.shape().dimensions(),
        ) else {
            return Err(Error::invalid_argument(
                "moe needs [T, H] tokens, [T, k] experts and weights, [E, 2I, H] and [E, H, I] weights",
            ));
        };
        if doubled != 2 * intermediate {
            return Err(Error::invalid_argument(
                "moe gate and up projections need twice the down projection's width",
            ));
        }

        let routes = tokens * chosen;
        let flat = expert_ids.reshape([routes])?;
        let order = flat.argsort(0, false)?;
        let token = capture::literal(self.device(), |device| {
            Self::arange(device, 0_i64, tokens, 1)
        })?
        .repeat_interleave(chosen, 0)?
        .index_select(0, &order)?;
        let offsets = flat
            .one_hot(experts)?
            .sum(&[0], false)?
            .cumsum(0)?
            .pad(&[(1, 0)], 0.0)?;
        let projected = self
            .index_select(0, &token)?
            .grouped_matmul(&gate_up.transpose(1, 2)?, &offsets)?;
        let activated = projected
            .narrow(1, 0, intermediate)?
            .silu()?
            .mul(&projected.narrow(1, intermediate, intermediate)?)?;
        let weights = expert_weights
            .reshape([routes])?
            .index_select(0, &order)?
            .unsqueeze(1)?;
        // A half-precision output meets its weight in the weight's type, and rounds once, as the references do.
        let expert_outputs = activated
            .grouped_matmul(&down.transpose(1, 2)?, &offsets)?
            .cast(weights.data_type())?
            .mul(&weights)?
            .cast(self.data_type())?;
        let shape = Shape::create([tokens, hidden])?;
        let data_type = self.data_type();
        capture::literal(self.device(), |device| {
            Scalar::Float(0.0).full(device, data_type, &shape)
        })?
        .index_add(0, &token, &expert_outputs)
    }
}

impl Tensor {
    // DeepSeek-V3's routing without auxiliary loss, over router logits `[T, E]`: each expert's score is the sigmoid of its logit, and `bias` shifts the scores used for choosing only.
    // The experts form `groups` equal groups; a token keeps the `kept_groups` groups whose two best biased scores sum highest, and chooses its `top_k` best biased experts among them.
    // Returns the chosen experts `[T, top_k]` in `I64`, and their unbiased scores, normalized to sum to `scale` per token, in `F32`.
    pub fn grouped_topk_routing(
        &self,
        bias: &Self,
        groups: i64,
        kept_groups: i64,
        top_k: i64,
        scale: f64,
    ) -> Result<(Self, Self)> {
        let &[tokens, experts] = self.shape().dimensions() else {
            return Err(Error::invalid_argument("routing needs [T, E] logits"));
        };
        if groups <= 0 || experts % groups != 0 {
            return Err(Error::invalid_argument(
                "routing groups must divide the experts",
            ));
        }

        let per_group = experts / groups;
        let scores = self.cast(DataType::F32)?.sigmoid()?;
        let biased = scores.add(&bias.cast(DataType::F32)?)?;
        let (best, _) =
            biased
                .reshape([tokens, groups, per_group])?
                .topk(2.min(per_group), 2, true, false)?;
        let (_, kept) = best.sum(&[2], false)?.topk(kept_groups, 1, true, false)?;
        let mask = kept
            .one_hot(groups)?
            .sum(&[1], false)?
            .cast(DataType::Bool)?
            .unsqueeze(2)?
            .expand(&Shape::create([tokens, groups, per_group])?)?
            .reshape([tokens, experts])?;
        let shape = Shape::create([tokens, experts])?;
        let lowest = capture::literal(self.device(), |device| {
            Scalar::Float(f64::from(f32::MIN)).full(device, DataType::F32, &shape)
        })?;
        let (_, ids) = Self::where_select(&mask, &biased, &lowest)?.topk(top_k, 1, true, true)?;
        let chosen = scores.gather(1, &ids)?;
        let total = chosen.sum(&[1], true)?.add_scalar(1e-20)?;
        Ok((ids, chosen.div(&total)?.mul_scalar(scale)?))
    }

    // Block-scaled values, as DeepSeek stores FP8 weights and activations: each `block.0` by `block.1` block of this tensor's last two axes times its scale, whose last two axes count the blocks, in `F32`.
    pub fn dequantize_blocks(&self, scales: &Self, block: (i64, i64)) -> Result<Self> {
        let dimensions = self.shape().dimensions();
        let rank = dimensions.len();
        if rank < 2 || scales.shape().dimensions().len() != rank {
            return Err(Error::invalid_argument(
                "block dequantization needs values and scales of one rank of at least two",
            ));
        }

        let (rows, columns) = (dimensions[rank - 2], dimensions[rank - 1]);
        let (row_axis, column_axis) = (
            i64::try_from(rank - 2).map_err(|_| Error::DimensionOutOfBounds)?,
            i64::try_from(rank - 1).map_err(|_| Error::DimensionOutOfBounds)?,
        );
        let expanded = scales
            .cast(DataType::F32)?
            .repeat_interleave(block.0, row_axis)?
            .repeat_interleave(block.1, column_axis)?
            .narrow(row_axis, 0, rows)?
            .narrow(column_axis, 0, columns)?;
        self.cast(DataType::F32)?.mul(&expanded)
    }
}
