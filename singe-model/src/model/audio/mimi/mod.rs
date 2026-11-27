//! Mimi: a `SEANet` encoder without an LSTM, a causal transformer over its frames with a sliding window, a strided convolution to half the frame rate, and a split residual quantizer, whose one semantic and many acoustic codebooks each quantize the same latents; the decoder mirrors it.
//! The transformers' blocks norm before attention and the feed-forward and scale each branch by a learned vector before adding it back.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    graph::{ComputePrecision, with_convolution_precision},
    nn::{
        Activation, Conv1d, Conv1dConfig, Layer, LayerNorm, LayerScale, Linear, Mlp, ModelState,
        RotaryFrequencies, RotaryLayout, RotarySpec, StateKey,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{PadMode, PaddedConv, Seanet, SeanetConfig, TrimmedTranspose},
    error::Result,
    format::Loader,
    head::{AudioCodec, Boxed, CodecLayers, HeadLoader},
    model::{ModelEntry, audio::quantizer},
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodecTransformerConfig {
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    pub theta: f64,
    /// The keys each query attends to, counting itself.
    pub window: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MimiConfig {
    pub sample_rate: i64,
    pub seanet: SeanetConfig,
    pub transformer: CodecTransformerConfig,
    /// The stride from the `SEANet`'s frame rate to the quantizer's.
    pub latent_stride: i64,
    pub upsample_groups: i64,
    pub codebook_size: i64,
    pub codebook_dim: i64,
    pub semantic_codebooks: i64,
    pub codebooks: i64,
}

/// A transformer block: windowed causal self-attention with rotary positions, then a feed-forward, each normed before and scaled after.
#[derive(Debug, Clone)]
struct Block {
    attention_norm: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    attention_scale: LayerScale,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_scale: LayerScale,
}

impl Block {
    fn create(schema: &mut Schema<'_>, config: &CodecTransformerConfig) -> SingeResult<Self> {
        let (width, eps) = (config.width, config.eps);
        let inner = config.heads * config.head_dim;
        Ok(Self {
            attention_norm: schema.layer_norm("attention_norm", width, eps)?,
            query: schema.linear("attention.query", width, inner, false)?,
            key: schema.linear("attention.key", width, inner, false)?,
            value: schema.linear("attention.value", width, inner, false)?,
            output: schema.linear("attention.output", inner, width, false)?,
            attention_scale: LayerScale::new(schema.parameter("attention_scale", [width])?),
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.inner, width],
                config.activation,
                false,
            )?,
            feed_forward_scale: LayerScale::new(schema.parameter("feed_forward_scale", [width])?),
        })
    }
}

/// A codec's transformer, from frames, `[B, T, W]`, to as many frames.
#[derive(Debug, Clone)]
struct CodecTransformer {
    blocks: Vec<Block>,
    config: CodecTransformerConfig,
}

impl CodecTransformer {
    fn create(schema: &mut Schema<'_>, config: &CodecTransformerConfig) -> SingeResult<Self> {
        Ok(Self {
            blocks: (0..config.layers)
                .map(|index| Block::create(&mut schema.scope(format!("layers.{index}")), config))
                .collect::<SingeResult<Vec<_>>>()?,
            config: config.clone(),
        })
    }
}

impl Layer for CodecTransformer {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let config = &self.config;
        let &[batch, length, _] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "codec transformer frames, [B, T, W]",
            ));
        };
        let positions = Tensor::range(input.device(), 0_i64, length)?;
        let angles = RotarySpec {
            head_dim: config.head_dim,
            frequencies: RotaryFrequencies::new(config.theta),
            layout: RotaryLayout::HalfSplit,
        }
        .angles(&positions, input.data_type())?;
        let heads = |input: &Tensor, linear: &Linear| {
            input
                .apply(linear, state)?
                .reshape([batch, length, config.heads, config.head_dim])
        };
        let options = AttentionOptions {
            causal: true,
            window: Some(config.window),
            ..AttentionOptions::default()
        };
        let mut hidden = input.clone();
        for block in &self.blocks {
            let normed = hidden.apply(&block.attention_norm, state)?;
            let query = angles
                .rotate(&heads(&normed, &block.query)?)?
                .transpose(1, 2)?;
            let key = angles
                .rotate(&heads(&normed, &block.key)?)?
                .transpose(1, 2)?;
            let value = heads(&normed, &block.value)?.transpose(1, 2)?;
            let attended = query
                .attend(&key, &value, None, None, options)?
                .transpose(1, 2)?
                .reshape([batch, length, config.heads * config.head_dim])?
                .apply(&block.output, state)?
                .apply(&block.attention_scale, state)?;
            hidden = hidden.add(&attended)?;
            let fed = hidden
                .apply(&block.feed_forward_norm, state)?
                .apply(&block.feed_forward, state)?
                .apply(&block.feed_forward_scale, state)?;
            hidden = hidden.add(&fed)?;
        }
        Ok(hidden)
    }
}

/// One half of the split quantizer: a projection into the codebooks' space, the codebooks, and a projection back.
#[derive(Debug, Clone)]
struct Quantizer {
    project_in: Conv1d,
    codebooks: Vec<StateKey>,
    project_out: Conv1d,
}

impl Quantizer {
    fn create(schema: &mut Schema<'_>, config: &MimiConfig, count: i64) -> SingeResult<Self> {
        let width = config.transformer.width;
        let pointwise = Conv1dConfig::default();
        Ok(Self {
            project_in: schema.conv1d(
                "project_in",
                width,
                config.codebook_dim,
                1,
                pointwise,
                false,
            )?,
            codebooks: (0..count)
                .map(|index| {
                    schema.parameter(
                        &format!("codebooks.{index}"),
                        [config.codebook_size, config.codebook_dim],
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
            project_out: schema.conv1d(
                "project_out",
                config.codebook_dim,
                width,
                1,
                pointwise,
                false,
            )?,
        })
    }

    /// Returns the projected sum of each codebook's entries at `codes`, `[B, K, T]`, as `[B, W, T]`, as `MimiResidualVectorQuantizer.decode` gives it.
    fn decode(&self, codes: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        quantizer::summed_entries(&self.codebooks, codes, state)?.apply(&self.project_out, state)
    }

    /// Returns each codebook's entry nearest the residual the codebooks before it leave, `[B, K, T]`, as `MimiResidualVectorQuantizer.encode` gives them.
    fn codes(&self, latents: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let projected = latents.apply(&self.project_in, state)?;
        let &[batch, width, frames] = projected.shape().dimensions() else {
            return Err(SingeError::invalid_argument("Mimi latents, [B, D, T]"));
        };
        let mut residual = projected
            .transpose(1, 2)?
            .reshape([batch * frames, width])?;
        let mut codes = Vec::with_capacity(self.codebooks.len());
        for codebook in &self.codebooks {
            let entries = state.tensor(codebook)?;
            let indices = quantizer::nearest(&residual, entries)?.reshape([batch * frames])?;
            residual = residual.sub(&entries.index_select(0, &indices)?)?;
            codes.push(indices.reshape([batch, 1, frames])?);
        }
        Tensor::concat(&codes, 1)
    }

    /// Returns the projected sum of each codebook's entry nearest the residual the codebooks before it leave, `[B, W, T]`, as `MimiResidualVectorQuantizer` encodes and decodes.
    fn quantize(&self, latents: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.decode(&self.codes(latents, state)?, state)
    }
}

/// Mimi: from clips' samples, `[B, N]`, to their reconstructions through every codebook, `[B, N]`.
#[derive(Debug, Clone)]
pub struct MimiLayers {
    encoder: Seanet,
    encoder_transformer: CodecTransformer,
    downsample: PaddedConv,
    semantic: Quantizer,
    acoustic: Quantizer,
    upsample: TrimmedTranspose,
    decoder_transformer: CodecTransformer,
    decoder: Seanet,
}

impl MimiLayers {
    pub fn create(schema: &mut Schema<'_>, config: &MimiConfig) -> SingeResult<Self> {
        let width = config.transformer.width;
        let stride = config.latent_stride;
        let causal = config.seanet.causal;
        Ok(Self {
            encoder: Seanet::encoder(&mut schema.scope("encoder"), &config.seanet)?,
            encoder_transformer: CodecTransformer::create(
                &mut schema.scope("encoder_transformer"),
                &config.transformer,
            )?,
            downsample: PaddedConv::create(
                &mut schema.scope("downsample"),
                (causal, PadMode::Replicate),
                [width, width, 2 * stride, stride, 1],
                false,
            )?,
            semantic: Quantizer::create(
                &mut schema.scope("semantic"),
                config,
                config.semantic_codebooks,
            )?,
            acoustic: Quantizer::create(
                &mut schema.scope("acoustic"),
                config,
                config.codebooks - config.semantic_codebooks,
            )?,
            upsample: TrimmedTranspose::create(
                &mut schema.scope("upsample"),
                (causal, config.seanet.trim_right),
                [width, width, 2 * stride, stride, config.upsample_groups],
                false,
            )?,
            decoder_transformer: CodecTransformer::create(
                &mut schema.scope("decoder_transformer"),
                &config.transformer,
            )?,
            decoder: Seanet::decoder(&mut schema.scope("decoder"), &config.seanet)?,
        })
    }

    /// Encodes samples, `[B, N]`, to codes, `[B, K, T]`, the semantic codebooks' first, as `MimiModel.encode` does.
    pub fn encode_codes(&self, samples: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        with_convolution_precision(ComputePrecision::Full, || {
            let latents = self.latents(samples, state)?;
            Tensor::concat(
                &[
                    self.semantic.codes(&latents, state)?,
                    self.acoustic.codes(&latents, state)?,
                ],
                1,
            )
        })
    }

    /// Returns the latents the quantizers read, `[B, D, T]` in `F32`: the encoder's frames through its transformer and the downsampling, as `cdist` reads them.
    fn latents(&self, samples: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        samples
            .unsqueeze(1)?
            .apply(&self.encoder, state)?
            .transpose(1, 2)?
            .apply(&self.encoder_transformer, state)?
            .transpose(1, 2)?
            .apply(&self.downsample, state)?
            .cast(DataType::F32)
    }

    /// Decodes codes, `[B, K, T]`, the semantic codebooks' first, to samples, `[B, N]`, as `MimiModel.decode` does.
    pub fn decode_codes(&self, codes: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        with_convolution_precision(ComputePrecision::Full, || {
            let &[_, count, _] = codes.shape().dimensions() else {
                return Err(SingeError::invalid_argument("Mimi codes, [B, K, T]"));
            };
            let semantic = i64::try_from(self.semantic.codebooks.len())
                .map_err(|_| SingeError::out_of_range("semantic codebooks"))?;
            let quantized = self
                .semantic
                .decode(&codes.narrow(1, 0, semantic)?, state)?
                .add(
                    &self
                        .acoustic
                        .decode(&codes.narrow(1, semantic, count - semantic)?, state)?,
                )?;
            self.samples(&quantized, state)
        })
    }

    /// Returns the samples, `[B, N]`, the decoder gives of the quantizers' summed output, `[B, W, T]`.
    fn samples(&self, quantized: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        quantized
            .apply(&self.upsample, state)?
            .transpose(1, 2)?
            .apply(&self.decoder_transformer, state)?
            .transpose(1, 2)?
            .apply(&self.decoder, state)?
            .squeeze(1)
    }
}

/// A bandwidth is `EnCodec`'s alone; this codec reconstructs through all its codebooks.
impl CodecLayers for MimiLayers {
    fn reconstruct(
        &self,
        samples: &Tensor,
        bandwidth: Option<f64>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        if bandwidth.is_some() {
            return Err(SingeError::invalid_argument(
                "a bandwidth for a codec that reconstructs through all its codebooks",
            ));
        }

        self.forward(samples, state)
    }
}

impl Layer for MimiLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        // A codec's reconstruction is its output, which TF32 convolutions move, as DAC's.
        with_convolution_precision(ComputePrecision::Full, || {
            let &[_, length] = input.shape().dimensions() else {
                return Err(SingeError::invalid_argument("Mimi samples, [B, N]"));
            };
            // Both halves quantize the same latents.
            let latents = self.latents(input, state)?;
            let quantized = self
                .semantic
                .quantize(&latents, state)?
                .add(&self.acoustic.quantize(&latents, state)?)?;
            let decoded = self.samples(&quantized, state)?;
            // Cut to the input's length, as `MimiDecoderModel` cuts it to the padding mask's.
            let decoded_length = decoded.shape().dimensions()[1];
            decoded.narrow(1, 0, length.min(decoded_length))
        })
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "mimi",
    formats: &[(Format::Transformers, &["mimi"])],
    loaders: &[HeadLoader::AudioCodec(load_codec)],
};

pub fn load_codec(loader: &mut Loader<'_>) -> Result<AudioCodec> {
    let config: MimiConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        MimiLayers::create(schema, &config).map(|layers| Boxed::<dyn CodecLayers>(Box::new(layers)))
    })?;
    Ok(AudioCodec::new(
        loader.load_whole(&model, &transformers::rules(&config, ""))?,
        config.sample_rate,
    ))
}
