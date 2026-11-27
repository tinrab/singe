//! `EnCodec`: a `SEANet` encoder of strided convolutions, residual blocks, and an LSTM, a residual vector quantizer of Euclidean codebooks, and a decoder mirroring the encoder with transposed convolutions.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    graph::{ComputePrecision, with_convolution_precision},
    nn::{ModelState, StateKey},
};

use crate::{
    assembly::{Seanet, SeanetConfig},
    error::Result,
    format::Loader,
    head::{AudioCodec, Boxed, CodecLayers, HeadLoader},
    model::{ModelEntry, audio::quantizer},
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncodecConfig {
    pub sample_rate: i64,
    pub seanet: SeanetConfig,
    pub codebook_size: i64,
    pub codebook_dim: i64,
    /// The codebooks the default bandwidth uses.
    pub codebooks: i64,
    /// The bandwidths the model was trained for, in kilobits a second.
    pub bandwidths: Vec<f64>,
    /// The bits a second each codebook adds: its size's bits at the frame rate.
    pub codebook_bits: f64,
}

impl EncodecConfig {
    /// Returns the codebooks `bandwidth` takes, as `get_num_quantizers_for_bandwidth` counts them, or `None` for a bandwidth the model was not trained for.
    #[must_use]
    pub fn codebooks_for(&self, bandwidth: f64) -> Option<i64> {
        self.bandwidths
            .contains(&bandwidth)
            .then(|| ((bandwidth * 1000.0 / self.codebook_bits).floor() as i64).max(1))
    }

    /// Returns the codebooks the largest bandwidth takes, which the model holds.
    #[must_use]
    pub fn held_codebooks(&self) -> i64 {
        self.bandwidths
            .iter()
            .filter_map(|&bandwidth| self.codebooks_for(bandwidth))
            .max()
            .unwrap_or(self.codebooks)
    }
}

/// `EnCodec`: from clips' samples, `[B, N]`, to their reconstructions through a bandwidth's codebooks, `[B, N]`.
#[derive(Debug, Clone)]
pub struct EncodecLayers {
    encoder: Seanet,
    codebooks: Vec<StateKey>,
    decoder: Seanet,
    config: EncodecConfig,
}

impl EncodecLayers {
    pub fn create(schema: &mut Schema<'_>, config: &EncodecConfig) -> SingeResult<Self> {
        let codebooks = (0..config.held_codebooks())
            .map(|index| {
                schema.parameter(
                    &format!("codebooks.{index}"),
                    [config.codebook_size, config.codebook_dim],
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            encoder: Seanet::encoder(&mut schema.scope("encoder"), &config.seanet)?,
            codebooks,
            decoder: Seanet::decoder(&mut schema.scope("decoder"), &config.seanet)?,
            config: config.clone(),
        })
    }

    /// Returns the sum of each of the first `count` codebooks' entry nearest the residual the codebooks before it leave, `[B, D, T]`, as `EncodecResidualVectorQuantizer` encodes and decodes.
    fn quantize(&self, latents: &Tensor, count: usize, state: &ModelState) -> SingeResult<Tensor> {
        let &[batch, width, frames] = latents.shape().dimensions() else {
            return Err(SingeError::invalid_argument("EnCodec latents, [B, D, T]"));
        };
        let mut residual = latents.transpose(1, 2)?.reshape([batch * frames, width])?;
        let mut quantized: Option<Tensor> = None;
        for codebook in self.codebooks.iter().take(count) {
            let entries = state.tensor(codebook)?;
            let indices = quantizer::nearest(&residual, entries)?;
            let chosen = entries.index_select(0, &indices.reshape([batch * frames])?)?;
            residual = residual.sub(&chosen)?;
            quantized = Some(match quantized {
                Some(quantized) => quantized.add(&chosen)?,
                None => chosen,
            });
        }
        quantized
            .ok_or_else(|| SingeError::invalid_argument("an EnCodec without codebooks"))?
            .reshape([batch, frames, width])?
            .transpose(1, 2)
    }

    /// Returns the samples, `[B, N]`, of clips' codes, `[B, K, T]`: each codebook's entries summed, then decoded.
    pub fn decode_codes(&self, codes: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        with_convolution_precision(ComputePrecision::Full, || {
            quantizer::summed_entries(&self.codebooks, codes, state)?
                .apply(&self.decoder, state)?
                .squeeze(1)
        })
    }
}

impl CodecLayers for EncodecLayers {
    fn reconstruct(
        &self,
        input: &Tensor,
        bandwidth: Option<f64>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let count = match bandwidth {
            Some(bandwidth) => self.config.codebooks_for(bandwidth).ok_or_else(|| {
                SingeError::invalid_argument(format!(
                    "a bandwidth of {bandwidth} kbps, rather than one of {:?}",
                    self.config.bandwidths
                ))
            })?,
            None => self.config.codebooks,
        };
        let count = usize::try_from(count).map_err(|_| SingeError::out_of_range("codebooks"))?;
        // A codec's reconstruction is its output, which TF32 convolutions move, as DAC's.
        with_convolution_precision(ComputePrecision::Full, || {
            let &[_, length] = input.shape().dimensions() else {
                return Err(SingeError::invalid_argument("EnCodec samples, [B, N]"));
            };
            let latents = input.unsqueeze(1)?.apply(&self.encoder, state)?;
            let decoded = self
                .quantize(&latents.cast(DataType::F32)?, count, state)?
                .apply(&self.decoder, state)?
                .squeeze(1)?;
            // Cut to the input's length, as `EncodecModel.decode` cuts it to the padding mask's.
            let decoded_length = decoded.shape().dimensions()[1];
            decoded.narrow(1, 0, length.min(decoded_length))
        })
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "encodec",
    formats: &[(Format::Transformers, &["encodec"])],
    loaders: &[HeadLoader::AudioCodec(load_codec)],
};

pub fn load_codec(loader: &mut Loader<'_>) -> Result<AudioCodec> {
    let config: EncodecConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        EncodecLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn CodecLayers>(Box::new(layers)))
    })?;
    Ok(AudioCodec::new(
        loader.load_whole(&model, &transformers::rules(&config, ""))?,
        config.sample_rate,
    ))
}
