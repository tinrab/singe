//! DAC, Descript's audio codec: strided convolutions with Snake activations encode a clip into frames, each quantized by a stack of codebooks, each coding the residual the codebooks before it leave, and transposed convolutions decode the codes back to samples.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    graph::{ComputePrecision, with_convolution_precision},
    nn::{
        Activation, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTransposeConfig, Layer, ModelState,
        StateKey,
    },
};

use crate::{
    assembly::{same_1d, unit_vectors},
    error::Result,
    format::Loader,
    head::{AudioCodec, Boxed, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

pub(crate) const DILATIONS: [i64; 3] = [1, 3, 9];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DacConfig {
    pub encoder_width: i64,
    pub decoder_width: i64,
    pub latent: i64,
    pub downsampling: Vec<i64>,
    pub upsampling: Vec<i64>,
    pub codebooks: i64,
    pub codebook_size: i64,
    pub codebook_dim: i64,
    pub sample_rate: i64,
}

/// The Snake activation, `x + sin(α x)² / (α + 1e-9)`, of each channel's `α`.
#[derive(Debug, Clone)]
struct Snake(StateKey);

impl Snake {
    fn create(schema: &mut Schema<'_>, name: &str, width: i64) -> SingeResult<Self> {
        Ok(Self(schema.parameter(name, [1, width, 1])?))
    }
}

impl Layer for Snake {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let alpha = state.tensor(&self.0)?;
        let wave = input.mul(alpha)?.sin()?;
        input.add(&wave.mul(&wave)?.div(&alpha.add_scalar(1e-9)?)?)
    }
}

#[derive(Debug, Clone)]
struct ResidualUnit {
    first_snake: Snake,
    first: Conv1d,
    second_snake: Snake,
    second: Conv1d,
}

impl ResidualUnit {
    fn create(schema: &mut Schema<'_>, width: i64, dilation: i64) -> SingeResult<Self> {
        Ok(Self {
            first_snake: Snake::create(schema, "first_snake", width)?,
            first: schema.conv1d("first", width, width, 7, same_1d(7, dilation), true)?,
            second_snake: Snake::create(schema, "second_snake", width)?,
            second: schema.conv1d("second", width, width, 1, Conv1dConfig::default(), true)?,
        })
    }

    fn stack(schema: &mut Schema<'_>, width: i64) -> SingeResult<Vec<Self>> {
        DILATIONS
            .iter()
            .enumerate()
            .map(|(index, &dilation)| {
                Self::create(&mut schema.scope(format!("units.{index}")), width, dilation)
            })
            .collect()
    }
}

impl Layer for ResidualUnit {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let output = input
            .apply(&self.first_snake, state)?
            .apply(&self.first, state)?
            .apply(&self.second_snake, state)?
            .apply(&self.second, state)?;
        // A unit whose convolutions shorten the clip adds the input's middle.
        let &[_, _, length] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument("DAC residual unit input"));
        };
        let &[_, _, shortened] = output.shape().dimensions() else {
            return Err(SingeError::invalid_argument("DAC residual unit output"));
        };
        let crop = (length - shortened) / 2;
        let input = if crop > 0 {
            input.narrow(2, crop, shortened)?
        } else {
            input.clone()
        };
        input.add(&output)
    }
}

#[derive(Debug, Clone)]
enum Resample {
    Down(Conv1d),
    Up(ConvTranspose1d),
}

#[derive(Debug, Clone)]
struct Block {
    units: Vec<ResidualUnit>,
    snake: Snake,
    resample: Resample,
}

impl Block {
    fn down(schema: &mut Schema<'_>, width: i64, stride: i64) -> SingeResult<Self> {
        let padding = (stride + 1) / 2;
        Ok(Self {
            units: ResidualUnit::stack(schema, width)?,
            snake: Snake::create(schema, "snake", width)?,
            resample: Resample::Down(schema.conv1d(
                "resample",
                width,
                width * 2,
                2 * stride,
                Conv1dConfig {
                    stride,
                    padding: (padding, padding),
                    ..Conv1dConfig::default()
                },
                true,
            )?),
        })
    }

    fn up(schema: &mut Schema<'_>, width: i64, stride: i64) -> SingeResult<Self> {
        let snake = Snake::create(schema, "snake", width)?;
        let resample = Resample::Up(schema.conv_transpose1d(
            "resample",
            width,
            width / 2,
            2 * stride,
            ConvTransposeConfig {
                stride: [stride],
                padding: [(stride + 1) / 2],
                ..ConvTransposeConfig::default()
            },
            true,
        )?);
        Ok(Self {
            units: ResidualUnit::stack(schema, width / 2)?,
            snake,
            resample,
        })
    }
}

impl Layer for Block {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        match &self.resample {
            // Encoders refine, then downsample; decoders upsample, then refine.
            Resample::Down(convolution) => {
                let mut hidden = input.clone();
                for unit in &self.units {
                    hidden = hidden.apply(unit, state)?;
                }
                hidden.apply(&self.snake, state)?.apply(convolution, state)
            }
            Resample::Up(convolution) => {
                let mut hidden = input.apply(&self.snake, state)?.apply(convolution, state)?;
                for unit in &self.units {
                    hidden = hidden.apply(unit, state)?;
                }
                Ok(hidden)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Codebook {
    project_in: Conv1d,
    entries: StateKey,
    project_out: Conv1d,
}

impl Codebook {
    fn create(schema: &mut Schema<'_>, config: &DacConfig) -> SingeResult<Self> {
        Ok(Self {
            project_in: schema.conv1d(
                "project_in",
                config.latent,
                config.codebook_dim,
                1,
                Conv1dConfig::default(),
                true,
            )?,
            entries: schema.parameter("entries", [config.codebook_size, config.codebook_dim])?,
            project_out: schema.conv1d(
                "project_out",
                config.codebook_dim,
                config.latent,
                1,
                Conv1dConfig::default(),
                true,
            )?,
        })
    }

    fn decoded(&self, residual: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let projected = residual.apply(&self.project_in, state)?;
        let &[batch, width, frames] = projected.shape().dimensions() else {
            return Err(SingeError::invalid_argument("codebook projections"));
        };
        // Every clip's frames, `[B · T, D]`, are quantized together.
        let encodings = unit_vectors(
            &projected
                .transpose(1, 2)?
                .reshape([batch * frames, width])?,
            -1,
        )?;
        let entries = state.tensor(&self.entries)?;
        let normalized = unit_vectors(entries, -1)?;
        // `decode_latents`' distance, whose sign on the entries' norm the argmax keeps.
        let distance = encodings
            .matmul(&normalized.transpose(0, 1)?)?
            .mul_scalar(2.0)?
            .sub(&encodings.mul(&encodings)?.sum(&[-1], true)?)?
            .add(
                &normalized
                    .mul(&normalized)?
                    .sum(&[-1], true)?
                    .transpose(0, 1)?,
            )?;
        let (_, indices) = distance.topk(1, -1, true, false)?;
        let chosen = entries
            .index_select(0, &indices.reshape([batch * frames])?)?
            .reshape([batch, frames, width])?
            .transpose(1, 2)?;
        // The straight-through estimate, whose value is the entry's up to rounding.
        projected
            .add(&chosen.sub(&projected)?)?
            .apply(&self.project_out, state)
    }
}

#[derive(Debug, Clone)]
pub struct DacLayers {
    encoder_in: Conv1d,
    encoder: Vec<Block>,
    encoder_snake: Snake,
    encoder_out: Conv1d,
    codebooks: Vec<Codebook>,
    decoder_in: Conv1d,
    decoder: Vec<Block>,
    decoder_snake: Snake,
    decoder_out: Conv1d,
}

impl DacLayers {
    pub fn create(schema: &mut Schema<'_>, config: &DacConfig) -> SingeResult<Self> {
        let mut width = config.encoder_width;
        let mut encoder = Vec::with_capacity(config.downsampling.len());
        for (index, &stride) in config.downsampling.iter().enumerate() {
            encoder.push(Block::down(
                &mut schema.scope(format!("encoder.{index}")),
                width,
                stride,
            )?);
            width *= 2;
        }

        let encoder_snake = Snake::create(schema, "encoder_snake", width)?;
        let encoder_out =
            schema.conv1d("encoder_out", width, config.latent, 3, same_1d(3, 1), true)?;
        let codebooks = (0..config.codebooks)
            .map(|index| Codebook::create(&mut schema.scope(format!("codebooks.{index}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        let mut width = config.decoder_width;
        let mut decoder = Vec::with_capacity(config.upsampling.len());
        for (index, &stride) in config.upsampling.iter().enumerate() {
            decoder.push(Block::up(
                &mut schema.scope(format!("decoder.{index}")),
                width,
                stride,
            )?);
            width /= 2;
        }

        Ok(Self {
            encoder_in: schema.conv1d(
                "encoder_in",
                1,
                config.encoder_width,
                7,
                same_1d(7, 1),
                true,
            )?,
            encoder,
            encoder_snake,
            encoder_out,
            codebooks,
            decoder_in: schema.conv1d(
                "decoder_in",
                config.latent,
                config.decoder_width,
                7,
                same_1d(7, 1),
                true,
            )?,
            decoder,
            decoder_snake: Snake::create(schema, "decoder_snake", width)?,
            decoder_out: schema.conv1d("decoder_out", width, 1, 7, same_1d(7, 1), true)?,
        })
    }

    /// Returns the residual vector quantizer's reconstruction of `latents`.
    /// Each codebook takes its entry nearest by cosine to the projection of the residual the codebooks before it leave, and the entries, projected back, are summed.
    fn quantize(&self, latents: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let Some((first, rest)) = self.codebooks.split_first() else {
            return Err(SingeError::invalid_argument("a DAC without codebooks"));
        };
        let mut quantized = first.decoded(latents, state)?;
        let mut residual = latents.sub(&quantized)?;
        for codebook in rest {
            let decoded = codebook.decoded(&residual, state)?;
            residual = residual.sub(&decoded)?;
            quantized = quantized.add(&decoded)?;
        }
        Ok(quantized)
    }

    fn encode(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.unsqueeze(1)?.apply(&self.encoder_in, state)?;
        for block in &self.encoder {
            hidden = hidden.apply(block, state)?;
        }

        hidden
            .apply(&self.encoder_snake, state)?
            .apply(&self.encoder_out, state)
    }

    fn decode(&self, quantized: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = quantized.apply(&self.decoder_in, state)?;
        for block in &self.decoder {
            hidden = hidden.apply(block, state)?;
        }

        hidden
            .apply(&self.decoder_snake, state)?
            .apply(&self.decoder_out, state)?
            .apply(&Activation::Tanh, state)?
            .squeeze(1)
    }
}

impl Layer for DacLayers {
    /// Returns the reconstructions, `[B, N]`, of a batch of clips' samples, `[B, N]`.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        // A codec's reconstruction is its output, which TF32 convolutions take from 117 dB of SNR against `transformers`' to 29 dB: full precision.
        with_convolution_precision(ComputePrecision::Full, || {
            let &[_, length] = input.shape().dimensions() else {
                return Err(SingeError::invalid_argument("DAC samples"));
            };
            let latents = self.encode(input, state)?;
            let decoded = self.decode(&self.quantize(&latents, state)?, state)?;
            // Cut to the input's length when longer, as `DacModel` cuts it.
            let &[_, decoded_length] = decoded.shape().dimensions() else {
                return Err(SingeError::invalid_argument("DAC reconstructions"));
            };
            decoded.narrow(1, 0, length.min(decoded_length))
        })
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "dac",
    formats: &[(Format::Transformers, &["dac"])],
    loaders: &[HeadLoader::AudioCodec(load_codec)],
};

pub fn load_codec(loader: &mut Loader<'_>) -> Result<AudioCodec> {
    let config: DacConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        DacLayers::create(schema, &config).map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    Ok(AudioCodec::new(
        loader.load_whole(&model, &transformers::rules())?,
        config.sample_rate,
    ))
}
