// DAC, Descript's audio codec: strided convolutions with Snake activations encode a clip into frames, each quantized by a stack of codebooks, each coding the residual the codebooks before it leave, and transposed convolutions decode the codes back to samples.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{
        Activation, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTransposeConfig, Layer, Model,
        ModelState, StateBuilder, StateKey,
    },
};

use crate::{
    error::Result,
    format::Loader,
    head::{AudioCodec, AudioLayers, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

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

// `x + sin(α x)² / (α + 1e-9)`, of each channel's `α`.
#[derive(Debug, Clone)]
struct Snake(StateKey);

impl Layer for Snake {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let alpha = state.tensor(&self.0)?;
        let wave = input.mul(alpha)?.sin()?;
        input.add(&wave.mul(&wave)?.div(&alpha.add_scalar(1e-9)?)?)
    }
}

#[derive(Debug, Clone)]
struct ResidualUnit {
    first: (Snake, Conv1d),
    second: (Snake, Conv1d),
}

impl Layer for ResidualUnit {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let output = input
            .apply(&self.first.0, state)?
            .apply(&self.first.1, state)?
            .apply(&self.second.0, state)?
            .apply(&self.second.1, state)?;
        // A unit whose convolutions shorten the clip adds the input's middle.
        let (length, shortened) = (
            input.shape().dimensions()[2],
            output.shape().dimensions()[2],
        );
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

fn unit_vectors(tensor: &Tensor) -> SingeResult<Tensor> {
    let norm = tensor
        .mul(tensor)?
        .sum(&[-1], true)?
        .sqrt()?
        .clamp(1e-12, f32::MAX)?;
    tensor.div(&norm)
}

impl DacLayers {
    // The residual vector quantizer: each codebook's nearest entry by cosine to the residual's projection, its entry projected back and taken from the residual.
    fn quantize(&self, latents: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut residual = latents.clone();
        let mut quantized: Option<Tensor> = None;
        for codebook in &self.codebooks {
            let projected = residual.apply(&codebook.project_in, state)?;
            let &[batch, width, frames] = projected.shape().dimensions() else {
                return Err(singe::Error::invalid_argument("codebook projections"));
            };
            // Every clip's frames, `[B · T, D]`, are quantized together.
            let encodings = unit_vectors(
                &projected
                    .transpose(1, 2)?
                    .reshape([batch * frames, width])?,
            )?;
            let entries = state.tensor(&codebook.entries)?;
            let normalized = unit_vectors(entries)?;
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
            let chosen = projected.add(&chosen.sub(&projected)?)?;
            let decoded = chosen.apply(&codebook.project_out, state)?;
            residual = residual.sub(&decoded)?;
            quantized = Some(match quantized {
                Some(sum) => sum.add(&decoded)?,
                None => decoded,
            });
        }
        quantized.ok_or_else(|| singe::Error::invalid_argument("no codebooks"))
    }
}

impl Layer for DacLayers {
    // A batch of clips' samples, `[B, N]`, to their reconstructions, `[B, N]`.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let length = input.shape().dimensions()[1];
        let mut hidden = input.unsqueeze(1)?.apply(&self.encoder_in, state)?;
        for block in &self.encoder {
            hidden = hidden.apply(block, state)?;
        }

        let latents = hidden
            .apply(&self.encoder_snake, state)?
            .apply(&self.encoder_out, state)?;
        let mut hidden = self
            .quantize(&latents, state)?
            .apply(&self.decoder_in, state)?;
        for block in &self.decoder {
            hidden = hidden.apply(block, state)?;
        }

        let decoded = hidden
            .apply(&self.decoder_snake, state)?
            .apply(&self.decoder_out, state)?
            .apply(&Activation::Tanh, state)?
            .squeeze(1)?;
        // Cut to the input's length when longer, as `DacModel` cuts it.
        let decoded_length = decoded.shape().dimensions()[1];
        decoded.narrow(1, 0, length.min(decoded_length))
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "dac",
    formats: &[(Format::Transformers, &["dac"])],
    heads: &[HeadLoader::AudioCodec(load_codec)],
};

fn padded(padding: i64) -> Conv1dConfig {
    Conv1dConfig {
        padding: (padding, padding),
        ..Conv1dConfig::default()
    }
}

fn residual_units(schema: &mut Schema<'_>, width: i64) -> SingeResult<Vec<ResidualUnit>> {
    [1, 3, 9]
        .iter()
        .enumerate()
        .map(|(index, &dilation)| {
            let mut schema = schema.scope(format!("units.{index}"));
            Ok(ResidualUnit {
                first: (
                    Snake(schema.parameter("first_snake", [1, width, 1])?),
                    schema.conv1d(
                        "first",
                        width,
                        width,
                        7,
                        Conv1dConfig {
                            padding: (3 * dilation, 3 * dilation),
                            dilation,
                            ..Conv1dConfig::default()
                        },
                        true,
                    )?,
                ),
                second: (
                    Snake(schema.parameter("second_snake", [1, width, 1])?),
                    schema.conv1d("second", width, width, 1, Conv1dConfig::default(), true)?,
                ),
            })
        })
        .collect()
}

pub fn load_codec(loader: &mut Loader<'_>) -> Result<AudioCodec> {
    let config: DacConfig = loader.config(transformers::config)?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let encoder_in =
            schema.conv1d("encoder_in", 1, config.encoder_width, 7, padded(3), true)?;
        let mut width = config.encoder_width;
        let encoder = config
            .downsampling
            .iter()
            .enumerate()
            .map(|(index, &stride)| {
                let mut schema = schema.scope(format!("encoder.{index}"));
                let block = Block {
                    units: residual_units(&mut schema, width)?,
                    snake: Snake(schema.parameter("snake", [1, width, 1])?),
                    resample: Resample::Down(schema.conv1d(
                        "resample",
                        width,
                        width * 2,
                        2 * stride,
                        Conv1dConfig {
                            stride,
                            padding: ((stride + 1) / 2, (stride + 1) / 2),
                            ..Conv1dConfig::default()
                        },
                        true,
                    )?),
                };
                width *= 2;
                Ok(block)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let encoder_snake = Snake(schema.parameter("encoder_snake", [1, width, 1])?);
        let encoder_out = schema.conv1d("encoder_out", width, config.latent, 3, padded(1), true)?;
        let codebooks = (0..config.codebooks)
            .map(|index| {
                let mut schema = schema.scope(format!("codebooks.{index}"));
                Ok(Codebook {
                    project_in: schema.conv1d(
                        "project_in",
                        config.latent,
                        config.codebook_dim,
                        1,
                        Conv1dConfig::default(),
                        true,
                    )?,
                    entries: schema
                        .parameter("entries", [config.codebook_size, config.codebook_dim])?,
                    project_out: schema.conv1d(
                        "project_out",
                        config.codebook_dim,
                        config.latent,
                        1,
                        Conv1dConfig::default(),
                        true,
                    )?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let decoder_in = schema.conv1d(
            "decoder_in",
            config.latent,
            config.decoder_width,
            7,
            padded(3),
            true,
        )?;
        let mut width = config.decoder_width;
        let decoder = config
            .upsampling
            .iter()
            .enumerate()
            .map(|(index, &stride)| {
                let mut schema = schema.scope(format!("decoder.{index}"));
                let snake = Snake(schema.parameter("snake", [1, width, 1])?);
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
                width /= 2;
                Ok(Block {
                    units: residual_units(&mut schema, width)?,
                    snake,
                    resample,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        DacLayers {
            encoder_in,
            encoder,
            encoder_snake,
            encoder_out,
            codebooks,
            decoder_in,
            decoder,
            decoder_snake: Snake(schema.parameter("decoder_snake", [1, width, 1])?),
            decoder_out: schema.conv1d("decoder_out", width, 1, 7, padded(3), true)?,
        }
    };
    let schema = Model::new(AudioLayers(Box::new(layers)), builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(AudioCodec::new(model, config.sample_rate))
}
