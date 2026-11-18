// Vision encoder-decoders, as `transformers`' `VisionEncoderDecoderModel` pairs them: an image encoder whose states a text decoder's cross-attention reads, such as TrOCR's DeiT and BART-style decoder, and Donut's Swin and MBart decoder.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Shape, Tensor,
    nn::{Conv2dConfig, Layer, Linear, Model, ModelState, PatchEmbedding, StateBuilder, StateKey},
};

use crate::{
    assembly::{PreNormEncoder, PreNormEncoderSpec, Swin, SwinConfig},
    error::Result,
    format::{Loader, transformers::seq2seq_generation},
    head::{HeadLoader, Seq2SeqInput, Seq2SeqLayers, Seq2SeqLm},
    model::{
        ModelEntry,
        text::bart::{BartStack, BartStackConfig},
    },
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImageEncoderConfig {
    // DeiT: a ViT with a class token and a distillation token before the patches.
    Deit {
        encoder: PreNormEncoderSpec,
        layers: i64,
        image_size: i64,
        patch_size: i64,
        channels: i64,
    },
    Swin(SwinConfig),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisionEncoderDecoderConfig {
    pub encoder: ImageEncoderConfig,
    pub decoder: BartStackConfig,
    // Whether the decoder's token embedding is its output's weight.
    pub tied: bool,
}

// DeiT's tokens: its class and distillation tokens, then its patches, with learned positions.
#[derive(Debug, Clone)]
struct DeitTokens {
    patches: PatchEmbedding,
    class: StateKey,
    distillation: StateKey,
    positions: StateKey,
}

impl Layer for DeitTokens {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let patches = input.apply(&self.patches, state)?;
        let dimensions = patches.shape().dimensions();
        let (batch, width) = (dimensions[0], dimensions[2]);
        let token = |key: &StateKey| state.tensor(key)?.expand(Shape::create([batch, 1, width])?);
        Tensor::concat(
            &[token(&self.class)?, token(&self.distillation)?, patches],
            1,
        )?
        .add(state.tensor(&self.positions)?)
    }
}

#[derive(Debug, Clone)]
enum ImageEncoder {
    Deit {
        tokens: DeitTokens,
        encoder: PreNormEncoder,
    },
    Swin(Swin),
}

impl Layer for ImageEncoder {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        match self {
            Self::Deit { tokens, encoder } => encoder.forward(&input.apply(tokens, state)?, state),
            Self::Swin(swin) => swin.forward(input, state),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VisionEncoderDecoderLayers {
    encoder: ImageEncoder,
    decoder: BartStack,
    head: Linear,
}

impl Seq2SeqLayers for VisionEncoderDecoderLayers {
    fn encode(&self, input: Seq2SeqInput<'_>, state: &ModelState) -> SingeResult<Tensor> {
        let Seq2SeqInput::Pixels(pixels) = input else {
            return Err(singe::Error::invalid_argument(
                "a vision encoder-decoder reads pixels",
            ));
        };
        pixels.apply(&self.encoder, state)
    }

    fn next_scores(
        &self,
        encoded: &Tensor,
        tokens: &[i64],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        self.decoder
            .last_state(encoded, tokens, state)?
            .apply(&self.head, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "vision_encoder_decoder",
    formats: &[(Format::Transformers, &["vision-encoder-decoder"])],
    heads: &[HeadLoader::Seq2SeqLm(load_seq2seq)],
};

fn image_encoder(
    schema: &mut Schema<'_>,
    config: &ImageEncoderConfig,
) -> SingeResult<ImageEncoder> {
    match config {
        ImageEncoderConfig::Deit {
            encoder,
            layers,
            image_size,
            patch_size,
            channels,
        } => {
            let width = encoder.width;
            let side = image_size / patch_size;
            let tokens = {
                let mut schema = schema.scope("embeddings");
                DeitTokens {
                    patches: PatchEmbedding::new(schema.conv2d(
                        "patches",
                        *channels,
                        width,
                        *patch_size,
                        Conv2dConfig {
                            stride: (*patch_size, *patch_size),
                            ..Conv2dConfig::default()
                        },
                        true,
                    )?),
                    class: schema.parameter("class", [1, 1, width])?,
                    distillation: schema.parameter("distillation", [1, 1, width])?,
                    positions: schema.parameter("positions", [1, side * side + 2, width])?,
                }
            };
            Ok(ImageEncoder::Deit {
                tokens,
                encoder: PreNormEncoder::create(schema, encoder, *layers, true)?,
            })
        }
        ImageEncoderConfig::Swin(config) => Ok(ImageEncoder::Swin(Swin::create(schema, config)?)),
    }
}

pub fn load_seq2seq(loader: &mut Loader<'_>) -> Result<Seq2SeqLm> {
    let config: VisionEncoderDecoderConfig = loader.config(transformers::config)?;
    let generation = seq2seq_generation(loader.source())?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let decoder = &config.decoder;
        let head = schema.linear("head", decoder.width, decoder.vocab_size, false)?;
        // A tied decoder reads its tokens from the output's weight, which its checkpoint stores.
        let tied = config.tied.then(|| head.weight().clone());
        VisionEncoderDecoderLayers {
            encoder: image_encoder(&mut schema.scope("encoder"), &config.encoder)?,
            decoder: BartStack::create(&mut schema.scope("decoder"), decoder, true, tied.as_ref())?,
            head,
        }
    };
    let layers: Box<dyn Seq2SeqLayers> = Box::new(layers);
    let schema = Model::new(layers, builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules(&config))?;
    loader.finish()?;
    Ok(Seq2SeqLm::new(model, generation))
}
