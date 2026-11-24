//! Vision encoder-decoders, as `transformers`' `VisionEncoderDecoderModel` pairs them: an image encoder whose states a text decoder's cross-attention reads, such as `TrOCR`'s `DeiT` and BART-style decoder, and Donut's Swin and `MBart` decoder.

mod transformers;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Layer, ModelState, PatchTokens},
};

use crate::{
    assembly::{
        BartDecoder, BartStack, BartStackConfig, FinalNorm, PreNormEncoder, StackRole, Swin,
        SwinConfig, VisionTowerConfig, patch_embedding,
    },
    error::Result,
    format::{Loader, transformers::seq2seq_generation},
    head::{Boxed, HeadLoader, Seq2SeqDecoding, Seq2SeqInput, Seq2SeqLayers, Seq2SeqLm},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImageEncoderConfig {
    /// `DeiT`: a `ViT` with a class token and a distillation token before the patches.
    Deit(VisionTowerConfig),
    Swin(SwinConfig),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisionEncoderDecoderConfig {
    pub encoder: ImageEncoderConfig,
    pub decoder: BartStackConfig,
    /// Whether the decoder's token embedding is its output's weight.
    pub tied: bool,
}

#[derive(Debug, Clone)]
enum ImageEncoder {
    Deit {
        embeddings: PatchTokens,
        encoder: PreNormEncoder,
    },
    Swin(Swin),
}

impl ImageEncoder {
    fn create(schema: &mut Schema<'_>, config: &ImageEncoderConfig) -> SingeResult<Self> {
        match config {
            ImageEncoderConfig::Deit(tower) => {
                let width = tower.encoder.width;
                let embeddings = {
                    let mut schema = schema.scope("embeddings");
                    let patches = patch_embedding(
                        &mut schema,
                        tower.channels,
                        width,
                        tower.patch_size,
                        true,
                    )?;
                    PatchTokens::new(patches)
                        .with_leading_token(schema.parameter("class", [1, 1, width])?)
                        .with_leading_token(schema.parameter("distillation", [1, 1, width])?)
                        .with_positions(
                            schema.parameter("positions", [1, tower.patches() + 2, width])?,
                        )
                };
                Ok(Self::Deit {
                    embeddings,
                    encoder: PreNormEncoder::create(
                        schema,
                        &tower.encoder,
                        tower.layers,
                        FinalNorm::Included,
                    )?,
                })
            }
            ImageEncoderConfig::Swin(config) => Ok(Self::Swin(Swin::create(schema, config)?)),
        }
    }
}

impl Layer for ImageEncoder {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        match self {
            Self::Deit {
                embeddings,
                encoder,
            } => encoder.forward(&input.apply(embeddings, state)?, state),
            Self::Swin(swin) => swin.forward(input, state),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VisionEncoderDecoderLayers {
    encoder: ImageEncoder,
    decoder: Arc<dyn Seq2SeqDecoding>,
}

impl VisionEncoderDecoderLayers {
    pub fn create(
        schema: &mut Schema<'_>,
        config: &VisionEncoderDecoderConfig,
    ) -> SingeResult<Self> {
        let decoder = &config.decoder;
        let head = schema.linear("head", decoder.width, decoder.vocab_size, false)?;
        // A tied decoder reads its tokens from the output's weight, which its checkpoint stores.
        let tied = config.tied.then(|| head.weight().clone());
        Ok(Self {
            encoder: ImageEncoder::create(&mut schema.scope("encoder"), &config.encoder)?,
            decoder: Arc::new(BartDecoder::new(
                BartStack::create(
                    &mut schema.scope("decoder"),
                    decoder,
                    StackRole::Decoder,
                    tied.as_ref(),
                )?,
                head,
                None,
            )),
        })
    }
}

impl Seq2SeqLayers for VisionEncoderDecoderLayers {
    fn encode(&self, input: Seq2SeqInput<'_>, state: &ModelState) -> SingeResult<Tensor> {
        let Seq2SeqInput::Pixels(pixels) = input else {
            return Err(SingeError::invalid_argument(
                "a vision encoder-decoder reads pixels",
            ));
        };
        pixels.apply(&self.encoder, state)
    }

    fn decoder(&self) -> Arc<dyn Seq2SeqDecoding> {
        Arc::clone(&self.decoder)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "vision_encoder_decoder",
    formats: &[(Format::Transformers, &["vision-encoder-decoder"])],
    loaders: &[HeadLoader::Seq2SeqLm(load_seq2seq)],
};

pub fn load_seq2seq(loader: &mut Loader<'_>) -> Result<Seq2SeqLm> {
    let config: VisionEncoderDecoderConfig = loader.config(transformers::config)?;
    let generation = seq2seq_generation(loader.source())?;
    let model = Schema::model(loader.options().data_type, |schema| {
        VisionEncoderDecoderLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Seq2SeqLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(&config))?;
    Ok(Seq2SeqLm::new(model, generation))
}
