// VideoMAE: a vision transformer over a clip's tubelets, patches spanning frames, with fixed sinusoid positions, whose mean final state a video classifier reads.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Conv3d, Conv3dConfig, Layer, LayerNorm, Linear, Model, ModelState, StateBuilder},
};

use crate::{
    assembly::{PreNormEncoder, PreNormEncoderSpec},
    error::Result,
    format::Loader,
    head::{Classifier, ClassifierConfig, HeadLoader, PixelLayers, VideoClassifier},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoMaeConfig {
    pub encoder: PreNormEncoderSpec,
    pub layers: i64,
    pub image_size: i64,
    pub patch_size: i64,
    pub frames: i64,
    // The frames one tubelet spans.
    pub tubelet: i64,
    pub channels: i64,
    pub classifier: ClassifierConfig,
}

impl VideoMaeConfig {
    const fn tubelets(&self) -> i64 {
        let side = self.image_size / self.patch_size;
        self.frames / self.tubelet * side * side
    }
}

// The sinusoid table of `count` positions and `width` features, row by row, as `get_sinusoid_encoding_table` computes it in `F64`: sines in even features and cosines in odd ones, of each position over 10000 to the power of its feature pair's share of the width.
fn sinusoids(count: i64, width: i64) -> Vec<f32> {
    (0..count)
        .flat_map(|position| {
            (0..width).map(move |feature| {
                #[expect(clippy::cast_precision_loss, reason = "positions and widths are small")]
                let angle =
                    position as f64 / 10_000_f64.powf(2.0 * (feature / 2) as f64 / width as f64);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "`transformers` stores the table in `F32`"
                )]
                let value = if feature % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                } as f32;
                value
            })
        })
        .collect()
}

// VideoMAE and its classifier: from clips `[B, T, C, H, W]` to each clip's scores, `[B, labels]`.
#[derive(Debug, Clone)]
pub struct VideoMaeClassifierLayers {
    tubelets: Conv3d,
    positions: Vec<f32>,
    encoder: PreNormEncoder,
    norm: LayerNorm,
    classifier: Linear,
}

impl Layer for VideoMaeClassifierLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let tubelets = input
            .permute(&[0, 2, 1, 3, 4])?
            .apply(&self.tubelets, state)?
            .flatten(2, 4)?
            .transpose(1, 2)?;
        let &[_, count, width] = tubelets.shape().dimensions() else {
            return Err(SingeError::invalid_argument("tubelet tokens"));
        };
        let positions = Tensor::from_vec(
            input.device(),
            Shape::create([1, count, width])?,
            self.positions.clone(),
        )?
        .cast(tubelets.data_type())?;
        self.encoder
            .forward(&tubelets.add(&positions)?, state)?
            .mean(&[1], false)?
            .apply(&self.norm, state)?
            .apply(&self.classifier, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "videomae",
    formats: &[(Format::Transformers, &["videomae"])],
    heads: &[HeadLoader::VideoClassification(load_classifier)],
};

pub fn load_classifier(loader: &mut Loader<'_>) -> Result<VideoClassifier> {
    let config: VideoMaeConfig = loader.config(transformers::config)?;
    let width = config.encoder.width;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let tubelets = schema.conv3d(
            "embeddings.tubelets",
            config.channels,
            width,
            [config.tubelet, config.patch_size, config.patch_size],
            Conv3dConfig {
                stride: (config.tubelet, config.patch_size, config.patch_size),
                ..Conv3dConfig::default()
            },
            true,
        )?;
        let encoder = PreNormEncoder::create(&mut schema, &config.encoder, config.layers, false)?;
        // `fc_norm` keeps PyTorch's default epsilon instead of `layer_norm_eps`.
        let norm = schema.layer_norm("norm", width, 1e-5)?;
        let labels = i64::try_from(config.classifier.labels.len())
            .map_err(|_| SingeError::out_of_range("labels"))?;
        let classifier = schema.linear("classifier.output", width, labels, true)?;
        VideoMaeClassifierLayers {
            tubelets,
            positions: sinusoids(config.tubelets(), width),
            encoder,
            norm,
            classifier,
        }
    };
    let schema = Model::new(PixelLayers(Box::new(layers)), builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(VideoClassifier(Classifier::new(model, config.classifier)))
}
