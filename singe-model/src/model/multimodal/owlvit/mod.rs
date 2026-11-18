// OWL-ViT: CLIP's towers made a detector, each image patch's state, scaled by the class token's, giving a box and an embedding scored against each text query's.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Shape, Tensor,
    nn::{Activation, Layer, LayerNorm, Linear, Model, ModelState, StateBuilder},
};

use crate::{
    error::Result,
    format::Loader,
    head::{HeadLoader, OpenVocabularyDetector, PixelLayers, TextEmbeddingLayers},
    model::{
        ModelEntry,
        multimodal::clip::{ClipConfig, ClipText, ClipTextLayers, ClipVision},
    },
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OwlVitConfig {
    pub clip: ClipConfig,
}

// The box head's bias for each patch of a `rows × columns` grid, `[cx, cy, w, h]` row by row: the logits of the patch's far corner and of a patch's size, as `compute_box_bias` computes them, so a head output of zero puts a box on its patch.
fn box_bias(rows: i64, columns: i64) -> Vec<f32> {
    let logit = |value: f32| {
        let value = value.clamp(0.0, 1.0);
        (value + 1e-4).ln() - (-value + 1e-4).ln_1p()
    };
    #[expect(clippy::cast_precision_loss, reason = "patch grids are tiny")]
    let (rows_f, columns_f) = (rows as f32, columns as f32);
    (0..rows)
        .flat_map(|row| {
            (0..columns).flat_map(move |column| {
                #[expect(clippy::cast_precision_loss, reason = "patch grids are tiny")]
                let (x, y) = ((column + 1) as f32 / columns_f, (row + 1) as f32 / rows_f);
                [x, y, 1.0 / columns_f, 1.0 / rows_f]
            })
        })
        .map(logit)
        .collect()
}

// OWL-ViT's image side: from images to each patch's class embedding, box, logit shift, and logit scale, `[B, P, D + 4 + 2]`.
#[derive(Debug, Clone)]
struct OwlVitImageLayers {
    vision: ClipVision,
    norm: LayerNorm,
    boxes: [Linear; 3],
    class: Linear,
    shift: Linear,
    scale: Linear,
    box_bias: Vec<f32>,
}

impl Layer for OwlVitImageLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self
            .vision
            .hidden(input, state)?
            .apply(self.vision.post_norm(), state)?;
        let count = hidden.shape().dimensions()[1];
        let patches = hidden
            .narrow(1, 1, count - 1)?
            .mul(&hidden.narrow(1, 0, 1)?)?
            .apply(&self.norm, state)?;
        let mut boxes = patches.clone();
        for (index, layer) in self.boxes.iter().enumerate() {
            boxes = boxes.apply(layer, state)?;
            if index < 2 {
                boxes = boxes.apply(&Activation::GeluExact, state)?;
            }
        }

        let bias = Tensor::from_vec(
            input.device(),
            Shape::create([1, count - 1, 4])?,
            self.box_bias.clone(),
        )?
        .cast(boxes.data_type())?;
        let boxes = boxes.add(&bias)?.apply(&Activation::Sigmoid, state)?;
        Tensor::concat(
            &[
                patches.apply(&self.class, state)?,
                boxes,
                patches.apply(&self.shift, state)?,
                patches.apply(&self.scale, state)?,
            ],
            2,
        )
    }
}

#[derive(Debug, Clone)]
struct OwlVitLayers {
    images: OwlVitImageLayers,
    texts: ClipTextLayers,
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "owlvit",
    formats: &[(Format::Transformers, &["owlvit"])],
    heads: &[HeadLoader::ZeroShotObjectDetection(load_detector)],
};

pub fn load_detector(loader: &mut Loader<'_>) -> Result<OpenVocabularyDetector> {
    let config: OwlVitConfig = loader.config(transformers::config)?;
    let (vision, text) = (&config.clip.vision, &config.clip.text);
    let width = vision.encoder.width;
    let side = vision.image_size / vision.patch_size;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let clip_vision = ClipVision::create(&mut schema.scope("vision"), vision)?;
        let clip_text = ClipText::create(&mut schema.scope("text"), text)?;
        let projection = schema.linear(
            "text_projection",
            text.encoder.width,
            config.clip.projection,
            false,
        )?;
        let norm = schema.layer_norm("norm", width, vision.encoder.eps)?;
        let boxes = {
            let mut schema = schema.scope("boxes");
            [
                schema.linear("0", width, width, true)?,
                schema.linear("1", width, width, true)?,
                schema.linear("2", width, 4, true)?,
            ]
        };
        let (class, shift, scale) = {
            let mut schema = schema.scope("classes");
            (
                schema.linear("embedding", width, config.clip.projection, true)?,
                schema.linear("shift", width, 1, true)?,
                schema.linear("scale", width, 1, true)?,
            )
        };
        OwlVitLayers {
            images: OwlVitImageLayers {
                vision: clip_vision,
                norm,
                boxes,
                class,
                shift,
                scale,
                box_bias: box_bias(side, side),
            },
            texts: ClipTextLayers {
                text: clip_text,
                projection,
            },
        }
    };
    let schema = Model::new(layers, builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    let state = model.state().clone();
    Ok(OpenVocabularyDetector::new(
        Model::new(
            PixelLayers(Box::new(model.layer().images.clone())),
            state.clone(),
        ),
        Model::new(
            TextEmbeddingLayers(Box::new(model.layer().texts.clone())),
            state,
        ),
    ))
}
