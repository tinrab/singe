//! OWL-ViT: CLIP's towers made a detector, each image patch's state, scaled by the class token's, giving a box and an embedding scored against each text query's.

mod transformers;

use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, Layer, LayerNorm, Linear, ModelState},
};

use crate::{
    assembly::LinearStack,
    error::Result,
    format::Loader,
    head::{HeadLoader, OpenVocabularyDetector},
    model::{
        ModelEntry,
        multimodal::clip::{
            self, ClipConfig, ClipText, ClipTextLayers, ClipVision, ImageTextLayers,
        },
    },
    schema::Schema,
    source::Format,
};

/// Returns the box head's bias for each patch of a `rows × columns` grid, `[cx, cy, w, h]` row by row: the logits of the patch's far corner and of a patch's size, as `compute_box_bias` computes them, so a head output of zero gives a patch-sized box centered on the patch's far corner.
fn box_bias(rows: i64, columns: i64) -> Vec<f32> {
    let logit = |value: f32| {
        let value = value.clamp(0.0, 1.0);
        (value + 1e-4).ln() - (-value + 1e-4).ln_1p()
    };
    let (rows_f, columns_f) = (rows as f32, columns as f32);
    (0..rows)
        .flat_map(|row| {
            (0..columns).flat_map(move |column| {
                let (x, y) = ((column + 1) as f32 / columns_f, (row + 1) as f32 / rows_f);
                [x, y, 1.0 / columns_f, 1.0 / rows_f]
            })
        })
        .map(logit)
        .collect()
}

/// OWL-ViT's image side: from images to each patch's class embedding, box, logit shift, and logit scale, `[B, P, D + 4 + 2]`.
#[derive(Debug, Clone)]
struct OwlVitImageLayers {
    vision: ClipVision,
    norm: LayerNorm,
    boxes: LinearStack,
    class: Linear,
    shift: Linear,
    scale: Linear,
    box_bias: Vec<f32>,
}

impl OwlVitImageLayers {
    fn create(schema: &mut Schema<'_>, config: &ClipConfig) -> SingeResult<Self> {
        let vision = &config.vision;
        let width = vision.encoder.width;
        let side = vision.side();
        let mut classes = schema.scope("classes");
        let (class, shift, scale) = (
            classes.linear("embedding", width, config.projection, true)?,
            classes.linear("shift", width, 1, true)?,
            classes.linear("scale", width, 1, true)?,
        );
        Ok(Self {
            vision: ClipVision::create(&mut schema.scope("vision"), vision)?,
            norm: schema.layer_norm("norm", width, vision.encoder.eps)?,
            boxes: LinearStack::create(
                &mut schema.scope("boxes"),
                [width, width, 4],
                3,
                Activation::GeluExact,
            )?,
            class,
            shift,
            scale,
            box_bias: box_bias(side, side),
        })
    }
}

impl Layer for OwlVitImageLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self
            .vision
            .hidden(input, state)?
            .apply(self.vision.post_norm(), state)?;
        let &[_, count, _] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("the vision tower's states"));
        };
        let patches = hidden
            .narrow(1, 1, count - 1)?
            .mul(&hidden.narrow(1, 0, 1)?)?
            .apply(&self.norm, state)?;
        let bias = Tensor::from_vec(
            input.device(),
            Shape::create([1, count - 1, 4])?,
            self.box_bias.clone(),
        )?;
        let boxes = patches.apply(&self.boxes, state)?;
        let boxes = boxes
            .add(&bias.cast(boxes.data_type())?)?
            .apply(&Activation::Sigmoid, state)?;
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

type OwlVitLayers = ImageTextLayers<OwlVitImageLayers, ClipTextLayers>;

fn layers(schema: &mut Schema<'_>, config: &ClipConfig) -> SingeResult<OwlVitLayers> {
    let text = &config.text;
    Ok(ImageTextLayers {
        images: OwlVitImageLayers::create(schema, config)?,
        texts: ClipTextLayers {
            text: ClipText::create(&mut schema.scope("text"), text)?,
            projection: schema.linear(
                "text_projection",
                text.encoder.width,
                config.projection,
                false,
            )?,
        },
    })
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "owlvit",
    formats: &[(Format::Transformers, &["owlvit"])],
    loaders: &[HeadLoader::ZeroShotObjectDetection(load_detector)],
};

pub fn load_detector(loader: &mut Loader<'_>) -> Result<OpenVocabularyDetector> {
    let config: ClipConfig = loader.config(clip::transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| layers(schema, &config))?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    let (images, texts) = model.layer().split(model.state());
    Ok(OpenVocabularyDetector::new(images, texts))
}
