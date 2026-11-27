//! Deformable DETR: a `ResNet`'s last three maps and a fourth strided from the last, each projected and group-normed, through a post-norm encoder of deformable self-attention over all the levels, then a decoder of learned object queries attending to each other and sampling the levels around reference points they predict, read by a class head and a box head that moves each query's reference.

mod transformers;

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    graph::{ComputePrecision, with_convolution_precision},
    nn::{Activation, Layer, Linear, ModelState, StateKey},
};

use crate::{
    assembly::{
        ConvGroupNorm, DeformableAttentionSpec, DeformableDecoderLayer, DeformableEncoderLayer,
        FeaturePyramid, LinearStack, PYTORCH_NORM_EPS, ResNet, ResNetConfig, SinePositionSpec,
        inverse_sigmoid, sine_positions, to_tokens,
    },
    error::Result,
    format::Loader,
    head::{Boxed, ClassifierConfig, DetectionScoring, HeadLoader, ObjectDetector},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

/// The groups of each level's projection norm.
const PROJECTION_GROUPS: i64 = 32;

/// The positions of every level, normalized sines of each point's center.
const POSITIONS: SinePositionSpec = SinePositionSpec {
    offset: 0.5,
    temperature: 10_000.0,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeformableDetrConfig {
    pub backbone: ResNetConfig,
    /// The backbone's last stages read as levels, from the first; any levels beyond them are strided from the last.
    pub backbone_levels: usize,
    pub levels: i64,
    pub width: i64,
    pub encoder_layers: i64,
    pub decoder_layers: i64,
    pub encoder_heads: i64,
    pub decoder_heads: i64,
    pub encoder_inner: i64,
    pub decoder_inner: i64,
    pub encoder_points: i64,
    pub decoder_points: i64,
    pub activation: Activation,
    pub queries: i64,
    pub classifier: ClassifierConfig,
    pub scoring: DetectionScoring,
}

/// Deformable DETR and its heads: from images to each query's label scores and box, `[B, Q, labels + 4]`.
#[derive(Debug, Clone)]
pub struct DeformableDetrLayers {
    width: i64,
    backbone: ResNet,
    backbone_levels: usize,
    projections: Vec<ConvGroupNorm>,
    level_embeddings: StateKey,
    encoder: Vec<DeformableEncoderLayer>,
    decoder: Vec<DeformableDecoderLayer>,
    queries: StateKey,
    references: Linear,
    classes: Linear,
    boxes: LinearStack,
}

impl DeformableDetrLayers {
    pub fn create(schema: &mut Schema<'_>, config: &DeformableDetrConfig) -> SingeResult<Self> {
        let width = config.width;
        let backbone = ResNet::create(&mut schema.scope("backbone"), &config.backbone)?;
        let stages = config.backbone.depths.len();
        let mut projections = Vec::new();
        for level in 0..config.levels {
            let mut schema = schema.scope(format!("projections.{level}"));
            let index = usize::try_from(level).unwrap_or(usize::MAX);
            let (input, kernel, stride) = match index.cmp(&config.backbone_levels) {
                Ordering::Less => {
                    let stage = stages - config.backbone_levels + index;
                    (config.backbone.channels(stage), 1, 1)
                }
                Ordering::Equal => (config.backbone.channels(stages - 1), 3, 2),
                Ordering::Greater => (width, 3, 2),
            };
            projections.push(ConvGroupNorm::create(
                &mut schema,
                [input, width, kernel, stride, PROJECTION_GROUPS],
                true,
            )?);
        }

        let encoder_spec = DeformableAttentionSpec {
            width,
            heads: config.encoder_heads,
            levels: config.levels,
            points: config.encoder_points,
        };
        let encoder = (0..config.encoder_layers)
            .map(|index| {
                DeformableEncoderLayer::create(
                    &mut schema.scope(format!("encoder.{index}")),
                    encoder_spec,
                    config.encoder_inner,
                    config.activation,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let decoder_spec = DeformableAttentionSpec {
            width,
            heads: config.decoder_heads,
            levels: config.levels,
            points: config.decoder_points,
        };
        let decoder = (0..config.decoder_layers)
            .map(|index| {
                DeformableDecoderLayer::create(
                    &mut schema.scope(format!("decoder.{index}")),
                    decoder_spec,
                    config.decoder_inner,
                    config.activation,
                    PYTORCH_NORM_EPS,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let labels = config.classifier.label_count()?;
        let boxes = LinearStack::create(
            &mut schema.scope("boxes"),
            [width, width, 4],
            3,
            Activation::Relu,
        )?;
        Ok(Self {
            width,
            backbone,
            backbone_levels: config.backbone_levels,
            projections,
            level_embeddings: schema.parameter("level_embeddings", [config.levels, width])?,
            encoder,
            decoder,
            queries: schema.parameter("queries", [config.queries, 2 * width])?,
            references: schema.linear("references", width, 2, true)?,
            classes: schema.linear("classes", width, labels, true)?,
            boxes,
        })
    }

    /// Returns each level's map, `[B, W, H_l, W_l]`.
    fn levels(&self, input: &Tensor, state: &ModelState) -> SingeResult<Vec<Tensor>> {
        let stages = self.backbone.stages(input, state)?;
        let first = stages.len() - self.backbone_levels;
        let mut maps = Vec::with_capacity(self.projections.len());
        for (index, projection) in self.projections.iter().enumerate() {
            let source = match index.checked_sub(self.backbone_levels) {
                None => &stages[first + index],
                Some(0) => &stages[stages.len() - 1],
                Some(_) => &maps[index - 1],
            };
            let map = source.apply(projection, state)?;
            maps.push(map);
        }
        Ok(maps)
    }
}

impl Layer for DeformableDetrLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        // Detections are thresholded scores, which TF32 convolutions move, as in DETR.
        let maps =
            with_convolution_precision(ComputePrecision::Full, || self.levels(input, state))?;
        let device = input.device();
        let data_type = maps[0].data_type();
        let width = self.width;
        let level_embeddings = state.tensor(&self.level_embeddings)?;
        let mut tokens = Vec::with_capacity(maps.len());
        let mut positions = Vec::with_capacity(maps.len());
        let mut sizes = Vec::with_capacity(maps.len());
        for (index, map) in maps.iter().enumerate() {
            let level = i64::try_from(index).unwrap_or(i64::MAX);
            let embedding = level_embeddings.narrow(0, level, 1)?.unsqueeze(0)?;
            let sines = sine_positions(map, width, POSITIONS)?;
            positions.push(sines.cast(data_type)?.add(&embedding)?);
            tokens.push(to_tokens(map)?);
            sizes.push([map.dimension(2)?, map.dimension(3)?]);
        }

        let positions = Tensor::concat(&positions, 1)?;
        let levels = FeaturePyramid::create(device, sizes)?;
        let references = levels.reference_points()?.cast(data_type)?;
        let mut memory = Tensor::concat(&tokens, 1)?;
        for layer in &self.encoder {
            memory = layer.forward(&memory, &positions, &references, &levels, state)?;
        }

        let queries = state.tensor(&self.queries)?;
        let count = queries.dimension(0)?;
        let batched = [memory.dimension(0)?, count, width.into()];
        let query_positions = queries
            .narrow(1, 0, width)?
            .unsqueeze(0)?
            .expand_with(&batched)?;
        let mut hidden = queries
            .narrow(1, width, width)?
            .unsqueeze(0)?
            .expand_with(&batched)?;
        let references = query_positions
            .apply(&self.references, state)?
            .apply(&Activation::Sigmoid, state)?;
        for layer in &self.decoder {
            hidden = layer.forward(
                &hidden,
                &query_positions,
                &references,
                &memory,
                &levels,
                state,
            )?;
        }

        let scores = hidden.apply(&self.classes, state)?;
        let deltas = hidden.apply(&self.boxes, state)?;
        let centers = deltas
            .narrow(2, 0, 2)?
            .add(&inverse_sigmoid(&references, 1e-5)?)?;
        let boxes = Tensor::concat(&[centers, deltas.narrow(2, 2, 2)?], 2)?
            .apply(&Activation::Sigmoid, state)?;
        Tensor::concat(&[scores, boxes], 2)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "deformable_detr",
    formats: &[(Format::Transformers, &["deformable_detr"])],
    loaders: &[HeadLoader::ObjectDetection(load_detector)],
};

pub fn load_detector(loader: &mut Loader<'_>) -> Result<ObjectDetector> {
    let config: DeformableDetrConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        DeformableDetrLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(ObjectDetector::new(
        model,
        config.classifier,
        config.scoring,
    ))
}
