//! RT-DETR: a `ResNet`-vd's last three maps, projected, through a hybrid encoder of one attention layer on the coarsest map and a top-down and a bottom-up path of convolutions across the levels.
//! The encoder's best-scoring points become the decoder's object queries and box references, which each decoder layer refines after sampling the levels around them.
//! RT-DETRv2 checkpoints whose decoder samples as RT-DETR's does load as RT-DETR.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::{ComputePrecision, capture, with_convolution_precision},
    nn::{Activation, Layer, LayerNorm, Linear, Mlp, ModelState},
};

use crate::{
    assembly::{
        Attention, AttentionSpec, ConvNorm, DeformableAttentionSpec, DeformableDecoderLayer,
        FeaturePyramid, LinearStack, ResNetConfig, ResNetVd, anchor_proposals, gather_rows,
        inverse_sigmoid, to_map, to_tokens,
    },
    error::Result,
    format::Loader,
    head::{Boxed, ClassifierConfig, DetectionScoring, HeadLoader, ObjectDetector},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

/// The repeated blocks in each of the encoder's cross-stage layers.
const REP_BLOCKS: usize = 3;

/// The backbone's maps the model reads: its last three.
const LEVELS: usize = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RtDetrConfig {
    pub backbone: ResNetConfig,
    pub width: i64,
    /// The channels inside each cross-stage layer.
    pub hidden: i64,
    /// The levels, indexed from the finest, that the encoder's attention layers run on.
    pub attended_levels: Vec<usize>,
    pub encoder_layers: i64,
    pub encoder_heads: i64,
    pub encoder_inner: i64,
    pub encoder_activation: Activation,
    /// The activation of the encoder's convolutions.
    pub activation: Activation,
    pub temperature: f32,
    pub decoder_layers: i64,
    pub decoder_heads: i64,
    pub decoder_inner: i64,
    pub decoder_points: i64,
    pub decoder_activation: Activation,
    pub norm_eps: f32,
    pub queries: i64,
    pub classifier: ClassifierConfig,
    pub scoring: DetectionScoring,
}

/// A post-norm encoder layer of self-attention, whose positions join its queries and keys, and a feed-forward network.
#[derive(Debug, Clone)]
struct EncoderLayer {
    attention: Attention,
    attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
}

impl EncoderLayer {
    fn create(schema: &mut Schema<'_>, config: &RtDetrConfig) -> SingeResult<Self> {
        let width = config.width;
        Ok(Self {
            attention: Attention::create(
                &mut schema.scope("attention"),
                AttentionSpec::square(width, config.encoder_heads),
            )?,
            attention_norm: schema.layer_norm("attention_norm", width, config.norm_eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.encoder_inner, width],
                config.encoder_activation,
                true,
            )?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.norm_eps)?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        positions: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let positioned = hidden.add(positions)?;
        let hidden = self
            .attention
            .forward(&positioned, &positioned, hidden, state)?
            .add(hidden)?
            .apply(&self.attention_norm, state)?;
        hidden
            .apply(&self.feed_forward, state)?
            .add(&hidden)?
            .apply(&self.feed_forward_norm, state)
    }
}

/// A cross-stage layer: two 1 × 1 branches of the input, one through repeated 3 × 3 and 1 × 1 convolutions summed, added together and projected to the width when the branches are narrower.
#[derive(Debug, Clone)]
struct CrossStage {
    first: ConvNorm,
    second: ConvNorm,
    blocks: Vec<[ConvNorm; 2]>,
    output: Option<ConvNorm>,
    activation: Activation,
}

impl CrossStage {
    fn create(schema: &mut Schema<'_>, config: &RtDetrConfig) -> SingeResult<Self> {
        let (width, hidden) = (config.width, config.hidden);
        let conv = |schema: &mut Schema<'_>, scope: &str, input, out, kernel| {
            let names = ("convolution", "norm");
            ConvNorm::create(&mut schema.scope(scope), names, input, out, kernel, 1)
        };
        let blocks = (0..REP_BLOCKS)
            .map(|index| {
                Ok([
                    conv(schema, &format!("blocks.{index}.wide"), hidden, hidden, 3)?,
                    conv(schema, &format!("blocks.{index}.narrow"), hidden, hidden, 1)?,
                ])
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            first: conv(schema, "first", 2 * width, hidden, 1)?.with_activation(config.activation),
            second: conv(schema, "second", 2 * width, hidden, 1)?
                .with_activation(config.activation),
            blocks,
            output: (hidden != width)
                .then(|| {
                    conv(schema, "output", hidden, width, 1)
                        .map(|layer| layer.with_activation(config.activation))
                })
                .transpose()?,
            activation: config.activation,
        })
    }
}

impl Layer for CrossStage {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut first = input.apply(&self.first, state)?;
        for [wide, narrow] in &self.blocks {
            first = first
                .apply(wide, state)?
                .add(&first.apply(narrow, state)?)?
                .apply(&self.activation, state)?;
        }

        let sum = first.add(&input.apply(&self.second, state)?)?;
        match &self.output {
            Some(output) => sum.apply(output, state),
            None => Ok(sum),
        }
    }
}

/// Returns the 2D sine and cosine positions of a `height` × `width` map, `[1, H · W, features]`, as `build_2d_sinusoidal_position_embedding` computes them in `F64`: each point's row's sines and cosines, then its column's, at frequencies `temperature^(-k / quarter)`.
fn grid_positions(map: &Tensor, features: i64, temperature: f32) -> SingeResult<Tensor> {
    let &[_, _, height, width] = map.shape().dimensions() else {
        return Err(SingeError::invalid_argument("RT-DETR map, [B, C, H, W]"));
    };
    let quarter = features / 4;
    let frequencies = (0..quarter)
        .map(|index| 1.0 / f64::from(temperature).powf(index as f64 / quarter as f64))
        .collect::<Vec<_>>();
    let mut values = Vec::new();
    for row in 0..height {
        for column in 0..width {
            for coordinate in [row, column] {
                let angles = frequencies
                    .iter()
                    .map(|frequency| coordinate as f64 * frequency);
                values.extend(angles.clone().map(|angle| angle.sin() as f32));
                values.extend(angles.map(|angle| angle.cos() as f32));
            }
        }
    }
    capture::literal(map.device(), |device| {
        Tensor::from_vec(
            device,
            Shape::create([1, height * width, features])?,
            values,
        )
    })
}

/// RT-DETR and its heads: from images to each query's label scores and box, `[B, Q, labels + 4]`.
#[derive(Debug, Clone)]
pub struct RtDetrLayers {
    width: i64,
    temperature: f32,
    queries: i64,
    backbone: ResNetVd,
    input_projections: Vec<ConvNorm>,
    attended_levels: Vec<usize>,
    encoder: Vec<EncoderLayer>,
    lateral: Vec<ConvNorm>,
    top_down: Vec<CrossStage>,
    downsample: Vec<ConvNorm>,
    bottom_up: Vec<CrossStage>,
    decoder_projections: Vec<ConvNorm>,
    proposal: Linear,
    proposal_norm: LayerNorm,
    proposal_scores: Linear,
    proposal_boxes: LinearStack,
    query_positions: LinearStack,
    decoder: Vec<DeformableDecoderLayer>,
    /// The last decoder layer's class head, the only one the detections read.
    classes: Linear,
    boxes: Vec<LinearStack>,
}

impl RtDetrLayers {
    pub fn create(schema: &mut Schema<'_>, config: &RtDetrConfig) -> SingeResult<Self> {
        let width = config.width;
        let backbone = ResNetVd::create(&mut schema.scope("backbone"), &config.backbone)?;
        let stages = config.backbone.depths.len();
        let conv = |schema: &mut Schema<'_>, scope: &str, input, kernel, stride| {
            let names = ("convolution", "norm");
            ConvNorm::create(
                &mut schema.scope(scope),
                names,
                input,
                width,
                kernel,
                stride,
            )
        };
        let input_projections = (0..LEVELS)
            .map(|level| {
                let channels = config.backbone.channels(stages - LEVELS + level);
                conv(
                    schema,
                    &format!("input_projections.{level}"),
                    channels,
                    1,
                    1,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let encoder = (0..config.encoder_layers)
            .map(|index| {
                EncoderLayer::create(&mut schema.scope(format!("encoder.{index}")), config)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let mut lateral = Vec::new();
        let mut top_down = Vec::new();
        let mut downsample = Vec::new();
        let mut bottom_up = Vec::new();
        for index in 0..LEVELS - 1 {
            lateral.push(
                conv(schema, &format!("lateral.{index}"), width, 1, 1)?
                    .with_activation(config.activation),
            );
            top_down.push(CrossStage::create(
                &mut schema.scope(format!("top_down.{index}")),
                config,
            )?);
            downsample.push(
                conv(schema, &format!("downsample.{index}"), width, 3, 2)?
                    .with_activation(config.activation),
            );
            bottom_up.push(CrossStage::create(
                &mut schema.scope(format!("bottom_up.{index}")),
                config,
            )?);
        }

        let decoder_projections = (0..LEVELS)
            .map(|level| conv(schema, &format!("decoder_projections.{level}"), width, 1, 1))
            .collect::<SingeResult<Vec<_>>>()?;
        let labels = config.classifier.label_count()?;
        let box_head = |schema: &mut Schema<'_>, scope: &str| {
            LinearStack::create(
                &mut schema.scope(scope),
                [width, width, 4],
                3,
                Activation::Relu,
            )
        };
        let spec = DeformableAttentionSpec {
            width,
            heads: config.decoder_heads,
            levels: i64::try_from(LEVELS).unwrap_or(i64::MAX),
            points: config.decoder_points,
        };
        let decoder = (0..config.decoder_layers)
            .map(|index| {
                DeformableDecoderLayer::create(
                    &mut schema.scope(format!("decoder.{index}")),
                    spec,
                    config.decoder_inner,
                    config.decoder_activation,
                    config.norm_eps,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let boxes = (0..config.decoder_layers)
            .map(|index| box_head(schema, &format!("boxes.{index}")))
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            width,
            temperature: config.temperature,
            queries: config.queries,
            backbone,
            input_projections,
            attended_levels: config.attended_levels.clone(),
            encoder,
            lateral,
            top_down,
            downsample,
            bottom_up,
            decoder_projections,
            proposal: schema.linear("proposal", width, width, true)?,
            proposal_norm: schema.layer_norm("proposal_norm", width, config.norm_eps)?,
            proposal_scores: schema.linear("proposal_scores", width, labels, true)?,
            proposal_boxes: box_head(schema, "proposal_boxes")?,
            query_positions: LinearStack::create(
                &mut schema.scope("query_positions"),
                [4, 2 * width, width],
                2,
                Activation::Relu,
            )?,
            decoder,
            classes: schema.linear("classes", width, labels, true)?,
            boxes,
        })
    }

    /// Returns the encoder's map at each level, `[B, W, H_l, W_l]`, from the finest.
    fn encode(&self, input: &Tensor, state: &ModelState) -> SingeResult<Vec<Tensor>> {
        let stages = self.backbone.stages(input, state)?;
        let first = stages.len() - self.input_projections.len();
        let mut maps = self
            .input_projections
            .iter()
            .zip(&stages[first..])
            .map(|(projection, map)| map.apply(projection, state))
            .collect::<SingeResult<Vec<_>>>()?;
        for &level in &self.attended_levels {
            let map = &maps[level];
            let positions =
                grid_positions(map, self.width, self.temperature)?.cast(map.data_type())?;
            let mut hidden = to_tokens(map)?;
            for layer in &self.encoder {
                hidden = layer.forward(&hidden, &positions, state)?;
            }

            maps[level] = to_map(&hidden, (map.dimension(2)?, map.dimension(3)?))?;
        }

        let count = maps.len();
        let mut top_down = vec![maps[count - 1].clone()];
        for (index, (lateral, block)) in self.lateral.iter().zip(&self.top_down).enumerate() {
            let last = top_down.len() - 1;
            let reduced = top_down[last].apply(lateral, state)?;
            top_down[last] = reduced.clone();
            let fused = Tensor::concat(&[upsample(&reduced)?, maps[count - 2 - index].clone()], 1)?;
            top_down.push(fused.apply(block, state)?);
        }

        top_down.reverse();
        let mut bottom_up = vec![top_down[0].clone()];
        for (index, (downsample, block)) in self.downsample.iter().zip(&self.bottom_up).enumerate()
        {
            let reduced = bottom_up[bottom_up.len() - 1].apply(downsample, state)?;
            let fused = Tensor::concat(&[reduced, top_down[index + 1].clone()], 1)?;
            bottom_up.push(fused.apply(block, state)?);
        }
        Ok(bottom_up)
    }
}

/// Returns `map`, `[B, C, H, W]`, at twice its height and width, each point repeated, as nearest interpolation by two does.
fn upsample(map: &Tensor) -> SingeResult<Tensor> {
    let (batch, channels) = (map.dimension(0)?, map.dimension(1)?);
    let (height, width) = (map.dimension(2)?, map.dimension(3)?);
    map.unsqueeze(3)?
        .unsqueeze(5)?
        .expand_with(&[
            batch.clone(),
            channels.clone(),
            height.clone(),
            2.into(),
            width.clone(),
            2.into(),
        ])?
        .reshape_with(&[batch, channels, height * 2, width * 2])
}

impl Layer for RtDetrLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        // Detections are thresholded scores, which TF32 convolutions move, as in DETR.
        let maps = with_convolution_precision(ComputePrecision::Full, || {
            let encoded = self.encode(input, state)?;
            self.decoder_projections
                .iter()
                .zip(&encoded)
                .map(|(projection, map)| map.apply(projection, state))
                .collect::<SingeResult<Vec<_>>>()
        })?;
        let device = input.device();
        let data_type = maps[0].data_type();
        let mut tokens = Vec::with_capacity(maps.len());
        let mut sizes = Vec::with_capacity(maps.len());
        let mut extents = Vec::with_capacity(maps.len());
        for map in &maps {
            let &[_, _, height, width] = map.shape().dimensions() else {
                return Err(SingeError::invalid_argument("RT-DETR level, [B, C, H, W]"));
            };
            tokens.push(to_tokens(map)?);
            sizes.push([map.dimension(2)?, map.dimension(3)?]);
            extents.push([height, width]);
        }

        let memory = Tensor::concat(&tokens, 1)?;
        let levels = FeaturePyramid::create(device, sizes)?;

        let (anchors, valid) = anchor_proposals(device, &extents, f32::MAX)?;
        let proposals = memory
            .mul(&valid.cast(data_type)?)?
            .apply(&self.proposal, state)?
            .apply(&self.proposal_norm, state)?;
        let scores = proposals.apply(&self.proposal_scores, state)?;
        let boxes = proposals
            .apply(&self.proposal_boxes, state)?
            .add(&anchors.cast(data_type)?)?;
        let (_, chosen) = scores
            .max_reduce(&[2], false)?
            .topk(self.queries, 1, true, true)?;
        let mut references = gather_rows(&boxes, &chosen)?.apply(&Activation::Sigmoid, state)?;
        let mut hidden = gather_rows(&proposals, &chosen)?;

        for (index, layer) in self.decoder.iter().enumerate() {
            let positions = references.apply(&self.query_positions, state)?;
            hidden = layer.forward(&hidden, &positions, &references, &memory, &levels, state)?;
            references = hidden
                .apply(&self.boxes[index], state)?
                .add(&inverse_sigmoid(&references, 1e-5)?)?
                .apply(&Activation::Sigmoid, state)?;
        }

        let scores = hidden.apply(&self.classes, state)?;
        Tensor::concat(&[scores, references], 2)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "rt_detr",
    formats: &[(Format::Transformers, &["rt_detr", "rt_detr_v2"])],
    loaders: &[HeadLoader::ObjectDetection(load_detector)],
};

pub fn load_detector(loader: &mut Loader<'_>) -> Result<ObjectDetector> {
    let config: RtDetrConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        RtDetrLayers::create(schema, &config).map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(config.decoder_layers))?;
    Ok(ObjectDetector::new(
        model,
        config.classifier,
        config.scoring,
    ))
}
