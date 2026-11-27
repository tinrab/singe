//! `TimeSformer`: each frame's patches with a class token and learned positions, then a learned time embedding per frame, through layers that attend over each patch's frames, then over each frame's patches with a copy of the class token, whose frames' outputs average back into one; the classifier reads the class token.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Activation, Conv2d, Conv2dConfig, Layer, LayerNorm, Linear, Mlp, ModelState, StateKey},
    shape::spatial::Spatial2d,
};

use crate::{
    assembly::{Attention, AttentionSpec, to_tokens},
    error::Result,
    format::Loader,
    head::{Boxed, Classifier, ClassifierConfig, HeadLoader, VideoClassifier},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimesformerConfig {
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    pub patch_size: i64,
    pub image_size: i64,
    pub frames: i64,
    pub channels: i64,
    pub classifier: ClassifierConfig,
}

/// A layer of divided space-time attention, as `TimesformerLayer` computes it with `divided_space_time`.
#[derive(Debug, Clone)]
struct DividedLayer {
    temporal_norm: LayerNorm,
    temporal_attention: Attention,
    temporal_output: Linear,
    spatial_norm: LayerNorm,
    spatial_attention: Attention,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
}

impl DividedLayer {
    fn create(schema: &mut Schema<'_>, config: &TimesformerConfig) -> SingeResult<Self> {
        let width = config.width;
        let spec = AttentionSpec::square(width, config.heads);
        Ok(Self {
            temporal_norm: schema.layer_norm("temporal_norm", width, config.eps)?,
            temporal_attention: Attention::create(&mut schema.scope("temporal_attention"), spec)?,
            temporal_output: schema.linear("temporal_output", width, width, true)?,
            spatial_norm: schema.layer_norm("spatial_norm", width, config.eps)?,
            spatial_attention: Attention::create(&mut schema.scope("spatial_attention"), spec)?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.inner, width],
                config.activation,
                true,
            )?,
        })
    }

    /// Runs the layer over `hidden`, `[B, 1 + P · T, W]`: the class token, then each patch's frames in turn.
    fn forward(
        &self,
        hidden: &Tensor,
        [patches, frames]: [i64; 2],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let &[batch, _, width] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "TimeSformer tokens, [B, N, W]",
            ));
        };
        let tokens = hidden.narrow(1, 1, patches * frames)?;
        let temporal = tokens
            .reshape([batch * patches, frames, width])?
            .apply(&self.temporal_norm, state)?;
        let temporal = self
            .temporal_attention
            .forward(&temporal, &temporal, &temporal, state)?
            .reshape([batch, patches * frames, width])?
            .apply(&self.temporal_output, state)?;
        let tokens = tokens.add(&temporal)?;

        // Each frame's patches, with the class token in front of every frame.
        let class = hidden.narrow(1, 0, 1)?;
        let spatial = Tensor::concat(
            &[
                class
                    .expand_with(&[batch.into(), frames.into(), width.into()])?
                    .reshape([batch * frames, 1, width])?,
                tokens
                    .reshape([batch, patches, frames, width])?
                    .permute(&[0, 2, 1, 3])?
                    .reshape([batch * frames, patches, width])?,
            ],
            1,
        )?
        .apply(&self.spatial_norm, state)?;
        let spatial = self
            .spatial_attention
            .forward(&spatial, &spatial, &spatial, state)?;
        let class_output = spatial
            .narrow(1, 0, 1)?
            .reshape([batch, frames, width])?
            .mean(&[1], true)?;
        let patch_output = spatial
            .narrow(1, 1, patches)?
            .reshape([batch, frames, patches, width])?
            .permute(&[0, 2, 1, 3])?
            .reshape([batch, patches * frames, width])?;
        let hidden = Tensor::concat(&[class, tokens], 1)?
            .add(&Tensor::concat(&[class_output, patch_output], 1)?)?;
        hidden
            .apply(&self.feed_forward_norm, state)?
            .apply(&self.feed_forward, state)?
            .add(&hidden)
    }
}

/// `TimeSformer` and its classifier: from clips `[B, T, C, H, W]` to each label's score, `[B, labels]`.
#[derive(Debug, Clone)]
pub struct TimesformerLayers {
    patches: Conv2d,
    class_token: StateKey,
    positions: StateKey,
    times: StateKey,
    layers: Vec<DividedLayer>,
    norm: LayerNorm,
    classifier: Linear,
    frames: i64,
}

impl TimesformerLayers {
    pub fn create(schema: &mut Schema<'_>, config: &TimesformerConfig) -> SingeResult<Self> {
        let width = config.width;
        let side = config.image_size / config.patch_size;
        let layers = (0..config.layers)
            .map(|index| DividedLayer::create(&mut schema.scope(format!("layers.{index}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            patches: schema.conv2d(
                "patches",
                config.channels,
                width,
                config.patch_size,
                Conv2dConfig {
                    stride: Spatial2d::square(config.patch_size),
                    ..Conv2dConfig::default()
                },
                true,
            )?,
            class_token: schema.parameter("class_token", [1, 1, width])?,
            positions: schema.parameter("positions", [1, side * side + 1, width])?,
            times: schema.parameter("times", [1, config.frames, width])?,
            layers,
            norm: schema.layer_norm("norm", width, config.eps)?,
            classifier: schema.linear(
                "classifier",
                width,
                config.classifier.label_count()?,
                true,
            )?,
            frames: config.frames,
        })
    }
}

impl Layer for TimesformerLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[batch, frames, channels, height, width] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "TimeSformer clips, [B, T, C, H, W]",
            ));
        };
        if frames != self.frames {
            return Err(SingeError::invalid_argument(
                "TimeSformer clips of the frames its time embeddings hold",
            ));
        }

        let patches = to_tokens(
            &input
                .reshape([batch * frames, channels, height, width])?
                .apply(&self.patches, state)?,
        )?;
        let &[_, count, features] = patches.shape().dimensions() else {
            return Err(SingeError::invalid_argument("TimeSformer patches"));
        };
        let positions = state.tensor(&self.positions)?;
        // The class token and its position are the same for every frame.
        let class = state
            .tensor(&self.class_token)?
            .add(&positions.narrow(1, 0, 1)?)?
            .expand_with(&[batch.into(), 1.into(), features.into()])?;
        let patches = patches
            .add(&positions.narrow(1, 1, count)?)?
            .reshape([batch, frames, count, features])?
            .permute(&[0, 2, 1, 3])?
            .reshape([batch * count, frames, features])?
            .add(state.tensor(&self.times)?)?
            .reshape([batch, count * frames, features])?;
        let mut hidden = Tensor::concat(&[class, patches], 1)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, [count, frames], state)?;
        }
        hidden
            .apply(&self.norm, state)?
            .narrow(1, 0, 1)?
            .reshape([batch, features])?
            .apply(&self.classifier, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "timesformer",
    formats: &[(Format::Transformers, &["timesformer"])],
    loaders: &[HeadLoader::VideoClassification(load_classifier)],
};

pub fn load_classifier(loader: &mut Loader<'_>) -> Result<VideoClassifier> {
    let config: TimesformerConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        TimesformerLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(VideoClassifier(Classifier::new(model, config.classifier)))
}
