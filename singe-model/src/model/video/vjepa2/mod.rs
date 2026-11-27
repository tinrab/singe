//! V-JEPA 2: tubelets of a clip read by pre-norm blocks whose attention turns queries and keys by their frame, row, and column, and, for classification, pooled by attention from a learned query.
//! Its rotary positions repeat each section's frequencies end to end while pairing adjacent features, so a pair's two features turn by different angles, as the original V-JEPA code does; they are computed here as `x · cos + swap(x) · sin` with a table per feature.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, Conv3d, Conv3dConfig, Layer, LayerNorm, Linear, Mlp, ModelState, StateKey},
    shape::spatial::Spatial3d,
    tensor::AttentionOptions,
};

use crate::{
    assembly::{
        FinalNorm, PreNormEncoder, PreNormEncoderConfig, merge_heads, split_heads, swapped_pairs,
    },
    error::Result,
    format::Loader,
    head::{Boxed, Classifier, ClassifierConfig, HeadLoader, VideoClassifier, VideoEncoder},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vjepa2Config {
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    pub channels: i64,
    pub patch_size: i64,
    pub tubelet: i64,
    /// The rows and columns of tubelets the rotary positions number, the crop size over the patch size, whatever the clip's size.
    pub grid: i64,
    pub pooler_layers: i64,
    pub classifier: ClassifierConfig,
}

/// The cosines and sines of each of a section's features at each token, `[N, D]`, and the section's width.
struct SectionTables {
    cos: Tensor,
    sin: Tensor,
    width: i64,
}

/// A block's attention, with V-JEPA's rotary positions on its queries and keys.
#[derive(Debug, Clone)]
struct RopeAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
}

impl RopeAttention {
    fn create(schema: &mut Schema<'_>, config: &Vjepa2Config) -> SingeResult<Self> {
        let [query, key, value, output] = schema.attention_projections(config.width, [true; 4])?;
        Ok(Self {
            query,
            key,
            value,
            output,
            heads: config.heads,
        })
    }

    /// Returns `x`, `[B, H, N, d]`, each of its three sections turned by its own positions, and any features past them kept.
    fn rotated(x: &Tensor, sections: &[SectionTables]) -> SingeResult<Tensor> {
        let head = *x
            .shape()
            .dimensions()
            .last()
            .ok_or_else(|| SingeError::invalid_argument("attention heads"))?;
        let mut pieces = Vec::with_capacity(sections.len() + 1);
        let mut start = 0;
        for section in sections {
            let part = x.narrow(-1, start, section.width)?;
            pieces.push(
                part.mul(&section.cos)?
                    .add(&swapped_pairs(&part)?.mul(&section.sin)?)?,
            );
            start += section.width;
        }

        if start < head {
            pieces.push(x.narrow(-1, start, head - start)?);
        }
        Tensor::concat(&pieces, -1)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        sections: &[SectionTables],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let heads = |linear: &Linear| split_heads(&hidden.apply(linear, state)?, self.heads);
        let query = Self::rotated(&heads(&self.query)?, sections)?;
        let key = Self::rotated(&heads(&self.key)?, sections)?;
        let attended = query.attend(
            &key,
            &heads(&self.value)?,
            None,
            None,
            AttentionOptions::default(),
        )?;
        merge_heads(&attended)?.apply(&self.output, state)
    }
}

#[derive(Debug, Clone)]
struct Block {
    attention_norm: LayerNorm,
    attention: RopeAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
}

impl Block {
    fn create(schema: &mut Schema<'_>, config: &Vjepa2Config) -> SingeResult<Self> {
        let width = config.width;
        Ok(Self {
            attention_norm: schema.layer_norm("attention_norm", width, config.eps)?,
            attention: RopeAttention::create(&mut schema.scope("attention"), config)?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.inner, width],
                config.activation,
                true,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        sections: &[SectionTables],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let attended =
            self.attention
                .forward(&hidden.apply(&self.attention_norm, state)?, sections, state)?;
        let hidden = hidden.add(&attended)?;
        hidden.add(
            &hidden
                .apply(&self.feed_forward_norm, state)?
                .apply(&self.feed_forward, state)?,
        )
    }
}

/// V-JEPA 2's attentive pooler: pre-norm self-attention layers, then a learned query attending to the normed tokens without an output projection, its residual the query's, and a feed-forward.
#[derive(Debug, Clone)]
struct AttentivePooler {
    layers: PreNormEncoder,
    query: StateKey,
    cross_norm: LayerNorm,
    cross_query: Linear,
    cross_key: Linear,
    cross_value: Linear,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
    heads: i64,
}

impl AttentivePooler {
    fn create(schema: &mut Schema<'_>, config: &Vjepa2Config) -> SingeResult<Self> {
        let width = config.width;
        let encoder = PreNormEncoderConfig {
            width,
            heads: config.heads,
            inner: config.inner,
            activation: config.activation,
            eps: config.eps,
            projection_biases: [true; 4],
            causal: false,
            layer_scale: false,
        };
        Ok(Self {
            layers: PreNormEncoder::create(
                schema,
                &encoder,
                config.pooler_layers,
                FinalNorm::Omitted,
            )?,
            query: schema.parameter("query", [1, 1, width])?,
            cross_norm: schema.layer_norm("cross_norm", width, config.eps)?,
            cross_query: schema.linear("cross_attention.query", width, width, true)?,
            cross_key: schema.linear("cross_attention.key", width, width, true)?,
            cross_value: schema.linear("cross_attention.value", width, width, true)?,
            feed_forward_norm: schema.layer_norm("cross_feed_forward_norm", width, config.eps)?,
            feed_forward: schema.mlp(
                "cross_feed_forward",
                [width, config.inner, width],
                config.activation,
                true,
            )?,
            heads: config.heads,
        })
    }

    fn forward(&self, tokens: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self.layers.forward(tokens, state)?;
        let &[batch, _, width] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("pooled tokens, [B, N, W]"));
        };
        let query = state
            .tensor(&self.query)?
            .expand(Shape::create([batch, 1, width])?)?;
        let normed = hidden.apply(&self.cross_norm, state)?;
        let split =
            |input: &Tensor, linear: &Linear| split_heads(&input.apply(linear, state)?, self.heads);
        let attended = split(&query, &self.cross_query)?.attend(
            &split(&normed, &self.cross_key)?,
            &split(&normed, &self.cross_value)?,
            None,
            None,
            AttentionOptions::default(),
        )?;
        let pooled = query.add(&merge_heads(&attended)?)?;
        pooled
            .add(
                &pooled
                    .apply(&self.feed_forward_norm, state)?
                    .apply(&self.feed_forward, state)?,
            )?
            .select(1, 0)
    }
}

/// V-JEPA 2's encoder: from clips `[B, T, C, H, W]` to each tubelet's normed state, `[B, N, W]`.
#[derive(Debug, Clone)]
pub struct Vjepa2Encoder {
    tubelets: Conv3d,
    blocks: Vec<Block>,
    norm: LayerNorm,
    config: Vjepa2Config,
}

impl Vjepa2Encoder {
    pub fn create(schema: &mut Schema<'_>, config: &Vjepa2Config) -> SingeResult<Self> {
        let width = config.width;
        let tubelets = schema.conv3d(
            "tubelets",
            config.channels,
            width,
            [config.tubelet, config.patch_size, config.patch_size],
            Conv3dConfig {
                stride: Spatial3d::new(config.tubelet, config.patch_size, config.patch_size),
                ..Conv3dConfig::default()
            },
            true,
        )?;
        let blocks = (0..config.layers)
            .map(|index| Block::create(&mut schema.scope(format!("blocks.{index}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            tubelets,
            blocks,
            norm: schema.layer_norm("norm", width, config.eps)?,
            config: config.clone(),
        })
    }

    /// Returns the three sections' tables for `count` tokens, by frame, row, and column of the grid, as `get_position_ids` numbers them, in `like`'s type on its device.
    fn sections(&self, count: i64, like: &Tensor) -> SingeResult<Vec<SectionTables>> {
        let head = self.config.width / self.config.heads;
        let width = 2 * ((head / 3) / 2);
        let half = width / 2;
        // `1 / 10000^(k / half)` in `F32`, each repeated for both halves of the section.
        let frequencies = (0..width)
            .map(|feature| 1.0 / 10_000_f32.powf((feature % half) as f32 / half as f32))
            .collect::<Vec<_>>();
        let grid = self.config.grid;
        let per_frame = grid * grid;
        let tables = |position: &dyn Fn(i64) -> i64| -> SingeResult<SectionTables> {
            let angles = (0..count)
                .flat_map(|token| {
                    let at = position(token) as f32;
                    frequencies.iter().map(move |frequency| at * frequency)
                })
                .collect::<Vec<f32>>();
            let table = |values: Vec<f32>| -> SingeResult<Tensor> {
                Tensor::from_vec(like.device(), Shape::create([count, width])?, values)?
                    .cast(like.data_type())
            };
            Ok(SectionTables {
                cos: table(angles.iter().map(|angle| angle.cos()).collect())?,
                sin: table(angles.iter().map(|angle| angle.sin()).collect())?,
                width,
            })
        };
        Ok(vec![
            tables(&|token| token / per_frame)?,
            tables(&|token| (token % per_frame) / grid)?,
            tables(&|token| token % per_frame % grid)?,
        ])
    }
}

impl Layer for Vjepa2Encoder {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[_, frames, _, _, _] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument("clips, [B, T, C, H, W]"));
        };
        let clip = input.permute(&[0, 2, 1, 3, 4])?;
        // A clip shorter than a tubelet repeats its frames to fill one.
        let clip = if frames < self.config.tubelet {
            Tensor::concat(
                &vec![clip; usize::try_from(self.config.tubelet).unwrap_or(1)],
                2,
            )?
        } else {
            clip
        };
        let mut hidden = clip
            .apply(&self.tubelets, state)?
            .flatten(2, 4)?
            .transpose(1, 2)?;
        let sections = self.sections(hidden.shape().dimensions()[1], &hidden)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &sections, state)?;
        }

        hidden.apply(&self.norm, state)
    }
}

/// V-JEPA 2 and its classifier: from clips `[B, T, C, H, W]` to each clip's scores, `[B, labels]`.
#[derive(Debug, Clone)]
pub struct Vjepa2ClassifierLayers {
    encoder: Vjepa2Encoder,
    pooler: AttentivePooler,
    classifier: Linear,
}

impl Vjepa2ClassifierLayers {
    pub fn create(schema: &mut Schema<'_>, config: &Vjepa2Config) -> SingeResult<Self> {
        let pooler = AttentivePooler::create(&mut schema.scope("pooler"), config)?;
        Ok(Self {
            encoder: Vjepa2Encoder::create(schema, config)?,
            pooler,
            classifier: schema.linear(
                "classifier",
                config.width,
                config.classifier.label_count()?,
                true,
            )?,
        })
    }
}

impl Layer for Vjepa2ClassifierLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let tokens = input.apply(&self.encoder, state)?;
        self.pooler
            .forward(&tokens, state)?
            .apply(&self.classifier, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "vjepa2",
    formats: &[(Format::Transformers, &["vjepa2"])],
    loaders: &[
        HeadLoader::VideoClassification(load_classifier),
        HeadLoader::VideoFeatureExtraction(load_encoder),
    ],
};

pub fn load_classifier(loader: &mut Loader<'_>) -> Result<VideoClassifier> {
    let config: Vjepa2Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Vjepa2ClassifierLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(VideoClassifier(Classifier::new(model, config.classifier)))
}

/// Loads the encoder alone: a `VJEPA2Model`'s, stored at the checkpoint's root, or a classifier's, stored under `vjepa2`.
pub fn load_encoder(loader: &mut Loader<'_>) -> Result<VideoEncoder> {
    let config: Vjepa2Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Vjepa2Encoder::create(schema, &config).map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let is_bare = loader
        .source()
        .classes()?
        .iter()
        .any(|class| class == "VJEPA2Model");
    let prefix = if is_bare { "" } else { "vjepa2" };
    let model = loader.load_whole(&model, &transformers::encoder_rules(prefix))?;
    Ok(VideoEncoder(model))
}
