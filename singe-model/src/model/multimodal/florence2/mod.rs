//! Florence-2: a `DaViT` backbone, whose blocks alternate attention within windows and attention between channel groups, projected to 577 rows that take the place of a prompt's image placeholders before a BART encoder-decoder.
//! A task token, such as `<CAPTION>` or `<OD>`, stands for a fixed question, and answers keep the location tokens that boxes are written in.

pub mod answers;
mod transformers;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, Conv2d, Conv2dConfig, LayerNorm, Linear, Mlp, ModelState, StateKey},
    shape::spatial::{Padding, Padding2d, Spatial2d},
};

use crate::{
    assembly::{
        Attention, AttentionSpec, BartDecoder, BartEncoder, BartStackConfig, PYTORCH_NORM_EPS,
        merge_windows, partition_windows, to_tokens,
    },
    error::{Error, Result},
    format::{Loader, transformers::seq2seq_generation},
    head::{Boxed, HeadLoader, Seq2SeqDecoding, Seq2SeqInput, Seq2SeqLayers, Seq2SeqLm},
    model::ModelEntry,
    schema::Schema,
    source::{Format, Source},
};

/// One `DaViT` stage's sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DavitStage {
    pub width: i64,
    pub depth: i64,
    pub heads: i64,
    pub groups: i64,
    /// The kernel of the convolution that starts the stage.
    pub kernel: i64,
    pub stride: i64,
    pub padding: i64,
    /// Whether the stage's layer norm comes before its convolution rather than after.
    pub norm_first: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Florence2Config {
    pub channels: i64,
    pub stages: Vec<DavitStage>,
    pub window: i64,
    pub mlp_ratio: f64,
    pub activation: Activation,
    /// The rows and columns of the learned grid of positions added to the last map.
    pub max_positions: i64,
    pub projection: i64,
    pub encoder: BartStackConfig,
    pub decoder: BartStackConfig,
    pub image_token: i64,
}

/// Returns `map`, `[B, C, H, W]`, layer-normed over its channels.
fn channel_normed(map: &Tensor, norm: &LayerNorm, state: &ModelState) -> SingeResult<Tensor> {
    map.permute(&[0, 2, 3, 1])?
        .apply(norm, state)?
        .permute(&[0, 3, 1, 2])
}

/// How a half-block attends: within windows of the map, or between the channels of each group.
#[derive(Debug, Clone, Copy)]
enum Mixing {
    Windows { window: i64 },
    Channels { groups: i64 },
}

/// A `DaViT` half-block: a depthwise convolution added back, attention of the normed tokens added back, a depthwise convolution added back, and a feed-forward of the normed tokens added back.
#[derive(Debug, Clone)]
struct HalfBlock {
    first_conv: Conv2d,
    attention_norm: LayerNorm,
    attention: Attention,
    second_conv: Conv2d,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
}

impl HalfBlock {
    fn create(
        schema: &mut Schema<'_>,
        config: &Florence2Config,
        stage: DavitStage,
    ) -> SingeResult<Self> {
        let width = stage.width;
        let depthwise = |schema: &mut Schema<'_>, name: &str| {
            schema.conv2d(
                name,
                width,
                width,
                3,
                Conv2dConfig {
                    padding: Padding2d::symmetric(1, 1),
                    groups: width,
                    ..Conv2dConfig::default()
                },
                true,
            )
        };
        let inner = (width as f64 * config.mlp_ratio) as i64;
        Ok(Self {
            first_conv: depthwise(schema, "first_conv")?,
            attention_norm: schema.layer_norm("attention_norm", width, PYTORCH_NORM_EPS)?,
            attention: Attention::create(
                &mut schema.scope("attention"),
                AttentionSpec::square(width, stage.heads),
            )?,
            second_conv: depthwise(schema, "second_conv")?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, PYTORCH_NORM_EPS)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, inner, width],
                config.activation,
                true,
            )?,
        })
    }

    fn forward(&self, map: &Tensor, mixing: Mixing, state: &ModelState) -> SingeResult<Tensor> {
        let &[batch, channels, height, width] = map.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a DaViT map, [B, C, H, W]"));
        };
        let map = map.add(&map.apply(&self.first_conv, state)?)?;
        let tokens = to_tokens(&map)?;
        let normed = tokens.apply(&self.attention_norm, state)?;
        let attended = match mixing {
            Mixing::Windows { window } => {
                self.windowed(&normed, [batch, height, width, channels], window, state)?
            }
            Mixing::Channels { groups } => self.grouped(&normed, groups, state)?,
        };
        let tokens = tokens.add(&attended)?;
        let map = tokens
            .transpose(1, 2)?
            .reshape([batch, channels, height, width])?;
        let map = map.add(&map.apply(&self.second_conv, state)?)?;
        let tokens = to_tokens(&map)?;
        let tokens = tokens.add(
            &tokens
                .apply(&self.feed_forward_norm, state)?
                .apply(&self.feed_forward, state)?,
        )?;
        tokens
            .transpose(1, 2)?
            .reshape([batch, channels, height, width])
    }

    /// Attends within `window × window` windows of the tokens, `[B, H · W, C]`, the map padded with zeros at its bottom and right to whole windows, as `Florence2VisionWindowAttention` does.
    fn windowed(
        &self,
        tokens: &Tensor,
        [batch, height, width, channels]: [i64; 4],
        window: i64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let map = tokens.reshape([batch, height, width, channels])?;
        let padded = |side: i64| (window - side % window) % window;
        let (bottom, right) = (padded(height), padded(width));
        let map = if bottom > 0 || right > 0 {
            map.pad(
                &[
                    Padding::ZERO,
                    Padding::new(0, bottom),
                    Padding::new(0, right),
                    Padding::ZERO,
                ],
                0.0,
            )?
        } else {
            map
        };
        let (full_height, full_width) = (height + bottom, width + right);
        let windows = partition_windows(&map, window)?;
        let attended = self
            .attention
            .forward(&windows, &windows, &windows, state)?;
        merge_windows(&attended, window, batch, (full_height, full_width))?
            .narrow(1, 0, height)?
            .narrow(2, 0, width)?
            .reshape([batch, height * width, channels])
    }

    /// Attends between the channels of each of `groups` groups, each channel's tokens its features, scaled by the tokens' count, as `Florence2VisionChannelAttention` does.
    fn grouped(&self, tokens: &Tensor, groups: i64, state: &ModelState) -> SingeResult<Tensor> {
        let &[batch, count, width] = tokens.shape().dimensions() else {
            return Err(SingeError::invalid_argument("DaViT tokens, [B, N, C]"));
        };
        let per_group = width / groups;
        let split = |linear: &Linear| -> SingeResult<Tensor> {
            tokens
                .apply(linear, state)?
                .reshape([batch, count, groups, per_group])?
                .permute(&[0, 2, 3, 1])
        };
        let (query, key, value) = (
            split(self.attention.query())?,
            split(self.attention.key())?,
            split(self.attention.value())?,
        );
        let weights = query
            .matmul(&key.transpose(2, 3)?)?
            .mul_scalar((count as f64).powf(-0.5))?
            .softmax(-1)?;
        weights
            .matmul(&value)?
            .permute(&[0, 3, 1, 2])?
            .reshape([batch, count, width])?
            .apply(self.attention.output(), state)
    }
}

/// A `DaViT` stage: its starting convolution and norm, then blocks of a window half and a channel half.
#[derive(Debug, Clone)]
struct Stage {
    conv: Conv2d,
    norm: LayerNorm,
    blocks: Vec<[HalfBlock; 2]>,
    config: DavitStage,
}

/// Florence-2's vision side: from images, `[B, 3, H, W]`, to the rows that take the image placeholders' place, `[B, 1 + N, P]`.
#[derive(Debug, Clone)]
struct VisionLayers {
    stages: Vec<Stage>,
    rows: StateKey,
    columns: StateKey,
    temporal: StateKey,
    projection: Linear,
    norm: LayerNorm,
    window: i64,
}

impl VisionLayers {
    fn create(schema: &mut Schema<'_>, config: &Florence2Config) -> SingeResult<Self> {
        let mut input = config.channels;
        let mut stages = Vec::with_capacity(config.stages.len());
        for (index, &stage) in config.stages.iter().enumerate() {
            let mut scope = schema.scope(format!("vision.stages.{index}"));
            let conv = scope.conv2d(
                "conv",
                input,
                stage.width,
                stage.kernel,
                Conv2dConfig {
                    stride: Spatial2d::square(stage.stride),
                    padding: Padding2d::symmetric(stage.padding, stage.padding),
                    ..Conv2dConfig::default()
                },
                true,
            )?;
            let norm = scope.layer_norm(
                "norm",
                if stage.norm_first { input } else { stage.width },
                PYTORCH_NORM_EPS,
            )?;
            let blocks = (0..stage.depth)
                .map(|block| {
                    Ok([
                        HalfBlock::create(
                            &mut scope.scope(format!("blocks.{block}.spatial")),
                            config,
                            stage,
                        )?,
                        HalfBlock::create(
                            &mut scope.scope(format!("blocks.{block}.channel")),
                            config,
                            stage,
                        )?,
                    ])
                })
                .collect::<SingeResult<Vec<_>>>()?;
            stages.push(Stage {
                conv,
                norm,
                blocks,
                config: stage,
            });
            input = stage.width;
        }

        let last = input;
        Ok(Self {
            stages,
            rows: schema.parameter("projector.rows", [config.max_positions, last / 2])?,
            columns: schema
                .parameter("projector.columns", [config.max_positions, last - last / 2])?,
            temporal: schema.parameter("projector.temporal", [1, last])?,
            projection: schema.linear("projector.projection", last, config.projection, false)?,
            norm: schema.layer_norm("projector.norm", config.projection, PYTORCH_NORM_EPS)?,
            window: config.window,
        })
    }

    fn forward(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut map = pixels.clone();
        for stage in &self.stages {
            map = if stage.config.norm_first {
                channel_normed(&map, &stage.norm, state)?.apply(&stage.conv, state)?
            } else {
                channel_normed(&map.apply(&stage.conv, state)?, &stage.norm, state)?
            };
            for [spatial, channel] in &stage.blocks {
                map = spatial.forward(
                    &map,
                    Mixing::Windows {
                        window: self.window,
                    },
                    state,
                )?;
                map = channel.forward(
                    &map,
                    Mixing::Channels {
                        groups: stage.config.groups,
                    },
                    state,
                )?;
            }
        }

        // Each cell's learned column and row positions, concatenated, and the first frame's temporal embedding.
        let &[_, channels, height, width] = map.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a DaViT map, [B, C, H, W]"));
        };
        let columns = state.tensor(&self.columns)?.narrow(0, 0, width)?;
        let rows = state.tensor(&self.rows)?.narrow(0, 0, height)?;
        let positions = Tensor::concat(
            &[
                columns.unsqueeze(0)?.expand(Shape::create([
                    height,
                    width,
                    columns.shape().dimensions()[1],
                ])?)?,
                rows.unsqueeze(1)?.expand(Shape::create([
                    height,
                    width,
                    rows.shape().dimensions()[1],
                ])?)?,
            ],
            2,
        )?
        .reshape([1, height * width, channels])?;
        let tokens = to_tokens(&map)?
            .add(&positions)?
            .add(&state.tensor(&self.temporal)?.unsqueeze(0)?)?;
        let pooled = tokens.mean(&[1], true)?;
        Tensor::concat(&[pooled, tokens], 1)?
            .apply(&self.projection, state)?
            .apply(&self.norm, state)
    }
}

#[derive(Debug, Clone)]
pub struct Florence2Layers {
    vision: VisionLayers,
    encoder: BartEncoder,
    decoder: Arc<dyn Seq2SeqDecoding>,
    image_token: i64,
}

impl Florence2Layers {
    pub fn create(schema: &mut Schema<'_>, config: &Florence2Config) -> SingeResult<Self> {
        let vocabulary = config.decoder.vocab_size;
        let width = config.decoder.width;
        let tokens = schema.parameter("tokens.weight", [vocabulary, width])?;
        let head = Linear::new(schema.tie("head.weight", &tokens)?, None);
        Ok(Self {
            vision: VisionLayers::create(schema, config)?,
            encoder: BartEncoder::create(
                &mut schema.scope("encoder"),
                &config.encoder,
                Some(&tokens),
            )?,
            decoder: Arc::new(BartDecoder::create(
                &mut schema.scope("decoder"),
                &config.decoder,
                Some(&tokens),
                head,
                None,
            )?),
            image_token: config.image_token,
        })
    }
}

impl Seq2SeqLayers for Florence2Layers {
    fn encode(&self, input: Seq2SeqInput<'_>, state: &ModelState) -> SingeResult<Tensor> {
        let Seq2SeqInput::Prompted { tokens, pixels } = input else {
            return Err(SingeError::invalid_argument(
                "Florence-2 reads an image with a prompt",
            ));
        };
        let images = self.vision.forward(pixels, state)?;
        let rows = images.shape().dimensions()[1];
        // The image's rows take the place of the prompt's run of placeholders, as `masked_scatter` writes them.
        let ids = tokens.to_vec::<i64>()?;
        let first = ids
            .iter()
            .position(|&id| id == self.image_token)
            .ok_or_else(|| {
                SingeError::invalid_argument("a Florence-2 prompt without image placeholders")
            })?;
        let start = i64::try_from(first).map_err(|_| SingeError::out_of_range("prompt"))?;
        let count = ids.iter().filter(|&&id| id == self.image_token).count();
        if i64::try_from(count).ok() != Some(rows)
            || ids[first..first + count]
                .iter()
                .any(|&id| id != self.image_token)
        {
            return Err(SingeError::invalid_argument(
                "a Florence-2 prompt whose placeholders are not one run of the image's rows",
            ));
        }

        let embedded = self.encoder.embed_tokens(tokens, state)?;
        let length = embedded.shape().dimensions()[1];
        let spliced = Tensor::concat(
            &[
                embedded.narrow(1, 0, start)?,
                images.cast(embedded.data_type())?,
                embedded.narrow(1, start + rows, length - start - rows)?,
            ],
            1,
        )?;
        self.encoder.forward_embedded(&spliced, state)
    }

    fn decoder(&self) -> Arc<dyn Seq2SeqDecoding> {
        Arc::clone(&self.decoder)
    }
}

/// The tokens of a task with no input of its own, and the question it stands for.
const TASKS: &[(&str, &str)] = &[
    ("<OCR>", "What is the text in the image?"),
    (
        "<OCR_WITH_REGION>",
        "What is the text in the image, with regions?",
    ),
    ("<CAPTION>", "What does the image describe?"),
    (
        "<DETAILED_CAPTION>",
        "Describe in detail what is shown in the image.",
    ),
    (
        "<MORE_DETAILED_CAPTION>",
        "Describe with a paragraph what is shown in the image.",
    ),
    (
        "<OD>",
        "Locate the objects with category name in the image.",
    ),
    (
        "<DENSE_REGION_CAPTION>",
        "Locate the objects in the image, with their descriptions.",
    ),
    (
        "<REGION_PROPOSAL>",
        "Locate the region proposals in the image.",
    ),
];

/// The tokens of a task that reads the rest of the prompt, and the question it goes into.
const INPUT_TASKS: &[(&str, &str)] = &[
    (
        "<CAPTION_TO_PHRASE_GROUNDING>",
        "Locate the phrases in the caption: {input}",
    ),
    (
        "<REFERRING_EXPRESSION_SEGMENTATION>",
        "Locate {input} in the image with mask",
    ),
    (
        "<REGION_TO_SEGMENTATION>",
        "What is the polygon mask of region {input}",
    ),
    (
        "<OPEN_VOCABULARY_DETECTION>",
        "Locate {input} in the image.",
    ),
    ("<REGION_TO_CATEGORY>", "What is the region {input}?"),
    (
        "<REGION_TO_DESCRIPTION>",
        "What does the region {input} describe?",
    ),
    ("<REGION_TO_OCR>", "What text is in the region {input}?"),
];

/// Florence-2's prompts, as `Florence2Processor` builds them: the image's placeholders, then the question a task token stands for, or the text itself, between the tokenizer's start and end tokens.
#[derive(Debug, Clone)]
pub struct Florence2Prompts {
    image_rows: usize,
    /// The phrases phrase grounding never reports, from the processor's config.
    banned_phrases: Vec<String>,
}

impl Florence2Prompts {
    /// Returns the prompts of a snapshot whose processor is `Florence2Processor`, or `None` for another.
    pub fn load(source: &Source) -> Result<Option<Self>> {
        let processor: Value = source
            .optional_json("preprocessor_config.json")?
            .unwrap_or_default();
        if processor["processor_class"].as_str() != Some("Florence2Processor") {
            return Ok(None);
        }

        let rows = processor["image_seq_length"]
            .as_u64()
            .and_then(|rows| usize::try_from(rows).ok())
            .ok_or_else(|| {
                Error::unsupported("a Florence-2 processor without `image_seq_length`")
            })?;
        let banned_phrases = source
            .optional_json::<Value>("processor_config.json")?
            .and_then(|processor| {
                processor["post_processor_config"]["phrase_grounding"]["banned_grounding_tokens"]
                    .as_array()
                    .map(|phrases| {
                        phrases
                            .iter()
                            .filter_map(|phrase| phrase.as_str().map(str::to_owned))
                            .collect()
                    })
            })
            .unwrap_or_default();
        Ok(Some(Self {
            image_rows: rows,
            banned_phrases,
        }))
    }

    /// Returns the text to tokenize, without the tokenizer's own special tokens, for `prompt`, a task token or a question.
    pub fn text(&self, prompt: &str) -> Result<String> {
        let question = match TASKS.iter().find(|&&(task, _)| prompt.contains(task)) {
            Some(&(task, _)) if prompt != task => {
                return Err(Error::invalid(format!(
                    "the task token {task} with other text"
                )));
            }
            Some(&(_, asked)) => asked.to_owned(),
            None => prompt.to_owned(),
        };
        let question = match INPUT_TASKS
            .iter()
            .find(|&&(task, _)| question.contains(task))
        {
            Some(&(task, asked)) => asked.replace("{input}", question.replace(task, "").trim()),
            None => question,
        };
        Ok(format!(
            "{}<s>{question}</s>",
            "<image>".repeat(self.image_rows)
        ))
    }

    /// Returns the answer to `prompt` parsed for its task, on an image of `width × height`, or `None` when its task's answer is text.
    #[must_use]
    pub fn parsed(
        &self,
        prompt: &str,
        answer: &str,
        size: (u32, u32),
    ) -> Option<answers::ParsedAnswer> {
        let task = TASKS
            .iter()
            .map(|&(task, _)| task)
            .chain(INPUT_TASKS.iter().map(|&(task, _)| task))
            .find(|task| prompt.starts_with(task))?;
        answers::parsed(task, answer, size, &self.banned_phrases)
    }

    /// Returns a generated answer, decoded with its special tokens, without the start, end, and padding tokens and the space around it, as `post_process_generation` cleans it before parsing.
    #[must_use]
    pub fn answer(text: &str) -> String {
        text.replace("<s>", "")
            .replace("</s>", "")
            .replace("<pad>", "")
            .trim()
            .to_owned()
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "florence2",
    formats: &[(Format::Transformers, &["florence2"])],
    loaders: &[HeadLoader::Seq2SeqLm(load_seq2seq)],
};

pub fn load_seq2seq(loader: &mut Loader<'_>) -> Result<Seq2SeqLm> {
    let config: Florence2Config = loader.config(transformers::config)?;
    let generation = seq2seq_generation(loader.source())?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Florence2Layers::create(schema, &config)
            .map(|layers| Boxed::<dyn Seq2SeqLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(Seq2SeqLm::new(model, generation))
}
