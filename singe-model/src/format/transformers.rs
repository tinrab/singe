//! What `transformers` configs share, read into Singe's configs: activation names, rotary fields, the fields of Llama-family decoders and BERT-style encoders, generation settings, and a classifier's labels.

use std::{collections::BTreeMap, result::Result as StdResult};

use serde::{
    Deserialize, Deserializer,
    de::{DeserializeOwned, Error as DeError},
};
use serde_json::Value;
use singe::{
    Error as SingeError,
    generation::GenerationOptions,
    nn::{Activation, Llama3Frequencies, RotaryFrequencies, RotaryLayout, YarnFrequencies},
    shape::spatial::{Spatial2d, Spatial2dI64},
};

use crate::{
    assembly::{DecoderConfig, PostNormEncoderConfig},
    error::{Error, Result},
    head::{ClassifierConfig, DetectionScoring, ScoreFunction, Seq2SeqGeneration},
    source::{Source, token_ids},
};

pub fn parse<T: DeserializeOwned>(value: Value) -> Result<T> {
    serde_json::from_value(value).map_err(Error::json_value("parse a model config"))
}

#[must_use]
pub const fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct SampleExtractor {
    pub sampling_rate: i64,
    #[serde(default = "default_true")]
    pub do_normalize: bool,
}

#[must_use]
pub const fn default_channels() -> i64 {
    3
}

/// Returns the activation a config names, as `transformers` maps these names.
///
/// # Errors
///
/// Returns an error if `name` is not `relu`, `gelu`, `gelu_new`, `gelu_pytorch_tanh`, `silu`, `swish`, `sigmoid`, `tanh`, or `quick_gelu`.
pub fn activation(name: &str) -> Result<Activation> {
    Ok(match name {
        "relu" => Activation::Relu,
        "relu6" => Activation::Relu6,
        "relu2" => Activation::Relu2,
        "gelu" => Activation::GeluExact,
        "gelu_new" | "gelu_pytorch_tanh" => Activation::Gelu,
        "silu" | "swish" => Activation::Silu,
        "sigmoid" => Activation::Sigmoid,
        "tanh" => Activation::Tanh,
        "quick_gelu" => Activation::QuickGelu,
        other => return Err(Error::unsupported(format!("activation {other}"))),
    })
}

/// Reads a config field that names an activation, for `#[serde(deserialize_with = …)]`.
pub fn activation_field<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> StdResult<Activation, D::Error> {
    let name = String::deserialize(deserializer)?;
    activation(&name).map_err(DeError::custom)
}

/// Reads a config field that sizes both sides of an image, written as `[height, width]` or as one side of a square, for `#[serde(deserialize_with = …)]`.
pub fn spatial_field<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> StdResult<Spatial2dI64, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SideValue {
        Square(i64),
        Pair([i64; 2]),
    }

    Ok(match SideValue::deserialize(deserializer)? {
        SideValue::Square(side) => Spatial2d::square(side),
        SideValue::Pair([height, width]) => Spatial2d::new(height, width),
    })
}

/// The rotary fields of a `config.json`: `rope_parameters`, as `transformers` 5 saves them, or the older `rope_theta` and `rope_scaling`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[expect(
    clippy::struct_field_names,
    reason = "the fields are named as the config's keys"
)]
pub struct RotaryFields {
    #[serde(default)]
    rope_parameters: Option<RopeFields>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_scaling: Option<RopeFields>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
struct RopeFields {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    rope_type: Option<String>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    factor: Option<f64>,
    #[serde(default)]
    original_max_position_embeddings: Option<i64>,
    #[serde(default)]
    beta_fast: Option<f64>,
    #[serde(default)]
    beta_slow: Option<f64>,
    #[serde(default)]
    low_freq_factor: Option<f64>,
    #[serde(default)]
    high_freq_factor: Option<f64>,
    #[serde(default)]
    attention_factor: Option<f64>,
    #[serde(default)]
    mscale: Option<f64>,
    #[serde(default)]
    mscale_all_dim: Option<f64>,
    #[serde(default)]
    truncate: Option<bool>,
    #[serde(default)]
    mrope_section: Option<Vec<i64>>,
    #[serde(default)]
    mrope_interleaved: Option<bool>,
}

/// The theta `transformers` falls back on.
const DEFAULT_ROTARY_THETA: f64 = 10_000.0;

impl RotaryFields {
    fn fields(&self) -> Option<&RopeFields> {
        self.rope_parameters.as_ref().or(self.rope_scaling.as_ref())
    }

    /// Returns the fraction of each head's features that the rotation turns, from `partial_rotary_factor`, or 1 when the config gives none.
    /// The rotation turns the leading features, and the rest pass through.
    #[must_use]
    pub fn rotated_fraction(&self) -> f64 {
        self.fields()
            .and_then(|fields| fields.partial_rotary_factor)
            .unwrap_or(1.0)
    }

    #[must_use]
    pub fn mrope_section(&self) -> Option<&[i64]> {
        self.fields()
            .and_then(|fields| fields.mrope_section.as_deref())
    }

    #[must_use]
    pub fn mrope_interleaved(&self) -> bool {
        self.fields()
            .and_then(|fields| fields.mrope_interleaved)
            .unwrap_or(false)
    }

    pub fn frequencies(&self) -> Result<RotaryFrequencies> {
        let fields = self.fields();
        let theta = fields
            .and_then(|fields| fields.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(DEFAULT_ROTARY_THETA);
        let frequencies = RotaryFrequencies::new(theta);
        let Some(fields) = fields else {
            return Ok(frequencies);
        };
        match fields.kind.as_deref().or(fields.rope_type.as_deref()) {
            None | Some("default") => Ok(frequencies),
            Some("yarn") => Ok(frequencies.with_yarn(YarnFrequencies {
                factor: fields
                    .factor
                    .ok_or_else(|| Error::unsupported("yarn without a factor"))?,
                original_max_positions: fields
                    .original_max_position_embeddings
                    .ok_or_else(|| Error::unsupported("yarn without its original length"))?,
                beta_fast: fields.beta_fast.unwrap_or(32.0),
                beta_slow: fields.beta_slow.unwrap_or(1.0),
                attention_factor: fields.attention_factor,
                mscale: fields.mscale,
                mscale_all_dim: fields.mscale_all_dim,
                truncate: fields.truncate.unwrap_or(true),
            })),
            Some("llama3") => {
                let field = |value: Option<f64>, name: &str| {
                    value.ok_or_else(|| Error::unsupported(format!("llama3 rotary without {name}")))
                };
                Ok(frequencies.with_llama3(Llama3Frequencies {
                    factor: field(fields.factor, "a factor")?,
                    low_frequency_factor: field(fields.low_freq_factor, "a low frequency factor")?,
                    high_frequency_factor: field(
                        fields.high_freq_factor,
                        "a high frequency factor",
                    )?,
                    original_max_positions: fields.original_max_position_embeddings.ok_or_else(
                        || Error::unsupported("llama3 rotary without its original length"),
                    )?,
                }))
            }
            Some(other) => Err(Error::unsupported(format!("rotary type {other}"))),
        }
    }
}

/// Returns [`Activation::Silu`], which `transformers` gives a Llama-family decoder whose config leaves `hidden_act` out.
const fn silu() -> Activation {
    Activation::Silu
}

fn decoder_activation<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> StdResult<Activation, D::Error> {
    Option::<String>::deserialize(deserializer)?.map_or(Ok(Activation::Silu), |name| {
        activation(&name).map_err(DeError::custom)
    })
}

/// The fields Llama-family configs share.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DecoderFields {
    pub vocab_size: i64,
    pub hidden_size: i64,
    #[serde(default)]
    pub intermediate_size: Option<i64>,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    /// The key and value heads, as many as the query heads when absent.
    #[serde(default)]
    pub num_key_value_heads: Option<i64>,
    /// Each head's width; when absent, [`DecoderDefaults::head_dim`], or else the hidden size over the heads.
    #[serde(default)]
    pub head_dim: Option<i64>,
    pub rms_norm_eps: f32,
    /// The activation, [`Activation::Silu`] when absent or `null`.
    #[serde(default = "silu", deserialize_with = "decoder_activation")]
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    /// Whether the output head shares the embedding's weight; when absent, [`DecoderDefaults::tied`], since the default differs by model.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    #[serde(flatten)]
    pub rotary: RotaryFields,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// Granite-4.0-H names it `attention`.
    #[serde(alias = "attention")]
    FullAttention,
    SlidingAttention,
    /// A layer that carries a state, which Granite-4.0-H names `mamba`.
    #[serde(alias = "mamba")]
    LinearAttention,
}

/// What a model's `transformers` config class gives the fields its config leaves out, where models differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderDefaults {
    /// Whether the output head shares the embedding's weight.
    pub tied: bool,
    /// Each head's width, or `None` for the hidden size over the heads, as Llama's.
    pub head_dim: Option<i64>,
}

impl DecoderFields {
    /// Checks that every layer in `layer_types` is full attention, since a definition would compute any other layer type as full attention.
    ///
    /// # Errors
    ///
    /// Returns an error if `layer_types` names a layer other than full attention.
    pub fn refuse_sliding_layers(&self) -> Result<()> {
        if self
            .layer_types
            .iter()
            .flatten()
            .any(|&layer| layer != LayerType::FullAttention)
        {
            return Err(Error::unsupported("a layer other than full attention"));
        }
        Ok(())
    }

    pub fn layer_types(&self, default: impl Fn(usize) -> LayerType) -> Result<Vec<LayerType>> {
        let layers = usize::try_from(self.num_hidden_layers)
            .map_err(|_| SingeError::out_of_range("num_hidden_layers"))?;
        match &self.layer_types {
            Some(types) if types.len() == layers => Ok(types.clone()),
            Some(types) => Err(Error::unsupported(format!(
                "{} layer types for {layers} layers",
                types.len()
            ))),
            None => Ok((0..layers).map(default).collect()),
        }
    }

    /// Returns the decoder's config, filling the head width and the tying the config leaves out from `defaults`.
    pub fn config(&self, defaults: DecoderDefaults) -> Result<DecoderConfig> {
        let heads = self.num_attention_heads;
        let head_dim = self
            .head_dim
            .or(defaults.head_dim)
            .unwrap_or(self.hidden_size / heads);
        Ok(DecoderConfig {
            vocab_size: self.vocab_size,
            width: self.hidden_size,
            inner: self.intermediate_size,
            layers: self.num_hidden_layers,
            heads,
            kv_heads: self.num_key_value_heads.unwrap_or(heads),
            head_dim,
            rotary_dim: rotated_features(head_dim, self.rotary.rotated_fraction())?,
            eps: self.rms_norm_eps,
            activation: self.hidden_act,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            tied: self.tie_word_embeddings.unwrap_or(defaults.tied),
            head_norms: false,
            frequencies: self.rotary.frequencies()?,
            rotary_layout: RotaryLayout::HalfSplit,
        })
    }
}

fn absolute() -> String {
    "absolute".to_owned()
}

/// The fields BERT-style encoder configs share, such as BERT's and `RoBERTa`'s.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EncoderFields {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    #[serde(deserialize_with = "activation_field")]
    pub hidden_act: Activation,
    pub max_position_embeddings: i64,
    pub type_vocab_size: i64,
    pub layer_norm_eps: f32,
    #[serde(default = "absolute")]
    pub position_embedding_type: String,
    #[serde(default)]
    pub is_decoder: bool,
}

impl EncoderFields {
    /// Returns the encoder's config, whose positions start at `first_position`.
    ///
    /// # Errors
    ///
    /// Returns an error if `position_embedding_type` is not `absolute`, or `is_decoder` is set.
    pub fn config(&self, first_position: i64) -> Result<PostNormEncoderConfig> {
        // A checkpoint using one of these would load and then compute something else.
        if self.position_embedding_type != "absolute" {
            return Err(Error::unsupported("relative position embeddings"));
        }

        if self.is_decoder {
            return Err(Error::unsupported("an encoder configured as a decoder"));
        }
        Ok(PostNormEncoderConfig {
            vocab_size: self.vocab_size,
            width: self.hidden_size,
            layers: self.num_hidden_layers,
            heads: self.num_attention_heads,
            inner: self.intermediate_size,
            activation: self.hidden_act,
            max_positions: self.max_position_embeddings,
            first_position,
            token_types: self.type_vocab_size,
            eps: self.layer_norm_eps,
            keys: None,
        })
    }
}

/// Returns the features of a head of `head_dim` that a rotation of `fraction` of it turns, as `transformers` rounds them down.
fn rotated_features(head_dim: i64, fraction: f64) -> Result<i64> {
    let features = (head_dim as f64 * fraction) as i64;
    if features <= 0 || features > head_dim || features % 2 != 0 {
        return Err(Error::unsupported(format!(
            "a rotation of {features} of {head_dim} features"
        )));
    }
    Ok(features)
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct GenerationFields {
    #[serde(default)]
    suppress_tokens: Vec<i64>,
    #[serde(default)]
    begin_suppress_tokens: Vec<i64>,
}

/// Returns how `transformers`' `generate` continues a prompt for a snapshot: its stop tokens end a sequence, `generation_config.json`'s `suppress_tokens` are never chosen, and its `begin_suppress_tokens` are not chosen first.
/// The options generate no tokens until a caller sets `max_new_tokens`.
pub fn generation_options(source: &Source) -> Result<GenerationOptions> {
    let fields: GenerationFields = source.json("generation_config.json")?;
    Ok(GenerationOptions {
        stop_tokens: source.stop_tokens()?,
        suppressed_tokens: fields.suppress_tokens,
        first_suppressed_tokens: fields.begin_suppress_tokens,
        ..GenerationOptions::default()
    })
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Seq2SeqFields {
    decoder_start_token_id: Option<i64>,
    eos_token_id: Option<Value>,
    forced_bos_token_id: Option<i64>,
    forced_eos_token_id: Option<i64>,
    max_length: Option<usize>,
    min_length: Option<usize>,
    no_repeat_ngram_size: Option<usize>,
    encoder_no_repeat_ngram_size: Option<usize>,
    repetition_penalty: Option<f32>,
    num_beams: Option<usize>,
    length_penalty: Option<f32>,
    early_stopping: Option<Value>,
    bad_words_ids: Option<Vec<Vec<i64>>>,
}

/// Returns how `transformers`' `generate` runs an encoder-decoder snapshot: `generation_config.json`'s settings, or where a snapshot has none, the model config's, with a composite's decoder's filling what its own leaves out.
/// Settings left out take `generate`'s defaults: at most 20 tokens, no fewer than 0, and one beam.
///
/// # Errors
///
/// Returns an error if the config it reads cannot be read or parsed, if `early_stopping` is neither a boolean nor `"never"` with a length penalty of at most 0, or if `bad_words_ids` bans a sequence longer than one token.
pub fn seq2seq_generation(source: &Source) -> Result<Seq2SeqGeneration> {
    let fields: Seq2SeqFields =
        if let Some(fields) = source.optional_json::<Seq2SeqFields>("generation_config.json")? {
            fields
        } else {
            let config: Value = source.json("config.json")?;
            let mut merged = config.clone();
            if let (Some(merged), Some(decoder)) =
                (merged.as_object_mut(), config["decoder"].as_object())
            {
                for (key, value) in decoder {
                    if merged.get(key).is_none_or(Value::is_null) {
                        merged.insert(key.clone(), value.clone());
                    }
                }
            }
            parse(merged)?
        };
    // `"never"` searches as `false` does when the length penalty is not positive.
    let early_stopping = match &fields.early_stopping {
        None | Some(Value::Null | Value::Bool(false)) => false,
        Some(Value::Bool(true)) => true,
        Some(Value::String(never))
            if never == "never" && fields.length_penalty.unwrap_or(1.0) <= 0.0 =>
        {
            false
        }
        Some(other) => return Err(Error::unsupported(format!("early stopping {other}"))),
    };
    let stop_tokens = match fields.eos_token_id.as_ref().and_then(token_ids) {
        Some(tokens) => tokens,
        None => source.stop_tokens()?,
    };
    // A banned token is a one-token sequence; longer ones would ban only their last token after the rest.
    let banned = fields
        .bad_words_ids
        .unwrap_or_default()
        .into_iter()
        .map(|words| match words.as_slice() {
            [token] => Ok(*token),
            _ => Err(Error::unsupported("banned sequences longer than one token")),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Seq2SeqGeneration {
        decoder_start: fields.decoder_start_token_id,
        stop_tokens,
        forced_first: fields.forced_bos_token_id,
        forced_last: fields.forced_eos_token_id,
        max_length: fields.max_length.unwrap_or(20),
        min_length: fields.min_length.unwrap_or(0),
        no_repeat_ngram: fields.no_repeat_ngram_size.unwrap_or(0),
        encoder_no_repeat_ngram: fields.encoder_no_repeat_ngram_size.unwrap_or(0),
        repetition_penalty: fields.repetition_penalty.unwrap_or(1.0),
        banned,
        beams: fields.num_beams.unwrap_or(1),
        length_penalty: fields.length_penalty.unwrap_or(1.0),
        early_stopping,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ClassifierFields {
    id2label: Option<BTreeMap<String, String>>,
    num_labels: Option<usize>,
    problem_type: Option<String>,
    #[serde(default)]
    sentence_transformers: Option<CrossEncoderFields>,
    /// Where `sentence-transformers` before 4.0 named a cross-encoder's activation.
    sbert_ce_default_activation_function: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct CrossEncoderFields {
    activation_fn: Option<String>,
}

/// Returns the score function of a `sentence-transformers` activation's class path, such as `torch.nn.modules.activation.Sigmoid`.
fn activation_function(path: &str) -> Result<ScoreFunction> {
    match path.rsplit('.').next() {
        Some("Sigmoid") => Ok(ScoreFunction::Sigmoid),
        Some("Identity") => Ok(ScoreFunction::None),
        _ => Err(Error::unsupported(format!(
            "cross-encoder activation {path}"
        ))),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct DetectionFields {
    model_type: Option<String>,
    num_queries: Option<usize>,
}

/// Returns how a detector's scores become detections, as each image processor's `post_process_object_detection` picks them: Deformable DETR keeps the best 100 sigmoid scores over all queries and labels, RT-DETR as many as it has queries, and the others take a softmax with a no-object score, as DETR does.
///
/// # Errors
///
/// Returns an error if `config.json` cannot be read or parsed, or if an RT-DETR config has no `num_queries`.
pub fn detection_scoring(source: &Source) -> Result<DetectionScoring> {
    let fields: DetectionFields = source.json("config.json")?;
    Ok(match fields.model_type.as_deref() {
        Some("deformable_detr") => DetectionScoring::SigmoidTopK(100),
        Some("rt_detr" | "rt_detr_v2") => DetectionScoring::SigmoidTopK(
            fields
                .num_queries
                .ok_or_else(|| Error::unsupported("an RT-DETR config without `num_queries`"))?,
        ),
        _ => DetectionScoring::NoObject,
    })
}

/// Reads a classifier's labels from `id2label` in index order, or `LABEL_{i}` for each of `num_labels` (2 when absent) as `transformers` names them.
/// The classifier's scores stay as they are for a regression, go through a sigmoid for a multi-label problem or a single label, and through a softmax otherwise, as the text-classification pipeline chooses.
/// A `sentence-transformers` cross-encoder's config also names the function its ranking scores go through.
///
/// # Errors
///
/// Returns an error if `config.json` cannot be read or parsed, if an `id2label` key is not an index or the keys skip an index, if `num_labels` is 0, or if the cross-encoder's activation is neither `Sigmoid` nor `Identity`.
pub fn classifier_config(source: &Source) -> Result<ClassifierConfig> {
    let fields: ClassifierFields = source.json("config.json")?;
    let labels = match fields.id2label {
        Some(labels) if !labels.is_empty() => {
            let mut labels = labels
                .into_iter()
                .map(|(index, label)| {
                    let index = index.parse::<usize>().map_err(|_| {
                        Error::unsupported("an `id2label` key that is not an index")
                    })?;
                    Ok((index, label))
                })
                .collect::<Result<Vec<_>>>()?;
            labels.sort_unstable_by_key(|&(index, _)| index);
            if labels
                .iter()
                .enumerate()
                .any(|(order, &(index, _))| order != index)
            {
                return Err(Error::unsupported("an `id2label` that skips an index"));
            }
            labels.into_iter().map(|(_, label)| label).collect()
        }
        _ => (0..fields.num_labels.unwrap_or(2))
            .map(|index| format!("LABEL_{index}"))
            .collect::<Vec<_>>(),
    };
    let function = match fields.problem_type.as_deref() {
        Some("regression") => ScoreFunction::None,
        Some("multi_label_classification") => ScoreFunction::Sigmoid,
        _ if labels.len() == 1 => ScoreFunction::Sigmoid,
        _ => ScoreFunction::Softmax,
    };
    let ranking_function = fields
        .sentence_transformers
        .and_then(|fields| fields.activation_fn)
        .or(fields.sbert_ce_default_activation_function)
        .map(|path| activation_function(&path))
        .transpose()?;
    ClassifierConfig::create(labels, function, ranking_function)
}
