//! Heads over a text encoder, and the one loader every encoder model shares.
//! A model gives its `transformers` config and rules, its encoder's layers, and its heads' layouts; each head registers its own layers and reads its own config.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use singe::{
    Result as SingeResult,
    nn::{Activation, Layer, Model, StateKey, TokenBatch},
};

use crate::{
    assembly::{PostNormEncoder, PostNormEncoderConfig},
    error::Result,
    format::{Loader, Rules},
    head::{
        ClassifierLayout, Head, HeadLoader, LoadHead, MaskedLm, QuestionAnswerer,
        SequenceClassifier, TextEncoder, TokenClassifier,
    },
    schema::Schema,
    source::Source,
};

/// How a model lays out the shared heads it serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadLayouts {
    pub classifier: ClassifierLayout,
    /// The activation of the masked language model's projection, or the encoder's when absent.
    pub masked_lm_activation: Option<Activation>,
}

/// What heads read of a text encoder beside its layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncoderSizes {
    pub width: i64,
    pub vocab_size: i64,
    pub eps: f32,
    pub activation: Activation,
}

/// A text encoder heads sit on: its layers, from tokens to each token's last hidden state, `[B, T, W]`, its sizes, and its token table, which a masked language model's output may share.
#[derive(Debug)]
pub struct EncoderBody {
    pub layers: Box<dyn Layer<TokenBatch>>,
    pub sizes: EncoderSizes,
    pub tokens: StateKey,
}

/// Creates `assembly`'s post-norm encoder as the body heads sit on.
pub fn post_norm_body(
    schema: &mut Schema<'_>,
    config: &PostNormEncoderConfig,
) -> SingeResult<EncoderBody> {
    let encoder = PostNormEncoder::create(schema, config)?;
    Ok(EncoderBody {
        sizes: EncoderSizes {
            width: config.width,
            vocab_size: config.vocab_size,
            eps: config.eps,
            activation: config.activation,
        },
        tokens: encoder.tokens().clone(),
        layers: Box::new(encoder),
    })
}

/// A head that sits on a text encoder: what it reads from a `transformers` config, the layers it registers beside the encoder, and what it wraps the loaded model in.
pub trait EncoderHead: LoadHead {
    type Config: Serialize + DeserializeOwned;

    type Layers;

    /// Reads what the head adds to a `transformers` config, such as a classifier's labels.
    fn transformers_config(source: &Source) -> Result<Self::Config>;

    /// Registers the head's layers over `encoder` in the encoder's schema, laid out as `layouts` says.
    fn layers(
        schema: &mut Schema<'_>,
        encoder: EncoderBody,
        config: &Self::Config,
        layouts: &HeadLayouts,
    ) -> SingeResult<Self::Layers>;

    fn new(model: Model<Self::Layers>, config: Self::Config) -> Self;
}

/// What an encoder model gives the shared loader.
pub trait EncoderModel {
    type Config: Serialize + DeserializeOwned;

    const LAYOUTS: HeadLayouts;

    /// Reads the encoder's config from a `transformers` snapshot.
    fn config(source: &Source) -> Result<Self::Config>;

    /// Creates the encoder heads sit on.
    fn create(schema: &mut Schema<'_>, config: &Self::Config) -> SingeResult<EncoderBody>;

    /// Returns the rules of the encoder's tensors, which skip every head's.
    fn encoder_rules() -> Rules;

    /// Returns the rules of `head`'s tensors, which go before the encoder's.
    fn head_rules(head: Head) -> Rules;
}

/// An encoder's config with its head's, as a native config stores them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct EncoderWithHead<E, C> {
    encoder: E,
    head: C,
}

/// Loads model `M`'s encoder and head `H` over it.
pub fn load_encoder_head<M: EncoderModel, H: EncoderHead>(loader: &mut Loader<'_>) -> Result<H> {
    let config: EncoderWithHead<M::Config, H::Config> = loader.config(|source| {
        Ok(EncoderWithHead {
            encoder: M::config(source)?,
            head: H::transformers_config(source)?,
        })
    })?;
    let model = Schema::model(loader.options().data_type, |schema| {
        let encoder = M::create(schema, &config.encoder)?;
        H::layers(schema, encoder, &config.head, &M::LAYOUTS)
    })?;
    let rules = M::head_rules(H::HEAD)
        .then(shared_head_rules(H::HEAD))
        .then(M::encoder_rules());
    let model = loader.load_whole(&model, &rules)?;
    Ok(H::new(model, config.head))
}

fn shared_head_rules(head: Head) -> Rules {
    match head {
        Head::TokenClassification => Rules::new().rename("classifier", "classifier.output"),
        Head::QuestionAnswering => Rules::new().rename("qa_outputs", "span.output"),
        _ => Rules::new(),
    }
}

/// Returns a loader for every head over a text encoder, for a model that serves them all.
/// The bare encoder comes first, so a snapshot that names no head loads as a [`TextEncoder`].
#[must_use]
pub const fn encoder_heads<M: EncoderModel>() -> [HeadLoader; 5] {
    [
        HeadLoader::TextEncoder(load_encoder_head::<M, TextEncoder>),
        HeadLoader::SequenceClassification(load_encoder_head::<M, SequenceClassifier>),
        HeadLoader::TokenClassification(load_encoder_head::<M, TokenClassifier>),
        HeadLoader::QuestionAnswering(load_encoder_head::<M, QuestionAnswerer>),
        HeadLoader::MaskedLm(load_encoder_head::<M, MaskedLm>),
    ]
}
