//! Heads over a text encoder, and the one loader every model built on `assembly`'s post-norm encoder shares.
//! A model gives its `transformers` config and rules and its heads' layouts; each head registers its own layers and reads its own config.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use singe::{
    Result as SingeResult,
    nn::{Activation, Model},
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

/// A head that sits on a text encoder: what it reads from a `transformers` config, the layers it registers beside the encoder, and what it wraps the loaded model in.
pub trait EncoderHead: LoadHead {
    type Config: Serialize + DeserializeOwned;

    type Layers;

    /// Reads what the head adds to a `transformers` config, such as a classifier's labels.
    fn transformers_config(source: &Source) -> Result<Self::Config>;

    /// Registers the head's layers over `encoder` in the encoder's schema, laid out as `layouts` says.
    fn layers(
        schema: &mut Schema<'_>,
        encoder: PostNormEncoder,
        config: &Self::Config,
        layouts: &HeadLayouts,
    ) -> SingeResult<Self::Layers>;

    fn new(model: Model<Self::Layers>, config: Self::Config) -> Self;
}

/// What a model built on the post-norm encoder gives the shared loader.
pub trait EncoderModel {
    const LAYOUTS: HeadLayouts;

    /// Reads the encoder's config from a `transformers` snapshot.
    fn config(source: &Source) -> Result<PostNormEncoderConfig>;

    /// Returns the rules of the encoder's tensors, which skip every head's.
    fn encoder_rules() -> Rules;

    /// Returns the rules of `head`'s tensors, which go before the encoder's.
    fn head_rules(head: Head) -> Rules;
}

/// An encoder's config with its head's, as a native config stores them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct EncoderWithHead<C> {
    encoder: PostNormEncoderConfig,
    head: C,
}

/// Loads model `M`'s encoder and head `H` over it.
pub fn load_encoder_head<M: EncoderModel, H: EncoderHead>(loader: &mut Loader<'_>) -> Result<H> {
    let config: EncoderWithHead<H::Config> = loader.config(|source| {
        Ok(EncoderWithHead {
            encoder: M::config(source)?,
            head: H::transformers_config(source)?,
        })
    })?;
    let model = Schema::model(loader.options().data_type, |schema| {
        let encoder = PostNormEncoder::create(schema, &config.encoder)?;
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
