//! Token classification, as `transformers`' `*ForTokenClassification` heads: every token's hidden state to one score per label, `[B, T, labels]`.
//! It is written once for every model; its state is `classifier.output`.

use std::sync::Arc;

use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    nn::{Layer, Linear, Model, ModelState, TokenBatch},
};

use crate::{
    error::Result,
    format::transformers::classifier_config,
    function::{
        TensorFunction,
        forward::{External, ExternalHead, Forward},
    },
    head::{ClassifierConfig, EncoderBody, EncoderHead, EncoderHeadLayout},
    host,
    schema::Schema,
    source::Source,
};

/// A text encoder and a linear layer that scores each token's last hidden state, as token classifiers and question answerers have.
#[derive(Debug)]
pub struct TokenScoreLayers {
    encoder: Box<dyn Layer<TokenBatch>>,
    output: Linear,
}

impl TokenScoreLayers {
    /// Registers `{name}.output` in `schema`, from `encoder`'s width to `scores` scores per token.
    pub fn create(
        schema: &mut Schema<'_>,
        name: &str,
        encoder: EncoderBody,
        scores: i64,
    ) -> SingeResult<Self> {
        let output = schema
            .scope(name)
            .linear("output", encoder.sizes.width, scores, true)?;
        Ok(Self {
            encoder: encoder.layers,
            output,
        })
    }
}

impl Layer<TokenBatch> for TokenScoreLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.encoder
            .forward(input, state)?
            .apply(&self.output, state)
    }
}

/// A loaded token classifier: its layers and state, and its labels.
#[derive(Debug)]
pub struct TokenClassifier {
    model: Forward<TokenScoreLayers>,
    config: ClassifierConfig,
}

impl TokenClassifier {
    pub fn get_model(&self) -> Result<&Model<TokenScoreLayers>> {
        self.model.get_native()
    }

    #[must_use]
    pub const fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    /// Returns each row's tokens' scores, one per label, before any score function, padded tokens included.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch`'s tokens are not `[B, T]`, or if running the model fails.
    pub fn scores(&self, batch: &TokenBatch) -> Result<Vec<Vec<Vec<f32>>>> {
        let &[_, length] = batch.tokens().shape().dimensions() else {
            return Err(SingeError::invalid_argument("token batch, [B, T]").into());
        };
        let length =
            usize::try_from(length).map_err(|_| SingeError::out_of_range("batch length"))?;
        let labels = self.config.labels().len();
        let scores = self
            .model
            .forward_group(batch)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?;
        Ok(scores
            .chunks(length * labels)
            .map(|row| host::rows(row, labels))
            .collect())
    }
}

impl EncoderHead for TokenClassifier {
    type Config = ClassifierConfig;

    type Layers = TokenScoreLayers;

    fn transformers_config(source: &Source) -> Result<ClassifierConfig> {
        classifier_config(source)
    }

    fn layers(
        schema: &mut Schema<'_>,
        encoder: EncoderBody,
        config: &ClassifierConfig,
        _layouts: &EncoderHeadLayout,
    ) -> SingeResult<TokenScoreLayers> {
        TokenScoreLayers::create(schema, "classifier", encoder, config.label_count()?)
    }

    fn new(model: Model<TokenScoreLayers>, config: ClassifierConfig) -> Self {
        Self {
            model: Forward::Native(model),
            config,
        }
    }
}

impl ExternalHead for TokenClassifier {
    fn external(source: &Source, function: Arc<dyn TensorFunction>) -> Result<Self> {
        let (model, config) = External::classifier(source, function, 2)?;
        Ok(Self {
            model: Forward::External(model),
            config,
        })
    }
}
