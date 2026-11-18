// Token classification, as `transformers`' `*ForTokenClassification` heads: every token's hidden state to one score per label, `[B, T, labels]`.
// It is written once for every model; its state is `classifier.output`.

use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    nn::{Layer, Linear, Model, ModelState, TokenBatch},
};

use super::{ClassifierConfig, EncoderHead, HeadLayouts};
use crate::{
    assembly::PostNormEncoder,
    error::{Error, Result},
    format::transformers::classifier_config,
    schema::Schema,
    source::Source,
};

#[derive(Debug)]
pub struct TokenClassifierLayers {
    encoder: Box<dyn Layer<TokenBatch>>,
    output: Linear,
}

impl Layer<TokenBatch> for TokenClassifierLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.output
            .forward(&self.encoder.forward(input, state)?, state)
    }
}

// A loaded token classifier: its layers and state, and its labels.
#[derive(Debug)]
pub struct TokenClassifier {
    model: Model<TokenClassifierLayers>,
    config: ClassifierConfig,
}

impl TokenClassifier {
    #[must_use]
    pub const fn model(&self) -> &Model<TokenClassifierLayers> {
        &self.model
    }

    #[must_use]
    pub const fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    // Returns each row's tokens' scores, one per label, before any score function, padded tokens included.
    pub fn scores(&self, batch: &TokenBatch) -> Result<Vec<Vec<Vec<f32>>>> {
        if self.config.labels.is_empty() {
            return Err(Error::unsupported("a classifier of no labels"));
        }

        let length = usize::try_from(batch.tokens().shape().dimensions()[1])
            .map_err(|_| SingeError::out_of_range("batch length"))?;
        let labels = self.config.labels.len();
        let scores = self
            .model
            .forward_group(batch)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?;
        Ok(scores
            .chunks(length * labels)
            .map(|row| row.chunks(labels).map(<[f32]>::to_vec).collect())
            .collect())
    }
}

impl EncoderHead for TokenClassifier {
    type Config = ClassifierConfig;

    type Layers = TokenClassifierLayers;

    fn transformers_config(source: &Source) -> Result<ClassifierConfig> {
        classifier_config(source)
    }

    fn layers(
        schema: &mut Schema<'_>,
        encoder: PostNormEncoder,
        config: &ClassifierConfig,
        _layouts: &HeadLayouts,
    ) -> SingeResult<TokenClassifierLayers> {
        let labels = i64::try_from(config.labels.len())
            .map_err(|_| SingeError::out_of_range("classifier labels"))?;
        let output =
            schema
                .scope("classifier")
                .linear("output", encoder.config().width, labels, true)?;
        Ok(TokenClassifierLayers {
            encoder: Box::new(encoder),
            output,
        })
    }

    fn new(model: Model<TokenClassifierLayers>, config: ClassifierConfig) -> Self {
        Self { model, config }
    }
}
