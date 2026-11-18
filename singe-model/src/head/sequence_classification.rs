// Sequence classification, as `transformers`' `*ForSequenceClassification` heads: a text encoder's hidden states pooled to one vector per sequence, through a projection and its activation where the model has one, to one score per label.
// It is written once for every model: a model names its layout, and the import rules of its classifier's tensors.
// Its state is `classifier.projection`, when the layout has one, and `classifier.output`.

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    nn::{
        Activation, Layer, Linear, Model, ModelState, Pooling, PoolingInput, PoolingMode,
        TokenBatch,
    },
};

use super::{EncoderHead, HeadLayouts};
use crate::{
    assembly::PostNormEncoder,
    error::{Error, Result},
    format::transformers::classifier_config,
    schema::Schema,
    source::Source,
};

// How scores become what a classification reports, as `transformers`' `function_to_apply` names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreFunction {
    Sigmoid,
    Softmax,
    None,
}

impl ScoreFunction {
    // Applies the function to one row's scores, as `transformers`' pipelines do with NumPy.
    #[must_use]
    pub fn apply(self, scores: &[f32]) -> Vec<f32> {
        match self {
            Self::None => scores.to_vec(),
            Self::Sigmoid => scores
                .iter()
                .map(|score| 1.0 / (1.0 + (-score).exp()))
                .collect(),
            Self::Softmax => {
                let largest = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exponentials = scores
                    .iter()
                    .map(|score| (score - largest).exp())
                    .collect::<Vec<_>>();
                let total = exponentials.iter().sum::<f32>();
                exponentials.iter().map(|value| value / total).collect()
            }
        }
    }
}

// Returns the index and value of the first largest score, as NumPy's `argmax` picks it, or `None` for no scores.
#[must_use]
pub fn argmax(scores: &[f32]) -> Option<(usize, f32)> {
    scores
        .iter()
        .copied()
        .enumerate()
        .fold(None, |best, (index, score)| match best {
            Some((_, kept)) if kept >= score => best,
            _ => Some((index, score)),
        })
}

// A classifier's labels, in the order of its scores, and the function its scores go through unless a caller names another.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassifierConfig {
    pub labels: Vec<String>,
    pub function: ScoreFunction,
    // The function a `sentence-transformers` cross-encoder's config names for its ranking scores, when it names one.
    pub ranking_function: Option<ScoreFunction>,
}

// What a model's classifier reads, and what it does before its output projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassifierLayout {
    // The state that stands for a sequence: the first token's for BERT, the last real token's for decoders.
    pub pooling: PoolingMode,
    // A projection that keeps the width, and its activation, before the output, as BERT's pooler (tanh) and DistilBERT's pre-classifier (ReLU) are.
    pub projection: Option<Activation>,
    pub output_bias: bool,
}

// A text encoder and its classifier, from a batch of token sequences to each sequence's scores, `[B, labels]`.
#[derive(Debug)]
pub struct SequenceClassifierLayers {
    encoder: Box<dyn Layer<TokenBatch>>,
    pooling: Pooling,
    projection: Option<(Linear, Activation)>,
    output: Linear,
}

impl SequenceClassifierLayers {
    // Registers the classifier in `schema` as `classifier.projection` and `classifier.output`, over `encoder`'s states of `width`.
    pub fn create(
        schema: &mut Schema<'_>,
        encoder: Box<dyn Layer<TokenBatch>>,
        width: i64,
        labels: usize,
        layout: ClassifierLayout,
    ) -> SingeResult<Self> {
        let mut schema = schema.scope("classifier");
        let projection = layout
            .projection
            .map(|activation| {
                Ok::<_, SingeError>((schema.linear("projection", width, width, true)?, activation))
            })
            .transpose()?;
        let labels =
            i64::try_from(labels).map_err(|_| SingeError::out_of_range("classifier labels"))?;
        let output = schema.linear("output", width, labels, layout.output_bias)?;
        Ok(Self {
            encoder,
            pooling: Pooling::new(layout.pooling, false),
            projection,
            output,
        })
    }
}

impl Layer<TokenBatch> for SequenceClassifierLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self.encoder.forward(input, state)?;
        let mut pooled = self.pooling.forward(
            &PoolingInput {
                hidden: &hidden,
                padding: input.padding(),
            },
            state,
        )?;
        if let Some((projection, activation)) = &self.projection {
            pooled = activation.forward(&projection.forward(&pooled, state)?, state)?;
        }
        self.output.forward(&pooled, state)
    }
}

// A loaded classifier: its layers and state, and its labels.
#[derive(Debug)]
pub struct SequenceClassifier {
    model: Model<SequenceClassifierLayers>,
    config: ClassifierConfig,
}

impl SequenceClassifier {
    #[must_use]
    pub const fn model(&self) -> &Model<SequenceClassifierLayers> {
        &self.model
    }

    #[must_use]
    pub const fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    // Returns each row's scores, one per label, before any score function.
    pub fn scores(&self, batch: &TokenBatch) -> Result<Vec<Vec<f32>>> {
        if self.config.labels.is_empty() {
            return Err(Error::unsupported("a classifier of no labels"));
        }

        let scores = self
            .model
            .forward_group(batch)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?;
        Ok(scores
            .chunks(self.config.labels.len())
            .map(<[f32]>::to_vec)
            .collect())
    }
}

impl EncoderHead for SequenceClassifier {
    type Config = ClassifierConfig;

    type Layers = SequenceClassifierLayers;

    fn transformers_config(source: &Source) -> Result<ClassifierConfig> {
        classifier_config(source)
    }

    fn layers(
        schema: &mut Schema<'_>,
        encoder: PostNormEncoder,
        config: &ClassifierConfig,
        layouts: &HeadLayouts,
    ) -> SingeResult<SequenceClassifierLayers> {
        let width = encoder.config().width;
        SequenceClassifierLayers::create(
            schema,
            Box::new(encoder),
            width,
            config.labels.len(),
            layouts.classifier,
        )
    }

    fn new(model: Model<SequenceClassifierLayers>, config: ClassifierConfig) -> Self {
        Self { model, config }
    }
}
