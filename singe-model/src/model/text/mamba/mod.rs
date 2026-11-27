//! Mamba: blocks of a selective state space mixer, each after an RMS norm with a residual around it, with no attention and no positions.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult,
    nn::{
        self, CausalLm, CausalLmParts, Embedding, Linear, MambaMixer, MambaSpec, Model,
        PreNormResidual, RmsNorm,
    },
};

use crate::{
    error::Result, format::Loader, head::HeadLoader, model::ModelEntry, schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MambaConfig {
    pub vocab_size: i64,
    pub width: i64,
    pub layers: i64,
    pub mixer: MambaSpec,
    /// Whether the mixer's input and output projections have a bias.
    pub bias: bool,
    pub convolution_bias: bool,
    pub eps: f32,
    pub tied: bool,
}

pub type MambaBlock = PreNormResidual<MambaMixer>;

pub type MambaLayers = CausalLm<MambaBlock>;

pub type Mamba = Model<MambaLayers>;

/// Registers the mixer's `input`, `convolution`, `projection`, `step`, `decay_log`, `skip`, and `output`.
fn mixer(schema: &mut Schema<'_>, config: &MambaConfig) -> SingeResult<MambaMixer> {
    let (width, spec) = (config.width, config.mixer);
    let layers = nn::MambaLayers {
        input: schema.linear("input", width, 2 * spec.inner, config.bias)?,
        convolution: schema.parameter("convolution.weight", [spec.inner, 1, spec.kernel])?,
        convolution_bias: config
            .convolution_bias
            .then(|| schema.parameter("convolution.bias", [spec.inner]))
            .transpose()?,
        projection: schema.linear("projection", spec.inner, spec.rank + 2 * spec.state, false)?,
        step: schema.linear("step", spec.rank, spec.inner, true)?,
        decay_log: schema.parameter("decay_log", [spec.inner, spec.state])?,
        skip: schema.parameter("skip", [spec.inner])?,
        output: schema.linear("output", spec.inner, width, config.bias)?,
    };
    MambaMixer::create(layers, spec)
}

/// Creates Mamba's state: `embeddings.tokens`, `blocks.N` of a `norm` and a `mixer`, `norm`, and `head`, tied to the embeddings when the config says so.
pub fn layers(schema: &mut Schema<'_>, config: &MambaConfig) -> SingeResult<MambaLayers> {
    let table = [config.vocab_size, config.width];
    let embedding = schema.parameter("embeddings.tokens.weight", table)?;
    let blocks = (0..config.layers)
        .map(|layer| {
            let mut block = schema.scope(format!("blocks.{layer}"));
            let norm = block.rms_norm("norm", config.width, config.eps)?;
            Ok(PreNormResidual::new(
                norm,
                mixer(&mut block.scope("mixer"), config)?,
            ))
        })
        .collect::<SingeResult<Vec<_>>>()?;
    let norm: RmsNorm = schema.rms_norm("norm", config.width, config.eps)?;
    let head = if config.tied {
        schema.tie("head.weight", &embedding)?
    } else {
        schema.parameter("head.weight", table)?
    };
    let caches = vec![config.mixer.cache(); blocks.len()];
    Ok(CausalLm::new(CausalLmParts {
        embedding: Embedding::new(embedding),
        embedding_scale: None,
        logit_scale: None,
        logit_cap: None,
        rotary: None,
        blocks,
        norm,
        head: Linear::new(head, None),
        caches,
    }))
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "mamba",
    formats: &[(Format::Transformers, &["mamba"])],
    loaders: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Mamba> {
    let config: MambaConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| layers(schema, &config))?;
    loader.load_whole(&model, &transformers::rules())
}
