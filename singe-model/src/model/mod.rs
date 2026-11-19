// Every model, one module each, under the category Hugging Face documents it in: `transformers`' docs sections, and `diffusion` for `diffusers`' models.
// A model declares itself in its module's `MODEL`, and its category's `MODELS` lists it; a model left out of the list leaves `MODEL` unused, which the lints refuse.
// A component only pipelines load, such as `diffusion::shap_e`'s, which no snapshot names as a model, has no entry.

pub mod audio;
pub mod diffusion;
pub mod multimodal;
pub mod reinforcement_learning;
pub mod tabular;
pub mod text;
pub mod time_series;
pub mod video;
pub mod vision;

use std::path::Path;

use serde_json::Value;

use crate::{
    error::{Error, Result},
    format::{Loader, NativeConfig},
    head::{Head, HeadLoader, LoadHead},
    source::{Format, LoadOptions, Source},
};

// What the registry knows of a model: Singe's name for it, which a native config records, each other format's names for it, and a loader for each head it serves.
#[derive(Debug, Clone, Copy)]
pub struct ModelEntry {
    pub name: &'static str,
    pub formats: &'static [(Format, &'static [&'static str])],
    // The heads in the order a snapshot that names none chooses them.
    pub loaders: &'static [HeadLoader],
}

impl ModelEntry {
    pub fn heads(&self) -> impl Iterator<Item = Head> {
        self.loaders.iter().map(HeadLoader::head)
    }

    fn named(&self, format: Format, name: &str) -> bool {
        match format {
            Format::Native => self.name == name,
            _ => self
                .formats
                .iter()
                .any(|(stored, names)| *stored == format && names.contains(&name)),
        }
    }

    fn loader(&self, head: Head) -> Result<&HeadLoader> {
        self.loaders
            .iter()
            .find(|loader| loader.head() == head)
            .ok_or_else(|| self.unserved(head))
    }

    fn unserved(&self, head: Head) -> Error {
        Error::unsupported(format!("model {} as {}", self.name, head.id()))
    }
}

const CATEGORIES: &[&[&ModelEntry]] = &[
    audio::MODELS,
    diffusion::MODELS,
    multimodal::MODELS,
    reinforcement_learning::MODELS,
    tabular::MODELS,
    text::MODELS,
    time_series::MODELS,
    video::MODELS,
    vision::MODELS,
];

// Every model the registry knows.
pub fn all() -> impl Iterator<Item = &'static ModelEntry> {
    CATEGORIES.iter().flat_map(|models| models.iter().copied())
}

// Returns the model `source` holds, refusing a name two models claim.
pub fn find(source: &Source) -> Result<&'static ModelEntry> {
    let mut found = all().filter(|model| model.named(source.format(), source.kind()));
    let model = found
        .next()
        .ok_or_else(|| Error::unsupported(format!("model kind {}", source.kind())))?;
    if let Some(other) = found.next() {
        return Err(Error::unsupported(format!(
            "model kind {}, which both {} and {} claim",
            source.kind(),
            model.name,
            other.name
        )));
    }
    Ok(model)
}

// Loads `source`'s model as head `H`.
pub fn load<H: LoadHead>(source: &Source, options: LoadOptions) -> Result<H> {
    let model = find(source)?;
    let load = H::loader(model.loader(H::HEAD)?).ok_or_else(|| model.unserved(H::HEAD))?;
    let mut loader = Loader::new(source, options);
    load(&mut loader).map_err(|error| with_classes(source, H::HEAD, error))
}

// Adds the snapshot's classes to the failed load of a head none of them names, whose weights the snapshot may lack.
fn with_classes(source: &Source, head: Head, error: Error) -> Error {
    match source.classes() {
        Ok(classes)
            if !classes.is_empty() && !classes.iter().any(|class| head.holds_class(class)) =>
        {
            Error::HeadWeights {
                head: head.id(),
                classes,
                source: Box::new(error),
            }
        }
        _ => error,
    }
}

// The head `source`'s weights hold: the one a native config records, or the first of the model's heads a `transformers` class in `architectures` names, or the model's first.
pub fn default_head(source: &Source) -> Result<Head> {
    let model = find(source)?;
    let first = model
        .heads()
        .next()
        .ok_or_else(|| Error::unsupported(format!("model {}, which serves no head", model.name)))?;
    match source.format() {
        Format::Native => Head::from_id(&NativeConfig::read(source.directory())?.head),
        Format::Transformers => {
            let classes = source.classes()?;
            Ok(model
                .heads()
                .find(|head| classes.iter().any(|class| head.holds_class(class)))
                .unwrap_or(first))
        }
        Format::Diffusers => Ok(first),
        // A Tab2D snapshot holds the weights of the one task its config names.
        Format::AutoGluon => {
            let config = source.json::<Value>("config.json")?;
            let head = match config["task"].as_str() {
                Some("REGRESSION") => Head::TabularRegression,
                _ => Head::TabularClassification,
            };
            Ok(model.heads().find(|&found| found == head).unwrap_or(first))
        }
    }
}

// Loads `source`'s model as `head`, and writes what it read to `directory` in Singe's format.
pub fn convert(source: &Source, head: Head, directory: &Path, options: LoadOptions) -> Result<()> {
    let model = find(source)?;
    let mut loader = Loader::new(source, options);
    model
        .loader(head)?
        .run(&mut loader)
        .map_err(|error| with_classes(source, head, error))?;
    loader.save(directory, model.name, head)
}
