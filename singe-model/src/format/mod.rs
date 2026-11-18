// The model formats Singe reads: its own, and Hugging Face's `transformers` and `diffusers` snapshots, whose names and configs importers turn into Singe's.

mod native;
mod rule;
pub mod transformers;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use singe::nn::{Checkpoint, CheckpointOptions, LoadPlan, Model, ModelState};

pub use self::rule::{Matched, Rules};
use crate::{
    error::{Error, Result},
    head::Head,
    source::{Format, LoadOptions, Source},
};

// The safetensors files a snapshot stores its checkpoint in under `stem`: `{stem}.safetensors`, or the shards `{stem}.safetensors.index.json` names.
fn safetensors_files(directory: &Path, stem: &str) -> Result<Vec<PathBuf>> {
    #[derive(serde::Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }

    let index = directory.join(format!("{stem}.safetensors.index.json"));
    if !index.exists() {
        // A snapshot that ships only half-precision weights stores them as the `fp16` variant.
        let single = directory.join(format!("{stem}.safetensors"));
        let half = directory.join(format!("{stem}.fp16.safetensors"));
        return Ok(vec![if !single.exists() && half.exists() {
            half
        } else {
            single
        }]);
    }

    let text = fs::read_to_string(&index).map_err(Error::io(&index))?;
    let index: Index =
        serde_json::from_str(&text).map_err(Error::json("parse a safetensors index"))?;
    let files = index
        .weight_map
        .into_values()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|file| directory.join(file))
        .collect();
    Ok(files)
}

// Reads one model from a source: its Singe config, and the state of each of its parts.
// It records both, so what it loaded can be saved in Singe's format.
#[derive(Debug)]
pub struct Loader<'a> {
    source: &'a Source,
    options: LoadOptions,
    checkpoint: Option<Checkpoint<'static>>,
    // Stored names a part read or a rule skipped.
    consumed: BTreeSet<String>,
    config: Option<Value>,
    parts: Vec<(String, ModelState)>,
}

impl<'a> Loader<'a> {
    #[must_use]
    pub const fn new(source: &'a Source, options: LoadOptions) -> Self {
        Self {
            source,
            options,
            checkpoint: None,
            consumed: BTreeSet::new(),
            config: None,
            parts: Vec::new(),
        }
    }

    #[must_use]
    pub const fn source(&self) -> &'a Source {
        self.source
    }

    #[must_use]
    pub const fn options(&self) -> LoadOptions {
        self.options
    }

    // Reads the model's Singe config: a native directory's own, or what `import` makes of another format's files.
    pub fn config<C: Serialize + DeserializeOwned>(
        &mut self,
        import: impl FnOnce(&Source) -> Result<C>,
    ) -> Result<C> {
        let config = match self.source.format() {
            Format::Native => {
                let file = self.source.json::<Value>("config.json")?;
                serde_json::from_value(file["singe"]["config"].clone())
                    .map_err(Error::ConfigFormat)?
            }
            Format::Transformers | Format::Diffusers | Format::AutoGluon => import(self.source)?,
        };
        self.config =
            Some(serde_json::to_value(&config).map_err(Error::json("record the model config"))?);
        Ok(config)
    }

    // Loads part `part` of the model, whose layers and state `schema` builds on metadata.
    // A native directory stores each part under Singe's names in `{part}.safetensors`; another format's checkpoint goes through `rules`.
    pub fn load<L>(&mut self, part: &str, schema: &Model<L>, rules: &Rules) -> Result<Model<L>> {
        let options = CheckpointOptions {
            device: Some(self.options.device),
            convert: true,
        };
        let state = match self.source.format() {
            Format::Native => {
                let mut checkpoint = Checkpoint::open([self
                    .source
                    .directory()
                    .join(format!("{part}.safetensors"))])?;
                let plan = LoadPlan::identity(&checkpoint);
                schema
                    .state()
                    .load_checkpoint(&mut checkpoint, &plan, options)?
            }
            Format::Transformers | Format::Diffusers | Format::AutoGluon => {
                let checkpoint = self.checkpoint()?;
                let (plan, matched) = rules.plan(checkpoint.names())?;
                let matched = matched.into_iter().map(str::to_owned).collect::<Vec<_>>();
                let state = schema.state().load_checkpoint(checkpoint, &plan, options)?;
                self.consumed.extend(matched);
                state
            }
        };
        self.parts.push((part.to_owned(), state.clone()));
        Ok(schema.with_state(state)?)
    }

    // Opens the source's checkpoint on first use, and keeps it for the parts after.
    fn checkpoint(&mut self) -> Result<&mut Checkpoint<'static>> {
        match &mut self.checkpoint {
            Some(checkpoint) => Ok(checkpoint),
            empty @ None => {
                let stem = match self.source.format() {
                    Format::Diffusers => "diffusion_pytorch_model",
                    Format::Transformers | Format::Native | Format::AutoGluon => "model",
                };
                let files = safetensors_files(self.source.directory(), stem)?;
                Ok(empty.insert(Checkpoint::open(files)?))
            }
        }
    }

    // Refuses a checkpoint that stores a tensor no part read and no rule skipped.
    pub fn finish(&mut self) -> Result<()> {
        let Some(checkpoint) = &self.checkpoint else {
            return Ok(());
        };
        let unread = checkpoint
            .names()
            .into_iter()
            .filter(|name| !self.consumed.contains(*name))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if unread.is_empty() {
            Ok(())
        } else {
            Err(Error::UnmappedTensors(unread))
        }
    }

    // Writes what this loader read to `directory` in Singe's format, as model `model` serving `head`.
    pub fn save(&self, directory: impl AsRef<Path>, model: &str, head: Head) -> Result<()> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| Error::unsupported("saving a model before its config"))?;
        native::save(
            self.source,
            directory.as_ref(),
            (model, head),
            config,
            &self.parts,
        )
    }
}
