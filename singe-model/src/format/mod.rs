// The model formats Singe reads: its own, and Hugging Face's `transformers` and `diffusers` snapshots, whose names and configs importers turn into Singe's.

pub mod diffusers;
mod native;
mod rule;
pub mod transformers;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use singe::nn::{Checkpoint, CheckpointOptions, LoadPlan, Model, ModelState};

pub(crate) use self::native::NativeConfig;
pub use self::rule::{Matched, Rules, join};
use crate::{
    error::{Error, Result},
    head::Head,
    source::{Format, LoadOptions, Source},
};

// The safetensors files a snapshot stores its checkpoint in under `stem`: `{stem}.safetensors`, or the shards `{stem}.safetensors.index.json` names.
fn safetensors_files(directory: &Path, stem: &str) -> Result<Vec<PathBuf>> {
    #[derive(Deserialize)]
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
    let index: Index = serde_json::from_str(&text).map_err(Error::json(&index))?;
    let files = index
        .weight_map
        .into_values()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|file| directory.join(file))
        .collect();
    Ok(files)
}

// How a native directory stores each part's state: `{part}.safetensors`, or `{part}.flashpack`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Safetensors,
    Flashpack,
}

impl WeightFormat {
    // The file that holds `part`'s state.
    #[must_use]
    pub fn file(self, part: &str) -> String {
        match self {
            Self::Safetensors => format!("{part}.safetensors"),
            Self::Flashpack => format!("{part}.flashpack"),
        }
    }
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
                transformers::parse(NativeConfig::read(self.source.directory())?.config)?
            }
            _ => import(self.source)?,
        };
        self.config = Some(
            serde_json::to_value(&config).map_err(Error::json_value("record the model config"))?,
        );
        Ok(config)
    }

    // Loads part `part` of the model, whose layers and state `model` builds on metadata.
    // A native directory stores each part under Singe's names in `{part}.flashpack` or `{part}.safetensors`, read in that order; another format's checkpoint goes through `rules`.
    pub fn load<L>(&mut self, part: &str, model: &Model<L>, rules: &Rules) -> Result<Model<L>> {
        let options = CheckpointOptions {
            device: Some(self.options.device),
            convert: true,
        };
        let state = if self.source.format() == Format::Native {
            let directory = self.source.directory();
            let flashpack = directory.join(WeightFormat::Flashpack.file(part));
            let file = if flashpack.exists() {
                flashpack
            } else {
                directory.join(WeightFormat::Safetensors.file(part))
            };
            let checkpoint = Checkpoint::open([file])?;
            let plan = LoadPlan::identity(&checkpoint);
            model.state().load_checkpoint(&checkpoint, &plan, options)?
        } else {
            let checkpoint = self.checkpoint()?;
            let (plan, matched) = rules.plan(checkpoint.names())?;
            let matched = matched.into_iter().map(str::to_owned).collect::<Vec<_>>();
            let state = model.state().load_checkpoint(checkpoint, &plan, options)?;
            self.consumed.extend(matched);
            state
        };
        // Each sibling group of projections is stored once, joined, so eager layers multiply by it in one call.
        let state = if self.options.join_projections {
            state.join_projection_rows()?
        } else {
            state
        };
        self.parts.push((part.to_owned(), state.clone()));
        Ok(model.with_state(state)?)
    }

    // Opens the source's checkpoint on first use, and keeps it for the parts after.
    fn checkpoint(&mut self) -> Result<&Checkpoint<'static>> {
        match &mut self.checkpoint {
            Some(checkpoint) => Ok(checkpoint),
            empty @ None => {
                let stem = match self.source.format() {
                    Format::Diffusers => "diffusion_pytorch_model",
                    _ => "model",
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

    pub fn load_whole<L>(&mut self, model: &Model<L>, rules: &Rules) -> Result<Model<L>> {
        let model = self.load("model", model, rules)?;
        self.finish()?;
        Ok(model)
    }

    // Writes what this loader read to `directory` in Singe's format, as model `model` serving `head`, each part's state in `weights`, and returns each part with the file it wrote.
    pub fn save(
        &self,
        directory: &Path,
        model: &str,
        head: Head,
        weights: WeightFormat,
    ) -> Result<Vec<(String, PathBuf)>> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| Error::invalid("saving a model before reading its config"))?;
        native::save(
            self.source,
            directory,
            (model, head),
            (config, self.options.data_type),
            (&self.parts, weights),
        )
    }
}
