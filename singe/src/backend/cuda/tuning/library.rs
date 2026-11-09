use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use singe_cudnn::backend::execution::EngineChoice;

use crate::{
    backend::{
        cuda::{
            runtime::CublasLtMatmulAlgorithmFingerprint,
            tuning::records::{TunedChoice, TuningRecords},
        },
        registry::CudaGraphProvider,
    },
    error::{Error, Result},
    utility::Locked,
};

// What a node's key and provider listed.
type Listed = BTreeMap<(String, String), Vec<TunedChoice>>;

// What libraries could run each node with, which compiling a graph lists when asked, since only a library knows.
#[derive(Debug, Clone, Default)]
pub struct LibraryRecorder {
    listed: Arc<Mutex<Listed>>,
}

impl LibraryRecorder {
    pub(crate) fn note(
        &self,
        key: &str,
        provider: CudaGraphProvider,
        choices: Vec<TunedChoice>,
    ) -> Result<()> {
        self.listed
            .locked("library recorder")?
            .insert((key.to_owned(), format!("{provider:?}")), choices);
        Ok(())
    }

    // What `provider` listed for the node under `key`.
    pub(crate) fn take(&self, key: &str, provider: CudaGraphProvider) -> Result<Vec<TunedChoice>> {
        Ok(self
            .listed
            .locked("library recorder")?
            .remove(&(key.to_owned(), format!("{provider:?}")))
            .unwrap_or_default())
    }
}

// What records say a library should run a node with, and what to tell them about it.
#[derive(Debug, Clone)]
pub struct ForcedLibrary {
    pub(crate) records: Arc<TuningRecords>,
    pub(crate) key: String,
    pub(crate) choice: TunedChoice,
}

impl ForcedLibrary {
    #[must_use]
    pub(crate) const fn cublaslt(&self) -> Option<&CublasLtMatmulAlgorithmFingerprint> {
        match &self.choice {
            TunedChoice::CublasLt(config) => Some(config),
            _ => None,
        }
    }

    // The library the record's choice is for.
    #[must_use]
    pub(crate) const fn library(&self) -> &'static str {
        match &self.choice {
            TunedChoice::CublasLt(_) => "cuBLASLt",
            TunedChoice::Cudnn(_) => "cuDNN",
            _ => "a library",
        }
    }

    #[must_use]
    pub(crate) const fn cudnn(&self) -> Option<&EngineChoice> {
        match &self.choice {
            TunedChoice::Cudnn(choice) => Some(choice),
            _ => None,
        }
    }
}

// Builds with what records force, when they force something this library reads, and with the library's own choice when they force nothing.
//
// # Errors
//
// Returns `Error::ConfigurationRefused` if the library refuses what the records force, and otherwise what `build` returns.
pub fn build_forced<'a, C: ?Sized + 'a, T>(
    forced: Option<&'a ForcedLibrary>,
    read: impl Fn(&'a ForcedLibrary) -> Option<&'a C>,
    build: impl Fn(Option<&'a C>) -> Result<T>,
) -> Result<T> {
    if let Some(forced) = forced
        && let Some(choice) = read(forced)
    {
        let value = build(Some(choice)).map_err(|error| Error::ConfigurationRefused {
            library: forced.library().to_owned(),
            key: forced.key.clone(),
            source: Box::new(error),
        })?;
        forced.records.note_used(&forced.key);
        return Ok(value);
    }
    build(None)
}
