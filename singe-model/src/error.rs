use std::{
    io::Error as IoError,
    path::{Path, PathBuf},
    result::Result as StdResult,
};

use image::ImageError;
use serde_json::Error as JsonError;
use singe::Error as SingeError;
use thiserror::Error;
use tokenizers::Error as TokenizerError;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Singe(#[from] SingeError),
    #[error("failed to access {}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: IoError,
    },
    #[error("failed to parse {}", path.display())]
    Json {
        path: PathBuf,
        #[source]
        source: JsonError,
    },
    #[error("failed to {what}")]
    JsonValue {
        what: &'static str,
        #[source]
        source: JsonError,
    },
    #[error("unsupported model config: {0}")]
    UnsupportedConfig(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("checkpoint tensors no rule maps: {0:?}")]
    UnmappedTensors(Vec<String>),
    #[error("failed to render chat template")]
    Template(#[source] minijinja::Error),
    #[error("failed to access the image {}", path.display())]
    Image {
        path: PathBuf,
        #[source]
        source: ImageError,
    },
    #[error("failed to encode an image as PNG")]
    ImageEncoding(#[source] ImageError),
    #[error("unreadable media: {0}")]
    Media(String),
    // A load of a head the snapshot's classes do not name, which fails where its weights are missing.
    #[error("loading head {head} from a snapshot of {classes:?}")]
    HeadWeights {
        head: &'static str,
        classes: Vec<String>,
        #[source]
        source: Box<Self>,
    },
    #[error("failed to {what}")]
    Tokenizer {
        what: &'static str,
        #[source]
        source: TokenizerError,
    },
}

pub type Result<T> = StdResult<T, Error>;

impl Error {
    #[must_use]
    pub fn unsupported(what: impl Into<String>) -> Self {
        Self::UnsupportedConfig(what.into())
    }

    #[must_use]
    pub fn invalid(what: impl Into<String>) -> Self {
        Self::InvalidInput(what.into())
    }

    // Wraps a failed access to `path`, as `map_err` takes it.
    pub(crate) fn io(path: &Path) -> impl FnOnce(IoError) -> Self + use<> {
        let path = path.to_owned();
        move |source| Self::Io { path, source }
    }

    pub(crate) fn json(path: &Path) -> impl FnOnce(JsonError) -> Self + use<> {
        let path = path.to_owned();
        move |source| Self::Json { path, source }
    }

    pub(crate) fn json_value(what: &'static str) -> impl FnOnce(JsonError) -> Self + use<> {
        move |source| Self::JsonValue { what, source }
    }

    pub(crate) fn image(path: &Path) -> impl FnOnce(ImageError) -> Self + use<> {
        let path = path.to_owned();
        move |source| Self::Image { path, source }
    }
}
