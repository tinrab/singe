use std::{
    io::Error as IoError,
    path::{Path, PathBuf},
    result::Result as StdResult,
};

use image::ImageError;
use serde_json::Error as JsonError;
use singe::Error as SingeError;
#[cfg(feature = "onnx")]
use singe::onnx::Error as OnnxError;
use thiserror::Error;
use tokenizers::Error as TokenizerError;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Singe(#[from] SingeError),
    // Boxed, since the ONNX error would double the size of every `Result` here.
    #[cfg(feature = "onnx")]
    #[error(transparent)]
    Onnx(Box<OnnxError>),
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
    #[error("failed to encode an image")]
    ImageEncoding(#[source] ImageError),
    #[error("unreadable media: {0}")]
    Media(String),
    /// A failed load of a head that none of the snapshot's classes names, so the snapshot may lack the head's weights.
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

#[cfg(feature = "onnx")]
impl From<OnnxError> for Error {
    fn from(error: OnnxError) -> Self {
        Self::Onnx(Box::new(error))
    }
}

impl Error {
    #[must_use]
    pub fn unsupported(what: impl Into<String>) -> Self {
        Self::UnsupportedConfig(what.into())
    }

    #[must_use]
    pub fn invalid(what: impl Into<String>) -> Self {
        Self::InvalidInput(what.into())
    }

    /// Returns a closure for `map_err` that wraps a failed access to `path` as [`Error::Io`].
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
