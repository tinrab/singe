use std::{io::Error as IoError, result::Result as StdResult};

use serde_json::Error as JsonError;
use singe::Error as SingeError;
use thiserror::Error;
use tokenizers::Error as TokenizerError;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Singe(#[from] SingeError),
    #[error("failed to read model config")]
    ConfigIo(#[source] IoError),
    #[error("failed to parse model config")]
    ConfigFormat(#[source] JsonError),
    #[error("unsupported model config: {0}")]
    UnsupportedConfig(String),
    #[error("failed to {what}")]
    Tokenizer {
        what: &'static str,
        #[source]
        source: TokenizerError,
    },
}

pub type Result<T> = StdResult<T, Error>;
