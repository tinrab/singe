//! Models built on Singe, loaded from Singe's own format or from Hugging Face snapshots, with the processors that turn a task's inputs into a model's and back.

pub mod assembly;
mod batch;
mod decoder_cache;
mod error;
#[cfg(feature = "cuda_13_4")]
pub mod export;
pub mod format;
pub mod head;
mod host;
pub mod model;
pub mod processor;
pub mod schema;
pub mod source;
pub mod task;

pub use self::{
    batch::BatchOptions,
    error::{Error, Result},
    source::{Format, LoadOptions, Source},
};
