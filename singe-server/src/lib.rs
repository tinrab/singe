//! Serves Singe models over HTTP, with OpenAI-compatible endpoints and a task API for every `singe-model` task.
//!
//! This crate provides:
//!
//! - **Server:** [`server::serve`] loads the models a [`ServerConfig`](config::ServerConfig) names and serves them until SIGINT or SIGTERM, and [`server::app`] returns its router without binding a port.
//! - **Endpoints:** routers in [`handler`] for chat and text completions, embeddings, rerank, transcription, speech, image generation and edits, and `/v1/tasks/{task}` for any task, with request and response types in [`api`].
//! - **Models:** a [`Served`](registry::Served) set of named models, each answering the endpoints of its task, as [`registry::endpoints`] lists them.
//! - **Text generation:** [`TextEngine`](generation::engine::TextEngine) batches requests continuously, joining each into a free row between decoding steps.
//! - **Custom models:** [`Worker`](worker::Worker) runs a model on a thread of its own, and [`TaskRunner`](registry::TaskRunner) serves one through the task API.
//!
//! # Example
//!
//! Serve Qwen3-0.6B from the local Hugging Face cache on port 8000:
//!
//! ```rust,no_run
//! use singe_server::{
//!     config::{ModelConfig, ServerConfig},
//!     server,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = ServerConfig {
//!         models: vec![ModelConfig::new("Qwen/Qwen3-0.6B")],
//!         ..ServerConfig::default()
//!     };
//!     server::serve(config).await?;
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod auth;
pub mod config;
pub mod error;
pub mod generation;
pub mod handler;
pub mod media;
pub mod registry;
pub mod server;
pub mod text;
pub mod worker;
