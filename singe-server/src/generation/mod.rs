//! Text generation with continuous batching: [`engine::TextEngine`] runs a model's decoder on its own thread, which joins requests into free rows between steps, and [`output::TextGeneration`] turns a request's tokens into text on the HTTP side.

pub mod engine;
pub mod format;
pub mod output;
