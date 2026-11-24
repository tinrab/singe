// What turns a task's inputs into a model's and back: tokenizers, chat templates, images, video clips, audio clips, and diffusion's noise and schedulers.
// Processing a model alone needs, such as Whisper's log-mel front end or Qwen3.5's image patches, lives with the model, as `transformers` keeps it.

pub mod audio;
pub mod base64;
pub mod chat;
mod conversion;
pub mod fbank;
pub mod image;
pub mod random;
mod resize;
pub mod scheduler;
mod sentencepiece;
pub mod text;
pub mod video;
