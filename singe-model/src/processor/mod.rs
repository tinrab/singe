// What turns a task's inputs into a model's and back: tokenizers, chat templates, images, video clips, audio clips, and diffusion's noise and schedulers.
// Processing a model alone needs, such as Whisper's log-mel front end or Qwen3.5's image patches, lives with the model, as `transformers` keeps it.

pub mod audio;
pub mod chat;
pub mod fbank;
pub mod image;
pub mod random;
mod resize;
pub mod scheduler;
mod sentencepiece;
pub mod text;
pub mod video;

use singe::{DataType, Result as SingeResult, Tensor};

// A tensor's values on the host, as `F32`, for the work a pipeline or head does there.
pub(crate) fn host_values(tensor: &Tensor) -> SingeResult<Vec<f32>> {
    tensor.cast(DataType::F32)?.contiguous()?.to_vec::<f32>()
}
