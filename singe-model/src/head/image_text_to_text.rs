use std::fmt::Debug;

use singe::generation::GenerationOptions;

use super::{LoopLimits, LoopSteps};
use crate::{
    BatchOptions,
    error::{Error, Result},
    processor::{image::Image, text::TextTokenizer, video::Video},
};

// A model that writes text about images and a prompt.
pub trait ImageTextToText: Debug {
    // Generates after `prompt`, whose image tokens stand for `images` in order, as `options` asks.
    fn generate(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        images: &[Image],
        options: &BatchOptions,
    ) -> Result<Vec<i64>>;

    // The steps an exported loop runs within `limits`: the encode step, which reads one image's patches, and the decoder's steps.
    fn loop_steps(&self, limits: &LoopLimits, generation: &GenerationOptions) -> Result<LoopSteps>;

    // The token a prompt holds where an image's rows go.
    fn image_token(&self) -> i64;

    // The template a conversation about a clip renders through, where the model's processor replaces the snapshot's.
    fn video_template(&self) -> Option<&str> {
        None
    }

    // Generates after `prompt`, whose clip placeholder stands for `video`, decoded at `fps` frames per second, as `options` asks.
    fn generate_video(
        &self,
        _tokenizer: &TextTokenizer,
        _prompt: &str,
        _video: &Video,
        _fps: f64,
        _options: &BatchOptions,
    ) -> Result<Vec<i64>> {
        Err(Error::unsupported("a model that reads no clips"))
    }
}
