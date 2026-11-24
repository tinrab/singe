use std::fmt::Debug;

use singe::generation::GenerationOptions;

use crate::{
    BatchOptions,
    error::{Error, Result},
    head::{LoopLimits, LoopSteps},
    processor::{image::Image, text::TextTokenizer, video::Video},
};

/// A model that writes text about images and a prompt.
pub trait ImageTextToText: Debug {
    /// Generates tokens after `prompt`, whose image tokens stand for `images` in order, as `options` asks.
    fn generate(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        images: &[Image],
        options: &BatchOptions,
    ) -> Result<Vec<i64>>;

    /// Returns the steps an exported loop runs within `limits`: the decoder's steps, and an encode step that reads one image's patches when the model has one.
    fn loop_steps(&self, limits: &LoopLimits, generation: &GenerationOptions) -> Result<LoopSteps>;

    /// Returns the token a prompt holds where an image's rows go.
    fn image_token(&self) -> i64;

    /// Returns the template a conversation about a clip renders through, when the model's processor replaces the snapshot's template.
    /// The default is `None`, which keeps the snapshot's template.
    fn video_template(&self) -> Option<&str> {
        None
    }

    /// Generates tokens after `prompt`, whose clip placeholder stands for `video`, decoded at `fps` frames per second, as `options` asks.
    /// The default returns an error, for a model that reads no clips.
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
