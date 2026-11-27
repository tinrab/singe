//! Text to an image, generated as a grid of image tokens one after another and decoded, as Janus generates one.

use std::fmt::Debug;

use crate::{error::Result, head::SamplingRequest, processor::image::Image};

/// A model that generates an image's tokens from a prompt and decodes them.
pub trait ImageGenerator: Debug {
    /// Generates an image for `prompt` under classifier-free guidance at `guidance_scale`, or the model's own scale when `None`, choosing codes as `sampling` asks over the generation config's sampling.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt does not tokenize, or if a decoder step or the image decoder fails.
    fn generate_image(
        &self,
        prompt: &str,
        guidance_scale: Option<f32>,
        sampling: &SamplingRequest,
    ) -> Result<Image>;
}
