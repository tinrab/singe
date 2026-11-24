//! Promptable segmentation, as SAM gives it: an image embedded once, then masks decoded for points against the embedding.

use std::fmt::Debug;

use singe::{
    Result as SingeResult, Tensor,
    nn::{Model, ModelState},
};

use crate::{error::Result, head::Boxed};

/// A promptable segmenter's layers, which embed an image once and decode masks for points against the embedding.
pub trait PromptableLayers: Debug + Send + Sync {
    /// Returns an image's embedding from its pixels at the model's input size.
    fn embed(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<Tensor>;

    /// Returns, for `points` on the embedded image, `[x, y]` in pixels, each point's candidate masks' scores at a quarter of the input size, `[P, M, S / 4, S / 4]`, and their predicted quality, `[P, M]`.
    fn decode(
        &self,
        embedding: &Tensor,
        points: &[[f32; 2]],
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)>;

    /// Returns the side of the square images the model reads.
    fn image_size(&self) -> i64;
}

#[derive(Debug)]
pub struct MaskGenerator {
    model: Model<Boxed<dyn PromptableLayers>>,
}

impl MaskGenerator {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn PromptableLayers>>) -> Self {
        Self { model }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<Boxed<dyn PromptableLayers>> {
        &self.model
    }

    pub fn embed(&self, pixels: &Tensor) -> Result<Tensor> {
        Ok(self.model.layer().embed(pixels, self.model.state())?)
    }

    pub fn decode(&self, embedding: &Tensor, points: &[[f32; 2]]) -> Result<(Tensor, Tensor)> {
        Ok(self
            .model
            .layer()
            .decode(embedding, points, self.model.state())?)
    }

    #[must_use]
    pub fn image_size(&self) -> i64 {
        self.model.layer().image_size()
    }
}
