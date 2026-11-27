//! Promptable segmentation, as SAM gives it: an image embedded once, then masks decoded for points against the embedding.
//! A tracker carries one object, prompted by points on one frame of a clip, through the frames after it.
//! A concept segmenter finds every instance of what a text names, each a query with a score and a mask.

use std::fmt::Debug;

use singe::{
    DataType, Result as SingeResult, Tensor,
    nn::{Model, ModelState, TokenBatch},
};

use crate::{error::Result, head::Boxed};

/// A promptable segmenter's layers, which embed an image once and decode masks for points against the embedding.
pub trait PromptableLayers: Debug + Send + Sync {
    /// Returns an image's embedding, its maps from the finest the decoder reads to the coarsest, from its pixels at the model's input size.
    fn embed(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<Vec<Tensor>>;

    /// Returns, for `points` on the embedded image, `[x, y]` in pixels, each point's candidate masks' scores at a quarter of the input size, `[P, M, S / 4, S / 4]`, and their predicted quality, `[P, M]`.
    fn decode(
        &self,
        embedding: &[Tensor],
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

    pub fn embed(&self, pixels: &Tensor) -> Result<Vec<Tensor>> {
        Ok(self.model.layer().embed(pixels, self.model.state())?)
    }

    pub fn decode(&self, embedding: &[Tensor], points: &[[f32; 2]]) -> Result<(Tensor, Tensor)> {
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

/// Points on one frame of a clip that pick out an object to track.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackingPrompt {
    pub frame: usize,
    /// Each point, `[x, y]` in the model's input pixels.
    pub points: Vec<[f32; 2]>,
    /// Each point's label: 1 inside the object, 0 outside it, and 2 and 3 a box's top-left and bottom-right corners.
    pub labels: Vec<i64>,
    /// Whether tracking runs from the prompted frame back to the first, rather than on to the last.
    pub reverse: bool,
}

/// A tracked frame's mask scores, `[1, 1, S / 4, S / 4]` for an input of side `S`, and the score of the object being in the frame.
#[derive(Debug, Clone)]
pub struct TrackedFrame {
    pub scores: Tensor,
    pub object_score: f32,
}

/// A video segmenter's layers, which track the object a prompt picks out from the prompted frame through the last, or back through the first.
pub trait TrackingLayers: Debug + Send + Sync {
    /// Returns, for the prompted frame and each frame after it of the `frames` that `pixels` gives at the model's input size, or before it in reverse, nearest first, the object's mask.
    fn track(
        &self,
        frames: usize,
        pixels: &dyn Fn(usize) -> Result<Tensor>,
        prompt: &TrackingPrompt,
        state: &ModelState,
    ) -> Result<Vec<TrackedFrame>>;

    /// Returns the side of the square frames the model reads.
    fn image_size(&self) -> i64;
}

#[derive(Debug)]
pub struct MaskTracker {
    model: Model<Boxed<dyn TrackingLayers>>,
}

impl MaskTracker {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn TrackingLayers>>) -> Self {
        Self { model }
    }

    /// Tracks the object `prompt` picks out through the clip of `frames` frames whose pixels `pixels` gives.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompted frame is past the clip, if `pixels` fails, or if the model does.
    pub fn track(
        &self,
        frames: usize,
        pixels: &dyn Fn(usize) -> Result<Tensor>,
        prompt: &TrackingPrompt,
    ) -> Result<Vec<TrackedFrame>> {
        self.model
            .layer()
            .track(frames, pixels, prompt, self.model.state())
    }

    #[must_use]
    pub fn image_size(&self) -> i64 {
        self.model.layer().image_size()
    }
}

/// A concept segmenter's layers, which find what a text names in an image.
pub trait ConceptLayers: Debug + Send + Sync {
    /// Returns, for pixels at the model's input size and a tokenized text, each query's logit, `[Q]`, the logit that the concept is present at all, `[1]`, and each query's mask logits, `[Q, h, w]`.
    fn segment(
        &self,
        pixels: &Tensor,
        tokens: &TokenBatch,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor, Tensor)>;
}

/// Each query's score, the probability that it is an instance of the concept, and its mask logits, `[Q, h, w]`.
#[derive(Debug, Clone)]
pub struct ConceptQueries {
    pub scores: Vec<f32>,
    pub masks: Tensor,
}

#[derive(Debug)]
pub struct ConceptSegmenter {
    model: Model<Boxed<dyn ConceptLayers>>,
}

impl ConceptSegmenter {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn ConceptLayers>>) -> Self {
        Self { model }
    }

    /// Finds the concept `tokens` names in `pixels`: each query's score is the product of its probability and the concept's, as `post_process_instance_segmentation` scores it.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails.
    pub fn segment(&self, pixels: &Tensor, tokens: &TokenBatch) -> Result<ConceptQueries> {
        let (logits, presence, masks) =
            self.model
                .layer()
                .segment(pixels, tokens, self.model.state())?;
        let probability = |logit: f32| 1.0 / (1.0 + (-logit).exp());
        let presence = presence.cast(DataType::F32)?.to_vec::<f32>()?;
        let presence = presence.first().map_or(1.0, |&logit| probability(logit));
        Ok(ConceptQueries {
            scores: logits
                .cast(DataType::F32)?
                .contiguous()?
                .to_vec::<f32>()?
                .into_iter()
                .map(|logit| probability(logit) * presence)
                .collect(),
            masks,
        })
    }
}
