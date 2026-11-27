//! `mask-tracking`: a clip and points or a box on one of its frames to the object's mask on that frame and each after it, or each before it in reverse, as `transformers`' `Sam2VideoModel` propagates it from a session `Sam2VideoProcessor` fills.
//! The points scale to the model's input, each frame's mask scores are resized to the frame and cut at the threshold, and each frame's score is the probability that the object is in it.

use serde::{Deserialize, Serialize};
use singe::{DataType, Shape, Tensor, shape::spatial::Spatial2d, tensor::InterpolateMode};

use crate::{
    error::{Error, Result},
    head::{MaskTracker, TrackingPrompt},
    model,
    processor::{
        image::{Image, ImageTransforms},
        video::Video,
    },
    source::{LoadOptions, Source},
    task::{Pipeline, vision::pixels::mask},
};

#[derive(Debug)]
pub struct MaskTrackingPipeline {
    model: MaskTracker,
    processor: ImageTransforms,
    options: LoadOptions,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MaskTrackingInput {
    pub video: Video,
    /// The frame the points are on, the first by default.
    #[serde(default)]
    pub frame: usize,
    /// Each point, `[x, y]` in the frame's pixels.
    #[serde(default)]
    pub points: Vec<[f32; 2]>,
    /// A box around the object, `[x0, y0, x1, y1]` in the frame's pixels, whose corners prompt before the points.
    #[serde(default, rename = "box")]
    pub bounding_box: Option<[f32; 4]>,
    /// Each point's label, 1 inside the object and 0 outside it; every point is inside when there are none.
    #[serde(default)]
    pub labels: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct MaskTrackingParameters {
    /// The cutoff a pixel's score must exceed for the pixel to be in a mask, 0 by default.
    pub mask_threshold: f32,
    /// Whether to track from the prompted frame back to the first, as `propagate_in_video_iterator(reverse=True)` does, rather than on to the last.
    pub reverse: bool,
}

/// The tracked frames, from the prompted one on, or back in reverse, with each one's mask and the probability that the object is in it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrackedMasks {
    pub frames: Vec<usize>,
    pub masks: Vec<Image>,
    pub scores: Vec<f32>,
}

impl Pipeline for MaskTrackingPipeline {
    type Input = MaskTrackingInput;

    type Parameters = MaskTrackingParameters;

    type Output = TrackedMasks;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let frames = &input.video.frames;
        let prompted = frames.get(input.frame).ok_or_else(|| {
            Error::invalid(format!(
                "points on frame {} of a clip of {} frames",
                input.frame,
                frames.len()
            ))
        })?;
        let point_labels = if input.labels.is_empty() {
            vec![1; input.points.len()]
        } else if input.labels.len() == input.points.len() {
            if let Some(other) = input.labels.iter().find(|&&label| label != 0 && label != 1) {
                return Err(Error::invalid(format!(
                    "a point labelled {other}, rather than 0 or 1"
                )));
            }
            input.labels.clone()
        } else {
            return Err(Error::invalid(format!(
                "{} labels for {} points",
                input.labels.len(),
                input.points.len()
            )));
        };
        // `_normalize_coordinates` scales each axis to the warped frame.
        let side = self.model.image_size() as f32;
        let (width, height) = (prompted.width as f32, prompted.height as f32);
        // A box prompts as its two corners, labelled 2 and 3, before the points, as `Sam2VideoProcessor` adds it.
        let corners = input
            .bounding_box
            .map_or_else(Vec::new, |[x0, y0, x1, y1]| vec![[x0, y0], [x1, y1]]);
        let mut labels = if corners.is_empty() {
            Vec::new()
        } else {
            vec![2, 3]
        };
        labels.extend(point_labels);
        let prompt = TrackingPrompt {
            frame: input.frame,
            points: corners
                .iter()
                .chain(&input.points)
                .map(|&[x, y]| [x * side / width, y * side / height])
                .collect(),
            labels,
            reverse: parameters.reverse,
        };
        let pixels = |index: usize| {
            self.processor
                .preprocess(&frames[index], self.options.device, self.options.data_type)
        };
        let tracked = self.model.track(frames.len(), &pixels, &prompt)?;
        let threshold = Tensor::from_vec(
            self.options.device,
            Shape::create([1, 1, 1, 1])?,
            vec![parameters.mask_threshold],
        )?;
        let mut masks = TrackedMasks {
            frames: Vec::with_capacity(tracked.len()),
            masks: Vec::with_capacity(tracked.len()),
            scores: Vec::with_capacity(tracked.len()),
        };
        for (index, frame) in tracked.into_iter().enumerate() {
            let number = if parameters.reverse {
                input.frame - index
            } else {
                input.frame + index
            };
            let image = &frames[number];
            let inside = frame
                .scores
                .cast(DataType::F32)?
                .interpolate(
                    Spatial2d::new(i64::from(image.height), i64::from(image.width)),
                    InterpolateMode::Bilinear,
                )?
                .greater(&threshold)?
                .cast(DataType::F32)?
                .contiguous()?
                .to_vec::<f32>()?;
            masks.frames.push(number);
            masks.masks.push(mask(
                image.width,
                image.height,
                inside.into_iter().map(|value| value > 0.5),
            ));
            masks.scores.push(1.0 / (1.0 + (-frame.object_score).exp()));
        }
        Ok(masks)
    }
}
