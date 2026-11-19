// `keypoint-matching`: a pair of images to their matched keypoints, each at its image's whole pixels, with the match's score, best first, as `transformers`' pipeline gives them.

use serde::{Deserialize, Serialize};
use singe::Error as SingeError;

use crate::{
    error::Result,
    head::KeypointMatcher,
    host, model,
    processor::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::{Pipeline, vision::pixels::rescaled_point},
};

#[derive(Debug)]
pub struct KeypointMatchingPipeline {
    model: KeypointMatcher,
    processor: ImageTransforms,
    options: LoadOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default)]
pub struct MatchingParameters {
    // The score a match must pass to be kept.
    pub threshold: f32,
}

impl Default for MatchingParameters {
    fn default() -> Self {
        Self { threshold: 0.0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Keypoint {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Match {
    pub keypoint_image_0: Keypoint,
    pub keypoint_image_1: Keypoint,
    pub score: f32,
}

// A keypoint at the pixels of an image of `from` pixels, `(height, width)`, at `image`'s.
fn rescaled(point: [f32; 2], from: (i64, i64), image: &Image) -> Keypoint {
    let [x, y] = rescaled_point(point, from, image);
    Keypoint { x, y }
}

impl Pipeline for KeypointMatchingPipeline {
    type Input = [Image; 2];

    type Parameters = MatchingParameters;

    type Output = Vec<Match>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            options,
        })
    }

    fn run_one(&self, pair: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let pixels =
            self.processor
                .preprocess_batch(pair, self.options.device, self.options.data_type)?;
        let &[_, _, height, width] = pixels.shape().dimensions() else {
            return Err(SingeError::invalid_argument("images, [B, C, H, W]").into());
        };
        let size = (height, width);
        let found = self.model.match_pair(&pixels)?;
        let mut matches = found
            .matches
            .iter()
            .zip(&found.scores)
            .enumerate()
            .filter(|&(_, (&other, &score))| other >= 0 && score > parameters.threshold)
            .filter_map(|(index, (&other, &score))| {
                let other = found.keypoints[1].get(usize::try_from(other).ok()?)?;
                Some(Match {
                    keypoint_image_0: rescaled(found.keypoints[0][index], size, &pair[0]),
                    keypoint_image_1: rescaled(*other, size, &pair[1]),
                    score,
                })
            })
            .collect::<Vec<_>>();
        // Best first, equal scores in the first image's order, as Python's stable sort leaves them.
        host::best_first(&mut matches, usize::MAX, |found| found.score);
        Ok(matches)
    }
}
