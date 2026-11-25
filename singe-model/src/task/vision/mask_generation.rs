//! `mask-generation`: an image to masks of everything in it, as `transformers`' pipeline finds them with SAM: a grid of points, each decoded into candidate masks, kept by their predicted quality and their stability, resized to the image, and deduplicated by their boxes, best first.

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Shape, Tensor,
    shape::spatial::{Spatial2d, Spatial2dI64},
    tensor::InterpolateMode,
};

use crate::{
    error::Result,
    head::MaskGenerator,
    host, model,
    processor::image::{Image, ImageTransforms, Resize},
    source::{LoadOptions, Source},
    task::{Pipeline, vision::pixels::mask},
};

#[derive(Debug)]
pub struct MaskGenerationPipeline {
    model: MaskGenerator,
    processor: ImageTransforms,
    options: LoadOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default)]
pub struct MaskGenerationParameters {
    /// The points along each side of the grid, 32 by default.
    pub points_per_crop: usize,
    /// The points decoded together, 64 by default.
    pub points_per_batch: usize,
    /// The predicted quality a mask must exceed to be kept, 0.88 by default.
    pub pred_iou_thresh: f32,
    /// The stability a mask must exceed to be kept, 0.95 by default: the share of its points above the cutoff raised by `stability_score_offset` among those above the cutoff lowered by as much.
    pub stability_score_thresh: f32,
    /// How far the cutoff moves either way for a mask's stability, 1 by default.
    pub stability_score_offset: f32,
    /// The cutoff a point's score must exceed for the point to be in a mask, 0 by default.
    pub mask_threshold: f32,
    /// The most a kept mask's box may overlap a better kept one's, as intersection over union, 0.7 by default.
    pub crops_nms_thresh: f32,
}

impl Default for MaskGenerationParameters {
    fn default() -> Self {
        Self {
            points_per_crop: 32,
            points_per_batch: 64,
            pred_iou_thresh: 0.88,
            stability_score_thresh: 0.95,
            stability_score_offset: 1.0,
            mask_threshold: 0.0,
            crops_nms_thresh: 0.7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GeneratedMasks {
    pub masks: Vec<Image>,
    pub scores: Vec<f32>,
}

/// Returns a grid of `side × side` points at its cells' centers, `[x, y]` in `[0, 1]`, row by row, as `_build_point_grid` lays it out with `torch.linspace`.
fn point_grid(side: usize) -> Vec<[f32; 2]> {
    let offset = 1.0 / (2.0 * side as f32);
    let step = (1.0 - 2.0 * offset) / (side as f32 - 1.0).max(1.0);
    let along = (0..side)
        .map(|index| offset + step * index as f32)
        .collect::<Vec<_>>();
    along
        .iter()
        .flat_map(|&y| along.iter().map(move |&x| [x, y]))
        .collect()
}

/// Returns a mask's box, `[left, top, right, bottom]` of its first and last rows and columns, or zeros when it is empty, as `_batched_mask_to_box` gives it.
fn mask_box(inside: &[bool], width: usize) -> [f32; 4] {
    let (mut left, mut top, mut right, mut bottom) = (usize::MAX, usize::MAX, 0, 0);
    for index in (0..inside.len()).filter(|&index| inside[index]) {
        let (row, column) = (index / width, index % width);
        left = left.min(column);
        right = right.max(column);
        top = top.min(row);
        bottom = bottom.max(row);
    }
    if left == usize::MAX {
        [0.0; 4]
    } else {
        [left as f32, top as f32, right as f32, bottom as f32]
    }
}

fn overlap(first: &[f32; 4], second: &[f32; 4]) -> f32 {
    let area = |found: &[f32; 4]| (found[2] - found[0]) * (found[3] - found[1]);
    let width = (first[2].min(second[2]) - first[0].max(second[0])).max(0.0);
    let height = (first[3].min(second[3]) - first[1].max(second[1])).max(0.0);
    let intersection = width * height;
    intersection / (area(first) + area(second) - intersection)
}

/// Returns the indices `torchvision.ops.nms` keeps, best first: each box overlapping no better kept box by more than `threshold`.
fn suppress(boxes: &[[f32; 4]], scores: &[f32], threshold: f32) -> Vec<usize> {
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    host::best_first(&mut order, usize::MAX, |&index| scores[index]);
    let mut kept: Vec<usize> = Vec::new();
    for index in order {
        if kept
            .iter()
            .all(|&better| overlap(&boxes[better], &boxes[index]) <= threshold)
        {
            kept.push(index);
        }
    }
    kept
}

fn scaled_grid(
    side: usize,
    Spatial2d { height, width }: Spatial2dI64,
    resized: Spatial2dI64,
) -> Vec<[f32; 2]> {
    let (scale_x, scale_y) = (
        resized.width as f64 / width as f64,
        resized.height as f64 / height as f64,
    );
    point_grid(side)
        .into_iter()
        .map(|[x, y]| {
            [
                x * width as f32 * scale_x as f32,
                y * height as f32 * scale_y as f32,
            ]
        })
        .collect()
}

fn count_where(masks: &Tensor, level: &Tensor) -> Result<Vec<f32>> {
    Ok(masks
        .greater(level)?
        .cast(DataType::F32)?
        .sum(&[1, 2], false)?
        .to_vec::<f32>()?)
}

impl MaskGenerationPipeline {
    fn decode(
        &self,
        embedding: &Tensor,
        points: &[[f32; 2]],
        size: Spatial2dI64,
        resized: Spatial2dI64,
    ) -> Result<(Tensor, Vec<f32>)> {
        let target = self.model.image_size();
        let (low, quality) = self.model.decode(embedding, points)?;
        let &[count, candidates, ..] = low.shape().dimensions() else {
            return Err(SingeError::invalid_argument("candidate masks, [P, M, h, w]").into());
        };
        // Resized to the padded input, cropped to the resized image, and resized to the image, as `post_process_masks` does.
        let masks = low
            .cast(DataType::F32)?
            .interpolate(Spatial2d::square(target), InterpolateMode::Bilinear)?
            .narrow(2, 0, resized.height)?
            .narrow(3, 0, resized.width)?
            .interpolate(size, InterpolateMode::Bilinear)?
            .reshape([count * candidates, size.height, size.width])?;
        Ok((masks, quality.cast(DataType::F32)?.to_vec::<f32>()?))
    }
}

impl Pipeline for MaskGenerationPipeline {
    type Input = Image;

    type Parameters = MaskGenerationParameters;

    type Output = GeneratedMasks;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            options,
        })
    }

    fn run_one(&self, image: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let size = Spatial2d::new(i64::from(image.height), i64::from(image.width));
        let pixels =
            self.processor
                .preprocess(image, self.options.device, self.options.data_type)?;
        let embedding = self.model.embed(&pixels)?;
        // The resized image's size, which points scale to and masks are cropped to, as `_normalize_coordinates` rounds it.
        let resized = Resize::Longest(self.model.image_size()).size(size);
        let level = |value: f32| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                self.options.device,
                Shape::create([1, 1, 1])?,
                vec![value],
            )?)
        };
        let (above, below, inside) = (
            level(parameters.mask_threshold + parameters.stability_score_offset)?,
            level(parameters.mask_threshold - parameters.stability_score_offset)?,
            level(parameters.mask_threshold)?,
        );
        let (mut masks, mut scores, mut boxes) = (Vec::new(), Vec::new(), Vec::new());
        let points = scaled_grid(parameters.points_per_crop, size, resized);
        for batch in points.chunks(parameters.points_per_batch.max(1)) {
            let (resized_masks, quality) = self.decode(&embedding, batch, size, resized)?;
            let intersections = count_where(&resized_masks, &above)?;
            let unions = count_where(&resized_masks, &below)?;
            for (index, &score) in quality.iter().enumerate() {
                let stability = intersections[index] / unions[index];
                if score <= parameters.pred_iou_thresh
                    || stability.is_nan()
                    || stability <= parameters.stability_score_thresh
                {
                    continue;
                }

                let row = i64::try_from(index).map_err(|_| SingeError::out_of_range("masks"))?;
                let found = resized_masks
                    .narrow(0, row, 1)?
                    .greater(&inside)?
                    .cast(DataType::F32)?
                    .contiguous()?
                    .to_vec::<f32>()?
                    .into_iter()
                    .map(|value| value > 0.5)
                    .collect::<Vec<_>>();
                boxes.push(mask_box(&found, image.width as usize));
                masks.push(found);
                scores.push(score);
            }
        }

        let kept = suppress(&boxes, &scores, parameters.crops_nms_thresh);
        Ok(GeneratedMasks {
            masks: kept
                .iter()
                .map(|&index| mask(image.width, image.height, masks[index].iter().copied()))
                .collect(),
            scores: kept.iter().map(|&index| scores[index]).collect(),
        })
    }
}
