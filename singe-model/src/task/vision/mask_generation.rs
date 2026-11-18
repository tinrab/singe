// `mask-generation`: an image to masks of everything in it, as `transformers`' pipeline finds them with SAM: a grid of points, each decoded into candidate masks, kept by their predicted quality and their stability, resized to the image, and deduplicated by their boxes, best first.

use std::slice;

use serde::{Deserialize, Serialize};
use singe::{DataType, Shape, Tensor, tensor::InterpolateMode};

use crate::{
    error::Result,
    head::MaskGenerator,
    model,
    processing::{
        host_values,
        image::{Image, ImageTransforms, longest_edge},
    },
    source::{LoadOptions, Source},
    task::{Pipeline, Task, vision::image_segmentation::mask},
};

#[derive(Debug)]
pub struct MaskGenerationPipeline {
    model: MaskGenerator,
    processor: ImageTransforms,
    data_type: DataType,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default)]
pub struct MaskGenerationParameters {
    // The points along each side of the grid.
    pub points_per_crop: usize,
    // The points decoded together.
    pub points_per_batch: usize,
    // The least predicted quality a mask keeps.
    pub pred_iou_thresh: f32,
    // The least share a mask keeps of its points when its cutoff moves by the offset either way.
    pub stability_score_thresh: f32,
    pub stability_score_offset: f32,
    // The score a mask's points pass.
    pub mask_threshold: f32,
    // The most a kept mask's box may overlap a better one's.
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

// A grid of `side × side` points at its cells' centers, `[x, y]` in [0, 1], row by row, as `_build_point_grid` lays it out with `torch.linspace`.
#[expect(clippy::cast_precision_loss, reason = "grid sides are small")]
#[expect(
    clippy::suboptimal_flops,
    reason = "rounds each product as the reference does, which the verifications compare with"
)]
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

// A mask's box, `[left, top, right, bottom]` of its first and last rows and columns, or zeros when it is empty, as `_batched_mask_to_box` gives it.
fn mask_box(inside: &[bool], width: usize) -> [f32; 4] {
    let (mut left, mut top, mut right, mut bottom) = (usize::MAX, usize::MAX, 0, 0);
    for index in (0..inside.len()).filter(|&index| inside[index]) {
        let (row, column) = (index / width, index % width);
        left = left.min(column);
        right = right.max(column);
        top = top.min(row);
        bottom = bottom.max(row);
    }
    #[expect(clippy::cast_precision_loss, reason = "image sides are small")]
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

// The indices `torchvision.ops.nms` keeps, best first: each box overlapping no better kept box by more than `threshold`.
fn suppress(boxes: &[[f32; 4]], scores: &[f32], threshold: f32) -> Vec<usize> {
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
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

impl Pipeline for MaskGenerationPipeline {
    const TASK: Task = Task::MaskGeneration;

    type Input = Image;

    type Parameters = MaskGenerationParameters;

    type Output = GeneratedMasks;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            data_type: options.data_type,
        })
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "image sides and pixel counts are far below 2^24"
    )]
    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let device = self.model.model().state().device()?;
        let target = self.model.image_size();
        inputs
            .iter()
            .map(|image| {
                let (height, width) = (i64::from(image.height), i64::from(image.width));
                let pixels =
                    self.processor
                        .preprocess(slice::from_ref(image), device, self.data_type)?;
                let embedding = self.model.embed(&pixels)?;

                // The resized image's size, which points scale to and masks are cropped to, as `_normalize_coordinates` rounds it.
                let (resized_height, resized_width) = longest_edge(height, width, target);
                let (scale_x, scale_y) = (
                    resized_width as f64 / width as f64,
                    resized_height as f64 / height as f64,
                );
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "`transformers` scales `F32` points by these"
                )]
                let points = point_grid(parameters.points_per_crop)
                    .into_iter()
                    .map(|[x, y]| {
                        [
                            x * width as f32 * scale_x as f32,
                            y * height as f32 * scale_y as f32,
                        ]
                    })
                    .collect::<Vec<_>>();

                let pixel_count = image.width as usize * image.height as usize;
                let level = |value: f32| -> Result<Tensor> {
                    Ok(Tensor::from_vec(
                        device,
                        Shape::create([1, 1, 1])?,
                        vec![value],
                    )?)
                };
                let (above, below, inside) = (
                    level(parameters.mask_threshold + parameters.stability_score_offset)?,
                    level(parameters.mask_threshold - parameters.stability_score_offset)?,
                    level(parameters.mask_threshold)?,
                );
                let count_where = |masks: &Tensor, level: &Tensor| -> Result<Vec<f32>> {
                    Ok(masks
                        .greater(level)?
                        .cast(DataType::F32)?
                        .sum(&[1, 2], false)?
                        .to_vec::<f32>()?)
                };
                let (mut masks, mut scores, mut boxes) = (Vec::new(), Vec::new(), Vec::new());
                for batch in points.chunks(parameters.points_per_batch.max(1)) {
                    let (low, quality) = self.model.decode(&embedding, batch)?;
                    let dimensions = low.shape().dimensions();
                    let (count, candidates) = (dimensions[0], dimensions[1]);
                    // Resized to the padded input, cropped to the resized image, and resized to the image, as `post_process_masks` does.
                    let resized_masks = low
                        .cast(DataType::F32)?
                        .interpolate((target, target), InterpolateMode::Bilinear)?
                        .narrow(2, 0, resized_height)?
                        .narrow(3, 0, resized_width)?
                        .interpolate((height, width), InterpolateMode::Bilinear)?
                        .reshape([count * candidates, height, width])?;
                    let quality = host_values(&quality)?;
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

                        let row = i64::try_from(index)
                            .map_err(|_| singe::Error::out_of_range("masks"))?;
                        let found = resized_masks
                            .narrow(0, row, 1)?
                            .greater(&inside)?
                            .cast(DataType::F32)?
                            .contiguous()?
                            .to_vec::<f32>()?
                            .into_iter()
                            .map(|value| value > 0.5)
                            .collect::<Vec<_>>();
                        debug_assert_eq!(found.len(), pixel_count);
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
            })
            .collect()
    }
}
