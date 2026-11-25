//! Qwen3.5's image front end, as `transformers`' processor reads a still image: resized into `smart_resize`'s size, normalized, and cut into patches in the order that puts each merged block's patches together.

use serde::{Deserialize, Serialize};
use singe::{
    Device, Error as SingeError, Shape, Tensor,
    shape::spatial::{Spatial2d, Spatial2dI64},
    tensor::{InterpolateMode, Interpolation},
};

use crate::{error::Result, model::text::qwen3_5::Qwen35VisionConfig, processor::image::Image};

/// The fewest and most pixels an image is resized into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSizeBounds {
    pub min_pixels: i64,
    pub max_pixels: i64,
}

// no-slop-allow: no-vague-type-names -- this mirrors a `transformers` image processor.
/// How the image front end resizes and normalizes an image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35ImageProcessor {
    pub bounds: ImageSizeBounds,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// An image as the encoder reads it: its patches, `[N, C · T · P · P]` in `F32`, in the order that puts each merged block's patches together, and its grid of patches, rows then columns.
#[derive(Debug, Clone)]
pub(crate) struct ImagePatches {
    pub(crate) patches: Tensor,
    pub(crate) grid: [i64; 2],
}

impl Qwen35ImageProcessor {
    /// Returns one image's patches: the image resized bicubically into `smart_resize`'s size, as `transformers`' processor resizes it, then normalized and cut into patches.
    pub(crate) fn patches(
        &self,
        image: &Image,
        vision: &Qwen35VisionConfig,
        device: Device,
    ) -> Result<ImagePatches> {
        let size = Spatial2d::new(i64::from(image.height), i64::from(image.width));
        let resized = smart_resize(size, vision.size_factor(), self.bounds);
        let mut pixels = image.to_tensor(device)?;
        if resized != size {
            let interpolation = Interpolation::new(InterpolateMode::Bicubic).with_antialias(true);
            pixels = pixels
                .interpolate(resized, interpolation)?
                .clamp(0.0, 255.0)?
                .round()?;
        }

        let pixels = pixels
            .mul_scalar(1.0 / 255.0)?
            .contiguous()?
            .to_vec::<f32>()?;
        self.cut(&pixels, resized.height, resized.width, vision, device)
    }

    /// Cuts an image, `[3, H, W]` channel-major RGB in `[0, 1]` with each side a multiple of the size factor, into normalized patches on `device`, each frame of a patch the image itself, as `transformers`' processor does for a still image.
    fn cut(
        &self,
        pixels: &[f32],
        height: i64,
        width: i64,
        vision: &Qwen35VisionConfig,
        device: Device,
    ) -> Result<ImagePatches> {
        let factor = vision.size_factor();
        if height <= 0 || width <= 0 || height % factor != 0 || width % factor != 0 {
            return Err(SingeError::invalid_argument(format!(
                "an image's sides are positive multiples of {factor}"
            ))
            .into());
        }

        let (height, width) = (to_usize(height)?, to_usize(width)?);
        if pixels.len() != 3 * height * width {
            return Err(SingeError::length_mismatch(
                "image pixels",
                3 * height * width,
                pixels.len(),
            )
            .into());
        }

        let patch = to_usize(vision.patch_size)?;
        let frames = to_usize(vision.frames)?;
        let grid = [
            i64::try_from(height / patch).map_err(|_| SingeError::out_of_range("image"))?,
            i64::try_from(width / patch).map_err(|_| SingeError::out_of_range("image"))?,
        ];
        let places = patch_order(vision, grid);
        let mut values = Vec::with_capacity(places.len() * 3 * frames * patch * patch);
        for (row, column) in places {
            let (top, left) = (to_usize(row)? * patch, to_usize(column)? * patch);
            for channel in 0..3 {
                let (mean, std) = (self.mean[channel], self.std[channel]);
                for _ in 0..frames {
                    for y in top..top + patch {
                        let line = &pixels[(channel * height + y) * width + left..][..patch];
                        values.extend(line.iter().map(|value| (value - mean) / std));
                    }
                }
            }
        }

        let count = grid[0] * grid[1];
        Ok(ImagePatches {
            patches: Tensor::from_vec(
                device,
                Shape::create([count, vision.patch_features()])?,
                values,
            )?,
            grid,
        })
    }
}

/// Returns the size `transformers`' `smart_resize` gives an image of `size`: each side a multiple of `factor`, the area within `bounds`, and the aspect ratio kept as closely as that allows.
fn smart_resize(size: Spatial2dI64, factor: i64, bounds: ImageSizeBounds) -> Spatial2dI64 {
    let (h, w, f) = (size.height as f64, size.width as f64, factor as f64);
    let area = h * w;
    let rounded = size.map(|side| ((side as f64 / f).round_ties_even() as i64) * factor);
    if rounded.height * rounded.width > bounds.max_pixels {
        let beta = (area / bounds.max_pixels as f64).sqrt();
        size.map(|side| factor.max(((side as f64 / beta / f).floor() as i64) * factor))
    } else if rounded.height * rounded.width < bounds.min_pixels {
        let beta = (bounds.min_pixels as f64 / area).sqrt();
        size.map(|side| ((side as f64 * beta / f).ceil() as i64) * factor)
    } else {
        rounded
    }
}

/// Returns each patch's row and column, in the order that puts each merged block's patches together, as the image front end cuts them.
fn patch_order(config: &Qwen35VisionConfig, [rows, columns]: [i64; 2]) -> Vec<(i64, i64)> {
    let merge = config.merge_size;
    let mut places = Vec::with_capacity(usize::try_from(rows * columns).unwrap_or(0));
    for block_row in 0..rows / merge {
        for block_column in 0..columns / merge {
            for row in 0..merge {
                for column in 0..merge {
                    places.push((block_row * merge + row, block_column * merge + column));
                }
            }
        }
    }
    places
}

fn to_usize(value: i64) -> Result<usize> {
    usize::try_from(value).map_err(|_| SingeError::out_of_range("image size").into())
}
