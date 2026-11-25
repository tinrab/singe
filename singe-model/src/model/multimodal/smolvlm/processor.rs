//! `SmolVLM`'s processor, ported from `transformers`' `SmolVLMImageProcessor`, `SmolVLMVideoProcessor`, and `SmolVLMProcessor`: images cut into tiles, clips sampled into frames, both resized as torchvision resizes bytes, and the prompt's placeholders expanded into image tokens between their markers.

use std::fmt::Write;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Shape, Tensor,
    shape::spatial::{Spatial2d, Spatial2dI64},
};

use crate::{
    error::{Error, Result},
    processor::{
        image::{Image, Resample, extent},
        video::Video,
    },
};

/// The placeholder a clip leaves in a rendered conversation.
pub(crate) const VIDEO_TOKEN: &str = "<video>";

/// `SmolVLMProcessor`'s `DEFAULT_CHAT_TEMPLATE`, which it renders a conversation with a clip through instead of the snapshot's.
///
/// Unlike the original, it also renders a message whose content is a plain string.
pub(crate) const VIDEO_CHAT_TEMPLATE: &str = "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'] is string %}{{': ' + message['content']}}{% else %}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% elif line['type'] == 'video' %}{{ '<video>' }}{% endif %}{% endfor %}{% endif %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}";

/// The largest side `transformers` resizes an image to.
const MAX_IMAGE_SIZE: i64 = 4096;

/// The seconds `sample_frames` skips at each end of a long enough clip.
const SKIPPED_SECONDS: f64 = 1.0;

// no-slop-allow: no-vague-type-names -- this is `SmolVLMImageProcessor`, a `transformers` image processor.
/// How `SmolVLM`'s front end splits images into tiles, samples video frames, normalizes pixels, and expands a prompt's image placeholders into tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmolVlmProcessor {
    /// The longest side an image is resized to before it is cut.
    pub longest_edge: i64,
    /// The side of a tile, which the vision tower reads.
    pub tile: i64,
    /// Whether an image larger than a tile is cut into tiles, rather than shrunk to one.
    pub splitting: bool,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub resample: Resample,
    /// The number of image tokens one tile's rows take.
    pub image_tokens: usize,
    /// How clips are sampled, for a processor that reads them.
    pub video: Option<VideoSampling>,
    pub image_token: String,
    pub fake_image_token: String,
    pub global_image_token: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VideoSampling {
    /// The frames per second a clip is sampled at.
    pub fps: f64,
    /// The most frames sampled from a clip.
    pub max_frames: usize,
    /// The longest side a frame is resized to before it is squared to a tile.
    pub edge: i64,
}

impl VideoSampling {
    /// Returns the indices of the frames `sample_frames` takes from a clip of `total` frames decoded at `fps` frames per second.
    ///
    /// It takes about [`fps`](Self::fps) frames a second, up to `max_frames`, spread evenly, skipping a second at each end of a long clip, and fails for a clip without frames or a positive frame rate.
    fn sampled(&self, total: usize, fps: f64) -> Result<Vec<usize>> {
        if total == 0 || fps <= 0.0 {
            return Err(Error::Media(
                "a clip without frames or a frame rate".to_owned(),
            ));
        }

        let duration = total as f64 / fps;
        let estimated = (self.fps * duration).round_ties_even() as usize;
        let desired = estimated.min(self.max_frames).max(1);
        let (mut start, mut end) = (0, total - 1);
        if duration - 2.0 * SKIPPED_SECONDS > self.max_frames as f64 * self.fps {
            start = (SKIPPED_SECONDS * fps) as usize;
            end = (total as f64 - SKIPPED_SECONDS * fps) as usize;
        }

        end = end.min(total - 1);
        if start >= end {
            (start, end) = (0, total - 1);
        }

        let step = if desired > 1 {
            (end - start) as f64 / (desired - 1) as f64
        } else {
            0.0
        };
        let mut indices = (0..desired)
            .map(|index| {
                if desired > 1 && index == desired - 1 {
                    end
                } else {
                    (index as f64 * step + start as f64) as usize
                }
            })
            .collect::<Vec<_>>();
        indices.dedup();
        Ok(indices)
    }
}

/// Returns `value` truncated toward zero, as Python's `int` gives it for a positive float.
const fn truncated(value: f64) -> i64 {
    value as i64
}

const fn float(value: i64) -> f64 {
    value as f64
}

/// Returns the size `_resize_output_size_rescale_to_max_len` gives: the longer side becomes `longest`, and the other keeps the aspect ratio, truncated and rounded up to even.
fn rescaled_to_longest(size: Spatial2dI64, longest: i64) -> Spatial2dI64 {
    let Spatial2d {
        mut height,
        mut width,
    } = size;
    let aspect = float(width) / float(height);
    if width >= height {
        width = longest;
        height = truncated(float(width) / aspect);
        height += height % 2;
    } else {
        height = longest;
        width = truncated(float(height) * aspect);
        width += width % 2;
    }
    Spatial2d::new(height.max(1), width.max(1))
}

/// Returns the size `_resize_output_size_scale_below_upper_bound` gives: a side above [`MAX_IMAGE_SIZE`] shrinks to it, keeping the aspect ratio.
fn bounded(size: Spatial2dI64) -> Spatial2dI64 {
    let Spatial2d {
        mut height,
        mut width,
    } = size;
    let aspect = float(width) / float(height);
    if width >= height && width > MAX_IMAGE_SIZE {
        width = MAX_IMAGE_SIZE;
        height = truncated(float(width) / aspect);
    } else if height > width && height > MAX_IMAGE_SIZE {
        height = MAX_IMAGE_SIZE;
        width = truncated(float(height) * aspect);
    }
    Spatial2d::new(height.max(1), width.max(1))
}

/// Returns the size `resize_for_vision_encoder` gives: both sides rounded up to whole tiles, the shorter from the aspect ratio after the longer.
fn whole_tiles(Spatial2d { height, width }: Spatial2dI64, tile: i64) -> Spatial2dI64 {
    let aspect = float(width) / float(height);
    let up = |side: i64| (side + tile - 1) / tile * tile;
    if width >= height {
        let width = up(width);
        Spatial2d::new(up(truncated(float(width) / aspect)), width)
    } else {
        let height = up(height);
        Spatial2d::new(height, up(truncated(float(height) * aspect)))
    }
}

/// Returns the English words `num2words` gives for `number` below 1000, and its digits from 1000 up.
fn number_words(number: usize) -> String {
    const ONES: &[&str] = &[
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];

    const TENS: &[&str] = &[
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    match number {
        0..20 => ONES[number].to_owned(),
        20..100 if number.is_multiple_of(10) => TENS[number / 10].to_owned(),
        20..100 => format!("{}-{}", TENS[number / 10], ONES[number % 10]),
        100..1000 if number.is_multiple_of(100) => format!("{} hundred", ONES[number / 100]),
        100..1000 => format!(
            "{} hundred and {}",
            ONES[number / 100],
            number_words(number % 100)
        ),
        _ => number.to_string(),
    }
}

impl SmolVlmProcessor {
    fn resized(&self, image: &Image, height: i64, width: i64) -> Result<Image> {
        let side = |value: i64| extent(value.clamp(1, MAX_IMAGE_SIZE));
        let (height, width) = (side(height)?, side(width)?);
        Ok(if (image.height, image.width) == (height, width) {
            image.clone()
        } else {
            image.resized(height, width, self.resample)
        })
    }

    /// Cuts `image` into tiles as `SmolVLMImageProcessor` does.
    ///
    /// Returns the tiles row by row, then the whole image shrunk to one, with the grid of tiles; or the image as one tile and no grid when it fits one or splitting is off.
    pub fn tiles(&self, image: &Image) -> Result<(Vec<Image>, Option<Spatial2dI64>)> {
        let size = Spatial2d::new(i64::from(image.height), i64::from(image.width));
        let size = bounded(rescaled_to_longest(size, self.longest_edge));
        let image = self.resized(image, size.height, size.width)?;
        let tile = self.tile;
        if !self.splitting {
            return Ok((vec![self.resized(&image, tile, tile)?], None));
        }

        let Spatial2d { height, width } = whole_tiles(size, tile);
        let image = self.resized(&image, height, width)?;
        if height <= tile && width <= tile {
            return Ok((vec![image], None));
        }

        let grid = Spatial2d::new(height / tile, width / tile);
        let mut tiles = Vec::new();
        for row in 0..grid.height {
            for column in 0..grid.width {
                tiles.push(crop(&image, row * tile, column * tile, tile)?);
            }
        }

        tiles.push(self.resized(&image, tile, tile)?);
        Ok((tiles, Some(grid)))
    }

    /// Returns the text an image's placeholder becomes: each tile's marker and image tokens, row by row, then the whole image's, as `get_image_prompt_string` writes it.
    #[must_use]
    pub fn image_prompt(&self, grid: Option<Spatial2dI64>) -> String {
        let tokens = self.image_token.repeat(self.image_tokens);
        let whole = format!(
            "{fake}{global}{tokens}{fake}",
            fake = self.fake_image_token,
            global = self.global_image_token,
        );
        let Some(Spatial2d {
            height: rows,
            width: columns,
        }) = grid
        else {
            return whole;
        };
        let mut text = String::new();
        for row in 1..=rows {
            for column in 1..=columns {
                let _ = write!(
                    text,
                    "{}<row_{row}_col_{column}>{tokens}",
                    self.fake_image_token
                );
            }

            text.push('\n');
        }

        text.push('\n');
        text + &whole
    }

    /// Replaces each `placeholder` in `prompt` with the next of `expansions`.
    ///
    /// # Errors
    ///
    /// Returns an error if `prompt` does not hold exactly one `placeholder` for each of `expansions`.
    pub fn expand(&self, prompt: &str, placeholder: &str, expansions: &[String]) -> Result<String> {
        let pieces = prompt.split(placeholder).collect::<Vec<_>>();
        if pieces.len() != expansions.len() + 1 {
            return Err(SingeError::length_mismatch(
                format!("the prompt's {placeholder} placeholders"),
                expansions.len(),
                pieces.len() - 1,
            )
            .into());
        }

        let mut text = pieces[0].to_owned();
        for (expansion, piece) in expansions.iter().zip(&pieces[1..]) {
            text.push_str(expansion);
            text.push_str(piece);
        }
        Ok(text)
    }

    /// Returns a clip's sampled frames, each squared to a tile, and the text its placeholder becomes, as `SmolVLMVideoProcessor` samples and resizes them and `replace_video_token` introduces them, for a clip decoded at `fps` frames per second.
    ///
    /// # Errors
    ///
    /// Returns an error if the processor reads no clips, or if the clip has no frames or `fps` is not positive.
    pub fn frames(&self, video: &Video, fps: f64) -> Result<(Vec<Image>, String)> {
        let sampling = self
            .video
            .ok_or_else(|| Error::unsupported("a processor that reads no clips"))?;
        let indices = sampling.sampled(video.frames.len(), fps)?;
        let frames = indices
            .iter()
            .map(|&index| {
                let frame = &video.frames[index];
                let size = rescaled_to_longest(
                    Spatial2d::new(i64::from(frame.height), i64::from(frame.width)),
                    sampling.edge.min(MAX_IMAGE_SIZE),
                );
                let frame = self.resized(frame, size.height, size.width)?;
                self.resized(&frame, self.tile, self.tile)
            })
            .collect::<Result<Vec<_>>>()?;
        let (duration, seconds) = (
            (video.frames.len() as f64 / fps) as u64,
            indices
                .iter()
                .map(|&index| (index as f64 / fps) as u64)
                .collect::<Vec<_>>(),
        );
        let mut text = format!(
            "You are provided the following series of {} frames from a {}:{:02}:{:02} [H:MM:SS] video.\n",
            number_words(frames.len()),
            duration / 3600,
            duration / 60 % 60,
            duration % 60
        );
        let image = self.image_prompt(None);
        for second in seconds {
            let _ = write!(
                text,
                "\nFrame from {:02}:{:02}:{image}",
                second / 60,
                second % 60
            );
        }

        text.push_str("\n\n");
        Ok((frames, text))
    }

    /// Returns tiles as the tower reads them, `[N, 3, S, S]`, rescaled and normalized as `transformers` fuses the two.
    pub fn pixels(&self, tiles: &[Image], device: Device, data_type: DataType) -> Result<Tensor> {
        let side = usize::try_from(self.tile).map_err(|_| SingeError::out_of_range("tile side"))?;
        let mut values = Vec::with_capacity(tiles.len() * 3 * side * side);
        for tile in tiles {
            for channel in 0..3 {
                let (mean, std) = (self.mean[channel] * 255.0, self.std[channel] * 255.0);
                values.extend(
                    tile.pixels
                        .iter()
                        .skip(channel)
                        .step_by(3)
                        .map(|&value| (f32::from(value) - mean) / std),
                );
            }
        }

        let count = i64::try_from(tiles.len()).map_err(|_| SingeError::out_of_range("tiles"))?;
        Ok(Tensor::from_vec(
            device,
            Shape::create([count, 3, self.tile, self.tile])?,
            values,
        )?
        .cast(data_type)?)
    }
}

/// Returns the `side × side` square of `image` from `top`, `left`.
fn crop(image: &Image, top: i64, left: i64, side: i64) -> Result<Image> {
    let width = image.width as usize;
    let (top, left, side) = (top as usize, left as usize, side as usize);
    let pixels = (top..top + side)
        .flat_map(|row| {
            let start = (row * width + left) * 3;
            image.pixels[start..start + side * 3].iter().copied()
        })
        .collect();
    let side = u32::try_from(side).map_err(|_| SingeError::out_of_range("tile side"))?;
    Ok(Image {
        width: side,
        height: side,
        pixels,
    })
}
