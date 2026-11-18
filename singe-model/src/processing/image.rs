// Images as models read them, and the image processor most vision models share: resize on the CPU, then center crop, rescale, and normalize on the device, as `transformers`' default image processors compute them with torchvision.
// A model whose processor does more, such as Qwen3.5's patches of any size, keeps it with the model, as `transformers` does.

use std::{io::Cursor, path::Path, result::Result as StdResult};

use image::{ImageFormat, RgbImage};
use serde::{
    Deserialize, Deserializer, Serialize, Serializer, de::Error as DeError, ser::Error as SerError,
};
use serde_json::Value;
use singe::{DataType, Device, Shape, Tensor};

use crate::{
    error::{Error, Result},
    source::Source,
};

// An RGB image, `height × width × 3` bytes in row-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Image {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let image = image::open(path)
            .map_err(|error| Error::Media(format!("{}: {error}", path.display())))?
            .to_rgb8();
        Ok(Self {
            width: image.width(),
            height: image.height(),
            pixels: image.into_raw(),
        })
    }

    // Returns the image as `[1, 3, H, W]` of `F32` values in [0, 255] on `device`.
    pub fn to_tensor(&self, device: Device) -> Result<Tensor> {
        let values = self.pixels.iter().map(|&value| f32::from(value)).collect();
        let shape = Shape::create([i64::from(self.height), i64::from(self.width), 3])?;
        Ok(Tensor::from_vec(device, shape, values)?
            .permute(&[2, 0, 1])?
            .unsqueeze(0)?)
    }
}

// An image in JSON names the file it is read from.
impl<'de> Deserialize<'de> for Image {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> StdResult<Self, D::Error> {
        let path = String::deserialize(deserializer)?;
        Self::open(&path).map_err(DeError::custom)
    }
}

// The filter a resize uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resample {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos,
}

// The size a resize gives an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resize {
    // The shorter side becomes this long, and the longer keeps the aspect ratio, truncated.
    ShortestEdge(i64),
    // The shorter side becomes `shortest` unless the longer would pass `longest`, which then bounds it, and both sides round down to a multiple of `multiple`, as DETR-style processors size images.
    Bounded {
        shortest: i64,
        longest: i64,
        multiple: i64,
    },
    Exact {
        height: i64,
        width: i64,
    },
    // The longer side becomes this long, and the shorter keeps the aspect ratio, rounded, as SAM's processor sizes images.
    Longest(i64),
    // The shorter side becomes the shorter of `height` and `width`, then the image shrinks, keeping its aspect ratio, until it fits within `height × width`, as Donut's processor resizes and thumbnails images.
    Thumbnail {
        height: i64,
        width: i64,
    },
    // Toward `height × width`, scaling both sides alike by whichever of the two scales is nearer one when `keep_aspect_ratio` is set, each side rounded to a multiple of `multiple`, as DPT's processor sizes images.
    Fitted {
        height: i64,
        width: i64,
        keep_aspect_ratio: bool,
        multiple: i64,
    },
}

// How an image grows at its bottom and right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Padding {
    // Past the next multiple of `multiple`, by a whole multiple when the side is one already, mirroring the edge before rescaling, as Swin2SR's processor pads.
    Symmetric { multiple: i64 },
    // To `height × width` with zeros after normalizing, as SAM's processor pads.
    Zeros { height: i64, width: i64 },
    // To `height × width` with black around the image, centered, before rescaling, as Donut's processor pads.
    Centered { height: i64, width: i64 },
}

// Per-channel normalization after rescaling: `(x - mean) / std`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Normalization {
    pub mean: [f64; 3],
    pub std: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImageTransformsConfig {
    pub resize: Option<Resize>,
    pub resample: Resample,
    // The center crop's height and width.
    pub crop: Option<[i64; 2]>,
    // What pixel values in [0, 255] are multiplied by, such as `1 / 255`.
    pub rescale: Option<f64>,
    pub pad: Option<Padding>,
    // Whether a color image becomes gray after rescaling, in each of its three channels, by the NTSC weights, as LightGlue's processor converts it.
    pub grayscale: bool,
    pub normalization: Option<Normalization>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageTransforms {
    config: ImageTransformsConfig,
}

impl ImageTransforms {
    const fn new(config: ImageTransformsConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub const fn config(&self) -> &ImageTransformsConfig {
        &self.config
    }

    // Returns the images as one batch, `[B, 3, H, W]` of `data_type` on `device`, each resized, cropped, rescaled, and normalized; they must end at one size.
    pub fn preprocess(
        &self,
        images: &[Image],
        device: Device,
        data_type: DataType,
    ) -> Result<Tensor> {
        let images = images
            .iter()
            .map(|image| self.one(image, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::concat(&images, 0)?.cast(data_type)?)
    }

    fn one(&self, image: &Image, device: Device) -> Result<Tensor> {
        let config = self.config;
        let (height, width) = (i64::from(image.height), i64::from(image.width));
        let resized;
        let image = match config.resize {
            Some(resize) => {
                let (height, width) = match resize {
                    Resize::Exact { height, width } => (height, width),
                    Resize::ShortestEdge(edge) => shortest_edge(height, width, edge),
                    Resize::Bounded {
                        shortest,
                        longest,
                        multiple,
                    } => bounded(height, width, shortest, longest, multiple),
                    Resize::Longest(edge) => longest_edge(height, width, edge),
                    Resize::Thumbnail {
                        height: most_height,
                        width: most_width,
                    } => shortest_edge(height, width, most_height.min(most_width)),
                    Resize::Fitted {
                        height: target_height,
                        width: target_width,
                        keep_aspect_ratio,
                        multiple,
                    } => fitted(
                        (height, width),
                        (target_height, target_width),
                        keep_aspect_ratio,
                        multiple,
                    ),
                };
                let side = |side: i64| {
                    u32::try_from(side)
                        .map_err(|_| Error::Media(format!("an image side of {side}")))
                };
                let first = image.resized(side(height)?, side(width)?, config.resample);
                resized = match resize {
                    Resize::Thumbnail {
                        height: most_height,
                        width: most_width,
                    } => match thumbnail(
                        (i64::from(first.height), i64::from(first.width)),
                        (most_height, most_width),
                    ) {
                        Some((height, width)) => {
                            first.resized(side(height)?, side(width)?, config.resample)
                        }
                        None => first,
                    },
                    _ => first,
                };
                &resized
            }
            None => image,
        };
        let padded;
        let image = match config.pad {
            Some(Padding::Symmetric { multiple }) => {
                padded = image.padded_symmetric(multiple)?;
                &padded
            }
            Some(Padding::Centered { height, width }) => {
                padded = image.padded_centered(height, width)?;
                &padded
            }
            _ => image,
        };
        let mut tensor = image.to_tensor(device)?;

        if let Some([crop_height, crop_width]) = config.crop {
            let dimensions = tensor.shape().dimensions();
            let (height, width) = (dimensions[2], dimensions[3]);
            if crop_height > height || crop_width > width {
                return Err(Error::Media(format!(
                    "a {crop_height} × {crop_width} crop of a {height} × {width} image"
                )));
            }

            tensor = tensor
                .narrow(2, (height - crop_height) / 2, crop_height)?
                .narrow(3, (width - crop_width) / 2, crop_width)?;
        }

        if let Some(scale) = config.rescale {
            tensor = tensor.mul_scalar(scale)?;
        }

        // An image whose channels are already equal stays as it is.
        let gray = image
            .pixels
            .as_chunks::<3>()
            .0
            .iter()
            .all(|pixel| pixel[0] == pixel[1] && pixel[1] == pixel[2]);
        if config.grayscale && !gray {
            let channel = |index: i64, weight: f64| tensor.narrow(1, index, 1)?.mul_scalar(weight);
            let luma = channel(0, 0.2989)?
                .add(&channel(1, 0.587)?)?
                .add(&channel(2, 0.114)?)?;
            tensor = Tensor::concat(&[luma.clone(), luma.clone(), luma], 1)?;
        }

        if let Some(normalization) = config.normalization {
            let channels = |values: [f64; 3]| -> Result<Tensor> {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "normalization constants are small values"
                )]
                let values = values.map(|value| value as f32).to_vec();
                Ok(Tensor::from_vec(
                    device,
                    Shape::create([1, 3, 1, 1])?,
                    values,
                )?)
            };
            tensor = tensor
                .sub(&channels(normalization.mean)?)?
                .div(&channels(normalization.std)?)?;
        }

        if let Some(Padding::Zeros {
            height: padded_height,
            width: padded_width,
        }) = config.pad
        {
            let dimensions = tensor.shape().dimensions();
            let (height, width) = (dimensions[2], dimensions[3]);
            if height > padded_height || width > padded_width {
                return Err(Error::Media(format!(
                    "a {height} × {width} image padded to {padded_height} × {padded_width}"
                )));
            }

            tensor = tensor.pad(
                &[
                    (0, 0),
                    (0, 0),
                    (0, padded_height - height),
                    (0, padded_width - width),
                ],
                0.0,
            )?;
        }

        Ok(tensor.contiguous()?)
    }
}

// The size `transformers` gives an image whose shorter side becomes `edge`: the longer side scaled to match, truncated.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "image sides are far below 2^53"
)]
fn shortest_edge(height: i64, width: i64, edge: i64) -> (i64, i64) {
    let (short, long) = (height.min(width), height.max(width));
    let long = (edge as f64 * long as f64 / short as f64) as i64;
    if height <= width {
        (edge, long)
    } else {
        (long, edge)
    }
}

// The size Donut's `thumbnail` shrinks an image to so it fits within `most`, or `None` when it fits already: the sides capped, and the other side of the longer one scaled to keep the aspect ratio, truncated.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "image sides are far below 2^53"
)]
fn thumbnail(
    (height, width): (i64, i64),
    (most_height, most_width): (i64, i64),
) -> Option<(i64, i64)> {
    let (mut fitted_height, mut fitted_width) = (height.min(most_height), width.min(most_width));
    if fitted_height == height && fitted_width == width {
        return None;
    }

    if height > width {
        fitted_width = (width as f64 * fitted_height as f64 / height as f64) as i64;
    } else if width > height {
        fitted_height = (height as f64 * fitted_width as f64 / width as f64) as i64;
    }
    Some((fitted_height, fitted_width))
}

// The size SAM's processor gives an image whose longer side becomes `edge`, as `_get_preprocess_shape` rounds it.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "image sides are far below 2^53"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "rounds each product as the reference does, which the verifications compare with"
)]
pub(crate) fn longest_edge(height: i64, width: i64, edge: i64) -> (i64, i64) {
    let scale = edge as f64 / height.max(width) as f64;
    let side = |side: i64| (side as f64 * scale + 0.5) as i64;
    (side(height), side(width))
}

// The size DPT's processor gives an image toward `target`, as `get_resize_output_image_size` computes it, with Python's rounding of halves to even.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "image sides are far below 2^53"
)]
fn fitted(
    (height, width): (i64, i64),
    (target_height, target_width): (i64, i64),
    keep_aspect_ratio: bool,
    multiple: i64,
) -> (i64, i64) {
    let mut scale_height = target_height as f64 / height as f64;
    let mut scale_width = target_width as f64 / width as f64;
    if keep_aspect_ratio {
        // The scale nearer one changes the image least.
        if (1.0 - scale_width).abs() < (1.0 - scale_height).abs() {
            scale_height = scale_width;
        } else {
            scale_width = scale_height;
        }
    }

    let multiple = multiple as f64;
    let constrain = |value: f64| ((value / multiple).round_ties_even() * multiple) as i64;
    (
        constrain(scale_height * height as f64),
        constrain(scale_width * width as f64),
    )
}

// The size `transformers`' DETR-style processors give an image: its shorter side `shortest`, unless the longer side would pass `longest`, which then bounds it, both rounded down to a multiple of `multiple`, as `get_size_with_aspect_ratio_yolos` computes them.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "image sides are far below 2^53"
)]
fn bounded(height: i64, width: i64, shortest: i64, longest: i64, multiple: i64) -> (i64, i64) {
    let (short, long) = (height.min(width) as f64, height.max(width) as f64);
    let mut size = shortest;
    let mut raw = None;
    if long / short * shortest as f64 > longest as f64 {
        let bounded = longest as f64 * short / long;
        raw = Some(bounded);
        size = bounded.round_ties_even() as i64;
    }

    let scaled = |side: i64, other: i64| {
        raw.map_or_else(
            || (size as f64 * side as f64 / other as f64) as i64,
            |raw| (raw * side as f64 / other as f64) as i64,
        )
    };
    let (out_height, out_width) = if width < height {
        (scaled(height, width), size)
    } else if (height <= width && height == size) || (width <= height && width == size) {
        (height, width)
    } else {
        (size, scaled(width, height))
    };
    (
        out_height - out_height % multiple,
        out_width - out_width % multiple,
    )
}

impl ImageTransforms {
    // Reads a snapshot's `preprocessor_config.json`, as `transformers`' default image processors read it.
    pub fn load(source: &Source) -> Result<Self> {
        let fields: Value = source.json("preprocessor_config.json")?;
        let flag = |name: &str, default: bool| fields[name].as_bool().unwrap_or(default);
        let sides = |value: &Value| -> Option<[i64; 2]> {
            value
                .as_i64()
                .map(|side| [side, side])
                .or_else(|| Some([value["height"].as_i64()?, value["width"].as_i64()?]))
                .or_else(|| Some([value[0].as_i64()?, value[1].as_i64()?]))
        };
        // A processor's class, which decides what an older config's bare `size` means.
        let class = fields["image_processor_type"]
            .as_str()
            .or_else(|| fields["feature_extractor_type"].as_str())
            .unwrap_or_default();
        let shortest_edge_class = ["CLIP", "VideoMAE", "Bit", "ConvNext", "MobileNet"]
            .iter()
            .any(|prefix| class.starts_with(prefix));
        // A config without a size leaves resizing to its class, whose default this takes to be off, as Swin2SR's is.
        let resize = if !fields["size"].is_null() && flag("do_resize", true) {
            let size = &fields["size"];
            // A bare number is the shortest edge for processors that keep the aspect ratio, as `get_size_dict(size, default_to_square=False)` reads it, and a square side otherwise.
            let shortest = size["shortest_edge"]
                .as_i64()
                .or_else(|| size.as_i64().filter(|_| shortest_edge_class));
            Some(match shortest {
                None if size["longest_edge"].is_i64() && size["height"].is_null() => {
                    Resize::Longest(size["longest_edge"].as_i64().unwrap_or_default())
                }
                Some(edge) if size["longest_edge"].is_null() => Resize::ShortestEdge(edge),
                // YOLOS rounds its sides down to whole patches; DETR keeps them.
                Some(shortest) => Resize::Bounded {
                    shortest,
                    longest: size["longest_edge"].as_i64().unwrap_or(i64::MAX),
                    multiple: if class.starts_with("Yolos") { 16 } else { 1 },
                },
                None => {
                    let [height, width] = sides(size)
                        .ok_or_else(|| Error::unsupported("an image size of this form"))?;
                    let keep_aspect_ratio = flag("keep_aspect_ratio", false);
                    let multiple = fields["ensure_multiple_of"].as_i64().unwrap_or(1);
                    if keep_aspect_ratio || multiple > 1 {
                        Resize::Fitted {
                            height,
                            width,
                            keep_aspect_ratio,
                            multiple,
                        }
                    } else {
                        Resize::Exact { height, width }
                    }
                }
            })
        } else {
            None
        };
        // Donut's processor reads a list size as `[width, height]`, resizes the shorter side, thumbnails, and pads to the size, centered.
        let donut = if class == "DonutImageProcessor" {
            if flag("do_align_long_axis", false) {
                return Err(Error::unsupported("Donut's long-axis alignment"));
            }

            let size = &fields["size"];
            let [height, width] = match size.as_array().map(Vec::as_slice) {
                Some([width, height]) => [height.as_i64(), width.as_i64()],
                _ => [size["height"].as_i64(), size["width"].as_i64()],
            };
            let (height, width) = height
                .zip(width)
                .ok_or_else(|| Error::unsupported("a Donut size of this form"))?;
            Some((height, width))
        } else {
            None
        };
        let resize = match donut {
            Some((height, width)) => {
                flag("do_resize", true).then_some(Resize::Thumbnail { height, width })
            }
            None => resize,
        };
        // PIL's filter numbers; torchvision on a GPU resizes Lanczos bicubically.
        let resample = match fields["resample"].as_i64().unwrap_or(2) {
            0 => Resample::Nearest,
            1 => Resample::Lanczos,
            2 => Resample::Bilinear,
            3 => Resample::Bicubic,
            other => {
                return Err(Error::UnsupportedConfig(format!(
                    "resampling filter {other}"
                )));
            }
        };
        let crop = if flag("do_center_crop", false) {
            // Processors that crop default to 224 pixels square when the config leaves the size out.
            Some(if fields["crop_size"].is_null() {
                [224, 224]
            } else {
                sides(&fields["crop_size"])
                    .ok_or_else(|| Error::unsupported("a crop of this form"))?
            })
        } else {
            None
        };
        let rescale = flag("do_rescale", true)
            .then(|| fields["rescale_factor"].as_f64().unwrap_or(1.0 / 255.0));
        let channels = |name: &str| -> Result<[f64; 3]> {
            let values = fields[name].as_array().map_or_else(Vec::new, |values| {
                values.iter().filter_map(Value::as_f64).collect::<Vec<_>>()
            });
            values.try_into().map_err(|_| {
                Error::UnsupportedConfig(format!("an {name} of other than three channels"))
            })
        };
        // A config naming neither leaves normalizing to its class, whose default this takes to be off, as SuperPoint's is.
        let normalizes = fields["image_mean"].is_array() || fields["do_normalize"].is_boolean();
        let normalization = if normalizes && flag("do_normalize", true) {
            Some(Normalization {
                mean: channels("image_mean")?,
                std: channels("image_std")?,
            })
        } else {
            None
        };
        Ok(Self::new(ImageTransformsConfig {
            resize,
            resample,
            crop,
            rescale,
            pad: match class {
                "DonutImageProcessor" if flag("do_pad", true) => {
                    donut.map(|(height, width)| Padding::Centered { height, width })
                }
                "Swin2SRImageProcessor" if flag("do_pad", true) => Some(Padding::Symmetric {
                    multiple: fields["size_divisor"]
                        .as_i64()
                        .or_else(|| fields["pad_size"].as_i64())
                        .unwrap_or(8),
                }),
                "SamImageProcessor" if flag("do_pad", true) => {
                    let [height, width] = sides(&fields["pad_size"])
                        .ok_or_else(|| Error::unsupported("a SAM padding of this form"))?;
                    Some(Padding::Zeros { height, width })
                }
                _ => None,
            },
            grayscale: flag("do_grayscale", false),
            normalization,
        }))
    }
}

impl Image {
    // Returns the image in the middle of a black `height × width` image, its odd pixel of margin at the bottom and right, as Donut's processor pads it.
    fn padded_centered(&self, height: i64, width: i64) -> Result<Self> {
        let side = |side: i64| u32::try_from(side).map_err(|_| Error::unsupported("this padding"));
        let (height, width) = (side(height)?, side(width)?);
        if height < self.height || width < self.width {
            return Err(Error::Media(format!(
                "a {} × {} image padded to {height} × {width}",
                self.height, self.width
            )));
        }

        let (top, left) = ((height - self.height) / 2, (width - self.width) / 2);
        let mut pixels = vec![0; width as usize * height as usize * 3];
        for row in 0..self.height as usize {
            let from = row * self.width as usize * 3;
            let to = ((row + top as usize) * width as usize + left as usize) * 3;
            pixels[to..to + self.width as usize * 3]
                .copy_from_slice(&self.pixels[from..from + self.width as usize * 3]);
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    // Returns the image grown at the bottom and right to the next multiple of `multiple` past its sides, by rows and columns mirrored from its edge, the edge included, as torchvision's symmetric padding fills them.
    fn padded_symmetric(&self, multiple: i64) -> Result<Self> {
        let multiple = u32::try_from(multiple).map_err(|_| Error::unsupported("this padding"))?;
        let grown = |side: u32| (side / multiple + 1) * multiple;
        let (width, height) = (grown(self.width), grown(self.height));
        // Mirrors an index past `side`, as `side - 1 - (index - side)` for one pass of reflection.
        let source = |index: u32, side: u32| {
            if index < side {
                index
            } else {
                (2 * side - 1).saturating_sub(index)
            }
        };
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for row in 0..height {
            let row = source(row, self.height) as usize;
            for column in 0..width {
                let column = source(column, self.width) as usize;
                let at = (row * self.width as usize + column) * 3;
                pixels.extend_from_slice(&self.pixels[at..at + 3]);
            }
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    // Writes the image to `path`, whose extension picks the format.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        self.buffer()?
            .save(path)
            .map_err(|error| Error::Media(format!("{}: {error}", path.display())))
    }

    fn buffer(&self) -> Result<RgbImage> {
        RgbImage::from_raw(self.width, self.height, self.pixels.clone())
            .ok_or_else(|| Error::Media("pixels that do not fill the image".to_owned()))
    }

    // Returns the image as a PNG `data:` URL.
    fn to_data_url(&self) -> Result<String> {
        let mut png = Cursor::new(Vec::new());
        self.buffer()?
            .write_to(&mut png, ImageFormat::Png)
            .map_err(|error| Error::Media(error.to_string()))?;
        Ok(format!("data:image/png;base64,{}", base64(png.get_ref())))
    }
}

// An image in JSON output is a PNG `data:` URL.
impl Serialize for Image {
    fn serialize<S: Serializer>(&self, serializer: S) -> StdResult<S::Ok, S::Error> {
        let url = self.to_data_url().map_err(SerError::custom)?;
        serializer.serialize_str(&url)
    }
}

// Standard base64 with padding.
pub(crate) fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let word = chunk
            .iter()
            .enumerate()
            .fold(0_u32, |word, (index, &byte)| {
                word | u32::from(byte) << (16 - 8 * index)
            });
        for index in 0..4 {
            if index <= chunk.len() {
                let sextet = (word >> (18 - 6 * index)) & 0x3f;
                text.push(char::from(ALPHABET[sextet as usize]));
            } else {
                text.push('=');
            }
        }
    }
    text
}
