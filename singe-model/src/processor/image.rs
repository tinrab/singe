//! Images as models read them, and the image processor most vision models share: resize on the CPU, then center crop, rescale, and normalize on the device, as `transformers`' default image processors compute them with torchvision.
//! A model whose processor does more, such as Qwen3.5's patches of any size, keeps it with the model, as `transformers` does.

use std::{borrow::Cow, io::Cursor, path::Path, result::Result as StdResult};

use image::{ImageFormat, RgbImage};
use serde::{
    Deserialize, Deserializer, Serialize, Serializer, de::Error as DeError, ser::Error as SerError,
};
use serde_json::Value;
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    shape::spatial::{self, Spatial2d, Spatial2dI64},
};

use crate::{
    error::{Error, Result},
    processor::{base64, resize},
    source::Source,
};

/// The shortest edge from which a `ConvNeXt` processor warps images to a square instead of cropping them.
const CONVNEXT_WARP: i64 = 384;

/// An RGB image, `height × width × 3` bytes in row-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelRange {
    Unit,
    Signed,
}

pub(crate) fn extent(value: i64) -> SingeResult<u32> {
    u32::try_from(value).map_err(|_| SingeError::out_of_range("image extent"))
}

impl Image {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let image = image::open(path).map_err(Error::image(path))?.to_rgb8();
        Ok(Self::from_rgb(image))
    }

    /// Decodes an image file's bytes, in any format the `image` crate reads, telling the format from its contents.
    ///
    /// # Errors
    ///
    /// Returns an error if `bytes` do not decode as an image.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let image = image::load_from_memory(bytes)
            .map_err(|error| Error::Media(format!("an image that does not decode: {error}")))?
            .to_rgb8();
        Ok(Self::from_rgb(image))
    }

    /// Decodes an image carried as JSON carries one: a `data:` URL or bare base64 of its file.
    ///
    /// # Errors
    ///
    /// Returns an error if `text` is neither a `data:` URL nor base64, or if its bytes do not decode as an image.
    pub fn from_data(text: &str) -> Result<Self> {
        let bytes = base64::data_bytes(text).ok_or_else(|| {
            Error::Media("an image that is neither a data URL nor base64".to_owned())
        })?;
        Self::decode(&bytes)
    }

    fn from_rgb(image: RgbImage) -> Self {
        Self {
            width: image.width(),
            height: image.height(),
            pixels: image.into_raw(),
        }
    }

    /// Encodes the image as a file of `format`, such as PNG, JPEG, or WebP.
    ///
    /// # Errors
    ///
    /// Returns an error if the pixels do not fill the image, or if the `image` crate cannot encode it as `format`.
    pub fn encode(&self, format: ImageFormat) -> Result<Vec<u8>> {
        let mut file = Cursor::new(Vec::new());
        self.buffer()?
            .write_to(&mut file, format)
            .map_err(Error::ImageEncoding)?;
        Ok(file.into_inner())
    }

    #[must_use]
    pub fn grayscale(width: u32, height: u32, values: impl IntoIterator<Item = u8>) -> Self {
        Self {
            width,
            height,
            pixels: values.into_iter().flat_map(|value| [value; 3]).collect(),
        }
    }

    pub fn from_tensor(tensor: &Tensor, range: PixelRange) -> Result<Vec<Self>> {
        let &[_, 3, height, width] = tensor.shape().dimensions() else {
            return Err(SingeError::invalid_argument("images, [B, 3, H, W]").into());
        };
        let unit = match range {
            PixelRange::Unit => tensor.clone(),
            PixelRange::Signed => tensor.div_scalar(2.0)?.add_scalar(0.5)?,
        };
        let values = unit
            .clamp(0.0, 1.0)?
            .mul_scalar(255.0)?
            .round()?
            .permute(&[0, 2, 3, 1])?
            .cast(DataType::F32)?
            .to_vec::<f32>()?;
        let (width, height) = (extent(width)?, extent(height)?);
        let size = width as usize * height as usize * 3;
        let images = values
            .chunks(size.max(1))
            .map(|pixels| Self {
                width,
                height,
                pixels: pixels.iter().map(|&value| value as u8).collect(),
            })
            .collect();
        Ok(images)
    }

    /// Returns the image as `[1, 3, H, W]` of `F32` values from 0 to 255 on `device`.
    pub fn to_tensor(&self, device: Device) -> Result<Tensor> {
        let values = self.pixels.iter().map(|&value| f32::from(value)).collect();
        let shape = Shape::create([i64::from(self.height), i64::from(self.width), 3])?;
        Ok(Tensor::from_vec(device, shape, values)?
            .permute(&[2, 0, 1])?
            .unsqueeze(0)?)
    }

    /// Returns the image resized to `height × width` with `resample`, as `transformers` resizes images through torchvision on the CPU.
    #[must_use]
    pub fn resized(&self, height: u32, width: u32, resample: Resample) -> Self {
        Self {
            width,
            height,
            pixels: resize::pixels(self, height, width, resample),
        }
    }

    fn padded(&self, padding: Padding) -> Result<Option<Self>> {
        match padding {
            Padding::Symmetric { multiple } => self.padded_symmetric(multiple).map(Some),
            Padding::Centered { height, width } => {
                self.padded_centered(height, width, [0; 3]).map(Some)
            }
            Padding::Square { fill } => {
                let side = i64::from(self.height.max(self.width));
                self.padded_centered(side, side, fill).map(Some)
            }
            Padding::Zeros { .. } => Ok(None),
        }
    }

    /// Returns the image in the middle of a `height × width` image of `fill`, its odd pixel of margin at the bottom and right, as Donut's and Janus's processors pad it.
    fn padded_centered(&self, height: i64, width: i64, fill: [u8; 3]) -> Result<Self> {
        let (height, width) = (extent(height)?, extent(width)?);
        if height < self.height || width < self.width {
            return Err(Error::Media(format!(
                "a {} × {} image padded to {height} × {width}",
                self.height, self.width
            )));
        }

        let (top, left) = ((height - self.height) / 2, (width - self.width) / 2);
        let mut pixels = fill.repeat(width as usize * height as usize);
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

    /// Returns the image grown at the bottom and right to the next multiple of `multiple` past its sides, by rows and columns mirrored from its edge, the edge included, as torchvision's symmetric padding fills them.
    fn padded_symmetric(&self, multiple: i64) -> Result<Self> {
        let multiple = extent(multiple)?;
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

    fn is_gray(&self) -> bool {
        self.pixels
            .as_chunks::<3>()
            .0
            .iter()
            .all(|pixel| pixel[0] == pixel[1] && pixel[1] == pixel[2])
    }

    /// Writes the image to `path`, whose extension picks the format.
    ///
    /// # Errors
    ///
    /// Returns an error if the pixels do not fill the image, or if the `image` crate cannot write it to `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        self.buffer()?.save(path).map_err(Error::image(path))
    }

    fn buffer(&self) -> Result<RgbImage> {
        RgbImage::from_raw(self.width, self.height, self.pixels.clone())
            .ok_or_else(|| Error::invalid("pixels that do not fill the image"))
    }

    /// Returns the image as a PNG `data:` URL.
    pub fn to_data_url(&self) -> Result<String> {
        Ok(base64::data_url(
            "image/png",
            &self.encode(ImageFormat::Png)?,
        ))
    }
}

/// An image in JSON is its file's bytes, as a `data:` URL or bare base64, so JSON from a client never names a file to read.
impl<'de> Deserialize<'de> for Image {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> StdResult<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::from_data(&text).map_err(DeError::custom)
    }
}

/// An image in JSON output is a PNG `data:` URL.
impl Serialize for Image {
    fn serialize<S: Serializer>(&self, serializer: S) -> StdResult<S::Ok, S::Error> {
        let url = self.to_data_url().map_err(SerError::custom)?;
        serializer.serialize_str(&url)
    }
}

/// The filter a resize uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resample {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos,
}

impl Resample {
    /// Returns the filter PIL's filter number `filter` names: 0 nearest, 1 Lanczos, 2 bilinear, and 3 bicubic.
    pub(crate) fn from_pil(filter: i64) -> Result<Self> {
        match filter {
            0 => Ok(Self::Nearest),
            1 => Ok(Self::Lanczos),
            2 => Ok(Self::Bilinear),
            3 => Ok(Self::Bicubic),
            other => Err(Error::unsupported(format!("resampling filter {other}"))),
        }
    }
}

/// The size a resize gives an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resize {
    /// The shorter side becomes this long, and the longer keeps the aspect ratio, truncated.
    ShortestEdge(i64),
    /// The shorter side becomes `shortest` unless the longer would pass `longest`, which then bounds it, and both sides round down to a multiple of `multiple`, as DETR-style processors size images.
    Bounded {
        shortest: i64,
        longest: i64,
        multiple: i64,
    },
    Exact {
        height: i64,
        width: i64,
    },
    /// The longer side becomes this long, and the shorter keeps the aspect ratio, rounded, as SAM's processor sizes images.
    Longest(i64),
    /// The longer side becomes `edge` long and each side keeps the aspect ratio, rounded half to even as Python's `round` rounds, and at least `least`, as Janus's processor sizes images.
    LongestEven {
        edge: i64,
        least: i64,
    },
    /// The shorter side becomes the shorter of `height` and `width`, then the image shrinks, keeping its aspect ratio, until it fits within `height × width`, as Donut's processor resizes and thumbnails images.
    Thumbnail {
        height: i64,
        width: i64,
    },
    /// Toward `height × width`, scaling both sides alike by whichever of the two scales is nearer one when `keep_aspect_ratio` is set, each side rounded to a multiple of `multiple`, as DPT's processor sizes images.
    Fitted {
        height: i64,
        width: i64,
        keep_aspect_ratio: bool,
        multiple: i64,
    },
    /// Both sides scaled alike, each rounded up to whole patches of `patch`, by the largest scale a binary search finds that keeps the patches at most `max_patches`, as `SigLIP` 2's `NaFlex` processor sizes images.
    Patches {
        patch: i64,
        max_patches: i64,
    },
}

impl Resize {
    #[must_use]
    pub fn size(self, image: Spatial2dI64) -> Spatial2dI64 {
        let Spatial2d { height, width } = image;
        match self {
            Self::Exact { height, width } => Spatial2d::new(height, width),
            Self::ShortestEdge(edge) => {
                let (short, long) = (height.min(width), height.max(width));
                let long = (edge as f64 * long as f64 / short as f64) as i64;
                if height <= width {
                    Spatial2d::new(edge, long)
                } else {
                    Spatial2d::new(long, edge)
                }
            }
            Self::Thumbnail {
                height: most_height,
                width: most_width,
            } => Self::ShortestEdge(most_height.min(most_width)).size(image),
            Self::Longest(edge) => {
                let scale = edge as f64 / height.max(width) as f64;
                image.map(|side| (side as f64 * scale + 0.5) as i64)
            }
            Self::LongestEven { edge, least } => {
                let scale = edge as f64 / height.max(width) as f64;
                image.map(|side| ((side as f64 * scale).round_ties_even() as i64).max(least))
            }
            Self::Patches { patch, max_patches } => {
                let scaled = |scale: f64, side: i64| {
                    let side = ((side as f64 * scale) / patch as f64).ceil() as i64 * patch;
                    side.max(patch)
                };
                // `get_image_size_for_max_num_patches`'s search, in the same floats.
                let epsilon = 1e-5;
                let (mut low, mut high) = (epsilon / 10.0, 100.0);
                loop {
                    if high - low < epsilon {
                        break;
                    }

                    let scale = f64::midpoint(low, high);
                    let patches = (scaled(scale, height) as f64 / patch as f64)
                        * (scaled(scale, width) as f64 / patch as f64);
                    if patches <= max_patches as f64 {
                        low = scale;
                    } else {
                        high = scale;
                    }
                }
                Spatial2d::new(scaled(low, height), scaled(low, width))
            }
            Self::Fitted {
                height: target_height,
                width: target_width,
                keep_aspect_ratio,
                multiple,
            } => {
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
                let constrain =
                    |value: f64| ((value / multiple).round_ties_even() * multiple) as i64;
                Spatial2d::new(
                    constrain(scale_height * height as f64),
                    constrain(scale_width * width as f64),
                )
            }
            Self::Bounded {
                shortest,
                longest,
                multiple,
            } => {
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
                } else if (height <= width && height == size) || (width <= height && width == size)
                {
                    (height, width)
                } else {
                    (size, scaled(width, height))
                };
                Spatial2d::new(
                    out_height - out_height % multiple,
                    out_width - out_width % multiple,
                )
            }
        }
    }

    /// Returns the size Donut's `thumbnail` shrinks an image to so it fits within a [`Resize::Thumbnail`]'s `height × width`: each side capped, then the shorter side scaled by as much as the longer to keep the aspect ratio, truncated.
    /// Returns `None` when the image fits already, or for another variant.
    fn thumbnail(self, Spatial2d { height, width }: Spatial2dI64) -> Option<Spatial2dI64> {
        let Self::Thumbnail {
            height: most_height,
            width: most_width,
        } = self
        else {
            return None;
        };
        let (mut fitted_height, mut fitted_width) =
            (height.min(most_height), width.min(most_width));
        if fitted_height == height && fitted_width == width {
            return None;
        }

        if height > width {
            fitted_width = (width as f64 * fitted_height as f64 / height as f64) as i64;
        } else if width > height {
            fitted_height = (height as f64 * fitted_width as f64 / width as f64) as i64;
        }
        Some(Spatial2d::new(fitted_height, fitted_width))
    }
}

/// How an image grows at its bottom and right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Padding {
    /// Past the next multiple of `multiple`, by a whole multiple when the side is one already, mirroring the edge before rescaling, as `Swin2SR`'s processor pads.
    Symmetric { multiple: i64 },
    /// To `height × width` with zeros after normalizing, as SAM's processor pads.
    Zeros { height: i64, width: i64 },
    /// To `height × width` with black around the image, centered, before rescaling, as Donut's processor pads.
    Centered { height: i64, width: i64 },
    /// To a square of the longer side with `fill` around the image, centered, before rescaling, as Janus's processor pads.
    Square { fill: [u8; 3] },
}

/// Per-channel normalization after rescaling: `(x - mean) / std`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Normalization {
    pub mean: [f64; 3],
    pub std: [f64; 3],
    /// Whether the result is divided by the std a second time, as `EfficientNet`'s processor does with `include_top`.
    #[serde(default)]
    pub repeats_std: bool,
}

impl Normalization {
    fn apply(&self, tensor: &Tensor, device: Device) -> Result<Tensor> {
        let channels = |values: [f64; 3]| -> Result<Tensor> {
            let values = values.map(|value| value as f32).to_vec();
            Ok(Tensor::from_vec(
                device,
                Shape::create([1, 3, 1, 1])?,
                values,
            )?)
        };
        let std = channels(self.std)?;
        let normalized = tensor.sub(&channels(self.mean)?)?.div(&std)?;
        Ok(if self.repeats_std {
            normalized.div(&std)?
        } else {
            normalized
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImagePreprocessor {
    pub resize: Option<Resize>,
    pub resample: Resample,
    /// The center crop's height and width.
    pub crop: Option<Spatial2dI64>,
    /// What pixel values from 0 to 255 are multiplied by, such as `1 / 255`.
    pub rescale: Option<f64>,
    pub pad: Option<Padding>,
    /// Whether a color image becomes gray after rescaling, in each of its three channels, by the NTSC weights, as `LightGlue`'s processor converts it.
    pub grayscale: bool,
    pub normalization: Option<Normalization>,
}

impl ImagePreprocessor {
    /// Reads a snapshot's `preprocessor_config.json`, as `transformers`' default image processors read it.
    ///
    /// # Errors
    ///
    /// Returns an error if `preprocessor_config.json` cannot be read or parsed, or if it asks for a size, crop, padding, normalization, or resampling filter these transforms do not support.
    pub fn load(source: &Source) -> Result<Self> {
        // A processor may keep its image processor's config inside its own, as SAM 3's does, and a video model may have only a video processor's config, whose frames these transforms process one by one.
        let nested = source
            .optional_json::<Value>("processor_config.json")?
            .and_then(|processor| processor.get("image_processor").cloned());
        let fields: Value = match source.optional_json("preprocessor_config.json")? {
            Some(fields) => fields,
            None => match nested {
                Some(fields) => fields,
                None => source.json("video_preprocessor_config.json")?,
            },
        };
        // A processor's class, which decides what an older config's bare `size` means.
        let class = fields["image_processor_type"]
            .as_str()
            .or_else(|| fields["feature_extractor_type"].as_str())
            .or_else(|| fields["video_processor_type"].as_str())
            .unwrap_or_default();
        let config = PreprocessorConfig {
            fields: &fields,
            class,
        };
        let donut = config.donut_size()?;
        let resize = match donut {
            Some(Spatial2d { height, width }) => config
                .flag("do_resize", true)
                .then_some(Resize::Thumbnail { height, width }),
            None => config.resize()?,
        };
        Ok(Self {
            resize,
            resample: Resample::from_pil(fields["resample"].as_i64().unwrap_or(2))?,
            crop: config.crop()?,
            rescale: config
                .flag("do_rescale", true)
                .then(|| fields["rescale_factor"].as_f64().unwrap_or(1.0 / 255.0)),
            pad: config.pad(donut)?,
            grayscale: config.flag("do_grayscale", false),
            normalization: config.normalization()?,
        })
    }

    /// Returns the image as `[1, 3, H, W]` of `data_type` on `device`, in this order resized, padded by mirroring or centering, center-cropped, rescaled, turned gray, normalized, and padded with zeros, as far as the transforms ask.
    ///
    /// # Errors
    ///
    /// Returns an error if the resized image is larger than a centered or zero padding's size, or smaller than the center crop.
    pub fn preprocess(&self, image: &Image, device: Device, data_type: DataType) -> Result<Tensor> {
        Ok(self.one(image, device)?.cast(data_type)?)
    }

    /// Returns the images as one batch, `[B, 3, H, W]` of `data_type` on `device`, each transformed as [`ImagePreprocessor::preprocess`] transforms one.
    ///
    /// # Errors
    ///
    /// Returns an error if [`ImagePreprocessor::preprocess`] would fail on an image, or if the transformed images differ in size.
    pub fn preprocess_batch(
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
        let resized = self.resized(image)?;
        let padded = match self.pad {
            Some(padding) => resized.padded(padding)?,
            None => None,
        };
        let image = padded.as_ref().unwrap_or(&*resized);
        let mut tensor = image.to_tensor(device)?;
        if let Some(crop) = self.crop {
            tensor = center_crop(&tensor, crop)?;
        }

        if let Some(scale) = self.rescale {
            tensor = tensor.mul_scalar(scale)?;
        }

        // An image whose channels are already equal stays as it is.
        if self.grayscale && !image.is_gray() {
            tensor = luma(&tensor)?;
        }

        if let Some(normalization) = self.normalization {
            tensor = normalization.apply(&tensor, device)?;
        }

        if let Some(Padding::Zeros { height, width }) = self.pad {
            tensor = zero_padded(&tensor, height, width)?;
        }
        Ok(tensor.contiguous()?)
    }

    fn resized<'a>(&self, image: &'a Image) -> Result<Cow<'a, Image>> {
        let Some(resize) = self.resize else {
            return Ok(Cow::Borrowed(image));
        };
        let size = resize.size(Spatial2d::new(
            i64::from(image.height),
            i64::from(image.width),
        ));
        let resized = image.resized(extent(size.height)?, extent(size.width)?, self.resample);
        let thumbnail = resize.thumbnail(Spatial2d::new(
            i64::from(resized.height),
            i64::from(resized.width),
        ));
        Ok(Cow::Owned(match thumbnail {
            Some(size) => resized.resized(extent(size.height)?, extent(size.width)?, self.resample),
            None => resized,
        }))
    }
}

fn center_crop(tensor: &Tensor, crop: Spatial2dI64) -> Result<Tensor> {
    let Spatial2d {
        height: crop_height,
        width: crop_width,
    } = crop;
    let &[_, _, height, width] = tensor.shape().dimensions() else {
        return Err(SingeError::invalid_argument("an image, [1, 3, H, W]").into());
    };
    if crop_height > height || crop_width > width {
        return Err(Error::Media(format!(
            "a {crop_height} × {crop_width} crop of a {height} × {width} image"
        )));
    }

    Ok(tensor
        .narrow(2, (height - crop_height) / 2, crop_height)?
        .narrow(3, (width - crop_width) / 2, crop_width)?)
}

fn luma(tensor: &Tensor) -> Result<Tensor> {
    let channel = |index: i64, weight: f64| tensor.narrow(1, index, 1)?.mul_scalar(weight);
    let luma = channel(0, 0.2989)?
        .add(&channel(1, 0.587)?)?
        .add(&channel(2, 0.114)?)?;
    Ok(Tensor::concat(&[luma.clone(), luma.clone(), luma], 1)?)
}

fn zero_padded(tensor: &Tensor, padded_height: i64, padded_width: i64) -> Result<Tensor> {
    let &[_, _, height, width] = tensor.shape().dimensions() else {
        return Err(SingeError::invalid_argument("an image, [1, 3, H, W]").into());
    };
    if height > padded_height || width > padded_width {
        return Err(Error::Media(format!(
            "a {height} × {width} image padded to {padded_height} × {padded_width}"
        )));
    }

    Ok(tensor.pad(
        &[
            spatial::Padding::ZERO,
            spatial::Padding::ZERO,
            spatial::Padding::new(0, padded_height - height),
            spatial::Padding::new(0, padded_width - width),
        ],
        0.0,
    )?)
}

/// A `preprocessor_config.json`'s fields, and the processor class that decides what an older config's bare `size` means.
struct PreprocessorConfig<'a> {
    fields: &'a Value,
    class: &'a str,
}

impl PreprocessorConfig<'_> {
    fn flag(&self, name: &str, default: bool) -> bool {
        self.fields[name].as_bool().unwrap_or(default)
    }

    fn resize(&self) -> Result<Option<Resize>> {
        let fields = self.fields;
        if self.class.starts_with("Siglip2ImageProcessor") && self.flag("do_resize", true) {
            return Ok(Some(Resize::Patches {
                patch: fields["patch_size"].as_i64().unwrap_or(16),
                max_patches: fields["max_num_patches"].as_i64().unwrap_or(256),
            }));
        }

        // A config without a size leaves resizing to its class, whose default this takes to be off, as Swin2SR's is.
        if fields["size"].is_null() || !self.flag("do_resize", true) {
            return Ok(None);
        }

        let size = &fields["size"];
        if self.class.starts_with("JanusImageProcessor") {
            let Spatial2d { height, width } = sides(size)
                .ok_or_else(|| Error::unsupported(format!("a Janus image size of {size}")))?;
            if height != width {
                return Err(Error::unsupported(format!(
                    "a Janus image size of {height} × {width}, which is not square"
                )));
            }
            return Ok(Some(Resize::LongestEven {
                edge: height,
                least: fields["min_size"].as_i64().unwrap_or(14),
            }));
        }

        let shortest_edge_class = ["CLIP", "VideoMAE", "Bit", "ConvNext", "MobileNet"]
            .iter()
            .any(|prefix| self.class.starts_with(prefix));
        // A bare number is the shortest edge for processors that keep the aspect ratio, as `get_size_dict(size, default_to_square=False)` reads it, and a square side otherwise.
        let shortest = size["shortest_edge"]
            .as_i64()
            .or_else(|| size.as_i64().filter(|_| shortest_edge_class));
        if let Some(edge) = self.convnext_edge() {
            // Below 384 the shortest edge grows by `crop_pct`, and `crop` takes the square back; from 384 the image is warped to a square.
            return Ok(Some(if edge < CONVNEXT_WARP {
                Resize::ShortestEdge((edge as f64 / self.convnext_crop_pct()) as i64)
            } else {
                Resize::Exact {
                    height: edge,
                    width: edge,
                }
            }));
        }
        Ok(Some(match shortest {
            None if size["longest_edge"].is_i64() && size["height"].is_null() => {
                Resize::Longest(size["longest_edge"].as_i64().unwrap_or_default())
            }
            Some(edge) if size["longest_edge"].is_null() => Resize::ShortestEdge(edge),
            // YOLOS rounds its sides down to whole patches; DETR keeps them.
            Some(shortest) => Resize::Bounded {
                shortest,
                longest: size["longest_edge"].as_i64().unwrap_or(i64::MAX),
                multiple: if self.class.starts_with("Yolos") {
                    16
                } else {
                    1
                },
            },
            None => {
                let Spatial2d { height, width } = sides(size)
                    .ok_or_else(|| Error::unsupported(format!("an image size of {size}")))?;
                let keep_aspect_ratio = self.flag("keep_aspect_ratio", false);
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
        }))
    }

    /// Returns the size a `DonutImageProcessor` config names, or `None` for another class.
    /// Donut's processor reads a list size as `[width, height]`, resizes the shorter side, thumbnails, and pads to the size, centered.
    fn donut_size(&self) -> Result<Option<Spatial2dI64>> {
        if self.class != "DonutImageProcessor" {
            return Ok(None);
        }

        if self.flag("do_align_long_axis", false) {
            return Err(Error::unsupported("Donut's long-axis alignment"));
        }

        let size = &self.fields["size"];
        let [height, width] = match size.as_array().map(Vec::as_slice) {
            Some([width, height]) => [height.as_i64(), width.as_i64()],
            _ => [size["height"].as_i64(), size["width"].as_i64()],
        };
        height
            .zip(width)
            .map(|(height, width)| Some(Spatial2d::new(height, width)))
            .ok_or_else(|| Error::unsupported(format!("a Donut size of {size}")))
    }

    /// Returns the shortest edge a `ConvNextImageProcessor` config resizes to, or `None` for another class.
    fn convnext_edge(&self) -> Option<i64> {
        if !self.class.starts_with("ConvNext") || !self.flag("do_resize", true) {
            return None;
        }

        // Older configs give the size as a bare shortest edge.
        let size = &self.fields["size"];
        size["shortest_edge"].as_i64().or_else(|| size.as_i64())
    }

    /// Returns the share of the resized image a `ConvNeXt` processor's crop keeps, `224 / 256` by default.
    fn convnext_crop_pct(&self) -> f64 {
        self.fields["crop_pct"].as_f64().unwrap_or(224.0 / 256.0)
    }

    fn crop(&self) -> Result<Option<Spatial2dI64>> {
        // A ConvNeXt processor crops a square of its shortest edge after resizing below 384, whatever `do_center_crop` says.
        if let Some(edge) = self.convnext_edge() {
            return Ok((edge < CONVNEXT_WARP).then(|| Spatial2d::square(edge)));
        }

        if !self.flag("do_center_crop", false) {
            return Ok(None);
        }

        // Processors that crop default to 224 pixels square when the config leaves the size out.
        if self.fields["crop_size"].is_null() {
            return Ok(Some(Spatial2d::square(224)));
        }

        let size = &self.fields["crop_size"];
        sides(size)
            .map(Some)
            .ok_or_else(|| Error::unsupported(format!("a crop size of {size}")))
    }

    fn normalization(&self) -> Result<Option<Normalization>> {
        let fields = self.fields;
        // A config naming neither leaves normalizing to its class, whose default this takes to be off, as SuperPoint's is.
        let normalizes = fields["image_mean"].is_array() || fields["do_normalize"].is_boolean();
        if !normalizes || !self.flag("do_normalize", true) {
            return Ok(None);
        }

        let channels = |name: &str| -> Result<[f64; 3]> {
            let values = fields[name].as_array().map_or_else(Vec::new, |values| {
                values.iter().filter_map(Value::as_f64).collect::<Vec<_>>()
            });
            values
                .try_into()
                .map_err(|_| Error::unsupported(format!("an {name} of other than three channels")))
        };
        Ok(Some(Normalization {
            mean: channels("image_mean")?,
            std: channels("image_std")?,
            repeats_std: self.class == "EfficientNetImageProcessor"
                && self.flag("include_top", true),
        }))
    }

    fn pad(&self, donut: Option<Spatial2dI64>) -> Result<Option<Padding>> {
        if !self.flag("do_pad", true) {
            return Ok(None);
        }

        Ok(match self.class {
            "DonutImageProcessor" => {
                donut.map(|Spatial2d { height, width }| Padding::Centered { height, width })
            }
            "Swin2SRImageProcessor" => Some(Padding::Symmetric {
                multiple: self.fields["size_divisor"]
                    .as_i64()
                    .or_else(|| self.fields["pad_size"].as_i64())
                    .unwrap_or(8),
            }),
            // `JanusImageProcessor.__init__` fills with the mean, ignoring a config's `background_color`.
            class if class.starts_with("JanusImageProcessor") => {
                let mut fill = [127; 3];
                if let Some(mean) = self.fields["image_mean"].as_array() {
                    for (channel, value) in fill.iter_mut().zip(mean) {
                        *channel = (value.as_f64().unwrap_or(0.5) * 255.0) as u8;
                    }
                }
                Some(Padding::Square { fill })
            }
            "SamImageProcessor" => {
                let size = &self.fields["pad_size"];
                let Spatial2d { height, width } = sides(size)
                    .ok_or_else(|| Error::unsupported(format!("a SAM pad size of {size}")))?;
                Some(Padding::Zeros { height, width })
            }
            _ => None,
        })
    }
}

fn sides(value: &Value) -> Option<Spatial2dI64> {
    value
        .as_i64()
        .map(Spatial2d::square)
        .or_else(|| {
            Some(Spatial2d::new(
                value["height"].as_i64()?,
                value["width"].as_i64()?,
            ))
        })
        .or_else(|| Some(Spatial2d::new(value[0].as_i64()?, value[1].as_i64()?)))
}
