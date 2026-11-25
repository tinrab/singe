//! Media in and out of requests: multipart forms, images encoded as clients ask, and transcripts as subtitles.

use std::{collections::HashMap, str::FromStr};

use axum::extract::Multipart;
use image::{
    ExtendedColorType, ImageEncoder, ImageFormat, RgbImage,
    codecs::{jpeg::JpegEncoder, png::PngEncoder, webp::WebPEncoder},
};
use singe::shape::spatial::{Spatial2d, Spatial2dI64};
use singe_model::processor::image::Image;

use crate::error::{ApiError, ApiResult};

/// A multipart form's text fields and files, by name, with array names' `[]` dropped.
#[derive(Debug, Default)]
pub struct Form {
    /// The text fields; of fields sharing a name, the first.
    pub fields: HashMap<String, String>,
    /// The fields sent with a file name, as bytes; of files sharing a name, the first.
    pub files: HashMap<String, Vec<u8>>,
}

impl Form {
    /// Reads every field of `multipart`.
    ///
    /// # Errors
    ///
    /// Returns a 400 error if the form is malformed or a text field is not UTF-8.
    pub async fn read(mut multipart: Multipart) -> ApiResult<Self> {
        let mut form = Self::default();
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| ApiError::invalid(format!("a malformed form: {error}")))?
        {
            let name = field
                .name()
                .unwrap_or_default()
                .trim_end_matches("[]")
                .to_owned();
            let file = field.file_name().is_some();
            let bytes = field
                .bytes()
                .await
                .map_err(|error| ApiError::invalid(format!("a malformed form: {error}")))?;
            if file {
                form.files.entry(name).or_insert_with(|| bytes.to_vec());
            } else {
                let text = String::from_utf8(bytes.to_vec()).map_err(|_| {
                    ApiError::invalid_field(&name, "a form field that is not UTF-8")
                })?;
                form.fields.entry(name).or_insert(text);
            }
        }
        Ok(form)
    }

    /// Returns the text field `name`, if the form has it.
    #[must_use]
    pub fn text(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }

    /// Returns the text field `name`.
    ///
    /// # Errors
    ///
    /// Returns a 400 error if the form does not have it.
    pub fn required_text(&self, name: &str) -> ApiResult<&str> {
        self.text(name)
            .ok_or_else(|| ApiError::invalid_field(name, format!("a form without `{name}`")))
    }

    /// Returns the file `name`.
    ///
    /// # Errors
    ///
    /// Returns a 400 error if the form does not have it.
    pub fn required_file(&self, name: &str) -> ApiResult<&[u8]> {
        self.files.get(name).map(Vec::as_slice).ok_or_else(|| {
            ApiError::invalid_field(name, format!("a form without the file `{name}`"))
        })
    }

    /// Returns the text field `name`, trimmed and parsed as `T`, if the form has it.
    ///
    /// # Errors
    ///
    /// Returns a 400 error if the field does not parse.
    pub fn parsed<T: FromStr>(&self, name: &str) -> ApiResult<Option<T>> {
        self.text(name)
            .map(|text| {
                text.trim()
                    .parse()
                    .map_err(|_| ApiError::invalid_field(name, format!("an invalid `{name}`")))
            })
            .transpose()
    }
}

/// The format a generated image is encoded in, as a request's `output_format` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageOutput {
    /// PNG, the default.
    #[default]
    Png,
    /// JPEG, whose quality follows `output_compression`.
    Jpeg,
    /// Lossless WebP.
    Webp,
}

impl ImageOutput {
    /// Parses `png`, `jpeg` or `jpg`, or `webp`; no format is PNG.
    ///
    /// # Errors
    ///
    /// Returns a 400 error for any other format.
    pub fn parse(format: Option<&str>) -> ApiResult<Self> {
        match format {
            None | Some("png") => Ok(Self::Png),
            Some("jpeg" | "jpg") => Ok(Self::Jpeg),
            Some("webp") => Ok(Self::Webp),
            Some(other) => Err(ApiError::invalid_field(
                "output_format",
                format!("an output format {other}, rather than png, jpeg, or webp"),
            )),
        }
    }

    /// Returns the format's name, as a response's `output_format` gives it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
        }
    }

    /// Encodes `image`; JPEG takes `compression`, from 0 to 100, as its quality's complement, as OpenAI's `output_compression` reads.
    ///
    /// # Errors
    ///
    /// Returns a 500 error if encoding fails.
    pub fn encode(self, image: &Image, compression: Option<u8>) -> ApiResult<Vec<u8>> {
        let mut bytes = Vec::new();
        let encoded = match self {
            Self::Png => PngEncoder::new(&mut bytes).write_image(
                &image.pixels,
                image.width,
                image.height,
                ExtendedColorType::Rgb8,
            ),
            Self::Jpeg => {
                let quality = 100 - compression.unwrap_or(0).min(99);
                JpegEncoder::new_with_quality(&mut bytes, quality).write_image(
                    &image.pixels,
                    image.width,
                    image.height,
                    ExtendedColorType::Rgb8,
                )
            }
            Self::Webp => WebPEncoder::new_lossless(&mut bytes).write_image(
                &image.pixels,
                image.width,
                image.height,
                ExtendedColorType::Rgb8,
            ),
        };
        encoded
            .map_err(|error| ApiError::internal(format!("failed to encode an image: {error}")))?;
        Ok(bytes)
    }
}

/// Decodes an uploaded PNG, JPEG, WebP, GIF, or BMP image as RGB, refusing one of more than `max_pixels`.
///
/// # Errors
///
/// Returns a 415 error for an unknown or other format, a 400 error if the image does not decode, and a 413 error if it has more than `max_pixels`.
pub fn decode_image(bytes: &[u8], max_pixels: u64) -> ApiResult<Image> {
    let format = image::guess_format(bytes)
        .map_err(|_| ApiError::unsupported_media("an image of an unknown format"))?;
    if !matches!(
        format,
        ImageFormat::Png
            | ImageFormat::Jpeg
            | ImageFormat::WebP
            | ImageFormat::Gif
            | ImageFormat::Bmp
    ) {
        return Err(ApiError::unsupported_media(format!(
            "an image in {format:?}"
        )));
    }

    let image = image::load_from_memory_with_format(bytes, format)
        .map_err(|error| ApiError::invalid(format!("an image that does not decode: {error}")))?;
    if u64::from(image.width()) * u64::from(image.height()) > max_pixels {
        return Err(ApiError::too_large(format!(
            "an image of more than {max_pixels} pixels"
        )));
    }

    let image: RgbImage = image.to_rgb8();
    Ok(Image {
        width: image.width(),
        height: image.height(),
        pixels: image.into_raw(),
    })
}

/// Parses a `WIDTHxHEIGHT` size, or `None` for `auto` or no size.
///
/// # Errors
///
/// Returns a 400 error if the size is malformed or a side is not a positive multiple of 8.
pub fn parse_size(size: Option<&str>) -> ApiResult<Option<Spatial2dI64>> {
    let Some(size) = size.filter(|size| *size != "auto") else {
        return Ok(None);
    };
    let invalid =
        || ApiError::invalid_field("size", format!("a size {size}, rather than WIDTHxHEIGHT"));
    let (width, height) = size.split_once('x').ok_or_else(invalid)?;
    let width = width.parse::<i64>().map_err(|_| invalid())?;
    let height = height.parse::<i64>().map_err(|_| invalid())?;
    if width <= 0 || height <= 0 || width % 8 != 0 || height % 8 != 0 {
        return Err(ApiError::invalid_field(
            "size",
            "a size whose sides are not positive multiples of 8",
        ));
    }
    Ok(Some(Spatial2d::new(height, width)))
}

/// Formats a transcript as one SubRip cue over the whole clip, `duration` seconds long.
#[must_use]
pub fn srt(text: &str, duration: f64) -> String {
    format!(
        "1\n{} --> {}\n{}\n",
        timestamp(0.0, ','),
        timestamp(duration, ','),
        text.trim()
    )
}

/// Formats a transcript as one WebVTT cue over the whole clip, `duration` seconds long.
#[must_use]
pub fn vtt(text: &str, duration: f64) -> String {
    format!(
        "WEBVTT\n\n{} --> {}\n{}\n",
        timestamp(0.0, '.'),
        timestamp(duration, '.'),
        text.trim()
    )
}

fn timestamp(seconds: f64, separator: char) -> String {
    let milliseconds = (seconds.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{separator}{:03}",
        milliseconds / 3_600_000,
        milliseconds / 60_000 % 60,
        milliseconds / 1000 % 60,
        milliseconds % 1000
    )
}
