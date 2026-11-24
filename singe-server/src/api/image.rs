//! `POST /v1/images/generations`, and `POST /v1/images/edits`, whose request is a multipart form.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A request to `POST /v1/images/generations`.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageGenerationRequest {
    /// The served model's name, as `/v1/models` lists it.
    pub model: String,
    /// The text describing the image to generate.
    pub prompt: String,
    /// The options generating shares with editing.
    #[serde(flatten)]
    pub options: ImageOptions,
    /// Fields the server does not read, which it logs and ignores.
    #[serde(flatten)]
    pub unknown: Map<String, Value>,
}

/// What generating and editing both read.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ImageOptions {
    /// How many images to make, from 1 to 10, and 1 by default.
    pub n: Option<usize>,
    /// `WIDTHxHEIGHT`, or `auto` for the model's own size; an edit refuses a size other than its input image's.
    pub size: Option<String>,
    /// Refused unless it is `b64_json`, since the server keeps no images to link.
    pub response_format: Option<String>,
    /// `png`, the default, `jpeg` (or `jpg`), or `webp`.
    pub output_format: Option<String>,
    /// From 0 to 100, the complement of a JPEG's quality; other formats ignore it.
    pub output_compression: Option<u8>,
    /// Accepted and ignored.
    pub quality: Option<String>,
    /// Refused when set.
    pub style: Option<String>,
    /// Refused when `transparent`, and otherwise ignored.
    pub background: Option<String>,
    /// Accepted and ignored.
    pub user: Option<String>,
    /// The first image's seed, which each further image adds 1 to; a fresh seed when unset.
    pub seed: Option<u64>,
    /// The denoising steps, or the pipeline's default when unset.
    pub num_inference_steps: Option<usize>,
    /// How closely the image follows the prompt, or the pipeline's default when unset.
    pub guidance_scale: Option<f32>,
    /// How closely an edit follows its input image; generating ignores it.
    pub image_guidance_scale: Option<f32>,
    /// Refused unless it is empty.
    pub negative_prompt: Option<String>,
}

/// The response to an image request.
#[derive(Debug, Clone, Serialize)]
pub struct ImageList {
    /// When the images were made, in seconds since the Unix epoch.
    pub created: u64,
    /// The images, in the order of their seeds.
    pub data: Vec<GeneratedImage>,
    /// The images' format: `png`, `jpeg`, or `webp`.
    pub output_format: &'static str,
}

/// One image of an [`ImageList`].
#[derive(Debug, Clone, Serialize)]
pub struct GeneratedImage {
    /// The encoded image, in base64.
    pub b64_json: String,
}
