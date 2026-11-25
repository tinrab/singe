//! `POST /v1/images/generations`: a prompt through a diffusion model; `POST /v1/images/edits`: an uploaded image and an instruction through an image editor.
//! Images return as base64 in the format asked; the server keeps nothing, so it serves no URLs.

use axum::{
    Json,
    extract::{Multipart, State},
};
use serde_json::{Map, Value, json};
use singe::shape::spatial::Spatial2d;
use singe_model::{
    processor::{base64, image::Image},
    task::{
        self, Pipeline, Task,
        inputs::{SeededImagePrompt, SeededPrompt},
        multimodal::image_text_to_image::{ImageTextToImageParameters, ImageTextToImagePipeline},
        vision::text_to_image::{TextToImageParameters, TextToImagePipeline},
    },
};

use crate::{
    api::{
        image::{GeneratedImage, ImageGenerationRequest, ImageList, ImageOptions},
        now, warn_unknown,
    },
    error::{ApiError, ApiResult},
    handler::{ApiJson, blocking, downcast, fresh_seed},
    media::{Form, ImageOutput, decode_image, parse_size},
    registry::Served,
};

const GENERATIONS: &str = "/v1/images/generations";

const EDITS: &str = "/v1/images/edits";

/// The most images one request asks for.
const MAX_IMAGES: usize = 10;

/// The pixels an uploaded image holds at most.
const MAX_PIXELS: u64 = 4096 * 4096;

/// Serves `POST /v1/images/generations`: generates `n` images, 1 to 10, from `prompt` with a text-to-image model, each with the seed after the last's, as base64 PNG, JPEG, or WebP.
///
/// # Errors
///
/// Returns a 404 error for an unknown model, 422 for a model of another task, 503 while it loads, 429 when its queue is full, and 400 for an invalid `n`, `size`, `output_format`, or parameter, a `response_format` other than `b64_json`, or a `negative_prompt`, transparent `background`, or `style`, which it does not support.
pub async fn generate(
    State(served): State<Served>,
    ApiJson(request): ApiJson<ImageGenerationRequest>,
) -> ApiResult<Json<ImageList>> {
    warn_unknown(GENERATIONS, &request.unknown);
    let model = served.get(&request.model)?;
    let worker = model.pipeline(&[Task::TextToImage], GENERATIONS)?;
    let options = &request.options;
    let (count, output) = check(options)?;
    let mut parameters = denoising(options);
    if let Some(size) = parse_size(options.size.as_deref())? {
        parameters.insert("width".to_owned(), json!(size.width));
        parameters.insert("height".to_owned(), json!(size.height));
    }

    let parameters = task::parameters::<TextToImageParameters>(&Value::Object(parameters))?;
    let seed = options.seed.unwrap_or_else(fresh_seed);
    let prompt = request.prompt.clone();
    let images = worker
        .call(move |pipeline| {
            let pipeline = downcast::<TextToImagePipeline>(&**pipeline)?;
            (0..count)
                .map(|index| {
                    let input = SeededPrompt {
                        prompt: prompt.clone(),
                        seed: seed.wrapping_add(index as u64),
                    };
                    Ok(pipeline.run_one(&input, &parameters)?)
                })
                .collect::<ApiResult<Vec<_>>>()
        })
        .await?;
    respond(images, output, options.output_compression).await
}

/// Serves `POST /v1/images/edits`: edits the multipart form's `image` as `prompt` instructs with an image editor, `n` times, as [`generate`] returns images.
///
/// # Errors
///
/// Returns the errors [`generate`] does, and also a 400 error for a malformed form, a `mask`, or a `size` other than the image's, a 415 error for an image of an unknown or unsupported format, and a 413 error for one of more than 4096 × 4096 pixels.
pub async fn edit(
    State(served): State<Served>,
    multipart: Multipart,
) -> ApiResult<Json<ImageList>> {
    let form = Form::read(multipart).await?;
    let model = served.get(form.required_text("model")?)?;
    let worker = model.pipeline(&[Task::ImageTextToImage], EDITS)?;
    let options = ImageOptions {
        n: form.parsed("n")?,
        size: form.text("size").map(str::to_owned),
        response_format: form.text("response_format").map(str::to_owned),
        output_format: form.text("output_format").map(str::to_owned),
        output_compression: form.parsed("output_compression")?,
        quality: form.text("quality").map(str::to_owned),
        style: form.text("style").map(str::to_owned),
        background: form.text("background").map(str::to_owned),
        user: form.text("user").map(str::to_owned),
        seed: form.parsed("seed")?,
        num_inference_steps: form.parsed("num_inference_steps")?,
        guidance_scale: form.parsed("guidance_scale")?,
        image_guidance_scale: form.parsed("image_guidance_scale")?,
        negative_prompt: form.text("negative_prompt").map(str::to_owned),
    };
    let (count, output) = check(&options)?;
    if form.files.contains_key("mask") {
        return Err(ApiError::unsupported_field("mask"));
    }

    let prompt = form.required_text("prompt")?.to_owned();
    let file = form.required_file("image")?.to_vec();
    let image = blocking(move || decode_image(&file, MAX_PIXELS)).await?;
    if let Some(size) = parse_size(options.size.as_deref())?
        && Spatial2d::new(i64::from(image.height), i64::from(image.width)) != size
    {
        return Err(ApiError::invalid_field(
            "size",
            "a size other than the input image's, which an edit keeps",
        ));
    }

    let mut parameters = denoising(&options);
    if let Some(scale) = options.image_guidance_scale {
        parameters.insert("image_guidance_scale".to_owned(), json!(scale));
    }

    let parameters = task::parameters::<ImageTextToImageParameters>(&Value::Object(parameters))?;
    let seed = options.seed.unwrap_or_else(fresh_seed);
    let images = worker
        .call(move |pipeline| {
            let pipeline = downcast::<ImageTextToImagePipeline>(&**pipeline)?;
            (0..count)
                .map(|index| {
                    let input = SeededImagePrompt {
                        image: image.clone(),
                        prompt: prompt.clone(),
                        seed: seed.wrapping_add(index as u64),
                    };
                    Ok(pipeline.run_one(&input, &parameters)?)
                })
                .collect::<ApiResult<Vec<_>>>()
        })
        .await?;
    respond(images, output, options.output_compression).await
}

/// Checks what generating and editing share, and returns how many images to make and their format.
fn check(options: &ImageOptions) -> ApiResult<(usize, ImageOutput)> {
    let count = options.n.unwrap_or(1);
    if !(1..=MAX_IMAGES).contains(&count) {
        return Err(ApiError::invalid_field(
            "n",
            format!("a count outside 1 to {MAX_IMAGES}"),
        ));
    }

    match options.response_format.as_deref() {
        None | Some("b64_json") => {}
        Some(_) => {
            return Err(ApiError::invalid_field(
                "response_format",
                "a response format other than b64_json, since the server keeps no images to link",
            ));
        }
    }

    if options
        .negative_prompt
        .as_ref()
        .is_some_and(|prompt| !prompt.is_empty())
    {
        return Err(ApiError::unsupported_field("negative_prompt"));
    }

    if options
        .background
        .as_deref()
        .is_some_and(|background| background == "transparent")
    {
        return Err(ApiError::unsupported_field("background"));
    }

    if options.style.is_some() {
        return Err(ApiError::unsupported_field("style"));
    }
    Ok((count, ImageOutput::parse(options.output_format.as_deref())?))
}

fn denoising(options: &ImageOptions) -> Map<String, Value> {
    let mut parameters = Map::new();
    if let Some(steps) = options.num_inference_steps {
        parameters.insert("num_inference_steps".to_owned(), json!(steps));
    }

    if let Some(scale) = options.guidance_scale {
        parameters.insert("guidance_scale".to_owned(), json!(scale));
    }
    parameters
}

async fn respond(
    images: Vec<Image>,
    output: ImageOutput,
    compression: Option<u8>,
) -> ApiResult<Json<ImageList>> {
    let data = blocking(move || {
        images
            .iter()
            .map(|image| {
                Ok(GeneratedImage {
                    b64_json: base64::encode(&output.encode(image, compression)?),
                })
            })
            .collect::<ApiResult<Vec<_>>>()
    })
    .await?;
    Ok(Json(ImageList {
        created: now(),
        data,
        output_format: output.name(),
    }))
}
