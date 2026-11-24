//! `POST /v1/audio/transcriptions`: an uploaded clip through a speech recognizer; `POST /v1/audio/speech`: text through a speech synthesizer, as a WAV file or raw PCM.

use axum::{
    Json,
    extract::{Multipart, State},
    http::{HeaderName, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
};
use singe_model::{
    processor::audio::Audio,
    task::{
        Pipeline, Task,
        audio::{
            automatic_speech_recognition::{
                AutomaticSpeechRecognitionParameters, AutomaticSpeechRecognitionPipeline,
            },
            text_to_audio::{TextToAudioParameters, TextToAudioPipeline},
        },
    },
};

use crate::{
    api::{
        audio::{
            SpeechRequest, Transcription, TranscriptionFormat, TranscriptionSegment,
            VerboseTranscription,
        },
        warn_unknown,
    },
    error::{ApiError, ApiResult},
    handler::{ApiJson, blocking, downcast},
    media::{Form, srt, vtt},
    registry::Served,
};

const TRANSCRIPTIONS: &str = "/v1/audio/transcriptions";

const SPEECH: &str = "/v1/audio/speech";

/// Serves `POST /v1/audio/transcriptions`: transcribes the multipart form's `file` with a speech recognizer, as JSON, text, verbose JSON with one segment, SubRip, or WebVTT, as `response_format` asks.
///
/// # Errors
///
/// Returns a 404 error for an unknown model, 422 for a model of another task, 503 while it loads, 429 when its queue is full, and 400 for a malformed form, an unknown `response_format`, or a `prompt`, `temperature` above 0, or `timestamp_granularities` other than `segment`, which it does not support.
pub async fn transcribe(State(served): State<Served>, multipart: Multipart) -> ApiResult<Response> {
    let form = Form::read(multipart).await?;
    let model = served.get(form.required_text("model")?)?;
    let worker = model.pipeline(&[Task::AutomaticSpeechRecognition], TRANSCRIPTIONS)?;
    let format = form
        .text("response_format")
        .map(|format| {
            TranscriptionFormat::parse(format).ok_or_else(|| {
                ApiError::invalid_field(
                    "response_format",
                    format!("a response format {format}, rather than json, text, verbose_json, srt, or vtt"),
                )
            })
        })
        .transpose()?
        .unwrap_or_default();
    if form
        .text("prompt")
        .is_some_and(|prompt| !prompt.trim().is_empty())
    {
        return Err(ApiError::unsupported_field("prompt"));
    }

    if form
        .parsed::<f32>("temperature")?
        .is_some_and(|temperature| temperature > 0.0)
    {
        return Err(ApiError::unsupported_field("temperature"));
    }

    if form
        .text("timestamp_granularities")
        .is_some_and(|granularity| granularity != "segment")
    {
        return Err(ApiError::unsupported_field("timestamp_granularities"));
    }

    let mut parameters = AutomaticSpeechRecognitionParameters {
        language: form
            .text("language")
            .map(str::trim)
            .filter(|language| !language.is_empty())
            .map(str::to_owned),
        ..AutomaticSpeechRecognitionParameters::default()
    };
    if let Some(tokens) = form.parsed::<usize>("max_new_tokens")? {
        parameters.max_new_tokens = tokens;
    }

    let file = form.required_file("file")?.to_vec();
    let clip = blocking(move || Ok(Audio::decode(&file)?)).await?;
    let duration = clip.duration_seconds();
    let transcription = worker
        .call(move |pipeline| {
            Ok(downcast::<AutomaticSpeechRecognitionPipeline>(&**pipeline)?
                .run_one(&clip, &parameters)?)
        })
        .await?;
    let text = transcription.text.trim().to_owned();
    Ok(match format {
        TranscriptionFormat::Json => Json(Transcription { text }).into_response(),
        TranscriptionFormat::Text => text.into_response(),
        TranscriptionFormat::VerboseJson => Json(VerboseTranscription {
            task: "transcribe",
            language: transcription.language,
            duration,
            segments: vec![TranscriptionSegment {
                id: 0,
                start: 0.0,
                end: duration,
                text: text.clone(),
            }],
            text,
        })
        .into_response(),
        TranscriptionFormat::Srt => srt(&text, duration).into_response(),
        TranscriptionFormat::Vtt => {
            ([(CONTENT_TYPE, "text/vtt")], vtt(&text, duration)).into_response()
        }
    })
}

/// Serves `POST /v1/audio/speech`: speaks `input` with a speech synthesizer, as a WAV file, or as raw PCM with its rate in `x-sample-rate` when `response_format` is `pcm`.
///
/// # Errors
///
/// Returns a 404 error for an unknown model, 422 for a model of another task, 503 while it loads, 429 when its queue is full, and 400 for an empty `input`, a `response_format` other than `wav` or `pcm`, or a `speed` other than 1 or any `instructions`, which it does not support.
pub async fn speak(
    State(served): State<Served>,
    ApiJson(request): ApiJson<SpeechRequest>,
) -> ApiResult<Response> {
    warn_unknown(SPEECH, &request.unknown);
    let model = served.get(&request.model)?;
    let worker = model.pipeline(&[Task::TextToAudio], SPEECH)?;
    if request.input.trim().is_empty() {
        return Err(ApiError::invalid_field("input", "no text to speak"));
    }

    if request
        .speed
        .is_some_and(|speed| (speed - 1.0).abs() > f32::EPSILON)
    {
        return Err(ApiError::unsupported_field("speed"));
    }

    if request
        .instructions
        .as_ref()
        .is_some_and(|instructions| !instructions.trim().is_empty())
    {
        return Err(ApiError::unsupported_field("instructions"));
    }

    let pcm = match request.response_format.as_deref() {
        None | Some("wav") => false,
        Some("pcm") => true,
        Some(other) => {
            return Err(ApiError::invalid_field(
                "response_format",
                format!("a response format {other}, rather than wav or pcm"),
            ));
        }
    };

    let parameters = TextToAudioParameters {
        noise_scale: request.noise_scale,
        noise_scale_duration: request.noise_scale_duration,
        seed: request.seed.unwrap_or(0),
    };
    let text = request.input;
    let generated = worker
        .call(move |pipeline| {
            Ok(downcast::<TextToAudioPipeline>(&**pipeline)?.run_one(&text, &parameters)?)
        })
        .await?;
    let rate = u32::try_from(generated.sampling_rate)
        .map_err(|_| ApiError::internal("a sample rate out of range"))?;
    let clip = Audio {
        samples: generated.audio,
        rate,
    };
    Ok(if pcm {
        (
            [
                (CONTENT_TYPE, "audio/pcm".to_owned()),
                (HeaderName::from_static("x-sample-rate"), rate.to_string()),
            ],
            clip.to_pcm(),
        )
            .into_response()
    } else {
        ([(CONTENT_TYPE, "audio/wav")], clip.to_wav()).into_response()
    })
}
