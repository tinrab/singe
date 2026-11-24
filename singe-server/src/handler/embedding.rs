//! `POST /v1/embeddings`: texts through a sentence embedder, its pooled and normalized vectors.

use axum::{Json, extract::State};
use singe_model::{
    processor::base64,
    task::{Task, text::sentence_similarity::SentenceSimilarityPipeline},
};

use crate::{
    api::{
        embedding::{
            Embedding, EmbeddingInput, EmbeddingList, EmbeddingRequest, EmbeddingUsage,
            EmbeddingVector, EncodingFormat,
        },
        warn_unknown,
    },
    error::{ApiError, ApiResult},
    handler::{ApiJson, downcast},
    registry::Served,
};

const ENDPOINT: &str = "/v1/embeddings";

/// Serves `POST /v1/embeddings`: embeds each text of `input` with a sentence embedder, as floats or base64 little-endian `f32`s; `dimensions` truncates each vector and normalizes it again.
///
/// # Errors
///
/// Returns a 404 error for an unknown model, 422 for a model of another task, 503 while it loads, 429 when its queue is full, and 400 for token inputs, an empty input, or zero `dimensions`.
pub async fn create(
    State(served): State<Served>,
    ApiJson(request): ApiJson<EmbeddingRequest>,
) -> ApiResult<Json<EmbeddingList>> {
    warn_unknown(ENDPOINT, &request.unknown);
    let model = served.get(&request.model)?;
    let worker = model.pipeline(&[Task::SentenceSimilarity], ENDPOINT)?;
    let texts = match request.input {
        EmbeddingInput::Text(text) => vec![text],
        EmbeddingInput::Texts(texts) => texts,
        EmbeddingInput::Tokens(_) | EmbeddingInput::TokenLists(_) => {
            return Err(ApiError::invalid_field(
                "input",
                "token inputs, where an embedder takes texts",
            ));
        }
    };
    if texts.is_empty() || texts.iter().any(String::is_empty) {
        return Err(ApiError::invalid_field("input", "an empty input"));
    }

    if request.dimensions == Some(0) {
        return Err(ApiError::invalid_field("dimensions", "zero dimensions"));
    }

    let prompt_tokens = match &model.tokenizer {
        Some(tokenizer) => texts
            .iter()
            .map(|text| tokenizer.encode(text, true).map(|tokens| tokens.len()))
            .sum::<singe_model::Result<usize>>()?,
        None => 0,
    };
    let (values, width) = worker
        .call(move |pipeline| {
            let pipeline = downcast::<SentenceSimilarityPipeline>(&**pipeline)?;
            let texts = texts.iter().map(String::as_str).collect::<Vec<_>>();
            let vectors = pipeline.embed(&texts, None)?;
            let width = vectors
                .shape()
                .dimensions()
                .get(1)
                .and_then(|&width| usize::try_from(width).ok())
                .ok_or_else(|| ApiError::internal("embeddings that are not [B, W]"))?;
            Ok((vectors.to_vec::<f32>()?, width))
        })
        .await?;

    let format = request.encoding_format.unwrap_or_default();
    let data = values
        .chunks(width.max(1))
        .enumerate()
        .map(|(index, vector)| {
            let vector = match request.dimensions {
                Some(dimensions) if dimensions < vector.len() => normalized(&vector[..dimensions]),
                _ => vector.to_vec(),
            };
            let embedding = match format {
                EncodingFormat::Float => EmbeddingVector::Float(vector),
                EncodingFormat::Base64 => {
                    let bytes = vector
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>();
                    EmbeddingVector::Base64(base64::encode(&bytes))
                }
            };
            Embedding {
                object: "embedding",
                index,
                embedding,
            }
        })
        .collect();
    Ok(Json(EmbeddingList {
        object: "list",
        data,
        model: model.name.clone(),
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
}

/// `vector` scaled to unit length, as a truncated embedding is.
fn normalized(vector: &[f32]) -> Vec<f32> {
    let length = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    vector
        .iter()
        .map(|value| value / length.max(f32::EPSILON))
        .collect()
}
