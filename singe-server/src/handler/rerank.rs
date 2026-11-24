//! `POST /v1/rerank`: a query and documents through a cross-encoder, the documents by relevance, best first.

use axum::{Json, extract::State, http::HeaderMap};
use singe_model::task::{
    Pipeline, Task,
    text::text_ranking::{TextRankingInput, TextRankingParameters, TextRankingPipeline},
};

use crate::{
    api::{
        rerank::{RerankRequest, RerankResponse, RerankResult, RerankUsage, RerankedText},
        warn_unknown,
    },
    error::{ApiError, ApiResult},
    handler::{ApiJson, downcast, request_id},
    registry::Served,
};

const ENDPOINT: &str = "/v1/rerank";

/// Serves `POST /v1/rerank`: scores each of `documents` against `query` with a cross-encoder, and returns them best first, the first `top_n` when it is set.
///
/// # Errors
///
/// Returns a 404 error for an unknown model, 422 for a model of another task, 503 while it loads, 429 when its queue is full, and 400 for no documents or a `top_n` of zero.
pub async fn create(
    State(served): State<Served>,
    headers: HeaderMap,
    ApiJson(request): ApiJson<RerankRequest>,
) -> ApiResult<Json<RerankResponse>> {
    warn_unknown(ENDPOINT, &request.unknown);
    let model = served.get(&request.model)?;
    let worker = model.pipeline(&[Task::TextRanking], ENDPOINT)?;
    if request.documents.is_empty() {
        return Err(ApiError::invalid_field("documents", "no documents"));
    }

    if request.top_n == Some(0) {
        return Err(ApiError::invalid_field("top_n", "a top_n of zero"));
    }

    let texts = request
        .documents
        .iter()
        .map(|document| document.text().to_owned())
        .collect::<Vec<_>>();
    let total_tokens = match &model.tokenizer {
        Some(tokenizer) => {
            let query = tokenizer.encode(&request.query, true)?.len();
            texts
                .iter()
                .map(|text| {
                    tokenizer
                        .encode(text, false)
                        .map(|tokens| query + tokens.len())
                })
                .sum::<singe_model::Result<usize>>()?
        }
        None => 0,
    };
    let input = TextRankingInput {
        query: request.query.clone(),
        texts,
    };
    let parameters = TextRankingParameters {
        top_k: request.top_n,
        ..TextRankingParameters::default()
    };
    let ranked = worker
        .call(move |pipeline| {
            Ok(downcast::<TextRankingPipeline>(&**pipeline)?.run_one(&input, &parameters)?)
        })
        .await?;
    let results = ranked
        .into_iter()
        .map(|ranked| RerankResult {
            index: ranked.corpus_id,
            relevance_score: ranked.score,
            document: request.return_documents.then(|| RerankedText {
                text: request.documents[ranked.corpus_id].text().to_owned(),
            }),
        })
        .collect();
    Ok(Json(RerankResponse {
        id: format!("rerank-{}", request_id(&headers)),
        model: model.name.clone(),
        results,
        usage: RerankUsage { total_tokens },
    }))
}
