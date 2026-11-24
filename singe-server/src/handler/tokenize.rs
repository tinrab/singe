//! `POST /tokenize` and `POST /detokenize`, with a model's own tokenizer and chat template.

use std::sync::Arc;

use axum::{Json, extract::State};

use crate::{
    api::tokenize::{DetokenizeRequest, DetokenizeResponse, TokenizeRequest, TokenizeResponse},
    error::{ApiError, ApiResult},
    handler::{ApiJson, blocking, chat::messages},
    registry::Served,
};

/// Serves `POST /tokenize`: the tokens of `prompt`, or of `messages` rendered through the model's chat template, with their count and the model's `max_model_len`.
///
/// # Errors
///
/// Returns a 404 error for an unknown model, 400 for a model without a tokenizer or a request with neither or both of `prompt` and `messages`, and 422 for `messages` to a model that does not generate text.
pub async fn tokenize(
    State(served): State<Served>,
    ApiJson(request): ApiJson<TokenizeRequest>,
) -> ApiResult<Json<TokenizeResponse>> {
    let model = served.get(&request.model)?;
    let tokenizer = Arc::clone(model.tokenizer()?);
    let tokens = match (request.prompt, request.messages) {
        (Some(prompt), None) => {
            let special_tokens = request.add_special_tokens.unwrap_or(true);
            blocking(move || Ok(tokenizer.encode(&prompt, special_tokens)?)).await?
        }
        (None, Some(conversation)) => {
            let (_, format) = model.text("/tokenize with messages")?;
            let format = Arc::clone(format);
            let conversation = messages(&conversation)?;
            let generation_prompt = request.add_generation_prompt.unwrap_or(true);
            let arguments = request.chat_template_kwargs;
            blocking(move || {
                let text = format.render(&conversation, generation_prompt, &arguments)?;
                format.encode(&text, false)
            })
            .await?
        }
        _ => {
            return Err(ApiError::invalid(
                "a request with either `prompt` or `messages`",
            ));
        }
    };
    Ok(Json(TokenizeResponse {
        count: tokens.len(),
        max_model_len: model.max_model_len(),
        tokens,
    }))
}

/// Serves `POST /detokenize`: the text of `tokens`, special tokens included.
///
/// # Errors
///
/// Returns a 404 error for an unknown model, and 400 for a model without a tokenizer.
pub async fn detokenize(
    State(served): State<Served>,
    ApiJson(request): ApiJson<DetokenizeRequest>,
) -> ApiResult<Json<DetokenizeResponse>> {
    let model = served.get(&request.model)?;
    let tokenizer = Arc::clone(model.tokenizer()?);
    let prompt = blocking(move || Ok(tokenizer.decode(&request.tokens, false)?)).await?;
    Ok(Json(DetokenizeResponse { prompt }))
}
