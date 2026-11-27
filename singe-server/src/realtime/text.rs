//! A text generator's session: generations by chat messages or a prompt, each streamed as it decodes, several at once, any of them cancellable.

use std::{collections::HashMap, sync::Arc};

use futures::future::BoxFuture;
use tokio::task::{AbortHandle, JoinSet};

use crate::{
    api::{
        Stop, Usage,
        realtime::{Generate, TextEvent, TextRequest},
    },
    error::{ApiError, ApiResult},
    generation::{
        engine::TextEngine,
        format::TextFormat,
        output::{Piece, TextGeneration},
    },
    handler::{blocking, chat, max_tokens},
    realtime::{Incoming, Outgoing, RealtimeSession},
    text::Detokenizer,
};

/// The session of a connection to a text generator.
pub struct TextSession {
    model: String,
    engine: TextEngine,
    format: Arc<TextFormat>,
    running: JoinSet<()>,
    /// The generations in progress, by id.
    generations: HashMap<String, AbortHandle>,
}

impl TextSession {
    /// Creates the session of a connection to the text generator `model`.
    #[must_use]
    pub fn new(model: String, engine: TextEngine, format: Arc<TextFormat>) -> Self {
        Self {
            model,
            engine,
            format,
            running: JoinSet::new(),
            generations: HashMap::new(),
        }
    }

    fn generate(&mut self, request: Box<Generate>, outgoing: &Outgoing) -> ApiResult<()> {
        while self.running.try_join_next().is_some() {}
        self.generations
            .retain(|_, generation| !generation.is_finished());
        if self.generations.contains_key(&request.id) {
            return Err(ApiError::invalid_field(
                "id",
                format!("a generation {} already in progress", request.id),
            ));
        }

        request.generation.check_supported()?;
        let id = request.id.clone();
        let (model, engine, format) = (
            self.model.clone(),
            self.engine.clone(),
            Arc::clone(&self.format),
        );
        let outgoing = outgoing.clone();
        let generation = self.running.spawn(async move {
            let id = request.id.clone();
            if let Err(error) = stream(*request, &model, &engine, format, &outgoing).await {
                outgoing.error(Some(&id), &error).await;
            }
        });
        self.generations.insert(id, generation);
        Ok(())
    }

    async fn cancel(&mut self, id: String, outgoing: &Outgoing) -> ApiResult<()> {
        let generation = self
            .generations
            .remove(&id)
            .filter(|generation| !generation.is_finished())
            .ok_or_else(|| {
                ApiError::invalid_field("id", format!("no generation {id} in progress"))
            })?;
        generation.abort();
        outgoing
            .json(&TextEvent::Done {
                id,
                finish_reason: "cancelled",
                usage: None,
            })
            .await;
        Ok(())
    }
}

impl RealtimeSession for TextSession {
    fn receive<'a>(
        &'a mut self,
        message: Incoming,
        outgoing: &'a Outgoing,
    ) -> BoxFuture<'a, ApiResult<()>> {
        Box::pin(async move {
            // An error about a generation carries its id.
            let (id, result) = match message.parse::<TextRequest>()? {
                TextRequest::Generate(request) => {
                    let id = request.id.clone();
                    (id, self.generate(request, outgoing))
                }
                TextRequest::Cancel { id } => {
                    let result = self.cancel(id.clone(), outgoing).await;
                    (id, result)
                }
            };
            if let Err(error) = result {
                outgoing.error(Some(&id), &error).await;
            }
            Ok(())
        })
    }
}

/// Generates `request` on `engine`, sending its text as it decodes and then `done`.
async fn stream(
    request: Generate,
    model: &str,
    engine: &TextEngine,
    format: Arc<TextFormat>,
    outgoing: &Outgoing,
) -> ApiResult<()> {
    let fields = request.generation;
    let rendering = Arc::clone(&format);
    let (messages, prompt, arguments) = (
        request.messages,
        request.prompt,
        request.chat_template_kwargs,
    );
    let prompt = blocking(move || match (messages, prompt) {
        (Some(messages), None) => {
            let text = rendering.render(&chat::messages(&messages)?, true, &arguments)?;
            rendering.encode(&text, false)
        }
        (None, Some(prompt)) => rendering.encode(&prompt, true),
        _ => Err(ApiError::invalid(
            "a generation of `messages` or of a `prompt`, and not both",
        )),
    })
    .await?;

    let limit = max_tokens(fields.max_tokens, prompt.len(), engine.limits())?;
    let sampling = fields.sampling(None)?;
    let options = format.sequence(limit, &sampling, &fields.stop_token_ids, fields.ignore_eos)?;
    let (stop_tokens, min_tokens) = (options.stop_tokens.clone(), options.min_tokens);
    let detokenizer = Detokenizer::new(
        Arc::clone(&format.tokenizer),
        fields.skip_special_tokens.unwrap_or(true),
        &prompt,
    )?;
    let prompt_tokens = prompt.len();
    let mut generation = TextGeneration::new(
        engine.submit(prompt, options)?,
        detokenizer,
        Stop::strings(fields.stop.as_ref()),
        stop_tokens,
        min_tokens,
    );
    let id = request.id;
    while let Some(piece) = generation.next().await? {
        let sent = match piece {
            Piece::Text { text, .. } if text.is_empty() => true,
            Piece::Text { text, .. } => {
                outgoing
                    .json(&TextEvent::Delta {
                        id: id.clone(),
                        text,
                    })
                    .await
            }
            Piece::Finished(reason) => {
                outgoing
                    .json(&TextEvent::Done {
                        id: id.clone(),
                        finish_reason: reason.as_str(),
                        usage: Some(Usage::new(prompt_tokens, generation.tokens().len())),
                    })
                    .await
            }
        };
        if !sent {
            return Ok(());
        }
    }

    generation.log(model, prompt_tokens);
    Ok(())
}
