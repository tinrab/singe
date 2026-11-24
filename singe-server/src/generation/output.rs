//! A request's generated text, piece by piece: its tokens detokenized, cut at its stop strings, without the stop token's own text.

use std::{collections::VecDeque, time::Instant};

use crate::{
    error::{ApiError, ApiResult},
    generation::engine::{FinishReason, GenerationEvent, GenerationStream},
    text::{Detokenizer, StopMatcher},
};

/// A part of a request's output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    /// Text, never empty.
    Text(String),
    /// The last piece, with why the text ended.
    Finished(FinishReason),
}

/// A request's [`GenerationStream`] turned into text.
#[derive(Debug)]
pub struct TextGeneration {
    /// Dropped once a stop string ends the text, which cancels the request.
    stream: Option<GenerationStream>,
    detokenizer: Detokenizer,
    stops: StopMatcher,
    stop_tokens: Vec<i64>,
    pending: VecDeque<Piece>,
    tokens: Vec<i64>,
    started: Instant,
    first_token: Option<Instant>,
}

impl TextGeneration {
    /// Creates the output of `stream`, cut at `stops`, without the text of `stop_tokens`.
    /// `detokenizer` starts after the request's prompt, which `stream` generates from.
    #[must_use]
    pub fn new(
        stream: GenerationStream,
        detokenizer: Detokenizer,
        stops: Vec<String>,
        stop_tokens: Vec<i64>,
    ) -> Self {
        Self {
            stream: Some(stream),
            detokenizer,
            stops: StopMatcher::new(stops),
            stop_tokens,
            pending: VecDeque::new(),
            tokens: Vec::new(),
            started: Instant::now(),
            first_token: None,
        }
    }

    /// Returns the next piece, or `None` after the last.
    /// A stop string ends the text and drops the stream, which cancels the request on the engine.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails, and a 500 error if generation failed or the engine dropped the request.
    pub async fn next(&mut self) -> ApiResult<Option<Piece>> {
        loop {
            if let Some(piece) = self.pending.pop_front() {
                return Ok(Some(piece));
            }

            let Some(stream) = self.stream.as_mut() else {
                return Ok(None);
            };
            match stream.recv().await {
                Some(GenerationEvent::Token(token)) => {
                    self.first_token.get_or_insert_with(Instant::now);
                    self.tokens.push(token);
                    if self.stop_tokens.contains(&token) {
                        continue;
                    }

                    let added = self.detokenizer.push(token)?;
                    let (text, stopped) = self.stops.push(&added);
                    if !text.is_empty() {
                        self.pending.push_back(Piece::Text(text));
                    }
                    if stopped {
                        self.pending.push_back(Piece::Finished(FinishReason::Stop));
                        self.stream = None;
                    }
                }
                Some(GenerationEvent::Finished(reason)) => {
                    let rest = self.detokenizer.finish()?;
                    let (mut text, stopped) = self.stops.push(&rest);
                    if !stopped {
                        text.push_str(&self.stops.finish());
                    }

                    if !text.is_empty() {
                        self.pending.push_back(Piece::Text(text));
                    }

                    let reason = if stopped { FinishReason::Stop } else { reason };
                    self.pending.push_back(Piece::Finished(reason));
                    self.stream = None;
                }
                Some(GenerationEvent::Failed(error)) => {
                    self.stream = None;
                    return Err(ApiError::internal(format!("generation failed: {error}")));
                }
                None => {
                    self.stream = None;
                    return Err(ApiError::internal("generation stopped"));
                }
            }
        }
    }

    /// Reads every piece, and returns the text and why it ended.
    ///
    /// # Errors
    ///
    /// Returns the first error [`TextGeneration::next`] returns.
    pub async fn collect(&mut self) -> ApiResult<(String, FinishReason)> {
        let mut text = String::new();
        let mut reason = FinishReason::Length;
        while let Some(piece) = self.next().await? {
            match piece {
                Piece::Text(piece) => text.push_str(&piece),
                Piece::Finished(finished) => reason = finished,
            }
        }
        Ok((text, reason))
    }

    /// Returns the tokens generated so far, the stop token included.
    #[must_use]
    pub fn tokens(&self) -> &[i64] {
        &self.tokens
    }

    /// Logs the request's tokens, its time to the first, and its rate after it.
    pub fn log(&self, model: &str, prompt_tokens: usize) {
        let elapsed = self.started.elapsed();
        let first = self
            .first_token
            .map(|first| first.duration_since(self.started));
        let after = first.map(|first| elapsed.saturating_sub(first));
        let rate = after
            .filter(|after| !after.is_zero() && self.tokens.len() > 1)
            .map(|after| (self.tokens.len() - 1) as f64 / after.as_secs_f64());
        tracing::info!(
            model,
            prompt_tokens,
            completion_tokens = self.tokens.len(),
            first_token_ms = first.map(|first| first.as_millis()),
            tokens_per_second = rate.map(f64::round),
            elapsed_ms = elapsed.as_millis(),
            "generated"
        );
    }
}
