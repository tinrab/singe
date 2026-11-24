//! Text out of tokens as they come: incremental detokenization, and stop strings that hold back text that may begin one.

use std::{mem, sync::Arc};

use singe_model::processor::text::TextTokenizer;

use crate::error::ApiResult;

/// Decodes tokens one at a time into the text they add, as `tokenizers`' `DecodeStream` does.
/// Each step decodes a window of recent tokens and gives what it adds beyond the text of the window's first part, once that text ends in whole characters.
/// The window starts with the end of the prompt, so the first token decodes as it does after its prompt, with the space a SentencePiece decoder would drop at the start of a text.
#[derive(Debug)]
pub struct Detokenizer {
    tokenizer: Arc<TextTokenizer>,
    skip_special_tokens: bool,
    /// The window, whose first `given` tokens decode to `prefix`, text already given or the prompt's.
    window: Vec<i64>,
    prefix: String,
    given: usize,
}

/// The prompt tokens with text of their own that the window starts with.
const PROMPT_TOKENS: usize = 5;

/// The most prompt tokens the window reaches back for them.
const PROMPT_REACH: usize = 32;

impl Detokenizer {
    /// Creates a detokenizer for the tokens generated after `prompt`.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding the prompt's end fails.
    pub fn new(
        tokenizer: Arc<TextTokenizer>,
        skip_special_tokens: bool,
        prompt: &[i64],
    ) -> ApiResult<Self> {
        let mut detokenizer = Self {
            tokenizer,
            skip_special_tokens,
            window: Vec::new(),
            prefix: String::new(),
            given: 0,
        };
        detokenizer.start_after(prompt)?;
        Ok(detokenizer)
    }

    /// Adds `token` and returns the text it completes, which may be empty.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn push(&mut self, token: i64) -> ApiResult<String> {
        self.window.push(token);
        let text = self.decode(&self.window)?;
        if text.len() <= self.prefix.len() || text.ends_with('\u{fffd}') {
            return Ok(String::new());
        }

        let Some(added) = text.strip_prefix(&self.prefix).map(str::to_owned) else {
            // The new token changed how the text before it decodes, and that text was given already.
            // Only the text past its length is given, never what was given again, and the window restarts after this token.
            let added = text[text.ceil_char_boundary(self.prefix.len())..].to_owned();
            let window = mem::take(&mut self.window);
            self.start_after(&window)?;
            return Ok(added);
        };
        self.window.drain(..self.given);
        self.prefix = self.decode(&self.window)?;
        self.given = self.window.len();
        Ok(added)
    }

    /// Returns the text of the tokens whose text was held back, at the end of the sequence, such as a character a token left unfinished.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn finish(&mut self) -> ApiResult<String> {
        let text = self.decode(&self.window)?;
        let added = text
            .get(text.ceil_char_boundary(self.prefix.len())..)
            .unwrap_or_default()
            .to_owned();
        self.window.clear();
        self.prefix.clear();
        self.given = 0;
        Ok(added)
    }

    /// Starts the window with the end of `tokens`: the last few that have text of their own, and more while their text ends in an unfinished character.
    fn start_after(&mut self, tokens: &[i64]) -> ApiResult<()> {
        let reach = tokens.len().saturating_sub(PROMPT_REACH);
        let mut start = tokens.len();
        let mut needed = PROMPT_TOKENS;
        let mut prefix = String::new();
        while start > reach {
            start -= 1;
            if needed > 0 && !self.decode(&tokens[start..=start])?.is_empty() {
                needed -= 1;
            }
            if needed == 0 {
                prefix = self.decode(&tokens[start..])?;
                if !prefix.ends_with('\u{fffd}') {
                    break;
                }
            }
        }

        if needed > 0 {
            // A short prompt: the window holds what is in reach.
            prefix = self.decode(&tokens[start..])?;
        }

        if prefix.ends_with('\u{fffd}') {
            // No start in reach decodes to whole characters; the window starts empty.
            start = tokens.len();
            prefix.clear();
        }

        self.window = tokens[start..].to_vec();
        self.prefix = prefix;
        self.given = self.window.len();
        Ok(())
    }

    fn decode(&self, tokens: &[i64]) -> ApiResult<String> {
        Ok(self.tokenizer.decode(tokens, self.skip_special_tokens)?)
    }
}

/// Ends text at the first of its stop strings, holding back the end of the text so far that could begin one.
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
}

impl StopMatcher {
    /// Creates a matcher for `stops`, ignoring empty ones.
    #[must_use]
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|stop| !stop.is_empty()).collect(),
            held: String::new(),
        }
    }

    /// Adds `text`, and returns the text that can be given now, and whether a stop string ended it.
    pub fn push(&mut self, text: &str) -> (String, bool) {
        self.held.push_str(text);
        let found = self
            .stops
            .iter()
            .filter_map(|stop| self.held.find(stop.as_str()))
            .min();
        if let Some(position) = found {
            let given = self.held[..position].to_owned();
            self.held.clear();
            return (given, true);
        }

        let keep = self.held_suffix();
        let split = self.held.len() - keep;
        let given = self.held[..split].to_owned();
        self.held.drain(..split);
        (given, false)
    }

    /// Returns the text held back, at the end of the sequence.
    pub fn finish(&mut self) -> String {
        mem::take(&mut self.held)
    }

    /// The length of the longest end of the held text that begins a stop string.
    fn held_suffix(&self) -> usize {
        self.stops
            .iter()
            .filter_map(|stop| {
                (1..stop.len())
                    .rev()
                    .filter(|&length| stop.is_char_boundary(length))
                    .find(|&length| self.held.ends_with(&stop[..length]))
            })
            .max()
            .unwrap_or(0)
    }
}
