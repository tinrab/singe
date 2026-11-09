// Text to token ids and back, through a snapshot's `tokenizer.json`, as `transformers`' fast tokenizers read it.

use std::{path::Path, result::Result as StdResult};

use singe::{DataType, Device, Shape, Tensor, nn::TokenBatch};
use tokenizers::{EncodeInput, Tokenizer, TruncationParams};

use crate::{Error, Result};

// A snapshot's tokenizer.
#[derive(Debug, Clone)]
pub struct TextTokenizer {
    inner: Tokenizer,
}

fn ids(ids: &[u32]) -> Vec<i64> {
    ids.iter().map(|&id| i64::from(id)).collect()
}

impl TextTokenizer {
    // Loads `tokenizer.json` from a snapshot directory, refusing one without it.
    pub fn load(snapshot: impl AsRef<Path>) -> Result<Self> {
        let path = snapshot.as_ref().join("tokenizer.json");
        let mut inner = Tokenizer::from_file(&path).map_err(|source| Error::Tokenizer {
            what: "load tokenizer.json",
            source,
        })?;
        // A snapshot's padding and truncation settings are what it was trained with, which `transformers` drops on loading and applies only when asked.
        inner.with_padding(None);
        inner
            .with_truncation(None)
            .map_err(|source| Error::Tokenizer {
                what: "clear truncation",
                source,
            })?;
        Ok(Self { inner })
    }

    // Encodes `text`, with the special tokens the tokenizer's post-processor adds when `special_tokens` is set, such as BERT's `[CLS]` and `[SEP]`.
    pub fn encode(&self, text: &str, special_tokens: bool) -> Result<Vec<i64>> {
        let encoding =
            self.inner
                .encode(text, special_tokens)
                .map_err(|source| Error::Tokenizer {
                    what: "encode text",
                    source,
                })?;
        Ok(ids(encoding.get_ids()))
    }

    // Encodes `texts` with special tokens into a batch for an encoder on `device`, each truncated to `max_length` tokens if given, keeping its special tokens, padded with id zero at its end, with the token types the tokenizer gives.
    pub fn encode_batch(
        &self,
        texts: &[&str],
        max_length: Option<usize>,
        device: Device,
    ) -> Result<TokenBatch> {
        let inputs = texts
            .iter()
            .map(|&text| EncodeInput::from(text))
            .collect::<Vec<_>>();
        // Truncation keeps the special tokens, as `transformers` truncates.
        let mut tokenizer = self.inner.clone();
        tokenizer
            .with_truncation(max_length.map(|max_length| TruncationParams {
                max_length,
                ..TruncationParams::default()
            }))
            .map_err(|source| Error::Tokenizer {
                what: "set truncation",
                source,
            })?;
        let encodings =
            tokenizer
                .encode_batch(inputs, true)
                .map_err(|source| Error::Tokenizer {
                    what: "encode texts",
                    source,
                })?;
        let sequences = encodings
            .iter()
            .map(|encoding| ids(encoding.get_ids()))
            .collect::<Vec<_>>();
        let types = encodings
            .iter()
            .map(|encoding| ids(encoding.get_type_ids()))
            .collect::<Vec<_>>();
        let rows = sequences.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let batch = TokenBatch::from_sequences(device, &rows)?;
        let length = batch.tokens().shape().dimensions()[1];
        let width = usize::try_from(length)
            .map_err(|_| Error::UnsupportedConfig("a batch this long".to_owned()))?;
        let padded = types
            .iter()
            .flat_map(|types| {
                let mut types = types.clone();
                types.resize(width, 0);
                types
            })
            .collect::<Vec<_>>();
        let count = i64::try_from(texts.len())
            .map_err(|_| Error::UnsupportedConfig("this many texts".to_owned()))?;
        let token_types = Tensor::from_vec(device, Shape::create([count, length])?, padded)?
            .cast(DataType::I64)?;
        Ok(batch.with_token_types(token_types)?)
    }

    // Decodes generated ids to text, leaving out special tokens when `skip_special_tokens` is set.
    pub fn decode(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String> {
        let ids = ids
            .iter()
            .map(|&id| u32::try_from(id))
            .collect::<StdResult<Vec<_>, _>>()
            .map_err(|_| Error::UnsupportedConfig("a token id outside u32".to_owned()))?;
        self.inner
            .decode(&ids, skip_special_tokens)
            .map_err(|source| Error::Tokenizer {
                what: "decode ids",
                source,
            })
    }

    // Returns the id of a token by its text, such as Whisper's `<|en|>`.
    pub fn token_id(&self, token: &str) -> Result<i64> {
        self.inner
            .token_to_id(token)
            .map(i64::from)
            .ok_or_else(|| Error::UnsupportedConfig(format!("a tokenizer without {token}")))
    }
}
