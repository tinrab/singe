//! Text to token ids and back, through a snapshot's `tokenizer.json`, as `transformers`' fast tokenizers read it.
//! A snapshot with only slow-tokenizer files, such as BERT's `vocab.txt`, gets the tokenizer `transformers` converts them to.

use std::{collections::HashMap, result::Result as StdResult};

use serde_json::{Map, Value};
use singe::{DataType, Device, Error as SingeError, Shape, Tensor, nn::TokenBatch};
use tokenizers::{
    DecodeStream, Decoder, DecoderWrapper, EncodeInput, Encoding, ModelWrapper, NormalizerWrapper,
    PostProcessorWrapper, PreTokenizerWrapper, Tokenizer, TruncationParams, TruncationStrategy,
    normalizers::{Lowercase, Sequence as NormalizerSequence},
};

use crate::{
    error::{Error, Result},
    processor::conversion::{self, special_or, tokenizer_config},
    source::Source,
};

/// A snapshot's tokenizer, with the longest input its model reads and the special tokens its configs name.
#[derive(Debug, Clone)]
pub struct TextTokenizer {
    inner: Tokenizer,
    /// The longest input the snapshot's model reads, as `tokenizer_config.json`'s `model_max_length` gives it.
    max_length: Option<usize>,
    /// The fields of `special_tokens_map.json` and `tokenizer_config.json` in one object, the latter's winning, among them the special tokens they name by role, such as `mask_token`.
    config: Value,
}

/// What a pipeline encodes: one text, or a pair whose second text follows the first, as an encoder reads a question and its context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextInput<'a> {
    Single(&'a str),
    Pair(&'a str, &'a str),
}

/// How an input longer than its limit is cut, as `transformers` names the strategies: from the longer text of a pair first, or from one text of it alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Truncation {
    #[default]
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

impl Truncation {
    const fn strategy(self) -> TruncationStrategy {
        match self {
            Self::LongestFirst => TruncationStrategy::LongestFirst,
            Self::OnlyFirst => TruncationStrategy::OnlyFirst,
            Self::OnlySecond => TruncationStrategy::OnlySecond,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeOptions {
    /// The most tokens of a row, special tokens included, or no limit.
    pub max_length: Option<usize>,
    pub truncation: Truncation,
    /// With `max_length`, splits what truncation cuts into further rows, each repeating this many tokens of the window before it.
    pub stride: Option<usize>,
}

/// One row of an encoded batch: the input it came from, and for each of its tokens, its id, its token type, its characters in the text it came from, its word, and which text of the input it came from, none for a special token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedRow {
    pub input: usize,
    pub ids: Vec<i64>,
    pub types: Vec<i64>,
    pub offsets: Vec<(usize, usize)>,
    pub words: Vec<Option<u32>>,
    pub sequences: Vec<Option<usize>>,
}

impl EncodedRow {
    fn new(input: usize, encoding: &Encoding) -> Self {
        Self {
            input,
            ids: ids(encoding.get_ids()),
            types: ids(encoding.get_type_ids()),
            offsets: encoding.get_offsets().to_vec(),
            words: encoding.get_word_ids().to_vec(),
            sequences: encoding.get_sequence_ids(),
        }
    }

    /// Returns the row's tokens at `indices`, in order.
    fn select(&self, indices: impl Iterator<Item = usize> + Clone) -> Self {
        Self {
            input: self.input,
            ids: indices.clone().map(|index| self.ids[index]).collect(),
            types: indices.clone().map(|index| self.types[index]).collect(),
            offsets: indices.clone().map(|index| self.offsets[index]).collect(),
            words: indices.clone().map(|index| self.words[index]).collect(),
            sequences: indices.map(|index| self.sequences[index]).collect(),
        }
    }

    /// Splits the row into windows of at most `max_length` tokens, cutting only the text `truncation` names, each window repeating `stride` of its tokens from the one before and keeping the special tokens and any other text whole, as `transformers`' fast tokenizers split a truncated input.
    fn windows(
        self,
        truncation: Truncation,
        max_length: usize,
        stride: usize,
    ) -> Result<Vec<Self>> {
        let cut = match truncation {
            Truncation::OnlyFirst => 0,
            Truncation::OnlySecond if self.sequences.contains(&Some(1)) => 1,
            Truncation::OnlySecond | Truncation::LongestFirst => {
                return Err(Error::unsupported(
                    "windows of other than one text of an input",
                ));
            }
        };
        let positions = (0..self.ids.len())
            .filter(|&index| self.sequences[index] == Some(cut))
            .collect::<Vec<_>>();
        let (Some(&first), Some(&last)) = (positions.first(), positions.last()) else {
            return Ok(vec![self]);
        };
        let kept = self.ids.len() - positions.len();
        let budget = max_length
            .checked_sub(kept)
            .filter(|&budget| budget > stride)
            .ok_or_else(|| {
                Error::unsupported("a window too short for its stride and the text it keeps")
            })?;
        if positions.len() <= budget {
            return Ok(vec![self]);
        }

        // Each window starts `budget - stride` tokens after the one before, and the last ends with the text.
        let mut windows = Vec::new();
        for start in (0..positions.len()).step_by(budget - stride) {
            let stop = (start + budget).min(positions.len());
            let indices = (0..first)
                .chain(first + start..first + stop)
                .chain(last + 1..self.ids.len());
            windows.push(self.select(indices));
            if stop == positions.len() {
                break;
            }
        }
        Ok(windows)
    }
}

/// A batch for an encoder, and the rows it was made of.
#[derive(Debug)]
pub struct Encoded {
    pub batch: TokenBatch,
    pub rows: Vec<EncodedRow>,
}

/// Pads `rows` into a batch on `device`, with id zero and token type zero after each row's end.
fn batch(rows: &[EncodedRow], device: Device) -> Result<TokenBatch> {
    let sequences = rows
        .iter()
        .map(|row| row.ids.as_slice())
        .collect::<Vec<_>>();
    let batch = TokenBatch::from_sequences(device, &sequences)?;
    let &[_, length] = batch.tokens().shape().dimensions() else {
        return Err(SingeError::invalid_argument("token batch").into());
    };
    let width = usize::try_from(length).map_err(|_| SingeError::out_of_range("batch length"))?;
    let padded = rows
        .iter()
        .flat_map(|row| {
            let mut types = row.types.clone();
            types.resize(width, 0);
            types
        })
        .collect::<Vec<_>>();
    let count = i64::try_from(rows.len()).map_err(|_| SingeError::out_of_range("texts"))?;
    let token_types =
        Tensor::from_vec(device, Shape::create([count, length])?, padded)?.cast(DataType::I64)?;
    Ok(batch.with_token_types(token_types)?)
}

/// Returns the text of the special token `name` a tokenizer config gives, as a string or, in newer configs, an object of its `content`.
pub(crate) fn special_token<'a>(config: &'a Value, name: &str) -> Option<&'a str> {
    let value = &config[name];
    value.as_str().or_else(|| value["content"].as_str())
}

/// Removes the spaces a word-piece decode leaves before punctuation and contractions, as `clean_up_tokenization` does.
fn clean_up_tokenization(text: &str) -> String {
    [
        (" .", "."),
        (" ?", "?"),
        (" !", "!"),
        (" ,", ","),
        (" ' ", "'"),
        (" n't", "n't"),
        (" 'm", "'m"),
        (" 's", "'s"),
        (" 've", "'ve"),
        (" 're", "'re"),
    ]
    .iter()
    .fold(text.to_owned(), |text, (from, to)| text.replace(from, to))
}

fn ids(ids: &[u32]) -> Vec<i64> {
    ids.iter().map(|&id| i64::from(id)).collect()
}

impl TextTokenizer {
    /// Loads `tokenizer.json` from a snapshot directory, or builds BERT's `WordPiece` tokenizer from its `vocab.txt`, CLIP's or `RoBERTa`'s byte-level BPE from its `vocab.json` and `merges.txt`, or `XLM-RoBERTa`'s from its `sentencepiece.bpe.model`, when it has only those.
    ///
    /// # Errors
    ///
    /// Returns an error if `tokenizer.json` does not load, if the snapshot has neither it nor files to convert, or if `special_tokens_map.json` or `tokenizer_config.json` is there but cannot be read or parsed.
    pub fn load(source: &Source) -> Result<Self> {
        let mut inner = if source.has("tokenizer.json") {
            Tokenizer::from_file(source.directory().join("tokenizer.json")).map_err(|source| {
                Error::Tokenizer {
                    what: "load tokenizer.json",
                    source,
                }
            })?
        } else {
            conversion::convert(source)?
        };
        // A snapshot's padding and truncation settings are what it was trained with, which `transformers` drops on loading and applies only when asked.
        inner.with_padding(None);
        inner
            .with_truncation(None)
            .map_err(|source| Error::Tokenizer {
                what: "clear truncation",
                source,
            })?;
        // `transformers` loads SigLIP 2's tokenizer as `Siglip2Tokenizer`, which lowercases before the snapshot's normalizer.
        let model: Value = source.optional_json("config.json")?.unwrap_or_default();
        if model["model_type"].as_str() == Some("siglip2") {
            let mut normalizers = vec![NormalizerWrapper::Lowercase(Lowercase)];
            normalizers.extend(inner.get_normalizer().cloned());
            inner
                .with_normalizer(Some(NormalizerWrapper::Sequence(NormalizerSequence::new(
                    normalizers,
                ))))
                .map_err(|source| Error::Tokenizer {
                    what: "add lowercasing",
                    source,
                })?;
        }

        let mut config = Map::new();
        for file in ["special_tokens_map.json", "tokenizer_config.json"] {
            if let Some(Value::Object(fields)) = source.optional_json(file)? {
                config.extend(fields);
            }
        }

        let max_length = config
            .get("model_max_length")
            .and_then(Value::as_u64)
            .and_then(|length| usize::try_from(length).ok());
        Ok(Self {
            inner,
            max_length,
            config: Value::Object(config),
        })
    }

    /// Encodes `text`, with the special tokens the tokenizer's post-processor adds when `special_tokens` is set, such as BERT's `[CLS]` and `[SEP]`.
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

    /// Encodes `texts` with special tokens into a batch for an encoder on `device`, each truncated to `max_length` tokens if given, keeping its special tokens.
    pub fn encode_batch(
        &self,
        texts: &[&str],
        max_length: Option<usize>,
        device: Device,
    ) -> Result<TokenBatch> {
        let inputs = texts
            .iter()
            .map(|&text| TextInput::Single(text))
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            max_length,
            ..EncodeOptions::default()
        };
        Ok(self.encode_inputs(&inputs, &options, device)?.batch)
    }

    /// Encodes `inputs` with special tokens into a batch for an encoder on `device`, one row per input and per window it overflows into, padded with id zero at each row's end, with the token types the tokenizer gives.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer fails to encode an input, or, with a stride, if the truncation would cut other than one text of an input, or if `max_length` leaves no more than `stride` tokens of the cut text in a window.
    pub fn encode_inputs(
        &self,
        inputs: &[TextInput<'_>],
        options: &EncodeOptions,
        device: Device,
    ) -> Result<Encoded> {
        // Windows are cut here from whole encodings, so the tokenizer truncates only inputs read in one row.
        let mut tokenizer = self.inner.clone();
        let truncation =
            options
                .max_length
                .filter(|_| options.stride.is_none())
                .map(|max_length| TruncationParams {
                    max_length,
                    strategy: options.truncation.strategy(),
                    ..TruncationParams::default()
                });
        tokenizer
            .with_truncation(truncation)
            .map_err(|source| Error::Tokenizer {
                what: "set truncation",
                source,
            })?;
        let encodings = tokenizer
            .encode_batch_char_offsets(
                inputs
                    .iter()
                    .map(|&input| match input {
                        TextInput::Single(text) => EncodeInput::from(text),
                        TextInput::Pair(first, second) => EncodeInput::from((first, second)),
                    })
                    .collect(),
                true,
            )
            .map_err(|source| Error::Tokenizer {
                what: "encode texts",
                source,
            })?;
        let mut rows = Vec::new();
        for (input, encoding) in encodings.iter().enumerate() {
            let row = EncodedRow::new(input, encoding);
            match (options.max_length, options.stride) {
                (Some(max_length), Some(stride)) => {
                    rows.extend(row.windows(options.truncation, max_length, stride)?);
                }
                _ => rows.push(row),
            }
        }

        let batch = batch(&rows, device)?;
        Ok(Encoded { batch, rows })
    }

    /// Decodes generated ids to text, leaving out special tokens when `skip_special_tokens` is set.
    /// It removes the spaces before punctuation and contractions where `transformers` would: when the config sets `clean_up_tokenization_spaces`, and for a BPE model only when the config also insists.
    ///
    /// # Errors
    ///
    /// Returns an error if an id is outside `u32`, or if the tokenizer fails to decode the ids.
    pub fn decode(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String> {
        let ids = ids
            .iter()
            .map(|&id| u32::try_from(id))
            .collect::<StdResult<Vec<_>, _>>()
            .map_err(|_| Error::invalid("a token id outside u32"))?;
        let text = self
            .inner
            .decode(&ids, skip_special_tokens)
            .map_err(|source| Error::Tokenizer {
                what: "decode ids",
                source,
            })?;
        // `transformers` cleans up only when the config asks, and never a BPE model's text, unless told to anyway.
        let flag = |name: &str| self.config[name].as_bool().unwrap_or(false);
        let bpe = matches!(self.inner.get_model(), ModelWrapper::BPE(_));
        if flag("clean_up_tokenization_spaces")
            && (!bpe
                || flag("clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output"))
        {
            return Ok(clean_up_tokenization(&text));
        }
        Ok(text)
    }

    /// Starts turning generated ids into text one at a time, leaving out special tokens when `skip_special_tokens` is set.
    #[must_use]
    pub fn stream(&self, skip_special_tokens: bool) -> TextStream<'_> {
        TextStream {
            inner: self.inner.decode_stream(skip_special_tokens),
        }
    }

    /// Returns the number of tokens the vocabulary holds, added tokens included.
    #[must_use]
    pub fn vocabulary_size(&self) -> i64 {
        i64::try_from(self.inner.get_vocab_size(true)).unwrap_or(i64::MAX)
    }

    /// Returns the id of a token by its text, such as Whisper's `<|en|>`.
    ///
    /// # Errors
    ///
    /// Returns an error if the vocabulary has no token `token`.
    pub fn token_id(&self, token: &str) -> Result<i64> {
        self.inner
            .token_to_id(token)
            .map(i64::from)
            .ok_or_else(|| Error::unsupported(format!("a tokenizer without {token}")))
    }

    /// Returns the token of `id` as the vocabulary writes it, such as `WordPiece`'s `##ing`.
    #[must_use]
    pub fn token(&self, id: i64) -> Option<String> {
        u32::try_from(id)
            .ok()
            .and_then(|id| self.inner.id_to_token(id))
    }

    /// Joins tokens into text with the tokenizer's decoder, or with spaces without one, as `transformers`' `convert_tokens_to_string` does.
    pub fn join_tokens(&self, tokens: Vec<String>) -> Result<String> {
        match self.inner.get_decoder() {
            Some(decoder) => decoder.decode(tokens).map_err(|source| Error::Tokenizer {
                what: "join tokens",
                source,
            }),
            None => Ok(tokens.join(" ")),
        }
    }

    /// Returns the prefix the model marks a word's continuing pieces with, such as `WordPiece`'s `##`, or `None` if it marks none.
    #[must_use]
    pub fn subword_prefix(&self) -> Option<&str> {
        match self.inner.get_model() {
            ModelWrapper::WordPiece(model) => Some(model.continuing_subword_prefix.as_str()),
            ModelWrapper::BPE(model) => model.continuing_subword_prefix.as_deref(),
            ModelWrapper::WordLevel(_) | ModelWrapper::Unigram(_) => None,
        }
        .filter(|prefix| !prefix.is_empty())
    }

    /// Returns the longest input the model reads, if the snapshot says.
    #[must_use]
    pub const fn max_length(&self) -> Option<usize> {
        self.max_length
    }

    /// Returns the id of the special token the tokenizer's configs name for `role`, such as `mask_token`, as a string or an object of its `content`.
    #[must_use]
    pub fn special_id(&self, role: &str) -> Option<i64> {
        let token = special_token(&self.config, role)?;
        self.inner.token_to_id(token).map(i64::from)
    }

    /// Returns the id of the mask token: the one the configs name, or the added special token named `mask` in brackets, as `[MASK]` and `<mask>` are.
    #[must_use]
    pub fn mask_id(&self) -> Option<i64> {
        self.special_id("mask_token").or_else(|| {
            self.inner
                .get_added_tokens_decoder()
                .iter()
                .find(|(_, token)| {
                    token.special
                        && token
                            .content
                            .trim_matches(|character| "[]<>".contains(character))
                            .eq_ignore_ascii_case("mask")
                })
                .map(|(&id, _)| i64::from(id))
        })
    }

    /// Returns the id of the token the post-processor puts before a text, as BERT's `[CLS]` and `RoBERTa`'s `<s>` are, if it puts one.
    pub fn classifier_id(&self) -> Result<Option<i64>> {
        let encoding = self
            .inner
            .encode("", true)
            .map_err(|source| Error::Tokenizer {
                what: "encode an empty text",
                source,
            })?;
        Ok(encoding
            .get_special_tokens_mask()
            .first()
            .filter(|&&special| special == 1)
            .and_then(|_| encoding.get_ids().first())
            .map(|&id| i64::from(id)))
    }

    /// Returns the id of the model's unknown token, if it has one.
    #[must_use]
    pub fn unknown_id(&self) -> Option<i64> {
        let token = match self.inner.get_model() {
            ModelWrapper::WordPiece(model) => Some(model.unk_token.as_str()),
            ModelWrapper::BPE(model) => model.unk_token.as_deref(),
            ModelWrapper::WordLevel(_) | ModelWrapper::Unigram(_) => None,
        }?;
        self.inner.token_to_id(token).map(i64::from)
    }
}

/// Generated ids turned into text as they arrive.
///
/// An id that ends partway through a character, as byte-level tokenizers split one, gives no text until the ids after it complete the character, so each piece is whole text and the pieces join to what [`TextTokenizer::decode`] gives.
pub struct TextStream<'a> {
    inner: DecodeStream<
        'a,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
}

impl TextStream<'_> {
    /// Adds the next id, and returns the text it completes, or `None` while the text it starts is not whole.
    ///
    /// # Errors
    ///
    /// Returns an error if `id` is outside `u32`, or if the tokenizer fails to decode it.
    pub fn push(&mut self, id: i64) -> Result<Option<String>> {
        let id = u32::try_from(id).map_err(|_| Error::invalid("a token id outside u32"))?;
        self.inner.step(id).map_err(|source| Error::Tokenizer {
            what: "decode a streamed id",
            source,
        })
    }
}

/// How a CTC model's frames' best tokens become text: repeats collapsed, the blank dropped, and the rest read as the snapshot's tokenizer reads them.
#[derive(Debug, Clone)]
pub enum CtcTokenizer {
    /// Characters, as `Wav2Vec2CTCTokenizer` reads them from `vocab.json`: the blank is the padding token, and the word delimiter a space.
    Characters(CtcVocabulary),
    /// Pieces of `tokenizer.json`, as `ParakeetTokenizer` decodes them: the blank is the padding token, and the rest decode without special tokens.
    Pieces {
        tokenizer: Box<TextTokenizer>,
        blank: i64,
    },
}

#[derive(Debug, Clone)]
pub struct CtcVocabulary {
    tokens: Vec<String>,
    padding: String,
    delimiter: String,
    lowercase: bool,
    clean_up: bool,
}

impl CtcTokenizer {
    /// Reads `vocab.json`'s characters when the snapshot has one, and `tokenizer.json`'s pieces otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if `vocab.json` has an id that is not a non-negative integer, if a snapshot without it has no tokenizer or no padding token, or if a tokenizer config cannot be read or parsed.
    pub fn load(source: &Source) -> Result<Self> {
        let Some(vocabulary) = source.optional_json::<Map<String, Value>>("vocab.json")? else {
            let tokenizer = TextTokenizer::load(source)?;
            let blank = tokenizer
                .special_id("pad_token")
                .ok_or_else(|| Error::unsupported("a CTC tokenizer without a padding token"))?;
            return Ok(Self::Pieces {
                tokenizer: Box::new(tokenizer),
                blank,
            });
        };
        let config = tokenizer_config(source)?;
        let mut tokens = Vec::new();
        for (token, id) in &vocabulary {
            let id = id
                .as_u64()
                .and_then(|id| usize::try_from(id).ok())
                .ok_or_else(|| Error::unsupported(format!("a vocab.json id for {token}")))?;
            if tokens.len() <= id {
                tokens.resize(id + 1, String::new());
            }

            tokens[id].clone_from(token);
        }
        Ok(Self::Characters(CtcVocabulary {
            tokens,
            padding: special_or(&config, "pad_token", "<pad>"),
            delimiter: special_or(&config, "word_delimiter_token", "|"),
            lowercase: config["do_lower_case"].as_bool().unwrap_or(false),
            clean_up: config["clean_up_tokenization_spaces"]
                .as_bool()
                .unwrap_or(false),
        }))
    }

    /// Decodes `ids`, each frame's best token, to text, as the speech recognition pipeline decodes a CTC model's output.
    ///
    /// # Errors
    ///
    /// Returns an error if an id is outside `vocab.json`'s characters, or, for pieces, if the tokenizer fails to decode the ids.
    pub fn decode(&self, ids: &[i64]) -> Result<String> {
        match self {
            Self::Characters(characters) => characters.decode(ids),
            Self::Pieces { tokenizer, blank } => {
                let mut kept = Vec::with_capacity(ids.len());
                let mut previous = None;
                for &id in ids {
                    if previous != Some(id) && id != *blank {
                        kept.push(id);
                    }

                    previous = Some(id);
                }
                Ok(tokenizer.decode(&kept, true)?.trim().to_owned())
            }
        }
    }
}

impl CtcVocabulary {
    /// Decodes `ids`, collapsing repeats, dropping the blank but keeping other special tokens, and writing the word delimiter as a space.
    fn decode(&self, ids: &[i64]) -> Result<String> {
        let mut text = String::new();
        let mut previous = None;
        for &id in ids {
            if previous == Some(id) {
                continue;
            }

            previous = Some(id);
            let token = usize::try_from(id)
                .ok()
                .and_then(|id| self.tokens.get(id))
                .ok_or_else(|| Error::invalid(format!("token {id} outside vocab.json")))?;
            if *token == self.padding {
                continue;
            }

            text.push_str(if *token == self.delimiter { " " } else { token });
        }

        let mut text = text.trim().to_owned();
        if self.lowercase {
            text = text.to_lowercase();
        }
        Ok(if self.clean_up {
            clean_up_tokenization(&text)
        } else {
            text
        })
    }
}

/// A character tokenizer, as `VitsTokenizer` reads `vocab.json`.
/// Unless the config turns `normalize` off, text is lowercased where its characters are not in the vocabulary, and characters still not in it are dropped.
/// Unless the config turns `add_blank` off, a blank, id zero, goes between every two characters and at both ends.
#[derive(Debug, Clone)]
pub struct CharacterTokenizer {
    vocabulary: HashMap<char, i64>,
    normalize: bool,
    add_blank: bool,
}

impl CharacterTokenizer {
    pub fn load(source: &Source) -> Result<Self> {
        let vocabulary: Map<String, Value> = source
            .optional_json("vocab.json")?
            .ok_or_else(|| Error::unsupported("a character snapshot without vocab.json"))?;
        let config = tokenizer_config(source)?;
        let mut characters = HashMap::new();
        for (token, id) in &vocabulary {
            let mut chars = token.chars();
            if let (Some(character), None, Some(id)) = (chars.next(), chars.next(), id.as_i64()) {
                characters.insert(character, id);
            }
        }

        if config["phonemize"].as_bool().unwrap_or(false)
            || config["is_uroman"].as_bool().unwrap_or(false)
        {
            return Err(Error::unsupported("phonemized or romanized text"));
        }
        Ok(Self {
            vocabulary: characters,
            normalize: config["normalize"].as_bool().unwrap_or(true),
            add_blank: config["add_blank"].as_bool().unwrap_or(true),
        })
    }

    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let text = if self.normalize {
            let lowered = text
                .chars()
                .flat_map(|character| {
                    if self.vocabulary.contains_key(&character) {
                        vec![character]
                    } else {
                        character.to_lowercase().collect()
                    }
                })
                .filter(|character| self.vocabulary.contains_key(character))
                .collect::<String>();
            lowered.trim().to_owned()
        } else {
            text.to_owned()
        };
        let ids = text
            .chars()
            .filter_map(|character| self.vocabulary.get(&character).copied())
            .collect::<Vec<_>>();
        if !self.add_blank {
            return ids;
        }

        let mut interspersed = vec![0; ids.len() * 2 + 1];
        for (index, id) in ids.into_iter().enumerate() {
            interspersed[2 * index + 1] = id;
        }
        interspersed
    }
}

/// A tokenizer of space-separated `ARPAbet` phonemes, as `FastSpeech2ConformerTokenizer` maps what its grapheme-to-phoneme model gives: each phoneme's id, or the unknown token's, then the end token.
/// It reads phonemes, not text, since Singe has no grapheme-to-phoneme model.
#[derive(Debug, Clone)]
pub struct PhonemeTokenizer {
    vocabulary: HashMap<String, i64>,
    unknown: i64,
    end: i64,
}

impl PhonemeTokenizer {
    pub fn load(source: &Source) -> Result<Self> {
        let vocabulary: HashMap<String, i64> = source
            .optional_json("vocab.json")?
            .ok_or_else(|| Error::unsupported("a phoneme snapshot without vocab.json"))?;
        let config = tokenizer_config(source)?;
        let id = |name: &str, default: &str| {
            let token = special_or(&config, name, default);
            vocabulary.get(&token).copied().ok_or_else(|| {
                Error::unsupported(format!("a phoneme vocabulary without `{token}`"))
            })
        };
        Ok(Self {
            unknown: id("unk_token", "<unk>")?,
            end: id("eos_token", "<sos/eos>")?,
            vocabulary,
        })
    }

    #[must_use]
    pub fn encode(&self, phonemes: &str) -> Vec<i64> {
        phonemes
            .split_whitespace()
            .map(|phoneme| {
                self.vocabulary
                    .get(phoneme)
                    .copied()
                    .unwrap_or(self.unknown)
            })
            .chain([self.end])
            .collect()
    }
}

/// A text-to-speech model's tokenizer: characters, as VITS reads them, or phonemes, as `FastSpeech2Conformer` does.
#[derive(Debug, Clone)]
pub enum SpeechTokenizer {
    Characters(CharacterTokenizer),
    Phonemes(PhonemeTokenizer),
}

impl SpeechTokenizer {
    /// Loads the phoneme tokenizer for a snapshot whose tokenizer is `FastSpeech2ConformerTokenizer`, and the character tokenizer otherwise.
    pub fn load(source: &Source) -> Result<Self> {
        let config = tokenizer_config(source)?;
        Ok(
            if config["tokenizer_class"].as_str() == Some("FastSpeech2ConformerTokenizer") {
                Self::Phonemes(PhonemeTokenizer::load(source)?)
            } else {
                Self::Characters(CharacterTokenizer::load(source)?)
            },
        )
    }

    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<i64> {
        match self {
            Self::Characters(tokenizer) => tokenizer.encode(text),
            Self::Phonemes(tokenizer) => tokenizer.encode(text),
        }
    }
}
