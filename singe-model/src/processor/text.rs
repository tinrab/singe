// Text to token ids and back, through a snapshot's `tokenizer.json`, as `transformers`' fast tokenizers read it.
// A snapshot with only BERT's `vocab.txt` gets the tokenizer `transformers` converts it to.

use std::{collections::HashMap, fs, path::Path, result::Result as StdResult, str::FromStr};

use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use singe::{DataType, Device, Shape, Tensor, nn::TokenBatch};
use tokenizers::{
    DecodeStream, Decoder, DecoderWrapper, EncodeInput, Encoding, ModelWrapper, NormalizerWrapper,
    PostProcessorWrapper, PreTokenizerWrapper, Tokenizer, TruncationParams, TruncationStrategy,
};

use crate::{Error, Result};

// A snapshot's tokenizer.
#[derive(Debug, Clone)]
pub struct TextTokenizer {
    inner: Tokenizer,
    // The longest input the snapshot's model reads, as `tokenizer_config.json`'s `model_max_length` gives it.
    max_length: Option<usize>,
    // The special tokens `tokenizer_config.json` and `special_tokens_map.json` name by role, such as `mask_token`, in one object.
    special: Value,
}

// What a pipeline encodes: one text, or a pair whose second text follows the first, as an encoder reads a question and its context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextInput<'a> {
    Single(&'a str),
    Pair(&'a str, &'a str),
}

// How an input longer than its limit is cut, as `transformers` names the strategies: from the longer text of a pair first, or from one text of it alone.
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
    // The most tokens of a row, special tokens included, or no limit.
    pub max_length: Option<usize>,
    pub truncation: Truncation,
    // With `max_length`, splits what truncation cuts into further rows, each repeating this many tokens of the window before it.
    pub stride: Option<usize>,
}

// One row of an encoded batch: the input it came from, and for each of its tokens, its id, its characters in the text it came from, its word, and which text of the input it came from, none for a special token.
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

    // The row's tokens at `indices`, in order.
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

    // Splits the row into windows of at most `max_length` tokens, cutting only the text `truncation` names, each window repeating `stride` of its tokens from the one before and keeping the special tokens and any other text whole, as `transformers`' fast tokenizers split a truncated input.
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

// A batch for an encoder, and the rows it was made of.
#[derive(Debug)]
pub struct Encoded {
    pub batch: TokenBatch,
    pub rows: Vec<EncodedRow>,
}

// Pads `rows` into a batch on `device`, with id zero and token type zero after each row's end.
fn batch(rows: &[EncodedRow], device: Device) -> Result<TokenBatch> {
    let sequences = rows
        .iter()
        .map(|row| row.ids.as_slice())
        .collect::<Vec<_>>();
    let batch = TokenBatch::from_sequences(device, &sequences)?;
    let length = batch.tokens().shape().dimensions()[1];
    let width = usize::try_from(length).map_err(|_| Error::unsupported("a batch this long"))?;
    let padded = rows
        .iter()
        .flat_map(|row| {
            let mut types = row.types.clone();
            types.resize(width, 0);
            types
        })
        .collect::<Vec<_>>();
    let count = i64::try_from(rows.len()).map_err(|_| Error::unsupported("this many texts"))?;
    let token_types =
        Tensor::from_vec(device, Shape::create([count, length])?, padded)?.cast(DataType::I64)?;
    Ok(batch.with_token_types(token_types)?)
}

// The tokenizer `transformers`' `BertConverter` makes of a slow `BertTokenizer`: `vocab.txt`'s WordPiece vocabulary, with the casing, accents, and special tokens `tokenizer_config.json` gives.
fn wordpiece(directory: &Path) -> Result<Tokenizer> {
    let config = read_optional(&directory.join("tokenizer_config.json"))?.unwrap_or(Value::Null);
    let special =
        |name: &str, default: &str| special_token(&config, name).unwrap_or(default).to_owned();
    let flag = |name: &str| config[name].as_bool().unwrap_or(true);
    let (unknown, separator, padding, classifier, mask) = (
        special("unk_token", "[UNK]"),
        special("sep_token", "[SEP]"),
        special("pad_token", "[PAD]"),
        special("cls_token", "[CLS]"),
        special("mask_token", "[MASK]"),
    );

    let path = directory.join("vocab.txt");
    let text = fs::read_to_string(&path).map_err(Error::io(&path))?;
    let mut vocabulary = Map::new();
    // A token listed twice takes its later id, as `transformers`' `load_vocab` reads it.
    for (id, token) in text.lines().enumerate() {
        vocabulary.insert(token.to_owned(), json!(id));
    }

    let id = |token: &str| vocabulary_id(&vocabulary, "vocab.txt", token);
    let added = [&padding, &unknown, &classifier, &separator, &mask]
        .into_iter()
        .map(|token| added_token(&vocabulary, "vocab.txt", token, false))
        .collect::<Result<Vec<_>>>()?;
    let piece = |id: &str, type_id: u32| json!({"SpecialToken": {"id": id, "type_id": type_id}});
    let sequence = |id: &str, type_id: u32| json!({"Sequence": {"id": id, "type_id": type_id}});
    let tokenizer = json!({
        "version": "1.0",
        "added_tokens": added,
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": true,
            "handle_chinese_chars": flag("tokenize_chinese_chars"),
            "strip_accents": config["strip_accents"].as_bool(),
            "lowercase": flag("do_lower_case"),
        },
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [piece(&classifier, 0), sequence("A", 0), piece(&separator, 0)],
            "pair": [
                piece(&classifier, 0), sequence("A", 0), piece(&separator, 0),
                sequence("B", 1), piece(&separator, 1),
            ],
            "special_tokens": {
                &classifier: {"id": &classifier, "ids": [id(&classifier)?], "tokens": [&classifier]},
                &separator: {"id": &separator, "ids": [id(&separator)?], "tokens": [&separator]},
            },
        },
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
        "model": {
            "type": "WordPiece",
            "unk_token": unknown,
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocabulary,
        },
    });
    Tokenizer::from_str(&tokenizer.to_string()).map_err(|source| Error::Tokenizer {
        what: "build a WordPiece tokenizer from vocab.txt",
        source,
    })
}

// CLIP's byte-level BPE tokenizer, built from `vocab.json` and `merges.txt` as `transformers`' `CLIPTokenizer` builds it: text normalized, lowercased, and split into words, each word's last piece marked with `</w>`, between `<|startoftext|>` and `<|endoftext|>`.
fn clip_bpe(directory: &Path) -> Result<Tokenizer> {
    let config = read_optional(&directory.join("tokenizer_config.json"))?.unwrap_or(Value::Null);
    let class = config["tokenizer_class"].as_str().unwrap_or_default();
    if ["Bart", "Roberta", "Tapex"]
        .iter()
        .any(|prefix| class.starts_with(prefix))
    {
        return roberta_bpe(directory, &config);
    }

    if !class.starts_with("CLIP") {
        return Err(Error::UnsupportedConfig(format!(
            "a {class} snapshot with merges.txt but no tokenizer.json"
        )));
    }

    let special =
        |name: &str, default: &str| special_token(&config, name).unwrap_or(default).to_owned();
    let (start, end, unknown) = (
        special("bos_token", "<|startoftext|>"),
        special("eos_token", "<|endoftext|>"),
        special("unk_token", "<|endoftext|>"),
    );
    let (vocabulary, merges) = bpe_files(directory)?;
    let id = |token: &str| vocabulary_id(&vocabulary, "vocab.json", token);
    let added = [&start, &end]
        .into_iter()
        .map(|token| added_token(&vocabulary, "vocab.json", token, false))
        .collect::<Result<Vec<_>>>()?;
    let tokenizer = json!({
        "version": "1.0",
        "added_tokens": added,
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "NFC"},
                {"type": "Replace", "pattern": {"Regex": r"\s+"}, "content": " "},
                {"type": "Lowercase"},
            ],
        },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {
                        "Regex": r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
                    },
                    "behavior": "Removed",
                    "invert": true,
                },
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            ],
        },
        "post_processor": {
            "type": "RobertaProcessing",
            "sep": [&end, id(&end)?],
            "cls": [&start, id(&start)?],
            "trim_offsets": false,
            "add_prefix_space": false,
        },
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": unknown,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "</w>",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocabulary,
            "merges": merges,
        },
    });
    Tokenizer::from_str(&tokenizer.to_string()).map_err(|source| Error::Tokenizer {
        what: "build CLIP's BPE tokenizer from vocab.json and merges.txt",
        source,
    })
}

// RoBERTa's byte-level BPE tokenizer, which BART and TAPEX share, built from `vocab.json` and `merges.txt` as `transformers`' `RobertaTokenizer` builds it: bytes as characters, words with their leading space, between `<s>` and `</s>`.
fn roberta_bpe(directory: &Path, config: &Value) -> Result<Tokenizer> {
    let special =
        |name: &str, default: &str| special_token(config, name).unwrap_or(default).to_owned();
    let (vocabulary, merges) = bpe_files(directory)?;
    let id = |token: &str| vocabulary_id(&vocabulary, "vocab.json", token);
    let (start, end, unknown, padding, mask) = (
        special("bos_token", "<s>"),
        special("eos_token", "</s>"),
        special("unk_token", "<unk>"),
        special("pad_token", "<pad>"),
        special("mask_token", "<mask>"),
    );
    let added = [&start, &padding, &end, &unknown, &mask]
        .into_iter()
        .map(|token| added_token(&vocabulary, "vocab.json", token, token == &mask))
        .collect::<Result<Vec<_>>>()?;
    let add_prefix_space = config["add_prefix_space"].as_bool().unwrap_or(false);
    let tokenizer = json!({
        "version": "1.0",
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": add_prefix_space, "trim_offsets": true, "use_regex": true},
        "post_processor": {
            "type": "RobertaProcessing",
            "sep": [&end, id(&end)?],
            "cls": [&start, id(&start)?],
            "trim_offsets": true,
            "add_prefix_space": add_prefix_space,
        },
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocabulary,
            "merges": merges,
        },
    });
    Tokenizer::from_str(&tokenizer.to_string()).map_err(|source| Error::Tokenizer {
        what: "build RoBERTa's BPE tokenizer from vocab.json and merges.txt",
        source,
    })
}

// The text of the special token `name` a tokenizer config gives, as a string or, in newer configs, an object of its `content`.
pub(crate) fn special_token<'a>(config: &'a Value, name: &str) -> Option<&'a str> {
    let value = &config[name];
    value.as_str().or_else(|| value["content"].as_str())
}

// The id `vocabulary`, read from `file`, gives `token`.
fn vocabulary_id(vocabulary: &Map<String, Value>, file: &str, token: &str) -> Result<Value> {
    vocabulary
        .get(token)
        .cloned()
        .ok_or_else(|| Error::UnsupportedConfig(format!("a {file} without {token}")))
}

// The special token `token` as `tokenizer.json` lists an added token, with the id `vocabulary`, read from `file`, gives it.
fn added_token(
    vocabulary: &Map<String, Value>,
    file: &str,
    token: &str,
    lstrip: bool,
) -> Result<Value> {
    Ok(json!({
        "id": vocabulary_id(vocabulary, file, token)?, "content": token, "single_word": false,
        "lstrip": lstrip, "rstrip": false, "normalized": false, "special": true,
    }))
}

// A byte-level BPE snapshot's vocabulary and its merges' pairs.
type BpeFiles = (Map<String, Value>, Vec<Vec<String>>);

// A byte-level BPE snapshot's `vocab.json`, and the merges of its `merges.txt`, whose first line may name the format's version.
fn bpe_files(directory: &Path) -> Result<BpeFiles> {
    let vocabulary = read_optional(&directory.join("vocab.json"))?
        .ok_or_else(|| Error::unsupported("a BPE snapshot without vocab.json"))?;
    let path = directory.join("merges.txt");
    let text = fs::read_to_string(&path).map_err(Error::io(&path))?;
    let merges = text
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with("#version"))
        .map(|line| line.split(' ').map(str::to_owned).collect())
        .collect();
    Ok((vocabulary, merges))
}

fn read_optional<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    if !path.exists() {
        return Ok(None);
    }

    let text = fs::read_to_string(path).map_err(Error::io(path))?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(Error::json("parse a tokenizer file"))
}

// Removes the spaces a word-piece decode leaves before punctuation and contractions, as `clean_up_tokenization` does.
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
    // Loads `tokenizer.json` from a snapshot directory, or builds BERT's WordPiece tokenizer from its `vocab.txt`, CLIP's byte-level BPE from its `vocab.json` and `merges.txt`, or XLM-RoBERTa's from its `sentencepiece.bpe.model`, when it has only those.
    pub fn load(snapshot: impl AsRef<Path>) -> Result<Self> {
        let directory = snapshot.as_ref();
        let path = directory.join("tokenizer.json");
        let mut inner = if !path.exists() && directory.join("vocab.txt").exists() {
            wordpiece(directory)?
        } else if !path.exists() && directory.join("merges.txt").exists() {
            clip_bpe(directory)?
        } else if !path.exists() && directory.join("sentencepiece.bpe.model").exists() {
            let built =
                super::sentencepiece::xlm_roberta(&directory.join("sentencepiece.bpe.model"))?;
            Tokenizer::from_str(&built.to_string()).map_err(|source| Error::Tokenizer {
                what: "build XLM-RoBERTa's tokenizer from sentencepiece.bpe.model",
                source,
            })?
        } else {
            Tokenizer::from_file(&path).map_err(|source| Error::Tokenizer {
                what: "load tokenizer.json",
                source,
            })?
        };
        // A snapshot's padding and truncation settings are what it was trained with, which `transformers` drops on loading and applies only when asked.
        inner.with_padding(None);
        inner
            .with_truncation(None)
            .map_err(|source| Error::Tokenizer {
                what: "clear truncation",
                source,
            })?;
        let mut special = Map::new();
        for file in ["special_tokens_map.json", "tokenizer_config.json"] {
            if let Some(Value::Object(fields)) = read_optional(&directory.join(file))? {
                special.extend(fields);
            }
        }

        let max_length = special
            .get("model_max_length")
            .and_then(Value::as_u64)
            .and_then(|length| usize::try_from(length).ok());
        Ok(Self {
            inner,
            max_length,
            special: Value::Object(special),
        })
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

    // Encodes `texts` with special tokens into a batch for an encoder on `device`, each truncated to `max_length` tokens if given, keeping its special tokens.
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

    // Encodes `inputs` with special tokens into a batch for an encoder on `device`, one row per input and per window it overflows into, padded with id zero at each row's end, with the token types the tokenizer gives.
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

    // Decodes generated ids to text, leaving out special tokens when `skip_special_tokens` is set.
    pub fn decode(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String> {
        let ids = ids
            .iter()
            .map(|&id| u32::try_from(id))
            .collect::<StdResult<Vec<_>, _>>()
            .map_err(|_| Error::unsupported("a token id outside u32"))?;
        let text = self
            .inner
            .decode(&ids, skip_special_tokens)
            .map_err(|source| Error::Tokenizer {
                what: "decode ids",
                source,
            })?;
        // `transformers` cleans up only when the config asks, and never a BPE model's text, unless told to anyway.
        let flag = |name: &str| self.special[name].as_bool().unwrap_or(false);
        let bpe = matches!(self.inner.get_model(), ModelWrapper::BPE(_));
        if flag("clean_up_tokenization_spaces")
            && (!bpe
                || flag("clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output"))
        {
            return Ok(clean_up_tokenization(&text));
        }
        Ok(text)
    }

    // Starts turning generated ids into text one at a time, leaving out special tokens when `skip_special_tokens` is set.
    #[must_use]
    pub fn stream(&self, skip_special_tokens: bool) -> TextStream<'_> {
        TextStream {
            inner: self.inner.decode_stream(skip_special_tokens),
        }
    }

    // Returns the id of a token by its text, such as Whisper's `<|en|>`.
    pub fn token_id(&self, token: &str) -> Result<i64> {
        self.inner
            .token_to_id(token)
            .map(i64::from)
            .ok_or_else(|| Error::UnsupportedConfig(format!("a tokenizer without {token}")))
    }

    // Returns the token of `id` as the vocabulary writes it, such as WordPiece's `##ing`.
    #[must_use]
    pub fn token(&self, id: i64) -> Option<String> {
        u32::try_from(id)
            .ok()
            .and_then(|id| self.inner.id_to_token(id))
    }

    // Joins tokens into text with the tokenizer's decoder, or with spaces without one, as `transformers`' `convert_tokens_to_string` does.
    pub fn join_tokens(&self, tokens: Vec<String>) -> Result<String> {
        match self.inner.get_decoder() {
            Some(decoder) => decoder.decode(tokens).map_err(|source| Error::Tokenizer {
                what: "join tokens",
                source,
            }),
            None => Ok(tokens.join(" ")),
        }
    }

    // The prefix the model marks a word's continuing pieces with, such as WordPiece's `##`.
    #[must_use]
    pub fn subword_prefix(&self) -> Option<&str> {
        match self.inner.get_model() {
            ModelWrapper::WordPiece(model) => Some(model.continuing_subword_prefix.as_str()),
            ModelWrapper::BPE(model) => model.continuing_subword_prefix.as_deref(),
            ModelWrapper::WordLevel(_) | ModelWrapper::Unigram(_) => None,
        }
        .filter(|prefix| !prefix.is_empty())
    }

    // The longest input the model reads, if the snapshot says.
    #[must_use]
    pub const fn max_length(&self) -> Option<usize> {
        self.max_length
    }

    // The id of the special token the tokenizer's configs name for `role`, such as `mask_token`, as a string or an object of its `content`.
    #[must_use]
    pub fn special_id(&self, role: &str) -> Option<i64> {
        let token = special_token(&self.special, role)?;
        self.inner.token_to_id(token).map(i64::from)
    }

    // The id of the mask token: the one the configs name, or the added special token named `mask` in brackets, as `[MASK]` and `<mask>` are.
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

    // The id of the token the post-processor puts before a text, as BERT's `[CLS]` and RoBERTa's `<s>` are, if it puts one.
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

    // The id of the model's unknown token, if it has one.
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

// Generated ids turned into text as they arrive.
//
// An id that ends partway through a character, as byte-level tokenizers split one, gives no text until the ids after it complete the character, so each piece is whole text and the pieces join to what `decode` gives.
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
    // Adds the next id, and returns the text it completes, or `None` while the text it starts is not whole.
    pub fn push(&mut self, id: i64) -> Result<Option<String>> {
        let id = u32::try_from(id).map_err(|_| Error::unsupported("a token id outside u32"))?;
        self.inner.step(id).map_err(|source| Error::Tokenizer {
            what: "decode a streamed id",
            source,
        })
    }
}

// A CTC model's characters, as `Wav2Vec2CTCTokenizer` reads them from `vocab.json`, and how it turns frames' tokens into text: repeats collapsed, the blank padding token dropped, and the word delimiter a space.
#[derive(Debug, Clone)]
pub struct CtcVocabulary {
    tokens: Vec<String>,
    padding: String,
    delimiter: String,
    lowercase: bool,
    clean_up: bool,
}

impl CtcVocabulary {
    pub fn load(snapshot: impl AsRef<Path>) -> Result<Self> {
        let directory = snapshot.as_ref();
        let vocabulary: Map<String, Value> = read_optional(&directory.join("vocab.json"))?
            .ok_or_else(|| Error::unsupported("a CTC snapshot without vocab.json"))?;
        let config =
            read_optional(&directory.join("tokenizer_config.json"))?.unwrap_or(Value::Null);
        let text =
            |name: &str, default: &str| special_token(&config, name).unwrap_or(default).to_owned();
        let mut tokens = Vec::new();
        for (token, id) in &vocabulary {
            let id = id
                .as_u64()
                .and_then(|id| usize::try_from(id).ok())
                .ok_or_else(|| Error::UnsupportedConfig(format!("a vocab.json id for {token}")))?;
            if tokens.len() <= id {
                tokens.resize(id + 1, String::new());
            }

            tokens[id].clone_from(token);
        }
        Ok(Self {
            tokens,
            padding: text("pad_token", "<pad>"),
            delimiter: text("word_delimiter_token", "|"),
            lowercase: config["do_lower_case"].as_bool().unwrap_or(false),
            clean_up: config["clean_up_tokenization_spaces"]
                .as_bool()
                .unwrap_or(false),
        })
    }

    // The text of each frame's best token, as the speech recognition pipeline decodes a CTC model's, keeping special tokens but the blank.
    pub fn decode(&self, ids: &[i64]) -> Result<String> {
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
                .ok_or_else(|| {
                    Error::UnsupportedConfig(format!("token {id} outside vocab.json"))
                })?;
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

// A character tokenizer, as `VitsTokenizer` reads `vocab.json`: text lowercased where its characters are not in the vocabulary, characters still not in it dropped, and a blank, id zero, between every two characters and at both ends.
#[derive(Debug, Clone)]
pub struct CharacterTokenizer {
    vocabulary: HashMap<char, i64>,
    normalize: bool,
    add_blank: bool,
}

impl CharacterTokenizer {
    pub fn load(snapshot: impl AsRef<Path>) -> Result<Self> {
        let directory = snapshot.as_ref();
        let vocabulary: Map<String, Value> = read_optional(&directory.join("vocab.json"))?
            .ok_or_else(|| Error::unsupported("a character snapshot without vocab.json"))?;
        let config =
            read_optional(&directory.join("tokenizer_config.json"))?.unwrap_or(Value::Null);
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
