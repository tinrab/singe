// The tokenizers `transformers` converts a snapshot's slow-tokenizer files to when it has no `tokenizer.json`: BERT's WordPiece from `vocab.txt`, CLIP's and RoBERTa's byte-level BPE from `vocab.json` and `merges.txt`, and XLM-RoBERTa's Unigram from `sentencepiece.bpe.model`.

use std::{fs, str::FromStr};

use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use crate::{
    error::{Error, Result},
    processor::{base64, sentencepiece, text::special_token},
    source::Source,
};

pub(crate) fn convert(source: &Source) -> Result<Tokenizer> {
    if source.has("vocab.txt") {
        wordpiece(source)
    } else if source.has("merges.txt") {
        byte_level_bpe(source)
    } else if source.has("sentencepiece.bpe.model") {
        xlm_roberta(source)
    } else {
        Err(Error::unsupported(
            "a snapshot without tokenizer.json, vocab.txt, merges.txt, or sentencepiece.bpe.model",
        ))
    }
}

pub(crate) fn tokenizer_config(source: &Source) -> Result<Value> {
    Ok(source
        .optional_json("tokenizer_config.json")?
        .unwrap_or(Value::Null))
}

pub(crate) fn special_or(config: &Value, name: &str, default: &str) -> String {
    special_token(config, name).unwrap_or(default).to_owned()
}

fn build(tokenizer: &Value, what: &'static str) -> Result<Tokenizer> {
    Tokenizer::from_str(&tokenizer.to_string()).map_err(|source| Error::Tokenizer { what, source })
}

// The tokenizer `transformers`' `BertConverter` makes of a slow `BertTokenizer`: `vocab.txt`'s WordPiece vocabulary, with the casing, accents, and special tokens `tokenizer_config.json` gives.
fn wordpiece(source: &Source) -> Result<Tokenizer> {
    let config = tokenizer_config(source)?;
    let flag = |name: &str| config[name].as_bool().unwrap_or(true);
    let (unknown, separator, padding, classifier, mask) = (
        special_or(&config, "unk_token", "[UNK]"),
        special_or(&config, "sep_token", "[SEP]"),
        special_or(&config, "pad_token", "[PAD]"),
        special_or(&config, "cls_token", "[CLS]"),
        special_or(&config, "mask_token", "[MASK]"),
    );

    let path = source.directory().join("vocab.txt");
    let text = fs::read_to_string(&path).map_err(Error::io(&path))?;
    let mut vocabulary = Map::new();
    // A token listed twice takes its later id, as `transformers`' `load_vocab` reads it.
    for (id, token) in text.lines().enumerate() {
        vocabulary.insert(token.to_owned(), json!(id));
    }

    let id = |token: &str| vocabulary_id(&vocabulary, "vocab.txt", token);
    let added = [&padding, &unknown, &classifier, &separator, &mask]
        .into_iter()
        .map(|token| Ok(added_token(id(token)?, token, false)))
        .collect::<Result<Vec<_>>>()?;
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
            "single": [
                special_piece(&classifier, 0), sequence_piece("A", 0), special_piece(&separator, 0),
            ],
            "pair": [
                special_piece(&classifier, 0), sequence_piece("A", 0), special_piece(&separator, 0),
                sequence_piece("B", 1), special_piece(&separator, 1),
            ],
            "special_tokens": {
                &classifier: template_token(&classifier, id(&classifier)?),
                &separator: template_token(&separator, id(&separator)?),
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
    build(&tokenizer, "build a WordPiece tokenizer from vocab.txt")
}

fn byte_level_bpe(source: &Source) -> Result<Tokenizer> {
    let config = tokenizer_config(source)?;
    let class = config["tokenizer_class"].as_str().unwrap_or_default();
    if ["Bart", "Roberta", "Tapex"]
        .iter()
        .any(|prefix| class.starts_with(prefix))
    {
        return roberta_bpe(source, &config);
    }

    if !class.starts_with("CLIP") {
        return Err(Error::unsupported(format!(
            "a {class} snapshot with merges.txt but no tokenizer.json"
        )));
    }

    clip_bpe(source, &config)
}

// CLIP's byte-level BPE tokenizer, built from `vocab.json` and `merges.txt` as `transformers`' `CLIPTokenizer` builds it: text normalized, lowercased, and split into words, each word's last piece marked with `</w>`, between `<|startoftext|>` and `<|endoftext|>`.
fn clip_bpe(source: &Source, config: &Value) -> Result<Tokenizer> {
    let (start, end, unknown) = (
        special_or(config, "bos_token", "<|startoftext|>"),
        special_or(config, "eos_token", "<|endoftext|>"),
        special_or(config, "unk_token", "<|endoftext|>"),
    );
    let (vocabulary, merges) = bpe_files(source)?;
    let id = |token: &str| vocabulary_id(&vocabulary, "vocab.json", token);
    let added = [&start, &end]
        .into_iter()
        .map(|token| Ok(added_token(id(token)?, token, false)))
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
        "model": bpe_model(Some(unknown.as_str()), "</w>", &vocabulary, &merges),
    });
    build(
        &tokenizer,
        "build CLIP's BPE tokenizer from vocab.json and merges.txt",
    )
}

// RoBERTa's byte-level BPE tokenizer, which BART and TAPEX share, built from `vocab.json` and `merges.txt` as `transformers`' `RobertaTokenizer` builds it: bytes as characters, words with their leading space, between `<s>` and `</s>`.
fn roberta_bpe(source: &Source, config: &Value) -> Result<Tokenizer> {
    let (vocabulary, merges) = bpe_files(source)?;
    let id = |token: &str| vocabulary_id(&vocabulary, "vocab.json", token);
    let (start, end, unknown, padding, mask) = (
        special_or(config, "bos_token", "<s>"),
        special_or(config, "eos_token", "</s>"),
        special_or(config, "unk_token", "<unk>"),
        special_or(config, "pad_token", "<pad>"),
        special_or(config, "mask_token", "<mask>"),
    );
    let added = [&start, &padding, &end, &unknown, &mask]
        .into_iter()
        .map(|token| Ok(added_token(id(token)?, token, token == &mask)))
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
        "model": bpe_model(None, "", &vocabulary, &merges),
    });
    build(
        &tokenizer,
        "build RoBERTa's BPE tokenizer from vocab.json and merges.txt",
    )
}

// XLM-RoBERTa's tokenizer as `transformers`' converter builds it from `sentencepiece.bpe.model`: fairseq's four special tokens first, the model's pieces but its own first three, and `<mask>` last, under a Unigram model with the model's normalizer and `▁` for spaces.
fn xlm_roberta(source: &Source) -> Result<Tokenizer> {
    let model = sentencepiece::read(&source.directory().join("sentencepiece.bpe.model"))?;
    let mut vocabulary = ["<s>", "<pad>", "</s>", "<unk>"]
        .iter()
        .map(|token| json!([token, 0.0]))
        .collect::<Vec<_>>();
    vocabulary.extend(
        model
            .pieces
            .iter()
            .skip(3)
            .map(|(piece, score)| json!([piece, score])),
    );
    vocabulary.push(json!(["<mask>", 0.0]));
    let mask = vocabulary.len() - 1;
    let mut normalizers = vec![
        json!({"type": "Replace", "pattern": {"String": "``"}, "content": "\""}),
        json!({"type": "Replace", "pattern": {"String": "''"}, "content": "\""}),
    ];
    normalizers.push(if model.charsmap.is_empty() {
        json!({"type": "NFKC"})
    } else {
        json!({"type": "Precompiled", "precompiled_charsmap": base64::encode(&model.charsmap)})
    });
    normalizers.push(json!({"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}));
    let metaspace =
        json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true});
    let tokenizer = json!({
        "version": "1.0",
        "added_tokens": [
            added_token(0, "<s>", false), added_token(1, "<pad>", false),
            added_token(2, "</s>", false), added_token(3, "<unk>", false),
            added_token(mask, "<mask>", true),
        ],
        "normalizer": {"type": "Sequence", "normalizers": normalizers},
        "pre_tokenizer": metaspace,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [special_piece("<s>", 0), sequence_piece("A", 0), special_piece("</s>", 0)],
            "pair": [
                special_piece("<s>", 0), sequence_piece("A", 0), special_piece("</s>", 0),
                special_piece("</s>", 0), sequence_piece("B", 0), special_piece("</s>", 0),
            ],
            "special_tokens": {
                "<s>": template_token("<s>", 0),
                "</s>": template_token("</s>", 2),
            },
        },
        "decoder": metaspace,
        "model": {"type": "Unigram", "unk_id": 3, "vocab": vocabulary, "byte_fallback": false},
    });
    build(
        &tokenizer,
        "build XLM-RoBERTa's tokenizer from sentencepiece.bpe.model",
    )
}

// The id `vocabulary`, read from `file`, gives `token`.
fn vocabulary_id(vocabulary: &Map<String, Value>, file: &str, token: &str) -> Result<usize> {
    vocabulary
        .get(token)
        .and_then(Value::as_u64)
        .and_then(|id| usize::try_from(id).ok())
        .ok_or_else(|| Error::unsupported(format!("a {file} without {token}")))
}

// The special token `token` as `tokenizer.json` lists an added token.
fn added_token(id: usize, token: &str, lstrip: bool) -> Value {
    json!({
        "id": id, "content": token, "single_word": false,
        "lstrip": lstrip, "rstrip": false, "normalized": false, "special": true,
    })
}

fn special_piece(id: &str, type_id: u32) -> Value {
    json!({"SpecialToken": {"id": id, "type_id": type_id}})
}

fn sequence_piece(id: &str, type_id: u32) -> Value {
    json!({"Sequence": {"id": id, "type_id": type_id}})
}

fn template_token(token: &str, id: usize) -> Value {
    json!({"id": token, "ids": [id], "tokens": [token]})
}

fn bpe_model(
    unknown: Option<&str>,
    end_of_word_suffix: &str,
    vocabulary: &Map<String, Value>,
    merges: &[Vec<String>],
) -> Value {
    json!({
        "type": "BPE",
        "dropout": null,
        "unk_token": unknown,
        "continuing_subword_prefix": "",
        "end_of_word_suffix": end_of_word_suffix,
        "fuse_unk": false,
        "byte_fallback": false,
        "ignore_merges": false,
        "vocab": vocabulary,
        "merges": merges,
    })
}

// A byte-level BPE snapshot's vocabulary and its merges' pairs.
type BpeFiles = (Map<String, Value>, Vec<Vec<String>>);

// A byte-level BPE snapshot's `vocab.json`, and the merges of its `merges.txt`, whose first line may name the format's version.
fn bpe_files(source: &Source) -> Result<BpeFiles> {
    let vocabulary = source
        .optional_json("vocab.json")?
        .ok_or_else(|| Error::unsupported("a BPE snapshot without vocab.json"))?;
    let path = source.directory().join("merges.txt");
    let text = fs::read_to_string(&path).map_err(Error::io(&path))?;
    let merges = text
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with("#version"))
        .map(|line| line.split(' ').map(str::to_owned).collect())
        .collect();
    Ok((vocabulary, merges))
}
