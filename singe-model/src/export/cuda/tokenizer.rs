use std::collections::HashMap;

use regex_syntax::hir::{Class, HirKind};
use serde_json::Value;
use singe::backend::cuda::export::{
    CharacterClassTable, Composition, LoopEncoder, LoopToken, LoopTokenizer,
};
use unicode_normalization_alignments::char::{
    canonical_combining_class, compose, decompose_canonical,
};

use crate::{
    error::{Error, Result},
    source::Source,
};

const QWEN3_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN35_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const CL100K_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplate {
    pub before: String,
    pub after: String,
    pub placeholder: Option<i64>,
}

fn byte_alphabet() -> HashMap<char, u8> {
    let mut shifted = 0;
    (0..=u8::MAX)
        .map(|byte| {
            let character = if matches!(byte, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff) {
                char::from(byte)
            } else {
                shifted += 1;
                char::from_u32(255 + shifted).unwrap_or(char::REPLACEMENT_CHARACTER)
            };
            (character, byte)
        })
        .collect()
}

fn id_of(vocabulary: &serde_json::Map<String, Value>, token: &str) -> Result<i64> {
    vocabulary
        .get(token)
        .and_then(Value::as_i64)
        .ok_or_else(|| Error::unsupported("a merge of tokens outside the vocabulary"))
}

fn class_ranges(pattern: &str) -> Result<Vec<[u32; 2]>> {
    let hir = regex_syntax::parse(pattern).map_err(|_| Error::unsupported("a character class"))?;
    match hir.kind() {
        HirKind::Class(Class::Unicode(class)) => Ok(class
            .ranges()
            .iter()
            .map(|range| [u32::from(range.start()), u32::from(range.end())])
            .collect()),
        _ => Err(Error::unsupported("a character class")),
    }
}

fn composition() -> Composition {
    let mut decompositions = Vec::new();
    let mut combining_classes: Vec<[u32; 3]> = Vec::new();
    let mut pairs = Vec::new();
    for character in (0..=u32::from(char::MAX)).filter_map(char::from_u32) {
        let point = u32::from(character);
        let class = u32::from(canonical_combining_class(character));
        if class != 0 {
            match combining_classes.last_mut() {
                Some([_, end, last]) if *end + 1 == point && *last == class => *end = point,
                _ => combining_classes.push([point, point, class]),
            }
        }

        let mut parts = Vec::new();
        decompose_canonical(character, |part| parts.push(part));
        if parts == [character] {
            continue;
        }

        // The pair composition reaches this character from: its decomposition but the last code point, composed, and the last.
        if let Some((&last, [first, rest @ ..])) = parts.split_last()
            && let Some(first) = rest
                .iter()
                .try_fold(*first, |first, &next| compose(first, next))
            && compose(first, last) == Some(character)
        {
            pairs.push([u32::from(first), u32::from(last), point]);
        }

        decompositions.push((point, parts.into_iter().map(u32::from).collect()));
    }
    Composition {
        decompositions,
        combining_classes,
        pairs,
    }
}

fn encoder(
    json: &Value,
    vocabulary: &serde_json::Map<String, Value>,
    added: Vec<i64>,
    template: PromptTemplate,
) -> Result<Option<LoopEncoder>> {
    let composes = match &json["normalizer"] {
        Value::Null => false,
        normalizer if normalizer["type"] == "NFC" => true,
        _ => return Ok(None),
    };
    let steps = &json["pre_tokenizer"]["pretokenizers"];
    let (split, bytes) = (&steps[0], &steps[1]);
    let (marks_are_letters, max_digits) = match split["pattern"]["Regex"].as_str() {
        Some(QWEN3_SPLIT) => (false, 1),
        Some(QWEN35_SPLIT) => (true, 1),
        Some(CL100K_SPLIT) => (false, 3),
        _ => return Ok(None),
    };
    // Every character matches the pattern, so keeping its matches and removing what lies between them splits as isolating them does.
    let isolates = (split["behavior"] == "Isolated" && split["invert"] == false)
        || (split["behavior"] == "Removed" && split["invert"] == true);
    if json["pre_tokenizer"]["type"] != "Sequence"
        || steps.as_array().map(Vec::len) != Some(2)
        || split["type"] != "Split"
        || !isolates
        || bytes["type"] != "ByteLevel"
        || bytes["use_regex"] != false
        || bytes["add_prefix_space"] != false
    {
        return Ok(None);
    }

    let merges = json["model"]["merges"]
        .as_array()
        .ok_or_else(|| Error::unsupported("a BPE model without merges"))?
        .iter()
        .map(|merge| {
            let (left, right) = match merge {
                Value::String(merge) => merge
                    .split_once(' ')
                    .ok_or_else(|| Error::unsupported("a merge that is not a pair"))?,
                Value::Array(pair) => match pair.as_slice() {
                    [Value::String(left), Value::String(right)] => (left.as_str(), right.as_str()),
                    _ => return Err(Error::unsupported("a merge that is not a pair")),
                },
                _ => return Err(Error::unsupported("a merge that is not a pair")),
            };
            Ok([
                id_of(vocabulary, left)?,
                id_of(vocabulary, right)?,
                id_of(vocabulary, &format!("{left}{right}"))?,
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(LoopEncoder {
        template: [template.before, template.after],
        placeholder: template.placeholder,
        added,
        merges,
        classes: CharacterClassTable {
            letters: class_ranges(r"\p{L}")?,
            marks: class_ranges(r"\p{M}")?,
            numbers: class_ranges(r"\p{N}")?,
            whitespace: class_ranges(r"\s")?,
        },
        marks_are_letters,
        max_digits,
        composition: composes.then(composition),
    }))
}

pub fn loop_tokenizer(
    source: &Source,
    template: Option<PromptTemplate>,
) -> Result<Option<LoopTokenizer>> {
    let json: Value = source.json("tokenizer.json")?;
    let model = &json["model"];
    let affixes = ["continuing_subword_prefix", "end_of_word_suffix"]
        .iter()
        .all(|affix| model[affix].is_null() || model[affix] == "");
    if model["type"] != "BPE"
        || model["byte_fallback"] == true
        || json["decoder"]["type"] != "ByteLevel"
        || !affixes
    {
        return Ok(None);
    }

    let vocabulary = model["vocab"]
        .as_object()
        .ok_or_else(|| Error::unsupported("a BPE model without a vocabulary"))?;
    let alphabet = byte_alphabet();
    let mut tokens = Vec::new();
    let mut place = |id: Option<i64>, token: LoopToken| {
        let index = id
            .and_then(|id| usize::try_from(id).ok())
            .ok_or_else(|| Error::unsupported("a token without an id"))?;
        if tokens.len() <= index {
            tokens.resize(
                index + 1,
                LoopToken {
                    bytes: Vec::new(),
                    special: true,
                },
            );
        }

        tokens[index] = token;
        Result::Ok(())
    };
    for (text, id) in vocabulary {
        let bytes = text
            .chars()
            .map(|character| alphabet.get(&character).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::unsupported("a byte-level token outside its alphabet"))?;
        place(
            id.as_i64(),
            LoopToken {
                bytes,
                special: false,
            },
        )?;
    }

    let mut added = Vec::new();
    for token in json["added_tokens"].as_array().into_iter().flatten() {
        let content = token["content"]
            .as_str()
            .ok_or_else(|| Error::unsupported("an added token without content"))?;
        place(
            token["id"].as_i64(),
            LoopToken {
                bytes: content.as_bytes().to_vec(),
                special: token["special"] == true,
            },
        )?;
        added.extend(token["id"].as_i64());
    }

    // A tokenizer whose normalizer or pre-tokenizer the program cannot apply leaves it without an encoder, so it reads prompts as tokens.
    let encoder = match template {
        Some(template) => encoder(&json, vocabulary, added, template)?,
        None => None,
    };
    Ok(Some(LoopTokenizer { tokens, encoder }))
}
