// SentencePiece models, read from their protobuf files for the tokenizers `transformers` converts from them when a snapshot has no `tokenizer.json`: XLM-RoBERTa's, which TrOCR's snapshots name.

use std::{fs, path::Path};

use serde_json::{Value, json};

use super::image::base64;
use crate::error::{Error, Result};

// A protobuf field's value, by its wire type.
enum Field<'a> {
    Varint,
    Fixed32([u8; 4]),
    Bytes(&'a [u8]),
}

fn varint(bytes: &[u8], at: &mut usize) -> Result<u64> {
    let mut value = 0_u64;
    for shift in (0..64).step_by(7) {
        let byte = *bytes
            .get(*at)
            .ok_or_else(|| Error::unsupported("a truncated SentencePiece model"))?;
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(Error::unsupported(
        "an overlong varint in a SentencePiece model",
    ))
}

// A message's fields, each its number and value, in order.
fn fields(bytes: &[u8]) -> Result<Vec<(u64, Field<'_>)>> {
    let truncated = || Error::unsupported("a truncated SentencePiece model");
    let mut at = 0;
    let mut found = Vec::new();
    while at < bytes.len() {
        let key = varint(bytes, &mut at)?;
        let value = match key & 7 {
            0 => {
                varint(bytes, &mut at)?;
                Field::Varint
            }
            1 => {
                at += 8;
                Field::Varint
            }
            2 => {
                let length = usize::try_from(varint(bytes, &mut at)?).map_err(|_| truncated())?;
                let value = bytes.get(at..at + length).ok_or_else(truncated)?;
                at += length;
                Field::Bytes(value)
            }
            5 => {
                let value = bytes.get(at..at + 4).ok_or_else(truncated)?;
                at += 4;
                Field::Fixed32([value[0], value[1], value[2], value[3]])
            }
            other => {
                return Err(Error::UnsupportedConfig(format!(
                    "wire type {other} in a SentencePiece model"
                )));
            }
        };
        found.push((key >> 3, value));
    }
    Ok(found)
}

// A model's pieces with their scores, and its normalizer's precompiled character map.
struct Model {
    pieces: Vec<(String, f32)>,
    charsmap: Vec<u8>,
}

fn read(path: &Path) -> Result<Model> {
    let bytes = fs::read(path).map_err(Error::io(path))?;
    let mut model = Model {
        pieces: Vec::new(),
        charsmap: Vec::new(),
    };
    for (number, value) in fields(&bytes)? {
        match (number, value) {
            (1, Field::Bytes(piece)) => {
                let (mut text, mut score) = (String::new(), 0.0);
                for (number, value) in fields(piece)? {
                    match (number, value) {
                        (1, Field::Bytes(bytes)) => {
                            text = String::from_utf8_lossy(bytes).into_owned();
                        }
                        (2, Field::Fixed32(bytes)) => score = f32::from_le_bytes(bytes),
                        _ => {}
                    }
                }

                model.pieces.push((text, score));
            }
            (3, Field::Bytes(normalizer)) => {
                for (number, value) in fields(normalizer)? {
                    if let (2, Field::Bytes(bytes)) = (number, value) {
                        model.charsmap = bytes.to_vec();
                    }
                }
            }
            _ => {}
        }
    }
    Ok(model)
}

// XLM-RoBERTa's tokenizer as `transformers`' converter builds it from `sentencepiece.bpe.model`: fairseq's four special tokens first, the model's pieces but its own first three, and `<mask>` last, under a Unigram model with the model's normalizer and `▁` for spaces.
pub fn xlm_roberta(path: &Path) -> Result<Value> {
    let model = read(path)?;
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
    let special = |id: usize, content: &str| {
        json!({
            "id": id, "content": content, "single_word": false, "lstrip": content == "<mask>",
            "rstrip": false, "normalized": false, "special": true,
        })
    };
    let mut normalizers = vec![
        json!({"type": "Replace", "pattern": {"String": "``"}, "content": "\""}),
        json!({"type": "Replace", "pattern": {"String": "''"}, "content": "\""}),
    ];
    normalizers.push(if model.charsmap.is_empty() {
        json!({"type": "NFKC"})
    } else {
        json!({"type": "Precompiled", "precompiled_charsmap": base64(&model.charsmap)})
    });
    normalizers.push(json!({"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}));
    let metaspace =
        json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true});
    let piece = |id: &str| json!({"SpecialToken": {"id": id, "type_id": 0}});
    let sequence = |id: &str| json!({"Sequence": {"id": id, "type_id": 0}});
    Ok(json!({
        "version": "1.0",
        "added_tokens": [special(0, "<s>"), special(1, "<pad>"), special(2, "</s>"), special(3, "<unk>"), special(mask, "<mask>")],
        "normalizer": {"type": "Sequence", "normalizers": normalizers},
        "pre_tokenizer": metaspace,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [piece("<s>"), sequence("A"), piece("</s>")],
            "pair": [piece("<s>"), sequence("A"), piece("</s>"), piece("</s>"), sequence("B"), piece("</s>")],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]},
                "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]},
            },
        },
        "decoder": metaspace,
        "model": {"type": "Unigram", "unk_id": 3, "vocab": vocabulary, "byte_fallback": false},
    }))
}
