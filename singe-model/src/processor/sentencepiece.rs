//! SentencePiece models, read from their protobuf files for the tokenizers `transformers` converts from them when a snapshot has no `tokenizer.json`: `XLM-RoBERTa`'s, which `TrOCR`'s snapshots name, and `BigBird`'s.

use std::{fs, path::Path};

use crate::error::{Error, Result};

/// A protobuf field's value, by its wire type.
enum Field<'a> {
    Varint(u64),
    Fixed64,
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

/// Returns a message's fields, each its number and value, in order.
fn fields(bytes: &[u8]) -> Result<Vec<(u64, Field<'_>)>> {
    let truncated = || Error::unsupported("a truncated SentencePiece model");
    let mut at = 0;
    let mut found = Vec::new();
    while at < bytes.len() {
        let key = varint(bytes, &mut at)?;
        let value = match key & 7 {
            0 => Field::Varint(varint(bytes, &mut at)?),
            1 => {
                at += 8;
                Field::Fixed64
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
                return Err(Error::unsupported(format!(
                    "wire type {other} in a SentencePiece model"
                )));
            }
        };
        found.push((key >> 3, value));
    }
    Ok(found)
}

/// A model's piece: its text, its score, and whether it is a normal piece rather than a control, unknown, or user-defined one.
pub(crate) struct Piece {
    pub(crate) text: String,
    pub(crate) score: f32,
    pub(crate) is_normal: bool,
}

/// A model's pieces, and its normalizer's precompiled character map.
pub(crate) struct Model {
    pub(crate) pieces: Vec<Piece>,
    pub(crate) charsmap: Vec<u8>,
}

/// The type of a normal piece, which a piece without a type has.
const NORMAL_PIECE: u64 = 1;

pub(crate) fn read(path: &Path) -> Result<Model> {
    let bytes = fs::read(path).map_err(Error::io(path))?;
    let mut model = Model {
        pieces: Vec::new(),
        charsmap: Vec::new(),
    };
    for (number, value) in fields(&bytes)? {
        match (number, value) {
            (1, Field::Bytes(piece)) => {
                let (mut text, mut score, mut kind) = (String::new(), 0.0, NORMAL_PIECE);
                for (number, value) in fields(piece)? {
                    match (number, value) {
                        (1, Field::Bytes(bytes)) => {
                            text = String::from_utf8_lossy(bytes).into_owned();
                        }
                        (2, Field::Fixed32(bytes)) => score = f32::from_le_bytes(bytes),
                        (3, Field::Varint(value)) => kind = value,
                        _ => {}
                    }
                }

                model.pieces.push(Piece {
                    text,
                    score,
                    is_normal: kind == NORMAL_PIECE,
                });
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
