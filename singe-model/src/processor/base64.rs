//! Base64, written standard with padding, and the `data:` URLs media travels in through JSON.

use std::{fs, path::Path};

use serde_json::Value;

use crate::error::{Error, Result};

#[must_use]
pub fn encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let word = chunk
            .iter()
            .enumerate()
            .fold(0_u32, |word, (index, &byte)| {
                word | u32::from(byte) << (16 - 8 * index)
            });
        for index in 0..4 {
            if index <= chunk.len() {
                let sextet = (word >> (18 - 6 * index)) & 0x3f;
                text.push(char::from(ALPHABET[sextet as usize]));
            } else {
                text.push('=');
            }
        }
    }
    text
}

/// Decodes standard or URL-safe base64, with or without padding, skipping whitespace and stopping at the first `=`.
/// Returns `None` when `text` holds any other character.
#[must_use]
pub fn decode(text: &str) -> Option<Vec<u8>> {
    let mut bytes = Vec::with_capacity(text.len() / 4 * 3);
    let (mut word, mut bits) = (0_u32, 0_u32);
    for character in text.bytes() {
        let sextet = match character {
            b'A'..=b'Z' => character - b'A',
            b'a'..=b'z' => character - b'a' + 26,
            b'0'..=b'9' => character - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        };
        word = word << 6 | u32::from(sextet);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push(u8::try_from(word >> bits & 0xff).ok()?);
        }
    }
    Some(bytes)
}

/// Returns the bytes `text` carries: a `data:` URL's base64 payload, or bare base64.
/// Returns `None` for a `data:` URL that is not base64, or for text [`decode`] rejects.
#[must_use]
pub fn data_bytes(text: &str) -> Option<Vec<u8>> {
    let payload = match text.strip_prefix("data:") {
        Some(url) => {
            let (header, payload) = url.split_once(',')?;
            if !header.ends_with(";base64") {
                return None;
            }
            payload
        }
        None => text,
    };
    decode(payload)
}

/// Returns `bytes` as a base64 `data:` URL of media type `media_type`.
#[must_use]
pub fn data_url(media_type: &str, bytes: &[u8]) -> String {
    format!("data:{media_type};base64,{}", encode(bytes))
}

/// Returns a file as JSON carries media: a `data:` URL of its bytes, or for a directory, an array of one for each `png` or `jpg` image in it, in name order, as a clip's frames.
///
/// # Errors
///
/// Returns an error if the file or the directory cannot be read.
pub fn file_value(path: &Path) -> Result<Value> {
    if !path.is_dir() {
        let bytes = fs::read(path).map_err(Error::io(path))?;
        return Ok(Value::String(data_url("application/octet-stream", &bytes)));
    }

    let mut paths = fs::read_dir(path)
        .map_err(Error::io(path))?
        .map(|entry| entry.map(|entry| entry.path()).map_err(Error::io(path)))
        .collect::<Result<Vec<_>>>()?;
    paths.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "png" || extension == "jpg")
    });
    paths.sort();
    paths
        .iter()
        .map(|path| file_value(path))
        .collect::<Result<Vec<_>>>()
        .map(Value::Array)
}
