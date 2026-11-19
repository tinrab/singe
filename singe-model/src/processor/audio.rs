// Clips of audio as models hear them: mono samples in [-1, 1) at a known rate, decoded from WAV files.
// Each model's feature extractor, such as Whisper's log-mel front end, lives with its model, as `transformers` keeps them.

use std::{fs, path::Path, result::Result as StdResult};

use serde::{Deserialize, Deserializer, de::Error as DeError};
use singe::{Device, Error as SingeError, Shape, Tensor};

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub rate: u32,
}

impl Audio {
    // Reads a 16-bit PCM WAV file, averaging its channels into one.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let invalid = |what: &str| Error::Media(format!("{}: {what}", path.display()));
        let bytes = fs::read(path).map_err(Error::io(path))?;
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(invalid("not a WAV file"));
        }

        let (mut format, mut data) = (None, None);
        let mut offset = 12;
        while offset + 8 <= bytes.len() {
            let id = &bytes[offset..offset + 4];
            let size = usize::try_from(u32::from_le_bytes([
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]))
            .map_err(|_| invalid("a chunk too large"))?;
            let body = &bytes[offset + 8..(offset + 8 + size).min(bytes.len())];
            match id {
                b"fmt " => format = Some(body),
                b"data" => data = Some(body),
                _ => {}
            }

            offset += 8 + size + size % 2;
        }

        let format = format
            .filter(|format| format.len() >= 16)
            .ok_or_else(|| invalid("no format chunk"))?;
        let field = |at: usize| u16::from_le_bytes([format[at], format[at + 1]]);
        let rate = u32::from_le_bytes([format[4], format[5], format[6], format[7]]);
        let channels = usize::from(field(2));
        if field(0) != 1 || field(14) != 16 || channels == 0 {
            return Err(invalid("not 16-bit PCM"));
        }

        let values = data
            .ok_or_else(|| invalid("no data chunk"))?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from(i16::from_le_bytes(*pair)) / 32_768.0)
            .collect::<Vec<_>>();
        let samples = values
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
        Ok(Self { samples, rate })
    }

    pub fn to_tensor(&self, device: Device) -> Result<Tensor> {
        let count = i64::try_from(self.samples.len())
            .map_err(|_| SingeError::out_of_range("clip samples"))?;
        Ok(Tensor::from_vec(
            device,
            Shape::create([count])?,
            self.samples.clone(),
        )?)
    }

    // Refuses a clip at another rate than `rate`, since no pipeline resamples yet.
    pub fn expect_rate(&self, rate: i64) -> Result<()> {
        if i64::from(self.rate) == rate {
            Ok(())
        } else {
            Err(Error::Media(format!(
                "a clip at {} Hz for a model that hears {rate} Hz",
                self.rate
            )))
        }
    }
}

// A clip in JSON names the file it is read from.
impl<'de> Deserialize<'de> for Audio {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> StdResult<Self, D::Error> {
        let path = String::deserialize(deserializer)?;
        Self::open(&path).map_err(DeError::custom)
    }
}
