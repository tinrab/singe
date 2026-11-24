// Clips of audio as models hear them: mono samples in [-1, 1) at a known rate, decoded from WAV files.
// Each model's feature extractor, such as Whisper's log-mel front end, lives with its model, as `transformers` keeps them.

use std::{borrow::Cow, f64::consts::PI, fs, path::Path, result::Result as StdResult};

use serde::{Deserialize, Deserializer, de::Error as DeError};
use singe::{Device, Error as SingeError, Shape, Tensor};

use crate::{
    error::{Error, Result},
    processor::base64,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub rate: u32,
}

// The zero crossings of the windowed sinc on each side of a resampled point.
const RESAMPLING_TAPS: f64 = 16.0;

impl Audio {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(Error::io(path))?;
        Self::decode(&bytes).map_err(|error| match error {
            Error::Media(message) => Error::Media(format!("{}: {message}", path.display())),
            other => other,
        })
    }

    // Decodes a WAV file of 16-bit or 24-bit PCM or 32-bit floats, averaging its channels into one.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let invalid = |what: &str| Error::Media(format!("a WAV file with {what}"));
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(Error::Media("audio that is not a WAV file".to_owned()));
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
        // An extensible format names its encoding in its subformat's first two bytes.
        let encoding = if field(0) == 0xfffe && format.len() >= 26 {
            field(24)
        } else {
            field(0)
        };
        let data = data.ok_or_else(|| invalid("no data chunk"))?;
        let values = match (encoding, field(14)) {
            (1, 16) => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| f32::from(i16::from_le_bytes(*pair)) / 32_768.0)
                .collect::<Vec<_>>(),
            (1, 24) => data
                .as_chunks::<3>()
                .0
                .iter()
                .map(|&[low, middle, high]| {
                    i32::from_le_bytes([0, low, middle, high]) as f32 / 2_147_483_648.0
                })
                .collect(),
            (3, 32) => data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect(),
            _ => {
                return Err(invalid(
                    "samples other than 16-bit or 24-bit PCM or 32-bit floats",
                ));
            }
        };
        if channels == 0 || rate == 0 {
            return Err(invalid("no channels or no rate"));
        }

        let samples = values
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
        Ok(Self { samples, rate })
    }

    // Encodes the clip as a WAV file of 16-bit PCM.
    #[must_use]
    pub fn to_wav(&self) -> Vec<u8> {
        let data = self.to_pcm();
        let length = |count: usize| u32::try_from(count).unwrap_or(u32::MAX).to_le_bytes();
        let mut file = Vec::with_capacity(44 + data.len());
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&length(36 + data.len()));
        file.extend_from_slice(b"WAVEfmt ");
        file.extend_from_slice(&16_u32.to_le_bytes());
        file.extend_from_slice(&1_u16.to_le_bytes());
        file.extend_from_slice(&1_u16.to_le_bytes());
        file.extend_from_slice(&self.rate.to_le_bytes());
        file.extend_from_slice(&(self.rate * 2).to_le_bytes());
        file.extend_from_slice(&2_u16.to_le_bytes());
        file.extend_from_slice(&16_u16.to_le_bytes());
        file.extend_from_slice(b"data");
        file.extend_from_slice(&length(data.len()));
        file.extend_from_slice(&data);
        file
    }

    // The samples as 16-bit little-endian PCM, clipped to its range.
    #[must_use]
    pub fn to_pcm(&self) -> Vec<u8> {
        self.samples
            .iter()
            .flat_map(|&sample| {
                // The clamp keeps the product within `i16`, so the cast cannot wrap.
                ((sample.clamp(-1.0, 1.0) * 32_767.0).round() as i16).to_le_bytes()
            })
            .collect()
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

    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.rate)
    }

    // The clip at `rate`: itself when it is at that rate already, or else resampled.
    pub fn at_rate(&self, rate: i64) -> Result<Cow<'_, Self>> {
        let rate = u32::try_from(rate)
            .ok()
            .filter(|&rate| rate > 0)
            .ok_or_else(|| Error::Media(format!("a sample rate of {rate} Hz")))?;
        if rate == self.rate {
            return Ok(Cow::Borrowed(self));
        }
        Ok(Cow::Owned(self.resampled(rate)))
    }

    // Resamples the clip to `rate` with a Hann-windowed sinc, low-passed below the lower rate's Nyquist frequency.
    #[must_use]
    pub fn resampled(&self, rate: u32) -> Self {
        let ratio = f64::from(rate) / f64::from(self.rate);
        let cutoff = ratio.min(1.0);
        let reach = RESAMPLING_TAPS / cutoff;
        let length = (self.samples.len() as f64 * ratio).round() as usize;
        let last = i64::try_from(self.samples.len()).unwrap_or(i64::MAX) - 1;
        let samples = (0..length)
            .map(|index| {
                let center = index as f64 / ratio;
                let first = (center - reach).ceil().max(0.0) as i64;
                let end = ((center + reach).floor() as i64).min(last);
                let (mut sum, mut weights) = (0.0, 0.0);
                for source in first..=end {
                    let distance = center - source as f64;
                    let x = distance * cutoff;
                    let sinc = if x.abs() < 1e-9 {
                        1.0
                    } else {
                        (PI * x).sin() / (PI * x)
                    };
                    let window = 0.5 + 0.5 * (PI * distance / reach).cos();
                    let weight = sinc * window;
                    // `source` lies within `0..=last`, so it indexes the samples.
                    sum += weight * f64::from(self.samples[source as usize]);
                    weights += weight;
                }
                if weights.abs() < 1e-12 {
                    0.0
                } else {
                    (sum / weights) as f32
                }
            })
            .collect();
        Self { samples, rate }
    }
}

// A clip in JSON is its WAV file's bytes, as a `data:` URL or bare base64, so JSON from a client never names a file to read.
impl<'de> Deserialize<'de> for Audio {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> StdResult<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        let bytes = base64::data_bytes(&text)
            .ok_or_else(|| DeError::custom("audio that is neither a data URL nor base64"))?;
        Self::decode(&bytes).map_err(DeError::custom)
    }
}
