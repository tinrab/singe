//! Clips of audio as models hear them: mono samples at a known rate, decoded from WAV, FLAC, MP3, Ogg, MP4, and the other files Symphonia reads.
//! Each model's feature extractor, such as Whisper's log-mel front end, lives with its model, as `transformers` keeps them.

use std::{borrow::Cow, f64::consts::PI, fs, io::Cursor, path::Path, result::Result as StdResult};

use serde::{Deserialize, Deserializer, de::Error as DeError};
use singe::{Device, Error as SingeError, Shape, Tensor};
use symphonium::{DecodeConfig, error::LoadError};

use crate::{
    error::{Error, Result},
    processor::base64,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub rate: u32,
}

/// The zero crossings of the windowed sinc on each side of a resampled point.
const RESAMPLING_TAPS: f64 = 16.0;

/// The most bytes a clip's decoded `f32` samples may take, all channels together, 4 GiB.
/// Three hours of 48 kHz stereo, 3.9 GiB, fits, where `symphonium`'s default of 1 GB stops at 43 minutes of it.
const MAX_DECODED_BYTES: usize = 4 << 30;

impl Audio {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(Error::io(path))?;
        Self::decode(&bytes).map_err(|error| match error {
            Error::Media(message) => Error::Media(format!("{}: {message}", path.display())),
            other => other,
        })
    }

    /// Decodes a clip in any format and encoding Symphonia reads, such as WAV, FLAC, MP3, Ogg Vorbis, or AAC in MP4, averaging its channels into one.
    ///
    /// # Errors
    ///
    /// Returns an error if `symphonium` cannot decode `bytes`, if the decoded samples would take more than 4 GiB, or if the clip has no channels.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let undecodable = |error: LoadError| Error::Media(format!("undecodable audio: {error}"));
        let probed =
            symphonium::probe_from_source(Box::new(Cursor::new(bytes.to_vec())), None, None)
                .map_err(undecodable)?;
        let config = DecodeConfig {
            max_bytes: MAX_DECODED_BYTES,
            ..DecodeConfig::default()
        };
        let decoded =
            symphonium::decode_f32(probed, &config, None, None, None).map_err(undecodable)?;
        let channels = decoded.data.len();
        if channels == 0 {
            return Err(Error::Media("audio without channels".to_owned()));
        }

        let samples = (0..decoded.frames())
            .map(|frame| {
                decoded
                    .data
                    .iter()
                    .map(|channel| channel[frame])
                    .sum::<f32>()
                    / channels as f32
            })
            .collect();
        Ok(Self {
            samples,
            rate: decoded.sample_rate.get(),
        })
    }

    /// Encodes the clip as a mono WAV file of 16-bit PCM.
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

    /// Returns the samples as 16-bit little-endian PCM, clipped to its range.
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

    /// Returns the clip at `rate`: itself when it is at that rate already, or else [`Audio::resampled`].
    ///
    /// # Errors
    ///
    /// Returns an error if `rate` is not a positive `u32`.
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

    /// Resamples the clip to `rate` with a Hann-windowed sinc, low-passed below the lower rate's Nyquist frequency.
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

/// A clip in JSON is its audio file's bytes, in any format [`Audio::decode`] reads, as a `data:` URL or bare base64, so JSON from a client never names a file to read.
impl<'de> Deserialize<'de> for Audio {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> StdResult<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        let bytes = base64::data_bytes(&text)
            .ok_or_else(|| DeError::custom("audio that is neither a data URL nor base64"))?;
        Self::decode(&bytes).map_err(DeError::custom)
    }
}
