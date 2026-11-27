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
        let kernel = Kernel::new(self.rate, rate);
        let length = (self.samples.len() as f64 * kernel.ratio).round() as usize;
        let last = i64::try_from(self.samples.len()).unwrap_or(i64::MAX) - 1;
        let samples = (0..length)
            .map(|index| {
                let center = index as f64 / kernel.ratio;
                let (first, end) = kernel.reach(center);
                kernel.point(center, first, end.min(last), |source| {
                    // `source` lies within `0..=last`, so it indexes the samples.
                    self.samples[source as usize]
                })
            })
            .collect();
        Self { samples, rate }
    }
}

/// Resamples a stream that arrives in pieces, as [`Audio::resampled`] resamples a whole clip: each output sample comes once every input sample it reads has arrived.
#[derive(Debug, Clone)]
pub struct StreamingResampler {
    kernel: Kernel,
    /// The input samples later outputs still read.
    pending: Vec<f32>,
    /// The stream position of `pending`'s first sample.
    start: i64,
    /// The outputs given so far.
    produced: i64,
}

impl StreamingResampler {
    /// Creates a resampler of a stream at `from` to `to`, both in hertz.
    #[must_use]
    pub fn new(from: u32, to: u32) -> Self {
        Self {
            kernel: Kernel::new(from, to),
            pending: Vec::new(),
            start: 0,
            produced: 0,
        }
    }

    /// Takes the stream's next `samples` and returns the outputs they complete.
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        self.pending.extend_from_slice(samples);
        let available = self.start + i64::try_from(self.pending.len()).unwrap_or(i64::MAX);
        let mut outputs = Vec::new();
        loop {
            let center = self.produced as f64 / self.kernel.ratio;
            let (first, end) = self.kernel.reach(center);
            if end >= available {
                break;
            }

            let (pending, start) = (&self.pending, self.start);
            // Samples before `start` were dropped only once no output read them, so `source - start` indexes `pending`.
            outputs.push(self.kernel.point(center, first, end, |source| {
                pending[(source - start) as usize]
            }));
            self.produced += 1;
        }

        let (needed, _) = self.kernel.reach(self.produced as f64 / self.kernel.ratio);
        let dropped = usize::try_from(needed - self.start)
            .unwrap_or(0)
            .min(self.pending.len());
        self.pending.drain(..dropped);
        self.start += i64::try_from(dropped).unwrap_or(0);
        outputs
    }
}

/// The Hann-windowed sinc a resampling from one rate to another weighs inputs by.
#[derive(Debug, Clone, Copy)]
struct Kernel {
    /// Output samples per input sample.
    ratio: f64,
    /// The low-pass cutoff, as a fraction of the input's Nyquist frequency.
    cutoff: f64,
    /// How far from a point, in input samples, the window reaches.
    reach: f64,
}

impl Kernel {
    fn new(from: u32, to: u32) -> Self {
        let ratio = f64::from(to) / f64::from(from);
        let cutoff = ratio.min(1.0);
        Self {
            ratio,
            cutoff,
            reach: RESAMPLING_TAPS / cutoff,
        }
    }

    /// Returns the first and last input samples an output at `center` reads, the first no earlier than the stream's start.
    fn reach(&self, center: f64) -> (i64, i64) {
        (
            (center - self.reach).ceil().max(0.0) as i64,
            (center + self.reach).floor() as i64,
        )
    }

    /// Returns the output at `center`, weighing inputs `first..=end`, which `sample` reads.
    fn point(&self, center: f64, first: i64, end: i64, sample: impl Fn(i64) -> f32) -> f32 {
        let (mut sum, mut weights) = (0.0, 0.0);
        for source in first..=end {
            let distance = center - source as f64;
            let x = distance * self.cutoff;
            let sinc = if x.abs() < 1e-9 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            let window = 0.5 + 0.5 * (PI * distance / self.reach).cos();
            let weight = sinc * window;
            sum += weight * f64::from(sample(source));
            weights += weight;
        }
        if weights.abs() < 1e-12 {
            0.0
        } else {
            (sum / weights) as f32
        }
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
