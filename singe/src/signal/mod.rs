// Front ends that turn raw signals into model inputs and learn nothing, such as an audio clip's log-mel spectrogram.

use std::f64::consts::{LN_10, PI};

use crate::{
    data_type::DataType,
    error::{Error, Result},
    graph::capture,
    nn::{Layer, ModelState},
    shape::Shape,
    tensor::Tensor,
};

// How a log-mel spectrogram is computed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogMelOptions {
    pub sample_rate: i64,
    // The samples of each frame's Fourier transform, and of its Hann window.
    pub window: i64,
    // The samples between frames.
    pub hop: i64,
    pub mels: i64,
    // The highest frequency the mel filters cover, in hertz.
    pub max_frequency: f64,
    // The samples a clip is padded with zeros or cut to.
    pub samples: i64,
    // How far below each clip's loudest value its log values are clamped, in decades.
    pub dynamic_range: f64,
}

impl LogMelOptions {
    // Whisper's: 16 kHz clips of 30 seconds, 25 ms frames every 10 ms, and `mels` filters up to 8 kHz.
    #[must_use]
    pub const fn whisper(mels: i64) -> Self {
        Self {
            sample_rate: 16_000,
            window: 400,
            hop: 160,
            mels,
            max_frequency: 8_000.0,
            samples: 480_000,
            dynamic_range: 8.0,
        }
    }

    // The frames a clip yields: one per hop, of which the last is dropped.
    #[must_use]
    pub const fn frames(&self) -> i64 {
        self.samples / self.hop
    }

    const fn bins(&self) -> i64 {
        self.window / 2 + 1
    }
}

// A clip's log-mel spectrogram, `[B, mels, frames]` from samples `[B, N]` or `[N]`, as Whisper's feature extractor computes it.
//
// Each clip is padded or cut to its length, padded at each end by half a window of its own reflection, and cut into Hann-windowed frames.
// Each frame's power spectrum goes through Slaney-normalized mel filters on the Slaney mel scale, then `log10` of at least 1e-10, clamped to the dynamic range below the clip's loudest value, and mapped by `(x + 4) / 4`.
#[derive(Debug, Clone)]
pub struct LogMelSpectrogram {
    options: LogMelOptions,
    window: Vec<f32>,
    filters: Vec<f32>,
}

impl LogMelSpectrogram {
    pub fn create(options: LogMelOptions) -> Result<Self> {
        if options.window < 2
            || options.hop <= 0
            || options.mels <= 0
            || options.samples < options.window
            || options.sample_rate <= 0
        {
            return Err(Error::invalid_argument("log-mel spectrogram options"));
        }
        Ok(Self {
            window: hann_window(options.window),
            filters: mel_filters(&options),
            options,
        })
    }

    #[must_use]
    pub const fn options(&self) -> &LogMelOptions {
        &self.options
    }

    // The mel filters, `[bins, mels]`, row-major.
    #[must_use]
    pub fn mel_filters(&self) -> &[f32] {
        &self.filters
    }
}

// A periodic Hann window of `length` samples, as `torch.hann_window` makes it.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "window lengths are small, and the values lie in [0, 1]"
)]
fn hann_window(length: i64) -> Vec<f32> {
    (0..length)
        .map(|index| (-(2.0 * PI * index as f64 / length as f64).cos()).mul_add(0.5, 0.5) as f32)
        .collect()
}

// The Slaney mel scale: linear below 1 kHz and logarithmic above.
const SLANEY_STEP: f64 = 200.0 / 3.0;
const SLANEY_KNEE_HZ: f64 = 1_000.0;

fn slaney_log_step() -> f64 {
    6.4_f64.ln() / 27.0
}

fn hertz_to_mel(hertz: f64) -> f64 {
    if hertz < SLANEY_KNEE_HZ {
        return hertz / SLANEY_STEP;
    }
    SLANEY_KNEE_HZ / SLANEY_STEP + (hertz / SLANEY_KNEE_HZ).ln() / slaney_log_step()
}

fn mel_to_hertz(mel: f64) -> f64 {
    let knee = SLANEY_KNEE_HZ / SLANEY_STEP;
    if mel < knee {
        return mel * SLANEY_STEP;
    }
    SLANEY_KNEE_HZ * (slaney_log_step() * (mel - knee)).exp()
}

// Triangular filters, `[bins, mels]`, between mel-spaced edges from zero to the highest frequency, each scaled to unit area, as `transformers`' `mel_filter_bank` computes them with Slaney's norm and scale.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "bin and filter counts are small, and filter weights fit F32"
)]
fn mel_filters(options: &LogMelOptions) -> Vec<f32> {
    let bins = options.bins();
    let mels = options.mels;
    let top = hertz_to_mel(options.max_frequency);
    let edges = (0..mels + 2)
        .map(|index| mel_to_hertz(top * index as f64 / (mels + 1) as f64))
        .collect::<Vec<_>>();
    let nyquist = (options.sample_rate / 2) as f64;
    let mut filters = Vec::with_capacity(usize::try_from(bins * mels).unwrap_or(0));
    for bin in 0..bins {
        let frequency = nyquist * bin as f64 / (bins - 1) as f64;
        for mel in 0..usize::try_from(mels).unwrap_or(0) {
            let rising = (frequency - edges[mel]) / (edges[mel + 1] - edges[mel]);
            let falling = (edges[mel + 2] - frequency) / (edges[mel + 2] - edges[mel + 1]);
            let area = 2.0 / (edges[mel + 2] - edges[mel]);
            filters.push((rising.min(falling).max(0.0) * area) as f32);
        }
    }
    filters
}

// The smallest power `log10` reads, as Whisper's extractor floors it.
const POWER_FLOOR: f32 = 1e-10;

impl Layer for LogMelSpectrogram {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        let options = &self.options;
        let samples = match input.shape().dimensions().len() {
            1 => input.unsqueeze(0)?,
            2 => input.clone(),
            _ => {
                return Err(Error::invalid_argument(
                    "a log-mel spectrogram reads samples [batch, samples] or [samples]",
                ));
            }
        }
        .cast(DataType::F32)?;
        let length = samples.shape().dimensions()[1];
        let samples = if length < options.samples {
            samples.pad(&[(0, 0), (0, options.samples - length)], 0.0)?
        } else {
            samples.narrow(1, 0, options.samples)?
        };
        let half = options.window / 2;
        let window = capture::literal(samples.device(), |device| {
            Tensor::from_vec(
                device,
                Shape::create([options.window])?,
                self.window.clone(),
            )
        })?;
        let filters = capture::literal(samples.device(), |device| {
            Tensor::from_vec(
                device,
                Shape::create([options.bins(), options.mels])?,
                self.filters.clone(),
            )
        })?;
        let spectrum = samples
            .pad_reflect(1, half, half)?
            .contiguous()?
            .unfold(1, options.window, options.hop)?
            .mul(&window)?
            .rfft()?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the range is a few decades"
        )]
        let range = options.dynamic_range as f32;
        #[expect(clippy::cast_possible_truncation, reason = "a constant that fits F32")]
        let decade = LN_10 as f32;
        let log = spectrum
            .mul(&spectrum)?
            .sum(&[-1], false)?
            .matmul(&filters)?
            .transpose(1, 2)?
            .clamp(POWER_FLOOR, f32::MAX)?
            .log()?
            .div_scalar(decade)?
            .narrow(2, 0, options.frames())?;
        let floor = log.max_reduce(&[1, 2], true)?.sub_scalar(range)?;
        log.maximum(&floor)?.add_scalar(4.0)?.div_scalar(4.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_mel_scale_round_trips_on_both_sides_of_its_knee() {
        for hertz in [0.0, 440.0, 999.0, 1_000.0, 4_000.0, 8_000.0] {
            assert!((mel_to_hertz(hertz_to_mel(hertz)) - hertz).abs() < 1e-6);
        }
    }

    #[test]
    fn test_a_periodic_hann_window_starts_at_zero_and_peaks_at_its_middle() {
        let window = hann_window(400);
        assert!(window[0].abs() < 1e-7);
        assert!((window[200] - 1.0).abs() < 1e-7);
        assert!((window[1] - window[399]).abs() < 1e-7);
    }
}
