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

        let bins = usize::try_from(options.bins())
            .map_err(|_| Error::out_of_range("log-mel frequency bins"))?;
        let mels =
            usize::try_from(options.mels).map_err(|_| Error::out_of_range("log-mel mels"))?;
        let filters =
            slaney_mel_filters(options.sample_rate, bins, mels, 0.0, options.max_frequency)
                .into_iter()
                .map(|weight| weight as f32)
                .collect();
        Ok(Self {
            window: hann_window(options.window),
            filters,
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
fn hann_window(length: i64) -> Vec<f32> {
    (0..length)
        .map(|index| (-(2.0 * PI * index as f64 / length as f64).cos()).mul_add(0.5, 0.5) as f32)
        .collect()
}

fn slaney_mels_per_log() -> f64 {
    27.0 / 6.4_f64.ln()
}

fn hertz_to_mel(hertz: f64) -> f64 {
    if hertz >= 1_000.0 {
        15.0 + (hertz / 1_000.0).ln() * slaney_mels_per_log()
    } else {
        3.0 * hertz / 200.0
    }
}

fn mel_to_hertz(mel: f64) -> f64 {
    if mel >= 15.0 {
        1_000.0 * ((mel - 15.0) / slaney_mels_per_log()).exp()
    } else {
        200.0 * mel / 3.0
    }
}

// Triangular filters, `[bins, mels]` row-major, between edges spaced evenly on the Slaney mel scale from `low` to `high` hertz, each scaled to unit area, over `bins` frequencies from zero to half of `sample_rate`, as `transformers`' `mel_filter_bank` computes them with Slaney's norm and scale.
#[must_use]
pub fn slaney_mel_filters(
    sample_rate: i64,
    bins: usize,
    mels: usize,
    low: f64,
    high: f64,
) -> Vec<f64> {
    let (low, high) = (hertz_to_mel(low), hertz_to_mel(high));
    let edges = (0..mels + 2)
        .map(|index| mel_to_hertz(low + (high - low) * index as f64 / (mels + 1) as f64))
        .collect::<Vec<_>>();
    let nyquist = (sample_rate / 2) as f64;
    let mut filters = Vec::with_capacity(bins * mels);
    for bin in 0..bins {
        let frequency = nyquist * bin as f64 / (bins - 1) as f64;
        for mel in 0..mels {
            let rising = (frequency - edges[mel]) / (edges[mel + 1] - edges[mel]);
            let falling = (edges[mel + 2] - frequency) / (edges[mel + 2] - edges[mel + 1]);
            let area = 2.0 / (edges[mel + 2] - edges[mel]);
            filters.push(rising.min(falling).max(0.0) * area);
        }
    }
    filters
}

// The smallest power `log10` reads, as Whisper's extractor floors it.
const POWER_FLOOR: f32 = 1e-10;

impl LogMelSpectrogram {
    // The log10 mel power of each frame of `padded`, `[B, N + window]`, samples already padded by half a window at each end: `[B, mels, N / hop]`.
    fn log_power(&self, padded: &Tensor) -> Result<Tensor> {
        let options = &self.options;
        let window = capture::literal(padded.device(), |device| {
            Tensor::from_vec(
                device,
                Shape::create([options.window])?,
                self.window.clone(),
            )
        })?;
        let filters = capture::literal(padded.device(), |device| {
            Tensor::from_vec(
                device,
                Shape::create([options.bins(), options.mels])?,
                self.filters.clone(),
            )
        })?;
        let spectrum = padded
            .contiguous()?
            .unfold(1, options.window, options.hop)?
            .mul(&window)?
            .rfft()?;
        let decade = LN_10 as f32;
        spectrum
            .mul(&spectrum)?
            .sum(&[-1], false)?
            .matmul(&filters)?
            .transpose(1, 2)?
            .clamp(POWER_FLOOR, f32::MAX)?
            .log()?
            .div_scalar(decade)?
            .narrow(2, 0, options.frames())
    }

    const fn range(&self) -> f32 {
        self.options.dynamic_range as f32
    }

    // The spectrogram of one clip, `[1, mels, frames]`, whose first `count` of `samples` are its own, `[N]` with `N` the options' samples, and `count` an `I64` `[1]`.
    // It reflects the clip at its own end rather than the buffer's, clamps against the loudest of the clip's own frames, `count / hop` of them, and leaves the frames after them zero, as Qwen3-ASR's extractor computes a clip alone and then pads its features.
    pub fn counted(&self, samples: &Tensor, count: &Tensor) -> Result<Tensor> {
        let options = &self.options;
        if samples.shape().dimensions() != [options.samples] {
            return Err(Error::invalid_argument(
                "a counted log-mel spectrogram reads one clip's samples, [samples]",
            ));
        }

        let device = samples.device();
        let half = options.window / 2;
        let length = count.cast(DataType::F32)?;
        // Each padded sample's source: the reflection at the clip's start, the clip, and the reflection at its own end.
        let offsets = capture::literal(device, |device| {
            Tensor::from_vec(
                device,
                Shape::create([options.samples + 2 * half])?,
                (-half..options.samples + half)
                    .map(|offset| offset as f32)
                    .collect(),
            )
        })?
        .abs()?;
        let mirrored = length.mul_scalar(2.0)?.sub_scalar(2.0)?.sub(&offsets)?;
        // Far past the clip a mirrored index falls below zero; those samples feed only frames after the clip's own, which come out zero, so any sample of the buffer serves.
        let sources = Tensor::where_select(&offsets.greater_equal(&length)?, &mirrored, &offsets)?
            .clamp(0.0, (options.samples - 1) as f32)?
            .cast(DataType::I64)?;
        let log = self.log_power(
            &samples
                .cast(DataType::F32)?
                .index_select(0, &sources)?
                .unsqueeze(0)?,
        )?;
        let frames = capture::literal(device, |device| {
            Tensor::from_vec(
                device,
                Shape::create([1, 1, options.frames()])?,
                (0..options.frames()).map(|frame| frame as f32).collect(),
            )
        })?;
        let own = frames.less(&length.div_scalar(options.hop as f64)?.floor()?)?;
        let lowest = capture::literal(device, |device| {
            Tensor::from_vec(device, Shape::create([1])?, vec![f32::NEG_INFINITY])
        })?;
        let floor = Tensor::where_select(&own, &log, &lowest)?
            .max_reduce(&[1, 2], true)?
            .sub_scalar(f64::from(self.range()))?;
        let features = log.maximum(&floor)?.add_scalar(4.0)?.div_scalar(4.0)?;
        let zero = capture::literal(device, |device| {
            Tensor::from_vec(device, Shape::create([1])?, vec![0.0_f32])
        })?;
        Tensor::where_select(&own, &features, &zero)
    }
}

impl Layer for LogMelSpectrogram {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        let options = &self.options;
        let (samples, length) = match *input.shape().dimensions() {
            [length] => (input.unsqueeze(0)?, length),
            [_, length] => (input.clone(), length),
            _ => {
                return Err(Error::invalid_argument(
                    "a log-mel spectrogram reads samples [batch, samples] or [samples]",
                ));
            }
        };
        let samples = samples.cast(DataType::F32)?;
        let samples = if length < options.samples {
            samples.pad(&[(0, 0), (0, options.samples - length)], 0.0)?
        } else {
            samples.narrow(1, 0, options.samples)?
        };
        let half = options.window / 2;
        let log = self.log_power(&samples.pad_reflect(1, half, half)?)?;
        let floor = log
            .max_reduce(&[1, 2], true)?
            .sub_scalar(f64::from(self.range()))?;
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
