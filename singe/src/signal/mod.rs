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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    Hann,
    Hamming,
    Blackman,
    Povey,
}

#[must_use]
pub fn window(function: WindowFunction, length: i64, periodic: bool) -> Vec<f64> {
    let span = if periodic && function != WindowFunction::Povey {
        length
    } else {
        length - 1
    };
    if span <= 0 {
        return vec![1.0; usize::try_from(length).unwrap_or(0)];
    }

    (0..length)
        .map(|index| {
            let phase = 2.0 * PI * index as f64 / span as f64;
            match function {
                WindowFunction::Hann => 0.5 - 0.5 * phase.cos(),
                WindowFunction::Hamming => 0.54 - 0.46 * phase.cos(),
                WindowFunction::Blackman => {
                    0.08_f64.mul_add((2.0 * phase).cos(), 0.42 - 0.5 * phase.cos())
                }
                WindowFunction::Povey => (0.5 - 0.5 * phase.cos()).powf(0.85),
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramePadding {
    Reflect,
    Zeros,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogScale {
    Whisper { dynamic_range: f64 },
    Natural { guard: f64 },
}

// How a log-mel spectrogram is computed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogMelOptions {
    pub sample_rate: i64,
    pub window: i64,
    pub window_function: WindowFunction,
    pub periodic: bool,
    pub fft: i64,
    // The samples between frames.
    pub hop: i64,
    pub mels: i64,
    // The highest frequency the mel filters cover, in hertz.
    pub max_frequency: f64,
    pub samples: i64,
    pub padding: FramePadding,
    pub preemphasis: Option<f64>,
    pub log: LogScale,
    pub normalize: bool,
}

impl LogMelOptions {
    // Whisper's: 16 kHz clips of 30 seconds, 25 ms frames every 10 ms, and `mels` filters up to 8 kHz.
    #[must_use]
    pub const fn whisper(mels: i64) -> Self {
        Self {
            sample_rate: 16_000,
            window: 400,
            window_function: WindowFunction::Hann,
            periodic: true,
            fft: 400,
            hop: 160,
            mels,
            max_frequency: 8_000.0,
            samples: 480_000,
            padding: FramePadding::Reflect,
            preemphasis: None,
            log: LogScale::Whisper { dynamic_range: 8.0 },
            normalize: false,
        }
    }

    #[must_use]
    pub const fn nemo(mels: i64) -> Self {
        Self {
            sample_rate: 16_000,
            window: 400,
            window_function: WindowFunction::Hann,
            periodic: false,
            fft: 512,
            hop: 160,
            mels,
            max_frequency: 8_000.0,
            samples: 0,
            padding: FramePadding::Zeros,
            preemphasis: Some(0.97),
            log: LogScale::Natural {
                guard: 5.960_464_477_539_063e-8,
            },
            normalize: true,
        }
    }

    // The frames a clip yields: one per hop, of which the last is dropped.
    #[must_use]
    pub const fn frames(&self) -> i64 {
        self.samples / self.hop
    }

    const fn bins(&self) -> i64 {
        self.fft / 2 + 1
    }
}

#[derive(Debug, Clone)]
pub struct LogMelSpectrogram {
    options: LogMelOptions,
    window: Vec<f32>,
    filters: Vec<f32>,
}

impl LogMelSpectrogram {
    pub fn create(options: LogMelOptions) -> Result<Self> {
        if options.window < 2
            || options.fft < options.window
            || options.hop <= 0
            || options.mels <= 0
            || options.samples < 0
            || (matches!(options.log, LogScale::Whisper { .. }) && options.samples < options.fft)
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
        // The window, zero-padded on both sides to the transform's length.
        let before = usize::try_from((options.fft - options.window) / 2)
            .map_err(|_| Error::out_of_range("log-mel window padding"))?;
        let mut padded = vec![0.0_f32; usize::try_from(options.fft).unwrap_or(0)];
        for (index, value) in window(options.window_function, options.window, options.periodic)
            .into_iter()
            .enumerate()
        {
            padded[before + index] = value as f32;
        }
        Ok(Self {
            window: padded,
            filters,
            options,
        })
    }

    #[must_use]
    pub const fn own_frames(&self, samples: i64) -> i64 {
        let options = &self.options;
        let frames = (samples + options.fft / 2 * 2 - options.fft).div_euclid(options.hop);
        if frames < 0 { 0 } else { frames }
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
    fn log_power(&self, padded: &Tensor) -> Result<Tensor> {
        let options = &self.options;
        let &[batch, length] = padded.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "log-mel features read padded samples, [batch, samples]",
            ));
        };
        let frames = (length - options.fft).div_euclid(options.hop) + 1;
        let per_frame = (batch * options.fft) as f64 * DataType::F32.size_bytes() as f64;
        let span = ((FRAME_BUDGET / per_frame) as i64).max(1);
        if frames <= span {
            return self.log_power_span(padded);
        }

        let mut pieces = Vec::new();
        let mut first = 0;
        while first < frames {
            let count = span.min(frames - first);
            pieces.push(self.log_power_span(&padded.narrow(
                1,
                first * options.hop,
                (count - 1) * options.hop + options.fft,
            )?)?);
            first += count;
        }
        Tensor::concat(&pieces, 2)
    }

    fn log_power_span(&self, padded: &Tensor) -> Result<Tensor> {
        let options = &self.options;
        let window = capture::literal(padded.device(), |device| {
            Tensor::from_vec(device, Shape::create([options.fft])?, self.window.clone())
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
            .unfold(1, options.fft, options.hop)?
            .mul(&window)?
            .rfft()?;
        let power = spectrum
            .mul(&spectrum)?
            .sum(&[-1], false)?
            .matmul(&filters)?
            .transpose(1, 2)?;
        match options.log {
            LogScale::Whisper { .. } => power
                .clamp(POWER_FLOOR, f32::MAX)?
                .log()?
                .div_scalar(LN_10 as f32),
            LogScale::Natural { guard } => power.add_scalar(guard)?.log(),
        }
    }

    fn range(&self) -> Result<f32> {
        match self.options.log {
            LogScale::Whisper { dynamic_range } => Ok(dynamic_range as f32),
            LogScale::Natural { .. } => Err(Error::invalid_argument(
                "a fixed-length log-mel spectrogram on a scale other than Whisper's",
            )),
        }
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
        let half = options.fft / 2;
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
        let log = self
            .log_power(
                &samples
                    .cast(DataType::F32)?
                    .index_select(0, &sources)?
                    .unsqueeze(0)?,
            )?
            .narrow(2, 0, options.frames())?;
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
            .sub_scalar(f64::from(self.range()?))?;
        let features = log.maximum(&floor)?.add_scalar(4.0)?.div_scalar(4.0)?;
        let zero = capture::literal(device, |device| {
            Tensor::from_vec(device, Shape::create([1])?, vec![0.0_f32])
        })?;
        Tensor::where_select(&own, &features, &zero)
    }
}

impl LogMelSpectrogram {
    pub fn features(&self, samples: &Tensor, lengths: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let options = &self.options;
        let &[batch, count] = samples.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "log-mel features read clips' samples, [batch, samples]",
            ));
        };
        if count < 2 {
            return Err(Error::invalid_argument(
                "log-mel features of under two samples",
            ));
        }

        let device = samples.device();
        let mut samples = samples.cast(DataType::F32)?;
        let lengths = match lengths {
            Some(lengths) => Some(lengths.cast(DataType::I64)?.reshape([batch, 1])?),
            None => None,
        };
        if let Some(coefficient) = options.preemphasis {
            let previous = samples.narrow(1, 0, count - 1)?.mul_scalar(coefficient)?;
            let emphasized = samples.narrow(1, 1, count - 1)?.sub(&previous)?;
            samples = Tensor::concat(&[samples.narrow(1, 0, 1)?, emphasized], 1)?;
        }

        if let Some(lengths) = &lengths {
            let positions =
                capture::literal(device, |device| Tensor::arange(device, 0_i64, count, 1))?
                    .unsqueeze(0)?;
            samples = samples.mul(&positions.less(lengths)?.cast(DataType::F32)?)?;
        }

        let half = options.fft / 2;
        let padded = match options.padding {
            FramePadding::Reflect => samples.pad_reflect(1, half, half)?,
            FramePadding::Zeros => samples.pad(&[(0, 0), (half, half)], 0.0)?,
        };
        let log = self.log_power(&padded)?;
        let total = (count + 2 * half - options.fft).div_euclid(options.hop) + 1;
        let frames = match &lengths {
            // Frame `f` is a clip's own when `(f + 1) · hop + fft - 2 · ⌊fft / 2⌋ ≤ length`, as `own_frames` counts them, compared in integers, since F32 rounds lengths past 2^24 samples.
            Some(lengths) => capture::literal(device, |device| {
                Tensor::from_vec(
                    device,
                    Shape::create([1, total])?,
                    (1..=total)
                        .map(|frame| frame * options.hop + options.fft - 2 * half - 1)
                        .collect(),
                )
            })?
            .less(lengths)?
            .cast(DataType::F32)?
            .sum(&[1], false)?
            .cast(DataType::I64)?,
            None => capture::literal(device, |device| {
                Tensor::from_vec(
                    device,
                    Shape::create([batch])?,
                    vec![self.own_frames(count); usize::try_from(batch).unwrap_or(0)],
                )
            })?,
        };
        let steps = capture::literal(device, |device| Tensor::arange(device, 0_i64, total, 1))?
            .reshape([1, 1, total])?;
        let own = steps
            .less(&frames.reshape([batch, 1, 1])?)?
            .cast(DataType::F32)?;
        let log = if options.normalize {
            let counted = frames
                .cast(DataType::F32)?
                .reshape([batch, 1, 1])?
                .clamp(1.0, f32::MAX)?;
            let mean = log.mul(&own)?.sum(&[2], true)?.div(&counted)?;
            let centered = log.sub(&mean)?.mul(&own)?;
            let deviation = centered
                .mul(&centered)?
                .sum(&[2], true)?
                .div(&counted.sub_scalar(1.0)?.clamp(1.0, f32::MAX)?)?
                .sqrt()?;
            log.sub(&mean)?
                .div(&deviation.add_scalar(NORMALIZE_EPSILON)?)?
        } else {
            log
        };
        Ok((log.mul(&own)?, frames))
    }
}

const FRAME_BUDGET: f64 = (256 << 20) as f64;

const NORMALIZE_EPSILON: f64 = 1e-5;

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
        let half = options.fft / 2;
        let log = self
            .log_power(&samples.pad_reflect(1, half, half)?)?
            .narrow(2, 0, options.frames())?;
        let floor = log
            .max_reduce(&[1, 2], true)?
            .sub_scalar(f64::from(self.range()?))?;
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
        let periodic = window(WindowFunction::Hann, 400, true);
        assert!(periodic[0].abs() < 1e-12);
        assert!((periodic[200] - 1.0).abs() < 1e-12);
        assert!((periodic[1] - periodic[399]).abs() < 1e-12);
        let symmetric = window(WindowFunction::Hann, 400, false);
        assert!(symmetric[399].abs() < 1e-12);
    }
}
