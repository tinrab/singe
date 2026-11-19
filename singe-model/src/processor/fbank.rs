// Log mel spectrograms as `transformers`' NumPy feature extractors compute them in `F64`: AST's Kaldi-style filter banks, and CLAP's spectrograms in decibels through Slaney's filters.

use std::f64::consts::PI;

// The FFT of `real`, whose length is a power of two, as `(re, im)` pairs of its first `n / 2 + 1` bins.
#[expect(
    clippy::suboptimal_flops,
    reason = "rounds each product as the reference does, which the verifications compare with"
)]
fn rfft(real: &[f64]) -> Vec<(f64, f64)> {
    let count = real.len();
    let mut values = real.iter().map(|&value| (value, 0.0)).collect::<Vec<_>>();
    // Bit-reversal permutation, then iterative butterflies.
    let bits = count.trailing_zeros();
    for index in 0..count {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    let mut size = 2;
    while size <= count {
        #[expect(clippy::cast_precision_loss, reason = "FFT sizes are small")]
        let angle = -2.0 * PI / size as f64;
        for start in (0..count).step_by(size) {
            for offset in 0..size / 2 {
                #[expect(clippy::cast_precision_loss, reason = "FFT sizes are small")]
                let (sin, cos) = (angle * offset as f64).sin_cos();
                let (re, im) = values[start + offset + size / 2];
                let twiddled = (re * cos - im * sin, re * sin + im * cos);
                let (even_re, even_im) = values[start + offset];
                values[start + offset] = (even_re + twiddled.0, even_im + twiddled.1);
                values[start + offset + size / 2] = (even_re - twiddled.0, even_im - twiddled.1);
            }
        }

        size *= 2;
    }

    values.truncate(count / 2 + 1);
    values
}

fn kaldi_mel(frequency: f64) -> f64 {
    #[expect(clippy::imprecise_flops, reason = "Kaldi computes `log(1 + f / 700)`")]
    let mel = 1127.0 * (1.0 + frequency / 700.0).ln();
    mel
}

// Triangular filters `[bins, filters]`, row by row, spaced evenly in Kaldi's mel scale between `low` and `high` Hz and triangular in it, as `mel_filter_bank(..., mel_scale="kaldi", triangularize_in_mel_space=True)` makes them.
#[expect(
    clippy::cast_precision_loss,
    reason = "bin and filter counts are small"
)]
fn mel_filters(bins: usize, filters: usize, low: f64, high: f64, rate: f64) -> Vec<f64> {
    let (low, high) = (kaldi_mel(low), kaldi_mel(high));
    let centers = (0..filters + 2)
        .map(|index| low + (high - low) * index as f64 / (filters + 1) as f64)
        .collect::<Vec<_>>();
    let width = rate / ((bins - 1) * 2) as f64;
    let mut weights = vec![0.0; bins * filters];
    for bin in 0..bins {
        let mel = kaldi_mel(width * bin as f64);
        for filter in 0..filters {
            let down = (mel - centers[filter]) / (centers[filter + 1] - centers[filter]);
            let up = (centers[filter + 2] - mel) / (centers[filter + 2] - centers[filter + 1]);
            weights[bin * filters + filter] = down.min(up).max(0.0);
        }
    }
    weights
}

// The features AST reads from 16 kHz samples: `frames × mels` log filter-bank energies, padded with zeros or cut to `frames`, then normalized by `mean` and twice `std`, row by row.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "frames and bins are small, and features are `F32` as NumPy returns them"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "rounds each product as the reference does, which the verifications compare with"
)]
pub fn ast_features(
    samples: &[f32],
    rate: f64,
    mels: usize,
    frames: usize,
    mean: f64,
    std: f64,
) -> Vec<f32> {
    const FRAME: usize = 400;
    const HOP: usize = 160;
    const FFT: usize = 512;
    const PREEMPHASIS: f64 = 0.97;

    // `np.finfo(np.float32).eps`, as the extractor floors the filter outputs.
    const FLOOR: f64 = 1.192_092_955_078_125e-07;
    let window = (0..FRAME)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f64 / (FRAME - 1) as f64).cos())
        .collect::<Vec<_>>();
    let bins = FFT / 2 + 1;
    let filters = mel_filters(bins, mels, 20.0, rate / 2.0, rate);
    let count = if samples.len() < FRAME {
        0
    } else {
        1 + (samples.len() - FRAME) / HOP
    };
    let mut features = vec![0.0_f32; frames * mels];
    for frame in 0..count.min(frames) {
        let mut buffer = vec![0.0_f64; FFT];
        for (index, value) in buffer.iter_mut().take(FRAME).enumerate() {
            *value = f64::from(samples[frame * HOP + index]);
        }

        let average = buffer[..FRAME].iter().sum::<f64>() / FRAME as f64;
        for value in &mut buffer[..FRAME] {
            *value -= average;
        }

        for index in (1..FRAME).rev() {
            buffer[index] -= PREEMPHASIS * buffer[index - 1];
        }

        buffer[0] *= 1.0 - PREEMPHASIS;
        for (value, weight) in buffer.iter_mut().zip(&window) {
            *value *= weight;
        }

        // The spectrum stored in `complex64`, as the extractor's buffer holds it.
        let power = rfft(&buffer)
            .into_iter()
            .map(|(re, im)| {
                let (re, im) = (f64::from(re as f32), f64::from(im as f32));
                re * re + im * im
            })
            .collect::<Vec<_>>();
        for filter in 0..mels {
            let energy = (0..bins)
                .map(|bin| filters[bin * mels + filter] * power[bin])
                .sum::<f64>();
            features[frame * mels + filter] = energy.max(FLOOR).ln() as f32;
        }
    }
    features
        .into_iter()
        .map(|value| ((f64::from(value) - mean) / (std * 2.0)) as f32)
        .collect()
}

// Slaney's mel scale: linear below 1 kHz, logarithmic above.
#[expect(
    clippy::suboptimal_flops,
    reason = "rounds each product as the reference does, which the verifications compare with"
)]
fn slaney_mel(frequency: f64) -> f64 {
    let step = 27.0 / 6.4_f64.ln();
    if frequency >= 1000.0 {
        15.0 + (frequency / 1000.0).ln() * step
    } else {
        3.0 * frequency / 200.0
    }
}

fn slaney_hertz(mel: f64) -> f64 {
    let step = 27.0 / 6.4_f64.ln();
    if mel >= 15.0 {
        1000.0 * ((mel - 15.0) / step).exp()
    } else {
        200.0 * mel / 3.0
    }
}

// Triangular filters `[bins, filters]` in Hz, spaced evenly in Slaney's mel scale between `low` and `high`, each scaled to unit area, as `mel_filter_bank(..., norm="slaney", mel_scale="slaney")` makes them.
#[expect(
    clippy::cast_precision_loss,
    reason = "bin and filter counts are small"
)]
fn slaney_filters(bins: usize, filters: usize, low: f64, high: f64, rate: f64) -> Vec<f64> {
    let (low, high) = (slaney_mel(low), slaney_mel(high));
    let centers = (0..filters + 2)
        .map(|index| slaney_hertz(low + (high - low) * index as f64 / (filters + 1) as f64))
        .collect::<Vec<_>>();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`transformers` halves the integer sampling rate"
    )]
    let nyquist = (rate as i64 / 2) as f64;
    let mut weights = vec![0.0; bins * filters];
    for bin in 0..bins {
        let frequency = nyquist * bin as f64 / (bins - 1) as f64;
        for filter in 0..filters {
            let down = (frequency - centers[filter]) / (centers[filter + 1] - centers[filter]);
            let up =
                (centers[filter + 2] - frequency) / (centers[filter + 2] - centers[filter + 1]);
            let area = 2.0 / (centers[filter + 2] - centers[filter]);
            weights[bin * filters + filter] = down.min(up).max(0.0) * area;
        }
    }
    weights
}

// The spectrogram CLAP reads from samples at `rate`: the clip repeated whole to `length` samples and padded with silence, frames of `fft` samples every `hop` centered by reflection, a periodic Hann window, power spectra through Slaney's filters, in decibels, `frames × mels` row by row.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "frames and bins are small, and features are `F32` as NumPy returns them"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "rounds each product as the reference does, which the verifications compare with"
)]
pub fn clap_features(
    samples: &[f32],
    rate: f64,
    length: usize,
    fft: usize,
    hop: usize,
    mels: usize,
    low: f64,
    high: f64,
) -> Vec<f32> {
    let mut clip = Vec::with_capacity(length);
    if samples.len() >= length || samples.is_empty() {
        clip.extend(samples.iter().take(length).map(|&sample| f64::from(sample)));
    } else {
        for _ in 0..length / samples.len() {
            clip.extend(samples.iter().map(|&sample| f64::from(sample)));
        }
    }

    clip.resize(length, 0.0);
    // Reflected by half a frame at each end.
    let half = fft / 2;
    let mut padded = Vec::with_capacity(length + fft);
    padded.extend((1..=half).rev().map(|index| clip[index]));
    padded.extend_from_slice(&clip);
    padded.extend((1..=half).map(|index| clip[length - 1 - index]));
    let window = (0..fft)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f64 / fft as f64).cos())
        .collect::<Vec<_>>();
    let bins = fft / 2 + 1;
    let filters = slaney_filters(bins, mels, low, high, rate);
    let frames = 1 + (padded.len() - fft) / hop;
    let mut features = Vec::with_capacity(frames * mels);
    for frame in 0..frames {
        let buffer = (0..fft)
            .map(|index| padded[frame * hop + index] * window[index])
            .collect::<Vec<_>>();
        let power = rfft(&buffer)
            .into_iter()
            .map(|(re, im)| {
                let (re, im) = (f64::from(re as f32), f64::from(im as f32));
                re * re + im * im
            })
            .collect::<Vec<_>>();
        for filter in 0..mels {
            let energy = (0..bins)
                .map(|bin| filters[bin * mels + filter] * power[bin])
                .sum::<f64>();
            features.push((10.0 * energy.max(1e-10).log10()) as f32);
        }
    }
    features
}
