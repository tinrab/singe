//! Log mel spectrograms as `transformers`' `NumPy` feature extractors compute them in `F64`: AST's Kaldi-style filter banks, and CLAP's spectrograms in decibels through Slaney's filters.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};
use singe::signal::slaney_mel_filters;

/// The log mel filter banks an audio classifier reads, as AST's feature extractor computes them: `frames` of `mels` from clips at `sample_rate`, normalized by `mean` and twice `std`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FilterBankOptions {
    pub sample_rate: i64,
    pub mels: usize,
    pub frames: usize,
    pub mean: f64,
    pub std: f64,
}

/// The spectrogram a contrastive audio model reads, as CLAP's feature extractor computes it: clips at `sample_rate` cut or repeated to `samples`, frames of `fft` every `hop`, and `mels` Slaney filters between `low` and `high` Hz.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpectrogramOptions {
    pub sample_rate: i64,
    pub samples: usize,
    pub fft: usize,
    pub hop: usize,
    pub mels: usize,
    pub low: f64,
    pub high: f64,
}

/// Returns the FFT of `real`, whose length `n` is a power of two, as `(re, im)` pairs of its first `n / 2 + 1` bins.
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
        let angle = -2.0 * PI / size as f64;
        for start in (0..count).step_by(size) {
            for offset in 0..size / 2 {
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

fn hann(length: usize, period: usize) -> Vec<f64> {
    (0..length)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f64 / period as f64).cos())
        .collect()
}

fn filter_energies(frame: &[f64], filters: &[f64], mels: usize) -> Vec<f64> {
    // The spectrum stored in `complex64`, as the extractor's buffer holds it.
    let power = rfft(frame)
        .into_iter()
        .map(|(re, im)| {
            let (re, im) = (f64::from(re as f32), f64::from(im as f32));
            re * re + im * im
        })
        .collect::<Vec<_>>();
    (0..mels)
        .map(|filter| {
            (0..power.len())
                .map(|bin| filters[bin * mels + filter] * power[bin])
                .sum::<f64>()
        })
        .collect()
}

fn kaldi_mel(frequency: f64) -> f64 {
    #[expect(clippy::imprecise_flops, reason = "Kaldi computes `log(1 + f / 700)`")]
    let mel = 1127.0 * (1.0 + frequency / 700.0).ln();
    mel
}

/// Returns triangular filters `[bins, filters]`, row by row, spaced evenly in Kaldi's mel scale between `low` and `high` Hz and triangular in it, as `mel_filter_bank(..., mel_scale="kaldi", triangularize_in_mel_space=True)` makes them.
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

/// Returns the features AST reads from samples at `options.sample_rate`, 16 kHz for AST: `frames × mels` log filter-bank energies of 400-sample frames every 160 samples, padded with zeros or cut to `frames`, then normalized by `mean` and twice `std`, row by row.
#[must_use]
pub fn ast_features(samples: &[f32], options: &FilterBankOptions) -> Vec<f32> {
    const FRAME: usize = 400;
    const HOP: usize = 160;
    const FFT: usize = 512;
    const PREEMPHASIS: f64 = 0.97;

    // `np.finfo(np.float32).eps`, as the extractor floors the filter outputs.
    const FLOOR: f64 = 1.192_092_955_078_125e-07;
    let FilterBankOptions {
        sample_rate,
        mels,
        frames,
        mean,
        std,
    } = *options;
    let rate = sample_rate as f64;
    let window = hann(FRAME, FRAME - 1);
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

        for (filter, energy) in filter_energies(&buffer, &filters, mels)
            .into_iter()
            .enumerate()
        {
            features[frame * mels + filter] = energy.max(FLOOR).ln() as f32;
        }
    }
    features
        .into_iter()
        .map(|value| ((f64::from(value) - mean) / (std * 2.0)) as f32)
        .collect()
}

/// Returns the spectrogram CLAP reads from samples at `options.sample_rate`, `frames × mels` row by row.
/// The clip is cut to `options.samples` samples, or repeated whole and padded with silence up to it.
/// Frames of `fft` samples every `hop`, centered by reflection, take a periodic Hann window, and their power spectra pass through Slaney's filters into decibels.
#[must_use]
pub fn clap_features(samples: &[f32], options: &SpectrogramOptions) -> Vec<f32> {
    let SpectrogramOptions {
        sample_rate,
        samples: length,
        fft,
        hop,
        mels,
        low,
        high,
    } = *options;
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
    let window = hann(fft, fft);
    let bins = fft / 2 + 1;
    let filters = slaney_mel_filters(sample_rate, bins, mels, low, high);
    let frames = 1 + (padded.len() - fft) / hop;
    let mut features = Vec::with_capacity(frames * mels);
    for frame in 0..frames {
        let buffer = (0..fft)
            .map(|index| padded[frame * hop + index] * window[index])
            .collect::<Vec<_>>();
        for energy in filter_energies(&buffer, &filters, mels) {
            features.push((10.0 * energy.max(1e-10).log10()) as f32);
        }
    }
    features
}
