//! Audio feature extraction helpers used by speech models.

use crate::{
    audio::{
        AudioFeatureLayout, AudioFrontendConfig, MelFilterConfig, PadMode, StftConfig, WindowKind,
    },
    error::{Error, Result},
    utility::checked_element_count,
};

pub fn stft_power(input: &[f32], config: StftConfig) -> Result<Vec<f32>> {
    Ok(power_from_r2c(&stft_r2c(input, config)?))
}

pub fn stft_r2c(input: &[f32], config: StftConfig) -> Result<Vec<f32>> {
    validate_stft_input(input.len(), config)?;
    let window = stft_window(config.window, config.win_length);
    let (dft_real, dft_imag) = dft_tables(config.n_fft, config.frequency_bin_count())?;
    let frames = config.frame_count(input.len())?;
    let mut out = vec![0.0; frames * config.frequency_bin_count() * 2];
    let pad = if config.center { config.n_fft / 2 } else { 0 };

    for frame in 0..frames {
        for frequency in 0..config.frequency_bin_count() {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for (sample, window_value) in window.iter().copied().enumerate().take(config.n_fft) {
                let raw_index = frame * config.hop_length + sample;
                let value =
                    padded_sample(input, raw_index as isize - pad as isize, config.pad_mode)?
                        * window_value;
                let table_offset = frequency * config.n_fft + sample;
                real += value * dft_real[table_offset];
                imag += value * dft_imag[table_offset];
            }
            let output_offset = (frame * config.frequency_bin_count() + frequency) * 2;
            out[output_offset] = real;
            out[output_offset + 1] = imag;
        }
    }

    Ok(out)
}

pub fn padded_sample(input: &[f32], index: isize, pad_mode: PadMode) -> Result<f32> {
    match pad_mode {
        PadMode::Reflect => Ok(input[reflect_index(index, input.len())?]),
        PadMode::Zero => {
            if index < 0 || index >= input.len() as isize {
                Ok(0.0)
            } else {
                Ok(input[index as usize])
            }
        }
    }
}

pub fn power_from_r2c(spectrum: &[f32]) -> Vec<f32> {
    spectrum
        .chunks_exact(2)
        .map(|complex| complex[0] * complex[0] + complex[1] * complex[1])
        .collect()
}

pub fn log_mel(power: &[f32], frames: usize, config: AudioFrontendConfig) -> Result<Vec<f32>> {
    config.validate()?;
    let expected_power = checked_element_count(frames, config.mel.frequency_bin_count)?;
    if power.len() < expected_power || config.log_mel.dynamic_range.is_none() {
        return Err(Error::InvalidLength);
    }
    let filters = slaney_mel_filters(config.mel)?;
    let dynamic_range = config.log_mel.dynamic_range.ok_or(Error::InvalidLength)?;
    let mut raw = vec![0.0; frames * config.mel.mel_bin_count];

    for frame in 0..frames {
        for mel in 0..config.mel.mel_bin_count {
            let mut sum = 0.0f32;
            for frequency in 0..config.mel.frequency_bin_count {
                sum += power[frame * config.mel.frequency_bin_count + frequency]
                    * filters[mel * config.mel.frequency_bin_count + frequency];
            }
            raw[frame * config.mel.mel_bin_count + mel] = sum.max(config.log_mel.floor).log10();
        }
    }

    let reference_max = config
        .log_mel
        .reference_max
        .or_else(|| raw.iter().copied().reduce(f32::max))
        .ok_or(Error::InvalidLength)?;
    let lower = reference_max - dynamic_range;
    let mut out = vec![0.0; frames * config.mel.mel_bin_count];

    for frame in 0..frames {
        for mel in 0..config.mel.mel_bin_count {
            let mut value = raw[frame * config.mel.mel_bin_count + mel];
            value = value.max(lower);
            value = (value + config.log_mel.offset) / config.log_mel.scale;
            let output_offset = match config.layout {
                AudioFeatureLayout::FramesFirst => frame * config.mel.mel_bin_count + mel,
                AudioFeatureLayout::MelsFirst => mel * frames + frame,
            };
            out[output_offset] = value;
        }
    }

    Ok(out)
}

pub fn sparse_log_mel_range(
    full: &[f32],
    config: AudioFrontendConfig,
    first_frame: usize,
    frames: usize,
    total_frames: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; full.len()];
    for local_frame in 0..frames {
        let frame = first_frame + local_frame;
        for mel in 0..config.mel.mel_bin_count {
            let offset = match config.layout {
                AudioFeatureLayout::FramesFirst => frame * config.mel.mel_bin_count + mel,
                AudioFeatureLayout::MelsFirst => mel * total_frames + frame,
            };
            out[offset] = full[offset];
        }
    }
    out
}

pub fn convert_log_mel_layout(
    input: &[f32],
    batch: usize,
    frames: usize,
    mel_bins: usize,
    input_layout: AudioFeatureLayout,
    output_layout: AudioFeatureLayout,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for batch_index in 0..batch {
        for frame in 0..frames {
            for mel in 0..mel_bins {
                let input_offset =
                    log_mel_offset(batch_index, frame, mel, frames, mel_bins, input_layout);
                let output_offset =
                    log_mel_offset(batch_index, frame, mel, frames, mel_bins, output_layout);
                output[output_offset] = input[input_offset];
            }
        }
    }
    output
}

pub fn log_mel_offset(
    batch_index: usize,
    frame: usize,
    mel: usize,
    frames: usize,
    mel_bins: usize,
    layout: AudioFeatureLayout,
) -> usize {
    let feature_len = frames * mel_bins;
    match layout {
        AudioFeatureLayout::FramesFirst => batch_index * feature_len + frame * mel_bins + mel,
        AudioFeatureLayout::MelsFirst => batch_index * feature_len + mel * frames + frame,
    }
}

pub fn reflect_index(index: isize, len: usize) -> Result<usize> {
    if len == 0 {
        return Err(Error::InvalidLength);
    }
    if len == 1 {
        return if index == 0 {
            Ok(0)
        } else {
            Err(Error::InvalidLength)
        };
    }
    let reflected = if index < 0 {
        (-index) as usize
    } else if index >= len as isize {
        (2 * len - 2) - index as usize
    } else {
        index as usize
    };
    if reflected >= len {
        return Err(Error::InvalidLength);
    }
    Ok(reflected)
}

fn validate_stft_input(input_len: usize, config: StftConfig) -> Result<()> {
    validate_stft_config(config)?;
    if config.pad_mode != PadMode::Reflect {
        return Ok(());
    }
    if input_len == 0 {
        return Err(Error::InvalidLength);
    }
    if config.center && input_len <= config.n_fft / 2 {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_stft_config(config: StftConfig) -> Result<()> {
    if config.n_fft == 0 || config.hop_length == 0 || config.win_length == 0 {
        return Err(Error::InvalidLength);
    }
    if config.win_length > config.n_fft {
        return Err(Error::InvalidLength);
    }
    if !config.n_fft.is_multiple_of(2) {
        return Err(Error::UnsupportedParameter {
            op: "stft".into(),
            parameter: "n_fft".into(),
            value: config.n_fft,
        });
    }
    Ok(())
}

fn validate_mel_filter_config(config: MelFilterConfig) -> Result<()> {
    if config.sample_rate == 0 || config.frequency_bin_count == 0 || config.mel_bin_count == 0 {
        return Err(Error::InvalidLength);
    }
    let nyquist = config.sample_rate as f32 / 2.0;
    if !config.min_frequency.is_finite()
        || !config.max_frequency.is_finite()
        || config.min_frequency < 0.0
        || config.max_frequency <= config.min_frequency
        || config.max_frequency > nyquist
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn stft_window(kind: WindowKind, len: usize) -> Vec<f32> {
    match kind {
        WindowKind::Rectangular => vec![1.0; len],
        WindowKind::PeriodicHann => (0..len)
            .map(|index| {
                let angle = 2.0 * std::f32::consts::PI * index as f32 / len as f32;
                0.5 * (1.0 - angle.cos())
            })
            .collect(),
        WindowKind::PeriodicHamming => {
            const ALPHA: f32 = 0.54;
            const BETA: f32 = 1.0 - ALPHA;
            (0..len)
                .map(|index| {
                    let angle = 2.0 * std::f32::consts::PI * index as f32 / len as f32;
                    ALPHA - BETA * angle.cos()
                })
                .collect()
        }
    }
}

fn dft_tables(n_fft: usize, frequency_bin_count: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let len = checked_element_count(n_fft, frequency_bin_count)?;
    let mut real = Vec::with_capacity(len);
    let mut imag = Vec::with_capacity(len);
    for frequency in 0..frequency_bin_count {
        for sample in 0..n_fft {
            let angle =
                2.0 * std::f32::consts::PI * frequency as f32 * sample as f32 / n_fft as f32;
            real.push(angle.cos());
            imag.push(-angle.sin());
        }
    }
    Ok((real, imag))
}

fn slaney_mel_filters(config: MelFilterConfig) -> Result<Vec<f32>> {
    validate_mel_filter_config(config)?;
    let len = checked_element_count(config.mel_bin_count, config.frequency_bin_count)?;
    let mut filters = vec![0.0; len];

    let min_mel = hertz_to_slaney_mel(config.min_frequency);
    let max_mel = hertz_to_slaney_mel(config.max_frequency);
    let step = (max_mel - min_mel) / (config.mel_bin_count + 1) as f32;

    let mut mel_edges = Vec::with_capacity(config.mel_bin_count + 2);
    for index in 0..config.mel_bin_count + 2 {
        mel_edges.push(slaney_mel_to_hertz(min_mel + step * index as f32));
    }

    let mut fft_frequencies = Vec::with_capacity(config.frequency_bin_count);
    for bin in 0..config.frequency_bin_count {
        let frequency =
            bin as f32 * config.sample_rate as f32 / (2 * (config.frequency_bin_count - 1)) as f32;
        fft_frequencies.push(frequency);
    }

    for mel in 0..config.mel_bin_count {
        let lower = mel_edges[mel];
        let center = mel_edges[mel + 1];
        let upper = mel_edges[mel + 2];
        let enorm = 2.0 / (upper - lower);

        for (frequency_bin, frequency) in fft_frequencies.iter().copied().enumerate() {
            let lower_slope = (frequency - lower) / (center - lower);
            let upper_slope = (upper - frequency) / (upper - center);
            let weight = lower_slope.min(upper_slope).max(0.0) * enorm;
            filters[mel * config.frequency_bin_count + frequency_bin] = weight;
        }
    }

    Ok(filters)
}

fn hertz_to_slaney_mel(frequency: f32) -> f32 {
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    const LOG_STEP: f32 = 0.06875178;

    if frequency < MIN_LOG_HZ {
        frequency * 0.015
    } else {
        MIN_LOG_MEL + (frequency / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn slaney_mel_to_hertz(mel: f32) -> f32 {
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    const LOG_STEP: f32 = 0.06875178;

    if mel < MIN_LOG_MEL {
        mel / 0.015
    } else {
        MIN_LOG_HZ * (LOG_STEP * (mel - MIN_LOG_MEL)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{LogMelConfig, SpectralNormalization, SpectrumKind};

    #[test]
    fn padded_sample_rejects_empty_reflect_input() {
        assert!(matches!(
            padded_sample(&[], 0, PadMode::Reflect),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn stft_power_rejects_too_short_centered_reflect_input() {
        let config = test_stft_config(PadMode::Reflect);

        assert!(matches!(
            stft_power(&[0.0, 1.0], config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn stft_power_allows_empty_centered_zero_padded_input() -> Result<()> {
        let config = test_stft_config(PadMode::Zero);

        let actual = stft_power(&[], config)?;

        assert_eq!(
            actual.len(),
            config.frame_count(0)? * config.frequency_bin_count()
        );
        assert!(actual.iter().all(|value| *value == 0.0));
        Ok(())
    }

    #[test]
    fn log_mel_rejects_empty_dynamic_reference_input() {
        let mut config = test_frontend_config();
        config.log_mel.reference_max = None;

        assert!(matches!(log_mel(&[], 0, config), Err(Error::InvalidLength)));
    }

    fn test_stft_config(pad_mode: PadMode) -> StftConfig {
        StftConfig {
            n_fft: 8,
            hop_length: 4,
            win_length: 8,
            center: true,
            pad_mode,
            window: WindowKind::Rectangular,
            spectrum: SpectrumKind::OneSide,
            normalization: SpectralNormalization::None,
            drop_last_frame: false,
        }
    }

    fn test_frontend_config() -> AudioFrontendConfig {
        AudioFrontendConfig {
            stft: test_stft_config(PadMode::Zero),
            mel: MelFilterConfig {
                sample_rate: 16_000,
                frequency_bin_count: 5,
                mel_bin_count: 2,
                min_frequency: 0.0,
                max_frequency: 8_000.0,
            },
            log_mel: LogMelConfig {
                floor: 1e-10,
                reference_max: Some(0.0),
                dynamic_range: Some(8.0),
                offset: 4.0,
                scale: 4.0,
            },
            layout: AudioFeatureLayout::FramesFirst,
        }
    }
}
