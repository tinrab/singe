//! Audio frontend configuration used by audio feature extraction kernels.

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowKind {
    Rectangular,
    PeriodicHann,
    PeriodicHamming,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PadMode {
    Reflect,
    Zero,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpectrumKind {
    OneSide,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpectralNormalization {
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioFeatureLayout {
    FramesFirst,
    MelsFirst,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub center: bool,
    pub pad_mode: PadMode,
    pub window: WindowKind,
    pub spectrum: SpectrumKind,
    pub normalization: SpectralNormalization,
    pub drop_last_frame: bool,
}

impl StftConfig {
    pub const fn frequency_bin_count(self) -> usize {
        match self.spectrum {
            SpectrumKind::OneSide => self.n_fft / 2 + 1,
        }
    }

    pub fn centered_input_len(self, input_len: usize) -> Result<usize> {
        if !self.center {
            return Ok(input_len);
        }
        let pad = self.n_fft / 2;
        input_len
            .checked_add(pad)
            .and_then(|len| len.checked_add(pad))
            .ok_or(Error::SizeOverflow)
    }

    pub fn frame_count(self, input_len: usize) -> Result<usize> {
        validate_stft_config(self)?;
        let padded_len = self.centered_input_len(input_len)?;
        if padded_len < self.n_fft {
            return Ok(0);
        }

        let mut frames = (padded_len - self.n_fft) / self.hop_length + 1;
        if self.drop_last_frame {
            frames = frames.saturating_sub(1);
        }
        Ok(frames)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MelFilterConfig {
    pub sample_rate: usize,
    pub frequency_bin_count: usize,
    pub mel_bin_count: usize,
    pub min_frequency: f32,
    pub max_frequency: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogMelConfig {
    pub floor: f32,
    pub reference_max: Option<f32>,
    pub dynamic_range: Option<f32>,
    pub offset: f32,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioFrontendConfig {
    pub stft: StftConfig,
    pub mel: MelFilterConfig,
    pub log_mel: LogMelConfig,
    pub layout: AudioFeatureLayout,
}

impl AudioFrontendConfig {
    pub fn validate(self) -> Result<()> {
        validate_stft_config(self.stft)?;
        validate_mel_filter_config(self.mel)?;
        validate_log_mel_config(self.log_mel)?;
        if self.stft.frequency_bin_count() != self.mel.frequency_bin_count {
            return Err(Error::LengthMismatch);
        }
        Ok(())
    }
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

fn validate_log_mel_config(config: LogMelConfig) -> Result<()> {
    if !config.floor.is_finite()
        || config.floor <= 0.0
        || !config.offset.is_finite()
        || !config.scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    if let Some(dynamic_range) = config.dynamic_range
        && (!dynamic_range.is_finite() || dynamic_range <= 0.0)
    {
        return Err(Error::InvalidLength);
    }
    if let Some(reference_max) = config.reference_max
        && !reference_max.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}
