//! Audio feature extraction helpers used by speech models.

use singe_cuda::{
    memory::DeviceMemory,
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    audio::{
        AudioFeatureLayout, AudioFrontendConfig, MelFilterConfig, PadMode, SpectralNormalization,
        StftConfig, WindowKind,
    },
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value, ensure_len},
};

#[derive(Debug, PartialEq)]
pub struct AudioFrontendPlan {
    config: AudioFrontendConfig,
    window: DeviceMemory<f32>,
    dft_real: DeviceMemory<f32>,
    dft_imag: DeviceMemory<f32>,
    mel_filters: DeviceMemory<f32>,
}

impl AudioFrontendPlan {
    pub fn create(config: AudioFrontendConfig) -> Result<Self> {
        config.validate()?;

        let window = create_window(config.stft)?;
        let (dft_real, dft_imag) =
            create_dft_tables(config.stft.n_fft, config.stft.frequency_bin_count())?;
        let mel_filters = create_mel_filters(config.mel)?;

        Ok(Self {
            config,
            window,
            dft_real,
            dft_imag,
            mel_filters,
        })
    }

    pub const fn config(&self) -> AudioFrontendConfig {
        self.config
    }

    pub const fn window_len(&self) -> usize {
        self.config.stft.win_length
    }

    pub const fn dft_table_len(&self) -> usize {
        self.config.stft.n_fft * self.config.stft.frequency_bin_count()
    }

    pub const fn mel_filter_len(&self) -> usize {
        self.config.mel.mel_bin_count * self.config.mel.frequency_bin_count
    }

    fn validate_stft_r2c_f32(
        &self,
        output_len: usize,
        input_len: usize,
        batch: usize,
        first_frame: usize,
    ) -> Result<StftSpectrumShape> {
        self.config.validate()?;
        validate_supported_stft_power(self.config.stft)?;
        validate_padding(self.config.stft, input_len)?;
        if batch == 0 {
            return Err(Error::InvalidLength);
        }

        let frequency_bins = self.config.stft.frequency_bin_count();
        let expected_unit =
            checked_element_count(checked_element_count(frequency_bins, 2)?, batch)?;
        if output_len == 0 || !output_len.is_multiple_of(expected_unit) {
            return Err(Error::LengthMismatch);
        }
        let frames = output_len / expected_unit;
        validate_frame_range(self.config.stft, input_len, first_frame, frames)?;
        if frames == 0 {
            return Err(Error::InvalidLength);
        }
        let complex_values = checked_element_count(frames, frequency_bins)?;
        let expected_output =
            checked_element_count(checked_element_count(batch, complex_values)?, 2)?;
        checked_i32_value(expected_output)?;
        ensure_len(output_len, expected_output)?;
        let expected_input = checked_element_count(batch, input_len)?;
        checked_i32_value(expected_input)?;

        Ok(StftSpectrumShape {
            batch,
            frames,
            frequency_bins,
        })
    }

    fn validate_stft_power_f32(
        &self,
        output_len: usize,
        input_len: usize,
        batch: usize,
        first_frame: usize,
    ) -> Result<StftPowerShape> {
        self.config.validate()?;
        validate_supported_stft_power(self.config.stft)?;
        validate_padding(self.config.stft, input_len)?;
        if batch == 0 {
            return Err(Error::InvalidLength);
        }

        let frequency_bins = self.config.stft.frequency_bin_count();
        let expected_unit = checked_element_count(batch, frequency_bins)?;
        if output_len == 0 || !output_len.is_multiple_of(expected_unit) {
            return Err(Error::LengthMismatch);
        }
        let frames = output_len / expected_unit;
        validate_frame_range(self.config.stft, input_len, first_frame, frames)?;
        if frames == 0 {
            return Err(Error::InvalidLength);
        }
        let expected_output =
            checked_element_count(checked_element_count(batch, frames)?, frequency_bins)?;
        checked_i32_value(expected_output)?;
        ensure_len(output_len, expected_output)?;
        let expected_input = checked_element_count(batch, input_len)?;
        checked_i32_value(expected_input)?;

        Ok(StftPowerShape {
            batch,
            frames,
            frequency_bins,
        })
    }

    fn validate_log_mel_f32(
        &self,
        output_len: usize,
        power_len: usize,
        batch: usize,
        frames: usize,
    ) -> Result<LogMelShape> {
        self.config.validate()?;
        validate_supported_log_mel(self.config)?;
        if batch == 0 || frames == 0 {
            return Err(Error::InvalidLength);
        }

        let expected_power = checked_element_count(
            checked_element_count(batch, frames)?,
            self.config.mel.frequency_bin_count,
        )?;
        let expected_output = checked_element_count(
            checked_element_count(batch, frames)?,
            self.config.mel.mel_bin_count,
        )?;
        ensure_len(power_len, expected_power)?;
        ensure_len(output_len, expected_output)?;
        checked_i32_value(expected_output)?;

        Ok(LogMelShape {
            batch,
            frames,
            mel_bins: self.config.mel.mel_bin_count,
            frequency_bins: self.config.mel.frequency_bin_count,
        })
    }

    fn validate_log_mel_range_f32(
        &self,
        output_len: usize,
        power_len: usize,
        batch: usize,
        first_frame: usize,
        frames: usize,
        total_frames: usize,
    ) -> Result<LogMelShape> {
        self.config.validate()?;
        validate_supported_log_mel(self.config)?;
        if batch == 0 || frames == 0 || total_frames == 0 {
            return Err(Error::InvalidLength);
        }
        let end_frame = first_frame.checked_add(frames).ok_or(Error::SizeOverflow)?;
        if end_frame > total_frames {
            return Err(Error::LengthMismatch);
        }

        let expected_power = checked_element_count(
            checked_element_count(batch, frames)?,
            self.config.mel.frequency_bin_count,
        )?;
        let expected_output = checked_element_count(
            checked_element_count(batch, total_frames)?,
            self.config.mel.mel_bin_count,
        )?;
        ensure_len(power_len, expected_power)?;
        ensure_len(output_len, expected_output)?;
        checked_i32_value(expected_output)?;

        Ok(LogMelShape {
            batch,
            frames,
            mel_bins: self.config.mel.mel_bin_count,
            frequency_bins: self.config.mel.frequency_bin_count,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StftPowerShape {
    batch: usize,
    frames: usize,
    frequency_bins: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StftSpectrumShape {
    batch: usize,
    frames: usize,
    frequency_bins: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LogMelShape {
    batch: usize,
    frames: usize,
    mel_bins: usize,
    frequency_bins: usize,
}

#[cfg(feature = "dtype-f32")]
pub fn stft_power_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
) -> Result<()> {
    stft_power_f32_range(stream, out, input, plan, 0)
}

#[cfg(feature = "dtype-f32")]
pub fn stft_power_f32_range(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
    first_frame: usize,
) -> Result<()> {
    stft_power_f32_batched_range(stream, out, input, plan, 1, input.len(), first_frame)
}

#[cfg(feature = "dtype-f32")]
pub fn stft_power_f32_batched(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
    batch: usize,
    input_len: usize,
) -> Result<()> {
    stft_power_f32_batched_range(stream, out, input, plan, batch, input_len, 0)
}

#[cfg(feature = "dtype-f32")]
pub fn stft_power_f32_batched_range(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
    batch: usize,
    input_len: usize,
    first_frame: usize,
) -> Result<()> {
    ensure_len(input.len(), checked_element_count(batch, input_len)?)?;
    #[cfg(feature = "dtype-f32")]
    let shape = plan.validate_stft_power_f32(out.len(), input_len, batch, first_frame)?;
    let spectrum_len = checked_element_count(
        checked_element_count(
            checked_element_count(shape.batch, shape.frames)?,
            shape.frequency_bins,
        )?,
        2,
    )?;
    let workspace_len = stft_r2c_workspace_len(shape.batch, shape.frames, plan.config.stft.n_fft)?;
    let mut spectrum = DeviceMemory::<f32>::zeroes(spectrum_len)?;
    let mut workspace = DeviceMemory::<f32>::zeroes(workspace_len)?;
    let stream = borrowed_stream(stream)?;
    cutile::audio::stft_r2c_f32(
        &stream,
        output_pointer(&mut spectrum),
        output_pointer(&mut workspace),
        input_pointer(input),
        input_pointer(&plan.window),
        input_pointer(&plan.dft_real),
        input_pointer(&plan.dft_imag),
        input_len,
        shape.batch,
        first_frame,
        shape.frames,
        plan.config.stft.n_fft,
        plan.config.stft.hop_length,
        plan.config.stft.center,
        pad_mode_code(plan.config.stft.pad_mode),
        shape.frequency_bins,
    )?;
    cutile::audio::r2c_power_f32(
        &stream,
        output_pointer(out),
        input_pointer(&spectrum),
        shape.batch,
        shape.frames,
        shape.frequency_bins,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn stft_r2c_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
) -> Result<()> {
    stft_r2c_f32_range(stream, out, input, plan, 0)
}

#[cfg(feature = "dtype-f32")]
pub fn stft_r2c_f32_range(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
    first_frame: usize,
) -> Result<()> {
    stft_r2c_f32_batched_range(stream, out, input, plan, 1, input.len(), first_frame)
}

#[cfg(feature = "dtype-f32")]
pub fn stft_r2c_f32_batched(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
    batch: usize,
    input_len: usize,
) -> Result<()> {
    stft_r2c_f32_batched_range(stream, out, input, plan, batch, input_len, 0)
}

#[cfg(feature = "dtype-f32")]
pub fn stft_r2c_f32_batched_range(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
    batch: usize,
    input_len: usize,
    first_frame: usize,
) -> Result<()> {
    ensure_len(input.len(), checked_element_count(batch, input_len)?)?;
    #[cfg(feature = "dtype-f32")]
    let shape = plan.validate_stft_r2c_f32(out.len(), input_len, batch, first_frame)?;
    let workspace_len = stft_r2c_workspace_len(shape.batch, shape.frames, plan.config.stft.n_fft)?;
    let mut workspace = DeviceMemory::<f32>::zeroes(workspace_len)?;
    let stream = borrowed_stream(stream)?;
    cutile::audio::stft_r2c_f32(
        &stream,
        output_pointer(out),
        output_pointer(&mut workspace),
        input_pointer(input),
        input_pointer(&plan.window),
        input_pointer(&plan.dft_real),
        input_pointer(&plan.dft_imag),
        input_len,
        shape.batch,
        first_frame,
        shape.frames,
        plan.config.stft.n_fft,
        plan.config.stft.hop_length,
        plan.config.stft.center,
        pad_mode_code(plan.config.stft.pad_mode),
        shape.frequency_bins,
    )
}

fn stft_r2c_workspace_len(batch: usize, frames: usize, n_fft: usize) -> Result<usize> {
    checked_element_count(
        checked_element_count(checked_element_count(batch, frames)?, n_fft)?,
        2,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn log_mel_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    power: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
    frames: usize,
) -> Result<()> {
    #[cfg(feature = "dtype-f32")]
    let shape = plan.validate_log_mel_f32(out.len(), power.len(), 1, frames)?;
    log_mel_f32_inner(stream, out, power, plan, shape, 0, shape.frames)
}

#[cfg(feature = "dtype-f32")]
pub fn log_mel_f32_batched(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    power: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
    batch: usize,
    frames: usize,
) -> Result<()> {
    #[cfg(feature = "dtype-f32")]
    let shape = plan.validate_log_mel_f32(out.len(), power.len(), batch, frames)?;
    log_mel_f32_inner(stream, out, power, plan, shape, 0, shape.frames)
}

#[cfg(feature = "dtype-f32")]
pub fn log_mel_f32_range(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    power: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
    first_frame: usize,
    frames: usize,
    total_frames: usize,
) -> Result<()> {
    let shape = plan.validate_log_mel_range_f32(
        out.len(),
        power.len(),
        1,
        first_frame,
        frames,
        total_frames,
    )?;
    log_mel_f32_inner(stream, out, power, plan, shape, first_frame, total_frames)
}

#[cfg(feature = "dtype-f32")]
pub fn convert_log_mel_layout_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    batch: usize,
    frames: usize,
    mel_bins: usize,
    input_layout: AudioFeatureLayout,
    output_layout: AudioFeatureLayout,
) -> Result<()> {
    if batch == 0 || frames == 0 || mel_bins == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(checked_element_count(batch, frames)?, mel_bins)?;
    ensure_len(input.len(), len)?;
    ensure_len(out.len(), len)?;

    let (dimensions, input_strides) =
        log_mel_layout_materialize_shape(batch, frames, mel_bins, input_layout, output_layout)?;
    let stream = borrowed_stream(stream)?;
    cutile::shape::materialize_rank4_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        dimensions,
        input_strides,
    )
}

fn log_mel_f32_inner(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    power: &impl DeviceSlice<f32>,
    plan: &AudioFrontendPlan,
    shape: LogMelShape,
    first_frame: usize,
    output_frame_stride: usize,
) -> Result<()> {
    let log_config = plan.config.log_mel;
    let dynamic_range = log_config.dynamic_range.ok_or(Error::MissingParameter {
        op: "log_mel_f32".into(),
        parameter: "dynamic_range".into(),
    })?;
    let layout = match plan.config.layout {
        AudioFeatureLayout::FramesFirst => 0,
        AudioFeatureLayout::MelsFirst => 1,
    };

    let stream = borrowed_stream(stream)?;
    if let Some(reference_max) = log_config.reference_max {
        cutile::audio::log_mel_f32(
            &stream,
            output_pointer(out),
            input_pointer(power),
            input_pointer(&plan.mel_filters),
            shape.batch,
            shape.frames,
            shape.mel_bins,
            shape.frequency_bins,
            log_config.floor,
            reference_max,
            dynamic_range,
            log_config.offset,
            log_config.scale,
            layout,
            first_frame,
            output_frame_stride,
        )
    } else {
        let partial_per_batch = log_mel_dynamic_max_partial_len(shape.frames, shape.mel_bins)?;
        let partial_len = checked_element_count(shape.batch, partial_per_batch)?;
        let mut partial = DeviceMemory::<f32>::zeroes(partial_len)?;
        let mut reference_max = DeviceMemory::<f32>::zeroes(shape.batch)?;
        cutile::audio::log_mel_dynamic_max_f32(
            &stream,
            output_pointer(&mut reference_max),
            output_pointer(&mut partial),
            input_pointer(power),
            input_pointer(&plan.mel_filters),
            shape.batch,
            shape.frames,
            shape.mel_bins,
            shape.frequency_bins,
            log_config.floor,
            partial_per_batch,
        )?;
        cutile::audio::log_mel_dynamic_reference_f32(
            &stream,
            output_pointer(out),
            input_pointer(power),
            input_pointer(&plan.mel_filters),
            input_pointer(&reference_max),
            shape.batch,
            shape.frames,
            shape.mel_bins,
            shape.frequency_bins,
            log_config.floor,
            dynamic_range,
            log_config.offset,
            log_config.scale,
            layout,
            first_frame,
            output_frame_stride,
        )
    }
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

fn validate_supported_stft_power(config: StftConfig) -> Result<()> {
    if config.n_fft != 400 || config.frequency_bin_count() != 201 {
        return Err(Error::UnsupportedConfiguration {
            op: "stft_power_f32".into(),
            reason: "requires n_fft = 400 and frequency bins = 201".into(),
        });
    }
    if config.win_length != config.n_fft {
        return Err(Error::UnsupportedConfiguration {
            op: "stft_power_f32".into(),
            reason: "requires win_length = n_fft".into(),
        });
    }
    if config.normalization != SpectralNormalization::None {
        return Err(Error::UnsupportedConfiguration {
            op: "stft_power_f32".into(),
            reason: "requires unnormalized spectra".into(),
        });
    }
    Ok(())
}

fn validate_padding(config: StftConfig, input_len: usize) -> Result<()> {
    if !config.center || config.pad_mode != PadMode::Reflect {
        return Ok(());
    }
    let pad = config.n_fft / 2;
    if input_len <= pad {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

const fn pad_mode_code(mode: PadMode) -> i32 {
    match mode {
        PadMode::Reflect => 0,
        PadMode::Zero => 1,
    }
}

fn validate_frame_range(
    config: StftConfig,
    input_len: usize,
    first_frame: usize,
    frames: usize,
) -> Result<()> {
    if frames == 0 {
        return Err(Error::InvalidLength);
    }
    let total_frames = config.frame_count(input_len)?;
    let end_frame = first_frame.checked_add(frames).ok_or(Error::SizeOverflow)?;
    if end_frame > total_frames {
        return Err(Error::LengthMismatch);
    }
    Ok(())
}

fn validate_supported_log_mel(config: AudioFrontendConfig) -> Result<()> {
    if config.mel.frequency_bin_count != 201 || config.mel.mel_bin_count != 128 {
        return Err(Error::UnsupportedConfiguration {
            op: "log_mel_f32".into(),
            reason: "requires frequency bins = 201 and mel bins = 128".into(),
        });
    }
    if config.log_mel.dynamic_range.is_none() {
        return Err(Error::MissingParameter {
            op: "log_mel_f32".into(),
            parameter: "dynamic_range".into(),
        });
    }
    Ok(())
}

fn log_mel_layout_materialize_shape(
    batch: usize,
    frames: usize,
    mel_bins: usize,
    input_layout: AudioFeatureLayout,
    output_layout: AudioFeatureLayout,
) -> Result<([usize; 4], [usize; 4])> {
    let feature_len = checked_element_count(frames, mel_bins)?;
    let dimensions = match output_layout {
        AudioFeatureLayout::FramesFirst => [batch, frames, mel_bins, 1],
        AudioFeatureLayout::MelsFirst => [batch, mel_bins, frames, 1],
    };
    let input_strides = match (input_layout, output_layout) {
        (AudioFeatureLayout::FramesFirst, AudioFeatureLayout::FramesFirst) => {
            [feature_len, mel_bins, 1, 1]
        }
        (AudioFeatureLayout::FramesFirst, AudioFeatureLayout::MelsFirst) => {
            [feature_len, 1, mel_bins, 1]
        }
        (AudioFeatureLayout::MelsFirst, AudioFeatureLayout::FramesFirst) => {
            [feature_len, 1, frames, 1]
        }
        (AudioFeatureLayout::MelsFirst, AudioFeatureLayout::MelsFirst) => {
            [feature_len, frames, 1, 1]
        }
    };
    Ok((dimensions, input_strides))
}

fn log_mel_dynamic_max_partial_len(frames: usize, mel_bins: usize) -> Result<usize> {
    let output_len = checked_element_count(frames, mel_bins)?;
    if output_len == 0 {
        return Err(Error::InvalidLength);
    }
    let partial_len = output_len.div_ceil(128);
    if partial_len > 1024 {
        return Err(Error::UnsupportedBlockCount {
            op: "dynamic log_mel max".into(),
            blocks: partial_len,
        });
    }
    Ok(partial_len)
}

fn create_window(config: StftConfig) -> Result<DeviceMemory<f32>> {
    let window = stft_window(config.window, config.win_length);
    let dm = DeviceMemory::from_slice(&window)?;
    Ok(dm)
}

fn create_dft_tables(
    n_fft: usize,
    frequency_bin_count: usize,
) -> Result<(DeviceMemory<f32>, DeviceMemory<f32>)> {
    let (real, imag) = dft_tables(n_fft, frequency_bin_count)?;
    let real = DeviceMemory::from_slice(&real)?;
    let imag = DeviceMemory::from_slice(&imag)?;
    Ok((real, imag))
}

fn create_mel_filters(config: MelFilterConfig) -> Result<DeviceMemory<f32>> {
    let filters = slaney_mel_filters(config)?;
    let dm = DeviceMemory::from_slice(&filters)?;
    Ok(dm)
}

pub(crate) fn stft_window(kind: WindowKind, len: usize) -> Vec<f32> {
    match kind {
        WindowKind::Rectangular => rectangular_window(len),
        WindowKind::PeriodicHann => periodic_hann_window(len),
        WindowKind::PeriodicHamming => periodic_hamming_window(len),
    }
}

fn rectangular_window(len: usize) -> Vec<f32> {
    vec![1.0; len]
}

fn periodic_hann_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let angle = 2.0 * std::f32::consts::PI * index as f32 / len as f32;
            0.5 * (1.0 - angle.cos())
        })
        .collect()
}

fn periodic_hamming_window(len: usize) -> Vec<f32> {
    const ALPHA: f32 = 0.54;
    const BETA: f32 = 1.0 - ALPHA;
    (0..len)
        .map(|index| {
            let angle = 2.0 * std::f32::consts::PI * index as f32 / len as f32;
            ALPHA - BETA * angle.cos()
        })
        .collect()
}

pub(crate) fn dft_tables(n_fft: usize, frequency_bin_count: usize) -> Result<(Vec<f32>, Vec<f32>)> {
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

pub(crate) fn slaney_mel_filters(config: MelFilterConfig) -> Result<Vec<f32>> {
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
    use singe_cuda::{
        context::Context as CudaContext, device::Device, memory::DeviceMemory, stream::StreamFlags,
    };

    use crate::{
        audio::{LogMelConfig, SpectrumKind},
        cpu::audio as cpu_audio,
        cuda::audio::{
            self, AudioFeatureLayout, AudioFrontendConfig, AudioFrontendPlan, MelFilterConfig,
            PadMode, SpectralNormalization, StftConfig, WindowKind, dft_tables,
            periodic_hamming_window, periodic_hann_window, rectangular_window, slaney_mel_filters,
            validate_padding,
        },
        error::{Error, Result},
    };

    fn standard_log_mel_config() -> AudioFrontendConfig {
        let stft = StftConfig {
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            center: true,
            pad_mode: PadMode::Reflect,
            window: WindowKind::PeriodicHann,
            spectrum: SpectrumKind::OneSide,
            normalization: SpectralNormalization::None,
            drop_last_frame: true,
        };
        AudioFrontendConfig {
            stft,
            mel: MelFilterConfig {
                sample_rate: 16_000,
                frequency_bin_count: stft.frequency_bin_count(),
                mel_bin_count: 128,
                min_frequency: 0.0,
                max_frequency: 8_000.0,
            },
            log_mel: LogMelConfig {
                floor: 1e-10,
                reference_max: Some(1.5),
                dynamic_range: Some(8.0),
                offset: 4.0,
                scale: 4.0,
            },
            layout: AudioFeatureLayout::MelsFirst,
        }
    }

    #[test]
    fn periodic_hann_matches_reference_values() -> Result<()> {
        let window = periodic_hann_window(4);
        assert_eq!(window[0], 0.0);
        assert!((window[1] - 0.5).abs() < 1e-6);
        assert!((window[2] - 1.0).abs() < 1e-6);
        assert!((window[3] - 0.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn rectangular_window_is_all_ones() -> Result<()> {
        let window = rectangular_window(5);
        assert_eq!(window, vec![1.0; 5]);
        Ok(())
    }

    #[test]
    fn periodic_hamming_matches_reference_values() -> Result<()> {
        let window = periodic_hamming_window(4);
        assert!((window[0] - 0.08).abs() < 1e-6);
        assert!((window[1] - 0.54).abs() < 1e-6);
        assert!((window[2] - 1.0).abs() < 1e-6);
        assert!((window[3] - 0.54).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn dft_tables_are_row_major_by_frequency() -> Result<()> {
        let (real, imag) = dft_tables(4, 3)?;
        assert_eq!(real.len(), 12);
        assert_eq!(imag.len(), 12);
        assert!((real[0] - 1.0).abs() < 1e-6);
        assert!(imag[0].abs() < 1e-6);
        assert!((real[5] - 0.0).abs() < 1e-6);
        assert!((imag[5] + 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn standard_log_mel_profile_validates_as_general_config() -> Result<()> {
        let config = standard_log_mel_config();
        config.validate()?;
        assert_eq!(config.stft.frequency_bin_count(), 201);
        assert_eq!(config.stft.frame_count(16_000)?, 100);
        Ok(())
    }

    #[test]
    fn mel_filters_have_expected_shape_and_non_negative_weights() -> Result<()> {
        let config = standard_log_mel_config().mel;
        let filters = slaney_mel_filters(config)?;
        assert_eq!(filters.len(), 128 * 201);
        assert!(filters.iter().all(|weight| *weight >= 0.0));
        assert!(filters.iter().any(|weight| *weight > 0.0));
        Ok(())
    }

    #[test]
    fn frontend_validation_rejects_mismatched_frequency_bins() -> Result<()> {
        let mut config = standard_log_mel_config();
        config.mel.frequency_bin_count -= 1;
        assert!(matches!(config.validate(), Err(Error::LengthMismatch)));
        Ok(())
    }

    #[test]
    fn frontend_validation_rejects_invalid_log_floor() -> Result<()> {
        let mut config = standard_log_mel_config();
        config.log_mel.floor = 0.0;
        assert!(matches!(config.validate(), Err(Error::InvalidLength)));
        Ok(())
    }

    #[test]
    fn stft_power_f32_matches_host_direct_dft() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let config = standard_log_mel_config();
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(640);
        let frames = config.stft.frame_count(input_host.len())?;
        let output_len = frames * config.stft.frequency_bin_count();
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::stft_power_f32(&stream, &mut actual, &input, &plan)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected = cpu_audio::stft_power(&input_host, config.stft.into())?;
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-2);
        Ok(())
    }

    #[test]
    fn stft_r2c_f32_matches_host_direct_dft() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let config = standard_log_mel_config();
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(640);
        let frames = config.stft.frame_count(input_host.len())?;
        let output_len = frames * config.stft.frequency_bin_count() * 2;
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::stft_r2c_f32(&stream, &mut actual, &input, &plan)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected = cpu_audio::stft_r2c(&input_host, config.stft.into())?;
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-2);
        Ok(())
    }

    #[test]
    fn stft_power_f32_matches_host_periodic_hamming_window() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let mut config = standard_log_mel_config();
        config.stft.window = WindowKind::PeriodicHamming;
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(640);
        let frames = config.stft.frame_count(input_host.len())?;
        let output_len = frames * config.stft.frequency_bin_count();
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::stft_power_f32(&stream, &mut actual, &input, &plan)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected = cpu_audio::stft_power(&input_host, config.stft.into())?;
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-2);
        Ok(())
    }

    #[test]
    fn stft_power_f32_matches_host_centered_zero_padding() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let mut config = standard_log_mel_config();
        config.stft.pad_mode = PadMode::Zero;
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(320);
        let frames = config.stft.frame_count(input_host.len())?;
        let output_len = frames * config.stft.frequency_bin_count();
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::stft_power_f32(&stream, &mut actual, &input, &plan)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected = cpu_audio::stft_power(&input_host, config.stft.into())?;
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-2);
        Ok(())
    }

    #[test]
    fn stft_power_f32_matches_host_non_centered_zero_padding_config() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let mut config = standard_log_mel_config();
        config.stft.center = false;
        config.stft.pad_mode = PadMode::Zero;
        config.stft.drop_last_frame = false;
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(960);
        let frames = config.stft.frame_count(input_host.len())?;
        let output_len = frames * config.stft.frequency_bin_count();
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::stft_power_f32(&stream, &mut actual, &input, &plan)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected = cpu_audio::stft_power(&input_host, config.stft.into())?;
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-2);
        Ok(())
    }

    #[test]
    fn stft_power_f32_matches_host_very_short_centered_zero_padded_input() -> Result<()> {
        let mut config = standard_log_mel_config();
        config.stft.pad_mode = PadMode::Zero;
        config.stft.drop_last_frame = false;
        assert_stft_power_matches_host(config, 32)
    }

    #[test]
    fn stft_power_f32_matches_host_exact_frame_non_centered_input() -> Result<()> {
        let mut config = standard_log_mel_config();
        config.stft.center = false;
        config.stft.pad_mode = PadMode::Zero;
        config.stft.drop_last_frame = false;
        assert_stft_power_matches_host(config, config.stft.n_fft)
    }

    #[test]
    fn stft_power_f32_matches_host_non_centered_partial_final_frame_input() -> Result<()> {
        let mut config = standard_log_mel_config();
        config.stft.center = false;
        config.stft.pad_mode = PadMode::Zero;
        config.stft.drop_last_frame = false;
        let input_len = config.stft.n_fft + config.stft.hop_length - 1;
        assert_eq!(config.stft.frame_count(input_len)?, 1);
        assert_stft_power_matches_host(config, input_len)
    }

    #[test]
    fn stft_power_f32_matches_host_centered_reflect_finish_padding() -> Result<()> {
        let mut config = standard_log_mel_config();
        config.stft.drop_last_frame = false;
        assert_stft_power_matches_host(config, 401)
    }

    #[test]
    fn stft_power_f32_matches_host_centered_reflect_sub_fft_input() -> Result<()> {
        let mut config = standard_log_mel_config();
        config.stft.drop_last_frame = false;
        assert_stft_power_matches_host(config, 320)
    }

    #[test]
    fn stft_r2c_f32_range_matches_host_direct_dft_slice() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let config = standard_log_mel_config();
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(960);
        let first_frame = 2;
        let frames = 3;
        let frequency_bins = config.stft.frequency_bin_count();
        let output_len = frames * frequency_bins * 2;
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::stft_r2c_f32_range(&stream, &mut actual, &input, &plan, first_frame)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected_full = cpu_audio::stft_r2c(&input_host, config.stft.into())?;
        let start = first_frame * frequency_bins * 2;
        let end = start + output_len;
        singe_core::assert_close!(&actual, &expected_full[start..end], 2e-2);
        Ok(())
    }

    #[test]
    fn stft_power_f32_range_matches_host_direct_dft_slice() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let config = standard_log_mel_config();
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(960);
        let first_frame = 1;
        let frames = 4;
        let frequency_bins = config.stft.frequency_bin_count();
        let output_len = frames * frequency_bins;
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::stft_power_f32_range(&stream, &mut actual, &input, &plan, first_frame)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected_full = cpu_audio::stft_power(&input_host, config.stft.into())?;
        let start = first_frame * frequency_bins;
        let end = start + output_len;
        singe_core::assert_close!(&actual, &expected_full[start..end], 2e-2);
        Ok(())
    }

    #[test]
    fn stft_power_f32_batched_matches_host_direct_dft() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let config = standard_log_mel_config();
        let plan = AudioFrontendPlan::create(config)?;
        let batch = 2;
        let input_len = 640;
        let input_host = deterministic_audio_input(batch * input_len);
        let frames = config.stft.frame_count(input_len)?;
        let output_len = batch * frames * config.stft.frequency_bin_count();
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::stft_power_f32_batched(&stream, &mut actual, &input, &plan, batch, input_len)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let mut expected = Vec::with_capacity(output_len);
        for batch_index in 0..batch {
            let start = batch_index * input_len;
            let end = start + input_len;
            expected.extend(cpu_audio::stft_power(
                &input_host[start..end],
                config.stft.into(),
            )?);
        }
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-2);
        Ok(())
    }

    #[test]
    fn stft_r2c_f32_batched_range_matches_host_direct_dft_slice() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let config = standard_log_mel_config();
        let plan = AudioFrontendPlan::create(config)?;
        let batch = 2;
        let input_len = 960;
        let input_host = deterministic_audio_input(batch * input_len);
        let first_frame = 1;
        let frames = 4;
        let frequency_bins = config.stft.frequency_bin_count();
        let output_len = batch * frames * frequency_bins * 2;
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        audio::stft_r2c_f32_batched_range(
            &stream,
            &mut actual,
            &input,
            &plan,
            batch,
            input_len,
            first_frame,
        )?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let mut expected = Vec::with_capacity(output_len);
        for batch_index in 0..batch {
            let input_start = batch_index * input_len;
            let input_end = input_start + input_len;
            let expected_full =
                cpu_audio::stft_r2c(&input_host[input_start..input_end], config.stft.into())?;
            let output_start = first_frame * frequency_bins * 2;
            let output_end = output_start + frames * frequency_bins * 2;
            expected.extend_from_slice(&expected_full[output_start..output_end]);
        }
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-2);
        Ok(())
    }

    #[test]
    fn log_mel_f32_matches_host_projection() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let config = standard_log_mel_config();
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(640);
        let power_host = cpu_audio::stft_power(&input_host, config.stft.into())?;
        let frames = config.stft.frame_count(input_host.len())?;
        let output_len = frames * config.mel.mel_bin_count;
        let power = DeviceMemory::<f32>::from_slice(&power_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::log_mel_f32(&stream, &mut actual, &power, &plan, frames)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected = cpu_audio::log_mel(&power_host, frames, config.into())?;
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-4);
        Ok(())
    }

    #[test]
    fn log_mel_f32_matches_host_centered_reflect_sub_fft_input() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let mut config = standard_log_mel_config();
        config.stft.drop_last_frame = false;
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(320);
        let frames = config.stft.frame_count(input_host.len())?;
        assert_eq!(frames, 3);
        let power_len = frames * config.stft.frequency_bin_count();
        let output_len = frames * config.mel.mel_bin_count;
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut power = DeviceMemory::<f32>::zeroes(power_len)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        audio::stft_power_f32(&stream, &mut power, &input, &plan)?;
        audio::log_mel_f32(&stream, &mut actual, &power, &plan, frames)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected_power = cpu_audio::stft_power(&input_host, config.stft.into())?;
        let expected = cpu_audio::log_mel(&expected_power, frames, config.into())?;
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 3e-2);
        Ok(())
    }

    #[test]
    fn log_mel_f32_batched_matches_host_projection() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let config = standard_log_mel_config();
        let plan = AudioFrontendPlan::create(config)?;
        let batch = 2;
        let input_len = 640;
        let frames = config.stft.frame_count(input_len)?;
        let feature_len = frames * config.mel.mel_bin_count;
        let mut power_host = Vec::with_capacity(batch * frames * config.mel.frequency_bin_count);
        let mut expected = Vec::with_capacity(batch * feature_len);
        let input_host = deterministic_audio_input(batch * input_len);
        for batch_index in 0..batch {
            let input_start = batch_index * input_len;
            let input_end = input_start + input_len;
            let batch_power =
                cpu_audio::stft_power(&input_host[input_start..input_end], config.stft.into())?;
            expected.extend(cpu_audio::log_mel(&batch_power, frames, config.into())?);
            power_host.extend(batch_power);
        }
        let power = DeviceMemory::<f32>::from_slice(&power_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(batch * feature_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::log_mel_f32_batched(&stream, &mut actual, &power, &plan, batch, frames)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-4);
        Ok(())
    }

    #[test]
    fn log_mel_f32_batched_matches_host_dynamic_reference_max_per_batch() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let mut config = standard_log_mel_config();
        config.log_mel.reference_max = None;
        let plan = AudioFrontendPlan::create(config)?;
        let batch = 2;
        let input_len = 640;
        let frames = config.stft.frame_count(input_len)?;
        let feature_len = frames * config.mel.mel_bin_count;
        let mut power_host = Vec::with_capacity(batch * frames * config.mel.frequency_bin_count);
        let mut expected = Vec::with_capacity(batch * feature_len);
        let input_host = deterministic_audio_input(batch * input_len);
        for batch_index in 0..batch {
            let input_start = batch_index * input_len;
            let input_end = input_start + input_len;
            let batch_power =
                cpu_audio::stft_power(&input_host[input_start..input_end], config.stft.into())?;
            expected.extend(cpu_audio::log_mel(&batch_power, frames, config.into())?);
            power_host.extend(batch_power);
        }
        let power = DeviceMemory::<f32>::from_slice(&power_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(batch * feature_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::log_mel_f32_batched(&stream, &mut actual, &power, &plan, batch, frames)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-4);
        Ok(())
    }

    #[test]
    fn log_mel_f32_matches_host_dynamic_reference_max() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let mut config = standard_log_mel_config();
        config.log_mel.reference_max = None;
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(640);
        let power_host = cpu_audio::stft_power(&input_host, config.stft.into())?;
        let frames = config.stft.frame_count(input_host.len())?;
        let output_len = frames * config.mel.mel_bin_count;
        let power = DeviceMemory::<f32>::from_slice(&power_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::log_mel_f32(&stream, &mut actual, &power, &plan, frames)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected = cpu_audio::log_mel(&power_host, frames, config.into())?;
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-4);
        Ok(())
    }

    #[test]
    fn log_mel_f32_range_writes_absolute_mel_frame_columns() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let config = standard_log_mel_config();
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(960);
        let power_full = cpu_audio::stft_power(&input_host, config.stft.into())?;
        let total_frames = config.stft.frame_count(input_host.len())?;
        let first_frame = 2;
        let frames = 3;
        let frequency_bins = config.mel.frequency_bin_count;
        let power_start = first_frame * frequency_bins;
        let power_end = power_start + frames * frequency_bins;
        let power = DeviceMemory::<f32>::from_slice(&power_full[power_start..power_end])?;
        let output_len = total_frames * config.mel.mel_bin_count;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        audio::log_mel_f32_range(
            &stream,
            &mut actual,
            &power,
            &plan,
            first_frame,
            frames,
            total_frames,
        )?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected_full = cpu_audio::log_mel(&power_full, total_frames, config.into())?;
        let expected = cpu_audio::sparse_log_mel_range(
            &expected_full,
            config.into(),
            first_frame,
            frames,
            total_frames,
        );
        singe_core::assert_close!(&actual, &expected, 2e-4);
        Ok(())
    }

    #[test]
    fn convert_log_mel_layout_f32_converts_batched_frames_first_to_mels_first() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let batch = 2;
        let frames = 3;
        let mel_bins = 5;
        let input_host = deterministic_feature_input(batch * frames * mel_bins);
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(input_host.len())?;

        audio::convert_log_mel_layout_f32(
            &stream,
            &mut actual,
            &input,
            batch,
            frames,
            mel_bins,
            AudioFeatureLayout::FramesFirst,
            AudioFeatureLayout::MelsFirst,
        )?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected = cpu_audio::convert_log_mel_layout(
            &input_host,
            batch,
            frames,
            mel_bins,
            AudioFeatureLayout::FramesFirst.into(),
            AudioFeatureLayout::MelsFirst.into(),
        );
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn convert_log_mel_layout_f32_converts_batched_mels_first_to_frames_first() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let batch = 2;
        let frames = 4;
        let mel_bins = 3;
        let input_host = deterministic_feature_input(batch * frames * mel_bins);
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(input_host.len())?;

        audio::convert_log_mel_layout_f32(
            &stream,
            &mut actual,
            &input,
            batch,
            frames,
            mel_bins,
            AudioFeatureLayout::MelsFirst,
            AudioFeatureLayout::FramesFirst,
        )?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected = cpu_audio::convert_log_mel_layout(
            &input_host,
            batch,
            frames,
            mel_bins,
            AudioFeatureLayout::MelsFirst.into(),
            AudioFeatureLayout::FramesFirst.into(),
        );
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn convert_log_mel_layout_f32_handles_unbatched_noop_layout() -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let batch = 1;
        let frames = 4;
        let mel_bins = 6;
        let input_host = deterministic_feature_input(batch * frames * mel_bins);
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(input_host.len())?;

        audio::convert_log_mel_layout_f32(
            &stream,
            &mut actual,
            &input,
            batch,
            frames,
            mel_bins,
            AudioFeatureLayout::FramesFirst,
            AudioFeatureLayout::FramesFirst,
        )?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        assert_eq!(actual, input_host);
        Ok(())
    }

    #[test]
    fn stft_power_validation_rejects_short_reflect_input() -> Result<()> {
        let config = standard_log_mel_config();
        assert!(matches!(
            validate_padding(config.stft, 200),
            Err(Error::InvalidLength)
        ));
        Ok(())
    }

    #[test]
    fn stft_power_validation_allows_short_centered_zero_padding_input() -> Result<()> {
        let mut config = standard_log_mel_config();
        config.stft.pad_mode = PadMode::Zero;
        validate_padding(config.stft, 200)?;
        Ok(())
    }

    fn deterministic_audio_input(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let a = (index % 17) as f32 * 0.03125;
                let b = (index % 7) as f32 * 0.015625;
                a - b - 0.25
            })
            .collect()
    }

    fn deterministic_feature_input(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| (index % 23) as f32 * 0.5 - (index / 7) as f32 * 0.125)
            .collect()
    }

    fn assert_stft_power_matches_host(config: AudioFrontendConfig, input_len: usize) -> Result<()> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let plan = AudioFrontendPlan::create(config)?;
        let input_host = deterministic_audio_input(input_len);
        let frames = config.stft.frame_count(input_host.len())?;
        let output_len = frames * config.stft.frequency_bin_count();
        let input = DeviceMemory::<f32>::from_slice(&input_host)?;
        let mut actual = DeviceMemory::<f32>::zeroes(output_len)?;

        #[cfg(feature = "dtype-f32")]
        audio::stft_power_f32(&stream, &mut actual, &input, &plan)?;
        stream.synchronize()?;

        let actual = actual.copy_to_host_vec()?;
        let expected = cpu_audio::stft_power(&input_host, config.stft.into())?;
        assert_eq!(actual.len(), expected.len());
        singe_core::assert_close!(&actual, &expected, 2e-2);
        Ok(())
    }
}
