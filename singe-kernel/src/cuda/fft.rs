//! Small FFT helpers, interleaved complex transforms, and normalization.

use std::{f32::consts::PI, mem::MaybeUninit, sync::Arc};

use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    context::Context,
    error::Error as CudaError,
    memory::DeviceMemory,
    stream::Stream,
    view::{DeviceInput, DeviceOutput, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{input_pointer, output_pointer},
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value, ensure_len},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftDirection {
    Forward,
    Inverse,
}

impl FftDirection {
    const fn twiddle_imag_scale(self) -> f32 {
        match self {
            Self::Forward => 1.0,
            Self::Inverse => -1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftTransformKind {
    ComplexToComplex,
    RealToComplex,
    ComplexToReal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftPrecision {
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftComplexLayout {
    Interleaved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftNormalization {
    Backward,
    Forward,
    Ortho,
}

impl FftNormalization {
    fn scale(self, n: usize, direction: FftDirection) -> Result<f32> {
        if n == 0 {
            return Err(Error::InvalidLength);
        }
        Ok(match self {
            Self::Backward => match direction {
                FftDirection::Forward => 1.0,
                FftDirection::Inverse => 1.0 / n as f32,
            },
            Self::Forward => match direction {
                FftDirection::Forward => 1.0 / n as f32,
                FftDirection::Inverse => 1.0,
            },
            Self::Ortho => 1.0 / (n as f32).sqrt(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FftPlanConfig {
    transform: FftTransformKind,
    precision: FftPrecision,
    complex_layout: FftComplexLayout,
    direction: FftDirection,
    n: usize,
    batch: usize,
    factors: [usize; 3],
    packing: usize,
    input_stride: usize,
    output_stride: usize,
    input_distance: usize,
    output_distance: usize,
}

impl FftPlanConfig {
    pub fn c2c_f32_interleaved_supported(
        n: usize,
        batch: usize,
        direction: FftDirection,
    ) -> Result<Self> {
        let (factors, packing) = supported_f32_plan_shape(n).ok_or(Error::UnsupportedFftSize {
            transform: "c2c".into(),
            n,
        })?;
        Self::c2c_f32_interleaved(n, batch, factors, packing, direction)
    }

    pub fn c2r_f32_interleaved_supported(n: usize, batch: usize) -> Result<Self> {
        let (factors, packing) = supported_f32_plan_shape(n).ok_or(Error::UnsupportedFftSize {
            transform: "c2r".into(),
            n,
        })?;
        Self::c2r_f32_interleaved(n, batch, factors, packing)
    }

    pub fn r2c_f32_interleaved_supported(n: usize, batch: usize) -> Result<Self> {
        let (factors, packing) = supported_f32_plan_shape(n).ok_or(Error::UnsupportedFftSize {
            transform: "r2c".into(),
            n,
        })?;
        Self::r2c_f32_interleaved(n, batch, factors, packing)
    }

    pub fn c2c_f32_interleaved(
        n: usize,
        batch: usize,
        factors: [usize; 3],
        packing: usize,
        direction: FftDirection,
    ) -> Result<Self> {
        let distance = checked_element_count(n, 2)?;
        let config = Self {
            transform: FftTransformKind::ComplexToComplex,
            precision: FftPrecision::F32,
            complex_layout: FftComplexLayout::Interleaved,
            direction,
            n,
            batch,
            factors,
            packing,
            input_stride: 2,
            output_stride: 2,
            input_distance: distance,
            output_distance: distance,
        };
        validate_fft_plan_config(config)?;
        Ok(config)
    }

    pub fn c2r_f32_interleaved(
        n: usize,
        batch: usize,
        factors: [usize; 3],
        packing: usize,
    ) -> Result<Self> {
        let config = Self {
            transform: FftTransformKind::ComplexToReal,
            precision: FftPrecision::F32,
            complex_layout: FftComplexLayout::Interleaved,
            direction: FftDirection::Inverse,
            n,
            batch,
            factors,
            packing,
            input_stride: 2,
            output_stride: 1,
            input_distance: checked_element_count(n / 2 + 1, 2)?,
            output_distance: n,
        };
        validate_fft_plan_config(config)?;
        Ok(config)
    }

    pub fn r2c_f32_interleaved(
        n: usize,
        batch: usize,
        factors: [usize; 3],
        packing: usize,
    ) -> Result<Self> {
        let config = Self {
            transform: FftTransformKind::RealToComplex,
            precision: FftPrecision::F32,
            complex_layout: FftComplexLayout::Interleaved,
            direction: FftDirection::Forward,
            n,
            batch,
            factors,
            packing,
            input_stride: 1,
            output_stride: 2,
            input_distance: n,
            output_distance: checked_element_count(n / 2 + 1, 2)?,
        };
        validate_fft_plan_config(config)?;
        Ok(config)
    }

    pub fn with_strided_layout(
        mut self,
        input_stride: usize,
        output_stride: usize,
        input_distance: usize,
        output_distance: usize,
    ) -> Result<Self> {
        self.input_stride = input_stride;
        self.output_stride = output_stride;
        self.input_distance = input_distance;
        self.output_distance = output_distance;
        validate_fft_plan_config(self)?;
        Ok(self)
    }

    pub const fn transform(self) -> FftTransformKind {
        self.transform
    }

    pub const fn precision(self) -> FftPrecision {
        self.precision
    }

    pub const fn complex_layout(self) -> FftComplexLayout {
        self.complex_layout
    }

    pub const fn direction(self) -> FftDirection {
        self.direction
    }

    pub const fn n(self) -> usize {
        self.n
    }

    pub const fn batch(self) -> usize {
        self.batch
    }

    pub const fn factors(self) -> [usize; 3] {
        self.factors
    }

    pub const fn packing(self) -> usize {
        self.packing
    }

    pub const fn input_stride(self) -> usize {
        self.input_stride
    }

    pub const fn output_stride(self) -> usize {
        self.output_stride
    }

    pub const fn input_distance(self) -> usize {
        self.input_distance
    }

    pub const fn output_distance(self) -> usize {
        self.output_distance
    }

    const fn is_c2c_f32_interleaved(self) -> bool {
        matches!(self.transform, FftTransformKind::ComplexToComplex)
            && matches!(self.precision, FftPrecision::F32)
            && matches!(self.complex_layout, FftComplexLayout::Interleaved)
    }

    const fn is_r2c_f32_interleaved(self) -> bool {
        matches!(self.transform, FftTransformKind::RealToComplex)
            && matches!(self.precision, FftPrecision::F32)
            && matches!(self.complex_layout, FftComplexLayout::Interleaved)
            && matches!(self.direction, FftDirection::Forward)
    }

    const fn is_c2r_f32_interleaved(self) -> bool {
        matches!(self.transform, FftTransformKind::ComplexToReal)
            && matches!(self.precision, FftPrecision::F32)
            && matches!(self.complex_layout, FftComplexLayout::Interleaved)
            && matches!(self.direction, FftDirection::Inverse)
    }
}

#[derive(Debug)]
pub struct FftPlan {
    config: FftPlanConfig,
    workspace_bytes: usize,
    twiddle_real: DeviceMemory<f32>,
    twiddle_imag: DeviceMemory<f32>,
}

impl FftPlan {
    pub fn create(context: &Arc<Context>, config: FftPlanConfig) -> Result<Self> {
        validate_fft_plan_config(config)?;
        let workspace_bytes = estimate_fft_workspace_bytes(config)?;
        match config.transform {
            FftTransformKind::ComplexToComplex
            | FftTransformKind::RealToComplex
            | FftTransformKind::ComplexToReal => {
                let (twiddle_real, twiddle_imag) = create_twiddle_tables(context, config.n)?;
                Ok(Self {
                    config,
                    workspace_bytes,
                    twiddle_real,
                    twiddle_imag,
                })
            }
        }
    }

    pub fn context(&self) -> &Arc<Context> {
        self.twiddle_real.context()
    }

    fn validate_stream(&self, stream: &Stream) -> Result<()> {
        if self.context() != stream.context() {
            return Err(CudaError::StreamContextMismatch.into());
        }
        Ok(())
    }

    pub const fn config(&self) -> FftPlanConfig {
        self.config
    }

    pub const fn n(&self) -> usize {
        self.config.n
    }

    pub const fn batch(&self) -> usize {
        self.config.batch
    }

    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    pub fn create_workspace(&self) -> Result<Option<DeviceMemory<MaybeUninit<u8>>>> {
        if self.workspace_bytes == 0 {
            Ok(None)
        } else {
            Ok(
                DeviceMemory::<u8>::create_uninit(self.context(), self.workspace_bytes)
                    .map(Some)?,
            )
        }
    }

    fn validate_workspace<W: DeviceOutput<u8> + ?Sized>(
        &self,
        workspace: Option<&W>,
    ) -> Result<()> {
        if let Some(workspace) = workspace
            && workspace.context() != self.context()
        {
            return Err(CudaError::MemoryContextMismatch.into());
        }
        self.validate_workspace_len(workspace.map(DeviceInput::len))
    }

    fn validate_workspace_len(&self, workspace_len: Option<usize>) -> Result<()> {
        match workspace_len {
            Some(workspace_len) if workspace_len >= self.workspace_bytes => Ok(()),
            Some(_) => Err(Error::LengthMismatch),
            None if self.workspace_bytes == 0 => Ok(()),
            None => Err(Error::LengthMismatch),
        }
    }

    fn validate_c2c_f32_interleaved(&self, out_len: usize, input_len: usize) -> Result<()> {
        validate_fft_plan_config(self.config)?;
        if !self.config.is_c2c_f32_interleaved() {
            return Err(Error::UnsupportedConfiguration {
                op: "fft_f32_interleaved".into(),
                reason: "requires a c2c f32 interleaved plan".into(),
            });
        }

        let lanes =
            checked_element_count(checked_element_count(self.config.batch, self.config.n)?, 2)?;
        checked_i32_value(lanes)?;
        let (input_layout, output_layout) = fft_plan_buffer_layouts(self.config)?;
        output_layout.ensure_len_at_least(out_len)?;
        input_layout.ensure_len_at_least(input_len)
    }

    fn validate_r2c_f32_interleaved(&self, out_len: usize, input_len: usize) -> Result<()> {
        validate_fft_plan_config(self.config)?;
        if !self.config.is_r2c_f32_interleaved() {
            return Err(Error::UnsupportedConfiguration {
                op: "fft_r2c_f32_interleaved".into(),
                reason: "requires an r2c f32 interleaved plan".into(),
            });
        }

        let output_lanes = checked_element_count(
            checked_element_count(self.config.batch, self.config.n / 2 + 1)?,
            2,
        )?;
        let input_values = checked_element_count(self.config.batch, self.config.n)?;
        checked_i32_value(output_lanes)?;
        checked_i32_value(input_values)?;
        let (input_layout, output_layout) = fft_plan_buffer_layouts(self.config)?;
        input_layout.ensure_len_at_least(input_len)?;
        output_layout.ensure_len_at_least(out_len)
    }

    fn validate_c2r_f32_interleaved(&self, out_len: usize, input_len: usize) -> Result<()> {
        self.validate_c2r_f32_output(out_len)?;
        let input_lanes = checked_element_count(
            checked_element_count(self.config.batch, self.config.n / 2 + 1)?,
            2,
        )?;
        checked_i32_value(input_lanes)?;
        let (input_layout, _) = fft_plan_buffer_layouts(self.config)?;
        input_layout.ensure_len_at_least(input_len)
    }

    fn validate_c2r_f32_output(&self, out_len: usize) -> Result<()> {
        validate_fft_plan_config(self.config)?;
        if !self.config.is_c2r_f32_interleaved() {
            return Err(Error::UnsupportedConfiguration {
                op: "ifft_c2r_f32_interleaved".into(),
                reason: "requires a c2r f32 interleaved plan".into(),
            });
        }

        let output_values = checked_element_count(self.config.batch, self.config.n)?;
        checked_i32_value(output_values)?;
        let (_, output_layout) = fft_plan_buffer_layouts(self.config)?;
        output_layout.ensure_len_at_least(out_len)
    }
}

pub fn estimate_fft_workspace_bytes(config: FftPlanConfig) -> Result<usize> {
    validate_fft_plan_config(config)?;
    match config.transform {
        FftTransformKind::ComplexToComplex
        | FftTransformKind::RealToComplex
        | FftTransformKind::ComplexToReal => Ok(0),
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct FftC2cPlan {
    n: usize,
    factors: [usize; 3],
    packing: usize,
    twiddle_real: DeviceMemory<f32>,
    twiddle_imag: DeviceMemory<f32>,
}

impl FftC2cPlan {
    pub fn create(
        context: &Arc<Context>,
        n: usize,
        factors: [usize; 3],
        packing: usize,
    ) -> Result<Self> {
        validate_plan_shape(n, factors, packing)?;
        let (twiddle_real, twiddle_imag) = create_twiddle_tables(context, n)?;
        Ok(Self {
            n,
            factors,
            packing,
            twiddle_real,
            twiddle_imag,
        })
    }

    pub fn context(&self) -> &Arc<Context> {
        self.twiddle_real.context()
    }

    fn validate_stream(&self, stream: &Stream) -> Result<()> {
        if self.context() != stream.context() {
            return Err(CudaError::StreamContextMismatch.into());
        }
        Ok(())
    }

    pub const fn n(&self) -> usize {
        self.n
    }

    pub const fn factors(&self) -> [usize; 3] {
        self.factors
    }

    pub const fn packing(&self) -> usize {
        self.packing
    }

    fn validate_shape(&self) -> Result<()> {
        validate_plan_shape(self.n, self.factors, self.packing)
    }

    fn validate_c2c_f32_interleaved(
        &self,
        batch: usize,
        out_len: usize,
        input_len: usize,
    ) -> Result<()> {
        self.validate_shape()?;
        if batch == 0 {
            return Err(Error::InvalidLength);
        }

        let complex_values = checked_element_count(batch, self.n)?;
        let lanes = checked_element_count(complex_values, 2)?;
        checked_i32_value(lanes)?;
        ensure_len(out_len, lanes)?;
        ensure_len(input_len, lanes)
    }
}

fn validate_fft_plan_config(config: FftPlanConfig) -> Result<()> {
    validate_plan_shape(config.n, config.factors, config.packing)?;
    if config.batch == 0
        || config.input_stride == 0
        || config.output_stride == 0
        || config.input_distance == 0
        || config.output_distance == 0
    {
        return Err(Error::InvalidLength);
    }

    let (input_layout, output_layout) = fft_plan_buffer_layouts(config)?;
    input_layout.validate_i32_reach()?;
    output_layout.validate_i32_reach()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FftBufferKind {
    Real,
    InterleavedComplex,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FftBufferLayout {
    kind: FftBufferKind,
    batch: usize,
    elements: usize,
    stride: usize,
    distance: usize,
}

impl FftBufferLayout {
    const fn real(batch: usize, elements: usize, stride: usize, distance: usize) -> Self {
        Self {
            kind: FftBufferKind::Real,
            batch,
            elements,
            stride,
            distance,
        }
    }

    const fn interleaved_complex(
        batch: usize,
        elements: usize,
        stride: usize,
        distance: usize,
    ) -> Self {
        Self {
            kind: FftBufferKind::InterleavedComplex,
            batch,
            elements,
            stride,
            distance,
        }
    }

    fn reach(self) -> Result<usize> {
        let batch_span = self
            .batch
            .checked_sub(1)
            .ok_or(Error::SizeOverflow)?
            .checked_mul(self.distance)
            .ok_or(Error::SizeOverflow)?;
        let element_span = self
            .elements
            .checked_sub(1)
            .ok_or(Error::SizeOverflow)?
            .checked_mul(self.stride)
            .ok_or(Error::SizeOverflow)?;
        let max_offset = batch_span
            .checked_add(element_span)
            .ok_or(Error::SizeOverflow)?;
        let max_lane = match self.kind {
            FftBufferKind::Real => max_offset,
            FftBufferKind::InterleavedComplex => {
                max_offset.checked_add(1).ok_or(Error::SizeOverflow)?
            }
        };
        max_lane.checked_add(1).ok_or(Error::SizeOverflow)
    }

    fn validate_i32_reach(self) -> Result<()> {
        let reach = self.reach()?;
        checked_i32_value(reach - 1)?;
        Ok(())
    }

    fn ensure_len_at_least(self, actual: usize) -> Result<()> {
        if actual >= self.reach()? {
            Ok(())
        } else {
            Err(Error::LengthMismatch)
        }
    }
}

fn fft_plan_buffer_layouts(config: FftPlanConfig) -> Result<(FftBufferLayout, FftBufferLayout)> {
    match config.transform {
        FftTransformKind::ComplexToComplex => Ok((
            FftBufferLayout::interleaved_complex(
                config.batch,
                config.n,
                config.input_stride,
                config.input_distance,
            ),
            FftBufferLayout::interleaved_complex(
                config.batch,
                config.n,
                config.output_stride,
                config.output_distance,
            ),
        )),
        FftTransformKind::RealToComplex => {
            if !config.n.is_multiple_of(2) || !matches!(config.direction, FftDirection::Forward) {
                return Err(Error::InvalidLength);
            }
            Ok((
                FftBufferLayout::real(
                    config.batch,
                    config.n,
                    config.input_stride,
                    config.input_distance,
                ),
                FftBufferLayout::interleaved_complex(
                    config.batch,
                    config.n / 2 + 1,
                    config.output_stride,
                    config.output_distance,
                ),
            ))
        }
        FftTransformKind::ComplexToReal => {
            if !config.n.is_multiple_of(2) || !matches!(config.direction, FftDirection::Inverse) {
                return Err(Error::InvalidLength);
            }
            Ok((
                FftBufferLayout::interleaved_complex(
                    config.batch,
                    config.n / 2 + 1,
                    config.input_stride,
                    config.input_distance,
                ),
                FftBufferLayout::real(
                    config.batch,
                    config.n,
                    config.output_stride,
                    config.output_distance,
                ),
            ))
        }
    }
}

fn validate_plan_shape(n: usize, factors: [usize; 3], packing: usize) -> Result<()> {
    if n == 0 || packing == 0 || factors.contains(&0) {
        return Err(Error::InvalidLength);
    }

    let radix = checked_element_count(factors[0], factors[1])?;
    let radix = checked_element_count(radix, factors[2])?;
    ensure_len(radix, n)?;

    let lanes = checked_element_count(n, 2)?;
    if !lanes.is_multiple_of(packing) {
        return Err(Error::InvalidLength);
    }

    Ok(())
}

fn supported_f32_plan_shape(n: usize) -> Option<([usize; 3], usize)> {
    Some(match n {
        8 => ([2, 2, 2], 4),
        12 => ([3, 4, 1], 8),
        16 => ([4, 4, 1], 8),
        20 => ([4, 5, 1], 8),
        24 => ([4, 6, 1], 8),
        32 => ([4, 4, 2], 8),
        40 => ([5, 8, 1], 8),
        48 => ([6, 8, 1], 8),
        60 => ([6, 10, 1], 8),
        80 => ([8, 10, 1], 8),
        96 => ([8, 12, 1], 8),
        120 => ([10, 12, 1], 8),
        144 => ([12, 12, 1], 8),
        _ => return None,
    })
}

fn create_twiddle_tables(
    context: &Arc<Context>,
    n: usize,
) -> Result<(DeviceMemory<f32>, DeviceMemory<f32>)> {
    let mut real = Vec::with_capacity(n);
    let mut imag = Vec::with_capacity(n);
    for phase in 0..n {
        let angle = -2.0 * PI * phase as f32 / n as f32;
        real.push(angle.cos());
        imag.push(angle.sin());
    }
    let real = DeviceMemory::from_slice(context, &real)?;
    let imag = DeviceMemory::from_slice(context, &imag)?;
    Ok((real, imag))
}

#[cfg(feature = "dtype_f32")]
/// Copies packed interleaved complex f32 values between device buffers.
///
/// # Safety
///
/// Inputs and read-write data must be valid when read in the stream's context.
/// Order preceding producers before use. Retain all buffers and plan tables,
/// and exclusive writable access,
/// until completion on every path and throughout captured graph replay.
pub unsafe fn copy_c2c_f32_interleaved(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    plan: &FftC2cPlan,
    batch: usize,
) -> Result<()> {
    plan.validate_stream(stream)?;
    #[cfg(feature = "dtype_f32")]
    plan.validate_c2c_f32_interleaved(batch, out.len(), input.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::fft::copy_c2c_f32_interleaved(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            plan.n,
            batch,
        )
    }
}

#[cfg(feature = "dtype_f32")]
/// Computes a batched complex f32 FFT in the requested direction.
///
/// # Safety
///
/// Inputs and read-write data must be valid when read in the stream's context.
/// Order preceding producers before use. Retain all buffers and plan tables,
/// and exclusive writable access,
/// until completion on every path and throughout captured graph replay.
/// Only logical output elements are initialized; padding and trailing capacity are excluded.
pub unsafe fn fft_c2c_f32_interleaved(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    plan: &FftC2cPlan,
    batch: usize,
    direction: FftDirection,
) -> Result<()> {
    plan.validate_stream(stream)?;
    #[cfg(feature = "dtype_f32")]
    plan.validate_c2c_f32_interleaved(batch, out.len(), input.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::fft::fft_c2c_f32_interleaved(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            input_pointer(&plan.twiddle_real),
            input_pointer(&plan.twiddle_imag),
            plan.n,
            batch,
            2,
            2,
            plan.n * 2,
            plan.n * 2,
            direction.twiddle_imag_scale(),
        )
    }
}

#[cfg(feature = "dtype_f16")]
/// Computes a batched complex FFT with interleaved f16 storage.
///
/// # Safety
///
/// Inputs and read-write data must be valid when read in the stream's context.
/// Order preceding producers before use. Retain all buffers and plan tables,
/// and exclusive writable access,
/// until completion on every path and throughout captured graph replay.
/// Only logical output elements are initialized; padding and trailing capacity are excluded.
pub unsafe fn fft_c2c_f16_interleaved(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    input: &impl DeviceInput<f16>,
    plan: &FftC2cPlan,
    batch: usize,
    direction: FftDirection,
) -> Result<()> {
    plan.validate_stream(stream)?;
    #[cfg(feature = "dtype_f32")]
    plan.validate_c2c_f32_interleaved(batch, out.len(), input.len())?;
    match plan.n {
        8 | 16 => {
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fft::fft_c2c_f16_interleaved(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    input_pointer(&plan.twiddle_real),
                    input_pointer(&plan.twiddle_imag),
                    plan.n,
                    batch,
                    direction.twiddle_imag_scale(),
                )
            }
        }
        _ => Err(Error::UnsupportedFftSize {
            transform: "c2c f16".into(),
            n: plan.n,
        }),
    }
}

#[cfg(feature = "dtype_bf16")]
/// Computes a batched complex FFT with interleaved bf16 storage.
///
/// # Safety
///
/// Inputs and read-write data must be valid when read in the stream's context.
/// Order preceding producers before use. Retain all buffers and plan tables,
/// and exclusive writable access,
/// until completion on every path and throughout captured graph replay.
/// Only logical output elements are initialized; padding and trailing capacity are excluded.
pub unsafe fn fft_c2c_bf16_interleaved(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    input: &impl DeviceInput<bf16>,
    plan: &FftC2cPlan,
    batch: usize,
    direction: FftDirection,
) -> Result<()> {
    plan.validate_stream(stream)?;
    #[cfg(feature = "dtype_f32")]
    plan.validate_c2c_f32_interleaved(batch, out.len(), input.len())?;
    match plan.n {
        8 | 16 => {
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fft::fft_c2c_bf16_interleaved(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    input_pointer(&plan.twiddle_real),
                    input_pointer(&plan.twiddle_imag),
                    plan.n,
                    batch,
                    direction.twiddle_imag_scale(),
                )
            }
        }
        _ => Err(Error::UnsupportedFftSize {
            transform: "c2c bf16".into(),
            n: plan.n,
        }),
    }
}

#[cfg(feature = "dtype_f32")]
/// Executes the complex f32 transform and layout described by the plan.
///
/// # Safety
///
/// Inputs and read-write data must be valid when read in the stream's context.
/// Order preceding producers before use. Retain all buffers and plan tables,
/// and exclusive writable access,
/// until completion on every path and throughout captured graph replay.
/// Only logical output elements are initialized; padding and trailing capacity are excluded.
pub unsafe fn fft_f32_interleaved(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    plan: &FftPlan,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { fft_f32_interleaved_inner(stream, out, input, plan) }
}

#[cfg(feature = "dtype_f32")]
/// Executes the planned complex f32 transform with caller-provided scratch storage.
///
/// # Safety
///
/// Inputs and read-write data must be valid when read in the stream's context.
/// Order preceding producers before use. Retain all buffers and plan tables,
/// and exclusive writable access,
/// until completion on every path and throughout captured graph replay.
/// Only logical output elements are initialized; padding and trailing capacity are excluded.
pub unsafe fn fft_f32_interleaved_with_workspace<W>(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    plan: &FftPlan,
    workspace: Option<&mut W>,
) -> Result<()>
where
    W: DeviceOutput<u8> + ?Sized,
{
    plan.validate_workspace(workspace.as_deref())?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { fft_f32_interleaved_inner(stream, out, input, plan) }
}

unsafe fn fft_f32_interleaved_inner(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    plan: &FftPlan,
) -> Result<()> {
    plan.validate_stream(stream)?;
    #[cfg(feature = "dtype_f32")]
    plan.validate_c2c_f32_interleaved(out.len(), input.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::fft::fft_c2c_f32_interleaved(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            input_pointer(&plan.twiddle_real),
            input_pointer(&plan.twiddle_imag),
            plan.config.n,
            plan.config.batch,
            plan.config.input_stride,
            plan.config.output_stride,
            plan.config.input_distance,
            plan.config.output_distance,
            plan.config.direction.twiddle_imag_scale(),
        )
    }
}

#[cfg(feature = "dtype_f32")]
/// Transforms real f32 input into an interleaved complex spectrum.
///
/// # Safety
///
/// Inputs and read-write data must be valid when read in the stream's context.
/// Order preceding producers before use. Retain all buffers and plan tables,
/// and exclusive writable access,
/// until completion on every path and throughout captured graph replay.
/// Only logical output elements are initialized; padding and trailing capacity are excluded.
pub unsafe fn fft_r2c_f32_interleaved(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    plan: &FftPlan,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { fft_r2c_f32_interleaved_inner(stream, out, input, plan) }
}

#[cfg(feature = "dtype_f32")]
/// Transforms real f32 input into a complex spectrum with caller-provided scratch storage.
///
/// # Safety
///
/// Inputs and read-write data must be valid when read in the stream's context.
/// Order preceding producers before use. Retain all buffers and plan tables,
/// and exclusive writable access,
/// until completion on every path and throughout captured graph replay.
/// Only logical output elements are initialized; padding and trailing capacity are excluded.
pub unsafe fn fft_r2c_f32_interleaved_with_workspace<W>(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    plan: &FftPlan,
    workspace: Option<&mut W>,
) -> Result<()>
where
    W: DeviceOutput<u8> + ?Sized,
{
    plan.validate_workspace(workspace.as_deref())?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { fft_r2c_f32_interleaved_inner(stream, out, input, plan) }
}

unsafe fn fft_r2c_f32_interleaved_inner(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    plan: &FftPlan,
) -> Result<()> {
    plan.validate_stream(stream)?;
    #[cfg(feature = "dtype_f32")]
    plan.validate_r2c_f32_interleaved(out.len(), input.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::fft::fft_r2c_f32_interleaved(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            input_pointer(&plan.twiddle_real),
            input_pointer(&plan.twiddle_imag),
            plan.config.n,
            plan.config.batch,
            plan.config.input_stride,
            plan.config.output_stride,
            plan.config.input_distance,
            plan.config.output_distance,
        )
    }
}

#[cfg(feature = "dtype_f32")]
/// Transforms a Hermitian complex spectrum into real f32 output.
///
/// # Safety
///
/// Inputs and read-write data must be valid when read in the stream's context.
/// Order preceding producers before use. Retain all buffers and plan tables,
/// and exclusive writable access,
/// until completion on every path and throughout captured graph replay.
/// Only logical output elements are initialized; padding and trailing capacity are excluded.
pub unsafe fn ifft_c2r_f32_interleaved(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    plan: &FftPlan,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { ifft_c2r_f32_interleaved_inner(stream, out, input, plan) }
}

#[cfg(feature = "dtype_f32")]
/// Computes a real inverse FFT with caller-provided scratch storage.
///
/// # Safety
///
/// Inputs and read-write data must be valid when read in the stream's context.
/// Order preceding producers before use. Retain all buffers and plan tables,
/// and exclusive writable access,
/// until completion on every path and throughout captured graph replay.
/// Only logical output elements are initialized; padding and trailing capacity are excluded.
pub unsafe fn ifft_c2r_f32_interleaved_with_workspace<W>(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    plan: &FftPlan,
    workspace: Option<&mut W>,
) -> Result<()>
where
    W: DeviceOutput<u8> + ?Sized,
{
    plan.validate_workspace(workspace.as_deref())?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { ifft_c2r_f32_interleaved_inner(stream, out, input, plan) }
}

unsafe fn ifft_c2r_f32_interleaved_inner(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    plan: &FftPlan,
) -> Result<()> {
    plan.validate_stream(stream)?;
    #[cfg(feature = "dtype_f32")]
    plan.validate_c2r_f32_interleaved(out.len(), input.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::fft::ifft_c2r_f32_interleaved(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            input_pointer(&plan.twiddle_real),
            input_pointer(&plan.twiddle_imag),
            plan.config.n,
            plan.config.batch,
            plan.config.input_stride,
            plan.config.output_stride,
            plan.config.input_distance,
            plan.config.output_distance,
        )
    }
}

#[cfg(feature = "dtype_f32")]
/// Scales real inverse-transform output under the requested normalization convention.
/// Only logical output elements are modified; stride gaps and trailing storage are preserved.
///
/// # Safety
///
/// Inputs and read-write data must be valid when read in the stream's context.
/// Order preceding producers before use. Retain all buffers and plan tables,
/// and exclusive writable access,
/// until completion on every path and throughout captured graph replay.
pub unsafe fn normalize_c2r_f32_interleaved(
    stream: &Stream,
    data: &mut impl DeviceSliceMut<f32>,
    plan: &FftPlan,
    normalization: FftNormalization,
) -> Result<()> {
    plan.validate_stream(stream)?;
    plan.validate_c2r_f32_output(data.len())?;
    let scale = normalization.scale(plan.config.n, FftDirection::Inverse)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::fft::scale_real_strided_f32(
            &stream,
            output_pointer(data),
            plan.config.n,
            plan.config.batch,
            plan.config.output_stride,
            plan.config.output_distance,
            scale,
        )
    }
}

#[cfg(feature = "dtype_f32")]
/// Multiplies every supplied real f32 element by the scale factor.
///
/// # Safety
///
/// Data must be initialized and belong to the stream's context. Retain the
/// buffer and exclusive access until completion on every path and throughout
/// captured graph replay.
pub unsafe fn scale_real_f32(
    stream: &Stream,
    data: &mut impl DeviceSliceMut<f32>,
    scale: f32,
) -> Result<()> {
    let len = data.len();
    checked_i32_value(len)?;
    let stream = stream.to_cuda_core()?;
    let data = output_pointer(data);
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { cutile::scalar::scale_f32(&stream, data, data, scale, len, None, Default::default()) }
}

#[cfg(feature = "dtype_f32")]
/// Multiplies both lanes of each supplied complex f32 value by the scale factor.
///
/// # Safety
///
/// Data must be initialized and belong to the stream's context. Retain the
/// buffer and exclusive access until completion on every path and throughout
/// captured graph replay.
pub unsafe fn scale_c2c_f32_interleaved(
    stream: &Stream,
    data: &mut impl DeviceSliceMut<f32>,
    scale: f32,
) -> Result<()> {
    let len = data.len();
    checked_i32_value(len)?;
    let stream = stream.to_cuda_core()?;
    let data = output_pointer(data);
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { cutile::scalar::scale_f32(&stream, data, data, scale, len, None, Default::default()) }
}
