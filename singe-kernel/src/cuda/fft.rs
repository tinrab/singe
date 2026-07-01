//! Small FFT helpers, interleaved complex transforms, and normalization.

#[cfg(feature = "dtype-bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    memory::DeviceMemory,
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
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
        Ok(Self::c2c_f32_interleaved(
            n, batch, factors, packing, direction,
        ))
    }

    pub fn c2r_f32_interleaved_supported(n: usize, batch: usize) -> Result<Self> {
        let (factors, packing) = supported_f32_plan_shape(n).ok_or(Error::UnsupportedFftSize {
            transform: "c2r".into(),
            n,
        })?;
        Ok(Self::c2r_f32_interleaved(n, batch, factors, packing))
    }

    pub fn r2c_f32_interleaved_supported(n: usize, batch: usize) -> Result<Self> {
        let (factors, packing) = supported_f32_plan_shape(n).ok_or(Error::UnsupportedFftSize {
            transform: "r2c".into(),
            n,
        })?;
        Ok(Self::r2c_f32_interleaved(n, batch, factors, packing))
    }

    pub const fn c2c_f32_interleaved(
        n: usize,
        batch: usize,
        factors: [usize; 3],
        packing: usize,
        direction: FftDirection,
    ) -> Self {
        Self {
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
            input_distance: n * 2,
            output_distance: n * 2,
        }
    }

    pub const fn c2r_f32_interleaved(
        n: usize,
        batch: usize,
        factors: [usize; 3],
        packing: usize,
    ) -> Self {
        Self {
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
            input_distance: (n / 2 + 1) * 2,
            output_distance: n,
        }
    }

    pub const fn r2c_f32_interleaved(
        n: usize,
        batch: usize,
        factors: [usize; 3],
        packing: usize,
    ) -> Self {
        Self {
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
            output_distance: (n / 2 + 1) * 2,
        }
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
    pub fn create(config: FftPlanConfig) -> Result<Self> {
        validate_fft_plan_config(config)?;
        let workspace_bytes = estimate_fft_workspace_bytes(config)?;
        match config.transform {
            FftTransformKind::ComplexToComplex
            | FftTransformKind::RealToComplex
            | FftTransformKind::ComplexToReal => {
                let (twiddle_real, twiddle_imag) = create_twiddle_tables(config.n)?;
                Ok(Self {
                    config,
                    workspace_bytes,
                    twiddle_real,
                    twiddle_imag,
                })
            }
        }
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

    pub fn create_workspace(&self) -> Result<Option<DeviceMemory<u8>>> {
        if self.workspace_bytes == 0 {
            Ok(None)
        } else {
            Ok(DeviceMemory::create(self.workspace_bytes).map(Some)?)
        }
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
        validate_fft_plan_config(self.config)?;
        if !self.config.is_c2r_f32_interleaved() {
            return Err(Error::UnsupportedConfiguration {
                op: "ifft_c2r_f32_interleaved".into(),
                reason: "requires a c2r f32 interleaved plan".into(),
            });
        }

        let input_lanes = checked_element_count(
            checked_element_count(self.config.batch, self.config.n / 2 + 1)?,
            2,
        )?;
        let output_values = checked_element_count(self.config.batch, self.config.n)?;
        checked_i32_value(input_lanes)?;
        checked_i32_value(output_values)?;
        let (input_layout, output_layout) = fft_plan_buffer_layouts(self.config)?;
        input_layout.ensure_len_at_least(input_len)?;
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
    pub fn create(n: usize, factors: [usize; 3], packing: usize) -> Result<Self> {
        validate_plan_shape(n, factors, packing)?;
        let (twiddle_real, twiddle_imag) = create_twiddle_tables(n)?;
        Ok(Self {
            n,
            factors,
            packing,
            twiddle_real,
            twiddle_imag,
        })
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

fn create_twiddle_tables(n: usize) -> Result<(DeviceMemory<f32>, DeviceMemory<f32>)> {
    let mut real = Vec::with_capacity(n);
    let mut imag = Vec::with_capacity(n);
    for phase in 0..n {
        let angle = -2.0 * std::f32::consts::PI * phase as f32 / n as f32;
        real.push(angle.cos());
        imag.push(angle.sin());
    }
    let real = DeviceMemory::from_slice(&real)?;
    let imag = DeviceMemory::from_slice(&imag)?;
    Ok((real, imag))
}

#[cfg(feature = "dtype-f32")]
pub fn copy_c2c_f32_interleaved(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &FftC2cPlan,
    batch: usize,
) -> Result<()> {
    #[cfg(feature = "dtype-f32")]
    plan.validate_c2c_f32_interleaved(batch, out.len(), input.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::fft::copy_c2c_f32_interleaved(
        &stream,
        output_pointer(out),
        input_pointer(input),
        plan.n,
        batch,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn fft_c2c_f32_interleaved(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &FftC2cPlan,
    batch: usize,
    direction: FftDirection,
) -> Result<()> {
    #[cfg(feature = "dtype-f32")]
    plan.validate_c2c_f32_interleaved(batch, out.len(), input.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::fft::fft_c2c_f32_interleaved(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f16")]
pub fn fft_c2c_f16_interleaved(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    input: &impl DeviceSlice<f16>,
    plan: &FftC2cPlan,
    batch: usize,
    direction: FftDirection,
) -> Result<()> {
    #[cfg(feature = "dtype-f32")]
    plan.validate_c2c_f32_interleaved(batch, out.len(), input.len())?;
    match plan.n {
        8 | 16 => {
            let stream = borrowed_stream(stream)?;
            cutile::fft::fft_c2c_f16_interleaved(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(&plan.twiddle_real),
                input_pointer(&plan.twiddle_imag),
                plan.n,
                batch,
                direction.twiddle_imag_scale(),
            )
        }
        _ => Err(Error::UnsupportedFftSize {
            transform: "c2c f16".into(),
            n: plan.n,
        }),
    }
}

#[cfg(feature = "dtype-bf16")]
pub fn fft_c2c_bf16_interleaved(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    input: &impl DeviceSlice<bf16>,
    plan: &FftC2cPlan,
    batch: usize,
    direction: FftDirection,
) -> Result<()> {
    #[cfg(feature = "dtype-f32")]
    plan.validate_c2c_f32_interleaved(batch, out.len(), input.len())?;
    match plan.n {
        8 | 16 => {
            let stream = borrowed_stream(stream)?;
            cutile::fft::fft_c2c_bf16_interleaved(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(&plan.twiddle_real),
                input_pointer(&plan.twiddle_imag),
                plan.n,
                batch,
                direction.twiddle_imag_scale(),
            )
        }
        _ => Err(Error::UnsupportedFftSize {
            transform: "c2c bf16".into(),
            n: plan.n,
        }),
    }
}

#[cfg(feature = "dtype-f32")]
pub fn fft_f32_interleaved(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &FftPlan,
) -> Result<()> {
    fft_f32_interleaved_inner(stream, out, input, plan)
}

#[cfg(feature = "dtype-f32")]
pub fn fft_f32_interleaved_with_workspace<W>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &FftPlan,
    workspace: Option<&mut W>,
) -> Result<()>
where
    W: DeviceSliceMut<u8> + ?Sized,
{
    plan.validate_workspace_len(workspace.as_ref().map(|workspace| workspace.len()))?;
    fft_f32_interleaved_inner(stream, out, input, plan)
}

fn fft_f32_interleaved_inner(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &FftPlan,
) -> Result<()> {
    #[cfg(feature = "dtype-f32")]
    plan.validate_c2c_f32_interleaved(out.len(), input.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::fft::fft_c2c_f32_interleaved(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn fft_r2c_f32_interleaved(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &FftPlan,
) -> Result<()> {
    fft_r2c_f32_interleaved_inner(stream, out, input, plan)
}

#[cfg(feature = "dtype-f32")]
pub fn fft_r2c_f32_interleaved_with_workspace<W>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &FftPlan,
    workspace: Option<&mut W>,
) -> Result<()>
where
    W: DeviceSliceMut<u8> + ?Sized,
{
    plan.validate_workspace_len(workspace.as_ref().map(|workspace| workspace.len()))?;
    fft_r2c_f32_interleaved_inner(stream, out, input, plan)
}

fn fft_r2c_f32_interleaved_inner(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &FftPlan,
) -> Result<()> {
    #[cfg(feature = "dtype-f32")]
    plan.validate_r2c_f32_interleaved(out.len(), input.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::fft::fft_r2c_f32_interleaved(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn ifft_c2r_f32_interleaved(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &FftPlan,
) -> Result<()> {
    ifft_c2r_f32_interleaved_inner(stream, out, input, plan)
}

#[cfg(feature = "dtype-f32")]
pub fn ifft_c2r_f32_interleaved_with_workspace<W>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &FftPlan,
    workspace: Option<&mut W>,
) -> Result<()>
where
    W: DeviceSliceMut<u8> + ?Sized,
{
    plan.validate_workspace_len(workspace.as_ref().map(|workspace| workspace.len()))?;
    ifft_c2r_f32_interleaved_inner(stream, out, input, plan)
}

fn ifft_c2r_f32_interleaved_inner(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    plan: &FftPlan,
) -> Result<()> {
    #[cfg(feature = "dtype-f32")]
    plan.validate_c2r_f32_interleaved(out.len(), input.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::fft::ifft_c2r_f32_interleaved(
        &stream,
        output_pointer(out),
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

#[cfg(feature = "dtype-f32")]
pub fn normalize_c2r_f32_interleaved(
    stream: &Stream,
    data: &mut impl DeviceSliceMut<f32>,
    plan: &FftPlan,
    normalization: FftNormalization,
) -> Result<()> {
    let packed_input_len = checked_element_count(
        checked_element_count(plan.config.batch, plan.config.n / 2 + 1)?,
        2,
    )?;
    #[cfg(feature = "dtype-f32")]
    plan.validate_c2r_f32_interleaved(data.len(), packed_input_len)?;
    let scale = normalization.scale(plan.config.n, FftDirection::Inverse)?;
    scale_real_f32(stream, data, scale)
}

#[cfg(feature = "dtype-f32")]
pub fn scale_real_f32(
    stream: &Stream,
    data: &mut impl DeviceSliceMut<f32>,
    scale: f32,
) -> Result<()> {
    let len = data.len();
    checked_i32_value(len)?;
    let stream = borrowed_stream(stream)?;
    let data = output_pointer(data);
    cutile::scalar::scale_f32(&stream, data, data, scale, len)
}

#[cfg(feature = "dtype-f32")]
pub fn scale_c2c_f32_interleaved(
    stream: &Stream,
    data: &mut impl DeviceSliceMut<f32>,
    scale: f32,
) -> Result<()> {
    let len = data.len();
    checked_i32_value(len)?;
    let stream = borrowed_stream(stream)?;
    let data = output_pointer(data);
    cutile::scalar::scale_f32(&stream, data, data, scale, len)
}
