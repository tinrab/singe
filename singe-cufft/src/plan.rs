use std::{mem::ManuallyDrop, ptr, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    memory::DeviceMemory,
    stream::Stream,
    types::{Complex32, Complex64},
};
use singe_cufft_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{Direction, PlanProperty, TransformType},
    utility::{
        optional_to_i32_vec, optional_to_i64_vec, to_i32, to_i32_vec, to_i64, to_i64_vec, to_usize,
    },
};

/// A stateful cuFFT plan handle.
///
/// Use one plan per host thread or concurrent task. The plan is movable
/// between threads, but it is intentionally not `Clone` or `Sync`.
/// Planning methods mutate the handle's transform configuration and return the
/// required workspace size in bytes. Execution methods optionally bind a CUDA
/// stream before launching work, and launches are ordered by that stream.
/// When automatic allocation is disabled, the caller-provided work area must
/// remain valid for every execution that uses the plan.
#[derive(Debug)]
pub struct Plan {
    handle: Handle,
}

#[derive(Debug)]
pub struct PlanBuilder<'a> {
    cuda_ctx: &'a Arc<CudaContext>,
    definition: Option<PlanDefinition>,
}

#[derive(Debug)]
enum PlanDefinition {
    OneDimensional {
        nx: usize,
        kind: TransformType,
        batch: usize,
    },
    TwoDimensional {
        nx: usize,
        ny: usize,
        kind: TransformType,
    },
    ThreeDimensional {
        nx: usize,
        ny: usize,
        nz: usize,
        kind: TransformType,
    },
    Many {
        n: Vec<usize>,
        inembed: Option<Vec<usize>>,
        istride: usize,
        idist: usize,
        onembed: Option<Vec<usize>>,
        ostride: usize,
        odist: usize,
        kind: TransformType,
        batch: usize,
    },
    Many64 {
        n: Vec<i64>,
        inembed: Option<Vec<i64>>,
        istride: i64,
        idist: i64,
        onembed: Option<Vec<i64>>,
        ostride: i64,
        odist: i64,
        kind: TransformType,
        batch: i64,
    },
}

#[derive(Debug)]
struct Handle {
    raw: sys::cufftHandle,
    cuda_ctx: Arc<CudaContext>,
}

// cuFFT plan handles are mutable planning/execution objects. They can be moved
// to another thread, but shared concurrent access would race plan state.
unsafe impl Send for Handle {}

impl<'a> PlanBuilder<'a> {
    pub fn new(cuda_ctx: &'a Arc<CudaContext>) -> Self {
        Self {
            cuda_ctx,
            definition: None,
        }
    }

    pub fn one_dimensional(mut self, nx: usize, kind: TransformType) -> Self {
        self.definition = Some(PlanDefinition::OneDimensional { nx, kind, batch: 1 });
        self
    }

    pub fn two_dimensional(mut self, nx: usize, ny: usize, kind: TransformType) -> Self {
        self.definition = Some(PlanDefinition::TwoDimensional { nx, ny, kind });
        self
    }

    pub fn three_dimensional(
        mut self,
        nx: usize,
        ny: usize,
        nz: usize,
        kind: TransformType,
    ) -> Self {
        self.definition = Some(PlanDefinition::ThreeDimensional { nx, ny, nz, kind });
        self
    }

    pub fn many(mut self, config: ManyPlanConfig<'_>) -> Self {
        self.definition = Some(match config {
            ManyPlanConfig::I32(config) => PlanDefinition::Many {
                n: config.n.to_vec(),
                inembed: config.inembed.map(<[_]>::to_vec),
                istride: config.istride,
                idist: config.idist,
                onembed: config.onembed.map(<[_]>::to_vec),
                ostride: config.ostride,
                odist: config.odist,
                kind: config.kind,
                batch: config.batch,
            },
            ManyPlanConfig::I64(config) => PlanDefinition::Many64 {
                n: config.n.to_vec(),
                inembed: config.inembed.map(<[_]>::to_vec),
                istride: config.istride,
                idist: config.idist,
                onembed: config.onembed.map(<[_]>::to_vec),
                ostride: config.ostride,
                odist: config.odist,
                kind: config.kind,
                batch: config.batch,
            },
        });
        self
    }

    pub fn batch(mut self, batch: usize) -> Self {
        if let Some(PlanDefinition::OneDimensional { batch: value, .. }) = &mut self.definition {
            *value = batch;
        }
        self
    }

    pub fn create(self) -> Result<Plan> {
        let plan = Plan::create(self.cuda_ctx)?;
        match self.definition.ok_or(Error::MissingPlanDefinition)? {
            PlanDefinition::OneDimensional { nx, kind, batch } => {
                plan.plan_1d(nx, kind, batch)?;
            }
            PlanDefinition::TwoDimensional { nx, ny, kind } => {
                plan.plan_2d(nx, ny, kind)?;
            }
            PlanDefinition::ThreeDimensional { nx, ny, nz, kind } => {
                plan.plan_3d(nx, ny, nz, kind)?;
            }
            PlanDefinition::Many {
                n,
                inembed,
                istride,
                idist,
                onembed,
                ostride,
                odist,
                kind,
                batch,
            } => {
                plan.plan_many(ManyPlanConfig::I32(ManyPlanConfigI32 {
                    n: &n,
                    inembed: inembed.as_deref(),
                    istride,
                    idist,
                    onembed: onembed.as_deref(),
                    ostride,
                    odist,
                    kind,
                    batch,
                }))?;
            }
            PlanDefinition::Many64 {
                n,
                inembed,
                istride,
                idist,
                onembed,
                ostride,
                odist,
                kind,
                batch,
            } => {
                plan.plan_many(ManyPlanConfig::I64(ManyPlanConfigI64 {
                    n: &n,
                    inembed: inembed.as_deref(),
                    istride,
                    idist,
                    onembed: onembed.as_deref(),
                    ostride,
                    odist,
                    kind,
                    batch,
                }))?;
            }
        }
        Ok(plan)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ManyPlanConfigI32<'a> {
    pub n: &'a [usize],
    pub inembed: Option<&'a [usize]>,
    pub istride: usize,
    pub idist: usize,
    pub onembed: Option<&'a [usize]>,
    pub ostride: usize,
    pub odist: usize,
    pub kind: TransformType,
    pub batch: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ManyPlanConfigI64<'a> {
    pub n: &'a [i64],
    pub inembed: Option<&'a [i64]>,
    pub istride: i64,
    pub idist: i64,
    pub onembed: Option<&'a [i64]>,
    pub ostride: i64,
    pub odist: i64,
    pub kind: TransformType,
    pub batch: i64,
}

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ManyPlanConfig<'a> {
    I32(ManyPlanConfigI32<'a>),
    I64(ManyPlanConfigI64<'a>),
}

impl Plan {
    /// Takes ownership of a raw cuFFT plan handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cufftHandle` associated with `cuda_ctx` and
    /// must not be owned by any other wrapper. The returned plan destroys it
    /// with `cufftDestroy`.
    pub unsafe fn from_raw(handle: sys::cufftHandle, cuda_ctx: Arc<CudaContext>) -> Result<Self> {
        if handle == sys::CUFFT_PLAN_NULL {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_ctx,
            },
        })
    }

    pub fn create(cuda_ctx: &Arc<CudaContext>) -> Result<Self> {
        cuda_ctx.bind()?;

        let mut handle = 0;
        unsafe {
            try_ffi!(sys::cufftCreate(&raw mut handle))?;
        }

        if handle == sys::CUFFT_PLAN_NULL {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_ctx: Arc::clone(cuda_ctx),
            },
        })
    }

    pub fn set_stream(&self, stream: Option<&Stream>) -> Result<()> {
        self.with_stream(stream)
    }

    pub fn set_auto_allocation(&self, auto_allocate: bool) -> Result<()> {
        self.bind()?;

        unsafe {
            try_ffi!(sys::cufftSetAutoAllocation(
                self.as_raw(),
                if auto_allocate { 1 } else { 0 },
            ))?;
        }

        Ok(())
    }

    pub fn set_work_area(&self, work_area: Option<&DeviceMemory<u8>>) -> Result<()> {
        self.bind()?;

        unsafe {
            try_ffi!(sys::cufftSetWorkArea(
                self.as_raw(),
                match work_area {
                    Some(memory) => memory.as_mut_ptr() as _,
                    None => ptr::null_mut(),
                },
            ))?;
        }

        Ok(())
    }

    pub fn set_plan_property(&self, property: PlanProperty, value: i64) -> Result<()> {
        self.bind()?;

        unsafe {
            try_ffi!(sys::cufftSetPlanPropertyInt64(
                self.as_raw(),
                property.into(),
                value,
            ))?;
        }

        Ok(())
    }

    pub fn plan_property(&self, property: PlanProperty) -> Result<i64> {
        self.bind()?;

        let mut value = 0i64;
        unsafe {
            try_ffi!(sys::cufftGetPlanPropertyInt64(
                self.as_raw(),
                property.into(),
                &raw mut value,
            ))?;
        }

        Ok(value)
    }

    pub fn reset_plan_property(&self, property: PlanProperty) -> Result<()> {
        self.bind()?;

        unsafe {
            try_ffi!(sys::cufftResetPlanProperty(self.as_raw(), property.into()))?;
        }

        Ok(())
    }

    pub fn plan_1d(&self, nx: usize, kind: TransformType, batch: usize) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftMakePlan1d(
                self.as_raw(),
                to_i32(nx, "nx")?,
                kind.into(),
                to_i32(batch, "batch")?,
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn plan_2d(&self, nx: usize, ny: usize, kind: TransformType) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftMakePlan2d(
                self.as_raw(),
                to_i32(nx, "nx")?,
                to_i32(ny, "ny")?,
                kind.into(),
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn plan_3d(&self, nx: usize, ny: usize, nz: usize, kind: TransformType) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftMakePlan3d(
                self.as_raw(),
                to_i32(nx, "nx")?,
                to_i32(ny, "ny")?,
                to_i32(nz, "nz")?,
                kind.into(),
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn plan_many(&self, config: ManyPlanConfig<'_>) -> Result<usize> {
        match config {
            ManyPlanConfig::I32(config) => self.plan_many_i32(config),
            ManyPlanConfig::I64(config) => self.plan_many_i64(config),
        }
    }

    fn plan_many_i32(&self, config: ManyPlanConfigI32<'_>) -> Result<usize> {
        self.bind()?;

        let rank = validate_rank(config.n)?;
        let mut n = to_i32_vec(config.n, "n")?;
        let mut inembed = optional_to_i32_vec(config.inembed, rank, "inembed")?;
        let mut onembed = optional_to_i32_vec(config.onembed, rank, "onembed")?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftMakePlanMany(
                self.as_raw(),
                to_i32(rank, "rank")?,
                n.as_mut_ptr(),
                if inembed.is_empty() {
                    ptr::null_mut()
                } else {
                    inembed.as_mut_ptr()
                },
                to_i32(config.istride, "istride")?,
                to_i32(config.idist, "idist")?,
                if onembed.is_empty() {
                    ptr::null_mut()
                } else {
                    onembed.as_mut_ptr()
                },
                to_i32(config.ostride, "ostride")?,
                to_i32(config.odist, "odist")?,
                config.kind.into(),
                to_i32(config.batch, "batch")?,
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    fn plan_many_i64(&self, config: ManyPlanConfigI64<'_>) -> Result<usize> {
        self.bind()?;

        let rank = validate_rank(config.n)?;
        let mut n = to_i64_vec(config.n, "n")?;
        let mut inembed = optional_to_i64_vec(config.inembed, rank, "inembed")?;
        let mut onembed = optional_to_i64_vec(config.onembed, rank, "onembed")?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftMakePlanMany64(
                self.as_raw(),
                to_i32(rank, "rank")?,
                n.as_mut_ptr(),
                if inembed.is_empty() {
                    ptr::null_mut()
                } else {
                    inembed.as_mut_ptr()
                },
                to_i64(config.istride, "istride")?,
                to_i64(config.idist, "idist")?,
                if onembed.is_empty() {
                    ptr::null_mut()
                } else {
                    onembed.as_mut_ptr()
                },
                to_i64(config.ostride, "ostride")?,
                to_i64(config.odist, "odist")?,
                config.kind.into(),
                to_i64(config.batch, "batch")?,
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn workspace_size_1d(&self, nx: usize, kind: TransformType, batch: usize) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftGetSize1d(
                self.as_raw(),
                to_i32(nx, "nx")?,
                kind.into(),
                to_i32(batch, "batch")?,
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn workspace_size_2d(&self, nx: usize, ny: usize, kind: TransformType) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftGetSize2d(
                self.as_raw(),
                to_i32(nx, "nx")?,
                to_i32(ny, "ny")?,
                kind.into(),
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn workspace_size_3d(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        kind: TransformType,
    ) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftGetSize3d(
                self.as_raw(),
                to_i32(nx, "nx")?,
                to_i32(ny, "ny")?,
                to_i32(nz, "nz")?,
                kind.into(),
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn workspace_size_many(&self, config: ManyPlanConfig<'_>) -> Result<usize> {
        match config {
            ManyPlanConfig::I32(config) => self.workspace_size_many_i32(config),
            ManyPlanConfig::I64(config) => self.workspace_size_many_i64(config),
        }
    }

    fn workspace_size_many_i32(&self, config: ManyPlanConfigI32<'_>) -> Result<usize> {
        self.bind()?;

        let rank = validate_rank(config.n)?;
        let mut n = to_i32_vec(config.n, "n")?;
        let mut inembed = optional_to_i32_vec(config.inembed, rank, "inembed")?;
        let mut onembed = optional_to_i32_vec(config.onembed, rank, "onembed")?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftGetSizeMany(
                self.as_raw(),
                to_i32(rank, "rank")?,
                n.as_mut_ptr(),
                if inembed.is_empty() {
                    ptr::null_mut()
                } else {
                    inembed.as_mut_ptr()
                },
                to_i32(config.istride, "istride")?,
                to_i32(config.idist, "idist")?,
                if onembed.is_empty() {
                    ptr::null_mut()
                } else {
                    onembed.as_mut_ptr()
                },
                to_i32(config.ostride, "ostride")?,
                to_i32(config.odist, "odist")?,
                config.kind.into(),
                to_i32(config.batch, "batch")?,
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    fn workspace_size_many_i64(&self, config: ManyPlanConfigI64<'_>) -> Result<usize> {
        self.bind()?;

        let rank = validate_rank(config.n)?;
        let mut n = to_i64_vec(config.n, "n")?;
        let mut inembed = optional_to_i64_vec(config.inembed, rank, "inembed")?;
        let mut onembed = optional_to_i64_vec(config.onembed, rank, "onembed")?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftGetSizeMany64(
                self.as_raw(),
                to_i32(rank, "rank")?,
                n.as_mut_ptr(),
                if inembed.is_empty() {
                    ptr::null_mut()
                } else {
                    inembed.as_mut_ptr()
                },
                to_i64(config.istride, "istride")?,
                to_i64(config.idist, "idist")?,
                if onembed.is_empty() {
                    ptr::null_mut()
                } else {
                    onembed.as_mut_ptr()
                },
                to_i64(config.ostride, "ostride")?,
                to_i64(config.odist, "odist")?,
                config.kind.into(),
                to_i64(config.batch, "batch")?,
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn workspace_size(&self) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftGetSize(self.as_raw(), &raw mut work_size))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn estimate_1d(nx: usize, kind: TransformType, batch: usize) -> Result<usize> {
        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftEstimate1d(
                to_i32(nx, "nx")?,
                kind.into(),
                to_i32(batch, "batch")?,
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn estimate_2d(nx: usize, ny: usize, kind: TransformType) -> Result<usize> {
        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftEstimate2d(
                to_i32(nx, "nx")?,
                to_i32(ny, "ny")?,
                kind.into(),
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn estimate_3d(nx: usize, ny: usize, nz: usize, kind: TransformType) -> Result<usize> {
        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftEstimate3d(
                to_i32(nx, "nx")?,
                to_i32(ny, "ny")?,
                to_i32(nz, "nz")?,
                kind.into(),
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn estimate_many(config: ManyPlanConfig<'_>) -> Result<usize> {
        match config {
            ManyPlanConfig::I32(config) => Self::estimate_many_i32(config),
            ManyPlanConfig::I64(config) => Self::estimate_many_i64(config),
        }
    }

    fn estimate_many_i32(config: ManyPlanConfigI32<'_>) -> Result<usize> {
        let rank = validate_rank(config.n)?;
        let mut n = to_i32_vec(config.n, "n")?;
        let mut inembed = optional_to_i32_vec(config.inembed, rank, "inembed")?;
        let mut onembed = optional_to_i32_vec(config.onembed, rank, "onembed")?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftEstimateMany(
                to_i32(rank, "rank")?,
                n.as_mut_ptr(),
                if inembed.is_empty() {
                    ptr::null_mut()
                } else {
                    inembed.as_mut_ptr()
                },
                to_i32(config.istride, "istride")?,
                to_i32(config.idist, "idist")?,
                if onembed.is_empty() {
                    ptr::null_mut()
                } else {
                    onembed.as_mut_ptr()
                },
                to_i32(config.ostride, "ostride")?,
                to_i32(config.odist, "odist")?,
                config.kind.into(),
                to_i32(config.batch, "batch")?,
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    fn estimate_many_i64(config: ManyPlanConfigI64<'_>) -> Result<usize> {
        let rank = validate_rank(config.n)?;
        let mut n = to_i32_vec(config.n, "n")?;
        let mut inembed = optional_to_i32_vec(config.inembed, rank, "inembed")?;
        let mut onembed = optional_to_i32_vec(config.onembed, rank, "onembed")?;

        let mut work_size: sys::size_t = 0;
        unsafe {
            try_ffi!(sys::cufftEstimateMany(
                to_i32(rank, "rank")?,
                n.as_mut_ptr(),
                if inembed.is_empty() {
                    ptr::null_mut()
                } else {
                    inembed.as_mut_ptr()
                },
                to_i32(config.istride, "istride")?,
                to_i32(config.idist, "idist")?,
                if onembed.is_empty() {
                    ptr::null_mut()
                } else {
                    onembed.as_mut_ptr()
                },
                to_i32(config.ostride, "ostride")?,
                to_i32(config.odist, "odist")?,
                config.kind.into(),
                to_i32(config.batch, "batch")?,
                &raw mut work_size,
            ))?;
        }

        to_usize(work_size, "work_size")
    }

    pub fn execute_c2c_f32(
        &self,
        input: &DeviceMemory<Complex32>,
        output: &mut DeviceMemory<Complex32>,
        direction: Direction,
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.with_stream(stream)?;

        unsafe {
            try_ffi!(sys::cufftExecC2C(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftComplex>().cast_mut(),
                output.as_mut_ptr().cast::<sys::cufftComplex>(),
                direction as _,
            ))?;
        }

        Ok(())
    }

    pub fn execute_c2c_f32_in_place(
        &self,
        data: &mut DeviceMemory<Complex32>,
        direction: Direction,
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.with_stream(stream)?;

        unsafe {
            try_ffi!(sys::cufftExecC2C(
                self.as_raw(),
                data.as_mut_ptr().cast::<sys::cufftComplex>(),
                data.as_mut_ptr().cast::<sys::cufftComplex>(),
                direction as _,
            ))?;
        }

        Ok(())
    }

    pub fn execute_r2c_f32(
        &self,
        input: &DeviceMemory<f32>,
        output: &mut DeviceMemory<Complex32>,
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.with_stream(stream)?;

        unsafe {
            try_ffi!(sys::cufftExecR2C(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftReal>().cast_mut(),
                output.as_mut_ptr().cast::<sys::cufftComplex>(),
            ))?;
        }

        Ok(())
    }

    pub fn execute_c2r_f32(
        &self,
        input: &DeviceMemory<Complex32>,
        output: &mut DeviceMemory<f32>,
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.with_stream(stream)?;

        unsafe {
            try_ffi!(sys::cufftExecC2R(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftComplex>().cast_mut(),
                output.as_mut_ptr().cast::<sys::cufftReal>(),
            ))?;
        }

        Ok(())
    }

    pub fn execute_z2z_f64(
        &self,
        input: &DeviceMemory<Complex64>,
        output: &mut DeviceMemory<Complex64>,
        direction: Direction,
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.with_stream(stream)?;

        unsafe {
            try_ffi!(sys::cufftExecZ2Z(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftDoubleComplex>().cast_mut(),
                output.as_mut_ptr().cast::<sys::cufftDoubleComplex>(),
                direction as _,
            ))?;
        }

        Ok(())
    }

    pub fn execute_z2z_f64_in_place(
        &self,
        data: &mut DeviceMemory<Complex64>,
        direction: Direction,
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.with_stream(stream)?;

        unsafe {
            try_ffi!(sys::cufftExecZ2Z(
                self.as_raw(),
                data.as_mut_ptr().cast::<sys::cufftDoubleComplex>(),
                data.as_mut_ptr().cast::<sys::cufftDoubleComplex>(),
                direction as _,
            ))?;
        }

        Ok(())
    }

    pub fn execute_d2z_f64(
        &self,
        input: &DeviceMemory<f64>,
        output: &mut DeviceMemory<Complex64>,
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.with_stream(stream)?;

        unsafe {
            try_ffi!(sys::cufftExecD2Z(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftDoubleReal>().cast_mut(),
                output.as_mut_ptr().cast::<sys::cufftDoubleComplex>(),
            ))?;
        }

        Ok(())
    }

    pub fn execute_z2d_f64(
        &self,
        input: &DeviceMemory<Complex64>,
        output: &mut DeviceMemory<f64>,
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.with_stream(stream)?;

        unsafe {
            try_ffi!(sys::cufftExecZ2D(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftDoubleComplex>().cast_mut(),
                output.as_mut_ptr().cast::<sys::cufftDoubleReal>(),
            ))?;
        }

        Ok(())
    }

    fn with_stream(&self, stream: Option<&Stream>) -> Result<()> {
        match stream {
            Some(stream) => {
                if self.cuda_context().as_ref() != stream.context() {
                    return Err(Error::StreamContextMismatch);
                }

                self.bind()?;
                unsafe {
                    try_ffi!(sys::cufftSetStream(self.as_raw(), stream.as_raw()))?;
                }
                Ok(())
            }
            None => {
                self.bind()?;
                unsafe {
                    try_ffi!(sys::cufftSetStream(self.as_raw(), ptr::null_mut()))?;
                }
                Ok(())
            }
        }
    }

    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_ctx
    }

    pub fn as_raw(&self) -> sys::cufftHandle {
        self.handle.raw
    }

    /// Consumes the plan and returns the raw cuFFT plan handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with cuFFT.
    pub fn into_raw(self) -> sys::cufftHandle {
        let plan = ManuallyDrop::new(self);
        plan.handle.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Err(err) = self.cuda_ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cufft handle: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cufftDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cufft handle: {err}");
            }
        }
    }
}

fn validate_rank<T>(values: &[T]) -> Result<usize> {
    if values.is_empty() {
        return Err(Error::OutOfRange {
            name: "rank".into(),
        });
    }

    Ok(values.len())
}
