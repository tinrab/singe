use std::{cell::RefCell, mem, ptr, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    stream::Stream,
    types::{Complex32, Complex64},
    view::{DeviceInput, DeviceOutputView, DeviceRegion},
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

/// Collects a transform definition before creating and configuring a cuFFT plan.
///
/// Shape setters replace the previous definition. Dimensions are checked when
/// [`create`](Self::create) builds the native plan.
#[derive(Debug)]
pub struct PlanBuilder<'a> {
    cuda_context: &'a Arc<CudaContext>,
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
    cuda_context: Arc<CudaContext>,
    streams: RefCell<Vec<Stream>>,
}

// SAFETY: cuFFT plan handles are mutable planning/execution objects. They can be moved
// to another thread, but shared concurrent access would race plan state.
unsafe impl Send for Handle {}

impl<'a> PlanBuilder<'a> {
    /// Starts a builder in `cuda_context` with no transform definition.
    #[must_use]
    pub fn new(cuda_context: &'a Arc<CudaContext>) -> Self {
        Self {
            cuda_context,
            definition: None,
        }
    }

    /// Selects a length-`nx` transform with one batch.
    ///
    /// Call [`batch`](Self::batch) afterward to select multiple transforms.
    #[must_use]
    pub fn one_dimensional(mut self, nx: usize, kind: TransformType) -> Self {
        self.definition = Some(PlanDefinition::OneDimensional { nx, kind, batch: 1 });
        self
    }

    /// Selects one row-major transform of shape `nx` by `ny`.
    ///
    /// `ny` is the innermost dimension.
    #[must_use]
    pub fn two_dimensional(mut self, nx: usize, ny: usize, kind: TransformType) -> Self {
        self.definition = Some(PlanDefinition::TwoDimensional { nx, ny, kind });
        self
    }

    /// Selects one row-major transform of shape `nx` by `ny` by `nz`.
    ///
    /// `nz` is the innermost dimension.
    #[must_use]
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

    /// Copies a batched transform definition and its optional embedding arrays.
    #[must_use]
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

    /// Sets the batch count of the current one-dimensional definition.
    ///
    /// Has no effect before selecting a shape or on other definition kinds.
    #[must_use]
    pub fn batch(mut self, batch: usize) -> Self {
        if let Some(PlanDefinition::OneDimensional { batch: value, .. }) = &mut self.definition {
            *value = batch;
        }
        self
    }

    /// Creates a native plan using the selected definition.
    ///
    /// # Errors
    ///
    /// Returns an error if no definition was selected, a dimension or stride cannot
    /// fit the native integer interface, an embedding length differs from the rank,
    /// the CUDA context cannot be bound, or cuFFT cannot create or configure the plan.
    pub fn create(self) -> Result<Plan> {
        let plan = Plan::create(self.cuda_context)?;
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

/// Batched transform dimensions and element strides for cuFFT's 32-bit planning interface.
///
/// Dimensions are ordered from outermost to innermost. An absent embedding uses
/// the native contiguous layout; cuFFT ignores the associated advanced strides
/// and distances. Values are validated when passed to a plan or size query.
#[derive(Debug, Clone, Copy)]
pub struct ManyPlanConfigI32<'a> {
    /// Logical transform extents, ordered from outermost to innermost; the rank must be nonzero.
    pub n: &'a [usize],
    /// Physical input extents for advanced layout, with one entry per dimension, or `None` for contiguous input.
    pub inembed: Option<&'a [usize]>,
    /// Distance in input elements between consecutive samples of the innermost dimension.
    pub istride: usize,
    /// Distance in input elements between the starts of successive batches.
    pub idist: usize,
    /// Physical output extents for advanced layout, with one entry per dimension, or `None` for contiguous output.
    pub onembed: Option<&'a [usize]>,
    /// Distance in output elements between consecutive samples of the innermost dimension.
    pub ostride: usize,
    /// Distance in output elements between the starts of successive batches.
    pub odist: usize,
    /// Transform direction of data conversion and scalar precision.
    pub kind: TransformType,
    /// Number of transforms sharing this shape and layout.
    pub batch: usize,
}

/// Batched transform dimensions and element strides for cuFFT's 64-bit planning interface.
///
/// Dimensions are ordered from outermost to innermost. An absent embedding uses
/// the native contiguous layout; cuFFT ignores the associated advanced strides
/// and distances. Values are validated when passed to a plan or size query.
#[derive(Debug, Clone, Copy)]
pub struct ManyPlanConfigI64<'a> {
    /// Logical transform extents, ordered from outermost to innermost; the rank must be nonzero.
    pub n: &'a [i64],
    /// Physical input extents for advanced layout, with one entry per dimension, or `None` for contiguous input.
    pub inembed: Option<&'a [i64]>,
    /// Distance in input elements between consecutive samples of the innermost dimension.
    pub istride: i64,
    /// Distance in input elements between the starts of successive batches.
    pub idist: i64,
    /// Physical output extents for advanced layout, with one entry per dimension, or `None` for contiguous output.
    pub onembed: Option<&'a [i64]>,
    /// Distance in output elements between consecutive samples of the innermost dimension.
    pub ostride: i64,
    /// Distance in output elements between the starts of successive batches.
    pub odist: i64,
    /// Transform direction of data conversion and scalar precision.
    pub kind: TransformType,
    /// Number of transforms sharing this shape and layout.
    pub batch: i64,
}

/// Selects the native integer width for a batched FFT definition.
///
/// Estimation uses the 32-bit native interface for both variants; values in an
/// `I64` definition must therefore also fit `i32` when estimating workspace.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ManyPlanConfig<'a> {
    /// Uses checked conversion from `usize` values to the 32-bit cuFFT interface.
    I32(ManyPlanConfigI32<'a>),
    /// Uses signed 64-bit dimensions, embedding extents, strides, and batch distances.
    I64(ManyPlanConfigI64<'a>),
}

/// A transferred native plan with its actual CUDA context and retained streams.
///
/// Destroy `handle` before releasing these owners, or reclaim it with `Plan::from_raw`.
/// External workspaces, callbacks, queued buffers and graph replays remain caller obligations.
#[derive(Debug)]
pub struct RawPlan {
    /// Owned cuFFT plan handle; destroy it with `cufftDestroy` or return it to `Plan`.
    pub handle: sys::cufftHandle,
    /// Actual CUDA context in which the handle was created.
    pub cuda_context: Arc<CudaContext>,
    /// Retained owners for streams still referenced by native plan state or queued transforms.
    pub streams: Vec<Stream>,
}

// SAFETY: transfer carries exclusive native ownership and the resources needed for cross-thread cleanup.
unsafe impl Send for RawPlan {}

impl Plan {
    /// Takes ownership of a raw cuFFT plan handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cufftHandle` associated with `cuda_context` and
    /// must not be owned by any other wrapper. The returned plan destroys it
    /// with `cufftDestroy`.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is `CUFFT_PLAN_NULL` or any retained stream
    /// belongs to a different CUDA context. No native destruction is performed on
    /// the rejected handle.
    pub unsafe fn from_raw_parts(
        handle: sys::cufftHandle,
        cuda_context: Arc<CudaContext>,
        streams: Vec<Stream>,
    ) -> Result<Self> {
        if streams
            .iter()
            .any(|stream| stream.context() != &cuda_context)
        {
            return Err(Error::StreamContextMismatch);
        }
        if handle == sys::CUFFT_PLAN_NULL {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_context,
                streams: RefCell::new(streams),
            },
        })
    }

    /// Creates an unconfigured cuFFT plan in `cuda_context`.
    ///
    /// Configure its transform before execution. Automatic workspace allocation is
    /// enabled by cuFFT unless changed with [`set_auto_allocation`](Self::set_auto_allocation).
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, cuFFT cannot create
    /// the plan, or the returned handle is null.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        cuda_context.bind()?;

        let mut handle = 0;
        // SAFETY: `cufftCreate` receives valid output locations.
        unsafe {
            try_ffi!(sys::cufftCreate(&raw mut handle))?;
        }

        if handle == sys::CUFFT_PLAN_NULL {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_context: Arc::clone(cuda_context),
                streams: RefCell::new(Vec::new()),
            },
        })
    }

    /// Selects and retains the execution stream; `None` selects the default stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream belongs to another CUDA context, context
    /// binding fails, or cuFFT rejects the stream association.
    pub fn set_stream(&self, stream: Option<&Stream>) -> Result<()> {
        self.with_stream(stream)
    }

    /// Enables or disables cuFFT allocation of workspace during planning.
    ///
    /// When disabled, supply a sufficient external work area before execution.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails or cuFFT rejects the allocation setting.
    pub fn set_auto_allocation(&self, auto_allocate: bool) -> Result<()> {
        self.bind()?;

        // SAFETY: `cufftSetAutoAllocation` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cufftSetAutoAllocation(
                self.as_raw(),
                i32::from(auto_allocate),
            ))?;
        }

        Ok(())
    }

    /// Installs externally owned workspace storage without claiming its bytes are initialized.
    ///
    /// # Safety
    ///
    /// The workspace must have the capacity and alignment required by every configured transform.
    /// Retain it exclusively through all queued uses and native plan destruction, or reset it after completion.
    /// Captured graph replays keep the same workspace lifetime and access obligations.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails, the supplied region belongs to
    /// another CUDA context, or cuFFT rejects the workspace pointer.
    pub unsafe fn set_work_area(&self, work_area: Option<DeviceOutputView<'_, u8>>) -> Result<()> {
        self.bind()?;
        let pointer = if let Some(work_area) = work_area {
            let region = work_area.into_region();
            if region.context() != self.cuda_context() {
                return Err(Error::StreamContextMismatch);
            }
            region.as_ptr().cast()
        } else {
            ptr::null_mut()
        };
        // SAFETY: `cufftSetWorkArea` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cufftSetWorkArea(self.as_raw(), pointer))?;
        }
        Ok(())
    }

    /// Sets an integer plan property before planning.
    ///
    /// Property-specific restrictions and supported values are defined by cuFFT.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails or cuFFT rejects the property,
    /// value, or current planning state.
    pub fn set_plan_property(&self, property: PlanProperty, value: i64) -> Result<()> {
        self.bind()?;

        // SAFETY: `cufftSetPlanPropertyInt64` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cufftSetPlanPropertyInt64(
                self.as_raw(),
                property.into(),
                value,
            ))?;
        }

        Ok(())
    }

    /// Reads the current integer value of a plan property.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails or cuFFT cannot query the property.
    pub fn plan_property(&self, property: PlanProperty) -> Result<i64> {
        self.bind()?;

        let mut value = 0i64;
        // SAFETY: `cufftGetPlanPropertyInt64` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftGetPlanPropertyInt64(
                self.as_raw(),
                property.into(),
                &raw mut value,
            ))?;
        }

        Ok(value)
    }

    /// Restores a plan property to its native default.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails or cuFFT rejects the property
    /// or the current planning state.
    pub fn reset_plan_property(&self, property: PlanProperty) -> Result<()> {
        self.bind()?;

        // SAFETY: `cufftResetPlanProperty` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cufftResetPlanProperty(self.as_raw(), property.into()))?;
        }

        Ok(())
    }

    /// Configures `batch` transforms of length `nx` and returns required workspace bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails, an argument does not fit `i32`,
    /// cuFFT cannot plan the requested transform, or the workspace size does not fit `usize`.
    pub fn plan_1d(&self, nx: usize, kind: TransformType, batch: usize) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        // SAFETY: `cufftMakePlan1d` receives live handles owned by the wrappers and valid output locations.
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

    /// Configures one row-major transform of shape `nx` by `ny` and returns required workspace bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails, an argument does not fit `i32`,
    /// cuFFT cannot plan the requested transform, or the workspace size does not fit `usize`.
    pub fn plan_2d(&self, nx: usize, ny: usize, kind: TransformType) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        // SAFETY: `cufftMakePlan2d` receives live handles owned by the wrappers and valid output locations.
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

    /// Configures one row-major transform of shape `nx` by `ny` by `nz` and returns required workspace bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails, an argument does not fit `i32`,
    /// cuFFT cannot plan the requested transform, or the workspace size does not fit `usize`.
    pub fn plan_3d(&self, nx: usize, ny: usize, nz: usize, kind: TransformType) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        // SAFETY: `cufftMakePlan3d` receives live handles owned by the wrappers and valid output locations.
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

    /// Configures a batched transform and returns required workspace bytes.
    ///
    /// The configuration selects the 32-bit or 64-bit native planning interface.
    ///
    /// # Errors
    ///
    /// Returns an error if the rank is zero or does not fit `i32`, an embedding
    /// length differs from the rank, a value exceeds the selected native integer
    /// width, context binding fails, cuFFT cannot plan the transform, or the
    /// workspace size does not fit `usize`.
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
        // SAFETY: `cufftMakePlanMany` receives live handles owned by the wrappers and valid output locations.
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
        // SAFETY: `cufftMakePlanMany64` receives live handles owned by the wrappers and valid output locations.
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

    /// Queries workspace bytes for `batch` transforms of length `nx` using this plan's settings.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails, an argument does not fit `i32`,
    /// cuFFT cannot compute the workspace size, or the result does not fit `usize`.
    pub fn workspace_size_1d(&self, nx: usize, kind: TransformType, batch: usize) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        // SAFETY: `cufftGetSize1d` receives live handles owned by the wrappers and valid output locations.
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

    /// Queries workspace bytes for one row-major transform of shape `nx` by `ny` using this plan's settings.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails, an argument does not fit `i32`,
    /// cuFFT cannot compute the workspace size, or the result does not fit `usize`.
    pub fn workspace_size_2d(&self, nx: usize, ny: usize, kind: TransformType) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        // SAFETY: `cufftGetSize2d` receives live handles owned by the wrappers and valid output locations.
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

    /// Queries workspace bytes for one row-major transform of shape `nx` by `ny` by `nz` using this plan's settings.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails, an argument does not fit `i32`,
    /// cuFFT cannot compute the workspace size, or the result does not fit `usize`.
    pub fn workspace_size_3d(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        kind: TransformType,
    ) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        // SAFETY: `cufftGetSize3d` receives live handles owned by the wrappers and valid output locations.
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

    /// Queries workspace bytes for a batched transform with this plan's settings.
    ///
    /// # Errors
    ///
    /// Returns an error if the rank is zero or does not fit `i32`, an embedding
    /// length differs from the rank, a value exceeds the selected native integer
    /// width, context binding fails, cuFFT cannot compute the workspace size, or
    /// the result does not fit `usize`.
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
        // SAFETY: `cufftGetSizeMany` receives live handles owned by the wrappers and valid output locations.
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
        // SAFETY: `cufftGetSizeMany64` receives live handles owned by the wrappers and valid output locations.
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

    /// Returns the workspace size in bytes for this plan's configured transform.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails, cuFFT cannot query the configured
    /// plan, or the workspace size does not fit `usize`.
    pub fn workspace_size(&self) -> Result<usize> {
        self.bind()?;

        let mut work_size: sys::size_t = 0;
        // SAFETY: `cufftGetSize` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftGetSize(self.as_raw(), &raw mut work_size))?;
        }

        to_usize(work_size, "work_size")
    }

    /// Estimates workspace bytes for `batch` transforms of length `nx` without creating a plan.
    ///
    /// The estimate may differ from the size returned by actual planning.
    ///
    /// # Errors
    ///
    /// Returns an error if an argument does not fit `i32`, cuFFT cannot estimate
    /// the workspace size, or the result does not fit `usize`.
    pub fn estimate_1d(nx: usize, kind: TransformType, batch: usize) -> Result<usize> {
        let mut work_size: sys::size_t = 0;
        // SAFETY: `cufftEstimate1d` receives valid output locations.
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

    /// Estimates workspace bytes for one row-major transform of shape `nx` by `ny` without creating a plan.
    ///
    /// The estimate may differ from the size returned by actual planning.
    ///
    /// # Errors
    ///
    /// Returns an error if an argument does not fit `i32`, cuFFT cannot estimate
    /// the workspace size, or the result does not fit `usize`.
    pub fn estimate_2d(nx: usize, ny: usize, kind: TransformType) -> Result<usize> {
        let mut work_size: sys::size_t = 0;
        // SAFETY: `cufftEstimate2d` receives valid output locations.
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

    /// Estimates workspace bytes for one row-major transform of shape `nx` by `ny` by `nz` without creating a plan.
    ///
    /// The estimate may differ from the size returned by actual planning.
    ///
    /// # Errors
    ///
    /// Returns an error if an argument does not fit `i32`, cuFFT cannot estimate
    /// the workspace size, or the result does not fit `usize`.
    pub fn estimate_3d(nx: usize, ny: usize, nz: usize, kind: TransformType) -> Result<usize> {
        let mut work_size: sys::size_t = 0;
        // SAFETY: `cufftEstimate3d` receives valid output locations.
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

    /// Estimates batched-transform workspace bytes through the 32-bit cuFFT interface.
    ///
    /// Both configuration variants must fit `i32`. The estimate may differ from
    /// the size returned by actual planning.
    ///
    /// # Errors
    ///
    /// Returns an error if the rank is zero, an embedding length differs from the
    /// rank, a configuration value does not fit `i32`, cuFFT cannot estimate the
    /// workspace size, or the result does not fit `usize`.
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
        // SAFETY: `cufftEstimateMany` receives valid output locations.
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
        // SAFETY: `cufftEstimateMany` receives valid output locations.
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

    /// Queues a single-precision complex transform into a separate destination.
    /// Pending destinations are accepted without asserting their samples or padding are initialized.
    ///
    /// # Safety
    ///
    /// The configured transform, strides, batch count, capacities and native alignment must match these regions.
    /// All samples read by the transform must be initialized when execution reaches them.
    /// Writable samples must remain exclusive until completion, including submission failures and graph replays.
    /// Retain the plan, buffers, workspaces and callback resources through every queued use; do not reconfigure them meanwhile.
    ///
    /// # Errors
    ///
    /// Returns an error if a memory region or stream belongs to a different CUDA
    /// context, context binding or dependency ordering fails, or cuFFT rejects
    /// the transform submission. With no explicit stream, synchronizing input
    /// dependencies can also fail. Success reports submission, not device completion.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the unsafe execution contract requires native alignment after region type erasure"
    )]
    pub unsafe fn execute_c2c_f32_async(
        &self,
        input: &(impl DeviceInput<Complex32> + ?Sized),
        output: DeviceOutputView<'_, Complex32>,
        direction: Direction,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let input = input.region();
        let output = output.into_region();
        self.prepare_execution(stream, &[input, output])?;
        // SAFETY: `cufftExecC2C` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cufftExecC2C(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftComplex>(),
                output.as_ptr().cast::<sys::cufftComplex>(),
                direction as _,
            ))?;
        }
        Ok(())
    }

    /// Queues a single-precision complex transform over the same writable samples.
    /// Pending destinations are accepted without asserting their samples or padding are initialized.
    ///
    /// # Safety
    ///
    /// The configured transform, strides, batch count, capacities and native alignment must match these regions.
    /// All samples read by the transform must be initialized when execution reaches them.
    /// Writable samples must remain exclusive until completion, including submission failures and graph replays.
    /// Retain the plan, buffers, workspaces and callback resources through every queued use; do not reconfigure them meanwhile.
    ///
    /// # Errors
    ///
    /// Returns an error if a memory region or stream belongs to a different CUDA
    /// context, context binding or dependency ordering fails, or cuFFT rejects
    /// the transform submission. With no explicit stream, synchronizing input
    /// dependencies can also fail. Success reports submission, not device completion.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the unsafe execution contract requires native alignment after region type erasure"
    )]
    pub unsafe fn execute_c2c_f32_in_place_async(
        &self,
        data: DeviceOutputView<'_, Complex32>,
        direction: Direction,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let data = data.into_region();
        self.prepare_execution(stream, &[data])?;
        // SAFETY: `cufftExecC2C` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cufftExecC2C(
                self.as_raw(),
                data.as_ptr().cast::<sys::cufftComplex>(),
                data.as_ptr().cast::<sys::cufftComplex>(),
                direction as _,
            ))?;
        }
        Ok(())
    }

    /// Queues a real-to-complex transform producing the nonredundant half spectrum.
    /// Pending destinations are accepted without asserting their samples or padding are initialized.
    ///
    /// # Safety
    ///
    /// The configured transform, strides, batch count, capacities and native alignment must match these regions.
    /// All samples read by the transform must be initialized when execution reaches them.
    /// Writable samples must remain exclusive until completion, including submission failures and graph replays.
    /// Retain the plan, buffers, workspaces and callback resources through every queued use; do not reconfigure them meanwhile.
    ///
    /// # Errors
    ///
    /// Returns an error if a memory region or stream belongs to a different CUDA
    /// context, context binding or dependency ordering fails, or cuFFT rejects
    /// the transform submission. With no explicit stream, synchronizing input
    /// dependencies can also fail. Success reports submission, not device completion.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the unsafe execution contract requires native alignment after region type erasure"
    )]
    pub unsafe fn execute_r2c_f32_async(
        &self,
        input: &(impl DeviceInput<f32> + ?Sized),
        output: DeviceOutputView<'_, Complex32>,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let input = input.region();
        let output = output.into_region();
        self.prepare_execution(stream, &[input, output])?;
        // SAFETY: `cufftExecR2C` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cufftExecR2C(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftReal>(),
                output.as_ptr().cast::<sys::cufftComplex>(),
            ))?;
        }
        Ok(())
    }

    /// Queues a complex-to-real inverse, which may overwrite the input spectrum.
    /// Pending destinations are accepted without asserting their samples or padding are initialized.
    ///
    /// # Safety
    ///
    /// The configured transform, strides, batch count, capacities and native alignment must match these regions.
    /// All samples read by the transform must be initialized when execution reaches them.
    /// Writable samples must remain exclusive until completion, including submission failures and graph replays.
    /// Retain the plan, buffers, workspaces and callback resources through every queued use; do not reconfigure them meanwhile.
    ///
    /// # Errors
    ///
    /// Returns an error if a memory region or stream belongs to a different CUDA
    /// context, context binding or dependency ordering fails, or cuFFT rejects
    /// the transform submission. With no explicit stream, synchronizing input
    /// dependencies can also fail. Success reports submission, not device completion.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the unsafe execution contract requires native alignment after region type erasure"
    )]
    pub unsafe fn execute_c2r_f32_async(
        &self,
        input: DeviceOutputView<'_, Complex32>,
        output: DeviceOutputView<'_, f32>,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let input = input.into_region();
        let output = output.into_region();
        self.prepare_execution(stream, &[input, output])?;
        // SAFETY: `cufftExecC2R` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cufftExecC2R(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftComplex>(),
                output.as_ptr().cast::<sys::cufftReal>(),
            ))?;
        }
        Ok(())
    }

    /// Queues a double-precision complex transform into a separate destination.
    /// Pending destinations are accepted without asserting their samples or padding are initialized.
    ///
    /// # Safety
    ///
    /// The configured transform, strides, batch count, capacities and native alignment must match these regions.
    /// All samples read by the transform must be initialized when execution reaches them.
    /// Writable samples must remain exclusive until completion, including submission failures and graph replays.
    /// Retain the plan, buffers, workspaces and callback resources through every queued use; do not reconfigure them meanwhile.
    ///
    /// # Errors
    ///
    /// Returns an error if a memory region or stream belongs to a different CUDA
    /// context, context binding or dependency ordering fails, or cuFFT rejects
    /// the transform submission. With no explicit stream, synchronizing input
    /// dependencies can also fail. Success reports submission, not device completion.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the unsafe execution contract requires native alignment after region type erasure"
    )]
    pub unsafe fn execute_z2z_f64_async(
        &self,
        input: &(impl DeviceInput<Complex64> + ?Sized),
        output: DeviceOutputView<'_, Complex64>,
        direction: Direction,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let input = input.region();
        let output = output.into_region();
        self.prepare_execution(stream, &[input, output])?;
        // SAFETY: `cufftExecZ2Z` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cufftExecZ2Z(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftDoubleComplex>(),
                output.as_ptr().cast::<sys::cufftDoubleComplex>(),
                direction as _,
            ))?;
        }
        Ok(())
    }

    /// Queues a double-precision complex transform over the same writable samples.
    /// Pending destinations are accepted without asserting their samples or padding are initialized.
    ///
    /// # Safety
    ///
    /// The configured transform, strides, batch count, capacities and native alignment must match these regions.
    /// All samples read by the transform must be initialized when execution reaches them.
    /// Writable samples must remain exclusive until completion, including submission failures and graph replays.
    /// Retain the plan, buffers, workspaces and callback resources through every queued use; do not reconfigure them meanwhile.
    ///
    /// # Errors
    ///
    /// Returns an error if a memory region or stream belongs to a different CUDA
    /// context, context binding or dependency ordering fails, or cuFFT rejects
    /// the transform submission. With no explicit stream, synchronizing input
    /// dependencies can also fail. Success reports submission, not device completion.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the unsafe execution contract requires native alignment after region type erasure"
    )]
    pub unsafe fn execute_z2z_f64_in_place_async(
        &self,
        data: DeviceOutputView<'_, Complex64>,
        direction: Direction,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let data = data.into_region();
        self.prepare_execution(stream, &[data])?;
        // SAFETY: `cufftExecZ2Z` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cufftExecZ2Z(
                self.as_raw(),
                data.as_ptr().cast::<sys::cufftDoubleComplex>(),
                data.as_ptr().cast::<sys::cufftDoubleComplex>(),
                direction as _,
            ))?;
        }
        Ok(())
    }

    /// Queues a double-precision real transform producing the nonredundant half spectrum.
    /// Pending destinations are accepted without asserting their samples or padding are initialized.
    ///
    /// # Safety
    ///
    /// The configured transform, strides, batch count, capacities and native alignment must match these regions.
    /// All samples read by the transform must be initialized when execution reaches them.
    /// Writable samples must remain exclusive until completion, including submission failures and graph replays.
    /// Retain the plan, buffers, workspaces and callback resources through every queued use; do not reconfigure them meanwhile.
    ///
    /// # Errors
    ///
    /// Returns an error if a memory region or stream belongs to a different CUDA
    /// context, context binding or dependency ordering fails, or cuFFT rejects
    /// the transform submission. With no explicit stream, synchronizing input
    /// dependencies can also fail. Success reports submission, not device completion.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the unsafe execution contract requires native alignment after region type erasure"
    )]
    pub unsafe fn execute_d2z_f64_async(
        &self,
        input: &(impl DeviceInput<f64> + ?Sized),
        output: DeviceOutputView<'_, Complex64>,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let input = input.region();
        let output = output.into_region();
        self.prepare_execution(stream, &[input, output])?;
        // SAFETY: `cufftExecD2Z` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cufftExecD2Z(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftDoubleReal>(),
                output.as_ptr().cast::<sys::cufftDoubleComplex>(),
            ))?;
        }
        Ok(())
    }

    /// Queues a double-precision complex-to-real inverse, which may overwrite its spectrum.
    /// Pending destinations are accepted without asserting their samples or padding are initialized.
    ///
    /// # Safety
    ///
    /// The configured transform, strides, batch count, capacities and native alignment must match these regions.
    /// All samples read by the transform must be initialized when execution reaches them.
    /// Writable samples must remain exclusive until completion, including submission failures and graph replays.
    /// Retain the plan, buffers, workspaces and callback resources through every queued use; do not reconfigure them meanwhile.
    ///
    /// # Errors
    ///
    /// Returns an error if a memory region or stream belongs to a different CUDA
    /// context, context binding or dependency ordering fails, or cuFFT rejects
    /// the transform submission. With no explicit stream, synchronizing input
    /// dependencies can also fail. Success reports submission, not device completion.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the unsafe execution contract requires native alignment after region type erasure"
    )]
    pub unsafe fn execute_z2d_f64_async(
        &self,
        input: DeviceOutputView<'_, Complex64>,
        output: DeviceOutputView<'_, f64>,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let input = input.into_region();
        let output = output.into_region();
        self.prepare_execution(stream, &[input, output])?;
        // SAFETY: `cufftExecZ2D` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cufftExecZ2D(
                self.as_raw(),
                input.as_ptr().cast::<sys::cufftDoubleComplex>(),
                output.as_ptr().cast::<sys::cufftDoubleReal>(),
            ))?;
        }
        Ok(())
    }

    pub(crate) fn prepare_execution(
        &self,
        stream: Option<&Stream>,
        regions: &[DeviceRegion<'_>],
    ) -> Result<()> {
        if regions
            .iter()
            .any(|region| region.context() != self.cuda_context())
        {
            return Err(Error::StreamContextMismatch);
        }
        self.with_stream(stream)?;
        if let Some(stream) = stream {
            stream.join_regions(regions)?;
        } else {
            // No explicit stream is available to receive readiness dependencies.
            for region in regions {
                region.synchronize()?;
            }
        }
        Ok(())
    }

    fn with_stream(&self, stream: Option<&Stream>) -> Result<()> {
        if let Some(stream) = stream {
            if self.cuda_context() != stream.context() {
                return Err(Error::StreamContextMismatch);
            }
            let mut streams = self.handle.streams.borrow_mut();
            if !streams.contains(stream) {
                streams.push(stream.clone());
            }
        }
        self.bind()?;
        // SAFETY: the live plan retains the optional stream before native configuration.
        unsafe {
            try_ffi!(sys::cufftSetStream(
                self.as_raw(),
                stream.map_or_else(ptr::null_mut, Stream::as_raw),
            ))?;
        }
        Ok(())
    }

    /// Makes this plan's CUDA context current on the calling thread.
    ///
    /// # Errors
    ///
    /// Returns a CUDA error if the context cannot be made current.
    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    /// Borrows the CUDA context retained by this plan.
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_context
    }

    /// Borrows the native plan handle without transferring ownership.
    pub fn as_raw(&self) -> sys::cufftHandle {
        self.handle.raw
    }

    /// Reclaims a native plan transferred together with its dependent owners.
    ///
    /// # Safety
    ///
    /// `handle` must remain valid, uniquely owned, and associated with `cuda_context`.
    /// Retained streams must cover all streams still referenced by native state or queued work.
    /// Complete all borrowed work before releasing or replacing its resources.
    ///
    /// # Errors
    ///
    /// Returns an error if the transferred handle is null or any retained stream
    /// belongs to a different CUDA context. The rejected native handle is not destroyed.
    pub unsafe fn from_raw(raw: RawPlan) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::from_raw_parts(raw.handle, raw.cuda_context, raw.streams) }
    }

    /// Transfers the native handle together with its actual CUDA context and streams.
    pub fn into_raw(mut self) -> RawPlan {
        let handle = mem::replace(&mut self.handle.raw, sys::CUFFT_PLAN_NULL);
        RawPlan {
            handle,
            cuda_context: Arc::clone(&self.handle.cuda_context),
            streams: self.handle.streams.take(),
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if self.raw == sys::CUFFT_PLAN_NULL {
            return;
        }
        if let Err(error) = self.cuda_context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cufft handle: {error}");
        }

        // SAFETY: `cufftDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cufftDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cufft handle: {error}");
            }
        }
    }
}

fn validate_rank<T>(values: &[T]) -> Result<usize> {
    if values.is_empty() {
        return Err(Error::out_of_range("rank"));
    }

    Ok(values.len())
}
