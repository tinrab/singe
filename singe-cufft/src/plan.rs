mod sealed {
    use std::ffi::c_void;

    use singe_cufft_sys as sys;

    use crate::{
        error::Result,
        types::{Direction, TransformType},
    };

    #[allow(
        unnameable_types,
        reason = "sealed supertrait that callers must not name or implement"
    )]
    pub trait Index: Copy + Into<i64> {
        // Selects cuFFT's 64-bit planning interface.
        const WIDE: bool;
    }

    #[allow(
        unnameable_types,
        reason = "sealed supertrait that callers must not name or implement"
    )]
    pub trait Transform<O> {
        const KIND: TransformType;

        // The only direction a real transform runs in; complex transforms run in both.
        const DIRECTION: Option<Direction>;

        // Alignment of cuFFT's native element types, which can exceed the Rust element's.
        const INPUT_ALIGNMENT: usize;

        const OUTPUT_ALIGNMENT: usize;

        unsafe fn execute(
            plan: sys::cufftHandle,
            input: *mut c_void,
            output: *mut c_void,
            direction: Direction,
        ) -> Result<()>;
    }
}

use std::{cell::RefCell, convert::Infallible, ffi::c_void, fmt, ptr, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    error::Error as CudaError,
    handle::{CreateHandle, Handle, NativeHandle, RawHandle},
    stream::{Stream, StreamScope},
    submission::{Access, AccessMode},
    types::{Complex32, Complex64},
    view::{DeviceOutputView, DeviceRange, DeviceRegion, DeviceRepr, DeviceSliceMut},
};
use singe_cufft_sys as sys;

use crate::{
    error::{Error, Result, Status},
    try_ffi,
    types::{Direction, PlanProperty, TransformType},
    utility::{to_i32, to_i64, to_usize},
};

/// Marker type for the native cuFFT plan handle owned by [`Plan`].
#[derive(Debug)]
pub struct CufftPlan(Infallible);

// SAFETY: cuFFT plan handles may be used and destroyed from any thread, one call at a time.
unsafe impl NativeHandle for CufftPlan {
    type Raw = sys::cufftHandle;

    type State = ();

    type Error = Error;

    const OBJECT: &'static str = "cufft plan";

    fn null_handle() -> Error {
        Error::NullHandle
    }

    fn is_null(raw: Self::Raw) -> bool {
        raw == sys::CUFFT_PLAN_NULL
    }

    unsafe fn destroy(raw: Self::Raw, (): ()) -> Result<()> {
        // SAFETY: the caller passes a live, uniquely owned plan with its context bound.
        unsafe { try_ffi!(sys::cufftDestroy(raw)) }
    }
}

// SAFETY: `cufftCreate` creates the plan for the bound context's device.
unsafe impl CreateHandle for CufftPlan {
    unsafe fn create(raw: *mut Self::Raw) -> Result<()> {
        // SAFETY: the caller provides a writable output location and binds the context.
        unsafe { try_ffi!(sys::cufftCreate(raw)) }
    }
}

/// Raw ownership of a cuFFT plan, its CUDA context, and the stream the plan is associated with.
///
/// External workspaces, callbacks, queued buffers, and graph replays remain caller obligations.
pub type RawPlan = RawHandle<CufftPlan>;

/// An integer type of cuFFT's planning interfaces: `i32` or `i64`.
///
/// The trait is sealed.
pub trait PlanIndex: sealed::Index + fmt::Debug {}

impl sealed::Index for i32 {
    const WIDE: bool = false;
}

impl sealed::Index for i64 {
    const WIDE: bool = true;
}

impl PlanIndex for i32 {}

impl PlanIndex for i64 {}

/// Batched transform dimensions and element strides.
///
/// `I` selects cuFFT's 32-bit (`i32`) or 64-bit (`i64`) planning interface.
/// Dimensions are ordered from outermost to innermost.
/// An absent embedding uses the native contiguous layout; cuFFT ignores the associated strides and distances.
#[derive(Debug, Clone, Copy)]
pub struct ManyLayout<'a, I: PlanIndex> {
    /// Logical transform extents, ordered from outermost to innermost; the rank must be nonzero.
    pub n: &'a [I],
    /// Physical input extents, one per dimension, or `None` for contiguous input.
    pub inembed: Option<&'a [I]>,
    /// Distance in input elements between consecutive samples of the innermost dimension.
    pub istride: I,
    /// Distance in input elements between the starts of successive batches.
    pub idist: I,
    /// Physical output extents, one per dimension, or `None` for contiguous output.
    pub onembed: Option<&'a [I]>,
    /// Distance in output elements between consecutive samples of the innermost dimension.
    pub ostride: I,
    /// Distance in output elements between the starts of successive batches.
    pub odist: I,
    /// Transform data conversion and precision.
    pub kind: TransformType,
    /// Number of transforms sharing this shape and layout.
    pub batch: I,
}

/// An element type pair a cuFFT plan transforms, from `Self` to `O`.
///
/// Implemented for complex-to-complex, real-to-complex, and complex-to-real pairs of [`Complex32`] and `f32`, and of [`Complex64`] and `f64`.
/// The trait is sealed.
pub trait Transform<O: DeviceRepr>: DeviceRepr + sealed::Transform<O> {}

macro_rules! transform {
    ($input:ty => $output:ty, $kind:ident, $direction:expr, $native_input:ty => $native_output:ty, |$direction_parameter:ident| $function:ident($($argument:expr)?)) => {
        impl sealed::Transform<$output> for $input {
            const KIND: TransformType = TransformType::$kind;

            const DIRECTION: Option<Direction> = $direction;

            const INPUT_ALIGNMENT: usize = align_of::<$native_input>();

            const OUTPUT_ALIGNMENT: usize = align_of::<$native_output>();

            unsafe fn execute(
                plan: sys::cufftHandle,
                input: *mut c_void,
                output: *mut c_void,
                $direction_parameter: Direction,
            ) -> Result<()> {
                // SAFETY: the caller passes a live plan configured for this transform and aligned operands it covers.
                unsafe {
                    try_ffi!(sys::$function(
                        plan,
                        input.cast::<$native_input>(),
                        output.cast::<$native_output>(),
                        $($argument)?
                    ))
                }
            }
        }

        impl Transform<$output> for $input {}
    };
}

transform!(Complex32 => Complex32, ComplexToComplex, None, sys::cufftComplex => sys::cufftComplex, |direction| cufftExecC2C(i32::from(direction)));

transform!(f32 => Complex32, RealToComplex, Some(Direction::Forward), sys::cufftReal => sys::cufftComplex, |_direction| cufftExecR2C());

transform!(Complex32 => f32, ComplexToReal, Some(Direction::Inverse), sys::cufftComplex => sys::cufftReal, |_direction| cufftExecC2R());

transform!(Complex64 => Complex64, DoubleComplexToDoubleComplex, None, sys::cufftDoubleComplex => sys::cufftDoubleComplex, |direction| cufftExecZ2Z(i32::from(direction)));

transform!(f64 => Complex64, DoubleRealToDoubleComplex, Some(Direction::Forward), sys::cufftDoubleReal => sys::cufftDoubleComplex, |_direction| cufftExecD2Z());

transform!(Complex64 => f64, DoubleComplexToDoubleReal, Some(Direction::Inverse), sys::cufftDoubleComplex => sys::cufftDoubleReal, |_direction| cufftExecZ2D());

// The transform type and the minimum element counts of its operands, recorded when the plan is configured.
#[derive(Debug, Clone, Copy)]
struct PlanShape {
    kind: TransformType,
    input_len: usize,
    output_len: usize,
}

// A layout widened to `i64`, or narrowed to `i32` for the 32-bit interface.
#[derive(Debug)]
struct NativeLayout<T> {
    rank: i32,
    n: Vec<T>,
    inembed: Option<Vec<T>>,
    istride: T,
    idist: T,
    onembed: Option<Vec<T>>,
    ostride: T,
    odist: T,
    kind: TransformType,
    batch: T,
}

/// A stateful cuFFT plan handle.
///
/// Use one plan per host thread or concurrent task.
/// The plan is movable between threads, but it is intentionally not `Clone` or `Sync`.
/// Planning methods configure the transform and return the required workspace size in bytes.
/// Executions take a stream per call; the plan retains the last stream it ran on.
#[derive(Debug)]
pub struct Plan {
    handle: Handle<CufftPlan>,
    // The stream the native plan is associated with; cuFFT references no earlier stream.
    stream: RefCell<Option<Stream>>,
    shape: Option<PlanShape>,
    auto_allocation: bool,
    work_area_bytes: Option<usize>,
}

impl Plan {
    /// Creates an unconfigured cuFFT plan in `cuda_context`.
    ///
    /// Configure its transform before execution.
    /// cuFFT allocates the workspace during planning unless [`set_auto_allocation`](Self::set_auto_allocation) disables it.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, cuFFT cannot create the plan, or the returned handle is null.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            handle: Handle::create(cuda_context)?,
            stream: RefCell::new(None),
            shape: None,
            auto_allocation: true,
            work_area_bytes: None,
        })
    }

    /// Enables or disables cuFFT allocation of workspace during planning.
    ///
    /// When disabled, install a sufficient work area with [`set_work_area`](Self::set_work_area) before execution.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails or cuFFT rejects the allocation setting.
    pub fn set_auto_allocation(&mut self, auto_allocate: bool) -> Result<()> {
        self.bind()?;
        // SAFETY: the plan handle is live and its context is bound.
        unsafe {
            try_ffi!(sys::cufftSetAutoAllocation(
                self.as_raw(),
                i32::from(auto_allocate),
            ))?;
        }

        self.auto_allocation = auto_allocate;
        Ok(())
    }

    /// Installs externally owned workspace storage without claiming its bytes are initialized, or clears it for `None`.
    ///
    /// # Safety
    ///
    /// Retain the workspace exclusively through all queued uses and native plan destruction, or replace it after completion.
    /// Captured graph replays keep the same workspace lifetime and access obligations.
    ///
    /// # Errors
    ///
    /// Returns an error if the work area belongs to another CUDA context, is smaller than [`workspace_size`](Self::workspace_size) of a configured plan, context binding fails, or cuFFT rejects the pointer.
    pub unsafe fn set_work_area(
        &mut self,
        work_area: Option<DeviceOutputView<'_, u8>>,
    ) -> Result<()> {
        let region = work_area.map(DeviceOutputView::into_region);
        if let Some(region) = region {
            if region.context() != self.cuda_context() {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            if self.shape.is_some() {
                let required = self.workspace_size()?;
                if region.byte_len() < required {
                    return Err(Error::BufferTooSmall {
                        name: "work area".into(),
                        required,
                        actual: region.byte_len(),
                    });
                }
            }
        }

        self.bind()?;
        let pointer = region.map_or(ptr::null_mut(), |region| region.as_ptr().cast());
        // SAFETY: the plan handle is live and its context is bound; the caller retains the work area.
        unsafe {
            try_ffi!(sys::cufftSetWorkArea(self.as_raw(), pointer))?;
        }

        self.work_area_bytes = region.map(DeviceRegion::byte_len);
        Ok(())
    }

    /// Sets an integer plan property before planning.
    ///
    /// Property-specific restrictions and supported values are defined by cuFFT.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails or cuFFT rejects the property, value, or current planning state.
    pub fn set_plan_property(&mut self, property: PlanProperty, value: i64) -> Result<()> {
        self.bind()?;
        // SAFETY: the plan handle is live and its context is bound.
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
        // SAFETY: the plan handle is live, its context is bound, and `value` is a valid output location.
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
    /// Returns an error if context binding fails or cuFFT rejects the property or the current planning state.
    pub fn reset_plan_property(&mut self, property: PlanProperty) -> Result<()> {
        self.bind()?;
        // SAFETY: the plan handle is live and its context is bound.
        unsafe {
            try_ffi!(sys::cufftResetPlanProperty(self.as_raw(), property.into()))?;
        }
        Ok(())
    }

    /// Configures `batch` transforms of length `nx` and returns required workspace bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if an argument is zero or does not fit `i32`, context binding fails, cuFFT cannot plan the transform, or the workspace size does not fit `usize`.
    pub fn plan_1d(&mut self, nx: usize, kind: TransformType, batch: usize) -> Result<usize> {
        let shape = PlanShape::contiguous(kind, &[nx], batch)?;
        self.bind()?;
        let mut work_size: sys::size_t = 0;
        // SAFETY: the plan handle is live, its context is bound, and `work_size` is a valid output location.
        unsafe {
            try_ffi!(sys::cufftMakePlan1d(
                self.as_raw(),
                to_i32(nx, "nx")?,
                kind.into(),
                to_i32(batch, "batch")?,
                &raw mut work_size,
            ))?;
        }

        self.shape = Some(shape);
        to_usize(work_size, "work_size")
    }

    /// Configures one row-major transform of shape `nx` by `ny` and returns required workspace bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if an argument is zero or does not fit `i32`, context binding fails, cuFFT cannot plan the transform, or the workspace size does not fit `usize`.
    pub fn plan_2d(&mut self, nx: usize, ny: usize, kind: TransformType) -> Result<usize> {
        let shape = PlanShape::contiguous(kind, &[nx, ny], 1)?;
        self.bind()?;
        let mut work_size: sys::size_t = 0;
        // SAFETY: the plan handle is live, its context is bound, and `work_size` is a valid output location.
        unsafe {
            try_ffi!(sys::cufftMakePlan2d(
                self.as_raw(),
                to_i32(nx, "nx")?,
                to_i32(ny, "ny")?,
                kind.into(),
                &raw mut work_size,
            ))?;
        }

        self.shape = Some(shape);
        to_usize(work_size, "work_size")
    }

    /// Configures one row-major transform of shape `nx` by `ny` by `nz` and returns required workspace bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if an argument is zero or does not fit `i32`, context binding fails, cuFFT cannot plan the transform, or the workspace size does not fit `usize`.
    pub fn plan_3d(
        &mut self,
        nx: usize,
        ny: usize,
        nz: usize,
        kind: TransformType,
    ) -> Result<usize> {
        let shape = PlanShape::contiguous(kind, &[nx, ny, nz], 1)?;
        self.bind()?;
        let mut work_size: sys::size_t = 0;
        // SAFETY: the plan handle is live, its context is bound, and `work_size` is a valid output location.
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

        self.shape = Some(shape);
        to_usize(work_size, "work_size")
    }

    /// Configures a batched transform and returns required workspace bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the rank is zero or does not fit `i32`, an embedding length differs from the rank, an extent, stride, or batch count is out of range, context binding fails, cuFFT cannot plan the transform, or the workspace size does not fit `usize`.
    pub fn plan_many<I: PlanIndex>(&mut self, layout: &ManyLayout<'_, I>) -> Result<usize> {
        let mut native = layout.widened()?;
        let shape = native.shape()?;
        self.bind()?;
        let mut work_size: sys::size_t = 0;
        if I::WIDE {
            // SAFETY: the plan handle is live, its context is bound, and every array holds `rank` values that outlive the call.
            unsafe {
                try_ffi!(sys::cufftMakePlanMany64(
                    self.as_raw(),
                    native.rank,
                    native.n.as_mut_ptr(),
                    embedding_ptr(&mut native.inembed),
                    native.istride,
                    native.idist,
                    embedding_ptr(&mut native.onembed),
                    native.ostride,
                    native.odist,
                    native.kind.into(),
                    native.batch,
                    &raw mut work_size,
                ))?;
            }
        } else {
            let mut native = native.narrowed()?;
            // SAFETY: the plan handle is live, its context is bound, and every array holds `rank` values that outlive the call.
            unsafe {
                try_ffi!(sys::cufftMakePlanMany(
                    self.as_raw(),
                    native.rank,
                    native.n.as_mut_ptr(),
                    embedding_ptr(&mut native.inembed),
                    native.istride,
                    native.idist,
                    embedding_ptr(&mut native.onembed),
                    native.ostride,
                    native.odist,
                    native.kind.into(),
                    native.batch,
                    &raw mut work_size,
                ))?;
            }
        }

        self.shape = Some(shape);
        to_usize(work_size, "work_size")
    }

    /// Queries workspace bytes for `batch` transforms of length `nx` using this plan's settings.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails, an argument does not fit `i32`, cuFFT cannot compute the workspace size, or the result does not fit `usize`.
    pub fn workspace_size_1d(&self, nx: usize, kind: TransformType, batch: usize) -> Result<usize> {
        self.bind()?;
        let mut work_size: sys::size_t = 0;
        // SAFETY: the plan handle is live, its context is bound, and `work_size` is a valid output location.
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
    /// Returns an error if context binding fails, an argument does not fit `i32`, cuFFT cannot compute the workspace size, or the result does not fit `usize`.
    pub fn workspace_size_2d(&self, nx: usize, ny: usize, kind: TransformType) -> Result<usize> {
        self.bind()?;
        let mut work_size: sys::size_t = 0;
        // SAFETY: the plan handle is live, its context is bound, and `work_size` is a valid output location.
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
    /// Returns an error if context binding fails, an argument does not fit `i32`, cuFFT cannot compute the workspace size, or the result does not fit `usize`.
    pub fn workspace_size_3d(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        kind: TransformType,
    ) -> Result<usize> {
        self.bind()?;
        let mut work_size: sys::size_t = 0;
        // SAFETY: the plan handle is live, its context is bound, and `work_size` is a valid output location.
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
    /// Returns an error if the rank is zero or does not fit `i32`, an embedding length differs from the rank, context binding fails, cuFFT cannot compute the workspace size, or the result does not fit `usize`.
    pub fn workspace_size_many<I: PlanIndex>(&self, layout: &ManyLayout<'_, I>) -> Result<usize> {
        let mut native = layout.widened()?;
        self.bind()?;
        let mut work_size: sys::size_t = 0;
        if I::WIDE {
            // SAFETY: the plan handle is live, its context is bound, and every array holds `rank` values that outlive the call.
            unsafe {
                try_ffi!(sys::cufftGetSizeMany64(
                    self.as_raw(),
                    native.rank,
                    native.n.as_mut_ptr(),
                    embedding_ptr(&mut native.inembed),
                    native.istride,
                    native.idist,
                    embedding_ptr(&mut native.onembed),
                    native.ostride,
                    native.odist,
                    native.kind.into(),
                    native.batch,
                    &raw mut work_size,
                ))?;
            }
        } else {
            let mut native = native.narrowed()?;
            // SAFETY: the plan handle is live, its context is bound, and every array holds `rank` values that outlive the call.
            unsafe {
                try_ffi!(sys::cufftGetSizeMany(
                    self.as_raw(),
                    native.rank,
                    native.n.as_mut_ptr(),
                    embedding_ptr(&mut native.inembed),
                    native.istride,
                    native.idist,
                    embedding_ptr(&mut native.onembed),
                    native.ostride,
                    native.odist,
                    native.kind.into(),
                    native.batch,
                    &raw mut work_size,
                ))?;
            }
        }
        to_usize(work_size, "work_size")
    }

    /// Returns the workspace size in bytes for this plan's configured transform.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails, cuFFT cannot query the configured plan, or the workspace size does not fit `usize`.
    pub fn workspace_size(&self) -> Result<usize> {
        self.bind()?;
        let mut work_size: sys::size_t = 0;
        // SAFETY: the plan handle is live, its context is bound, and `work_size` is a valid output location.
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
    /// Returns an error if an argument does not fit `i32`, cuFFT cannot estimate the workspace size, or the result does not fit `usize`.
    pub fn estimate_1d(nx: usize, kind: TransformType, batch: usize) -> Result<usize> {
        let mut work_size: sys::size_t = 0;
        // SAFETY: `work_size` is a valid output location.
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
    /// Returns an error if an argument does not fit `i32`, cuFFT cannot estimate the workspace size, or the result does not fit `usize`.
    pub fn estimate_2d(nx: usize, ny: usize, kind: TransformType) -> Result<usize> {
        let mut work_size: sys::size_t = 0;
        // SAFETY: `work_size` is a valid output location.
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
    /// Returns an error if an argument does not fit `i32`, cuFFT cannot estimate the workspace size, or the result does not fit `usize`.
    pub fn estimate_3d(nx: usize, ny: usize, nz: usize, kind: TransformType) -> Result<usize> {
        let mut work_size: sys::size_t = 0;
        // SAFETY: `work_size` is a valid output location.
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

    /// Estimates batched-transform workspace bytes without creating a plan.
    ///
    /// cuFFT estimates through its 32-bit interface only, so every value of an `i64` layout must also fit `i32`.
    /// The estimate may differ from the size returned by actual planning.
    ///
    /// # Errors
    ///
    /// Returns an error if the rank is zero, an embedding length differs from the rank, a value does not fit `i32`, cuFFT cannot estimate the workspace size, or the result does not fit `usize`.
    pub fn estimate_many<I: PlanIndex>(layout: &ManyLayout<'_, I>) -> Result<usize> {
        let mut native = layout.widened()?.narrowed()?;
        let mut work_size: sys::size_t = 0;
        // SAFETY: every array holds `rank` values that outlive the call, and `work_size` is a valid output location.
        unsafe {
            try_ffi!(sys::cufftEstimateMany(
                native.rank,
                native.n.as_mut_ptr(),
                embedding_ptr(&mut native.inembed),
                native.istride,
                native.idist,
                embedding_ptr(&mut native.onembed),
                native.ostride,
                native.odist,
                native.kind.into(),
                native.batch,
                &raw mut work_size,
            ))?;
        }
        to_usize(work_size, "work_size")
    }

    /// Enqueues a transform from `input` into a separate `output` on `stream`.
    ///
    /// Real-to-complex transforms run [`Direction::Forward`] and complex-to-real transforms [`Direction::Inverse`].
    /// The output is accepted without asserting its samples or padding are initialized.
    ///
    /// # Safety
    ///
    /// Follow the crate's raw execution contract.
    /// A complex-to-real transform may overwrite `input`, so the caller must also hold it exclusively through completion.
    ///
    /// # Errors
    ///
    /// Returns an error if the direction does not suit the transform, the element types differ from the configured transform, an operand is shorter than the configured layout or misaligned for cuFFT, a region or the stream belongs to another CUDA context, context binding or dependency ordering fails, or cuFFT rejects the submission.
    pub unsafe fn execute_async<I: Transform<O>, O: DeviceRepr>(
        &self,
        input: &(impl DeviceRange<I> + ?Sized),
        output: DeviceOutputView<'_, O>,
        direction: Direction,
        stream: &Stream,
    ) -> Result<()> {
        self.check_transform::<I, O>(direction, input.len(), output.len())?;
        let input = input.region();
        let output = output.into_region();
        input.ensure_aligned(I::INPUT_ALIGNMENT, "input")?;
        output.ensure_aligned(I::OUTPUT_ALIGNMENT, "output")?;
        self.prepare_execution(Some(stream), &[input, output])?;
        // SAFETY: `check_transform` and `ensure_aligned` matched the plan's transform, lengths, and alignment, `prepare_execution` ordered it on `stream` after the operands' producers, and the caller keeps the plan and operands alive and exclusive until completion.
        unsafe {
            I::execute(
                self.as_raw(),
                input.as_ptr().cast(),
                output.as_ptr().cast(),
                direction,
            )
        }
    }

    /// Enqueues a complex-to-complex transform over the same samples on `stream`.
    ///
    /// # Safety
    ///
    /// Follow the crate's raw execution contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the element type differs from the configured transform, `data` is shorter than the configured layout or misaligned for cuFFT, the region or the stream belongs to another CUDA context, context binding or dependency ordering fails, or cuFFT rejects the submission.
    pub unsafe fn execute_in_place_async<T: Transform<T>>(
        &self,
        data: DeviceOutputView<'_, T>,
        direction: Direction,
        stream: &Stream,
    ) -> Result<()> {
        self.check_transform::<T, T>(direction, data.len(), data.len())?;
        let data = data.into_region();
        data.ensure_aligned(T::INPUT_ALIGNMENT, "data")?;
        self.prepare_execution(Some(stream), &[data])?;
        // SAFETY: `check_transform` and `ensure_aligned` matched the plan's transform, length, and alignment, `prepare_execution` ordered it on `stream` after the data's producers, and the caller keeps the plan and `data` alive and exclusive until completion.
        unsafe {
            T::execute(
                self.as_raw(),
                data.as_ptr().cast(),
                data.as_ptr().cast(),
                direction,
            )
        }
    }

    /// Transforms `input` into a separate `output` on `stream` and waits for the transform.
    ///
    /// cuFFT may overwrite the input of complex-to-real transforms, so `input` is borrowed exclusively.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan has no configured transform or work area, `stream` is capturing, or [`execute_async`](Self::execute_async) or the wait fails.
    pub fn execute<I: Transform<O>, O: DeviceRepr>(
        &self,
        input: &mut (impl DeviceSliceMut<I> + ?Sized),
        output: DeviceOutputView<'_, O>,
        direction: Direction,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_executable()?;
        stream.complete(|| {
            // SAFETY: the configured shape bounds both exclusive operands, which outlive the scope, and the scope waits for the transform, also on failure.
            unsafe { self.execute_async(&*input, output, direction, stream) }
        })
    }

    /// Transforms `data` in place on `stream` and waits for the transform.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan has no configured transform or work area, `stream` is capturing, or [`execute_in_place_async`](Self::execute_in_place_async) or the wait fails.
    pub fn execute_in_place<T: Transform<T>>(
        &self,
        data: &mut (impl DeviceSliceMut<T> + ?Sized),
        direction: Direction,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_executable()?;
        stream.complete(|| {
            // SAFETY: the configured shape bounds the exclusive operand, which outlives the scope, and the scope waits for the transform, also on failure.
            unsafe {
                self.execute_in_place_async(DeviceOutputView::from_output(data), direction, stream)
            }
        })
    }

    /// Enqueues a transform from `input` into a separate `output` on the scope's stream.
    ///
    /// cuFFT may overwrite the input of complex-to-real transforms, so `input` is borrowed exclusively.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan has no configured transform or work area, access tracking fails, or [`execute_async`](Self::execute_async) fails.
    pub fn execute_scoped<'env, I: Transform<O>, O: DeviceRepr>(
        &'env self,
        input: &'env mut (impl DeviceSliceMut<I> + ?Sized),
        mut output: DeviceOutputView<'env, O>,
        direction: Direction,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        self.ensure_executable()?;
        let accesses = [
            Access::new(input.region(), AccessMode::ReadWrite),
            Access::new(output.reborrow().into_region(), AccessMode::Write),
        ];
        // SAFETY: the transform touches only the declared operands on the scope's stream, and the plan and both operands are borrowed for `'env`, which the scope keeps alive until it drains.
        unsafe {
            scope.enqueue(&accesses, || {
                Ok(self.execute_async(&*input, output, direction, scope.stream()))
            })
        }?
    }

    /// Enqueues an in-place transform of `data` on the scope's stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan has no configured transform or work area, access tracking fails, or [`execute_in_place_async`](Self::execute_in_place_async) fails.
    pub fn execute_in_place_scoped<'env, T: Transform<T>>(
        &'env self,
        data: &'env mut (impl DeviceSliceMut<T> + ?Sized),
        direction: Direction,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        self.ensure_executable()?;
        let accesses = [Access::new(data.region(), AccessMode::ReadWrite)];
        // SAFETY: the transform touches only the declared operand on the scope's stream, and the plan and the operand are borrowed for `'env`, which the scope keeps alive until it drains.
        unsafe {
            scope.enqueue(&accesses, || {
                Ok(self.execute_in_place_async(
                    DeviceOutputView::from_output(data),
                    direction,
                    scope.stream(),
                ))
            })
        }?
    }

    // Rejects a plan whose operands or workspace the wrapper cannot bound.
    fn ensure_executable(&self) -> Result<()> {
        if self.shape.is_none() {
            return Err(Error::PlanNotConfigured);
        }

        if !self.auto_allocation {
            let required = self.workspace_size()?;
            let actual = self.work_area_bytes.unwrap_or(0);
            if self.work_area_bytes.is_none() && required > 0 {
                return Err(Error::Cufft {
                    code: Status::NoWorkspace,
                });
            }

            if actual < required {
                return Err(Error::BufferTooSmall {
                    name: "work area".into(),
                    required,
                    actual,
                });
            }
        }
        Ok(())
    }

    fn check_transform<I: Transform<O>, O: DeviceRepr>(
        &self,
        direction: Direction,
        input_len: usize,
        output_len: usize,
    ) -> Result<()> {
        if let Some(expected) = I::DIRECTION
            && expected != direction
        {
            return Err(Error::UnsupportedDirection {
                kind: I::KIND,
                direction,
            });
        }

        let Some(shape) = self.shape else {
            return Ok(());
        };
        if shape.kind != I::KIND {
            return Err(Error::TransformTypeMismatch {
                expected: shape.kind,
                actual: I::KIND,
            });
        }

        for (name, required, actual) in [
            ("input", shape.input_len, input_len),
            ("output", shape.output_len, output_len),
        ] {
            if actual < required {
                return Err(Error::BufferTooSmall {
                    name: name.into(),
                    required,
                    actual,
                });
            }
        }
        Ok(())
    }

    // Associates the plan with `stream`, or the legacy default stream for `None`, after ordering it behind the regions' producers.
    // Without a stream, the regions' producers are waited for on the host.
    pub(crate) fn prepare_execution(
        &self,
        stream: Option<&Stream>,
        regions: &[DeviceRegion<'_>],
    ) -> Result<()> {
        if regions
            .iter()
            .any(|region| region.context() != self.cuda_context())
        {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        self.with_stream(stream)?;
        if let Some(stream) = stream {
            stream.join_regions(regions)?;
        } else {
            for region in regions {
                region.synchronize()?;
            }
        }
        Ok(())
    }

    // Associates the native plan with `stream`, or the legacy default stream for `None`, and retains it.
    // The previously associated stream is released, since cuFFT no longer references it.
    pub(crate) fn with_stream(&self, stream: Option<&Stream>) -> Result<()> {
        if stream.is_some_and(|stream| self.cuda_context() != stream.context()) {
            return Err(Error::StreamContextMismatch);
        }

        let previous = self.stream.borrow().clone();
        if previous.as_ref() == stream {
            return Ok(());
        }

        self.bind()?;
        // Executions of one plan share its work area, so work on the new stream waits for work queued on the previous one.
        if let (Some(next), Some(previous)) = (stream, &previous) {
            next.join(previous)?;
        }

        // SAFETY: the plan handle is live, its context is bound, and the stream is null or live; the plan retains it below.
        unsafe {
            try_ffi!(sys::cufftSetStream(
                self.as_raw(),
                stream.map_or_else(ptr::null_mut, Stream::as_raw),
            ))?;
        }

        *self.stream.borrow_mut() = stream.cloned();
        Ok(())
    }

    pub(crate) fn bind(&self) -> Result<()> {
        self.handle.bind()
    }

    /// Returns the CUDA context the plan was created in.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Returns the native plan handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cufftHandle {
        self.handle.as_raw()
    }

    /// Adopts a native plan with its CUDA context and associated stream.
    ///
    /// The adopted plan has no recorded transform, so only its unsafe executions run until it is planned again.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a live, uniquely owned plan handle created for `raw.cuda_context`, associated with `raw.stream`, or with the default stream for `None`.
    /// Its workspace must be allocated by cuFFT or supplied by the caller for every later execution.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is the null plan handle or `raw.stream` belongs to another CUDA context.
    /// The rejected native handle is not destroyed.
    pub unsafe fn from_raw(mut raw: RawPlan) -> Result<Self> {
        let stream = raw.stream.take();
        if stream
            .as_ref()
            .is_some_and(|stream| stream.context() != &raw.cuda_context)
        {
            return Err(Error::StreamContextMismatch);
        }

        Ok(Self {
            // SAFETY: the caller guarantees `raw.handle` is a live, uniquely owned plan created for `raw.cuda_context`; the plan retains its stream separately.
            handle: unsafe { Handle::from_raw(raw)? },
            stream: RefCell::new(stream),
            shape: None,
            auto_allocation: true,
            work_area_bytes: None,
        })
    }

    /// Transfers the native plan, its CUDA context, and its associated stream to the caller.
    #[must_use]
    pub fn into_raw(self) -> RawPlan {
        let Self { handle, stream, .. } = self;
        let mut raw = handle.into_raw();
        raw.stream = stream.into_inner();
        raw
    }
}

impl PlanShape {
    fn contiguous(kind: TransformType, n: &[usize], batch: usize) -> Result<Self> {
        let n = n
            .iter()
            .map(|&extent| to_i64(extent, "transform extent"))
            .collect::<Result<Vec<_>>>()?;
        Self::create(kind, &n, to_i64(batch, "batch")?, None, None)
    }

    // cuFFT's documented layouts address sample `x` of batch `b` at `b * dist + stride * (x[0] * embed[1] * ... + x[rank - 1])`.
    // When either embedding is absent, cuFFT may lay out both sides contiguously, so each side needs the larger of both spans.
    fn create(
        kind: TransformType,
        n: &[i64],
        batch: i64,
        input: Option<(&[i64], i64, i64)>,
        output: Option<(&[i64], i64, i64)>,
    ) -> Result<Self> {
        if n.is_empty() || batch <= 0 || n.iter().any(|&extent| extent <= 0) {
            return Err(Error::out_of_range("transform dimensions"));
        }

        let mut half = n.to_vec();
        if let Some(last) = half.last_mut() {
            *last = *last / 2 + 1;
        }

        let (input_n, output_n) = match kind {
            TransformType::RealToComplex | TransformType::DoubleRealToDoubleComplex => {
                (n.to_vec(), half)
            }
            TransformType::ComplexToReal | TransformType::DoubleComplexToDoubleReal => {
                (half, n.to_vec())
            }
            _ => (n.to_vec(), n.to_vec()),
        };
        Ok(Self {
            kind,
            input_len: required_elements(&input_n, batch, input)?,
            output_len: required_elements(&output_n, batch, output)?,
        })
    }
}

fn required_elements(
    n: &[i64],
    batch: i64,
    embedding: Option<(&[i64], i64, i64)>,
) -> Result<usize> {
    let overflow = || Error::out_of_range("transform span");
    let contiguous = n
        .iter()
        .try_fold(batch, |elements, &extent| elements.checked_mul(extent))
        .ok_or_else(overflow)?;
    let elements = match embedding {
        None => contiguous,
        Some((embedding, stride, distance)) => {
            if stride <= 0
                || distance < 0
                || embedding.len() != n.len()
                || embedding
                    .iter()
                    .zip(n)
                    .skip(1)
                    .any(|(embed, extent)| embed < extent)
            {
                return Err(Error::out_of_range("transform embedding"));
            }

            let mut offset = 0i64;
            for (&extent, &embed) in n.iter().zip(embedding) {
                offset = offset
                    .checked_mul(embed)
                    .and_then(|offset| offset.checked_add(extent - 1))
                    .ok_or_else(overflow)?;
            }

            let embedded = offset
                .checked_mul(stride)
                .and_then(|offset| {
                    (batch - 1)
                        .checked_mul(distance)
                        .and_then(|batches| offset.checked_add(batches))
                })
                .and_then(|offset| offset.checked_add(1))
                .ok_or_else(overflow)?;
            embedded.max(contiguous)
        }
    };
    to_usize(elements, "transform elements")
}

impl<I: PlanIndex> ManyLayout<'_, I> {
    fn widened(&self) -> Result<NativeLayout<i64>> {
        if self.n.is_empty() {
            return Err(Error::out_of_range("rank"));
        }

        let rank = self.n.len();
        let widen = |values: &[I]| {
            values
                .iter()
                .map(|&value| value.into())
                .collect::<Vec<i64>>()
        };
        let embedding = |values: Option<&[I]>, name: &'static str| {
            values
                .map(|values| {
                    if values.len() == rank {
                        Ok(widen(values))
                    } else {
                        Err(Error::LengthMismatch {
                            name: name.into(),
                            expected: rank,
                            actual: values.len(),
                        })
                    }
                })
                .transpose()
        };
        Ok(NativeLayout {
            rank: to_i32(rank, "rank")?,
            n: widen(self.n),
            inembed: embedding(self.inembed, "inembed")?,
            istride: self.istride.into(),
            idist: self.idist.into(),
            onembed: embedding(self.onembed, "onembed")?,
            ostride: self.ostride.into(),
            odist: self.odist.into(),
            kind: self.kind,
            batch: self.batch.into(),
        })
    }
}

impl NativeLayout<i64> {
    fn narrowed(&self) -> Result<NativeLayout<i32>> {
        let narrow = |values: &[i64], name: &str| {
            values
                .iter()
                .map(|&value| to_i32(value, name))
                .collect::<Result<Vec<_>>>()
        };
        Ok(NativeLayout {
            rank: self.rank,
            n: narrow(&self.n, "n")?,
            inembed: self
                .inembed
                .as_deref()
                .map(|values| narrow(values, "inembed"))
                .transpose()?,
            istride: to_i32(self.istride, "istride")?,
            idist: to_i32(self.idist, "idist")?,
            onembed: self
                .onembed
                .as_deref()
                .map(|values| narrow(values, "onembed"))
                .transpose()?,
            ostride: to_i32(self.ostride, "ostride")?,
            odist: to_i32(self.odist, "odist")?,
            kind: self.kind,
            batch: to_i32(self.batch, "batch")?,
        })
    }

    fn shape(&self) -> Result<PlanShape> {
        PlanShape::create(
            self.kind,
            &self.n,
            self.batch,
            self.inembed
                .as_deref()
                .map(|embedding| (embedding, self.istride, self.idist)),
            self.onembed
                .as_deref()
                .map(|embedding| (embedding, self.ostride, self.odist)),
        )
    }
}

fn embedding_ptr<T>(values: &mut Option<Vec<T>>) -> *mut T {
    values.as_mut().map_or(ptr::null_mut(), Vec::as_mut_ptr)
}
