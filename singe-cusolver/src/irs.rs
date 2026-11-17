//! Mixed-precision iterative-refinement parameters, results, and linear solvers.

#![expect(
    clippy::many_single_char_names,
    reason = "BLAS and factorization operand names follow the standard mathematical API"
)]
use std::{mem, ptr};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    data_type::DataTypeLike,
    view::{DeviceOutput, DeviceOutputView, DeviceRepr},
};

use crate::{
    context::Context,
    error::{Error, Result},
    layout::{MatrixMut, MatrixOutput, MatrixRef},
    sys, try_ffi,
    types::{IrsFallback, IrsRefinement, PrecisionType},
    utility::{require_workspace_bytes, to_i32, to_u64, to_usize, validate_matrix},
};

/// Owns iterative-refinement configuration and tracks its main and lowest precisions.
///
/// Complete each solve before reconfiguring or destroying the parameters.
/// [`Self::create`] selects [`IrsRefinement::None`] initially.
#[derive(Debug)]
pub struct IrsParams {
    handle: sys::cusolverDnIRSParams_t,
    main_precision: Option<PrecisionType>,
    lowest_precision: Option<PrecisionType>,
}

/// Owns iteration counts and optional residual history for one iterative-refinement solve.
///
/// Create native storage with [`Self::create`].
/// Complete the solve before inspecting its results; raw execution must retain this object through completion.
#[derive(Debug)]
pub struct IrsInfos {
    handle: sys::cusolverDnIRSInfos_t,
    residual_history_requested: bool,
    main_precision: Option<PrecisionType>,
}

/// A transferred native IRS params handle with the precisions configured through its wrapper.
#[derive(Debug)]
pub struct RawIrsParams {
    /// The owned native IRS params handle.
    pub handle: sys::cusolverDnIRSParams_t,
    /// The main precision configured on `handle`, if known.
    pub main_precision: Option<PrecisionType>,
    /// The lowest precision configured on `handle`, if known.
    pub lowest_precision: Option<PrecisionType>,
}

/// A transferred native IRS infos handle with its residual-history request state.
#[derive(Debug)]
pub struct RawIrsInfos {
    /// The owned native IRS infos handle.
    pub handle: sys::cusolverDnIRSInfos_t,
    /// Whether residual history was requested on `handle`.
    pub residual_history_requested: bool,
    /// The main precision of the last solve that used `handle`, if known.
    pub main_precision: Option<PrecisionType>,
}

/// Iteration count and residual norm recorded at one outer refinement step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualHistoryEntry<T> {
    /// Cumulative iteration count at this outer step, stored in the residual scalar type.
    pub total_iterations: T,
    /// Residual norm recorded at this outer step; row zero describes the initial residual.
    pub residual_norm: T,
}

/// Owned copy of the valid rows from a completed native residual-history matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualHistory<T> {
    /// Valid outer-iteration rows, including the initial residual; undefined native trailing rows are omitted.
    pub rows: Vec<ResidualHistoryEntry<T>>,
    /// Original native history column stride in scalar elements, equal to the maximum iteration count plus one.
    ///
    /// This can exceed `rows.len()` because only valid rows are copied.
    pub leading_dimension: usize,
}

/// Dimensions for an iterative-refinement solve of a square linear system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrsSolve {
    /// Order of the square coefficient matrix.
    pub n: usize,
    /// Number of right-hand-side columns.
    pub right_hand_sides: usize,
}

impl IrsSolve {
    /// Records the system order and right-hand-side count without validating their native limits.
    #[must_use]
    pub fn new(n: usize, right_hand_sides: usize) -> Self {
        Self {
            n,
            right_hand_sides,
        }
    }

    /// Returns the device scratch requirement in bytes for `T` and the current IRS parameters.
    ///
    /// Fills any unset precision selectors from `T`.
    /// Requery after changing the solver configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, dimensions are zero or exceed native integer limits, the element type cannot be mapped to a supported precision, configured main precision differs from the element type, native precision setup or workspace querying fails, or the returned byte count does not fit `usize`.
    pub fn workspace_size<T: DataTypeLike>(
        self,
        context: &Context,
        params: &mut IrsParams,
    ) -> Result<usize> {
        xgesv_workspace_size::<T>(context, params, self.n, self.right_hand_sides)
    }

    /// Enqueues iterative refinement on the context's stream; [`Self::execute`] is the waiting form.
    ///
    /// Returns the native iteration count, as [`xgesv_async`] describes.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution), which also covers `params` and `infos`.
    ///
    /// # Errors
    ///
    /// Returns errors from [`xgesv_async`], including a buffer from another CUDA context, invalid matrix or status capacity, insufficient workspace, checked conversion failure, CUDA binding failure, or a native solver error.
    pub unsafe fn execute_async<T: DataTypeLike>(
        self,
        context: &Context,
        mut bindings: IrsSolveBindings<'_, T>,
        params: &mut IrsParams,
        infos: &mut IrsInfos,
    ) -> Result<i32> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            xgesv_async(
                context,
                params,
                infos,
                self.n,
                self.right_hand_sides,
                bindings.a,
                bindings.b,
                bindings.x,
                &mut bindings.device_workspace,
                &mut bindings.dev_info,
            )
        }
    }

    /// Solves the system by iterative refinement and waits for completion before returning the host iteration report.
    /// Inspect `dev_info` before using the solution.
    ///
    /// # Errors
    ///
    /// Returns errors from [`xgesv_async`], and an error if stream capture is active or invalidated or waiting for completion fails.
    pub fn execute<T: DataTypeLike>(
        self,
        context: &Context,
        bindings: IrsSolveBindings<'_, T>,
        params: &mut IrsParams,
        infos: &mut IrsInfos,
    ) -> Result<i32> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings and handles are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params, infos) })
    }
}

/// Borrowed matrices, output capacity, and device scratch for a square IRS solve.
///
/// These views retain their storage borrows only for the Rust call.
/// Unsafe execution requires the caller to retain all resources through native completion.
#[derive(Debug)]
pub struct IrsSolveBindings<'a, T: DeviceRepr> {
    /// Initialized square coefficient matrix; native solving may overwrite its logical contents.
    pub a: MatrixMut<'a, T>,
    /// Initialized right-hand-side matrix with `n` rows per column.
    pub b: MatrixRef<'a, T>,
    /// Exclusive solution capacity for `n` rows per right-hand-side column.
    /// Padding is not produced.
    pub x: MatrixOutput<'a, T>,
    /// Exclusive device scratch capacity in bytes, sized for the current IRS configuration.
    pub device_workspace: DeviceOutputView<'a, u8>,
    /// Exclusive device status capacity; only the first `i32` element is produced.
    pub dev_info: DeviceOutputView<'a, i32>,
}

// SAFETY: the IRS parameter structure is host memory with no thread affinity, owned uniquely by this wrapper, so it may be configured and destroyed from any thread.
unsafe impl Send for IrsParams {}

// SAFETY: every native setter takes `&mut self`, and the `&self` methods only call native getters that read the host structure, so shared references cannot race.
unsafe impl Sync for IrsParams {}

// SAFETY: the IRS infos structure is host memory with no thread affinity, owned uniquely by this wrapper, so it may be read and destroyed from any thread.
unsafe impl Send for IrsInfos {}

// SAFETY: requesting residual history takes `&mut self`, and the `&self` methods only call native getters that read the host structure, so shared references cannot race.
unsafe impl Sync for IrsInfos {}

// SAFETY: the record uniquely owns a host-memory parameter structure with no thread affinity.
unsafe impl Send for RawIrsParams {}

// SAFETY: the record uniquely owns a host-memory infos structure with no thread affinity.
unsafe impl Send for RawIrsInfos {}

impl IrsParams {
    /// Creates IRS solver parameters, with [`IrsRefinement::None`] selected and both precisions unset.
    ///
    /// One value can configure any number of [`xgesv_async`] and [`xgels_async`] solves, and may be changed between solves once the previous one has completed.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER cannot allocate the parameters or returns a null handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: the call writes a new handle to the local `handle`.
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsCreate(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let mut params = Self {
            handle,
            main_precision: None,
            lowest_precision: None,
        };
        params.set_refinement_solver(IrsRefinement::None)?;
        Ok(params)
    }

    /// Selects the refinement solver for later IRS solves.
    ///
    /// [`IrsRefinement::None`] factorizes in the lowest precision and solves once without refinement.
    /// [`IrsRefinement::Classical`] and [`IrsRefinement::Gmres`] refine in one level, while [`IrsRefinement::ClassicalGmres`] and [`IrsRefinement::GmresGmres`] run an inner GMRES solve at each outer iteration.
    /// cuSOLVER recommends [`IrsRefinement::Gmres`].
    /// A solve with [`IrsRefinement::NotSet`] fails.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the value.
    pub fn set_refinement_solver(&mut self, refinement: IrsRefinement) -> Result<()> {
        // SAFETY: `self` owns the live handle and is borrowed mutably, so no solve or getter uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetRefinementSolver(
                self.as_raw(),
                refinement.into(),
            ))?;
        }
        Ok(())
    }

    /// Sets the main precision, the precision of the solve's input and output data.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the value.
    pub fn set_main_precision(&mut self, precision: PrecisionType) -> Result<()> {
        // SAFETY: `self` owns the live handle and is borrowed mutably, so no solve or getter uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetSolverMainPrecision(
                self.as_raw(),
                precision.into(),
            ))?;
        }

        self.main_precision = Some(precision);
        Ok(())
    }

    /// Sets the lowest precision the solver may use internally for the LU factorization.
    ///
    /// The ratio between the throughput of the lowest and the main precision bounds the speedup for large matrices, such as about 2x for [`PrecisionType::R32F`] under [`PrecisionType::R64F`].
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the value.
    pub fn set_lowest_precision(&mut self, precision: PrecisionType) -> Result<()> {
        // SAFETY: `self` owns the live handle and is borrowed mutably, so no solve or getter uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetSolverLowestPrecision(
                self.as_raw(),
                precision.into(),
            ))?;
        }

        self.lowest_precision = Some(precision);
        Ok(())
    }

    /// Sets the main and lowest precisions together.
    ///
    /// The lowest precision is the main precision or a lower one of the same domain: down to [`PrecisionType::C16F`], [`PrecisionType::C16Bf`], or [`PrecisionType::CTf32`] for complex data, and to [`PrecisionType::R16F`], [`PrecisionType::R16Bf`], or [`PrecisionType::RTf32`] for real data.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the combination.
    pub fn set_solver_precisions(
        &mut self,
        main_precision: PrecisionType,
        lowest_precision: PrecisionType,
    ) -> Result<()> {
        // SAFETY: `self` owns the live handle and is borrowed mutably, so no solve or getter uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetSolverPrecisions(
                self.as_raw(),
                main_precision.into(),
                lowest_precision.into(),
            ))?;
        }

        self.main_precision = Some(main_precision);
        self.lowest_precision = Some(lowest_precision);
        Ok(())
    }

    /// Sets the residual tolerance at which refinement stops.
    ///
    /// The default stops when every right-hand side satisfies `‖r‖∞ < sqrt(n) * ‖x‖∞ * ‖A‖∞ * eps`, with `eps` the machine epsilon of the main precision.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the value.
    pub fn set_tolerance(&mut self, tolerance: f64) -> Result<()> {
        // SAFETY: `self` owns the live handle and is borrowed mutably, so no solve or getter uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetTol(self.as_raw(), tolerance))?;
        }
        Ok(())
    }

    /// Sets the tolerance of the inner GMRES solve of a two-level refinement solver, 1e-4 by default.
    ///
    /// One-level refinement solvers ignore it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the value.
    pub fn set_inner_tolerance(&mut self, tolerance: f64) -> Result<()> {
        // SAFETY: `self` owns the live handle and is borrowed mutably, so no solve or getter uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetTolInner(
                self.as_raw(),
                tolerance,
            ))?;
        }
        Ok(())
    }

    /// Sets the maximum total of outer and inner refinement iterations, 50 by default.
    ///
    /// The inner limit is lowered to this value if it is larger.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the value.
    pub fn set_max_iterations(&mut self, max_iterations: i32) -> Result<()> {
        // SAFETY: `self` owns the live handle and is borrowed mutably, so no solve or getter uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetMaxIters(
                self.as_raw(),
                max_iterations,
            ))?;
        }
        Ok(())
    }

    /// Sets the maximum number of inner iterations per outer iteration of a two-level refinement solver, 50 by default.
    ///
    /// One-level refinement solvers ignore it.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_iterations` exceeds the total limit, or if cuSOLVER rejects the value.
    pub fn set_max_inner_iterations(&mut self, max_iterations: i32) -> Result<()> {
        // SAFETY: `self` owns the live handle and is borrowed mutably, so no solve or getter uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetMaxItersInner(
                self.as_raw(),
                max_iterations,
            ))?;
        }
        Ok(())
    }

    /// Returns the current limit on total refinement iterations.
    ///
    /// [`IrsInfos::max_iterations`] returns the limit that applied to one finished solve.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameters.
    pub fn max_iterations(&self) -> Result<i32> {
        let mut value = 0;
        // SAFETY: `self` owns the live handle, and the call writes one `i32` to the local `value`.
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsGetMaxIters(
                self.as_raw(),
                &raw mut value,
            ))?;
        }
        Ok(value)
    }

    /// Selects what a solve does when refinement does not converge, which it reports with a negative iteration count.
    ///
    /// [`IrsFallback::MainPrecision`], the default, solves again from scratch in the main precision.
    /// [`IrsFallback::Disabled`] keeps the unconverged solution.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameters.
    pub fn set_fallback(&mut self, fallback: IrsFallback) -> Result<()> {
        // SAFETY: `self` owns the live handle and is borrowed mutably, so no solve or getter uses it concurrently.
        unsafe {
            match fallback {
                IrsFallback::MainPrecision => {
                    try_ffi!(sys::cusolverDnIRSParamsEnableFallback(self.as_raw()))?;
                }
                IrsFallback::Disabled => {
                    try_ffi!(sys::cusolverDnIRSParamsDisableFallback(self.as_raw()))?;
                }
            }
        }
        Ok(())
    }

    fn ensure_type_precision<T: DataTypeLike>(&mut self) -> Result<()> {
        let precision = PrecisionType::from_data_type(T::data_type())
            .ok_or(Error::InvalidPrecisionConfiguration)?;
        match self.main_precision {
            Some(existing) if existing != precision => {
                return Err(Error::InvalidPrecisionConfiguration);
            }
            None => self.set_main_precision(precision)?,
            _ => {}
        }

        if self.lowest_precision.is_none() {
            self.set_lowest_precision(precision)?;
        }
        Ok(())
    }

    /// Returns the native parameter handle, which stays owned by this value.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusolverDnIRSParams_t {
        self.handle
    }

    /// Adopts a native IRS parameter handle and the precisions configured on it.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a live IRS parameter handle that no other owner destroys or uses afterwards.
    /// `raw.main_precision` and `raw.lowest_precision` must each be `None` or the precision configured on the handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the raw value transfers ownership of its handle even though its fields are `Copy`"
    )]
    pub unsafe fn from_raw(raw: RawIrsParams) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: raw.handle,
            main_precision: raw.main_precision,
            lowest_precision: raw.lowest_precision,
        })
    }

    /// Releases the native handle and its configured precisions without destroying it.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> RawIrsParams {
        let raw = RawIrsParams {
            handle: self.handle,
            main_precision: self.main_precision,
            lowest_precision: self.lowest_precision,
        };
        mem::forget(self);
        raw
    }
}

impl Drop for IrsParams {
    fn drop(&mut self) {
        // SAFETY: `self` uniquely owns the live handle, and the raw execution contract keeps it alive until queued work that uses it completes.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusolverDnIRSParamsDestroy(self.handle)) {
                report_cleanup_failure("destroy", "cusolver irs params", &error);
            }
        }
    }
}

impl IrsInfos {
    /// Creates the report for one IRS solve.
    ///
    /// A report records the iteration counts, the iteration limit, and the optional residual history of the single solve it is passed to; each solve needs its own.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER cannot allocate the report or returns a null handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: the call writes a new handle to the local `handle`.
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosCreate(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle,
            residual_history_requested: false,
            main_precision: None,
        })
    }

    /// Returns the total number of refinement iterations of the finished solve.
    ///
    /// A negative count means refinement did not converge, and with fallback enabled the solve was repeated in the main precision.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the report.
    pub fn iterations(&self) -> Result<i32> {
        let mut value = 0;
        // SAFETY: `self` owns the live handle, and the call writes one `i32` to the local `value`.
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosGetNiters(
                self.as_raw(),
                &raw mut value,
            ))?;
        }
        Ok(value)
    }

    /// Returns the number of outer refinement iterations of the finished solve.
    ///
    /// It equals [`Self::iterations`] for one-level refinement solvers.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the report.
    pub fn outer_iterations(&self) -> Result<i32> {
        let mut value = 0;
        // SAFETY: `self` owns the live handle, and the call writes one `i32` to the local `value`.
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosGetOuterNiters(
                self.as_raw(),
                &raw mut value,
            ))?;
        }
        Ok(value)
    }

    /// Returns the iteration limit that applied to the finished solve.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the report.
    pub fn max_iterations(&self) -> Result<i32> {
        let mut value = 0;
        // SAFETY: `self` owns the live handle, and the call writes one `i32` to the local `value`.
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosGetMaxIters(
                self.as_raw(),
                &raw mut value,
            ))?;
        }
        Ok(value)
    }

    /// Asks the next solve to record its residual history, which [`Self::residual_history_f32`] and [`Self::residual_history_f64`] return.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the report.
    pub fn request_residual_history(&mut self) -> Result<()> {
        // SAFETY: `self` owns the live handle and is borrowed mutably, so no solve or getter uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosRequestResidual(self.as_raw()))?;
        }

        self.residual_history_requested = true;
        Ok(())
    }

    /// Copies the residual history of a finished single-precision solve, real or complex.
    ///
    /// Row zero holds the initial residual, and row `i` the cumulative iteration count and residual norm after outer iteration `i`.
    ///
    /// # Errors
    ///
    /// Returns an error if residual history was not requested before the solve, the solve was not single-precision, cuSOLVER rejects the report, or the reported history is null or inconsistent.
    pub fn residual_history_f32(&self) -> Result<ResidualHistory<f32>> {
        self.require_history_precision(PrecisionType::R32F, PrecisionType::C32F)?;
        let (leading_dimension, valid_rows) = self.residual_history_layout()?;
        let mut history = ptr::null_mut();
        // SAFETY: `self` owns the live handle, and the call writes the history address to the local `history`.
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosGetResidualHistory(
                self.as_raw(),
                &raw mut history,
            ))?;
        }

        if history.is_null() {
            return Err(Error::InvalidResidualHistory);
        }

        // SAFETY: the native history matrix has two columns of `leading_dimension` elements whose first `valid_rows` rows hold the outer iterations, as `residual_history_layout` read, and `self` keeps it alive during the copy.
        Ok(unsafe { copy_residual_history(history.cast::<f32>(), leading_dimension, valid_rows) })
    }

    /// Copies the residual history of a finished double-precision solve, real or complex.
    ///
    /// Row zero holds the initial residual, and row `i` the cumulative iteration count and residual norm after outer iteration `i`.
    ///
    /// # Errors
    ///
    /// Returns an error if residual history was not requested before the solve, the solve was not double-precision, cuSOLVER rejects the report, or the reported history is null or inconsistent.
    pub fn residual_history_f64(&self) -> Result<ResidualHistory<f64>> {
        self.require_history_precision(PrecisionType::R64F, PrecisionType::C64F)?;
        let (leading_dimension, valid_rows) = self.residual_history_layout()?;
        let mut history = ptr::null_mut();
        // SAFETY: `self` owns the live handle, and the call writes the history address to the local `history`.
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosGetResidualHistory(
                self.as_raw(),
                &raw mut history,
            ))?;
        }

        if history.is_null() {
            return Err(Error::InvalidResidualHistory);
        }

        // SAFETY: the native history matrix has two columns of `leading_dimension` elements whose first `valid_rows` rows hold the outer iterations, as `residual_history_layout` read, and `self` keeps it alive during the copy.
        Ok(unsafe { copy_residual_history(history.cast::<f64>(), leading_dimension, valid_rows) })
    }

    /// Returns the native report handle, which stays owned by this value.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusolverDnIRSInfos_t {
        self.handle
    }

    /// Adopts a native IRS report handle and its residual-history request state.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a live IRS report handle that no other owner destroys or uses afterwards.
    /// `raw.residual_history_requested` may be `true` only if residual history was requested on the handle.
    /// `raw.main_precision` must be `None` or the main precision of the last solve that used the handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the raw value transfers ownership of its handle even though its fields are `Copy`"
    )]
    pub unsafe fn from_raw(raw: RawIrsInfos) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: raw.handle,
            residual_history_requested: raw.residual_history_requested,
            main_precision: raw.main_precision,
        })
    }

    /// Releases the native handle and its residual-history request state without destroying it.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> RawIrsInfos {
        let raw = RawIrsInfos {
            handle: self.handle,
            residual_history_requested: self.residual_history_requested,
            main_precision: self.main_precision,
        };
        mem::forget(self);
        raw
    }

    fn require_history_precision(&self, real: PrecisionType, complex: PrecisionType) -> Result<()> {
        if !self.residual_history_requested {
            return Err(Error::ResidualHistoryNotRequested);
        }

        match self.main_precision {
            Some(precision) if precision == real || precision == complex => Ok(()),
            _ => Err(Error::InvalidPrecisionConfiguration),
        }
    }

    fn residual_history_layout(&self) -> Result<(usize, usize)> {
        let leading_dimension = self
            .max_iterations()?
            .checked_add(1)
            .ok_or(Error::InvalidResidualHistory)
            .and_then(|value| to_usize(value, "residual history leading dimension"))?;
        let valid_rows = self
            .outer_iterations()?
            .checked_add(1)
            .ok_or(Error::InvalidResidualHistory)
            .and_then(|value| to_usize(value, "residual history rows"))?;

        if valid_rows > leading_dimension {
            return Err(Error::InvalidResidualHistory);
        }

        Ok((leading_dimension, valid_rows))
    }
}

impl Drop for IrsInfos {
    fn drop(&mut self) {
        // SAFETY: `self` uniquely owns the live handle, and the raw execution contract keeps it alive until queued work that uses it completes.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusolverDnIRSInfosDestroy(self.handle)) {
                report_cleanup_failure("destroy", "cusolver irs infos", &error);
            }
        }
    }
}

/// Returns the device workspace bytes that [`xgesv_async`] needs for an `n` by `n` system with `nrhs` right-hand sides under the current `params`.
///
/// Fills any unset precision selectors in `params` from `T`, so query again after changing `params`.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a dimension is zero or does not fit the native integer types, `T` has no IRS precision or differs from the configured main precision, or cuSOLVER rejects the configuration.
pub fn xgesv_workspace_size<T: DataTypeLike>(
    context: &Context,
    params: &mut IrsParams,
    n: usize,
    nrhs: usize,
) -> Result<usize> {
    context.bind()?;
    if n == 0 || nrhs == 0 {
        return Err(Error::InvalidMatrixShape);
    }

    params.ensure_type_precision::<T>()?;
    let mut workspace_bytes = 0;
    // SAFETY: the context and parameter handles are live and the context is bound, the dimensions were checked above, and the call writes only the local `workspace_bytes`.
    unsafe {
        try_ffi!(sys::cusolverDnIRSXgesv_bufferSize(
            context.as_raw(),
            params.as_raw(),
            to_i32(n, "n")?,
            to_i32(nrhs, "nrhs")?,
            &raw mut workspace_bytes,
        ))?;
    }
    to_usize(workspace_bytes, "workspace size")
}

/// Enqueues the iterative-refinement solve of the square system `A * X = B` and returns the native iteration count.
///
/// `params` selects the refinement solver, tolerances, iteration limits, fallback, and the lowest internal precision; unset precisions are filled from `T`.
/// A negative iteration count means refinement did not converge; [`IrsFallback`] decides what happens then.
/// `infos` receives the iteration counts and, if requested, the residual history.
/// `a` may be overwritten; on completion with a zero status, `x` holds `n` rows per right-hand-side column, and only the first `dev_info` element is written.
/// Size `device_workspace` with [`xgesv_workspace_size`] after the last change to `params`.
/// The lowest precision can be the main precision or a lower one of the same domain, such as [`PrecisionType::R16F`], [`PrecisionType::R16Bf`], or [`PrecisionType::RTf32`] for an [`PrecisionType::R32F`] system.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution), which also covers `params` and `infos`: neither may be inspected or changed while queued work may use them.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, `T` does not match the configured main precision, the workspace is smaller than the queried size, or cuSOLVER rejects the configuration or reports a failed solve.
pub unsafe fn xgesv_async<T: DataTypeLike>(
    context: &Context,
    params: &mut IrsParams,
    infos: &mut IrsInfos,
    n: usize,
    nrhs: usize,
    mut a: MatrixMut<'_, T>,
    b: MatrixRef<'_, T>,
    mut x: MatrixOutput<'_, T>,
    device_workspace: &mut impl DeviceOutput<u8>,
    dev_info: &mut impl DeviceOutput<i32>,
) -> Result<i32> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&b.data)?;
    context.prepare_region(&x.data)?;
    context.prepare_region(&*device_workspace)?;
    context.prepare_region(&*dev_info)?;
    validate_matrix(n, n, a.data.len(), a.leading_dimension)?;
    validate_matrix(n, nrhs, b.data.len(), b.leading_dimension)?;
    validate_matrix(n, nrhs, x.data.len(), x.leading_dimension)?;
    require_info_buffer(dev_info)?;
    let workspace_bytes = xgesv_workspace_size::<T>(context, params, n, nrhs)?;
    require_workspace_bytes(
        "device workspace",
        device_workspace.region().byte_len(),
        workspace_bytes,
    )?;
    let mut niters = 0;
    infos.main_precision = params.main_precision;
    // SAFETY: the handles are live and the context is bound, every buffer belongs to the context and was checked above against the dimensions and the queried workspace bytes, and `niters` is a writable local.
    // The caller retains the buffers and IRS handles through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnIRSXgesv(
            context.as_raw(),
            params.as_raw(),
            infos.as_raw(),
            to_i32(n, "n")?,
            to_i32(nrhs, "nrhs")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "ldda")?,
            b.data.as_ptr().cast_mut().cast(),
            to_i32(b.leading_dimension, "lddb")?,
            x.data.as_mut_ptr().cast(),
            to_i32(x.leading_dimension, "lddx")?,
            device_workspace.as_mut_ptr().cast(),
            to_u64(workspace_bytes, "lwork_bytes")?,
            &raw mut niters,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(niters)
}

/// Returns the device workspace bytes that [`xgels_async`] needs for an `m` by `n` system with `nrhs` right-hand sides under the current `params`.
///
/// Fills any unset precision selectors in `params` from `T`, so query again after changing `params`.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a dimension is zero or does not fit the native integer types, `n` exceeds `m`, `T` has no IRS precision or differs from the configured main precision, or cuSOLVER rejects the configuration.
pub fn xgels_workspace_size<T: DataTypeLike>(
    context: &Context,
    params: &mut IrsParams,
    m: usize,
    n: usize,
    nrhs: usize,
) -> Result<usize> {
    context.bind()?;
    if m == 0 || n == 0 || nrhs == 0 || n > m {
        return Err(Error::InvalidMatrixShape);
    }

    params.ensure_type_precision::<T>()?;
    let mut workspace_bytes = 0;
    // SAFETY: the context and parameter handles are live and the context is bound, the dimensions were checked above, and the call writes only the local `workspace_bytes`.
    unsafe {
        try_ffi!(sys::cusolverDnIRSXgels_bufferSize(
            context.as_raw(),
            params.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(nrhs, "nrhs")?,
            &raw mut workspace_bytes,
        ))?;
    }
    to_usize(workspace_bytes, "workspace size")
}

/// Enqueues the iterative-refinement least-squares solve of the overdetermined system `A * X = B` and returns the native iteration count.
///
/// `a` has `m` rows and `n <= m` columns, and `b` has `m` rows per right-hand side.
/// `params` selects the refinement solver, tolerances, iteration limits, fallback, and the lowest internal precision; unset precisions are filled from `T`.
/// A negative iteration count means refinement did not converge; [`IrsFallback`] decides what happens then.
/// `infos` receives the iteration counts and, if requested, the residual history.
/// `a` may be overwritten; on completion with a zero status, `x` holds `n` rows per right-hand-side column, and only the first `dev_info` element is written.
/// Size `device_workspace` with [`xgels_workspace_size`] after the last change to `params`.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution), which also covers `params` and `infos`: neither may be inspected or changed while queued work may use them.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, `n` exceeds `m`, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, `T` does not match the configured main precision, the workspace is smaller than the queried size, or cuSOLVER rejects the configuration or reports a failed solve.
pub unsafe fn xgels_async<T: DataTypeLike>(
    context: &Context,
    params: &mut IrsParams,
    infos: &mut IrsInfos,
    m: usize,
    n: usize,
    nrhs: usize,
    mut a: MatrixMut<'_, T>,
    b: MatrixRef<'_, T>,
    mut x: MatrixOutput<'_, T>,
    device_workspace: &mut impl DeviceOutput<u8>,
    dev_info: &mut impl DeviceOutput<i32>,
) -> Result<i32> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&b.data)?;
    context.prepare_region(&x.data)?;
    context.prepare_region(&*device_workspace)?;
    context.prepare_region(&*dev_info)?;
    if n > m {
        return Err(Error::InvalidMatrixShape);
    }

    validate_matrix(m, n, a.data.len(), a.leading_dimension)?;
    validate_matrix(m, nrhs, b.data.len(), b.leading_dimension)?;
    validate_matrix(n, nrhs, x.data.len(), x.leading_dimension)?;
    require_info_buffer(dev_info)?;
    let workspace_bytes = xgels_workspace_size::<T>(context, params, m, n, nrhs)?;
    require_workspace_bytes(
        "device workspace",
        device_workspace.region().byte_len(),
        workspace_bytes,
    )?;
    let mut niters = 0;
    infos.main_precision = params.main_precision;
    // SAFETY: the handles are live and the context is bound, every buffer belongs to the context and was checked above against the dimensions and the queried workspace bytes, and `niters` is a writable local.
    // The caller retains the buffers and IRS handles through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnIRSXgels(
            context.as_raw(),
            params.as_raw(),
            infos.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(nrhs, "nrhs")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "ldda")?,
            b.data.as_ptr().cast_mut().cast(),
            to_i32(b.leading_dimension, "lddb")?,
            x.data.as_mut_ptr().cast(),
            to_i32(x.leading_dimension, "lddx")?,
            device_workspace.as_mut_ptr().cast(),
            to_u64(workspace_bytes, "lwork_bytes")?,
            &raw mut niters,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(niters)
}

fn require_info_buffer(dev_info: &impl DeviceOutput<i32>) -> Result<()> {
    if dev_info.is_empty() {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

// `history` must address two columns of `leading_dimension` elements whose first `valid_rows` rows are initialized and stay alive during the call.
// Only those rows are read, because cuSOLVER leaves the later rows undefined.
unsafe fn copy_residual_history<T: Copy>(
    history: *const T,
    leading_dimension: usize,
    valid_rows: usize,
) -> ResidualHistory<T> {
    let mut rows = Vec::with_capacity(valid_rows);
    for row in 0..valid_rows {
        // SAFETY: `row < valid_rows <= leading_dimension`, so both reads address initialized rows of the caller's two columns.
        let (total_iterations, residual_norm) = unsafe {
            (
                history.add(row).read(),
                history.add(row + leading_dimension).read(),
            )
        };
        rows.push(ResidualHistoryEntry {
            total_iterations,
            residual_norm,
        });
    }
    ResidualHistory {
        rows,
        leading_dimension,
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::{memory::DeviceMemory, stream::StreamOptions};

    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_xgesv_solves_diagonal_system() -> Result<()> {
        let mut test = setup_context()?;
        let context = test.context_mut();
        let mut params = IrsParams::create()?;
        let mut infos = IrsInfos::create()?;

        let mut a = DeviceMemory::from_slice(
            context.cuda_context(),
            &[
                2.0_f32, 0.0, //
                0.0, 4.0,
            ],
        )?;
        let b = DeviceMemory::from_slice(
            context.cuda_context(),
            &[
                6.0_f32, //
                8.0,
            ],
        )?;
        let mut x = DeviceMemory::<f32>::create_uninit(context.cuda_context(), 2)?;
        let workspace_bytes = xgesv_workspace_size::<f32>(context, &mut params, 2, 1)?;
        let mut workspace =
            DeviceMemory::<u8>::create_uninit(context.cuda_context(), workspace_bytes)?;
        let mut dev_info = DeviceMemory::<i32>::create_uninit(context.cuda_context(), 1)?;

        let stream = context
            .cuda_context()
            .create_stream(StreamOptions::default())?;
        context.set_stream(Some(&stream))?;
        IrsSolve::new(2, 1).execute(
            context,
            IrsSolveBindings {
                a: MatrixMut::new(&mut a, 2),
                b: MatrixRef::new(&b, 2),
                x: MatrixOutput::new(&mut x, 2),
                device_workspace: workspace.output(),
                dev_info: dev_info.output(),
            },
            &mut params,
            &mut infos,
        )?;

        // SAFETY: successful completion writes the first status element.
        let dev_info = unsafe { dev_info.assume_init() };
        assert_eq!(dev_info.copy_to_host_vec()?, vec![0]);
        // SAFETY: status is zero and the compact 2x1 solution has been produced.
        let x = unsafe { x.assume_init() };
        assert_eq!(x.copy_to_host_vec()?, vec![3.0, 2.0]);
        Ok(())
    }
}
