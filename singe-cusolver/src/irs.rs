use std::{ptr, slice};

use singe_cuda::{data_type::DataTypeLike, memory::DeviceMemory};

use crate::{
    context::Context,
    error::{Error, Result},
    layout::{MatrixMut, MatrixRef},
    sys, try_ffi,
    types::{IrsRefinement, PrecisionType},
    utility::{to_i32, to_u64, to_usize},
};

#[derive(Debug)]
pub struct IrsParams {
    handle: sys::cusolverDnIRSParams_t,
    main_precision: Option<PrecisionType>,
    lowest_precision: Option<PrecisionType>,
}

#[derive(Debug, Default)]
pub struct IrsInfos {
    handle: sys::cusolverDnIRSInfos_t,
    residual_history_requested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualHistoryEntry<T> {
    pub total_iterations: T,
    pub residual_norm: T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidualHistory<T> {
    pub rows: Vec<ResidualHistoryEntry<T>>,
    pub leading_dimension: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrsSolve {
    pub n: usize,
    pub right_hand_sides: usize,
}

impl IrsSolve {
    pub fn new(n: usize, right_hand_sides: usize) -> Self {
        Self {
            n,
            right_hand_sides,
        }
    }

    pub fn workspace_size<T: DataTypeLike>(
        self,
        ctx: &Context,
        params: &mut IrsParams,
    ) -> Result<usize> {
        xgesv_buffer_size::<T>(ctx, params, self.n, self.right_hand_sides)
    }

    pub fn execute<T: DataTypeLike>(
        self,
        ctx: &Context,
        params: &mut IrsParams,
        infos: &IrsInfos,
        bindings: IrsSolveBindings<'_, T>,
    ) -> Result<i32> {
        xgesv(
            ctx,
            params,
            infos,
            self.n,
            self.right_hand_sides,
            bindings.a,
            bindings.b,
            bindings.x,
            bindings.device_workspace,
            bindings.dev_info,
        )
    }
}

#[derive(Debug)]
pub struct IrsSolveBindings<'a, T> {
    pub a: MatrixMut<'a, T>,
    pub b: MatrixRef<'a, T>,
    pub x: MatrixMut<'a, T>,
    pub device_workspace: &'a mut DeviceMemory<u8>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

// IRS parameter/info handles expose mutation through &mut self and inspection
// through shared references, so immutable sharing follows the cuSOLVER contract.
unsafe impl Send for IrsParams {}
unsafe impl Sync for IrsParams {}
unsafe impl Send for IrsInfos {}
unsafe impl Sync for IrsInfos {}

impl IrsParams {
    /// Creates and initializes the parameter structure for IRS solvers such as
    /// [`xgesv`] and [`xgels`].
    ///
    /// The returned parameter structure can be reused across calls to the same
    /// IRS solver or to different IRS solvers.
    ///
    /// In CUDA 10.2, the behavior was different and a new parameter structure
    /// was required for each IRS solve call.
    ///
    /// You can also reconfigure the parameters between solves, but only after
    /// the previous IRS call has completed.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER cannot allocate the required resources
    /// or does not return a valid handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
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

    /// Sets the refinement solver used by IRS operations such as [`xgesv`] and
    /// [`xgels`].
    ///
    /// Configure the refinement algorithm before the first IRS solve. Newly created [`IrsParams`] do not set one by default.
    ///
    /// The supported values are described below.
    ///
    /// [`IrsRefinement::NotSet`]: Solver is not set. The IRS solver returns an
    /// error if this value is used.
    ///
    /// [`IrsRefinement::None`]: No refinement solver; the IRS solver performs a factorization followed by a solve without any refinement.
    /// For example, if the IRS solver was [`xgesv`], this is equivalent to an
    /// [`xgesv`] solve without refinement, with the factorization carried out in
    /// the lowest configured precision.
    /// If both the main and lowest precision are [`PrecisionType::R64F`], the
    /// solve is effectively performed in `f64`.
    ///
    /// [`IrsRefinement::Classical`]: Classical iterative refinement solver.
    /// Similar to the value used in LAPACK operations.
    ///
    /// [`IrsRefinement::Gmres`]: GMRES (Generalized Minimal Residual) based iterative refinement solver.
    /// Recent studies use GMRES as a refinement solver that can outperform
    /// classical iterative refinement.
    /// Recommended setting based on cuSOLVER experimentation.
    ///
    /// [`IrsRefinement::ClassicalGmres`]: Classical iterative refinement solver that uses the GMRES (Generalized Minimal Residual) internally to solve the correction equation at each iteration.
    /// The classical refinement iteration is the outer iteration, and GMRES is
    /// the inner iteration.
    /// If the tolerance of the inner GMRES is set very low, for
    /// example near machine precision, then the outer *classical refinement
    /// iteration* performs only one iteration and this option behaves like
    /// [`IrsRefinement::Gmres`].
    ///
    /// [`IrsRefinement::GmresGmres`]: GMRES-based iterative refinement solver
    /// that uses another GMRES solve internally for the preconditioned system.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameter structure.
    pub fn set_refinement_solver(&mut self, refinement: IrsRefinement) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetRefinementSolver(
                self.as_raw(),
                refinement.into(),
            ))?;
        }
        Ok(())
    }

    /// Sets the main precision for the Iterative Refinement Solver (IRS).
    ///
    /// The main precision is the type of the input and output data.
    /// Configure both the main and lowest precision before the first IRS solve. Those
    /// values are not inferred when the parameter structure is created because
    /// they depend on the input/output data type and the requested solver
    /// configuration. You can set them independently or together with
    /// [`IrsParams::set_solver_precisions`].
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameter structure.
    pub fn set_main_precision(&mut self, precision: PrecisionType) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetSolverMainPrecision(
                self.as_raw(),
                precision.into(),
            ))?;
        }
        self.main_precision = Some(precision);
        Ok(())
    }

    /// Sets the lowest precision that the IRS solver may use.
    ///
    /// The lowest precision is the minimum compute precision used
    /// during the LU factorization process.
    ///
    /// Configure both the main and lowest precision before the first IRS solve. They
    /// are not inferred when creating the parameter structure because they
    /// depend on the input and output data types and the requested solver
    /// configuration.
    /// Usually the lowest precision defines the speedup that can be achieved.
    /// The ratio between the performance of the lowest precision and the main
    /// precision gives an approximate upper bound on the speedup.
    /// More precisely, it depends on many factors, but for large matrices it is
    /// often tied to the performance ratio of large GEMM-like kernels.
    /// For instance, if the input/output precision is real double precision
    /// [`PrecisionType::R64F`] and the lowest precision is
    /// [`PrecisionType::R32F`], then a speedup of at most about 2x is expected
    /// for large problem sizes.
    /// If the lowest precision is [`PrecisionType::R16F`], expect 3x-4x.
    /// A reasonable strategy accounts for the number of right-hand sides, the matrix size, and the convergence rate.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameter structure.
    pub fn set_lowest_precision(&mut self, precision: PrecisionType) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetSolverLowestPrecision(
                self.as_raw(),
                precision.into(),
            ))?;
        }
        self.lowest_precision = Some(precision);
        Ok(())
    }

    /// Sets both the main and lowest precision for the Iterative Refinement
    /// Solver (IRS).
    ///
    /// The main precision is the precision of the input and output data.
    /// The lowest precision is the minimum compute precision used
    /// during the LU factorization process.
    ///
    /// Configure both values before the first IRS solve. They are not inferred when
    /// creating the parameter structure because they depend on the input and
    /// output data types and the requested solver configuration.
    ///
    /// Convenience wrapper around
    /// [`IrsParams::set_main_precision`] and
    /// [`IrsParams::set_lowest_precision`].
    /// All possible combinations of main/lowest precision are described in the table below.
    /// Usually the lowest precision defines the speedup that can be achieved.
    /// The ratio between the performance of the lowest precision and the main
    /// precision gives an approximate upper bound on the speedup.
    /// More precisely, it depends on many factors, but for large matrices it is
    /// often tied to the performance ratio of large GEMM-like kernels.
    /// For instance, if the input/output precision is real double precision
    /// [`PrecisionType::R64F`] and the lowest precision is
    /// [`PrecisionType::R32F`], then a speedup of at most about 2x is expected
    /// for large problem sizes.
    /// If the lowest precision is [`PrecisionType::R16F`], expect 3x-4x.
    /// A reasonable strategy accounts for the number of right-hand sides, the matrix size, and the convergence rate.
    ///
    /// **Supported input/output data type and lower precision for the IRS solver**
    ///
    /// | **input/output Data Type (for example, main precision)** | **Supported values for the lowest precision** |
    /// | --- | --- |
    /// | [`PrecisionType::C64F`] | [`PrecisionType::C64F`], [`PrecisionType::C32F`], [`PrecisionType::C16F`], [`PrecisionType::C16Bf`], [`PrecisionType::CTf32`] |
    /// | [`PrecisionType::C32F`] | [`PrecisionType::C32F`], [`PrecisionType::C16F`], [`PrecisionType::C16Bf`], [`PrecisionType::CTf32`] |
    /// | [`PrecisionType::R64F`] | [`PrecisionType::R64F`], [`PrecisionType::R32F`], [`PrecisionType::R16F`], [`PrecisionType::R16Bf`], [`PrecisionType::RTf32`] |
    /// | [`PrecisionType::R32F`] | [`PrecisionType::R32F`], [`PrecisionType::R16F`], [`PrecisionType::R16Bf`], [`PrecisionType::RTf32`] |
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameter structure.
    pub fn set_solver_precisions(
        &mut self,
        main_precision: PrecisionType,
        lowest_precision: PrecisionType,
    ) -> Result<()> {
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

    /// Sets the tolerance for the refinement solver.
    /// By default it is such that all the RHS satisfy:
    ///
    /// `RNRM &lt; SQRT(N)*XNRM*ANRM*EPS*BWDMAX` where
    ///
    /// * RNRM is the infinity-norm of the residual
    /// * XNRM is the infinity-norm of the solution
    /// * ANRM is the infinity-operator-norm of the matrix A
    /// * EPS is the machine epsilon for the input/output data type that matches
    ///   LAPACK `xLAMCH('Epsilon')`
    /// * BWDMAX, the value BWDMAX is fixed to 1.0
    ///
    /// Use this to set the tolerance to a lower or higher value.
    /// The tolerance value is always stored in real double precision,
    /// regardless of the input and output data type.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameter structure.
    pub fn set_tolerance(&mut self, tolerance: f64) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetTol(self.as_raw(), tolerance))?;
        }
        Ok(())
    }

    /// Sets the tolerance for the inner refinement solver when
    /// the refinement solver consists of two levels, for example
    /// [`IrsRefinement::ClassicalGmres`] or [`IrsRefinement::GmresGmres`].
    /// Ignored for one-level refinement solvers such as [`IrsRefinement::Classical`] or [`IrsRefinement::Gmres`].
    /// The default value is 1e-4.
    /// This sets the tolerance for the inner solver, such as the inner GMRES.
    /// For example, if the refinement solver is
    /// [`IrsRefinement::ClassicalGmres`], setting this tolerance means that the
    /// inner GMRES solver converges to that tolerance at each outer
    /// iteration of the classical refinement solver.
    /// The tolerance value is always stored in real double precision,
    /// regardless of the input and output data type.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameter structure.
    pub fn set_inner_tolerance(&mut self, tolerance: f64) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetTolInner(
                self.as_raw(),
                tolerance,
            ))?;
        }
        Ok(())
    }

    /// Sets the total number of allowed refinement iterations before the solver stops.
    /// The total is the sum of the outer and inner iterations. Inner iterations are meaningful when a two-level refinement solver is configured.
    /// The default value is 50.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameter structure.
    pub fn set_max_iterations(&mut self, max_iterations: i32) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetMaxIters(
                self.as_raw(),
                max_iterations,
            ))?;
        }
        Ok(())
    }

    /// Sets the maximum number of iterations allowed for the inner refinement solver.
    /// Ignored for one-level refinement solvers such as [`IrsRefinement::Classical`] or [`IrsRefinement::Gmres`].
    /// The inner refinement solver stops after reaching either the inner tolerance or `MaxItersInner`.
    /// The default value is 50.
    /// Cannot be larger than `MaxIters` because `MaxIters` is the total number of allowed iterations.
    /// If [`IrsParams::set_max_iterations`] is called after this method, it has priority and overwrites `MaxItersInner` with `min(MaxIters, MaxItersInner)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_iterations` is larger than `MaxIters`, or if
    /// cuSOLVER rejects the parameter structure.
    pub fn set_max_inner_iterations(&mut self, max_iterations: i32) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsSetMaxItersInner(
                self.as_raw(),
                max_iterations,
            ))?;
        }
        Ok(())
    }

    /// Returns the current maximum-iteration setting in this parameter structure.
    /// Current parameter configuration, distinct from [`IrsInfos::max_iterations`], which returns the maximum number of iterations allowed for a particular IRS solver call.
    /// The parameter structure can be reused across many IRS solver calls.
    /// The allowed `MaxIters` value can change between calls, while the `Infos` structure contains information about one particular call and cannot be reused for different calls.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameter structure.
    pub fn max_iterations(&self) -> Result<i32> {
        let mut value = 0;
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsGetMaxIters(
                self.as_raw(),
                &raw mut value,
            ))?;
        }
        Ok(value)
    }

    /// Enables fallback to the main precision if the Iterative Refinement Solver (IRS) fails to converge.
    /// If the IRS solver fails to converge, it returns a non-convergence code such as `niter < 0`.
    /// With fallback disabled, it returns the non-convergent solution as-is.
    /// With fallback enabled, it falls back to the main precision, which is the input/output data precision, and solves the problem again from scratch.
    /// This fallback is the default behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameter structure.
    pub fn enable_fallback(&mut self) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsEnableFallback(self.as_raw()))?;
        }
        Ok(())
    }

    /// Disables fallback to the main precision if the Iterative Refinement Solver (IRS) fails to converge.
    /// If the IRS solver fails to converge, it returns a non-convergence code such as `niter < 0`.
    /// With fallback disabled, the returned solution is whatever the refinement solver reached before returning.
    /// Disabling fallback does not guarantee that the solution is accurate.
    /// Re-enable fallback with [`IrsParams::enable_fallback`].
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the parameter structure.
    pub fn disable_fallback(&mut self) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnIRSParamsDisableFallback(self.as_raw()))?;
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

    pub fn as_raw(&self) -> sys::cusolverDnIRSParams_t {
        self.handle
    }

    /// Takes ownership of a raw cuSOLVER IRS params handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusolverDnIRSParams_t` created by cuSOLVER.
    /// The returned wrapper takes ownership and will destroy it with
    /// `cusolverDnIRSParamsDestroy`; no other owner may destroy or keep using it.
    pub unsafe fn from_raw(handle: sys::cusolverDnIRSParams_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle,
            main_precision: None,
            lowest_precision: None,
        })
    }

    /// Releases ownership and returns the raw cuSOLVER IRS params handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::cusolverDnIRSParams_t {
        let handle = self.handle;
        std::mem::forget(self);
        handle
    }
}

impl Drop for IrsParams {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusolverDnIRSParamsDestroy(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusolver irs params: {err}");
            }
        }
    }
}

impl IrsInfos {
    /// Creates and initializes the `Infos` structure that holds refinement information for an Iterative Refinement Solver (IRS) call.
    /// Such information includes the total number of iterations needed to converge (`Niters`), the number of outer iterations (meaningful when a two-level preconditioner such as [`IrsRefinement::ClassicalGmres`] is used), the maximum number of iterations allowed for that call, and a pointer to the convergence-history residual norm matrix.
    /// Construct the `Infos` structure before calling an IRS solver.
    /// The `Infos` structure is valid for only one call to an IRS solver, since it holds information about that solve; each solve requires its own `Infos` structure.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER cannot allocate the required resources
    /// or does not return a valid handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosCreate(&raw mut handle))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle,
            residual_history_requested: false,
        })
    }

    /// Returns the total number of iterations performed by the IRS solver.
    /// If this value is negative, the IRS solver did not converge. If fallback to full precision was enabled, the solver fell back to a full-precision solution.
    /// See [`xgesv`] and [`xgels`] for the meaning of negative `niters` values.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the `Infos` structure.
    pub fn niters(&self) -> Result<i32> {
        let mut value = 0;
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosGetNiters(
                self.as_raw(),
                &raw mut value,
            ))?;
        }
        Ok(value)
    }

    /// Returns the number of iterations performed by the outer refinement loop of the IRS solver.
    /// For one-level solvers such as [`IrsRefinement::Classical`] or [`IrsRefinement::Gmres`], this is the same as `Niters`.
    /// For two-level solvers such as [`IrsRefinement::ClassicalGmres`] or [`IrsRefinement::GmresGmres`], this is the number of outer-loop iterations.
    /// See [`IrsRefinement`] for refinement mode details.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the `Infos` structure.
    pub fn outer_niters(&self) -> Result<i32> {
        let mut value = 0;
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosGetOuterNiters(
                self.as_raw(),
                &raw mut value,
            ))?;
        }
        Ok(value)
    }

    /// Returns the maximum number of iterations allowed for the corresponding IRS solver call.
    /// Setting used when that call happened, distinct from [`IrsParams::max_iterations`], which returns the current setting in the `params` configuration structure.
    /// The `params` structure can be reused for many IRS solver calls.
    /// The allowed `MaxIters` value can change between calls, while this `Infos` structure contains information about one particular call and cannot be reused for different calls.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the `Infos` structure.
    pub fn max_iterations(&self) -> Result<i32> {
        let mut value = 0;
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosGetMaxIters(
                self.as_raw(),
                &raw mut value,
            ))?;
        }
        Ok(value)
    }

    /// Asks the IRS solver to store the convergence history
    /// (residual norms) of the refinement phase so it can later be queried with
    /// [`IrsInfos::residual_history_f32`] or [`IrsInfos::residual_history_f64`].
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the `Infos` structure.
    pub fn request_residual_history(&mut self) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosRequestResidual(self.as_raw()))?;
        }
        self.residual_history_requested = true;
        Ok(())
    }

    /// Returns the convergence history stored by the IRS solver when [`IrsInfos::request_residual_history`] was called before solving.
    /// The residual norm type depends on the input and output precision.
    /// Double-precision real and complex configurations report `f64` residuals, while single-precision real and complex configurations report `f32` residuals.
    ///
    /// The residual history matrix has two columns, even for multiple right-hand sides, and `MaxIters + 1` rows.
    /// Only the first `OuterNiters + 1` rows contain residual norms; the remaining rows are undefined.
    /// In the first column, each row `i` contains the total number of iterations performed up to outer iteration `i`.
    /// In the second column, each row contains the residual norm for that outer iteration.
    /// Row 0 contains the initial residual before the refinement loop starts, and subsequent rows contain residuals obtained at each outer iteration.
    /// The history only covers the outer loop.
    ///
    /// If the refinement solver was [`IrsRefinement::Classical`] or [`IrsRefinement::Gmres`], then `OuterNiters == Niters`, and there are `Niters + 1` rows of norms corresponding to the `Niters` outer iterations.
    ///
    /// If the refinement solver was [`IrsRefinement::ClassicalGmres`] or [`IrsRefinement::GmresGmres`], then `OuterNiters <= Niters` corresponds to the outer iterations performed by the outer refinement loop.
    /// There are `OuterNiters + 1` residual norms. Row `i` corresponds to outer iteration `i`; the first column contains the total number of outer and inner iterations performed up to that step, and the second column contains the residual norm at that step.
    ///
    /// For example, if [`IrsRefinement::ClassicalGmres`] needs 3 outer iterations to converge and 4, 3, and 3 inner iterations at each outer iteration, it performs 10 total iterations.
    /// Row 0 corresponds to the first residual before the refinement start, so it has 0 in its first column.
    /// Row 1 corresponds to outer iteration 1 and contains 4 in its first column, row 2 contains 7, and row 3 contains 10.
    ///
    /// In summary, let `ldh = MaxIters + 1`, the leading dimension of the residual matrix. Then `residual_history[i]` contains the total number of iterations performed at outer iteration `i`, and `residual_history[i + ldh]` contains the residual norm at that outer iteration.
    ///
    /// # Errors
    ///
    /// Returns an error if residual history was not requested before solving,
    /// or if cuSOLVER rejects the `Infos` structure.
    pub fn residual_history_f32(&self) -> Result<ResidualHistory<f32>> {
        if !self.residual_history_requested {
            return Err(Error::InvalidPrecisionConfiguration);
        }
        let (leading_dimension, valid_rows) = self.residual_history_layout()?;
        let mut history = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosGetResidualHistory(
                self.as_raw(),
                &raw mut history,
            ))?;
            Ok(copy_residual_history(
                history.cast::<f32>(),
                leading_dimension,
                valid_rows,
            ))
        }
    }

    /// Returns the convergence history stored by the IRS solver when [`IrsInfos::request_residual_history`] was called before solving.
    /// The residual norm type depends on the input and output precision.
    /// Double-precision real and complex configurations report `f64` residuals, while single-precision real and complex configurations report `f32` residuals.
    ///
    /// The residual history matrix has two columns, even for multiple right-hand sides, and `MaxIters + 1` rows.
    /// Only the first `OuterNiters + 1` rows contain residual norms; the remaining rows are undefined.
    /// In the first column, each row `i` contains the total number of iterations performed up to outer iteration `i`.
    /// In the second column, each row contains the residual norm for that outer iteration.
    /// Row 0 contains the initial residual before the refinement loop starts, and subsequent rows contain residuals obtained at each outer iteration.
    /// The history only covers the outer loop.
    ///
    /// If the refinement solver was [`IrsRefinement::Classical`] or [`IrsRefinement::Gmres`], then `OuterNiters == Niters`, and there are `Niters + 1` rows of norms corresponding to the `Niters` outer iterations.
    ///
    /// If the refinement solver was [`IrsRefinement::ClassicalGmres`] or [`IrsRefinement::GmresGmres`], then `OuterNiters <= Niters` corresponds to the outer iterations performed by the outer refinement loop.
    /// There are `OuterNiters + 1` residual norms. Row `i` corresponds to outer iteration `i`; the first column contains the total number of outer and inner iterations performed up to that step, and the second column contains the residual norm at that step.
    ///
    /// For example, if [`IrsRefinement::ClassicalGmres`] needs 3 outer iterations to converge and 4, 3, and 3 inner iterations at each outer iteration, it performs 10 total iterations.
    /// Row 0 corresponds to the first residual before the refinement start, so it has 0 in its first column.
    /// Row 1 corresponds to outer iteration 1 and contains 4 in its first column, row 2 contains 7, and row 3 contains 10.
    ///
    /// In summary, let `ldh = MaxIters + 1`, the leading dimension of the residual matrix. Then `residual_history[i]` contains the total number of iterations performed at outer iteration `i`, and `residual_history[i + ldh]` contains the residual norm at that outer iteration.
    ///
    /// # Errors
    ///
    /// Returns an error if residual history was not requested before solving,
    /// or if cuSOLVER rejects the `Infos` structure.
    pub fn residual_history_f64(&self) -> Result<ResidualHistory<f64>> {
        if !self.residual_history_requested {
            return Err(Error::InvalidPrecisionConfiguration);
        }
        let (leading_dimension, valid_rows) = self.residual_history_layout()?;
        let mut history = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusolverDnIRSInfosGetResidualHistory(
                self.as_raw(),
                &raw mut history,
            ))?;
            Ok(copy_residual_history(
                history.cast::<f64>(),
                leading_dimension,
                valid_rows,
            ))
        }
    }

    pub fn as_raw(&self) -> sys::cusolverDnIRSInfos_t {
        self.handle
    }

    /// Takes ownership of a raw cuSOLVER IRS infos handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusolverDnIRSInfos_t` created by cuSOLVER.
    /// The returned wrapper takes ownership and will destroy it with
    /// `cusolverDnIRSInfosDestroy`; no other owner may destroy or keep using it.
    pub unsafe fn from_raw(handle: sys::cusolverDnIRSInfos_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle,
            residual_history_requested: false,
        })
    }

    /// Releases ownership and returns the raw cuSOLVER IRS infos handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::cusolverDnIRSInfos_t {
        let handle = self.handle;
        std::mem::forget(self);
        handle
    }

    fn residual_history_layout(&self) -> Result<(usize, usize)> {
        let leading_dimension = self
            .max_iterations()?
            .checked_add(1)
            .ok_or(Error::InvalidResidualHistory)
            .and_then(|value| {
                usize::try_from(value).map_err(|_| Error::OutOfRange {
                    name: "residual history leading dimension".into(),
                })
            })?;
        let valid_rows = self
            .outer_niters()?
            .checked_add(1)
            .ok_or(Error::InvalidResidualHistory)
            .and_then(|value| {
                usize::try_from(value).map_err(|_| Error::OutOfRange {
                    name: "residual history rows".into(),
                })
            })?;

        if valid_rows > leading_dimension {
            return Err(Error::InvalidResidualHistory);
        }

        Ok((leading_dimension, valid_rows))
    }
}

impl Drop for IrsInfos {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusolverDnIRSInfosDestroy(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusolver irs infos: {err}");
            }
        }
    }
}

pub fn xgesv_buffer_size<T: DataTypeLike>(
    ctx: &Context,
    params: &mut IrsParams,
    n: usize,
    nrhs: usize,
) -> Result<usize> {
    ctx.bind()?;
    if n == 0 || nrhs == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    params.ensure_type_precision::<T>()?;
    let mut workspace_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnIRSXgesv_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            to_i32(n, "n")?,
            to_i32(nrhs, "nrhs")?,
            &raw mut workspace_bytes,
        ))?;
    }
    to_usize(workspace_bytes, "workspace size")
}

/// Provides the same solve as the typed cuSOLVER `gesv` entry
/// points, but through a generic Rust wrapper that exposes IRS configuration
/// and reporting more directly.
/// [`xgesv`] allows additional control of the solver parameters such as setting:
///
/// * the main precision (input/output precision) of the solver
/// * the lowest precision to be used internally by the solver
/// * the refinement solver type
/// * the maximum allowed number of iterations in the refinement phase
/// * the tolerance of the refinement solver
/// * the fallback to main precision
/// * and more
///
/// through [`IrsParams`] and its helper methods.
/// Moreover, [`xgesv`] provides additional output information such as the convergence history (for example, residual norms) at each iteration and the number of iterations needed to converge.
/// [`IrsInfos`] exposes the information reported for a particular solve.
///
/// The returned value describes the solving process.
/// `Ok` indicates that the solve finished successfully. An error indicates that one of the arguments is incorrect, that the parameter or info structures are misconfigured, or that the solve did not finish successfully.
/// Check `niters` and `dinfo` for additional error details.
/// Provide the required device workspace through `workspace`.
/// Query the required byte count with [`xgesv_buffer_size`].
/// Apply any required configuration through the parameter structure before calling [`xgesv_buffer_size`] so the workspace size matches that configuration.
///
/// Tensor Float (TF32), introduced with NVIDIA Ampere architecture GPUs, is the most robust tensor core accelerated compute mode for the iterative refinement solver.
/// It solves a broad range of HPC problems and can provide up to 4x and 5x
/// speedups for real and complex systems, respectively.
/// On Volta and Turing architecture GPUs, half precision tensor core acceleration is recommended.
/// In cases where the iterative refinement solver fails to converge to the desired accuracy (main precision, input/output data precision), it is recommended to use main precision as internal lowest precision.
///
/// The following table provides all possible lowest-precision values corresponding to the input/output data type.
/// If the lowest precision matches the input/output data type, the main
/// precision factorization is used.
///
/// **Supported input/output data type and lower precision for the IRS solver**
///
/// | **input/output Data Type (for example, main precision)** | **Supported values for the lowest precision** |
/// | --- | --- |
/// | [`PrecisionType::C64F`] | [`PrecisionType::C64F`], [`PrecisionType::C32F`], [`PrecisionType::C16F`], [`PrecisionType::C16Bf`], [`PrecisionType::CTf32`] |
/// | [`PrecisionType::C32F`] | [`PrecisionType::C32F`], [`PrecisionType::C16F`], [`PrecisionType::C16Bf`], [`PrecisionType::CTf32`] |
/// | [`PrecisionType::R64F`] | [`PrecisionType::R64F`], [`PrecisionType::R32F`], [`PrecisionType::R16F`], [`PrecisionType::R16Bf`], [`PrecisionType::RTf32`] |
/// | [`PrecisionType::R32F`] | [`PrecisionType::R32F`], [`PrecisionType::R16F`], [`PrecisionType::R16Bf`], [`PrecisionType::RTf32`] |
///
/// [`xgesv_buffer_size`] returns the required workspace size in bytes for the
/// current [`IrsParams`] configuration.
///
/// # Errors
///
/// Returns an error if cuSOLVER rejects the matrix dimensions, leading
/// dimensions, parameter structure, info structure, or workspace. The workspace
/// can become invalid if [`xgesv_buffer_size`] is called and then an IRS
/// configuration value, such as the lowest precision, is changed. cuSOLVER can
/// also report an error if host memory allocation fails, if the selected IRS
/// configuration is not supported on the current GPU architecture, if the
/// library has not been initialized, or if the solve ends with an internal or
/// numerical failure. Check `niters` and `dinfo` for additional solver details.
pub fn xgesv<T: DataTypeLike>(
    ctx: &Context,
    params: &mut IrsParams,
    infos: &IrsInfos,
    n: usize,
    nrhs: usize,
    a: MatrixMut<'_, T>,
    b: MatrixRef<'_, T>,
    x: MatrixMut<'_, T>,
    device_workspace: &mut DeviceMemory<u8>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<i32> {
    ctx.bind()?;
    validate_matrix(n, n, a.data.len(), a.leading_dimension)?;
    validate_matrix(n, nrhs, b.data.len(), b.leading_dimension)?;
    validate_matrix(n, nrhs, x.data.len(), x.leading_dimension)?;
    require_info_buffer(dev_info)?;
    let workspace_bytes = xgesv_buffer_size::<T>(ctx, params, n, nrhs)?;
    require_workspace_bytes(device_workspace.byte_len(), workspace_bytes)?;
    let mut niters = 0;
    unsafe {
        try_ffi!(sys::cusolverDnIRSXgesv(
            ctx.as_raw(),
            params.as_raw(),
            infos.as_raw(),
            to_i32(n, "n")?,
            to_i32(nrhs, "nrhs")?,
            a.data.as_mut_ptr() as _,
            to_i32(a.leading_dimension, "ldda")?,
            b.data.as_ptr() as _,
            to_i32(b.leading_dimension, "lddb")?,
            x.data.as_mut_ptr() as _,
            to_i32(x.leading_dimension, "lddx")?,
            device_workspace.as_mut_ptr() as _,
            to_u64(workspace_bytes, "lwork_bytes")?,
            &raw mut niters,
            dev_info.as_mut_ptr() as _,
        ))?;
    }
    Ok(niters)
}

pub fn xgels_buffer_size<T: DataTypeLike>(
    ctx: &Context,
    params: &mut IrsParams,
    m: usize,
    n: usize,
    nrhs: usize,
) -> Result<usize> {
    ctx.bind()?;
    if m == 0 || n == 0 || nrhs == 0 || n > m {
        return Err(Error::InvalidMatrixShape);
    }
    params.ensure_type_precision::<T>()?;
    let mut workspace_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnIRSXgels_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(nrhs, "nrhs")?,
            &raw mut workspace_bytes,
        ))?;
    }
    to_usize(workspace_bytes, "workspace size")
}

/// Provides the same solve as the typed cuSOLVER `gels` entry
/// points, but through a generic Rust wrapper that exposes IRS configuration
/// and reporting more directly.
/// [`xgels`] allows additional control of the solver parameters such as setting:
///
/// * the main precision (input/output precision) of the solver,
/// * the lowest precision to be used internally by the solver,
/// * the refinement solver type
/// * the maximum allowed number of iterations in the refinement phase
/// * the tolerance of the refinement solver
/// * the fallback to main precision
/// * and others
///
/// through [`IrsParams`] and its helper methods.
/// Moreover, [`xgels`] provides additional output information such as the convergence history (for example, residual norms) at each iteration and the number of iterations needed to converge.
/// [`IrsInfos`] exposes the information reported for a particular solve.
///
/// The returned value describes the solving process.
/// `Ok` indicates that the solve finished successfully. An error indicates that one of the arguments is incorrect, that the parameter or info structures are misconfigured, or that the solve did not finish successfully.
/// Check `niters` and `dinfo` for additional error details.
/// Provide the required device workspace through `workspace`.
/// Query the required byte count with [`xgels_buffer_size`].
/// Apply any required configuration through the parameter structure before calling [`xgels_buffer_size`] so the workspace size matches that configuration.
///
/// The following table provides all possible lowest-precision values corresponding to the input/output data type.
/// If the lowest precision matches the input/output data type, the main
/// precision factorization is used.
///
/// Tensor Float (TF32), introduced with NVIDIA Ampere architecture GPUs, is the most robust tensor core accelerated compute mode for the iterative refinement solver.
/// It solves a broad range of HPC problems and can provide up to 4x and 5x
/// speedups for real and complex systems, respectively.
/// On Volta and Turing architecture GPUs, half precision tensor core acceleration is recommended.
/// In cases where the iterative refinement solver fails to converge to the desired accuracy (main precision, input/output data precision), it is recommended to use main precision as internal lowest precision.
///
/// **Supported input/output data type and lower precision for the IRS solver**
///
/// | **input/output Data Type (for example, main precision)** | **Supported values for the lowest precision** |
/// | --- | --- |
/// | [`PrecisionType::C64F`] | [`PrecisionType::C64F`], [`PrecisionType::C32F`], [`PrecisionType::C16F`], [`PrecisionType::C16Bf`], [`PrecisionType::CTf32`] |
/// | [`PrecisionType::C32F`] | [`PrecisionType::C32F`], [`PrecisionType::C16F`], [`PrecisionType::C16Bf`], [`PrecisionType::CTf32`] |
/// | [`PrecisionType::R64F`] | [`PrecisionType::R64F`], [`PrecisionType::R32F`], [`PrecisionType::R16F`], [`PrecisionType::R16Bf`], [`PrecisionType::RTf32`] |
/// | [`PrecisionType::R32F`] | [`PrecisionType::R32F`], [`PrecisionType::R16F`], [`PrecisionType::R16Bf`], [`PrecisionType::RTf32`] |
///
/// [`xgels_buffer_size`] returns the required workspace size in bytes for the
/// current [`IrsParams`] configuration.
///
/// # Errors
///
/// Returns an error if cuSOLVER rejects the matrix dimensions, leading
/// dimensions, parameter structure, info structure, or workspace. The workspace
/// can become invalid if [`xgels_buffer_size`] is called and then an IRS
/// configuration value, such as the lowest precision, is changed. cuSOLVER can
/// also report an error if host memory allocation fails, if the selected IRS
/// configuration is not supported on the current GPU architecture, if the
/// library has not been initialized, or if the solve ends with an internal or
/// numerical failure. Check `niters` and `dinfo` for additional solver details.
pub fn xgels<T: DataTypeLike>(
    ctx: &Context,
    params: &mut IrsParams,
    infos: &IrsInfos,
    m: usize,
    n: usize,
    nrhs: usize,
    a: MatrixMut<'_, T>,
    b: MatrixRef<'_, T>,
    x: MatrixMut<'_, T>,
    device_workspace: &mut DeviceMemory<u8>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<i32> {
    ctx.bind()?;
    if n > m {
        return Err(Error::InvalidMatrixShape);
    }
    validate_matrix(m, n, a.data.len(), a.leading_dimension)?;
    validate_matrix(m, nrhs, b.data.len(), b.leading_dimension)?;
    validate_matrix(n, nrhs, x.data.len(), x.leading_dimension)?;
    require_info_buffer(dev_info)?;
    let workspace_bytes = xgels_buffer_size::<T>(ctx, params, m, n, nrhs)?;
    require_workspace_bytes(device_workspace.byte_len(), workspace_bytes)?;
    let mut niters = 0;
    unsafe {
        try_ffi!(sys::cusolverDnIRSXgels(
            ctx.as_raw(),
            params.as_raw(),
            infos.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(nrhs, "nrhs")?,
            a.data.as_mut_ptr() as _,
            to_i32(a.leading_dimension, "ldda")?,
            b.data.as_ptr() as _,
            to_i32(b.leading_dimension, "lddb")?,
            x.data.as_mut_ptr() as _,
            to_i32(x.leading_dimension, "lddx")?,
            device_workspace.as_mut_ptr() as _,
            to_u64(workspace_bytes, "lwork_bytes")?,
            &raw mut niters,
            dev_info.as_mut_ptr() as _,
        ))?;
    }
    Ok(niters)
}

fn require_info_buffer(dev_info: &DeviceMemory<i32>) -> Result<()> {
    if dev_info.is_empty() {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn require_workspace_bytes(actual: usize, required: usize) -> Result<()> {
    if actual < required {
        return Err(Error::InsufficientWorkspaceSize { required, actual });
    }
    Ok(())
}

unsafe fn copy_residual_history<T: Copy>(
    history: *const T,
    leading_dimension: usize,
    valid_rows: usize,
) -> ResidualHistory<T> {
    let history = unsafe { slice::from_raw_parts(history, leading_dimension.saturating_mul(2)) };
    let mut rows = Vec::with_capacity(valid_rows);
    for row in 0..valid_rows {
        rows.push(ResidualHistoryEntry {
            total_iterations: history[row],
            residual_norm: history[row + leading_dimension],
        });
    }
    ResidualHistory {
        rows,
        leading_dimension,
    }
}

fn validate_matrix(rows: usize, cols: usize, len: usize, lda: usize) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    if lda < rows {
        return Err(Error::InvalidLeadingDimension);
    }
    let required = lda.checked_mul(cols).ok_or(Error::InvalidMatrixShape)?;
    if len < required {
        return Err(Error::InvalidMatrixShape);
    }
    Ok(())
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::memory::DeviceMemory;

    use super::*;
    use crate::testing::setup_context_if_available;

    #[test]
    fn test_xgesv_solves_diagonal_system() -> Result<()> {
        let Some(ctx) = setup_context_if_available()? else {
            return Ok(());
        };
        let mut params = IrsParams::create()?;
        let infos = IrsInfos::create()?;

        let mut a = DeviceMemory::from_slice(&[
            2.0_f32, 0.0, //
            0.0, 4.0,
        ])?;
        let b = DeviceMemory::from_slice(&[
            6.0_f32, //
            8.0,
        ])?;
        let mut x = DeviceMemory::create(2)?;
        let workspace_bytes = xgesv_buffer_size::<f32>(&ctx, &mut params, 2, 1)?;
        let mut workspace = DeviceMemory::create(workspace_bytes.max(1))?;
        let mut dev_info = DeviceMemory::create(1)?;

        let _ = xgesv(
            &ctx,
            &mut params,
            &infos,
            2,
            1,
            MatrixMut::new(&mut a, 2),
            MatrixRef::new(&b, 2),
            MatrixMut::new(&mut x, 2),
            &mut workspace,
            &mut dev_info,
        )?;

        assert_eq!(dev_info.copy_to_host_vec()?, vec![0]);
        assert_eq!(x.copy_to_host_vec()?, vec![3.0, 2.0]);
        Ok(())
    }
}
