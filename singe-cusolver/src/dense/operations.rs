use std::marker::PhantomData;

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    view::{DeviceOutputView, DeviceRepr, DeviceView},
};

use crate::{
    context::Context,
    dense::{
        GebrdScalar, OrgbrScalar, OrgqrScalar, OrgtrScalar, OrmqrScalar, OrmtrScalar,
        PotrfBatchedScalar, PotriScalar, PotrsBatchedScalar, SytrdScalar,
        legacy::qr::{xgeqrf, xgeqrf_buffer_size},
        xgetrf, xgetrf_buffer_size, xgetrs, xlarft, xlarft_buffer_size, xpotrf, xpotrf_buffer_size,
        xpotrs, xsytrf, xsytrf_buffer_size, xsytrs, xsytrs_buffer_size, xtrtri, xtrtri_buffer_size,
    },
    error::Result,
    layout::{
        BatchedMatrixRef, BatchedVectorRef, ByteWorkspaceMut, MatrixLayout, MatrixMut, MatrixRef,
        VectorOutput, VectorRef, WorkspaceSizes,
    },
    params::Params,
    types::{DiagonalType, DirectMode, FillMode, Operation, SideMode, StorevMode},
};

/// Typed Cholesky factorization operation.
///
/// This is the preferred operation-family API for POTRF. It keeps matrix
/// metadata grouped in [`MatrixRef`] and [`MatrixMut`] instead of exposing the
/// cuSOLVER `S`/`D`/`C`/`Z` entry point names and separate `a`/`lda` arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Potrf<T: DeviceRepr> {
    fill_mode: FillMode,
    n: usize,
    compute_type: DataType,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Potrf::run`].
#[derive(Debug)]
pub struct PotrfArgs<'a, T: DeviceRepr> {
    /// Initialized square matrix overwritten in the selected triangle by its Cholesky factor.
    pub a: MatrixMut<'a, T>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed Cholesky solve operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Potrs<T: DeviceRepr> {
    fill_mode: FillMode,
    n: usize,
    right_hand_sides: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Potrs::run`].
#[derive(Debug)]
pub struct PotrsArgs<'a, T: DeviceRepr> {
    /// Initialized Cholesky factor in the selected triangle.
    pub factor: MatrixRef<'a, T>,
    /// Initialized right-hand-side columns overwritten by the solution.
    pub rhs: MatrixMut<'a, T>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed Cholesky inverse operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Potri<T: DeviceRepr> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Potri::run`].
#[derive(Debug)]
pub struct PotriArgs<'a, T: DeviceRepr> {
    /// Initialized Cholesky factor overwritten in the selected triangle by the inverse.
    pub factor: MatrixMut<'a, T>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed batched Cholesky factorization operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedPotrf<T: DeviceRepr> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`BatchedPotrf::run`].
#[derive(Debug)]
pub struct BatchedPotrfArgs<'a, T: DeviceRepr> {
    /// Device pointer table for initialized square matrices, each overwritten by its Cholesky factor.
    ///
    /// The opposite triangle may be used as scratch. The caller must retain and exclusively borrow every pointee through completion.
    pub a: BatchedMatrixRef<'a, T>,
    /// Device output capacity for one solver status per matrix; inspect each completed status before using its factor.
    pub info: DeviceOutputView<'a, i32>,
}

/// Typed batched Cholesky solve operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedPotrs<T: DeviceRepr> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`BatchedPotrs::run`].
#[derive(Debug)]
pub struct BatchedPotrsArgs<'a, T: DeviceRepr> {
    /// Device pointer table for initialized Cholesky factors.
    ///
    /// The opposite triangle may be used as scratch. The caller must retain and exclusively borrow each factor allocation through completion.
    pub factors: BatchedMatrixRef<'a, T>,
    /// Device pointer table for initialized single right-hand sides, overwritten by solutions.
    ///
    /// The caller must retain and exclusively borrow every pointee through completion.
    pub rhs: BatchedVectorRef<'a, T>,
    /// Device output capacity for one status for the whole batch, rather than one per matrix.
    pub info: DeviceOutputView<'a, i32>,
}

/// Typed LU factorization operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Getrf<T: DeviceRepr> {
    rows: usize,
    cols: usize,
    compute_type: DataType,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Getrf::run`].
#[derive(Debug)]
pub struct GetrfArgs<'a, T: DeviceRepr> {
    /// Initialized matrix overwritten by combined lower and upper triangular LU factors.
    pub a: MatrixMut<'a, T>,
    /// Optional device capacity for `min(rows, columns)` native 64-bit pivot indices.
    ///
    /// `None` passes a null pointer to request factorization without pivoting.
    pub pivots: Option<DeviceOutputView<'a, i64>>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed LU solve operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Getrs<TA: DeviceRepr, TB: DeviceRepr = TA> {
    operation: Operation,
    n: usize,
    right_hand_sides: usize,
    _ty: PhantomData<(TA, TB)>,
}

/// Runtime buffers for [`Getrs::run`].
#[derive(Debug)]
pub struct GetrsArgs<'a, TA: DeviceRepr, TB: DeviceRepr = TA> {
    /// Initialized LU factors from a compatible factorization.
    pub factor: MatrixRef<'a, TA>,
    /// Initialized native 64-bit pivot indices associated with the LU factorization.
    pub pivots: DeviceView<'a, i64>,
    /// Initialized right-hand-side columns overwritten by the solution.
    pub rhs: MatrixMut<'a, TB>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed triangular inverse operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trtri<T: DeviceRepr> {
    fill_mode: FillMode,
    diagonal_type: DiagonalType,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Trtri::run`].
#[derive(Debug)]
pub struct TrtriArgs<'a, T: DeviceRepr> {
    /// Initialized triangular matrix overwritten in the selected triangle by its inverse.
    pub a: MatrixMut<'a, T>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed symmetric or Hermitian solve operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sytrs<TA: DeviceRepr, TB: DeviceRepr = TA> {
    fill_mode: FillMode,
    n: usize,
    right_hand_sides: usize,
    _ty: PhantomData<(TA, TB)>,
}

/// Runtime buffers for [`Sytrs::run`].
#[derive(Debug)]
pub struct SytrsArgs<'a, TA: DeviceRepr, TB: DeviceRepr = TA> {
    /// Initialized indefinite factorization in the selected triangle.
    pub factor: MatrixRef<'a, TA>,
    /// Optional initialized native 64-bit pivot data matching the factorization. `None` passes a null pointer.
    pub pivots: Option<DeviceView<'a, i64>>,
    /// Initialized right-hand-side columns overwritten by the solution.
    pub rhs: MatrixMut<'a, TB>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed symmetric or Hermitian indefinite factorization operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sytrf<T: DeviceRepr> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Sytrf::run`].
#[derive(Debug)]
pub struct SytrfArgs<'a, T: DeviceRepr> {
    /// Initialized square matrix overwritten in the selected triangle by its indefinite factorization.
    pub a: MatrixMut<'a, T>,
    /// Optional device capacity for `n` native 64-bit pivot entries. `None` passes a null pointer.
    pub pivots: Option<DeviceOutputView<'a, i64>>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed symmetric or Hermitian tridiagonal reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sytrd<T: DeviceRepr> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Sytrd::run`].
#[derive(Debug)]
pub struct SytrdArgs<'a, T>
where
    T: DeviceRepr + SytrdScalar,
{
    /// Initialized symmetric or Hermitian matrix overwritten with tridiagonal entries and elementary reflectors.
    pub a: MatrixMut<'a, T>,
    /// Device output capacity for `n` real tridiagonal diagonal values.
    pub diagonal: VectorOutput<'a, T::Real>,
    /// Device output capacity for `n - 1` real tridiagonal off-diagonal values.
    pub off_diagonal: VectorOutput<'a, T::Real>,
    /// Device output capacity for `n - 1` elementary-reflector coefficients.
    pub tau: VectorOutput<'a, T>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed operation that forms explicit tridiagonalization factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Orgtr<T: DeviceRepr> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Orgtr::run`].
#[derive(Debug)]
pub struct OrgtrArgs<'a, T: DeviceRepr> {
    /// Initialized tridiagonal-reduction reflectors overwritten by the explicit orthogonal or unitary factor.
    pub a: MatrixMut<'a, T>,
    /// Initialized elementary-reflector coefficients from the corresponding tridiagonal reduction.
    pub tau: VectorRef<'a, T>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed operation that applies tridiagonalization factors to a matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ormtr<T: DeviceRepr> {
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    rows: usize,
    cols: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Ormtr::run`].
#[derive(Debug)]
pub struct OrmtrArgs<'a, T: DeviceRepr> {
    /// Initialized elementary reflectors from a compatible tridiagonal reduction.
    pub reflectors: MatrixRef<'a, T>,
    /// Initialized coefficients associated with the tridiagonal-reduction reflectors.
    pub tau: VectorRef<'a, T>,
    /// Initialized matrix overwritten by the selected orthogonal or unitary transformation.
    pub matrix: MatrixMut<'a, T>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed LARFT triangular-factor operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Larft<TV: DeviceRepr, TTau: DeviceRepr = TV, TT: DeviceRepr = TV> {
    direct: DirectMode,
    storev: StorevMode,
    n: usize,
    k: usize,
    compute_type: DataType,
    _ty: PhantomData<(TV, TTau, TT)>,
}

/// Runtime buffers for [`Larft::run`].
#[derive(Debug)]
pub struct LarftArgs<'a, TV: DeviceRepr, TTau: DeviceRepr = TV, TT: DeviceRepr = TV> {
    /// Initialized elementary-reflector vectors in the configured rowwise or columnwise layout.
    pub v: MatrixRef<'a, TV>,
    /// Initialized scalar coefficient for each of the `k` elementary reflectors.
    pub tau: VectorRef<'a, TTau>,
    /// Initialized triangular-factor destination.
    ///
    /// The configured triangle contains the result; the other triangle and padding are outside the numerical output contract.
    pub t: MatrixMut<'a, TT>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
}

/// Typed QR factorization operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geqrf<TA: DeviceRepr, TTau: DeviceRepr = TA> {
    rows: usize,
    cols: usize,
    compute_type: DataType,
    _ty: PhantomData<(TA, TTau)>,
}

/// Runtime buffers for [`Geqrf::run`].
#[derive(Debug)]
pub struct GeqrfArgs<'a, TA: DeviceRepr, TTau: DeviceRepr = TA> {
    /// Initialized matrix overwritten with the upper triangular factor and elementary reflectors.
    pub a: MatrixMut<'a, TA>,
    /// Device output capacity for `min(rows, columns)` elementary-reflector coefficients.
    pub tau: VectorOutput<'a, TTau>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed operation that applies QR reflectors to a matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ormqr<T: DeviceRepr> {
    side: SideMode,
    operation: Operation,
    rows: usize,
    cols: usize,
    reflectors: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Ormqr::run`].
#[derive(Debug)]
pub struct OrmqrArgs<'a, T: DeviceRepr> {
    /// Initialized elementary reflectors from a compatible QR factorization.
    pub reflectors: MatrixRef<'a, T>,
    /// Initialized scalar coefficients associated with the QR reflectors.
    pub tau: VectorRef<'a, T>,
    /// Initialized matrix overwritten by the selected orthogonal or unitary transformation.
    pub matrix: MatrixMut<'a, T>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed operation that forms explicit QR factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Orgqr<T: DeviceRepr> {
    rows: usize,
    cols: usize,
    reflectors: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Orgqr::run`].
#[derive(Debug)]
pub struct OrgqrArgs<'a, T: DeviceRepr> {
    /// Initialized QR reflectors overwritten by the requested explicit orthogonal or unitary columns.
    pub a: MatrixMut<'a, T>,
    /// Initialized coefficients for the configured number of QR reflectors.
    pub tau: VectorRef<'a, T>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed bidiagonal reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gebrd<T: DeviceRepr> {
    rows: usize,
    cols: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Gebrd::run`].
#[derive(Debug)]
pub struct GebrdArgs<'a, T>
where
    T: DeviceRepr + GebrdScalar,
{
    /// Initialized matrix overwritten with bidiagonal entries and reflectors representing the left and right transformations.
    pub a: MatrixMut<'a, T>,
    /// Device output capacity for `min(rows, columns)` real bidiagonal diagonal values.
    pub diagonal: VectorOutput<'a, T::Real>,
    /// Device storage with capacity for `min(rows, columns)` real values; only the first `min(rows, columns) - 1` off-diagonal values are numerical results.
    pub off_diagonal: VectorOutput<'a, T::Real>,
    /// Device output capacity for `min(rows, columns)` coefficients of the left elementary reflectors.
    pub tau_q: VectorOutput<'a, T>,
    /// Device output capacity for `min(rows, columns)` coefficients of the right elementary reflectors.
    pub tau_p: VectorOutput<'a, T>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed operation that forms explicit bidiagonalization factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Orgbr<T: DeviceRepr> {
    side: SideMode,
    rows: usize,
    cols: usize,
    reflectors: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Orgbr::run`].
#[derive(Debug)]
pub struct OrgbrArgs<'a, T: DeviceRepr> {
    /// Initialized bidiagonal-reduction reflectors overwritten by the requested explicit orthogonal or unitary factor.
    pub a: MatrixMut<'a, T>,
    /// Initialized coefficients for the selected left or right bidiagonalization reflectors.
    pub tau: VectorRef<'a, T>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

impl<T> Potrf<T>
where
    T: DeviceRepr + DataTypeLike,
{
    /// Creates a typed POTRF operation using the matrix data type as compute type.
    #[must_use]
    pub fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            compute_type: T::data_type(),
            _ty: PhantomData,
        }
    }

    /// Sets the compute type passed to the generic cuSOLVER X API.
    #[must_use]
    pub const fn with_compute_type(mut self, compute_type: DataType) -> Self {
        self.compute_type = compute_type;
        self
    }

    /// Returns the cuSOLVER fill mode.
    #[must_use]
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the cuSOLVER compute type.
    #[must_use]
    pub const fn compute_type(&self) -> DataType {
        self.compute_type
    }

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix shape is invalid or cuSOLVER cannot
    /// report the workspace sizes.
    pub fn workspace_size(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xpotrf_buffer_size::<T>(
            context,
            params,
            self.fill_mode,
            self.n,
            a,
            self.compute_type,
        )
    }

    /// Computes the Cholesky factorization in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, params: &Params, args: PotrfArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.a.data)?;
            context.prepare_region(&args.workspace.device)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                xpotrf(
                    context,
                    params,
                    self.fill_mode,
                    self.n,
                    args.a,
                    self.compute_type,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> Potrs<T>
where
    T: DeviceRepr + DataTypeLike,
{
    /// Creates a typed POTRS operation.
    #[must_use]
    pub const fn new(fill_mode: FillMode, n: usize, right_hand_sides: usize) -> Self {
        Self {
            fill_mode,
            n,
            right_hand_sides,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    #[must_use]
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of right-hand sides.
    #[must_use]
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }

    /// Solves the linear system in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    pub fn run(&self, context: &Context, params: &Params, args: PotrsArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.factor.data)?;
            context.prepare_region(&args.rhs.data)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                xpotrs(
                    context,
                    params,
                    self.fill_mode,
                    self.n,
                    self.right_hand_sides,
                    args.factor,
                    args.rhs,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> Potri<T>
where
    T: DeviceRepr + PotriScalar,
{
    /// Creates a typed POTRI operation.
    #[must_use]
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    #[must_use]
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix shape is invalid or cuSOLVER cannot
    /// report the workspace size.
    pub fn workspace_len(&self, context: &Context, factor: MatrixLayout) -> Result<usize> {
        T::potri_buffer_size(context, self.fill_mode, self.n, factor)
    }

    /// Computes the inverse in place from a Cholesky factorization.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, args: PotriArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.factor.data)?;
            context.prepare_region(&args.workspace)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                T::potri(
                    context,
                    self.fill_mode,
                    self.n,
                    args.factor,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> BatchedPotrf<T>
where
    T: DeviceRepr + PotrfBatchedScalar,
{
    /// Creates a typed batched POTRF operation.
    #[must_use]
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    #[must_use]
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Computes batched Cholesky factorizations in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    /// # Safety
    ///
    /// Retain every device pointer-table pointee through this completed call,
    /// and establish its shape, initialization, actual context, and access exclusivity.
    pub unsafe fn run(&self, context: &Context, args: BatchedPotrfArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.a.pointers)?;
            context.prepare_region(&args.info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe { T::potrf_batched(context, self.fill_mode, self.n, args.a, args.info) }
        })
    }
}

impl<T> BatchedPotrs<T>
where
    T: DeviceRepr + PotrsBatchedScalar,
{
    /// Creates a typed batched POTRS operation.
    ///
    /// The cuSOLVER batched POTRS API supports one right-hand side per matrix.
    #[must_use]
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    #[must_use]
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Solves batched linear systems in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    /// # Safety
    ///
    /// Retain every device pointer-table pointee through this completed call,
    /// and establish its shape, initialization, actual context, and access exclusivity.
    pub unsafe fn run(&self, context: &Context, args: BatchedPotrsArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.factors.pointers)?;
            context.prepare_region(&args.rhs.pointers)?;
            context.prepare_region(&args.info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                T::potrs_batched(
                    context,
                    self.fill_mode,
                    self.n,
                    args.factors,
                    args.rhs,
                    args.info,
                )
            }
        })
    }
}

impl<T> Getrf<T>
where
    T: DeviceRepr + DataTypeLike,
{
    /// Creates a typed GETRF operation using the matrix data type as compute type.
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            compute_type: T::data_type(),
            _ty: PhantomData,
        }
    }

    /// Sets the compute type passed to the generic cuSOLVER X API.
    #[must_use]
    pub const fn with_compute_type(mut self, compute_type: DataType) -> Self {
        self.compute_type = compute_type;
        self
    }

    /// Returns the matrix row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the matrix column count.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the cuSOLVER compute type.
    #[must_use]
    pub const fn compute_type(&self) -> DataType {
        self.compute_type
    }

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix shape is invalid or cuSOLVER cannot
    /// report the workspace sizes.
    pub fn workspace_size(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xgetrf_buffer_size::<T>(context, params, self.rows, self.cols, a, self.compute_type)
    }

    /// Computes the LU factorization in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, params: &Params, args: GetrfArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.a.data)?;
            if let Some(value) = args.pivots.as_ref() {
                context.prepare_region(value)?;
            }
            context.prepare_region(&args.workspace.device)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                xgetrf(
                    context,
                    params,
                    self.rows,
                    self.cols,
                    args.a,
                    args.pivots,
                    self.compute_type,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<TA, TB> Getrs<TA, TB>
where
    TA: DeviceRepr + DataTypeLike,
    TB: DeviceRepr + DataTypeLike,
{
    /// Creates a typed GETRS operation.
    #[must_use]
    pub const fn new(operation: Operation, n: usize, right_hand_sides: usize) -> Self {
        Self {
            operation,
            n,
            right_hand_sides,
            _ty: PhantomData,
        }
    }

    /// Returns the operation applied to the factorized matrix.
    #[must_use]
    pub const fn operation(&self) -> Operation {
        self.operation
    }

    /// Returns the square factor dimension.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of right-hand sides.
    #[must_use]
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }

    /// Solves the linear system in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    pub fn run(
        &self,
        context: &Context,
        params: &Params,
        args: GetrsArgs<'_, TA, TB>,
    ) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.factor.data)?;
            context.prepare_region(&args.pivots)?;
            context.prepare_region(&args.rhs.data)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                xgetrs(
                    context,
                    params,
                    self.operation,
                    self.n,
                    self.right_hand_sides,
                    args.factor,
                    args.pivots,
                    args.rhs,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> Trtri<T>
where
    T: DeviceRepr + DataTypeLike,
{
    /// Creates a typed TRTRI operation.
    #[must_use]
    pub const fn new(fill_mode: FillMode, diagonal_type: DiagonalType, n: usize) -> Self {
        Self {
            fill_mode,
            diagonal_type,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    #[must_use]
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the diagonal type.
    #[must_use]
    pub const fn diagonal_type(&self) -> DiagonalType {
        self.diagonal_type
    }

    /// Returns the square matrix dimension.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix shape is invalid or cuSOLVER cannot
    /// report the workspace sizes.
    pub fn workspace_size(&self, context: &Context, a: MatrixLayout) -> Result<WorkspaceSizes> {
        xtrtri_buffer_size::<T>(context, self.fill_mode, self.diagonal_type, self.n, a)
    }

    /// Computes the triangular inverse in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, args: TrtriArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.a.data)?;
            context.prepare_region(&args.workspace.device)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                xtrtri(
                    context,
                    self.fill_mode,
                    self.diagonal_type,
                    self.n,
                    args.a,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<TA, TB> Sytrs<TA, TB>
where
    TA: DeviceRepr + DataTypeLike,
    TB: DeviceRepr + DataTypeLike,
{
    /// Creates a typed SYTRS operation.
    #[must_use]
    pub const fn new(fill_mode: FillMode, n: usize, right_hand_sides: usize) -> Self {
        Self {
            fill_mode,
            n,
            right_hand_sides,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    #[must_use]
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square factor dimension.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of right-hand sides.
    #[must_use]
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace sizes.
    pub fn workspace_size(
        &self,
        context: &Context,
        factor: MatrixLayout,
        rhs: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xsytrs_buffer_size::<TA, TB>(
            context,
            self.fill_mode,
            self.n,
            self.right_hand_sides,
            factor,
            rhs,
        )
    }

    /// Solves the symmetric or Hermitian system in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, args: SytrsArgs<'_, TA, TB>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.factor.data)?;
            if let Some(value) = args.pivots.as_ref() {
                context.prepare_region(value)?;
            }
            context.prepare_region(&args.rhs.data)?;
            context.prepare_region(&args.workspace.device)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                xsytrs(
                    context,
                    self.fill_mode,
                    self.n,
                    self.right_hand_sides,
                    args.factor,
                    args.pivots,
                    args.rhs,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> Sytrf<T>
where
    T: DeviceRepr + DataTypeLike,
{
    /// Creates a typed SYTRF operation.
    #[must_use]
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    #[must_use]
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the device and host workspace sizes before allocating buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace length.
    pub fn workspace_size(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xsytrf_buffer_size::<T>(context, params, self.fill_mode, self.n, a, T::data_type())
    }

    /// Computes the indefinite factorization in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, params: &Params, args: SytrfArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.a.data)?;
            if let Some(value) = args.pivots.as_ref() {
                context.prepare_region(value)?;
            }
            context.prepare_region(&args.workspace.device)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                xsytrf(
                    context,
                    params,
                    self.fill_mode,
                    self.n,
                    args.a,
                    args.pivots,
                    T::data_type(),
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> Sytrd<T>
where
    T: DeviceRepr + SytrdScalar,
{
    /// Creates a typed SYTRD/HETRD operation.
    #[must_use]
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    #[must_use]
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace length.
    pub fn workspace_len(&self, context: &Context, a: MatrixLayout) -> Result<usize> {
        T::sytrd_buffer_size(context, self.fill_mode, self.n, a)
    }

    /// Reduces the matrix to symmetric or Hermitian tridiagonal form in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, args: SytrdArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.a.data)?;
            context.prepare_region(&args.diagonal.data)?;
            context.prepare_region(&args.off_diagonal.data)?;
            context.prepare_region(&args.tau.data)?;
            context.prepare_region(&args.workspace)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                T::sytrd(
                    context,
                    self.fill_mode,
                    self.n,
                    args.a,
                    args.diagonal,
                    args.off_diagonal,
                    args.tau,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> Orgtr<T>
where
    T: DeviceRepr + OrgtrScalar,
{
    /// Creates a typed ORGTR/UNGTR operation.
    #[must_use]
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    #[must_use]
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace length.
    pub fn workspace_len(&self, context: &Context, a: MatrixLayout) -> Result<usize> {
        T::orgtr_buffer_size(context, self.fill_mode, self.n, a)
    }

    /// Forms the explicit tridiagonalization factor matrix in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, args: OrgtrArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.a.data)?;
            context.prepare_region(&args.tau.data)?;
            context.prepare_region(&args.workspace)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                T::orgtr(
                    context,
                    self.fill_mode,
                    self.n,
                    args.a,
                    args.tau,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> Ormtr<T>
where
    T: DeviceRepr + OrmtrScalar,
{
    /// Creates a typed ORMTR/UNMTR operation.
    #[must_use]
    pub const fn new(
        side: SideMode,
        fill_mode: FillMode,
        operation: Operation,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            side,
            fill_mode,
            operation,
            rows,
            cols,
            _ty: PhantomData,
        }
    }

    /// Returns which side receives the tridiagonalization factor product.
    #[must_use]
    pub const fn side(&self) -> SideMode {
        self.side
    }

    /// Returns the cuSOLVER fill mode.
    #[must_use]
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the operation applied to the factor product.
    #[must_use]
    pub const fn operation(&self) -> Operation {
        self.operation
    }

    /// Returns the target matrix row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the target matrix column count.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace length.
    pub fn workspace_len(
        &self,
        context: &Context,
        reflectors: MatrixLayout,
        matrix: MatrixLayout,
    ) -> Result<usize> {
        T::ormtr_buffer_size(
            context,
            self.side,
            self.fill_mode,
            self.operation,
            self.rows,
            self.cols,
            reflectors,
            matrix,
        )
    }

    /// Applies the tridiagonalization factor product in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, args: OrmtrArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.reflectors.data)?;
            context.prepare_region(&args.tau.data)?;
            context.prepare_region(&args.matrix.data)?;
            context.prepare_region(&args.workspace)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                T::ormtr(
                    context,
                    self.side,
                    self.fill_mode,
                    self.operation,
                    self.rows,
                    self.cols,
                    args.reflectors,
                    args.tau,
                    args.matrix,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<TV, TTau, TT> Larft<TV, TTau, TT>
where
    TV: DeviceRepr + DataTypeLike,
    TTau: DeviceRepr + DataTypeLike,
    TT: DeviceRepr + DataTypeLike,
{
    /// Creates a typed LARFT operation using `T`'s data type as compute type.
    #[must_use]
    pub fn new(direct: DirectMode, storev: StorevMode, n: usize, k: usize) -> Self {
        Self {
            direct,
            storev,
            n,
            k,
            compute_type: TT::data_type(),
            _ty: PhantomData,
        }
    }

    /// Sets the compute type passed to the generic cuSOLVER X API.
    #[must_use]
    pub const fn with_compute_type(mut self, compute_type: DataType) -> Self {
        self.compute_type = compute_type;
        self
    }

    /// Returns the reflector product direction.
    #[must_use]
    pub const fn direct(&self) -> DirectMode {
        self.direct
    }

    /// Returns the reflector storage mode.
    #[must_use]
    pub const fn storev(&self) -> StorevMode {
        self.storev
    }

    /// Returns the reflector order.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of elementary reflectors.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Returns the cuSOLVER compute type.
    #[must_use]
    pub const fn compute_type(&self) -> DataType {
        self.compute_type
    }

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace sizes.
    pub fn workspace_size(
        &self,
        context: &Context,
        params: &Params,
        v: MatrixLayout,
        t: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xlarft_buffer_size::<TV, TTau, TT>(
            context,
            params,
            self.direct,
            self.storev,
            self.n,
            self.k,
            v,
            t,
            self.compute_type,
        )
    }

    /// Forms the triangular factor in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(
        &self,
        context: &Context,
        params: &Params,
        args: LarftArgs<'_, TV, TTau, TT>,
    ) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.v.data)?;
            context.prepare_region(&args.tau.data)?;
            context.prepare_region(&args.t.data)?;
            context.prepare_region(&args.workspace.device)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                xlarft(
                    context,
                    params,
                    self.direct,
                    self.storev,
                    self.n,
                    self.k,
                    args.v,
                    args.tau,
                    args.t,
                    self.compute_type,
                    args.workspace,
                )
            }
        })
    }
}

impl<TA, TTau> Geqrf<TA, TTau>
where
    TA: DeviceRepr + DataTypeLike,
    TTau: DeviceRepr + DataTypeLike,
{
    /// Creates a typed GEQRF operation using the matrix data type as compute type.
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            compute_type: TA::data_type(),
            _ty: PhantomData,
        }
    }

    /// Sets the compute type passed to the generic cuSOLVER X API.
    #[must_use]
    pub const fn with_compute_type(mut self, compute_type: DataType) -> Self {
        self.compute_type = compute_type;
        self
    }

    /// Returns the matrix row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the matrix column count.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the cuSOLVER compute type.
    #[must_use]
    pub const fn compute_type(&self) -> DataType {
        self.compute_type
    }

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace sizes.
    pub fn workspace_size(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xgeqrf_buffer_size::<TA, TTau>(context, params, self.rows, self.cols, a, self.compute_type)
    }

    /// Computes the QR factorization in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(
        &self,
        context: &Context,
        params: &Params,
        args: GeqrfArgs<'_, TA, TTau>,
    ) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.a.data)?;
            context.prepare_region(&args.tau.data)?;
            context.prepare_region(&args.workspace.device)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                xgeqrf(
                    context,
                    params,
                    self.rows,
                    self.cols,
                    args.a,
                    args.tau,
                    self.compute_type,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> Ormqr<T>
where
    T: DeviceRepr + OrmqrScalar,
{
    /// Creates a typed ORMQR/UNMQR operation.
    #[must_use]
    pub const fn new(
        side: SideMode,
        operation: Operation,
        rows: usize,
        cols: usize,
        reflectors: usize,
    ) -> Self {
        Self {
            side,
            operation,
            rows,
            cols,
            reflectors,
            _ty: PhantomData,
        }
    }

    /// Returns which side receives the reflector product.
    #[must_use]
    pub const fn side(&self) -> SideMode {
        self.side
    }

    /// Returns the operation applied to the reflector product.
    #[must_use]
    pub const fn operation(&self) -> Operation {
        self.operation
    }

    /// Returns the output matrix row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the output matrix column count.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of elementary reflectors.
    #[must_use]
    pub const fn reflectors(&self) -> usize {
        self.reflectors
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace length.
    pub fn workspace_len(
        &self,
        context: &Context,
        reflectors: MatrixLayout,
        matrix: MatrixLayout,
    ) -> Result<usize> {
        T::ormqr_buffer_size(
            context,
            self.side,
            self.operation,
            self.rows,
            self.cols,
            self.reflectors,
            reflectors,
            matrix,
        )
    }

    /// Applies the QR reflector product in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, args: OrmqrArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.reflectors.data)?;
            context.prepare_region(&args.tau.data)?;
            context.prepare_region(&args.matrix.data)?;
            context.prepare_region(&args.workspace)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                T::ormqr(
                    context,
                    self.side,
                    self.operation,
                    self.rows,
                    self.cols,
                    self.reflectors,
                    args.reflectors,
                    args.tau,
                    args.matrix,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> Orgqr<T>
where
    T: DeviceRepr + OrgqrScalar,
{
    /// Creates a typed ORGQR/UNGQR operation.
    #[must_use]
    pub const fn new(rows: usize, cols: usize, reflectors: usize) -> Self {
        Self {
            rows,
            cols,
            reflectors,
            _ty: PhantomData,
        }
    }

    /// Returns the output matrix row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the output matrix column count.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of elementary reflectors.
    #[must_use]
    pub const fn reflectors(&self) -> usize {
        self.reflectors
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace length.
    pub fn workspace_len(&self, context: &Context, a: MatrixLayout) -> Result<usize> {
        T::orgqr_buffer_size(context, self.rows, self.cols, self.reflectors, a)
    }

    /// Forms the explicit QR factor matrix in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, args: OrgqrArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.a.data)?;
            context.prepare_region(&args.tau.data)?;
            context.prepare_region(&args.workspace)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                T::orgqr(
                    context,
                    self.rows,
                    self.cols,
                    self.reflectors,
                    args.a,
                    args.tau,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> Gebrd<T>
where
    T: DeviceRepr + GebrdScalar,
{
    /// Creates a typed GEBRD operation.
    #[must_use]
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            _ty: PhantomData,
        }
    }

    /// Returns the matrix row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the matrix column count.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or if cuSOLVER cannot report
    /// the workspace length.
    pub fn workspace_len(&self, context: &Context) -> Result<usize> {
        T::gebrd_buffer_size(context, self.rows, self.cols)
    }

    /// Reduces the matrix to bidiagonal form in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, args: GebrdArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.a.data)?;
            context.prepare_region(&args.diagonal.data)?;
            context.prepare_region(&args.off_diagonal.data)?;
            context.prepare_region(&args.tau_q.data)?;
            context.prepare_region(&args.tau_p.data)?;
            context.prepare_region(&args.workspace)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                T::gebrd(
                    context,
                    self.rows,
                    self.cols,
                    args.a,
                    args.diagonal,
                    args.off_diagonal,
                    args.tau_q,
                    args.tau_p,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<T> Orgbr<T>
where
    T: DeviceRepr + OrgbrScalar,
{
    /// Creates a typed ORGBR/UNGBR operation.
    #[must_use]
    pub const fn new(side: SideMode, rows: usize, cols: usize, reflectors: usize) -> Self {
        Self {
            side,
            rows,
            cols,
            reflectors,
            _ty: PhantomData,
        }
    }

    /// Returns whether to form `Q` or `P`.
    #[must_use]
    pub const fn side(&self) -> SideMode {
        self.side
    }

    /// Returns the output matrix row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the output matrix column count.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of elementary reflectors.
    #[must_use]
    pub const fn reflectors(&self) -> usize {
        self.reflectors
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace length.
    pub fn workspace_len(&self, context: &Context, a: MatrixLayout) -> Result<usize> {
        T::orgbr_buffer_size(context, self.side, self.rows, self.cols, self.reflectors, a)
    }

    /// Forms the explicit bidiagonalization factor matrix in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, context: &Context, args: OrgbrArgs<'_, T>) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&args.a.data)?;
            context.prepare_region(&args.tau.data)?;
            context.prepare_region(&args.workspace)?;
            context.prepare_region(&args.dev_info)?;
            // SAFETY: the prepared bindings remain borrowed until this scope completes.
            unsafe {
                T::orgbr(
                    context,
                    self.side,
                    self.rows,
                    self.cols,
                    self.reflectors,
                    args.a,
                    args.tau,
                    args.workspace,
                    args.dev_info,
                )
            }
        })
    }
}

impl<'a, T: DeviceRepr> PotrfArgs<'a, T> {
    /// Groups the matrix, workspace, and info buffer needed by [`Potrf::run`].
    #[must_use]
    pub const fn new(
        a: MatrixMut<'a, T>,
        workspace: ByteWorkspaceMut<'a>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            a,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T: DeviceRepr> GetrfArgs<'a, T> {
    /// Groups the matrix, optional pivot buffer, workspace, and info buffer.
    #[must_use]
    pub const fn new(
        a: MatrixMut<'a, T>,
        pivots: Option<DeviceOutputView<'a, i64>>,
        workspace: ByteWorkspaceMut<'a>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            a,
            pivots,
            workspace,
            dev_info,
        }
    }
}

impl<'a, TA: DeviceRepr, TB: DeviceRepr> GetrsArgs<'a, TA, TB> {
    /// Groups the factorized matrix, pivots, right-hand sides, and info buffer.
    #[must_use]
    pub const fn new(
        factor: MatrixRef<'a, TA>,
        pivots: DeviceView<'a, i64>,
        rhs: MatrixMut<'a, TB>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            factor,
            pivots,
            rhs,
            dev_info,
        }
    }
}

impl<'a, T: DeviceRepr> TrtriArgs<'a, T> {
    /// Groups the triangular matrix, workspace, and info buffer.
    #[must_use]
    pub const fn new(
        a: MatrixMut<'a, T>,
        workspace: ByteWorkspaceMut<'a>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            a,
            workspace,
            dev_info,
        }
    }
}

impl<'a, TA: DeviceRepr, TB: DeviceRepr> SytrsArgs<'a, TA, TB> {
    /// Groups the factorized matrix, optional pivots, right-hand sides,
    /// workspace, and info buffer.
    #[must_use]
    pub const fn new(
        factor: MatrixRef<'a, TA>,
        pivots: Option<DeviceView<'a, i64>>,
        rhs: MatrixMut<'a, TB>,
        workspace: ByteWorkspaceMut<'a>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            factor,
            pivots,
            rhs,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T: DeviceRepr> SytrfArgs<'a, T> {
    /// Groups the factor matrix, optional pivot buffer, workspace, and info
    /// buffer.
    #[must_use]
    pub const fn new(
        a: MatrixMut<'a, T>,
        pivots: Option<DeviceOutputView<'a, i64>>,
        workspace: ByteWorkspaceMut<'a>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            a,
            pivots,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T> SytrdArgs<'a, T>
where
    T: DeviceRepr + SytrdScalar,
{
    /// Groups the matrix, tridiagonal outputs, reflectors, workspace, and info
    /// buffer.
    #[must_use]
    pub const fn new(
        a: MatrixMut<'a, T>,
        diagonal: VectorOutput<'a, T::Real>,
        off_diagonal: VectorOutput<'a, T::Real>,
        tau: VectorOutput<'a, T>,
        workspace: DeviceOutputView<'a, T>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            a,
            diagonal,
            off_diagonal,
            tau,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T: DeviceRepr> OrgtrArgs<'a, T> {
    /// Groups the matrix, tau vector, workspace, and info buffer.
    #[must_use]
    pub const fn new(
        a: MatrixMut<'a, T>,
        tau: VectorRef<'a, T>,
        workspace: DeviceOutputView<'a, T>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            a,
            tau,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T: DeviceRepr> OrmtrArgs<'a, T> {
    /// Groups the reflector matrix, tau vector, target matrix, workspace, and
    /// info buffer.
    #[must_use]
    pub const fn new(
        reflectors: MatrixRef<'a, T>,
        tau: VectorRef<'a, T>,
        matrix: MatrixMut<'a, T>,
        workspace: DeviceOutputView<'a, T>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            reflectors,
            tau,
            matrix,
            workspace,
            dev_info,
        }
    }
}

impl<'a, TV: DeviceRepr, TTau: DeviceRepr, TT: DeviceRepr> LarftArgs<'a, TV, TTau, TT> {
    /// Groups the reflector matrix, tau vector, output factor, and workspace.
    #[must_use]
    pub const fn new(
        v: MatrixRef<'a, TV>,
        tau: VectorRef<'a, TTau>,
        t: MatrixMut<'a, TT>,
        workspace: ByteWorkspaceMut<'a>,
    ) -> Self {
        Self {
            v,
            tau,
            t,
            workspace,
        }
    }
}

impl<'a, TA: DeviceRepr, TTau: DeviceRepr> GeqrfArgs<'a, TA, TTau> {
    /// Groups the matrix, tau vector, workspace, and info buffer.
    #[must_use]
    pub const fn new(
        a: MatrixMut<'a, TA>,
        tau: VectorOutput<'a, TTau>,
        workspace: ByteWorkspaceMut<'a>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            a,
            tau,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T: DeviceRepr> OrmqrArgs<'a, T> {
    /// Groups the reflector matrix, tau vector, target matrix, workspace, and
    /// info buffer.
    #[must_use]
    pub const fn new(
        reflectors: MatrixRef<'a, T>,
        tau: VectorRef<'a, T>,
        matrix: MatrixMut<'a, T>,
        workspace: DeviceOutputView<'a, T>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            reflectors,
            tau,
            matrix,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T: DeviceRepr> OrgqrArgs<'a, T> {
    /// Groups the matrix, tau vector, workspace, and info buffer.
    #[must_use]
    pub const fn new(
        a: MatrixMut<'a, T>,
        tau: VectorRef<'a, T>,
        workspace: DeviceOutputView<'a, T>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            a,
            tau,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T> GebrdArgs<'a, T>
where
    T: DeviceRepr + GebrdScalar,
{
    /// Groups the matrix, bidiagonal outputs, reflectors, workspace, and info
    /// buffer.
    #[must_use]
    pub const fn new(
        a: MatrixMut<'a, T>,
        diagonal: VectorOutput<'a, T::Real>,
        off_diagonal: VectorOutput<'a, T::Real>,
        tau_q: VectorOutput<'a, T>,
        tau_p: VectorOutput<'a, T>,
        workspace: DeviceOutputView<'a, T>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            a,
            diagonal,
            off_diagonal,
            tau_q,
            tau_p,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T: DeviceRepr> OrgbrArgs<'a, T> {
    /// Groups the matrix, tau vector, workspace, and info buffer.
    #[must_use]
    pub const fn new(
        a: MatrixMut<'a, T>,
        tau: VectorRef<'a, T>,
        workspace: DeviceOutputView<'a, T>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            a,
            tau,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T: DeviceRepr> PotrsArgs<'a, T> {
    /// Groups the factorized matrix, right-hand sides, and info buffer needed by [`Potrs::run`].
    #[must_use]
    pub const fn new(
        factor: MatrixRef<'a, T>,
        rhs: MatrixMut<'a, T>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            factor,
            rhs,
            dev_info,
        }
    }
}

impl<'a, T: DeviceRepr> PotriArgs<'a, T> {
    /// Groups the factorized matrix, workspace, and info buffer needed by [`Potri::run`].
    #[must_use]
    pub const fn new(
        factor: MatrixMut<'a, T>,
        workspace: DeviceOutputView<'a, T>,
        dev_info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self {
            factor,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T: DeviceRepr> BatchedPotrfArgs<'a, T> {
    /// Groups the matrix pointer array and info buffer needed by [`BatchedPotrf::run`].
    #[must_use]
    pub const fn new(a: BatchedMatrixRef<'a, T>, info: DeviceOutputView<'a, i32>) -> Self {
        Self { a, info }
    }
}

impl<'a, T: DeviceRepr> BatchedPotrsArgs<'a, T> {
    /// Groups the factor pointer array, right-hand-side pointer array, and info buffer.
    #[must_use]
    pub const fn new(
        factors: BatchedMatrixRef<'a, T>,
        rhs: BatchedVectorRef<'a, T>,
        info: DeviceOutputView<'a, i32>,
    ) -> Self {
        Self { factors, rhs, info }
    }
}
