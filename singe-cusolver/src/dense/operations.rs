use std::marker::PhantomData;

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    memory::DeviceMemory,
};

use crate::{
    context::Context,
    dense::{
        GebrdScalar, OrgbrScalar, OrgqrScalar, OrgtrScalar, OrmqrScalar, OrmtrScalar,
        PotrfBatchedScalar, PotriScalar, PotrsBatchedScalar, SytrdScalar, SytrfScalar,
        legacy::qr::{xgeqrf, xgeqrf_buffer_size},
        xgetrf, xgetrf_buffer_size, xgetrs, xlarft, xlarft_buffer_size, xpotrf, xpotrf_buffer_size,
        xpotrs, xsytrs, xsytrs_buffer_size, xtrtri, xtrtri_buffer_size,
    },
    error::Result,
    layout::{
        BatchedMatrixRef, BatchedVectorRef, ByteWorkspaceMut, MatrixMut, MatrixRef, VectorMut,
        VectorRef, WorkspaceSizes,
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
pub struct Potrf<T> {
    fill_mode: FillMode,
    n: usize,
    compute_type: DataType,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Potrf::run`].
#[derive(Debug)]
pub struct PotrfArgs<'a, T> {
    pub a: MatrixMut<'a, T>,
    pub workspace: ByteWorkspaceMut<'a>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed Cholesky solve operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Potrs<T> {
    fill_mode: FillMode,
    n: usize,
    right_hand_sides: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Potrs::run`].
#[derive(Debug)]
pub struct PotrsArgs<'a, T> {
    pub factor: MatrixRef<'a, T>,
    pub rhs: MatrixMut<'a, T>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed Cholesky inverse operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Potri<T> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Potri::run`].
#[derive(Debug)]
pub struct PotriArgs<'a, T> {
    pub factor: MatrixMut<'a, T>,
    pub workspace: &'a mut DeviceMemory<T>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed batched Cholesky factorization operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedPotrf<T> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`BatchedPotrf::run`].
#[derive(Debug)]
pub struct BatchedPotrfArgs<'a, T> {
    pub a: BatchedMatrixRef<'a, T>,
    pub info: &'a mut DeviceMemory<i32>,
}

/// Typed batched Cholesky solve operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedPotrs<T> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`BatchedPotrs::run`].
#[derive(Debug)]
pub struct BatchedPotrsArgs<'a, T> {
    pub factors: BatchedMatrixRef<'a, T>,
    pub rhs: BatchedVectorRef<'a, T>,
    pub info: &'a mut DeviceMemory<i32>,
}

/// Typed LU factorization operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Getrf<T> {
    rows: usize,
    cols: usize,
    compute_type: DataType,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Getrf::run`].
#[derive(Debug)]
pub struct GetrfArgs<'a, T> {
    pub a: MatrixMut<'a, T>,
    pub pivots: Option<&'a mut DeviceMemory<i64>>,
    pub workspace: ByteWorkspaceMut<'a>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed LU solve operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Getrs<TA, TB = TA> {
    operation: Operation,
    n: usize,
    right_hand_sides: usize,
    _ty: PhantomData<(TA, TB)>,
}

/// Runtime buffers for [`Getrs::run`].
#[derive(Debug)]
pub struct GetrsArgs<'a, TA, TB = TA> {
    pub factor: MatrixRef<'a, TA>,
    pub pivots: &'a DeviceMemory<i64>,
    pub rhs: MatrixMut<'a, TB>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed triangular inverse operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trtri<T> {
    fill_mode: FillMode,
    diagonal_type: DiagonalType,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Trtri::run`].
#[derive(Debug)]
pub struct TrtriArgs<'a, T> {
    pub a: MatrixMut<'a, T>,
    pub workspace: ByteWorkspaceMut<'a>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed symmetric or Hermitian solve operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sytrs<TA, TB = TA> {
    fill_mode: FillMode,
    n: usize,
    right_hand_sides: usize,
    _ty: PhantomData<(TA, TB)>,
}

/// Runtime buffers for [`Sytrs::run`].
#[derive(Debug)]
pub struct SytrsArgs<'a, TA, TB = TA> {
    pub factor: MatrixRef<'a, TA>,
    pub pivots: Option<&'a DeviceMemory<i64>>,
    pub rhs: MatrixMut<'a, TB>,
    pub workspace: ByteWorkspaceMut<'a>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed symmetric or Hermitian indefinite factorization operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sytrf<T> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Sytrf::run`].
#[derive(Debug)]
pub struct SytrfArgs<'a, T> {
    pub a: MatrixMut<'a, T>,
    pub pivots: Option<&'a mut DeviceMemory<i32>>,
    pub workspace: &'a mut DeviceMemory<T>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed symmetric or Hermitian tridiagonal reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sytrd<T> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Sytrd::run`].
#[derive(Debug)]
pub struct SytrdArgs<'a, T>
where
    T: SytrdScalar,
{
    pub a: MatrixMut<'a, T>,
    pub diagonal: VectorMut<'a, T::Real>,
    pub off_diagonal: VectorMut<'a, T::Real>,
    pub tau: VectorMut<'a, T>,
    pub workspace: &'a mut DeviceMemory<T>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed operation that forms explicit tridiagonalization factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Orgtr<T> {
    fill_mode: FillMode,
    n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Orgtr::run`].
#[derive(Debug)]
pub struct OrgtrArgs<'a, T> {
    pub a: MatrixMut<'a, T>,
    pub tau: VectorRef<'a, T>,
    pub workspace: &'a mut DeviceMemory<T>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed operation that applies tridiagonalization factors to a matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ormtr<T> {
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    rows: usize,
    cols: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Ormtr::run`].
#[derive(Debug)]
pub struct OrmtrArgs<'a, T> {
    pub reflectors: MatrixMut<'a, T>,
    pub tau: VectorMut<'a, T>,
    pub matrix: MatrixMut<'a, T>,
    pub workspace: &'a mut DeviceMemory<T>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed LARFT triangular-factor operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Larft<TV, TTau = TV, TT = TV> {
    direct: DirectMode,
    storev: StorevMode,
    n: usize,
    k: usize,
    compute_type: DataType,
    _ty: PhantomData<(TV, TTau, TT)>,
}

/// Runtime buffers for [`Larft::run`].
#[derive(Debug)]
pub struct LarftArgs<'a, TV, TTau = TV, TT = TV> {
    pub v: MatrixRef<'a, TV>,
    pub tau: VectorRef<'a, TTau>,
    pub t: MatrixMut<'a, TT>,
    pub workspace: ByteWorkspaceMut<'a>,
}

/// Typed QR factorization operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geqrf<TA, TTau = TA> {
    rows: usize,
    cols: usize,
    compute_type: DataType,
    _ty: PhantomData<(TA, TTau)>,
}

/// Runtime buffers for [`Geqrf::run`].
#[derive(Debug)]
pub struct GeqrfArgs<'a, TA, TTau = TA> {
    pub a: MatrixMut<'a, TA>,
    pub tau: VectorMut<'a, TTau>,
    pub workspace: ByteWorkspaceMut<'a>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed operation that applies QR reflectors to a matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ormqr<T> {
    side: SideMode,
    operation: Operation,
    rows: usize,
    cols: usize,
    reflectors: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Ormqr::run`].
#[derive(Debug)]
pub struct OrmqrArgs<'a, T> {
    pub reflectors: MatrixRef<'a, T>,
    pub tau: VectorRef<'a, T>,
    pub matrix: MatrixMut<'a, T>,
    pub workspace: &'a mut DeviceMemory<T>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed operation that forms explicit QR factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Orgqr<T> {
    rows: usize,
    cols: usize,
    reflectors: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Orgqr::run`].
#[derive(Debug)]
pub struct OrgqrArgs<'a, T> {
    pub a: MatrixMut<'a, T>,
    pub tau: VectorRef<'a, T>,
    pub workspace: &'a mut DeviceMemory<T>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed bidiagonal reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gebrd<T> {
    rows: usize,
    cols: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Gebrd::run`].
#[derive(Debug)]
pub struct GebrdArgs<'a, T>
where
    T: GebrdScalar,
{
    pub a: MatrixMut<'a, T>,
    pub diagonal: VectorMut<'a, T::Real>,
    pub off_diagonal: VectorMut<'a, T::Real>,
    pub tau_q: VectorMut<'a, T>,
    pub tau_p: VectorMut<'a, T>,
    pub workspace: &'a mut DeviceMemory<T>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

/// Typed operation that forms explicit bidiagonalization factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Orgbr<T> {
    side: SideMode,
    rows: usize,
    cols: usize,
    reflectors: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Orgbr::run`].
#[derive(Debug)]
pub struct OrgbrArgs<'a, T> {
    pub a: MatrixMut<'a, T>,
    pub tau: VectorRef<'a, T>,
    pub workspace: &'a mut DeviceMemory<T>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

impl<T> Potrf<T>
where
    T: DataTypeLike,
{
    /// Creates a typed POTRF operation using the matrix data type as compute type.
    pub fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            compute_type: T::data_type(),
            _ty: PhantomData,
        }
    }

    /// Sets the compute type passed to the generic cuSOLVER X API.
    pub const fn with_compute_type(mut self, compute_type: DataType) -> Self {
        self.compute_type = compute_type;
        self
    }

    /// Returns the cuSOLVER fill mode.
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the cuSOLVER compute type.
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
        ctx: &Context,
        params: &Params,
        a: MatrixRef<'_, T>,
    ) -> Result<WorkspaceSizes> {
        xpotrf_buffer_size(ctx, params, self.fill_mode, self.n, a, self.compute_type)
    }

    /// Computes the Cholesky factorization in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, params: &Params, args: PotrfArgs<'_, T>) -> Result<()> {
        xpotrf(
            ctx,
            params,
            self.fill_mode,
            self.n,
            args.a,
            self.compute_type,
            args.workspace,
            args.dev_info,
        )
    }
}

impl<T> Potrs<T>
where
    T: DataTypeLike,
{
    /// Creates a typed POTRS operation.
    pub const fn new(fill_mode: FillMode, n: usize, right_hand_sides: usize) -> Self {
        Self {
            fill_mode,
            n,
            right_hand_sides,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of right-hand sides.
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }

    /// Solves the linear system in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    pub fn run(&self, ctx: &Context, params: &Params, args: PotrsArgs<'_, T>) -> Result<()> {
        xpotrs(
            ctx,
            params,
            self.fill_mode,
            self.n,
            self.right_hand_sides,
            args.factor,
            args.rhs,
            args.dev_info,
        )
    }
}

impl<T> Potri<T>
where
    T: PotriScalar,
{
    /// Creates a typed POTRI operation.
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix shape is invalid or cuSOLVER cannot
    /// report the workspace size.
    pub fn workspace_len(&self, ctx: &Context, factor: MatrixMut<'_, T>) -> Result<usize> {
        T::potri_buffer_size(ctx, self.fill_mode, self.n, factor)
    }

    /// Computes the inverse in place from a Cholesky factorization.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, args: PotriArgs<'_, T>) -> Result<()> {
        T::potri(
            ctx,
            self.fill_mode,
            self.n,
            args.factor,
            args.workspace,
            args.dev_info,
        )
    }
}

impl<T> BatchedPotrf<T>
where
    T: PotrfBatchedScalar,
{
    /// Creates a typed batched POTRF operation.
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Computes batched Cholesky factorizations in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    pub fn run(&self, ctx: &Context, args: BatchedPotrfArgs<'_, T>) -> Result<()> {
        T::potrf_batched(ctx, self.fill_mode, self.n, args.a, args.info)
    }
}

impl<T> BatchedPotrs<T>
where
    T: PotrsBatchedScalar,
{
    /// Creates a typed batched POTRS operation.
    ///
    /// The cuSOLVER batched POTRS API supports one right-hand side per matrix.
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Solves batched linear systems in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    pub fn run(&self, ctx: &Context, args: BatchedPotrsArgs<'_, T>) -> Result<()> {
        T::potrs_batched(
            ctx,
            self.fill_mode,
            self.n,
            args.factors,
            args.rhs,
            args.info,
        )
    }
}

impl<T> Getrf<T>
where
    T: DataTypeLike,
{
    /// Creates a typed GETRF operation using the matrix data type as compute type.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            compute_type: T::data_type(),
            _ty: PhantomData,
        }
    }

    /// Sets the compute type passed to the generic cuSOLVER X API.
    pub const fn with_compute_type(mut self, compute_type: DataType) -> Self {
        self.compute_type = compute_type;
        self
    }

    /// Returns the matrix row count.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the matrix column count.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the cuSOLVER compute type.
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
        ctx: &Context,
        params: &Params,
        a: MatrixRef<'_, T>,
    ) -> Result<WorkspaceSizes> {
        xgetrf_buffer_size(ctx, params, self.rows, self.cols, a, self.compute_type)
    }

    /// Computes the LU factorization in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, params: &Params, args: GetrfArgs<'_, T>) -> Result<()> {
        xgetrf(
            ctx,
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
}

impl<TA, TB> Getrs<TA, TB>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
{
    /// Creates a typed GETRS operation.
    pub const fn new(operation: Operation, n: usize, right_hand_sides: usize) -> Self {
        Self {
            operation,
            n,
            right_hand_sides,
            _ty: PhantomData,
        }
    }

    /// Returns the operation applied to the factorized matrix.
    pub const fn operation(&self) -> Operation {
        self.operation
    }

    /// Returns the square factor dimension.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of right-hand sides.
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }

    /// Solves the linear system in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    pub fn run(&self, ctx: &Context, params: &Params, args: GetrsArgs<'_, TA, TB>) -> Result<()> {
        xgetrs(
            ctx,
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
}

impl<T> Trtri<T>
where
    T: DataTypeLike,
{
    /// Creates a typed TRTRI operation.
    pub const fn new(fill_mode: FillMode, diagonal_type: DiagonalType, n: usize) -> Self {
        Self {
            fill_mode,
            diagonal_type,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the diagonal type.
    pub const fn diagonal_type(&self) -> DiagonalType {
        self.diagonal_type
    }

    /// Returns the square matrix dimension.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix shape is invalid or cuSOLVER cannot
    /// report the workspace sizes.
    pub fn workspace_size(&self, ctx: &Context, a: MatrixRef<'_, T>) -> Result<WorkspaceSizes> {
        xtrtri_buffer_size(ctx, self.fill_mode, self.diagonal_type, self.n, a)
    }

    /// Computes the triangular inverse in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, args: TrtriArgs<'_, T>) -> Result<()> {
        xtrtri(
            ctx,
            self.fill_mode,
            self.diagonal_type,
            self.n,
            args.a,
            args.workspace,
            args.dev_info,
        )
    }
}

impl<TA, TB> Sytrs<TA, TB>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
{
    /// Creates a typed SYTRS operation.
    pub const fn new(fill_mode: FillMode, n: usize, right_hand_sides: usize) -> Self {
        Self {
            fill_mode,
            n,
            right_hand_sides,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square factor dimension.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of right-hand sides.
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
        ctx: &Context,
        factor: MatrixRef<'_, TA>,
        pivots: Option<&DeviceMemory<i64>>,
        rhs: MatrixRef<'_, TB>,
    ) -> Result<WorkspaceSizes> {
        xsytrs_buffer_size(
            ctx,
            self.fill_mode,
            self.n,
            self.right_hand_sides,
            factor,
            pivots,
            rhs,
        )
    }

    /// Solves the symmetric or Hermitian system in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, args: SytrsArgs<'_, TA, TB>) -> Result<()> {
        xsytrs(
            ctx,
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
}

impl<T> Sytrf<T>
where
    T: SytrfScalar,
{
    /// Creates a typed SYTRF operation.
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace length.
    pub fn workspace_len(&self, ctx: &Context, a: MatrixMut<'_, T>) -> Result<usize> {
        T::sytrf_buffer_size(ctx, self.n, a)
    }

    /// Computes the indefinite factorization in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, args: SytrfArgs<'_, T>) -> Result<()> {
        T::sytrf(
            ctx,
            self.fill_mode,
            self.n,
            args.a,
            args.pivots,
            args.workspace,
            args.dev_info,
        )
    }
}

impl<T> Sytrd<T>
where
    T: SytrdScalar,
{
    /// Creates a typed SYTRD/HETRD operation.
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace length.
    pub fn workspace_len(
        &self,
        ctx: &Context,
        a: MatrixRef<'_, T>,
        diagonal: VectorRef<'_, T::Real>,
        off_diagonal: VectorRef<'_, T::Real>,
        tau: VectorRef<'_, T>,
    ) -> Result<usize> {
        T::sytrd_buffer_size(ctx, self.fill_mode, self.n, a, diagonal, off_diagonal, tau)
    }

    /// Reduces the matrix to symmetric or Hermitian tridiagonal form in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, args: SytrdArgs<'_, T>) -> Result<()> {
        T::sytrd(
            ctx,
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
}

impl<T> Orgtr<T>
where
    T: OrgtrScalar,
{
    /// Creates a typed ORGTR/UNGTR operation.
    pub const fn new(fill_mode: FillMode, n: usize) -> Self {
        Self {
            fill_mode,
            n,
            _ty: PhantomData,
        }
    }

    /// Returns the cuSOLVER fill mode.
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the square matrix dimension.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the
    /// workspace length.
    pub fn workspace_len(
        &self,
        ctx: &Context,
        a: MatrixRef<'_, T>,
        tau: VectorRef<'_, T>,
    ) -> Result<usize> {
        T::orgtr_buffer_size(ctx, self.fill_mode, self.n, a, tau)
    }

    /// Forms the explicit tridiagonalization factor matrix in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, args: OrgtrArgs<'_, T>) -> Result<()> {
        T::orgtr(
            ctx,
            self.fill_mode,
            self.n,
            args.a,
            args.tau,
            args.workspace,
            args.dev_info,
        )
    }
}

impl<T> Ormtr<T>
where
    T: OrmtrScalar,
{
    /// Creates a typed ORMTR/UNMTR operation.
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
    pub const fn side(&self) -> SideMode {
        self.side
    }

    /// Returns the cuSOLVER fill mode.
    pub const fn fill_mode(&self) -> FillMode {
        self.fill_mode
    }

    /// Returns the operation applied to the factor product.
    pub const fn operation(&self) -> Operation {
        self.operation
    }

    /// Returns the target matrix row count.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the target matrix column count.
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
        ctx: &Context,
        reflectors: MatrixRef<'_, T>,
        tau: VectorRef<'_, T>,
        matrix: MatrixRef<'_, T>,
    ) -> Result<usize> {
        T::ormtr_buffer_size(
            ctx,
            self.side,
            self.fill_mode,
            self.operation,
            self.rows,
            self.cols,
            reflectors,
            tau,
            matrix,
        )
    }

    /// Applies the tridiagonalization factor product in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, args: OrmtrArgs<'_, T>) -> Result<()> {
        T::ormtr(
            ctx,
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
}

impl<TV, TTau, TT> Larft<TV, TTau, TT>
where
    TV: DataTypeLike,
    TTau: DataTypeLike,
    TT: DataTypeLike,
{
    /// Creates a typed LARFT operation using `T`'s data type as compute type.
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
    pub const fn with_compute_type(mut self, compute_type: DataType) -> Self {
        self.compute_type = compute_type;
        self
    }

    /// Returns the reflector product direction.
    pub const fn direct(&self) -> DirectMode {
        self.direct
    }

    /// Returns the reflector storage mode.
    pub const fn storev(&self) -> StorevMode {
        self.storev
    }

    /// Returns the reflector order.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of elementary reflectors.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Returns the cuSOLVER compute type.
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
        ctx: &Context,
        params: &Params,
        v: MatrixRef<'_, TV>,
        tau: VectorRef<'_, TTau>,
        t: MatrixRef<'_, TT>,
    ) -> Result<WorkspaceSizes> {
        xlarft_buffer_size(
            ctx,
            params,
            self.direct,
            self.storev,
            self.n,
            self.k,
            v,
            tau,
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
        ctx: &Context,
        params: &Params,
        args: LarftArgs<'_, TV, TTau, TT>,
    ) -> Result<()> {
        xlarft(
            ctx,
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
}

impl<TA, TTau> Geqrf<TA, TTau>
where
    TA: DataTypeLike,
    TTau: DataTypeLike,
{
    /// Creates a typed GEQRF operation using the matrix data type as compute type.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            compute_type: TA::data_type(),
            _ty: PhantomData,
        }
    }

    /// Sets the compute type passed to the generic cuSOLVER X API.
    pub const fn with_compute_type(mut self, compute_type: DataType) -> Self {
        self.compute_type = compute_type;
        self
    }

    /// Returns the matrix row count.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the matrix column count.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the cuSOLVER compute type.
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
        ctx: &Context,
        params: &Params,
        a: MatrixRef<'_, TA>,
        tau: VectorRef<'_, TTau>,
    ) -> Result<WorkspaceSizes> {
        xgeqrf_buffer_size(ctx, params, self.rows, self.cols, a, tau, self.compute_type)
    }

    /// Computes the QR factorization in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, params: &Params, args: GeqrfArgs<'_, TA, TTau>) -> Result<()> {
        xgeqrf(
            ctx,
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
}

impl<T> Ormqr<T>
where
    T: OrmqrScalar,
{
    /// Creates a typed ORMQR/UNMQR operation.
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
    pub const fn side(&self) -> SideMode {
        self.side
    }

    /// Returns the operation applied to the reflector product.
    pub const fn operation(&self) -> Operation {
        self.operation
    }

    /// Returns the output matrix row count.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the output matrix column count.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of elementary reflectors.
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
        ctx: &Context,
        reflectors: MatrixRef<'_, T>,
        tau: VectorRef<'_, T>,
        matrix: MatrixRef<'_, T>,
    ) -> Result<usize> {
        T::ormqr_buffer_size(
            ctx,
            self.side,
            self.operation,
            self.rows,
            self.cols,
            self.reflectors,
            reflectors,
            tau,
            matrix,
        )
    }

    /// Applies the QR reflector product in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, args: OrmqrArgs<'_, T>) -> Result<()> {
        T::ormqr(
            ctx,
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
}

impl<T> Orgqr<T>
where
    T: OrgqrScalar,
{
    /// Creates a typed ORGQR/UNGQR operation.
    pub const fn new(rows: usize, cols: usize, reflectors: usize) -> Self {
        Self {
            rows,
            cols,
            reflectors,
            _ty: PhantomData,
        }
    }

    /// Returns the output matrix row count.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the output matrix column count.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of elementary reflectors.
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
        ctx: &Context,
        a: MatrixRef<'_, T>,
        tau: VectorRef<'_, T>,
    ) -> Result<usize> {
        T::orgqr_buffer_size(ctx, self.rows, self.cols, self.reflectors, a, tau)
    }

    /// Forms the explicit QR factor matrix in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, args: OrgqrArgs<'_, T>) -> Result<()> {
        T::orgqr(
            ctx,
            self.rows,
            self.cols,
            self.reflectors,
            args.a,
            args.tau,
            args.workspace,
            args.dev_info,
        )
    }
}

impl<T> Gebrd<T>
where
    T: GebrdScalar,
{
    /// Creates a typed GEBRD operation.
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            _ty: PhantomData,
        }
    }

    /// Returns the matrix row count.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the matrix column count.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or if cuSOLVER cannot report
    /// the workspace length.
    pub fn workspace_len(&self, ctx: &Context) -> Result<usize> {
        T::gebrd_buffer_size(ctx, self.rows, self.cols)
    }

    /// Reduces the matrix to bidiagonal form in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, args: GebrdArgs<'_, T>) -> Result<()> {
        T::gebrd(
            ctx,
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
}

impl<T> Orgbr<T>
where
    T: OrgbrScalar,
{
    /// Creates a typed ORGBR/UNGBR operation.
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
    pub const fn side(&self) -> SideMode {
        self.side
    }

    /// Returns the output matrix row count.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the output matrix column count.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of elementary reflectors.
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
        ctx: &Context,
        a: MatrixRef<'_, T>,
        tau: VectorRef<'_, T>,
    ) -> Result<usize> {
        T::orgbr_buffer_size(
            ctx,
            self.side,
            self.rows,
            self.cols,
            self.reflectors,
            a,
            tau,
        )
    }

    /// Forms the explicit bidiagonalization factor matrix in place.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER
    /// reports a failure.
    pub fn run(&self, ctx: &Context, args: OrgbrArgs<'_, T>) -> Result<()> {
        T::orgbr(
            ctx,
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
}

impl<'a, T> PotrfArgs<'a, T> {
    /// Groups the matrix, workspace, and info buffer needed by [`Potrf::run`].
    pub const fn new(
        a: MatrixMut<'a, T>,
        workspace: ByteWorkspaceMut<'a>,
        dev_info: &'a mut DeviceMemory<i32>,
    ) -> Self {
        Self {
            a,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T> GetrfArgs<'a, T> {
    /// Groups the matrix, optional pivot buffer, workspace, and info buffer.
    pub const fn new(
        a: MatrixMut<'a, T>,
        pivots: Option<&'a mut DeviceMemory<i64>>,
        workspace: ByteWorkspaceMut<'a>,
        dev_info: &'a mut DeviceMemory<i32>,
    ) -> Self {
        Self {
            a,
            pivots,
            workspace,
            dev_info,
        }
    }
}

impl<'a, TA, TB> GetrsArgs<'a, TA, TB> {
    /// Groups the factorized matrix, pivots, right-hand sides, and info buffer.
    pub const fn new(
        factor: MatrixRef<'a, TA>,
        pivots: &'a DeviceMemory<i64>,
        rhs: MatrixMut<'a, TB>,
        dev_info: &'a mut DeviceMemory<i32>,
    ) -> Self {
        Self {
            factor,
            pivots,
            rhs,
            dev_info,
        }
    }
}

impl<'a, T> TrtriArgs<'a, T> {
    /// Groups the triangular matrix, workspace, and info buffer.
    pub const fn new(
        a: MatrixMut<'a, T>,
        workspace: ByteWorkspaceMut<'a>,
        dev_info: &'a mut DeviceMemory<i32>,
    ) -> Self {
        Self {
            a,
            workspace,
            dev_info,
        }
    }
}

impl<'a, TA, TB> SytrsArgs<'a, TA, TB> {
    /// Groups the factorized matrix, optional pivots, right-hand sides,
    /// workspace, and info buffer.
    pub const fn new(
        factor: MatrixRef<'a, TA>,
        pivots: Option<&'a DeviceMemory<i64>>,
        rhs: MatrixMut<'a, TB>,
        workspace: ByteWorkspaceMut<'a>,
        dev_info: &'a mut DeviceMemory<i32>,
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

impl<'a, T> SytrfArgs<'a, T> {
    /// Groups the factor matrix, optional pivot buffer, workspace, and info
    /// buffer.
    pub const fn new(
        a: MatrixMut<'a, T>,
        pivots: Option<&'a mut DeviceMemory<i32>>,
        workspace: &'a mut DeviceMemory<T>,
        dev_info: &'a mut DeviceMemory<i32>,
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
    T: SytrdScalar,
{
    /// Groups the matrix, tridiagonal outputs, reflectors, workspace, and info
    /// buffer.
    pub const fn new(
        a: MatrixMut<'a, T>,
        diagonal: VectorMut<'a, T::Real>,
        off_diagonal: VectorMut<'a, T::Real>,
        tau: VectorMut<'a, T>,
        workspace: &'a mut DeviceMemory<T>,
        dev_info: &'a mut DeviceMemory<i32>,
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

impl<'a, T> OrgtrArgs<'a, T> {
    /// Groups the matrix, tau vector, workspace, and info buffer.
    pub const fn new(
        a: MatrixMut<'a, T>,
        tau: VectorRef<'a, T>,
        workspace: &'a mut DeviceMemory<T>,
        dev_info: &'a mut DeviceMemory<i32>,
    ) -> Self {
        Self {
            a,
            tau,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T> OrmtrArgs<'a, T> {
    /// Groups the reflector matrix, tau vector, target matrix, workspace, and
    /// info buffer.
    pub const fn new(
        reflectors: MatrixMut<'a, T>,
        tau: VectorMut<'a, T>,
        matrix: MatrixMut<'a, T>,
        workspace: &'a mut DeviceMemory<T>,
        dev_info: &'a mut DeviceMemory<i32>,
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

impl<'a, TV, TTau, TT> LarftArgs<'a, TV, TTau, TT> {
    /// Groups the reflector matrix, tau vector, output factor, and workspace.
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

impl<'a, TA, TTau> GeqrfArgs<'a, TA, TTau> {
    /// Groups the matrix, tau vector, workspace, and info buffer.
    pub const fn new(
        a: MatrixMut<'a, TA>,
        tau: VectorMut<'a, TTau>,
        workspace: ByteWorkspaceMut<'a>,
        dev_info: &'a mut DeviceMemory<i32>,
    ) -> Self {
        Self {
            a,
            tau,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T> OrmqrArgs<'a, T> {
    /// Groups the reflector matrix, tau vector, target matrix, workspace, and
    /// info buffer.
    pub const fn new(
        reflectors: MatrixRef<'a, T>,
        tau: VectorRef<'a, T>,
        matrix: MatrixMut<'a, T>,
        workspace: &'a mut DeviceMemory<T>,
        dev_info: &'a mut DeviceMemory<i32>,
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

impl<'a, T> OrgqrArgs<'a, T> {
    /// Groups the matrix, tau vector, workspace, and info buffer.
    pub const fn new(
        a: MatrixMut<'a, T>,
        tau: VectorRef<'a, T>,
        workspace: &'a mut DeviceMemory<T>,
        dev_info: &'a mut DeviceMemory<i32>,
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
    T: GebrdScalar,
{
    /// Groups the matrix, bidiagonal outputs, reflectors, workspace, and info
    /// buffer.
    pub const fn new(
        a: MatrixMut<'a, T>,
        diagonal: VectorMut<'a, T::Real>,
        off_diagonal: VectorMut<'a, T::Real>,
        tau_q: VectorMut<'a, T>,
        tau_p: VectorMut<'a, T>,
        workspace: &'a mut DeviceMemory<T>,
        dev_info: &'a mut DeviceMemory<i32>,
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

impl<'a, T> OrgbrArgs<'a, T> {
    /// Groups the matrix, tau vector, workspace, and info buffer.
    pub const fn new(
        a: MatrixMut<'a, T>,
        tau: VectorRef<'a, T>,
        workspace: &'a mut DeviceMemory<T>,
        dev_info: &'a mut DeviceMemory<i32>,
    ) -> Self {
        Self {
            a,
            tau,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T> PotrsArgs<'a, T> {
    /// Groups the factorized matrix, right-hand sides, and info buffer needed by [`Potrs::run`].
    pub const fn new(
        factor: MatrixRef<'a, T>,
        rhs: MatrixMut<'a, T>,
        dev_info: &'a mut DeviceMemory<i32>,
    ) -> Self {
        Self {
            factor,
            rhs,
            dev_info,
        }
    }
}

impl<'a, T> PotriArgs<'a, T> {
    /// Groups the factorized matrix, workspace, and info buffer needed by [`Potri::run`].
    pub const fn new(
        factor: MatrixMut<'a, T>,
        workspace: &'a mut DeviceMemory<T>,
        dev_info: &'a mut DeviceMemory<i32>,
    ) -> Self {
        Self {
            factor,
            workspace,
            dev_info,
        }
    }
}

impl<'a, T> BatchedPotrfArgs<'a, T> {
    /// Groups the matrix pointer array and info buffer needed by [`BatchedPotrf::run`].
    pub const fn new(a: BatchedMatrixRef<'a, T>, info: &'a mut DeviceMemory<i32>) -> Self {
        Self { a, info }
    }
}

impl<'a, T> BatchedPotrsArgs<'a, T> {
    /// Groups the factor pointer array, right-hand-side pointer array, and info buffer.
    pub const fn new(
        factors: BatchedMatrixRef<'a, T>,
        rhs: BatchedVectorRef<'a, T>,
        info: &'a mut DeviceMemory<i32>,
    ) -> Self {
        Self { factors, rhs, info }
    }
}
