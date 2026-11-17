use std::marker::PhantomData;

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    view::{DeviceOutputView, DeviceRepr, DeviceView},
};

use crate::{
    context::Context,
    data_type::SolverElement,
    dense::{
        xgeqrf_async, xgeqrf_workspace_size, xgetrf_async, xgetrf_workspace_size, xgetrs_async,
        xlarft_async, xlarft_workspace_size, xpotrf_async, xpotrf_workspace_size, xpotrs_async,
        xsytrf_async, xsytrf_workspace_size, xsytrs_async, xsytrs_workspace_size, xtrtri_async,
        xtrtri_workspace_size,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Potrf<T: DeviceRepr> {
    /// Triangle that holds the matrix data.
    pub fill_mode: FillMode,
    /// Square matrix dimension.
    pub n: usize,
    /// Compute type passed to the generic native interface; defaults to the matrix element type.
    pub compute_type: DataType,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Potrf::execute`].
#[derive(Debug)]
pub struct PotrfBindings<'a, T: DeviceRepr> {
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
    /// Triangle that holds the matrix data.
    pub fill_mode: FillMode,
    /// Square matrix dimension.
    pub n: usize,
    /// Number of right-hand sides.
    pub right_hand_sides: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Potrs::execute`].
#[derive(Debug)]
pub struct PotrsBindings<'a, T: DeviceRepr> {
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
    /// Triangle that holds the matrix data.
    pub fill_mode: FillMode,
    /// Square matrix dimension.
    pub n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Potri::execute`].
#[derive(Debug)]
pub struct PotriBindings<'a, T: DeviceRepr> {
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
    /// Triangle that holds the matrix data.
    pub fill_mode: FillMode,
    /// Square matrix dimension.
    pub n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`BatchedPotrf::execute`].
#[derive(Debug)]
pub struct BatchedPotrfBindings<'a, T: DeviceRepr> {
    /// Device pointer table for initialized square matrices, each overwritten by its Cholesky factor.
    ///
    /// The opposite triangle may be used as scratch.
    pub a: BatchedMatrixRef<'a, T>,
    /// Device output capacity for one solver status per matrix; inspect each completed status before using its factor.
    pub info: DeviceOutputView<'a, i32>,
}

/// Typed batched Cholesky solve operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedPotrs<T: DeviceRepr> {
    /// Triangle that holds the matrix data.
    pub fill_mode: FillMode,
    /// Square matrix dimension.
    pub n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`BatchedPotrs::execute`].
#[derive(Debug)]
pub struct BatchedPotrsBindings<'a, T: DeviceRepr> {
    /// Device pointer table for initialized Cholesky factors.
    ///
    /// The opposite triangle may be used as scratch.
    pub factors: BatchedMatrixRef<'a, T>,
    /// Device pointer table for initialized single right-hand sides, overwritten by solutions.
    pub rhs: BatchedVectorRef<'a, T>,
    /// Device output capacity for one status for the whole batch, rather than one per matrix.
    pub info: DeviceOutputView<'a, i32>,
}

/// Typed LU factorization operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Getrf<T: DeviceRepr> {
    /// Matrix row count.
    pub rows: usize,
    /// Matrix column count.
    pub cols: usize,
    /// Compute type passed to the generic native interface; defaults to the matrix element type.
    pub compute_type: DataType,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Getrf::execute`].
#[derive(Debug)]
pub struct GetrfBindings<'a, T: DeviceRepr> {
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
    /// Operation applied to the factorized matrix.
    pub operation: Operation,
    /// Square factor dimension.
    pub n: usize,
    /// Number of right-hand sides.
    pub right_hand_sides: usize,
    _ty: PhantomData<(TA, TB)>,
}

/// Runtime buffers for [`Getrs::execute`].
#[derive(Debug)]
pub struct GetrsBindings<'a, TA: DeviceRepr, TB: DeviceRepr = TA> {
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
    /// Triangle that holds the matrix data.
    pub fill_mode: FillMode,
    /// Diagonal type.
    pub diagonal_type: DiagonalType,
    /// Square matrix dimension.
    pub n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Trtri::execute`].
#[derive(Debug)]
pub struct TrtriBindings<'a, T: DeviceRepr> {
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
    /// Triangle that holds the matrix data.
    pub fill_mode: FillMode,
    /// Square factor dimension.
    pub n: usize,
    /// Number of right-hand sides.
    pub right_hand_sides: usize,
    _ty: PhantomData<(TA, TB)>,
}

/// Runtime buffers for [`Sytrs::execute`].
#[derive(Debug)]
pub struct SytrsBindings<'a, TA: DeviceRepr, TB: DeviceRepr = TA> {
    /// Initialized indefinite factorization in the selected triangle.
    pub factor: MatrixRef<'a, TA>,
    /// Optional initialized native 64-bit pivot data matching the factorization.
    /// `None` passes a null pointer.
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
    /// Triangle that holds the matrix data.
    pub fill_mode: FillMode,
    /// Square matrix dimension.
    pub n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Sytrf::execute`].
#[derive(Debug)]
pub struct SytrfBindings<'a, T: DeviceRepr> {
    /// Initialized square matrix overwritten in the selected triangle by its indefinite factorization.
    pub a: MatrixMut<'a, T>,
    /// Optional device capacity for `n` native 64-bit pivot entries.
    /// `None` passes a null pointer.
    pub pivots: Option<DeviceOutputView<'a, i64>>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status; inspect the completed first element before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Typed symmetric or Hermitian tridiagonal reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sytrd<T: DeviceRepr> {
    /// Triangle that holds the matrix data.
    pub fill_mode: FillMode,
    /// Square matrix dimension.
    pub n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Sytrd::execute`].
#[derive(Debug)]
pub struct SytrdBindings<'a, T>
where
    T: SolverElement,
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
    /// Triangle that holds the matrix data.
    pub fill_mode: FillMode,
    /// Square matrix dimension.
    pub n: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Orgtr::execute`].
#[derive(Debug)]
pub struct OrgtrBindings<'a, T: DeviceRepr> {
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
    /// Which side receives the tridiagonalization factor product.
    pub side: SideMode,
    /// Triangle that holds the matrix data.
    pub fill_mode: FillMode,
    /// Operation applied to the factor product.
    pub operation: Operation,
    /// Target matrix row count.
    pub rows: usize,
    /// Target matrix column count.
    pub cols: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Ormtr::execute`].
#[derive(Debug)]
pub struct OrmtrBindings<'a, T: DeviceRepr> {
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
    /// Reflector product direction.
    pub direct: DirectMode,
    /// Reflector storage mode.
    pub storev: StorevMode,
    /// Reflector order.
    pub n: usize,
    /// Number of elementary reflectors.
    pub k: usize,
    /// Compute type passed to the generic native interface; defaults to the matrix element type.
    pub compute_type: DataType,
    _ty: PhantomData<(TV, TTau, TT)>,
}

/// Runtime buffers for [`Larft::execute`].
#[derive(Debug)]
pub struct LarftBindings<'a, TV: DeviceRepr, TTau: DeviceRepr = TV, TT: DeviceRepr = TV> {
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
    /// Matrix row count.
    pub rows: usize,
    /// Matrix column count.
    pub cols: usize,
    /// Compute type passed to the generic native interface; defaults to the matrix element type.
    pub compute_type: DataType,
    _ty: PhantomData<(TA, TTau)>,
}

/// Runtime buffers for [`Geqrf::execute`].
#[derive(Debug)]
pub struct GeqrfBindings<'a, TA: DeviceRepr, TTau: DeviceRepr = TA> {
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
    /// Which side receives the reflector product.
    pub side: SideMode,
    /// Operation applied to the reflector product.
    pub operation: Operation,
    /// Output matrix row count.
    pub rows: usize,
    /// Output matrix column count.
    pub cols: usize,
    /// Number of elementary reflectors.
    pub reflectors: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Ormqr::execute`].
#[derive(Debug)]
pub struct OrmqrBindings<'a, T: DeviceRepr> {
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
    /// Output matrix row count.
    pub rows: usize,
    /// Output matrix column count.
    pub cols: usize,
    /// Number of elementary reflectors.
    pub reflectors: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Orgqr::execute`].
#[derive(Debug)]
pub struct OrgqrBindings<'a, T: DeviceRepr> {
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
    /// Matrix row count.
    pub rows: usize,
    /// Matrix column count.
    pub cols: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Gebrd::execute`].
#[derive(Debug)]
pub struct GebrdBindings<'a, T>
where
    T: SolverElement,
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
    /// Whether to form `Q` or `P`.
    pub side: SideMode,
    /// Output matrix row count.
    pub rows: usize,
    /// Output matrix column count.
    pub cols: usize,
    /// Number of elementary reflectors.
    pub reflectors: usize,
    _ty: PhantomData<T>,
}

/// Runtime buffers for [`Orgbr::execute`].
#[derive(Debug)]
pub struct OrgbrBindings<'a, T: DeviceRepr> {
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

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix shape is invalid or cuSOLVER cannot report the workspace sizes.
    pub fn workspace_size(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xpotrf_workspace_size::<T>(
            context,
            params,
            self.fill_mode,
            self.n,
            a,
            self.compute_type,
        )
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: PotrfBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            xpotrf_async(
                context,
                params,
                self.fill_mode,
                self.n,
                bindings.a,
                self.compute_type,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Computes the Cholesky factorization in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(
        &self,
        context: &Context,
        bindings: PotrfBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
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

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: PotrsBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            xpotrs_async(
                context,
                params,
                self.fill_mode,
                self.n,
                self.right_hand_sides,
                bindings.factor,
                bindings.rhs,
                bindings.dev_info,
            )
        }
    }

    /// Solves the linear system in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(
        &self,
        context: &Context,
        bindings: PotrsBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

impl<T> Potri<T>
where
    T: SolverElement,
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

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix shape is invalid or cuSOLVER cannot report the workspace size.
    pub fn workspace_len(&self, context: &Context, factor: MatrixLayout) -> Result<usize> {
        T::potri_buffer_size(context, self.fill_mode, self.n, factor)
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: PotriBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            T::potri(
                context,
                self.fill_mode,
                self.n,
                bindings.factor,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Computes the inverse in place from a Cholesky factorization.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: PotriBindings<'_, T>) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
    }
}

impl<T> BatchedPotrf<T>
where
    T: SolverElement,
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

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution), which also requires every pointer-table pointee to outlive the work.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: BatchedPotrfBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe { T::potrf_batched(context, self.fill_mode, self.n, bindings.a, bindings.info) }
    }

    /// Computes batched Cholesky factorizations in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: BatchedPotrfBindings<'_, T>) -> Result<()> {
        // SAFETY: the pointer-table constructors vouch for the pointees during the bindings' lifetime, and `sync_scope` rejects capture and waits for the work before the bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
    }
}

impl<T> BatchedPotrs<T>
where
    T: SolverElement,
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

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution), which also requires every pointer-table pointee to outlive the work.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: BatchedPotrsBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            T::potrs_batched(
                context,
                self.fill_mode,
                self.n,
                bindings.factors,
                bindings.rhs,
                bindings.info,
            )
        }
    }

    /// Solves batched linear systems in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: BatchedPotrsBindings<'_, T>) -> Result<()> {
        // SAFETY: the pointer-table constructors vouch for the pointees during the bindings' lifetime, and `sync_scope` rejects capture and waits for the work before the bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
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

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix shape is invalid or cuSOLVER cannot report the workspace sizes.
    pub fn workspace_size(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xgetrf_workspace_size::<T>(context, params, self.rows, self.cols, a, self.compute_type)
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: GetrfBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            xgetrf_async(
                context,
                params,
                self.rows,
                self.cols,
                bindings.a,
                bindings.pivots,
                self.compute_type,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Computes the LU factorization in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(
        &self,
        context: &Context,
        bindings: GetrfBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
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

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: GetrsBindings<'_, TA, TB>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            xgetrs_async(
                context,
                params,
                self.operation,
                self.n,
                self.right_hand_sides,
                bindings.factor,
                bindings.pivots,
                bindings.rhs,
                bindings.dev_info,
            )
        }
    }

    /// Solves the linear system in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(
        &self,
        context: &Context,
        bindings: GetrsBindings<'_, TA, TB>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
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

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix shape is invalid or cuSOLVER cannot report the workspace sizes.
    pub fn workspace_size(&self, context: &Context, a: MatrixLayout) -> Result<WorkspaceSizes> {
        xtrtri_workspace_size::<T>(context, self.fill_mode, self.diagonal_type, self.n, a)
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: TrtriBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            xtrtri_async(
                context,
                self.fill_mode,
                self.diagonal_type,
                self.n,
                bindings.a,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Computes the triangular inverse in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: TrtriBindings<'_, T>) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
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

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the workspace sizes.
    pub fn workspace_size(
        &self,
        context: &Context,
        factor: MatrixLayout,
        rhs: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xsytrs_workspace_size::<TA, TB>(
            context,
            self.fill_mode,
            self.n,
            self.right_hand_sides,
            factor,
            rhs,
        )
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: SytrsBindings<'_, TA, TB>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            xsytrs_async(
                context,
                self.fill_mode,
                self.n,
                self.right_hand_sides,
                bindings.factor,
                bindings.pivots,
                bindings.rhs,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Solves the symmetric or Hermitian system in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: SytrsBindings<'_, TA, TB>) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
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

    /// Returns the device and host workspace sizes before allocating buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the workspace length.
    pub fn workspace_size(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xsytrf_workspace_size::<T>(context, params, self.fill_mode, self.n, a, T::data_type())
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: SytrfBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            xsytrf_async(
                context,
                params,
                self.fill_mode,
                self.n,
                bindings.a,
                bindings.pivots,
                T::data_type(),
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Computes the indefinite factorization in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(
        &self,
        context: &Context,
        bindings: SytrfBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

impl<T> Sytrd<T>
where
    T: SolverElement,
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

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the workspace length.
    pub fn workspace_len(&self, context: &Context, a: MatrixLayout) -> Result<usize> {
        T::sytrd_buffer_size(context, self.fill_mode, self.n, a)
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: SytrdBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            T::sytrd(
                context,
                self.fill_mode,
                self.n,
                bindings.a,
                bindings.diagonal,
                bindings.off_diagonal,
                bindings.tau,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Reduces the matrix to symmetric or Hermitian tridiagonal form in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: SytrdBindings<'_, T>) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
    }
}

impl<T> Orgtr<T>
where
    T: SolverElement,
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

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the workspace length.
    pub fn workspace_len(&self, context: &Context, a: MatrixLayout) -> Result<usize> {
        T::orgtr_buffer_size(context, self.fill_mode, self.n, a)
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: OrgtrBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            T::orgtr(
                context,
                self.fill_mode,
                self.n,
                bindings.a,
                bindings.tau,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Forms the explicit tridiagonalization factor matrix in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: OrgtrBindings<'_, T>) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
    }
}

impl<T> Ormtr<T>
where
    T: SolverElement,
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

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the workspace length.
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

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: OrmtrBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            T::ormtr(
                context,
                self.side,
                self.fill_mode,
                self.operation,
                self.rows,
                self.cols,
                bindings.reflectors,
                bindings.tau,
                bindings.matrix,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Applies the tridiagonalization factor product in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: OrmtrBindings<'_, T>) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
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

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the workspace sizes.
    pub fn workspace_size(
        &self,
        context: &Context,
        params: &Params,
        v: MatrixLayout,
        t: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xlarft_workspace_size::<TV, TTau, TT>(
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

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: LarftBindings<'_, TV, TTau, TT>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            xlarft_async(
                context,
                params,
                self.direct,
                self.storev,
                self.n,
                self.k,
                bindings.v,
                bindings.tau,
                bindings.t,
                self.compute_type,
                bindings.workspace,
            )
        }
    }

    /// Forms the triangular factor in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(
        &self,
        context: &Context,
        bindings: LarftBindings<'_, TV, TTau, TT>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
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

    /// Returns the required device and host workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the workspace sizes.
    pub fn workspace_size(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xgeqrf_workspace_size::<TA, TTau>(
            context,
            params,
            self.rows,
            self.cols,
            a,
            self.compute_type,
        )
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: GeqrfBindings<'_, TA, TTau>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            xgeqrf_async(
                context,
                params,
                self.rows,
                self.cols,
                bindings.a,
                bindings.tau,
                self.compute_type,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Computes the QR factorization in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspaces are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(
        &self,
        context: &Context,
        bindings: GeqrfBindings<'_, TA, TTau>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

impl<T> Ormqr<T>
where
    T: SolverElement,
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

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the workspace length.
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

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: OrmqrBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            T::ormqr(
                context,
                self.side,
                self.operation,
                self.rows,
                self.cols,
                self.reflectors,
                bindings.reflectors,
                bindings.tau,
                bindings.matrix,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Applies the QR reflector product in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: OrmqrBindings<'_, T>) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
    }
}

impl<T> Orgqr<T>
where
    T: SolverElement,
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

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the workspace length.
    pub fn workspace_len(&self, context: &Context, a: MatrixLayout) -> Result<usize> {
        T::orgqr_buffer_size(context, self.rows, self.cols, self.reflectors, a)
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: OrgqrBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            T::orgqr(
                context,
                self.rows,
                self.cols,
                self.reflectors,
                bindings.a,
                bindings.tau,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Forms the explicit QR factor matrix in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: OrgqrBindings<'_, T>) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
    }
}

impl<T> Gebrd<T>
where
    T: SolverElement,
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

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or if cuSOLVER cannot report the workspace length.
    pub fn workspace_len(&self, context: &Context) -> Result<usize> {
        T::gebrd_buffer_size(context, self.rows, self.cols)
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: GebrdBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            T::gebrd(
                context,
                self.rows,
                self.cols,
                bindings.a,
                bindings.diagonal,
                bindings.off_diagonal,
                bindings.tau_q,
                bindings.tau_p,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Reduces the matrix to bidiagonal form in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: GebrdBindings<'_, T>) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
    }
}

impl<T> Orgbr<T>
where
    T: SolverElement,
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

    /// Returns the required legacy workspace length in elements.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or if cuSOLVER cannot report the workspace length.
    pub fn workspace_len(&self, context: &Context, a: MatrixLayout) -> Result<usize> {
        T::orgbr_buffer_size(context, self.side, self.rows, self.cols, self.reflectors, a)
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    pub unsafe fn execute_async(
        &self,
        context: &Context,
        bindings: OrgbrBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwards this method's raw execution contract unchanged; the callee validates the bindings and orders the work after their producers.
        unsafe {
            T::orgbr(
                context,
                self.side,
                self.rows,
                self.cols,
                self.reflectors,
                bindings.a,
                bindings.tau,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Forms the explicit bidiagonalization factor matrix in place.
    /// Waits for completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs or workspace are invalid, or if cuSOLVER reports a failure.
    /// Also returns an error if the context's stream is capturing or waiting for completion fails.
    pub fn execute(&self, context: &Context, bindings: OrgbrBindings<'_, T>) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
    }
}

impl<'a, T: DeviceRepr> PotrfBindings<'a, T> {
    /// Groups the matrix, workspace, and info buffer needed by [`Potrf::execute`].
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

impl<'a, T: DeviceRepr> GetrfBindings<'a, T> {
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

impl<'a, TA: DeviceRepr, TB: DeviceRepr> GetrsBindings<'a, TA, TB> {
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

impl<'a, T: DeviceRepr> TrtriBindings<'a, T> {
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

impl<'a, TA: DeviceRepr, TB: DeviceRepr> SytrsBindings<'a, TA, TB> {
    /// Groups the factorized matrix, optional pivots, right-hand sides, workspace, and info buffer.
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

impl<'a, T: DeviceRepr> SytrfBindings<'a, T> {
    /// Groups the factor matrix, optional pivot buffer, workspace, and info buffer.
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

impl<'a, T> SytrdBindings<'a, T>
where
    T: SolverElement,
{
    /// Groups the matrix, tridiagonal outputs, reflectors, workspace, and info buffer.
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

impl<'a, T: DeviceRepr> OrgtrBindings<'a, T> {
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

impl<'a, T: DeviceRepr> OrmtrBindings<'a, T> {
    /// Groups the reflector matrix, tau vector, target matrix, workspace, and info buffer.
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

impl<'a, TV: DeviceRepr, TTau: DeviceRepr, TT: DeviceRepr> LarftBindings<'a, TV, TTau, TT> {
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

impl<'a, TA: DeviceRepr, TTau: DeviceRepr> GeqrfBindings<'a, TA, TTau> {
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

impl<'a, T: DeviceRepr> OrmqrBindings<'a, T> {
    /// Groups the reflector matrix, tau vector, target matrix, workspace, and info buffer.
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

impl<'a, T: DeviceRepr> OrgqrBindings<'a, T> {
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

impl<'a, T> GebrdBindings<'a, T>
where
    T: SolverElement,
{
    /// Groups the matrix, bidiagonal outputs, reflectors, workspace, and info buffer.
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

impl<'a, T: DeviceRepr> OrgbrBindings<'a, T> {
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

impl<'a, T: DeviceRepr> PotrsBindings<'a, T> {
    /// Groups the factorized matrix, right-hand sides, and info buffer needed by [`Potrs::execute`].
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

impl<'a, T: DeviceRepr> PotriBindings<'a, T> {
    /// Groups the factorized matrix, workspace, and info buffer needed by [`Potri::execute`].
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

impl<'a, T: DeviceRepr> BatchedPotrfBindings<'a, T> {
    /// Groups the matrix pointer array and info buffer needed by [`BatchedPotrf::execute`].
    #[must_use]
    pub const fn new(a: BatchedMatrixRef<'a, T>, info: DeviceOutputView<'a, i32>) -> Self {
        Self { a, info }
    }
}

impl<'a, T: DeviceRepr> BatchedPotrsBindings<'a, T> {
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
