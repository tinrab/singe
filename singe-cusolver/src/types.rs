//! Typed native solver options and precision selectors.

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda::data_type::DataType;
use singe_cusolver_sys as sys;

singe_core::native_enum! {
    /// Selects the generalized eigenvalue problem type.
    ///
    /// This corresponds to LAPACK integer `1` (`A*x = lambda*B*x`), `2` (`A*B*x = lambda*x`), and `3` (`B*A*x = lambda*x`) arguments.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum EigenType: sys::cusolverEigType_t {
        /// A\*x = lambda\*B\*x.
        Type1 = CUSOLVER_EIG_TYPE_1,
        /// A\*B\*x = lambda\*x.
        Type2 = CUSOLVER_EIG_TYPE_2,
        /// B\*A\*x = lambda\*x.
        Type3 = CUSOLVER_EIG_TYPE_3,
    }
}

singe_core::native_enum! {
    /// Selects whether to compute eigenvectors.
    ///
    /// This corresponds to LAPACK `N` (eigenvalues only) and `V` (eigenvalues and eigenvectors) arguments.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum EigenMode: sys::cusolverEigMode_t {
        /// Compute only eigenvalues.
        NoVector = CUSOLVER_EIG_MODE_NOVECTOR,
        /// Compute both eigenvalues and eigenvectors.
        Vector = CUSOLVER_EIG_MODE_VECTOR,
    }
}

singe_core::native_enum! {
    /// Selection of all eigenvalues, an index interval, or a value interval for an eigensolver.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum EigenRange: sys::cusolverEigRange_t {
        /// Computes all eigenvalues.
        All = CUSOLVER_EIG_RANGE_ALL,
        /// Computes eigenvalues with one-based indices in the inclusive interval `il..=iu`.
        Index = CUSOLVER_EIG_RANGE_I,
        /// Computes eigenvalues in the half-open value interval `(vl, vu]`.
        Value = CUSOLVER_EIG_RANGE_V,
    }
}

singe_core::native_enum! {
    /// Matrix norm used by native solver error and convergence calculations.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Norm: sys::cusolverNorm_t {
        /// Uses the infinity norm: the maximum absolute row sum.
        Infinity = CUSOLVER_INF_NORM,
        /// Uses the largest absolute matrix element.
        Maximum = CUSOLVER_MAX_NORM,
        /// Uses the one norm: the maximum absolute column sum.
        One = CUSOLVER_ONE_NORM,
        /// Uses the Frobenius norm: the square root of the sum of squared element magnitudes.
        Frobenius = CUSOLVER_FRO_NORM,
    }
}

singe_core::native_enum! {
    /// Indicates which IRS refinement solver to use for a cuSOLVER operation.
    /// Empirically, [`IrsRefinement::Gmres`] is often the best option.
    ///
    /// More details about the refinement process are available in Azzam Haidar, Stanimire Tomov, Jack Dongarra, and Nicholas J. Higham, "Harnessing GPU tensor cores for fast FP16 arithmetic to speed up mixed-precision iterative refinement solvers," SC '18.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum IrsRefinement: sys::cusolverIRSRefinement_t {
        /// Selects no solver, which makes IRS solves fail.
        NotSet = CUSOLVER_IRS_REFINE_NOT_SET,
        /// No refinement solver; the IRS solver performs a factorization followed by a solve without any refinement.
        /// For example, when used with [`xgesv_async`](crate::irs::xgesv_async), this matches the non-refined solver path with the factorization carried out in the lowest precision.
        /// If both the main and lowest precision are [`PrecisionType::R64F`], this is equivalent to solving entirely in `f64`.
        None = CUSOLVER_IRS_REFINE_NONE,
        /// Classical iterative refinement solver.
        /// Similar to the value used in LAPACK operations.
        Classical = CUSOLVER_IRS_REFINE_CLASSICAL,
        /// Classical iterative refinement solver that uses the GMRES (Generalized Minimal Residual) internally to solve the correction equation at each iteration.
        /// The classical refinement iteration is the outer iteration, and GMRES is the inner iteration.
        /// If the tolerance of the inner GMRES is very low, for example near machine precision, the outer *classical refinement iteration* performs only one iteration and this option behaves like [`IrsRefinement::Gmres`].
        ClassicalGmres = CUSOLVER_IRS_REFINE_CLASSICAL_GMRES,
        /// GMRES (Generalized Minimal Residual) based iterative refinement solver.
        /// Recent studies use GMRES as a refinement solver that can outperform classical iterative refinement.
        /// Recommended setting based on cuSOLVER experimentation.
        Gmres = CUSOLVER_IRS_REFINE_GMRES,
        /// GMRES-based iterative refinement solver that uses another GMRES solve internally for the preconditioned system.
        GmresGmres = CUSOLVER_IRS_REFINE_GMRES_GMRES,
        /// Uses GMRES iterative refinement without preconditioning.
        GmresNoPcond = CUSOLVER_IRS_REFINE_GMRES_NOPCOND,
        /// Selects the legacy DD precision pair; [`IrsParams::set_solver_precisions`](crate::irs::IrsParams::set_solver_precisions) takes [`PrecisionType`] values instead.
        PrecDd = CUSOLVER_PREC_DD,
        /// Selects the legacy SS precision pair; [`IrsParams::set_solver_precisions`](crate::irs::IrsParams::set_solver_precisions) takes [`PrecisionType`] values instead.
        PrecSs = CUSOLVER_PREC_SS,
        /// Selects the legacy SHT precision pair; [`IrsParams::set_solver_precisions`](crate::irs::IrsParams::set_solver_precisions) takes [`PrecisionType`] values instead.
        PrecSht = CUSOLVER_PREC_SHT,
    }
}

singe_core::native_enum! {
    /// Native real or complex precision selector for iterative-refinement main and lowest internal precision.
    /// Supported combinations depend on the solver and device.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PrecisionType: sys::cusolverPrecType_t {
        /// Selects real signed 8-bit integer arithmetic for iterative-refinement precision configuration.
        R8I = CUSOLVER_R_8I,
        /// Selects real unsigned 8-bit integer arithmetic for iterative-refinement precision configuration.
        R8U = CUSOLVER_R_8U,
        /// Selects real 64-bit floating-point arithmetic for iterative-refinement precision configuration.
        R64F = CUSOLVER_R_64F,
        /// Selects real 32-bit floating-point arithmetic for iterative-refinement precision configuration.
        R32F = CUSOLVER_R_32F,
        /// Selects real 16-bit floating-point arithmetic for iterative-refinement precision configuration.
        R16F = CUSOLVER_R_16F,
        /// Selects real bfloat16 arithmetic for iterative-refinement precision configuration.
        R16Bf = CUSOLVER_R_16BF,
        /// Selects real TensorFloat-32 arithmetic for iterative-refinement precision configuration.
        RTf32 = CUSOLVER_R_TF32,
        /// Native real AP precision mode.
        /// The header exposes this selector, but the documented iterative-refinement supported-precision tables do not include it.
        RAp = CUSOLVER_R_AP,
        /// Selects complex signed 8-bit integer arithmetic for iterative-refinement precision configuration.
        C8I = CUSOLVER_C_8I,
        /// Selects complex unsigned 8-bit integer arithmetic for iterative-refinement precision configuration.
        C8U = CUSOLVER_C_8U,
        /// Selects complex 64-bit floating-point arithmetic for iterative-refinement precision configuration.
        C64F = CUSOLVER_C_64F,
        /// Selects complex 32-bit floating-point arithmetic for iterative-refinement precision configuration.
        C32F = CUSOLVER_C_32F,
        /// Selects complex 16-bit floating-point arithmetic for iterative-refinement precision configuration.
        C16F = CUSOLVER_C_16F,
        /// Selects complex bfloat16 arithmetic for iterative-refinement precision configuration.
        C16Bf = CUSOLVER_C_16BF,
        /// Selects complex TensorFloat-32 arithmetic for iterative-refinement precision configuration.
        CTf32 = CUSOLVER_C_TF32,
        /// Native complex AP precision mode.
        /// The header exposes this selector, but the documented iterative-refinement supported-precision tables do not include it.
        CAp = CUSOLVER_C_AP,
    }
}

impl PrecisionType {
    /// Maps supported real or complex floating-point storage types to iterative-refinement precision selectors.
    ///
    /// Returns `None` for other data types; a mapped selector does not guarantee that a particular solver or device supports it.
    #[must_use]
    pub const fn from_data_type(data_type: DataType) -> Option<Self> {
        match data_type {
            DataType::F64 => Some(Self::R64F),
            DataType::F32 => Some(Self::R32F),
            DataType::F16 => Some(Self::R16F),
            DataType::Bf16 => Some(Self::R16Bf),
            DataType::ComplexF64 => Some(Self::C64F),
            DataType::ComplexF32 => Some(Self::C32F),
            DataType::ComplexF16 => Some(Self::C16F),
            DataType::ComplexBf16 => Some(Self::C16Bf),
            _ => None,
        }
    }
}

singe_core::native_enum! {
    /// Algorithm selected by [`Params::set_algorithm`](crate::params::Params::set_algorithm).
    /// The set of algorithms supported for each operation is described with that operation's documentation.
    ///
    /// The default algorithm is [`AlgorithmMode::Default`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum AlgorithmMode: sys::cusolverAlgMode_t {
        /// default algorithm
        Default = CUSOLVER_ALG_0,
        /// Uses a single algorithm for consistent accuracy over all n.
        Algorithm1 = CUSOLVER_ALG_1,
        /// Selects native algorithm variant 2 for routines that support it.
        /// Algorithm availability and behavior are defined by the selected routine.
        Algorithm2 = CUSOLVER_ALG_2,
    }
}

singe_core::native_enum! {
    /// Specifies how the vectors which define the elementary reflectors are stored.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum StorevMode: sys::cusolverStorevMode_t {
        /// Columnwise.
        Columnwise = CUBLAS_STOREV_COLUMNWISE,
        /// Rowwise.
        Rowwise = CUBLAS_STOREV_ROWWISE,
    }
}

singe_core::native_enum! {
    /// Specifies the order in which the elementary reflectors are multiplied to form the block reflector.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum DirectMode: sys::cusolverDirectMode_t {
        /// Forward.
        Forward = CUBLAS_DIRECT_FORWARD,
        /// Backward.
        Backward = CUBLAS_DIRECT_BACKWARD,
    }
}

singe_core::native_enum! {
    /// Indicates whether repeated cuSOLVER executions with the same input are required to produce bitwise-identical results.
    ///
    /// Compared with cuBLAS atomics mode, [`DeterministicMode`] covers non-determinism beyond atomic operations.
    ///
    /// Use [`Context::set_deterministic_mode`](crate::context::Context::set_deterministic_mode) and [`Context::deterministic_mode`](crate::context::Context::deterministic_mode) to configure and query this setting.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum DeterministicMode: sys::cusolverDeterministicMode_t {
        /// Compute deterministic results.
        Deterministic = CUSOLVER_DETERMINISTIC_RESULTS,
        /// Allow non-deterministic results.
        AllowNonDeterministic = CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS,
    }
}

singe_core::native_enum! {
    /// Compute precision mode selected by [`set_math_mode`](crate::context::Context::set_math_mode).
    ///
    /// [`MathMode::Fp32Fp64Emulated`] is the only combination of emulated modes cuSOLVER allows.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MathMode: sys::cusolverMathMode_t {
        /// Default math mode.
        /// Tensor Cores are used whenever possible.
        Default = CUSOLVER_DEFAULT_MATH,
        /// Use FP32 emulation according to the configured emulation strategy (see [`set_emulation_strategy`](crate::context::Context::set_emulation_strategy)).
        Fp32EmulatedBf16x9 = CUSOLVER_FP32_EMULATED_BF16X9_MATH,
        /// Use FP64 emulation according to the configured emulation strategy.
        Fp64EmulatedFixedPoint = CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH,
        /// Use both FP32 and FP64 emulation according to the configured emulation strategy.
        Fp32Fp64Emulated = CUSOLVER_FP32_FP64_EMULATED_MATH,
    }
}

singe_core::native_enum! {
    /// Selects which filled part of a dense matrix is used by the operation.
    ///
    /// This corresponds to BLAS `L`/`l` (lower) and `U`/`u` (upper) arguments.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum FillMode: sys::cublasFillMode_t {
        /// The lower part of the matrix is filled.
        Lower = CUBLAS_FILL_MODE_LOWER,
        /// The upper part of the matrix is filled.
        Upper = CUBLAS_FILL_MODE_UPPER,
        /// The full matrix is filled.
        Full = CUBLAS_FILL_MODE_FULL,
    }
}

singe_core::native_enum! {
    /// Whether a triangular matrix stores its diagonal or treats every diagonal element as one.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum DiagonalType: sys::cublasDiagType_t {
        /// Uses the stored diagonal elements of the triangular matrix.
        NonUnit = CUBLAS_DIAG_NON_UNIT,
        /// Treats diagonal elements as one without referencing their stored values.
        Unit = CUBLAS_DIAG_UNIT,
    }
}

singe_core::native_enum! {
    /// Side on which a matrix appears in a BLAS matrix equation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SideMode: sys::cublasSideMode_t {
        /// Applies the matrix factor from the left.
        Left = CUBLAS_SIDE_LEFT,
        /// Applies the matrix factor from the right.
        Right = CUBLAS_SIDE_RIGHT,
    }
}

singe_core::native_enum! {
    /// Selects the operation to perform with a dense matrix.
    ///
    /// This corresponds to BLAS `N`/`n` (non-transpose), `T`/`t` (transpose), and `C`/`c` (conjugate transpose) arguments.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Operation: sys::cublasOperation_t {
        /// Non-transpose operation.
        NonTranspose = CUBLAS_OP_N,
        /// Transpose operation.
        Transpose = CUBLAS_OP_T,
        /// Conjugate transpose operation.
        ConjugateTranspose = CUBLAS_OP_C,
        /// Selects element-wise complex conjugation without transposition.
        Conjugate = CUBLAS_OP_CONJG,
    }
}

impl Operation {
    /// Alias for conjugate transposition.
    pub const HERMITIAN: Self = Self::ConjugateTranspose;
}

/// Singular-vector output mode for full or reduced SVD routines.
///
/// The selected routine determines which combinations are supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SvdMode {
    /// Computes all singular vectors.
    All,
    /// Computes the reduced set of singular vectors.
    Some,
    /// Overwrites the input matrix with the requested singular vectors, where the routine supports it.
    Overwrite,
    /// Computes no singular vectors for this side of the decomposition.
    None,
}

impl SvdMode {
    /// Returns the native LAPACK job character as a signed byte.
    #[must_use]
    pub const fn as_raw(self) -> i8 {
        match self {
            Self::All => b'A'.cast_signed(),
            Self::Some => b'S'.cast_signed(),
            Self::Overwrite => b'O'.cast_signed(),
            Self::None => b'N'.cast_signed(),
        }
    }
}

/// Whether a truncated SVD computes the requested singular vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TruncatedSvdMode {
    /// Computes the requested truncated set of singular vectors.
    Some,
    /// Computes singular values without singular vectors.
    None,
}

impl TruncatedSvdMode {
    /// Returns the native LAPACK job character as a signed byte.
    #[must_use]
    pub const fn as_raw(self) -> i8 {
        match self {
            Self::Some => b'S'.cast_signed(),
            Self::None => b'N'.cast_signed(),
        }
    }
}

/// Columns of the singular-vector matrices that a Jacobi or polar-decomposition SVD produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SvdVectorColumns {
    /// Square U with `rows` columns and square V with `columns` columns.
    #[default]
    Full,
    /// Only the first `min(rows, columns)` columns of U and V.
    Economy,
}

impl SvdVectorColumns {
    pub(crate) const fn is_economy(self) -> bool {
        matches!(self, Self::Economy)
    }
}

/// What an iterative-refinement solver does when refinement fails to converge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IrsFallback {
    /// Solves the problem again from scratch in the main precision; the cuSOLVER default.
    #[default]
    MainPrecision,
    /// Returns the solution the refinement solver reached, which may be inaccurate.
    Disabled,
}

singe_core::native_enum! {
    /// Indicates which operation is configured by [`Params::set_algorithm`](crate::params::Params::set_algorithm).
    /// [`Function::Getrf`] corresponds to the `getrf` operation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Function: sys::cusolverDnFunction_t {
        /// Selects the symmetric or Hermitian eigensolver.
        Syevd = CUSOLVERDN_SYEVD,
        /// Corresponds to `Getrf`.
        Getrf = CUSOLVERDN_GETRF,
        /// Selects Cholesky factorization when configuring dense-solver algorithm options.
        Potrf = CUSOLVERDN_POTRF,
        /// Selects batched symmetric eigensolving when configuring dense-solver algorithm options.
        SyevBatched = CUSOLVERDN_SYEVBATCHED,
    }
}

impl_enum_display!(EigenType, {
    EigenType::Type1 => "CUSOLVER_EIG_TYPE_1",
    EigenType::Type2 => "CUSOLVER_EIG_TYPE_2",
    EigenType::Type3 => "CUSOLVER_EIG_TYPE_3",
});

impl_enum_display!(EigenMode, {
    EigenMode::NoVector => "CUSOLVER_EIG_MODE_NOVECTOR",
    EigenMode::Vector => "CUSOLVER_EIG_MODE_VECTOR",
});

impl_enum_display!(EigenRange, {
    EigenRange::All => "CUSOLVER_EIG_RANGE_ALL",
    EigenRange::Index => "CUSOLVER_EIG_RANGE_I",
    EigenRange::Value => "CUSOLVER_EIG_RANGE_V",
});

impl_enum_display!(Norm, {
    Norm::Infinity => "CUSOLVER_INF_NORM",
    Norm::Maximum => "CUSOLVER_MAX_NORM",
    Norm::One => "CUSOLVER_ONE_NORM",
    Norm::Frobenius => "CUSOLVER_FRO_NORM",
});

impl_enum_display!(IrsRefinement, {
    IrsRefinement::NotSet => "CUSOLVER_IRS_REFINE_NOT_SET",
    IrsRefinement::None => "CUSOLVER_IRS_REFINE_NONE",
    IrsRefinement::Classical => "CUSOLVER_IRS_REFINE_CLASSICAL",
    IrsRefinement::ClassicalGmres => "CUSOLVER_IRS_REFINE_CLASSICAL_GMRES",
    IrsRefinement::Gmres => "CUSOLVER_IRS_REFINE_GMRES",
    IrsRefinement::GmresGmres => "CUSOLVER_IRS_REFINE_GMRES_GMRES",
    IrsRefinement::GmresNoPcond => "CUSOLVER_IRS_REFINE_GMRES_NOPCOND",
    IrsRefinement::PrecDd => "CUSOLVER_PREC_DD",
    IrsRefinement::PrecSs => "CUSOLVER_PREC_SS",
    IrsRefinement::PrecSht => "CUSOLVER_PREC_SHT",
});

impl_enum_display!(PrecisionType, {
    PrecisionType::R8I => "CUSOLVER_R_8I",
    PrecisionType::R8U => "CUSOLVER_R_8U",
    PrecisionType::R64F => "CUSOLVER_R_64F",
    PrecisionType::R32F => "CUSOLVER_R_32F",
    PrecisionType::R16F => "CUSOLVER_R_16F",
    PrecisionType::R16Bf => "CUSOLVER_R_16BF",
    PrecisionType::RTf32 => "CUSOLVER_R_TF32",
    PrecisionType::RAp => "CUSOLVER_R_AP",
    PrecisionType::C8I => "CUSOLVER_C_8I",
    PrecisionType::C8U => "CUSOLVER_C_8U",
    PrecisionType::C64F => "CUSOLVER_C_64F",
    PrecisionType::C32F => "CUSOLVER_C_32F",
    PrecisionType::C16F => "CUSOLVER_C_16F",
    PrecisionType::C16Bf => "CUSOLVER_C_16BF",
    PrecisionType::CTf32 => "CUSOLVER_C_TF32",
    PrecisionType::CAp => "CUSOLVER_C_AP",
});

impl_enum_display!(AlgorithmMode, {
    AlgorithmMode::Default => "CUSOLVER_ALG_0",
    AlgorithmMode::Algorithm1 => "CUSOLVER_ALG_1",
    AlgorithmMode::Algorithm2 => "CUSOLVER_ALG_2",
});

impl_enum_display!(StorevMode, {
    StorevMode::Columnwise => "CUBLAS_STOREV_COLUMNWISE",
    StorevMode::Rowwise => "CUBLAS_STOREV_ROWWISE",
});

impl_enum_display!(DirectMode, {
    DirectMode::Forward => "CUBLAS_DIRECT_FORWARD",
    DirectMode::Backward => "CUBLAS_DIRECT_BACKWARD",
});

impl_enum_display!(DeterministicMode, {
    DeterministicMode::Deterministic => "CUSOLVER_DETERMINISTIC_RESULTS",
    DeterministicMode::AllowNonDeterministic => "CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS",
});

impl_enum_display!(MathMode, {
    MathMode::Default => "CUSOLVER_DEFAULT_MATH",
    MathMode::Fp32EmulatedBf16x9 => "CUSOLVER_FP32_EMULATED_BF16X9_MATH",
    MathMode::Fp64EmulatedFixedPoint => "CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH",
    MathMode::Fp32Fp64Emulated => "CUSOLVER_FP32_FP64_EMULATED_MATH",
});

impl_enum_display!(FillMode, {
    FillMode::Lower => "CUBLAS_FILL_MODE_LOWER",
    FillMode::Upper => "CUBLAS_FILL_MODE_UPPER",
    FillMode::Full => "CUBLAS_FILL_MODE_FULL",
});

impl_enum_display!(DiagonalType, {
    DiagonalType::NonUnit => "CUBLAS_DIAG_NON_UNIT",
    DiagonalType::Unit => "CUBLAS_DIAG_UNIT",
});

impl_enum_display!(SideMode, {
    SideMode::Left => "CUBLAS_SIDE_LEFT",
    SideMode::Right => "CUBLAS_SIDE_RIGHT",
});

impl_enum_display!(Operation, {
    Operation::NonTranspose => "CUBLAS_OP_N",
    Operation::Transpose => "CUBLAS_OP_T",
    Operation::ConjugateTranspose => "CUBLAS_OP_C",
    Operation::Conjugate => "CUBLAS_OP_CONJG",
});

impl_enum_display!(SvdMode, {
    SvdMode::All => "A",
    SvdMode::Some => "S",
    SvdMode::Overwrite => "O",
    SvdMode::None => "N",
});

impl_enum_display!(TruncatedSvdMode, {
    TruncatedSvdMode::Some => "S",
    TruncatedSvdMode::None => "N",
});

impl_enum_display!(Function, {
    Self::Syevd => "CUSOLVERDN_SYEVD",
    Function::Getrf => "CUSOLVERDN_GETRF",
    Function::Potrf => "CUSOLVERDN_POTRF",
    Function::SyevBatched => "CUSOLVERDN_SYEVBATCHED",
});
