use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::data_type::DataType;
use singe_cusolver_sys as sys;

/// Selects the generalized eigenvalue problem type.
///
/// This corresponds to LAPACK integer `1` (`A*x = lambda*B*x`), `2`
/// (`A*B*x = lambda*x`), and `3` (`B*A*x = lambda*x`) arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EigenType {
    /// A\*x = lambda\*B\*x.
    Type1 = sys::cusolverEigType_t::CUSOLVER_EIG_TYPE_1 as _,
    /// A\*B\*x = lambda\*x.
    Type2 = sys::cusolverEigType_t::CUSOLVER_EIG_TYPE_2 as _,
    /// B\*A\*x = lambda\*x.
    Type3 = sys::cusolverEigType_t::CUSOLVER_EIG_TYPE_3 as _,
}

impl_enum_conversion!(sys::cusolverEigType_t, EigenType);

/// Selects whether to compute eigenvectors.
///
/// This corresponds to LAPACK `N` (eigenvalues only) and `V` (eigenvalues and
/// eigenvectors) arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EigenMode {
    /// Compute only eigenvalues.
    NoVector = sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_NOVECTOR as _,
    /// Compute both eigenvalues and eigenvectors.
    Vector = sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR as _,
}

impl_enum_conversion!(sys::cusolverEigMode_t, EigenMode);

/// Selection of all eigenvalues, an index interval, or a value interval for an eigensolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EigenRange {
    /// Computes all eigenvalues.
    All = sys::cusolverEigRange_t::CUSOLVER_EIG_RANGE_ALL as _,
    /// Computes eigenvalues with one-based indices in the inclusive interval `il..=iu`.
    Index = sys::cusolverEigRange_t::CUSOLVER_EIG_RANGE_I as _,
    /// Computes eigenvalues in the half-open value interval `(vl, vu]`.
    Value = sys::cusolverEigRange_t::CUSOLVER_EIG_RANGE_V as _,
}

impl_enum_conversion!(sys::cusolverEigRange_t, EigenRange);

/// Matrix norm used by native solver error and convergence calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Norm {
    /// Uses the infinity norm: the maximum absolute row sum.
    Infinity = sys::cusolverNorm_t::CUSOLVER_INF_NORM as _,
    /// Uses the largest absolute matrix element.
    Maximum = sys::cusolverNorm_t::CUSOLVER_MAX_NORM as _,
    /// Uses the one norm: the maximum absolute column sum.
    One = sys::cusolverNorm_t::CUSOLVER_ONE_NORM as _,
    /// Uses the Frobenius norm: the square root of the sum of squared element magnitudes.
    Frobenius = sys::cusolverNorm_t::CUSOLVER_FRO_NORM as _,
}

impl_enum_conversion!(sys::cusolverNorm_t, Norm);

/// Indicates which IRS refinement solver to use for a cuSOLVER operation.
/// Empirically, [`IrsRefinement::Gmres`] is often the best option.
///
/// More details about the refinement process are available in Azzam Haidar,
/// Stanimire Tomov, Jack Dongarra, and Nicholas J. Higham, "Harnessing GPU
/// tensor cores for fast FP16 arithmetic to speed up mixed-precision iterative
/// refinement solvers," SC '18.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum IrsRefinement {
    /// Solver is not set; this value is what is set when creating the `params` structure.
    /// The IRS solver returns an error if this value is used.
    NotSet = sys::cusolverIRSRefinement_t::CUSOLVER_IRS_REFINE_NOT_SET as _,
    /// No refinement solver; the IRS solver performs a factorization followed by a solve without any refinement.
    /// For example, when used with [`xgesv`](crate::irs::xgesv), this matches the non-refined solver path with the factorization carried out in the lowest precision.
    /// If both the main and lowest precision are [`PrecisionType::R64F`], this is equivalent to solving entirely in `f64`.
    None = sys::cusolverIRSRefinement_t::CUSOLVER_IRS_REFINE_NONE as _,
    /// Classical iterative refinement solver.
    /// Similar to the value used in LAPACK operations.
    Classical = sys::cusolverIRSRefinement_t::CUSOLVER_IRS_REFINE_CLASSICAL as _,
    /// Classical iterative refinement solver that uses the GMRES (Generalized Minimal Residual) internally to solve the correction equation at each iteration.
    /// The classical refinement iteration is the outer iteration, and GMRES is the inner iteration.
    /// If the tolerance of the inner GMRES is very low, for example near machine precision, the outer *classical refinement iteration* performs only one iteration and this option behaves like [`IrsRefinement::Gmres`].
    ClassicalGmres = sys::cusolverIRSRefinement_t::CUSOLVER_IRS_REFINE_CLASSICAL_GMRES as _,
    /// GMRES (Generalized Minimal Residual) based iterative refinement solver.
    /// Recent studies use GMRES as a refinement solver that can outperform classical iterative refinement.
    /// Recommended setting based on cuSOLVER experimentation.
    Gmres = sys::cusolverIRSRefinement_t::CUSOLVER_IRS_REFINE_GMRES as _,
    /// GMRES-based iterative refinement solver that uses another GMRES solve internally for the preconditioned system.
    GmresGmres = sys::cusolverIRSRefinement_t::CUSOLVER_IRS_REFINE_GMRES_GMRES as _,
    /// Uses GMRES iterative refinement without preconditioning.
    GmresNoPcond = sys::cusolverIRSRefinement_t::CUSOLVER_IRS_REFINE_GMRES_NOPCOND as _,
    /// Legacy DD precision-selection value in `cusolverIRSRefinement_t`. The explicit main- and lowest-precision configuration APIs instead accept `cusolverPrecType_t`.
    PrecDd = sys::cusolverIRSRefinement_t::CUSOLVER_PREC_DD as _,
    /// Legacy SS precision-selection value in `cusolverIRSRefinement_t`. The explicit main- and lowest-precision configuration APIs instead accept `cusolverPrecType_t`.
    PrecSs = sys::cusolverIRSRefinement_t::CUSOLVER_PREC_SS as _,
    /// Legacy SHT precision-selection value in `cusolverIRSRefinement_t`. The explicit main- and lowest-precision configuration APIs instead accept `cusolverPrecType_t`.
    PrecSht = sys::cusolverIRSRefinement_t::CUSOLVER_PREC_SHT as _,
}

impl_enum_conversion!(sys::cusolverIRSRefinement_t, IrsRefinement);

/// Native real or complex precision selector for iterative-refinement main and lowest internal precision. Supported combinations depend on the solver and device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PrecisionType {
    /// Selects real signed 8-bit integer arithmetic for iterative-refinement precision configuration.
    R8I = sys::cusolverPrecType_t::CUSOLVER_R_8I as _,
    /// Selects real unsigned 8-bit integer arithmetic for iterative-refinement precision configuration.
    R8U = sys::cusolverPrecType_t::CUSOLVER_R_8U as _,
    /// Selects real 64-bit floating-point arithmetic for iterative-refinement precision configuration.
    R64F = sys::cusolverPrecType_t::CUSOLVER_R_64F as _,
    /// Selects real 32-bit floating-point arithmetic for iterative-refinement precision configuration.
    R32F = sys::cusolverPrecType_t::CUSOLVER_R_32F as _,
    /// Selects real 16-bit floating-point arithmetic for iterative-refinement precision configuration.
    R16F = sys::cusolverPrecType_t::CUSOLVER_R_16F as _,
    /// Selects real bfloat16 arithmetic for iterative-refinement precision configuration.
    R16Bf = sys::cusolverPrecType_t::CUSOLVER_R_16BF as _,
    /// Selects real TensorFloat-32 arithmetic for iterative-refinement precision configuration.
    RTf32 = sys::cusolverPrecType_t::CUSOLVER_R_TF32 as _,
    /// Native real AP precision mode. The header exposes this selector, but the documented iterative-refinement supported-precision tables do not include it.
    RAp = sys::cusolverPrecType_t::CUSOLVER_R_AP as _,
    /// Selects complex signed 8-bit integer arithmetic for iterative-refinement precision configuration.
    C8I = sys::cusolverPrecType_t::CUSOLVER_C_8I as _,
    /// Selects complex unsigned 8-bit integer arithmetic for iterative-refinement precision configuration.
    C8U = sys::cusolverPrecType_t::CUSOLVER_C_8U as _,
    /// Selects complex 64-bit floating-point arithmetic for iterative-refinement precision configuration.
    C64F = sys::cusolverPrecType_t::CUSOLVER_C_64F as _,
    /// Selects complex 32-bit floating-point arithmetic for iterative-refinement precision configuration.
    C32F = sys::cusolverPrecType_t::CUSOLVER_C_32F as _,
    /// Selects complex 16-bit floating-point arithmetic for iterative-refinement precision configuration.
    C16F = sys::cusolverPrecType_t::CUSOLVER_C_16F as _,
    /// Selects complex bfloat16 arithmetic for iterative-refinement precision configuration.
    C16Bf = sys::cusolverPrecType_t::CUSOLVER_C_16BF as _,
    /// Selects complex TensorFloat-32 arithmetic for iterative-refinement precision configuration.
    CTf32 = sys::cusolverPrecType_t::CUSOLVER_C_TF32 as _,
    /// Native complex AP precision mode. The header exposes this selector, but the documented iterative-refinement supported-precision tables do not include it.
    CAp = sys::cusolverPrecType_t::CUSOLVER_C_AP as _,
}

impl_enum_conversion!(sys::cusolverPrecType_t, PrecisionType);

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

/// Algorithm selected by [`Params::set_adv_options`](crate::params::Params::set_adv_options).
/// The set of algorithms supported for each operation is described with that
/// operation's documentation.
///
/// The default algorithm is [`AlgorithmMode::Default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AlgorithmMode {
    /// default algorithm
    Default = sys::cusolverAlgMode_t::CUSOLVER_ALG_0 as _,
    /// Uses a single algorithm for consistent accuracy over all n.
    Algorithm1 = sys::cusolverAlgMode_t::CUSOLVER_ALG_1 as _,
    /// Selects native algorithm variant 2 for routines that support it. Algorithm availability and behavior are defined by the selected routine.
    Algorithm2 = sys::cusolverAlgMode_t::CUSOLVER_ALG_2 as _,
}

impl_enum_conversion!(sys::cusolverAlgMode_t, AlgorithmMode);

/// Specifies how the vectors which define the elementary reflectors are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum StorevMode {
    /// Columnwise.
    Columnwise = sys::cusolverStorevMode_t::CUBLAS_STOREV_COLUMNWISE as _,
    /// Rowwise.
    Rowwise = sys::cusolverStorevMode_t::CUBLAS_STOREV_ROWWISE as _,
}

impl_enum_conversion!(sys::cusolverStorevMode_t, StorevMode);

/// Specifies the order in which the elementary reflectors are multiplied to form the block reflector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DirectMode {
    /// Forward.
    Forward = sys::cusolverDirectMode_t::CUBLAS_DIRECT_FORWARD as _,
    /// Backward.
    Backward = sys::cusolverDirectMode_t::CUBLAS_DIRECT_BACKWARD as _,
}

impl_enum_conversion!(sys::cusolverDirectMode_t, DirectMode);

/// Indicates whether repeated cuSOLVER executions with the same input are
/// required to produce bitwise-identical results.
///
/// Compared with cuBLAS atomics mode, [`DeterministicMode`] covers
/// non-determinism beyond atomic operations.
///
/// Use [`Context::set_deterministic_mode`](crate::context::Context::set_deterministic_mode)
/// and [`Context::deterministic_mode`](crate::context::Context::deterministic_mode)
/// to configure and query this setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DeterministicMode {
    /// Compute deterministic results.
    Deterministic = sys::cusolverDeterministicMode_t::CUSOLVER_DETERMINISTIC_RESULTS as _,
    /// Allow non-deterministic results.
    AllowNonDeterministic =
        sys::cusolverDeterministicMode_t::CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS as _,
}

impl_enum_conversion!(sys::cusolverDeterministicMode_t, DeterministicMode);

/// Compute precision mode selected by [`set_math_mode`](crate::context::Context::set_math_mode).
///
/// The following combinations of [`MathMode`] using the bitwise OR operator are allowed.
/// Math mode selection for cuSOLVER operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MathMode {
    /// Default math mode.
    /// Tensor Cores are used whenever possible.
    Default = sys::cusolverMathMode_t::CUSOLVER_DEFAULT_MATH as _,
    /// Use FP32 emulation according to the configured emulation strategy (see [`set_emulation_strategy`](crate::context::Context::set_emulation_strategy)).
    Fp32EmulatedBf16x9 = sys::cusolverMathMode_t::CUSOLVER_FP32_EMULATED_BF16X9_MATH as _,
}

impl_enum_conversion!(sys::cusolverMathMode_t, MathMode);

/// Selects which filled part of a dense matrix is used by the operation.
///
/// This corresponds to BLAS `L`/`l` (lower) and `U`/`u` (upper) arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FillMode {
    /// The lower part of the matrix is filled.
    Lower = sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER as _,
    /// The upper part of the matrix is filled.
    Upper = sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER as _,
    /// The full matrix is filled.
    Full = sys::cublasFillMode_t::CUBLAS_FILL_MODE_FULL as _,
}

impl_enum_conversion!(sys::cublasFillMode_t, FillMode);

/// Whether a triangular matrix stores its diagonal or treats every diagonal element as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DiagonalType {
    /// Uses the stored diagonal elements of the triangular matrix.
    NonUnit = sys::cublasDiagType_t::CUBLAS_DIAG_NON_UNIT as _,
    /// Treats diagonal elements as one without referencing their stored values.
    Unit = sys::cublasDiagType_t::CUBLAS_DIAG_UNIT as _,
}

impl_enum_conversion!(sys::cublasDiagType_t, DiagonalType);

/// Side on which a matrix appears in a BLAS matrix equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SideMode {
    /// Applies the matrix factor from the left.
    Left = sys::cublasSideMode_t::CUBLAS_SIDE_LEFT as _,
    /// Applies the matrix factor from the right.
    Right = sys::cublasSideMode_t::CUBLAS_SIDE_RIGHT as _,
}

impl_enum_conversion!(sys::cublasSideMode_t, SideMode);

/// Selects the operation to perform with a dense matrix.
///
/// This corresponds to BLAS `N`/`n` (non-transpose), `T`/`t` (transpose), and
/// `C`/`c` (conjugate transpose) arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Operation {
    /// Non-transpose operation.
    NonTranspose = sys::cublasOperation_t::CUBLAS_OP_N as _,
    /// Transpose operation.
    Transpose = sys::cublasOperation_t::CUBLAS_OP_T as _,
    /// Conjugate transpose operation.
    ConjugateTranspose = sys::cublasOperation_t::CUBLAS_OP_C as _,
    /// Selects element-wise complex conjugation without transposition.
    Conjugate = sys::cublasOperation_t::CUBLAS_OP_CONJG as _,
}

impl Operation {
    /// Alias for conjugate transposition.
    pub const HERMITIAN: Self = Self::ConjugateTranspose;
}

impl_enum_conversion!(sys::cublasOperation_t, Operation);

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

/// Indicates which operation is configured by
/// [`Params::set_adv_options`](crate::params::Params::set_adv_options).
/// [`Function::Getrf`] corresponds to the `getrf` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Function {
    /// Selects the symmetric or Hermitian eigensolver.
    Syevd = sys::cusolverDnFunction_t::CUSOLVERDN_SYEVD as _,
    /// Corresponds to `Getrf`.
    Getrf = sys::cusolverDnFunction_t::CUSOLVERDN_GETRF as _,
    /// Selects Cholesky factorization when configuring dense-solver algorithm options.
    Potrf = sys::cusolverDnFunction_t::CUSOLVERDN_POTRF as _,
    /// Selects batched symmetric eigensolving when configuring dense-solver algorithm options.
    SyevBatched = sys::cusolverDnFunction_t::CUSOLVERDN_SYEVBATCHED as _,
}

impl_enum_conversion!(sys::cusolverDnFunction_t, Function);

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
