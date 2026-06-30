use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cublas_sys as sys;

/// Selects whether cuBLASXt may pin host memory passed to its routines.
///
/// This mode is used with
/// [`Context::set_pinning_memory_mode`](crate::xt::context::Context::set_pinning_memory_mode)
/// and
/// [`Context::pinning_memory_mode`](crate::xt::context::Context::pinning_memory_mode).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum PinningMemoryMode {
    /// cuBLASXt does not pin pageable host memory on behalf of the caller.
    Disabled = sys::cublasXtPinnedMemMode_t::CUBLASXT_PINNING_DISABLED as _,
    /// cuBLASXt may pin pageable host memory passed to subsequent cuBLASXt calls.
    Enabled = sys::cublasXtPinnedMemMode_t::CUBLASXT_PINNING_ENABLED as _,
}

impl_enum_conversion!(sys::cublasXtPinnedMemMode_t, PinningMemoryMode);

impl_enum_display!(PinningMemoryMode, {
    PinningMemoryMode::Disabled => "CUBLASXT_PINNING_DISABLED",
    PinningMemoryMode::Enabled => "CUBLASXT_PINNING_ENABLED",
});

/// Identifies the scalar type used by a cuBLASXt BLAS routine.
///
/// cuBLASXt uses this selector with
/// [`Context::set_cpu_ratio`](crate::xt::context::Context::set_cpu_ratio)
/// and the raw [`cublasXtSetCpuRoutine`](sys::cublasXtSetCpuRoutine) API to configure hybrid CPU/GPU
/// execution for a routine.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum OperationType {
    /// Single-precision real values.
    F32 = sys::cublasXtOpType_t::CUBLASXT_FLOAT as _,
    /// Double-precision real values.
    F64 = sys::cublasXtOpType_t::CUBLASXT_DOUBLE as _,
    /// Single-precision complex values.
    Complex32 = sys::cublasXtOpType_t::CUBLASXT_COMPLEX as _,
    /// Double-precision complex values.
    Complex64 = sys::cublasXtOpType_t::CUBLASXT_DOUBLECOMPLEX as _,
}

impl_enum_conversion!(sys::cublasXtOpType_t, OperationType);

impl_enum_display!(OperationType, {
    OperationType::F32 => "CUBLASXT_FLOAT",
    OperationType::F64 => "CUBLASXT_DOUBLE",
    OperationType::Complex32 => "CUBLASXT_COMPLEX",
    OperationType::Complex64 => "CUBLASXT_DOUBLECOMPLEX",
});

/// Identifies a BLAS3 or BLAS-like cuBLASXt routine.
///
/// cuBLASXt uses this selector with
/// [`Context::set_cpu_ratio`](crate::xt::context::Context::set_cpu_ratio)
/// and the raw [`cublasXtSetCpuRoutine`](sys::cublasXtSetCpuRoutine) API to configure hybrid CPU/GPU execution for a routine.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum BlasOperation {
    /// Matrix-matrix multiplication.
    Gemm = sys::cublasXtBlasOp_t::CUBLASXT_GEMM as _,
    /// Symmetric rank-k update.
    Syrk = sys::cublasXtBlasOp_t::CUBLASXT_SYRK as _,
    /// Hermitian rank-k update.
    Herk = sys::cublasXtBlasOp_t::CUBLASXT_HERK as _,
    /// Symmetric matrix-matrix multiplication.
    Symm = sys::cublasXtBlasOp_t::CUBLASXT_SYMM as _,
    /// Hermitian matrix-matrix multiplication.
    Hemm = sys::cublasXtBlasOp_t::CUBLASXT_HEMM as _,
    /// Triangular solve with multiple right-hand sides.
    Trsm = sys::cublasXtBlasOp_t::CUBLASXT_TRSM as _,
    /// Symmetric rank-2k update.
    Syr2k = sys::cublasXtBlasOp_t::CUBLASXT_SYR2K as _,
    /// Hermitian rank-2k update.
    Her2k = sys::cublasXtBlasOp_t::CUBLASXT_HER2K as _,
    /// Symmetric packed matrix-matrix multiplication.
    Spmm = sys::cublasXtBlasOp_t::CUBLASXT_SPMM as _,
    /// Symmetric rank-k update with separate input matrices.
    Syrkx = sys::cublasXtBlasOp_t::CUBLASXT_SYRKX as _,
    /// Hermitian rank-k update with separate input matrices.
    Herkx = sys::cublasXtBlasOp_t::CUBLASXT_HERKX as _,
    /// Triangular matrix-matrix multiplication.
    Trmm = sys::cublasXtBlasOp_t::CUBLASXT_TRMM as _,
}

impl_enum_conversion!(sys::cublasXtBlasOp_t, BlasOperation);

impl_enum_display!(BlasOperation, {
    BlasOperation::Gemm => "CUBLASXT_GEMM",
    BlasOperation::Syrk => "CUBLASXT_SYRK",
    BlasOperation::Herk => "CUBLASXT_HERK",
    BlasOperation::Symm => "CUBLASXT_SYMM",
    BlasOperation::Hemm => "CUBLASXT_HEMM",
    BlasOperation::Trsm => "CUBLASXT_TRSM",
    BlasOperation::Syr2k => "CUBLASXT_SYR2K",
    BlasOperation::Her2k => "CUBLASXT_HER2K",
    BlasOperation::Spmm => "CUBLASXT_SPMM",
    BlasOperation::Syrkx => "CUBLASXT_SYRKX",
    BlasOperation::Herkx => "CUBLASXT_HERKX",
    BlasOperation::Trmm => "CUBLASXT_TRMM",
});
