use std::ffi::{c_char, c_int};

use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::impl_enum_display;

use singe_cublas_sys as sys;
use singe_cuda::types::{Complex32, Complex64};

singe_core::native_enum! {
    /// Selects whether cuBLASXt may pin host memory passed to its routines.
    ///
    /// This mode is used with [`Context::set_pinning_memory_mode`](crate::xt::context::Context::set_pinning_memory_mode) and [`Context::pinning_memory_mode`](crate::xt::context::Context::pinning_memory_mode).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum PinningMemoryMode: sys::cublasXtPinnedMemMode_t {
        /// cuBLASXt does not pin pageable host memory on behalf of the caller.
        Disabled = CUBLASXT_PINNING_DISABLED,
        /// cuBLASXt may pin pageable host memory passed to subsequent cuBLASXt calls.
        Enabled = CUBLASXT_PINNING_ENABLED,
    }
}

impl_enum_display!(PinningMemoryMode, {
    PinningMemoryMode::Disabled => "CUBLASXT_PINNING_DISABLED",
    PinningMemoryMode::Enabled => "CUBLASXT_PINNING_ENABLED",
});

singe_core::native_enum! {
    /// Identifies the scalar type used by a cuBLASXt BLAS routine.
    ///
    /// cuBLASXt uses this selector with [`Context::set_cpu_ratio`](crate::xt::context::Context::set_cpu_ratio) and [`Context::set_cpu_routine`](crate::xt::context::Context::set_cpu_routine) to configure hybrid CPU/GPU execution for a routine.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum OperationType: sys::cublasXtOpType_t {
        /// Single-precision real values.
        F32 = CUBLASXT_FLOAT,
        /// Double-precision real values.
        F64 = CUBLASXT_DOUBLE,
        /// Single-precision complex values.
        Complex32 = CUBLASXT_COMPLEX,
        /// Double-precision complex values.
        Complex64 = CUBLASXT_DOUBLECOMPLEX,
    }
}

impl_enum_display!(OperationType, {
    OperationType::F32 => "CUBLASXT_FLOAT",
    OperationType::F64 => "CUBLASXT_DOUBLE",
    OperationType::Complex32 => "CUBLASXT_COMPLEX",
    OperationType::Complex64 => "CUBLASXT_DOUBLECOMPLEX",
});

singe_core::native_enum! {
    /// Identifies a BLAS3 or BLAS-like cuBLASXt routine.
    ///
    /// cuBLASXt uses this selector with [`Context::set_cpu_ratio`](crate::xt::context::Context::set_cpu_ratio) and [`Context::set_cpu_routine`](crate::xt::context::Context::set_cpu_routine) to configure hybrid CPU/GPU execution for a routine.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum BlasOperation: sys::cublasXtBlasOp_t {
        /// Matrix-matrix multiplication.
        Gemm = CUBLASXT_GEMM,
        /// Symmetric rank-k update.
        Syrk = CUBLASXT_SYRK,
        /// Hermitian rank-k update.
        Herk = CUBLASXT_HERK,
        /// Symmetric matrix-matrix multiplication.
        Symm = CUBLASXT_SYMM,
        /// Hermitian matrix-matrix multiplication.
        Hemm = CUBLASXT_HEMM,
        /// Triangular solve with multiple right-hand sides.
        Trsm = CUBLASXT_TRSM,
        /// Symmetric rank-2k update.
        Syr2k = CUBLASXT_SYR2K,
        /// Hermitian rank-2k update.
        Her2k = CUBLASXT_HER2K,
        /// Symmetric packed matrix-matrix multiplication.
        Spmm = CUBLASXT_SPMM,
        /// Symmetric rank-k update with separate input matrices.
        Syrkx = CUBLASXT_SYRKX,
        /// Hermitian rank-k update with separate input matrices.
        Herkx = CUBLASXT_HERKX,
        /// Triangular matrix-matrix multiplication.
        Trmm = CUBLASXT_TRMM,
    }
}

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

/// A host GEMM with the Fortran BLAS calling convention, such as `sgemm_`.
pub type CpuGemm<T> = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const c_int,
    n: *const c_int,
    k: *const c_int,
    alpha: *const T,
    a: *const T,
    lda: *const c_int,
    b: *const T,
    ldb: *const c_int,
    beta: *const T,
    c: *mut T,
    ldc: *const c_int,
);

/// A host GEMM routine for one element type, which cuBLASXt runs for part of the work.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum CpuRoutine {
    /// Single-precision real GEMM.
    Sgemm(CpuGemm<f32>),
    /// Double-precision real GEMM.
    Dgemm(CpuGemm<f64>),
    /// Single-precision complex GEMM.
    Cgemm(CpuGemm<Complex32>),
    /// Double-precision complex GEMM.
    Zgemm(CpuGemm<Complex64>),
}
