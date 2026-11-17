// Per-precision cuBLAS entry points behind one trait per routine, so the public operations are generic over the element type.
// Real-valued scalars of complex routines are passed as untyped pointers; the public signatures type them through `Element::Real`.
#![allow(
    clippy::too_many_arguments,
    reason = "each entry point mirrors one native BLAS signature"
)]

use std::ffi::c_int;

use singe_cuda::types::{Complex32, Complex64};

use crate::sys;

type Op = sys::cublasOperation_t;

type Fill = sys::cublasFillMode_t;

type Diag = sys::cublasDiagType_t;

type Side = sys::cublasSideMode_t;

// Declares a trait with one native entry point and implements it for each listed element type.
// Arguments marked `p` are pointers cast to the native element type; arguments marked `v` pass unchanged.
macro_rules! dispatch {
    ($trait:ident::$method:ident $args:tt { $($element:ty => $native:ident),+ $(,)? }) => {
        dispatch!(@declare $trait::$method $args);
        $(dispatch!(@implement $trait::$method $args $element => $native);)+
    };
    (@declare $trait:ident::$method:ident ($($kind:ident $arg:ident: $ty:ty),* $(,)?)) => {
        #[expect(
            unnameable_types,
            reason = "sealed supertrait of the element traits; users cannot name or implement it"
        )]
        pub trait $trait: Sized {
            unsafe fn $method(handle: sys::cublasHandle_t, $($arg: $ty),*) -> sys::cublasStatus_t;
        }
    };
    (@implement $trait:ident::$method:ident ($($kind:ident $arg:ident: $ty:ty),* $(,)?) $element:ty => $native:ident) => {
        impl $trait for $element {
            unsafe fn $method(handle: sys::cublasHandle_t, $($arg: $ty),*) -> sys::cublasStatus_t {
                // SAFETY: the calling operation checked every operand's extent, increment, and context against `handle`, and `Self` has the layout of the native element type.
                unsafe { sys::$native(handle, $(dispatch!(@arg $kind $arg)),*) }
            }
        }
    };
    (@arg p $arg:ident) => {
        $arg.cast()
    };
    (@arg v $arg:ident) => {
        $arg
    };
}

dispatch!(Scal::scal(v n: i64, p alpha: *const Self, p x: *mut Self, v incx: i64) {
    f32 => cublasSscal_v2_64,
    f64 => cublasDscal_v2_64,
    Complex32 => cublasCscal_v2_64,
    Complex64 => cublasZscal_v2_64,
});

dispatch!(Axpy::axpy(v n: i64, p alpha: *const Self, p x: *const Self, v incx: i64, p y: *mut Self, v incy: i64) {
    f32 => cublasSaxpy_v2_64,
    f64 => cublasDaxpy_v2_64,
    Complex32 => cublasCaxpy_v2_64,
    Complex64 => cublasZaxpy_v2_64,
});

dispatch!(CopyVector::copy(v n: i64, p x: *const Self, v incx: i64, p y: *mut Self, v incy: i64) {
    f32 => cublasScopy_v2_64,
    f64 => cublasDcopy_v2_64,
    Complex32 => cublasCcopy_v2_64,
    Complex64 => cublasZcopy_v2_64,
});

dispatch!(Swap::swap(v n: i64, p x: *mut Self, v incx: i64, p y: *mut Self, v incy: i64) {
    f32 => cublasSswap_v2_64,
    f64 => cublasDswap_v2_64,
    Complex32 => cublasCswap_v2_64,
    Complex64 => cublasZswap_v2_64,
});

dispatch!(Dot::dot(v n: i64, p x: *const Self, v incx: i64, p y: *const Self, v incy: i64, p result: *mut Self) {
    f32 => cublasSdot_v2_64,
    f64 => cublasDdot_v2_64,
    Complex32 => cublasCdotu_v2_64,
    Complex64 => cublasZdotu_v2_64,
});

dispatch!(Dotc::dotc(v n: i64, p x: *const Self, v incx: i64, p y: *const Self, v incy: i64, p result: *mut Self) {
    Complex32 => cublasCdotc_v2_64,
    Complex64 => cublasZdotc_v2_64,
});

dispatch!(Nrm2::nrm2(v n: i64, p x: *const Self, v incx: i64, p result: *mut ()) {
    f32 => cublasSnrm2_v2_64,
    f64 => cublasDnrm2_v2_64,
    Complex32 => cublasScnrm2_v2_64,
    Complex64 => cublasDznrm2_v2_64,
});

dispatch!(Asum::asum(v n: i64, p x: *const Self, v incx: i64, p result: *mut ()) {
    f32 => cublasSasum_v2_64,
    f64 => cublasDasum_v2_64,
    Complex32 => cublasScasum_v2_64,
    Complex64 => cublasDzasum_v2_64,
});

dispatch!(Iamax::iamax(v n: i64, p x: *const Self, v incx: i64, v result: *mut i64) {
    f32 => cublasIsamax_v2_64,
    f64 => cublasIdamax_v2_64,
    Complex32 => cublasIcamax_v2_64,
    Complex64 => cublasIzamax_v2_64,
});

dispatch!(Iamin::iamin(v n: i64, p x: *const Self, v incx: i64, v result: *mut i64) {
    f32 => cublasIsamin_v2_64,
    f64 => cublasIdamin_v2_64,
    Complex32 => cublasIcamin_v2_64,
    Complex64 => cublasIzamin_v2_64,
});

dispatch!(Rot::rot(v n: i64, p x: *mut Self, v incx: i64, p y: *mut Self, v incy: i64, p c: *const (), p s: *const Self) {
    f32 => cublasSrot_v2_64,
    f64 => cublasDrot_v2_64,
    Complex32 => cublasCrot_v2_64,
    Complex64 => cublasZrot_v2_64,
});

dispatch!(Rotg::rotg(p a: *mut Self, p b: *mut Self, p c: *mut (), p s: *mut Self) {
    f32 => cublasSrotg_v2,
    f64 => cublasDrotg_v2,
    Complex32 => cublasCrotg_v2,
    Complex64 => cublasZrotg_v2,
});

dispatch!(Rotm::rotm(v n: i64, p x: *mut Self, v incx: i64, p y: *mut Self, v incy: i64, p param: *const Self) {
    f32 => cublasSrotm_v2_64,
    f64 => cublasDrotm_v2_64,
});

dispatch!(Rotmg::rotmg(p d1: *mut Self, p d2: *mut Self, p x1: *mut Self, p y1: *const Self, p param: *mut Self) {
    f32 => cublasSrotmg_v2,
    f64 => cublasDrotmg_v2,
});

dispatch!(Gemv::gemv(
    v trans: Op, v m: i64, v n: i64, p alpha: *const Self, p a: *const Self, v lda: i64,
    p x: *const Self, v incx: i64, p beta: *const Self, p y: *mut Self, v incy: i64,
) {
    f32 => cublasSgemv_v2_64,
    f64 => cublasDgemv_v2_64,
    Complex32 => cublasCgemv_v2_64,
    Complex64 => cublasZgemv_v2_64,
});

dispatch!(Gbmv::gbmv(
    v trans: Op, v m: i64, v n: i64, v kl: i64, v ku: i64, p alpha: *const Self, p a: *const Self,
    v lda: i64, p x: *const Self, v incx: i64, p beta: *const Self, p y: *mut Self, v incy: i64,
) {
    f32 => cublasSgbmv_v2_64,
    f64 => cublasDgbmv_v2_64,
    Complex32 => cublasCgbmv_v2_64,
    Complex64 => cublasZgbmv_v2_64,
});

dispatch!(Trmv::trmv(v uplo: Fill, v trans: Op, v diag: Diag, v n: i64, p a: *const Self, v lda: i64, p x: *mut Self, v incx: i64) {
    f32 => cublasStrmv_v2_64,
    f64 => cublasDtrmv_v2_64,
    Complex32 => cublasCtrmv_v2_64,
    Complex64 => cublasZtrmv_v2_64,
});

dispatch!(Trsv::trsv(v uplo: Fill, v trans: Op, v diag: Diag, v n: i64, p a: *const Self, v lda: i64, p x: *mut Self, v incx: i64) {
    f32 => cublasStrsv_v2_64,
    f64 => cublasDtrsv_v2_64,
    Complex32 => cublasCtrsv_v2_64,
    Complex64 => cublasZtrsv_v2_64,
});

dispatch!(Tbmv::tbmv(v uplo: Fill, v trans: Op, v diag: Diag, v n: i64, v k: i64, p a: *const Self, v lda: i64, p x: *mut Self, v incx: i64) {
    f32 => cublasStbmv_v2_64,
    f64 => cublasDtbmv_v2_64,
    Complex32 => cublasCtbmv_v2_64,
    Complex64 => cublasZtbmv_v2_64,
});

dispatch!(Tbsv::tbsv(v uplo: Fill, v trans: Op, v diag: Diag, v n: i64, v k: i64, p a: *const Self, v lda: i64, p x: *mut Self, v incx: i64) {
    f32 => cublasStbsv_v2_64,
    f64 => cublasDtbsv_v2_64,
    Complex32 => cublasCtbsv_v2_64,
    Complex64 => cublasZtbsv_v2_64,
});

dispatch!(Tpmv::tpmv(v uplo: Fill, v trans: Op, v diag: Diag, v n: i64, p ap: *const Self, p x: *mut Self, v incx: i64) {
    f32 => cublasStpmv_v2_64,
    f64 => cublasDtpmv_v2_64,
    Complex32 => cublasCtpmv_v2_64,
    Complex64 => cublasZtpmv_v2_64,
});

dispatch!(Tpsv::tpsv(v uplo: Fill, v trans: Op, v diag: Diag, v n: i64, p ap: *const Self, p x: *mut Self, v incx: i64) {
    f32 => cublasStpsv_v2_64,
    f64 => cublasDtpsv_v2_64,
    Complex32 => cublasCtpsv_v2_64,
    Complex64 => cublasZtpsv_v2_64,
});

dispatch!(Symv::symv(
    v uplo: Fill, v n: i64, p alpha: *const Self, p a: *const Self, v lda: i64, p x: *const Self,
    v incx: i64, p beta: *const Self, p y: *mut Self, v incy: i64,
) {
    f32 => cublasSsymv_v2_64,
    f64 => cublasDsymv_v2_64,
    Complex32 => cublasCsymv_v2_64,
    Complex64 => cublasZsymv_v2_64,
});

dispatch!(Hemv::hemv(
    v uplo: Fill, v n: i64, p alpha: *const Self, p a: *const Self, v lda: i64, p x: *const Self,
    v incx: i64, p beta: *const Self, p y: *mut Self, v incy: i64,
) {
    Complex32 => cublasChemv_v2_64,
    Complex64 => cublasZhemv_v2_64,
});

dispatch!(Sbmv::sbmv(
    v uplo: Fill, v n: i64, v k: i64, p alpha: *const Self, p a: *const Self, v lda: i64,
    p x: *const Self, v incx: i64, p beta: *const Self, p y: *mut Self, v incy: i64,
) {
    f32 => cublasSsbmv_v2_64,
    f64 => cublasDsbmv_v2_64,
});

dispatch!(Hbmv::hbmv(
    v uplo: Fill, v n: i64, v k: i64, p alpha: *const Self, p a: *const Self, v lda: i64,
    p x: *const Self, v incx: i64, p beta: *const Self, p y: *mut Self, v incy: i64,
) {
    Complex32 => cublasChbmv_v2_64,
    Complex64 => cublasZhbmv_v2_64,
});

dispatch!(Spmv::spmv(
    v uplo: Fill, v n: i64, p alpha: *const Self, p ap: *const Self, p x: *const Self, v incx: i64,
    p beta: *const Self, p y: *mut Self, v incy: i64,
) {
    f32 => cublasSspmv_v2_64,
    f64 => cublasDspmv_v2_64,
});

dispatch!(Hpmv::hpmv(
    v uplo: Fill, v n: i64, p alpha: *const Self, p ap: *const Self, p x: *const Self, v incx: i64,
    p beta: *const Self, p y: *mut Self, v incy: i64,
) {
    Complex32 => cublasChpmv_v2_64,
    Complex64 => cublasZhpmv_v2_64,
});

dispatch!(Ger::ger(
    v m: i64, v n: i64, p alpha: *const Self, p x: *const Self, v incx: i64, p y: *const Self,
    v incy: i64, p a: *mut Self, v lda: i64,
) {
    f32 => cublasSger_v2_64,
    f64 => cublasDger_v2_64,
    Complex32 => cublasCgeru_v2_64,
    Complex64 => cublasZgeru_v2_64,
});

dispatch!(Gerc::gerc(
    v m: i64, v n: i64, p alpha: *const Self, p x: *const Self, v incx: i64, p y: *const Self,
    v incy: i64, p a: *mut Self, v lda: i64,
) {
    Complex32 => cublasCgerc_v2_64,
    Complex64 => cublasZgerc_v2_64,
});

dispatch!(Syr::syr(v uplo: Fill, v n: i64, p alpha: *const Self, p x: *const Self, v incx: i64, p a: *mut Self, v lda: i64) {
    f32 => cublasSsyr_v2_64,
    f64 => cublasDsyr_v2_64,
    Complex32 => cublasCsyr_v2_64,
    Complex64 => cublasZsyr_v2_64,
});

dispatch!(Her::her(v uplo: Fill, v n: i64, p alpha: *const (), p x: *const Self, v incx: i64, p a: *mut Self, v lda: i64) {
    Complex32 => cublasCher_v2_64,
    Complex64 => cublasZher_v2_64,
});

dispatch!(Spr::spr(v uplo: Fill, v n: i64, p alpha: *const Self, p x: *const Self, v incx: i64, p ap: *mut Self) {
    f32 => cublasSspr_v2_64,
    f64 => cublasDspr_v2_64,
});

dispatch!(Hpr::hpr(v uplo: Fill, v n: i64, p alpha: *const (), p x: *const Self, v incx: i64, p ap: *mut Self) {
    Complex32 => cublasChpr_v2_64,
    Complex64 => cublasZhpr_v2_64,
});

dispatch!(Syr2::syr2(
    v uplo: Fill, v n: i64, p alpha: *const Self, p x: *const Self, v incx: i64, p y: *const Self,
    v incy: i64, p a: *mut Self, v lda: i64,
) {
    f32 => cublasSsyr2_v2_64,
    f64 => cublasDsyr2_v2_64,
    Complex32 => cublasCsyr2_v2_64,
    Complex64 => cublasZsyr2_v2_64,
});

dispatch!(Her2::her2(
    v uplo: Fill, v n: i64, p alpha: *const Self, p x: *const Self, v incx: i64, p y: *const Self,
    v incy: i64, p a: *mut Self, v lda: i64,
) {
    Complex32 => cublasCher2_v2_64,
    Complex64 => cublasZher2_v2_64,
});

dispatch!(Spr2::spr2(
    v uplo: Fill, v n: i64, p alpha: *const Self, p x: *const Self, v incx: i64, p y: *const Self,
    v incy: i64, p ap: *mut Self,
) {
    f32 => cublasSspr2_v2_64,
    f64 => cublasDspr2_v2_64,
});

dispatch!(Hpr2::hpr2(
    v uplo: Fill, v n: i64, p alpha: *const Self, p x: *const Self, v incx: i64, p y: *const Self,
    v incy: i64, p ap: *mut Self,
) {
    Complex32 => cublasChpr2_v2_64,
    Complex64 => cublasZhpr2_v2_64,
});

dispatch!(Gemm::gemm(
    v transa: Op, v transb: Op, v m: i64, v n: i64, v k: i64, p alpha: *const Self,
    p a: *const Self, v lda: i64, p b: *const Self, v ldb: i64, p beta: *const Self,
    p c: *mut Self, v ldc: i64,
) {
    f32 => cublasSgemm_v2_64,
    f64 => cublasDgemm_v2_64,
    Complex32 => cublasCgemm_v2_64,
    Complex64 => cublasZgemm_v2_64,
});

dispatch!(Gemm3m::gemm3m(
    v transa: Op, v transb: Op, v m: i64, v n: i64, v k: i64, p alpha: *const Self,
    p a: *const Self, v lda: i64, p b: *const Self, v ldb: i64, p beta: *const Self,
    p c: *mut Self, v ldc: i64,
) {
    Complex32 => cublasCgemm3m_64,
    Complex64 => cublasZgemm3m_64,
});

dispatch!(Symm::symm(
    v side: Side, v uplo: Fill, v m: i64, v n: i64, p alpha: *const Self, p a: *const Self,
    v lda: i64, p b: *const Self, v ldb: i64, p beta: *const Self, p c: *mut Self, v ldc: i64,
) {
    f32 => cublasSsymm_v2_64,
    f64 => cublasDsymm_v2_64,
    Complex32 => cublasCsymm_v2_64,
    Complex64 => cublasZsymm_v2_64,
});

dispatch!(Hemm::hemm(
    v side: Side, v uplo: Fill, v m: i64, v n: i64, p alpha: *const Self, p a: *const Self,
    v lda: i64, p b: *const Self, v ldb: i64, p beta: *const Self, p c: *mut Self, v ldc: i64,
) {
    Complex32 => cublasChemm_v2_64,
    Complex64 => cublasZhemm_v2_64,
});

dispatch!(Syrk::syrk(
    v uplo: Fill, v trans: Op, v n: i64, v k: i64, p alpha: *const Self, p a: *const Self,
    v lda: i64, p beta: *const Self, p c: *mut Self, v ldc: i64,
) {
    f32 => cublasSsyrk_v2_64,
    f64 => cublasDsyrk_v2_64,
    Complex32 => cublasCsyrk_v2_64,
    Complex64 => cublasZsyrk_v2_64,
});

dispatch!(Herk::herk(
    v uplo: Fill, v trans: Op, v n: i64, v k: i64, p alpha: *const (), p a: *const Self,
    v lda: i64, p beta: *const (), p c: *mut Self, v ldc: i64,
) {
    Complex32 => cublasCherk_v2_64,
    Complex64 => cublasZherk_v2_64,
});

dispatch!(Syr2k::syr2k(
    v uplo: Fill, v trans: Op, v n: i64, v k: i64, p alpha: *const Self, p a: *const Self,
    v lda: i64, p b: *const Self, v ldb: i64, p beta: *const Self, p c: *mut Self, v ldc: i64,
) {
    f32 => cublasSsyr2k_v2_64,
    f64 => cublasDsyr2k_v2_64,
    Complex32 => cublasCsyr2k_v2_64,
    Complex64 => cublasZsyr2k_v2_64,
});

dispatch!(Her2k::her2k(
    v uplo: Fill, v trans: Op, v n: i64, v k: i64, p alpha: *const Self, p a: *const Self,
    v lda: i64, p b: *const Self, v ldb: i64, p beta: *const (), p c: *mut Self, v ldc: i64,
) {
    Complex32 => cublasCher2k_v2_64,
    Complex64 => cublasZher2k_v2_64,
});

dispatch!(Syrkx::syrkx(
    v uplo: Fill, v trans: Op, v n: i64, v k: i64, p alpha: *const Self, p a: *const Self,
    v lda: i64, p b: *const Self, v ldb: i64, p beta: *const Self, p c: *mut Self, v ldc: i64,
) {
    f32 => cublasSsyrkx_64,
    f64 => cublasDsyrkx_64,
    Complex32 => cublasCsyrkx_64,
    Complex64 => cublasZsyrkx_64,
});

dispatch!(Herkx::herkx(
    v uplo: Fill, v trans: Op, v n: i64, v k: i64, p alpha: *const Self, p a: *const Self,
    v lda: i64, p b: *const Self, v ldb: i64, p beta: *const (), p c: *mut Self, v ldc: i64,
) {
    Complex32 => cublasCherkx_64,
    Complex64 => cublasZherkx_64,
});

dispatch!(Trmm::trmm(
    v side: Side, v uplo: Fill, v trans: Op, v diag: Diag, v m: i64, v n: i64,
    p alpha: *const Self, p a: *const Self, v lda: i64, p b: *const Self, v ldb: i64,
    p c: *mut Self, v ldc: i64,
) {
    f32 => cublasStrmm_v2_64,
    f64 => cublasDtrmm_v2_64,
    Complex32 => cublasCtrmm_v2_64,
    Complex64 => cublasZtrmm_v2_64,
});

dispatch!(Trsm::trsm(
    v side: Side, v uplo: Fill, v trans: Op, v diag: Diag, v m: i64, v n: i64,
    p alpha: *const Self, p a: *const Self, v lda: i64, p b: *mut Self, v ldb: i64,
) {
    f32 => cublasStrsm_v2_64,
    f64 => cublasDtrsm_v2_64,
    Complex32 => cublasCtrsm_v2_64,
    Complex64 => cublasZtrsm_v2_64,
});

dispatch!(Geam::geam(
    v transa: Op, v transb: Op, v m: i64, v n: i64, p alpha: *const Self, p a: *const Self,
    v lda: i64, p beta: *const Self, p b: *const Self, v ldb: i64, p c: *mut Self, v ldc: i64,
) {
    f32 => cublasSgeam_64,
    f64 => cublasDgeam_64,
    Complex32 => cublasCgeam_64,
    Complex64 => cublasZgeam_64,
});

dispatch!(Dgmm::dgmm(
    v mode: Side, v m: i64, v n: i64, p a: *const Self, v lda: i64, p x: *const Self, v incx: i64,
    p c: *mut Self, v ldc: i64,
) {
    f32 => cublasSdgmm_64,
    f64 => cublasDdgmm_64,
    Complex32 => cublasCdgmm_64,
    Complex64 => cublasZdgmm_64,
});

dispatch!(Tpttr::tpttr(v uplo: Fill, v n: c_int, p ap: *const Self, p a: *mut Self, v lda: c_int) {
    f32 => cublasStpttr,
    f64 => cublasDtpttr,
    Complex32 => cublasCtpttr,
    Complex64 => cublasZtpttr,
});

dispatch!(Trttp::trttp(v uplo: Fill, v n: c_int, p a: *const Self, v lda: c_int, p ap: *mut Self) {
    f32 => cublasStrttp,
    f64 => cublasDtrttp,
    Complex32 => cublasCtrttp,
    Complex64 => cublasZtrttp,
});

dispatch!(GemmBatched::gemm_batched(
    v transa: Op, v transb: Op, v m: i64, v n: i64, v k: i64, p alpha: *const Self,
    p a: *const *const Self, v lda: i64, p b: *const *const Self, v ldb: i64,
    p beta: *const Self, p c: *const *mut Self, v ldc: i64, v batch_count: i64,
) {
    f32 => cublasSgemmBatched_64,
    f64 => cublasDgemmBatched_64,
    Complex32 => cublasCgemmBatched_64,
    Complex64 => cublasZgemmBatched_64,
});

dispatch!(GemmStridedBatched::gemm_strided_batched(
    v transa: Op, v transb: Op, v m: i64, v n: i64, v k: i64, p alpha: *const Self,
    p a: *const Self, v lda: i64, v stride_a: i64, p b: *const Self, v ldb: i64, v stride_b: i64,
    p beta: *const Self, p c: *mut Self, v ldc: i64, v stride_c: i64, v batch_count: i64,
) {
    f32 => cublasSgemmStridedBatched_64,
    f64 => cublasDgemmStridedBatched_64,
    Complex32 => cublasCgemmStridedBatched_64,
    Complex64 => cublasZgemmStridedBatched_64,
});

dispatch!(TrsmBatched::trsm_batched(
    v side: Side, v uplo: Fill, v trans: Op, v diag: Diag, v m: i64, v n: i64,
    p alpha: *const Self, p a: *const *const Self, v lda: i64, p b: *const *mut Self, v ldb: i64,
    v batch_count: i64,
) {
    f32 => cublasStrsmBatched_64,
    f64 => cublasDtrsmBatched_64,
    Complex32 => cublasCtrsmBatched_64,
    Complex64 => cublasZtrsmBatched_64,
});

dispatch!(GemmGroupedBatched::gemm_grouped_batched(
    v transa: *const Op, v transb: *const Op, v m: *const i64, v n: *const i64, v k: *const i64,
    p alpha: *const Self, p a: *const *const Self, v lda: *const i64, p b: *const *const Self,
    v ldb: *const i64, p beta: *const Self, p c: *const *mut Self, v ldc: *const i64,
    v group_count: i64, v group_size: *const i64,
) {
    f32 => cublasSgemmGroupedBatched_64,
    f64 => cublasDgemmGroupedBatched_64,
});
