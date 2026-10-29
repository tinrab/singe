#![expect(
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "BLAS and factorization operand names follow the standard mathematical API"
)]
use singe_cuda::{
    types::{Complex32, Complex64},
    view::{DeviceOutputView, DeviceRepr},
};

use crate::{
    context::Context,
    dense::legacy::{
        bidiagonal::{
            cgebrd, cgebrd_buffer_size, cungbr, cungbr_buffer_size, dgebrd, dgebrd_buffer_size,
            dorgbr, dorgbr_buffer_size, sgebrd, sgebrd_buffer_size, sorgbr, sorgbr_buffer_size,
            zgebrd, zgebrd_buffer_size, zungbr, zungbr_buffer_size,
        },
        cholesky::{
            cpotrf_batched, cpotri, cpotri_buffer_size, cpotrs_batched, dpotrf_batched, dpotri,
            dpotri_buffer_size, dpotrs_batched, spotrf_batched, spotri, spotri_buffer_size,
            spotrs_batched, zpotrf_batched, zpotri, zpotri_buffer_size, zpotrs_batched,
        },
        qr::{
            cungqr, cungqr_buffer_size, cunmqr, cunmqr_buffer_size, dorgqr, dorgqr_buffer_size,
            dormqr, dormqr_buffer_size, sorgqr, sorgqr_buffer_size, sormqr, sormqr_buffer_size,
            zungqr, zungqr_buffer_size, zunmqr, zunmqr_buffer_size,
        },
        sytrf::{
            csytrf, csytrf_buffer_size, dsytrf, dsytrf_buffer_size, ssytrf, ssytrf_buffer_size,
            zsytrf, zsytrf_buffer_size,
        },
        tridiagonal::{
            chetrd, chetrd_buffer_size, cungtr, cungtr_buffer_size, cunmtr, cunmtr_buffer_size,
            dorgtr, dorgtr_buffer_size, dormtr, dormtr_buffer_size, dsytrd, dsytrd_buffer_size,
            sorgtr, sorgtr_buffer_size, sormtr, sormtr_buffer_size, ssytrd, ssytrd_buffer_size,
            zhetrd, zhetrd_buffer_size, zungtr, zungtr_buffer_size, zunmtr, zunmtr_buffer_size,
        },
    },
    error::Result,
    layout::{
        BatchedMatrixRef, BatchedVectorRef, MatrixLayout, MatrixMut, MatrixRef, VectorOutput,
        VectorRef,
    },
    types::{FillMode, Operation, SideMode},
};

mod potrf_batched_scalar {
    use super::{
        BatchedMatrixRef, Complex32, Complex64, Context, DeviceOutputView, DeviceRepr, FillMode,
        Result, cpotrf_batched, dpotrf_batched, spotrf_batched, zpotrf_batched,
    };

    pub trait Sealed: DeviceRepr {
        unsafe fn potrf_batched(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            info: DeviceOutputView<'_, i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        unsafe fn potrf_batched(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe { spotrf_batched(context, fill_mode, n, a, info) }
        }
    }

    impl Sealed for f64 {
        unsafe fn potrf_batched(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe { dpotrf_batched(context, fill_mode, n, a, info) }
        }
    }

    impl Sealed for Complex32 {
        unsafe fn potrf_batched(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe { cpotrf_batched(context, fill_mode, n, a, info) }
        }
    }

    impl Sealed for Complex64 {
        unsafe fn potrf_batched(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe { zpotrf_batched(context, fill_mode, n, a, info) }
        }
    }
}

/// Supported scalar types for batched Cholesky factorization.
///
/// This sealed trait dispatches `f32`, `f64`, and their complex counterparts to the corresponding native entry points.
pub trait PotrfBatchedScalar: potrf_batched_scalar::Sealed {}

impl PotrfBatchedScalar for f32 {}

impl PotrfBatchedScalar for f64 {}

impl PotrfBatchedScalar for Complex32 {}

impl PotrfBatchedScalar for Complex64 {}

mod potrs_batched_scalar {
    use super::{
        BatchedMatrixRef, BatchedVectorRef, Complex32, Complex64, Context, DeviceOutputView,
        DeviceRepr, FillMode, Result, cpotrs_batched, dpotrs_batched, spotrs_batched,
        zpotrs_batched,
    };

    pub trait Sealed: DeviceRepr {
        unsafe fn potrs_batched(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            b: BatchedVectorRef<'_, Self>,
            info: DeviceOutputView<'_, i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        unsafe fn potrs_batched(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            b: BatchedVectorRef<'_, Self>,
            info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe { spotrs_batched(context, fill_mode, n, a, b, info) }
        }
    }

    impl Sealed for f64 {
        unsafe fn potrs_batched(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            b: BatchedVectorRef<'_, Self>,
            info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe { dpotrs_batched(context, fill_mode, n, a, b, info) }
        }
    }

    impl Sealed for Complex32 {
        unsafe fn potrs_batched(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            b: BatchedVectorRef<'_, Self>,
            info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe { cpotrs_batched(context, fill_mode, n, a, b, info) }
        }
    }

    impl Sealed for Complex64 {
        unsafe fn potrs_batched(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            b: BatchedVectorRef<'_, Self>,
            info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe { zpotrs_batched(context, fill_mode, n, a, b, info) }
        }
    }
}

/// Supported scalar types for batched Cholesky solves.
///
/// This sealed trait dispatches `f32`, `f64`, and their complex counterparts to the corresponding native entry points.
pub trait PotrsBatchedScalar: potrs_batched_scalar::Sealed {}

impl PotrsBatchedScalar for f32 {}

impl PotrsBatchedScalar for f64 {}

impl PotrsBatchedScalar for Complex32 {}

impl PotrsBatchedScalar for Complex64 {}

mod potri_scalar {
    use super::{
        Complex32, Complex64, Context, DeviceOutputView, DeviceRepr, FillMode, MatrixLayout,
        MatrixMut, Result, cpotri, cpotri_buffer_size, dpotri, dpotri_buffer_size, spotri,
        spotri_buffer_size, zpotri, zpotri_buffer_size,
    };

    pub trait Sealed: DeviceRepr {
        fn potri_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixLayout,
        ) -> Result<usize>
        where
            Self: Sized;

        unsafe fn potri(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn potri_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixLayout,
        ) -> Result<usize> {
            spotri_buffer_size(context, fill_mode, n, factor.leading_dimension)
        }

        unsafe fn potri(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                spotri(
                    context,
                    fill_mode,
                    n,
                    factor.data,
                    factor.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for f64 {
        fn potri_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixLayout,
        ) -> Result<usize> {
            dpotri_buffer_size(context, fill_mode, n, factor.leading_dimension)
        }

        unsafe fn potri(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                dpotri(
                    context,
                    fill_mode,
                    n,
                    factor.data,
                    factor.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex32 {
        fn potri_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixLayout,
        ) -> Result<usize> {
            cpotri_buffer_size(context, fill_mode, n, factor.leading_dimension)
        }

        unsafe fn potri(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                cpotri(
                    context,
                    fill_mode,
                    n,
                    factor.data,
                    factor.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex64 {
        fn potri_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixLayout,
        ) -> Result<usize> {
            zpotri_buffer_size(context, fill_mode, n, factor.leading_dimension)
        }

        unsafe fn potri(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                zpotri(
                    context,
                    fill_mode,
                    n,
                    factor.data,
                    factor.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }
}

/// Supported scalar types for Cholesky inversion.
///
/// This sealed trait dispatches `f32`, `f64`, and their complex counterparts to the corresponding native entry points.
pub trait PotriScalar: potri_scalar::Sealed {}

impl PotriScalar for f32 {}

impl PotriScalar for f64 {}

impl PotriScalar for Complex32 {}

impl PotriScalar for Complex64 {}

mod ormqr_scalar {
    use super::{
        Complex32, Complex64, Context, DeviceOutputView, DeviceRepr, MatrixLayout, MatrixMut,
        MatrixRef, Operation, Result, SideMode, VectorRef, cunmqr, cunmqr_buffer_size, dormqr,
        dormqr_buffer_size, sormqr, sormqr_buffer_size, zunmqr, zunmqr_buffer_size,
    };

    pub trait Sealed: DeviceRepr {
        fn ormqr_buffer_size(
            context: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixLayout,
            matrix: MatrixLayout,
        ) -> Result<usize>
        where
            Self: Sized;

        unsafe fn ormqr(
            context: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn ormqr_buffer_size(
            context: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixLayout,
            matrix: MatrixLayout,
        ) -> Result<usize> {
            sormqr_buffer_size(
                context,
                side,
                operation,
                m,
                n,
                k,
                reflectors.leading_dimension,
                matrix.leading_dimension,
            )
        }

        unsafe fn ormqr(
            context: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                sormqr(
                    context,
                    side,
                    operation,
                    m,
                    n,
                    k,
                    reflectors.data,
                    reflectors.leading_dimension,
                    tau.data,
                    matrix.data,
                    matrix.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for f64 {
        fn ormqr_buffer_size(
            context: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixLayout,
            matrix: MatrixLayout,
        ) -> Result<usize> {
            dormqr_buffer_size(
                context,
                side,
                operation,
                m,
                n,
                k,
                reflectors.leading_dimension,
                matrix.leading_dimension,
            )
        }

        unsafe fn ormqr(
            context: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                dormqr(
                    context,
                    side,
                    operation,
                    m,
                    n,
                    k,
                    reflectors.data,
                    reflectors.leading_dimension,
                    tau.data,
                    matrix.data,
                    matrix.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex32 {
        fn ormqr_buffer_size(
            context: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixLayout,
            matrix: MatrixLayout,
        ) -> Result<usize> {
            cunmqr_buffer_size(
                context,
                side,
                operation,
                m,
                n,
                k,
                reflectors.leading_dimension,
                matrix.leading_dimension,
            )
        }

        unsafe fn ormqr(
            context: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                cunmqr(
                    context,
                    side,
                    operation,
                    m,
                    n,
                    k,
                    reflectors.data,
                    reflectors.leading_dimension,
                    tau.data,
                    matrix.data,
                    matrix.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex64 {
        fn ormqr_buffer_size(
            context: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixLayout,
            matrix: MatrixLayout,
        ) -> Result<usize> {
            zunmqr_buffer_size(
                context,
                side,
                operation,
                m,
                n,
                k,
                reflectors.leading_dimension,
                matrix.leading_dimension,
            )
        }

        unsafe fn ormqr(
            context: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                zunmqr(
                    context,
                    side,
                    operation,
                    m,
                    n,
                    k,
                    reflectors.data,
                    reflectors.leading_dimension,
                    tau.data,
                    matrix.data,
                    matrix.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }
}

/// Supported scalar types for application of QR reflectors.
///
/// This sealed trait dispatches `f32`, `f64`, and their complex counterparts to the corresponding native entry points.
pub trait OrmqrScalar: ormqr_scalar::Sealed {}

impl OrmqrScalar for f32 {}

impl OrmqrScalar for f64 {}

impl OrmqrScalar for Complex32 {}

impl OrmqrScalar for Complex64 {}

mod orgqr_scalar {
    use super::{
        Complex32, Complex64, Context, DeviceOutputView, DeviceRepr, MatrixLayout, MatrixMut,
        Result, VectorRef, cungqr, cungqr_buffer_size, dorgqr, dorgqr_buffer_size, sorgqr,
        sorgqr_buffer_size, zungqr, zungqr_buffer_size,
    };

    pub trait Sealed: DeviceRepr {
        fn orgqr_buffer_size(
            context: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixLayout,
        ) -> Result<usize>
        where
            Self: Sized;

        unsafe fn orgqr(
            context: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn orgqr_buffer_size(
            context: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            sorgqr_buffer_size(context, m, n, k, a.leading_dimension)
        }

        unsafe fn orgqr(
            context: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                sorgqr(
                    context,
                    m,
                    n,
                    k,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for f64 {
        fn orgqr_buffer_size(
            context: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            dorgqr_buffer_size(context, m, n, k, a.leading_dimension)
        }

        unsafe fn orgqr(
            context: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                dorgqr(
                    context,
                    m,
                    n,
                    k,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex32 {
        fn orgqr_buffer_size(
            context: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            cungqr_buffer_size(context, m, n, k, a.leading_dimension)
        }

        unsafe fn orgqr(
            context: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                cungqr(
                    context,
                    m,
                    n,
                    k,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex64 {
        fn orgqr_buffer_size(
            context: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            zungqr_buffer_size(context, m, n, k, a.leading_dimension)
        }

        unsafe fn orgqr(
            context: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                zungqr(
                    context,
                    m,
                    n,
                    k,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }
}

/// Supported scalar types for explicit QR factors.
///
/// This sealed trait dispatches `f32`, `f64`, and their complex counterparts to the corresponding native entry points.
pub trait OrgqrScalar: orgqr_scalar::Sealed {}

impl OrgqrScalar for f32 {}

impl OrgqrScalar for f64 {}

impl OrgqrScalar for Complex32 {}

impl OrgqrScalar for Complex64 {}

mod sytrf_scalar {
    use super::{
        Complex32, Complex64, Context, DeviceOutputView, DeviceRepr, FillMode, MatrixLayout,
        MatrixMut, Result, csytrf, csytrf_buffer_size, dsytrf, dsytrf_buffer_size, ssytrf,
        ssytrf_buffer_size, zsytrf, zsytrf_buffer_size,
    };

    pub trait Sealed: DeviceRepr {
        fn sytrf_buffer_size(context: &Context, n: usize, a: MatrixLayout) -> Result<usize>
        where
            Self: Sized;

        unsafe fn sytrf(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            pivots: Option<DeviceOutputView<'_, i32>>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn sytrf_buffer_size(context: &Context, n: usize, a: MatrixLayout) -> Result<usize> {
            ssytrf_buffer_size(context, n, a.leading_dimension)
        }

        unsafe fn sytrf(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            pivots: Option<DeviceOutputView<'_, i32>>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                ssytrf(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    pivots,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for f64 {
        fn sytrf_buffer_size(context: &Context, n: usize, a: MatrixLayout) -> Result<usize> {
            dsytrf_buffer_size(context, n, a.leading_dimension)
        }

        unsafe fn sytrf(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            pivots: Option<DeviceOutputView<'_, i32>>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                dsytrf(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    pivots,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex32 {
        fn sytrf_buffer_size(context: &Context, n: usize, a: MatrixLayout) -> Result<usize> {
            csytrf_buffer_size(context, n, a.leading_dimension)
        }

        unsafe fn sytrf(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            pivots: Option<DeviceOutputView<'_, i32>>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                csytrf(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    pivots,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex64 {
        fn sytrf_buffer_size(context: &Context, n: usize, a: MatrixLayout) -> Result<usize> {
            zsytrf_buffer_size(context, n, a.leading_dimension)
        }

        unsafe fn sytrf(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            pivots: Option<DeviceOutputView<'_, i32>>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                zsytrf(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    pivots,
                    workspace,
                    dev_info,
                )
            }
        }
    }
}

/// Supported scalar types for symmetric indefinite factorization.
///
/// This sealed trait dispatches `f32`, `f64`, and their complex counterparts to the corresponding native entry points.
pub trait SytrfScalar: sytrf_scalar::Sealed {}

impl SytrfScalar for f32 {}

impl SytrfScalar for f64 {}

impl SytrfScalar for Complex32 {}

impl SytrfScalar for Complex64 {}

mod sytrd_scalar {
    use super::{
        Complex32, Complex64, Context, DeviceOutputView, DeviceRepr, FillMode, MatrixLayout,
        MatrixMut, Result, VectorOutput, chetrd, chetrd_buffer_size, dsytrd, dsytrd_buffer_size,
        ssytrd, ssytrd_buffer_size, zhetrd, zhetrd_buffer_size,
    };

    pub trait Sealed: DeviceRepr {
        type Real: DeviceRepr;

        fn sytrd_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixLayout,
        ) -> Result<usize>
        where
            Self: Sized;

        unsafe fn sytrd(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorOutput<'_, Self::Real>,
            e: VectorOutput<'_, Self::Real>,
            tau: VectorOutput<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        type Real = Self;

        fn sytrd_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            ssytrd_buffer_size(context, fill_mode, n, a.leading_dimension)
        }

        unsafe fn sytrd(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorOutput<'_, Self::Real>,
            e: VectorOutput<'_, Self::Real>,
            tau: VectorOutput<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                ssytrd(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    d.data,
                    e.data,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for f64 {
        type Real = Self;

        fn sytrd_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            dsytrd_buffer_size(context, fill_mode, n, a.leading_dimension)
        }

        unsafe fn sytrd(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorOutput<'_, Self::Real>,
            e: VectorOutput<'_, Self::Real>,
            tau: VectorOutput<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                dsytrd(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    d.data,
                    e.data,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex32 {
        type Real = f32;

        fn sytrd_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            chetrd_buffer_size(context, fill_mode, n, a.leading_dimension)
        }

        unsafe fn sytrd(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorOutput<'_, Self::Real>,
            e: VectorOutput<'_, Self::Real>,
            tau: VectorOutput<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                chetrd(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    d.data,
                    e.data,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex64 {
        type Real = f64;

        fn sytrd_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            zhetrd_buffer_size(context, fill_mode, n, a.leading_dimension)
        }

        unsafe fn sytrd(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorOutput<'_, Self::Real>,
            e: VectorOutput<'_, Self::Real>,
            tau: VectorOutput<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                zhetrd(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    d.data,
                    e.data,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }
}

/// Supported scalar types for symmetric or Hermitian tridiagonal reduction.
///
/// This sealed trait dispatches `f32`, `f64`, and their complex counterparts to the corresponding native entry points.
pub trait SytrdScalar: sytrd_scalar::Sealed {}

impl SytrdScalar for f32 {}

impl SytrdScalar for f64 {}

impl SytrdScalar for Complex32 {}

impl SytrdScalar for Complex64 {}

mod orgtr_scalar {
    use super::{
        Complex32, Complex64, Context, DeviceOutputView, DeviceRepr, FillMode, MatrixLayout,
        MatrixMut, Result, VectorRef, cungtr, cungtr_buffer_size, dorgtr, dorgtr_buffer_size,
        sorgtr, sorgtr_buffer_size, zungtr, zungtr_buffer_size,
    };

    pub trait Sealed: DeviceRepr {
        fn orgtr_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixLayout,
        ) -> Result<usize>
        where
            Self: Sized;

        unsafe fn orgtr(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn orgtr_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            sorgtr_buffer_size(context, fill_mode, n, a.leading_dimension)
        }

        unsafe fn orgtr(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                sorgtr(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for f64 {
        fn orgtr_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            dorgtr_buffer_size(context, fill_mode, n, a.leading_dimension)
        }

        unsafe fn orgtr(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                dorgtr(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex32 {
        fn orgtr_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            cungtr_buffer_size(context, fill_mode, n, a.leading_dimension)
        }

        unsafe fn orgtr(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                cungtr(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex64 {
        fn orgtr_buffer_size(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            zungtr_buffer_size(context, fill_mode, n, a.leading_dimension)
        }

        unsafe fn orgtr(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                zungtr(
                    context,
                    fill_mode,
                    n,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }
}

/// Supported scalar types for explicit tridiagonalization factors.
///
/// This sealed trait dispatches `f32`, `f64`, and their complex counterparts to the corresponding native entry points.
pub trait OrgtrScalar: orgtr_scalar::Sealed {}

impl OrgtrScalar for f32 {}

impl OrgtrScalar for f64 {}

impl OrgtrScalar for Complex32 {}

impl OrgtrScalar for Complex64 {}

mod ormtr_scalar {
    use super::{
        Complex32, Complex64, Context, DeviceOutputView, DeviceRepr, FillMode, MatrixLayout,
        MatrixMut, MatrixRef, Operation, Result, SideMode, VectorRef, cunmtr, cunmtr_buffer_size,
        dormtr, dormtr_buffer_size, sormtr, sormtr_buffer_size, zunmtr, zunmtr_buffer_size,
    };

    pub trait Sealed: DeviceRepr {
        fn ormtr_buffer_size(
            context: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixLayout,
            matrix: MatrixLayout,
        ) -> Result<usize>
        where
            Self: Sized;

        unsafe fn ormtr(
            context: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn ormtr_buffer_size(
            context: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixLayout,
            matrix: MatrixLayout,
        ) -> Result<usize> {
            sormtr_buffer_size(
                context,
                side,
                fill_mode,
                operation,
                m,
                n,
                reflectors.leading_dimension,
                matrix.leading_dimension,
            )
        }

        unsafe fn ormtr(
            context: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                sormtr(
                    context,
                    side,
                    fill_mode,
                    operation,
                    m,
                    n,
                    reflectors.data,
                    reflectors.leading_dimension,
                    tau.data,
                    matrix.data,
                    matrix.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for f64 {
        fn ormtr_buffer_size(
            context: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixLayout,
            matrix: MatrixLayout,
        ) -> Result<usize> {
            dormtr_buffer_size(
                context,
                side,
                fill_mode,
                operation,
                m,
                n,
                reflectors.leading_dimension,
                matrix.leading_dimension,
            )
        }

        unsafe fn ormtr(
            context: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                dormtr(
                    context,
                    side,
                    fill_mode,
                    operation,
                    m,
                    n,
                    reflectors.data,
                    reflectors.leading_dimension,
                    tau.data,
                    matrix.data,
                    matrix.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex32 {
        fn ormtr_buffer_size(
            context: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixLayout,
            matrix: MatrixLayout,
        ) -> Result<usize> {
            cunmtr_buffer_size(
                context,
                side,
                fill_mode,
                operation,
                m,
                n,
                reflectors.leading_dimension,
                matrix.leading_dimension,
            )
        }

        unsafe fn ormtr(
            context: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                cunmtr(
                    context,
                    side,
                    fill_mode,
                    operation,
                    m,
                    n,
                    reflectors.data,
                    reflectors.leading_dimension,
                    tau.data,
                    matrix.data,
                    matrix.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex64 {
        fn ormtr_buffer_size(
            context: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixLayout,
            matrix: MatrixLayout,
        ) -> Result<usize> {
            zunmtr_buffer_size(
                context,
                side,
                fill_mode,
                operation,
                m,
                n,
                reflectors.leading_dimension,
                matrix.leading_dimension,
            )
        }

        unsafe fn ormtr(
            context: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                zunmtr(
                    context,
                    side,
                    fill_mode,
                    operation,
                    m,
                    n,
                    reflectors.data,
                    reflectors.leading_dimension,
                    tau.data,
                    matrix.data,
                    matrix.leading_dimension,
                    workspace,
                    dev_info,
                )
            }
        }
    }
}

/// Supported scalar types for application of tridiagonalization reflectors.
///
/// This sealed trait dispatches `f32`, `f64`, and their complex counterparts to the corresponding native entry points.
pub trait OrmtrScalar: ormtr_scalar::Sealed {}

impl OrmtrScalar for f32 {}

impl OrmtrScalar for f64 {}

impl OrmtrScalar for Complex32 {}

impl OrmtrScalar for Complex64 {}

mod gebrd_scalar {
    use super::{
        Complex32, Complex64, Context, DeviceOutputView, DeviceRepr, MatrixMut, Result,
        VectorOutput, cgebrd, cgebrd_buffer_size, dgebrd, dgebrd_buffer_size, sgebrd,
        sgebrd_buffer_size, zgebrd, zgebrd_buffer_size,
    };

    pub trait Sealed: DeviceRepr {
        type Real: DeviceRepr;

        fn gebrd_buffer_size(context: &Context, m: usize, n: usize) -> Result<usize>;

        unsafe fn gebrd(
            context: &Context,
            m: usize,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorOutput<'_, Self::Real>,
            e: VectorOutput<'_, Self::Real>,
            tauq: VectorOutput<'_, Self>,
            taup: VectorOutput<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        type Real = Self;

        fn gebrd_buffer_size(context: &Context, m: usize, n: usize) -> Result<usize> {
            sgebrd_buffer_size(context, m, n)
        }

        unsafe fn gebrd(
            context: &Context,
            m: usize,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorOutput<'_, Self::Real>,
            e: VectorOutput<'_, Self::Real>,
            tauq: VectorOutput<'_, Self>,
            taup: VectorOutput<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                sgebrd(
                    context,
                    m,
                    n,
                    a.data,
                    a.leading_dimension,
                    d.data,
                    e.data,
                    tauq.data,
                    taup.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for f64 {
        type Real = Self;

        fn gebrd_buffer_size(context: &Context, m: usize, n: usize) -> Result<usize> {
            dgebrd_buffer_size(context, m, n)
        }

        unsafe fn gebrd(
            context: &Context,
            m: usize,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorOutput<'_, Self::Real>,
            e: VectorOutput<'_, Self::Real>,
            tauq: VectorOutput<'_, Self>,
            taup: VectorOutput<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                dgebrd(
                    context,
                    m,
                    n,
                    a.data,
                    a.leading_dimension,
                    d.data,
                    e.data,
                    tauq.data,
                    taup.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex32 {
        type Real = f32;

        fn gebrd_buffer_size(context: &Context, m: usize, n: usize) -> Result<usize> {
            cgebrd_buffer_size(context, m, n)
        }

        unsafe fn gebrd(
            context: &Context,
            m: usize,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorOutput<'_, Self::Real>,
            e: VectorOutput<'_, Self::Real>,
            tauq: VectorOutput<'_, Self>,
            taup: VectorOutput<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                cgebrd(
                    context,
                    m,
                    n,
                    a.data,
                    a.leading_dimension,
                    d.data,
                    e.data,
                    tauq.data,
                    taup.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex64 {
        type Real = f64;

        fn gebrd_buffer_size(context: &Context, m: usize, n: usize) -> Result<usize> {
            zgebrd_buffer_size(context, m, n)
        }

        unsafe fn gebrd(
            context: &Context,
            m: usize,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorOutput<'_, Self::Real>,
            e: VectorOutput<'_, Self::Real>,
            tauq: VectorOutput<'_, Self>,
            taup: VectorOutput<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                zgebrd(
                    context,
                    m,
                    n,
                    a.data,
                    a.leading_dimension,
                    d.data,
                    e.data,
                    tauq.data,
                    taup.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }
}

/// Supported scalar types for bidiagonal reduction.
///
/// This sealed trait dispatches `f32`, `f64`, and their complex counterparts to the corresponding native entry points.
pub trait GebrdScalar: gebrd_scalar::Sealed {}

impl GebrdScalar for f32 {}

impl GebrdScalar for f64 {}

impl GebrdScalar for Complex32 {}

impl GebrdScalar for Complex64 {}

mod orgbr_scalar {
    use super::{
        Complex32, Complex64, Context, DeviceOutputView, DeviceRepr, MatrixLayout, MatrixMut,
        Result, SideMode, VectorRef, cungbr, cungbr_buffer_size, dorgbr, dorgbr_buffer_size,
        sorgbr, sorgbr_buffer_size, zungbr, zungbr_buffer_size,
    };

    pub trait Sealed: DeviceRepr {
        fn orgbr_buffer_size(
            context: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixLayout,
        ) -> Result<usize>
        where
            Self: Sized;

        unsafe fn orgbr(
            context: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn orgbr_buffer_size(
            context: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            sorgbr_buffer_size(context, side, m, n, k, a.leading_dimension)
        }

        unsafe fn orgbr(
            context: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                sorgbr(
                    context,
                    side,
                    m,
                    n,
                    k,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for f64 {
        fn orgbr_buffer_size(
            context: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            dorgbr_buffer_size(context, side, m, n, k, a.leading_dimension)
        }

        unsafe fn orgbr(
            context: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                dorgbr(
                    context,
                    side,
                    m,
                    n,
                    k,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex32 {
        fn orgbr_buffer_size(
            context: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            cungbr_buffer_size(context, side, m, n, k, a.leading_dimension)
        }

        unsafe fn orgbr(
            context: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                cungbr(
                    context,
                    side,
                    m,
                    n,
                    k,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }

    impl Sealed for Complex64 {
        fn orgbr_buffer_size(
            context: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixLayout,
        ) -> Result<usize> {
            zungbr_buffer_size(context, side, m, n, k, a.leading_dimension)
        }

        unsafe fn orgbr(
            context: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: DeviceOutputView<'_, Self>,
            dev_info: DeviceOutputView<'_, i32>,
        ) -> Result<()> {
            // SAFETY: the caller retains and orders every raw solver binding.
            unsafe {
                zungbr(
                    context,
                    side,
                    m,
                    n,
                    k,
                    a.data,
                    a.leading_dimension,
                    tau.data,
                    workspace,
                    dev_info,
                )
            }
        }
    }
}

/// Supported scalar types for explicit bidiagonalization factors.
///
/// This sealed trait dispatches `f32`, `f64`, and their complex counterparts to the corresponding native entry points.
pub trait OrgbrScalar: orgbr_scalar::Sealed {}

impl OrgbrScalar for f32 {}

impl OrgbrScalar for f64 {}

impl OrgbrScalar for Complex32 {}

impl OrgbrScalar for Complex64 {}
