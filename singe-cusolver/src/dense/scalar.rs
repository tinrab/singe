use singe_cuda::{
    memory::DeviceMemory,
    types::{Complex32, Complex64},
};

use crate::{
    context::Context,
    dense::legacy::{bidiagonal::*, cholesky::*, qr::*, sytrf::*, tridiagonal::*},
    error::Result,
    layout::{BatchedMatrixRef, BatchedVectorRef, MatrixMut, MatrixRef, VectorMut, VectorRef},
    types::{FillMode, Operation, SideMode},
};

mod potrf_batched_scalar {
    use super::*;

    pub trait Sealed {
        fn potrf_batched(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            info: &mut DeviceMemory<i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn potrf_batched(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            spotrf_batched(ctx, fill_mode, n, a, info)
        }
    }

    impl Sealed for f64 {
        fn potrf_batched(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            dpotrf_batched(ctx, fill_mode, n, a, info)
        }
    }

    impl Sealed for Complex32 {
        fn potrf_batched(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            cpotrf_batched(ctx, fill_mode, n, a, info)
        }
    }

    impl Sealed for Complex64 {
        fn potrf_batched(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            zpotrf_batched(ctx, fill_mode, n, a, info)
        }
    }
}

pub trait PotrfBatchedScalar: potrf_batched_scalar::Sealed {}

impl PotrfBatchedScalar for f32 {}
impl PotrfBatchedScalar for f64 {}
impl PotrfBatchedScalar for Complex32 {}
impl PotrfBatchedScalar for Complex64 {}

mod potrs_batched_scalar {
    use super::*;

    pub trait Sealed {
        fn potrs_batched(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            b: BatchedVectorRef<'_, Self>,
            info: &mut DeviceMemory<i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn potrs_batched(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            b: BatchedVectorRef<'_, Self>,
            info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            spotrs_batched(ctx, fill_mode, n, a, b, info)
        }
    }

    impl Sealed for f64 {
        fn potrs_batched(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            b: BatchedVectorRef<'_, Self>,
            info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            dpotrs_batched(ctx, fill_mode, n, a, b, info)
        }
    }

    impl Sealed for Complex64 {
        fn potrs_batched(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: BatchedMatrixRef<'_, Self>,
            b: BatchedVectorRef<'_, Self>,
            info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            zpotrs_batched(ctx, fill_mode, n, a, b, info)
        }
    }
}

pub trait PotrsBatchedScalar: potrs_batched_scalar::Sealed {}

impl PotrsBatchedScalar for f32 {}
impl PotrsBatchedScalar for f64 {}
impl PotrsBatchedScalar for Complex64 {}

mod potri_scalar {
    use super::*;

    pub trait Sealed {
        fn potri_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
        ) -> Result<usize>
        where
            Self: Sized;

        fn potri(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn potri_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
        ) -> Result<usize> {
            spotri_buffer_size(ctx, fill_mode, n, factor.data, factor.leading_dimension)
        }

        fn potri(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            spotri(
                ctx,
                fill_mode,
                n,
                factor.data,
                factor.leading_dimension,
                workspace,
                dev_info,
            )
        }
    }

    impl Sealed for f64 {
        fn potri_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
        ) -> Result<usize> {
            dpotri_buffer_size(ctx, fill_mode, n, factor.data, factor.leading_dimension)
        }

        fn potri(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            dpotri(
                ctx,
                fill_mode,
                n,
                factor.data,
                factor.leading_dimension,
                workspace,
                dev_info,
            )
        }
    }

    impl Sealed for Complex32 {
        fn potri_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
        ) -> Result<usize> {
            cpotri_buffer_size(ctx, fill_mode, n, factor.data, factor.leading_dimension)
        }

        fn potri(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            cpotri(
                ctx,
                fill_mode,
                n,
                factor.data,
                factor.leading_dimension,
                workspace,
                dev_info,
            )
        }
    }

    impl Sealed for Complex64 {
        fn potri_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
        ) -> Result<usize> {
            zpotri_buffer_size(ctx, fill_mode, n, factor.data, factor.leading_dimension)
        }

        fn potri(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            factor: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            zpotri(
                ctx,
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

pub trait PotriScalar: potri_scalar::Sealed {}

impl PotriScalar for f32 {}
impl PotriScalar for f64 {}
impl PotriScalar for Complex32 {}
impl PotriScalar for Complex64 {}

mod ormqr_scalar {
    use super::*;

    pub trait Sealed {
        fn ormqr_buffer_size(
            ctx: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixRef<'_, Self>,
        ) -> Result<usize>
        where
            Self: Sized;

        fn ormqr(
            ctx: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn ormqr_buffer_size(
            ctx: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixRef<'_, Self>,
        ) -> Result<usize> {
            sormqr_buffer_size(
                ctx,
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
            )
        }

        fn ormqr(
            ctx: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            sormqr(
                ctx,
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

    impl Sealed for f64 {
        fn ormqr_buffer_size(
            ctx: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixRef<'_, Self>,
        ) -> Result<usize> {
            dormqr_buffer_size(
                ctx,
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
            )
        }

        fn ormqr(
            ctx: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            dormqr(
                ctx,
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

    impl Sealed for Complex32 {
        fn ormqr_buffer_size(
            ctx: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixRef<'_, Self>,
        ) -> Result<usize> {
            cunmqr_buffer_size(
                ctx,
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
            )
        }

        fn ormqr(
            ctx: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            cunmqr(
                ctx,
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

    impl Sealed for Complex64 {
        fn ormqr_buffer_size(
            ctx: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixRef<'_, Self>,
        ) -> Result<usize> {
            zunmqr_buffer_size(
                ctx,
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
            )
        }

        fn ormqr(
            ctx: &Context,
            side: SideMode,
            operation: Operation,
            m: usize,
            n: usize,
            k: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            zunmqr(
                ctx,
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

pub trait OrmqrScalar: ormqr_scalar::Sealed {}

impl OrmqrScalar for f32 {}
impl OrmqrScalar for f64 {}
impl OrmqrScalar for Complex32 {}
impl OrmqrScalar for Complex64 {}

mod orgqr_scalar {
    use super::*;

    pub trait Sealed {
        fn orgqr_buffer_size(
            ctx: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize>
        where
            Self: Sized;

        fn orgqr(
            ctx: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn orgqr_buffer_size(
            ctx: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            sorgqr_buffer_size(ctx, m, n, k, a.data, a.leading_dimension, tau.data)
        }

        fn orgqr(
            ctx: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            sorgqr(
                ctx,
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

    impl Sealed for f64 {
        fn orgqr_buffer_size(
            ctx: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            dorgqr_buffer_size(ctx, m, n, k, a.data, a.leading_dimension, tau.data)
        }

        fn orgqr(
            ctx: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            dorgqr(
                ctx,
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

    impl Sealed for Complex32 {
        fn orgqr_buffer_size(
            ctx: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            cungqr_buffer_size(ctx, m, n, k, a.data, a.leading_dimension, tau.data)
        }

        fn orgqr(
            ctx: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            cungqr(
                ctx,
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

    impl Sealed for Complex64 {
        fn orgqr_buffer_size(
            ctx: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            zungqr_buffer_size(ctx, m, n, k, a.data, a.leading_dimension, tau.data)
        }

        fn orgqr(
            ctx: &Context,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            zungqr(
                ctx,
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

pub trait OrgqrScalar: orgqr_scalar::Sealed {}

impl OrgqrScalar for f32 {}
impl OrgqrScalar for f64 {}
impl OrgqrScalar for Complex32 {}
impl OrgqrScalar for Complex64 {}

mod sytrf_scalar {
    use super::*;

    pub trait Sealed {
        fn sytrf_buffer_size(ctx: &Context, n: usize, a: MatrixMut<'_, Self>) -> Result<usize>
        where
            Self: Sized;

        fn sytrf(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            pivots: Option<&mut DeviceMemory<i32>>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn sytrf_buffer_size(ctx: &Context, n: usize, a: MatrixMut<'_, Self>) -> Result<usize> {
            ssytrf_buffer_size(ctx, n, a.data, a.leading_dimension)
        }

        fn sytrf(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            pivots: Option<&mut DeviceMemory<i32>>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            ssytrf(
                ctx,
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

    impl Sealed for f64 {
        fn sytrf_buffer_size(ctx: &Context, n: usize, a: MatrixMut<'_, Self>) -> Result<usize> {
            dsytrf_buffer_size(ctx, n, a.data, a.leading_dimension)
        }

        fn sytrf(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            pivots: Option<&mut DeviceMemory<i32>>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            dsytrf(
                ctx,
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

    impl Sealed for Complex32 {
        fn sytrf_buffer_size(ctx: &Context, n: usize, a: MatrixMut<'_, Self>) -> Result<usize> {
            csytrf_buffer_size(ctx, n, a.data, a.leading_dimension)
        }

        fn sytrf(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            pivots: Option<&mut DeviceMemory<i32>>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            csytrf(
                ctx,
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

    impl Sealed for Complex64 {
        fn sytrf_buffer_size(ctx: &Context, n: usize, a: MatrixMut<'_, Self>) -> Result<usize> {
            zsytrf_buffer_size(ctx, n, a.data, a.leading_dimension)
        }

        fn sytrf(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            pivots: Option<&mut DeviceMemory<i32>>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            zsytrf(
                ctx,
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

pub trait SytrfScalar: sytrf_scalar::Sealed {}

impl SytrfScalar for f32 {}
impl SytrfScalar for f64 {}
impl SytrfScalar for Complex32 {}
impl SytrfScalar for Complex64 {}

mod sytrd_scalar {
    use super::*;

    pub trait Sealed {
        type Real;

        fn sytrd_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixRef<'_, Self>,
            d: VectorRef<'_, Self::Real>,
            e: VectorRef<'_, Self::Real>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize>
        where
            Self: Sized;

        fn sytrd(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorMut<'_, Self::Real>,
            e: VectorMut<'_, Self::Real>,
            tau: VectorMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        type Real = f32;

        fn sytrd_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixRef<'_, Self>,
            d: VectorRef<'_, Self::Real>,
            e: VectorRef<'_, Self::Real>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            ssytrd_buffer_size(
                ctx,
                fill_mode,
                n,
                a.data,
                a.leading_dimension,
                d.data,
                e.data,
                tau.data,
            )
        }

        fn sytrd(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorMut<'_, Self::Real>,
            e: VectorMut<'_, Self::Real>,
            tau: VectorMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            ssytrd(
                ctx,
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

    impl Sealed for f64 {
        type Real = f64;

        fn sytrd_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixRef<'_, Self>,
            d: VectorRef<'_, Self::Real>,
            e: VectorRef<'_, Self::Real>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            dsytrd_buffer_size(
                ctx,
                fill_mode,
                n,
                a.data,
                a.leading_dimension,
                d.data,
                e.data,
                tau.data,
            )
        }

        fn sytrd(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorMut<'_, Self::Real>,
            e: VectorMut<'_, Self::Real>,
            tau: VectorMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            dsytrd(
                ctx,
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

    impl Sealed for Complex32 {
        type Real = f32;

        fn sytrd_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixRef<'_, Self>,
            d: VectorRef<'_, Self::Real>,
            e: VectorRef<'_, Self::Real>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            chetrd_buffer_size(
                ctx,
                fill_mode,
                n,
                a.data,
                a.leading_dimension,
                d.data,
                e.data,
                tau.data,
            )
        }

        fn sytrd(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorMut<'_, Self::Real>,
            e: VectorMut<'_, Self::Real>,
            tau: VectorMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            chetrd(
                ctx,
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

    impl Sealed for Complex64 {
        type Real = f64;

        fn sytrd_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixRef<'_, Self>,
            d: VectorRef<'_, Self::Real>,
            e: VectorRef<'_, Self::Real>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            zhetrd_buffer_size(
                ctx,
                fill_mode,
                n,
                a.data,
                a.leading_dimension,
                d.data,
                e.data,
                tau.data,
            )
        }

        fn sytrd(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorMut<'_, Self::Real>,
            e: VectorMut<'_, Self::Real>,
            tau: VectorMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            zhetrd(
                ctx,
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

pub trait SytrdScalar: sytrd_scalar::Sealed {}

impl SytrdScalar for f32 {}
impl SytrdScalar for f64 {}
impl SytrdScalar for Complex32 {}
impl SytrdScalar for Complex64 {}

mod orgtr_scalar {
    use super::*;

    pub trait Sealed {
        fn orgtr_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize>
        where
            Self: Sized;

        fn orgtr(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn orgtr_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            sorgtr_buffer_size(ctx, fill_mode, n, a.data, a.leading_dimension, tau.data)
        }

        fn orgtr(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            sorgtr(
                ctx,
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

    impl Sealed for f64 {
        fn orgtr_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            dorgtr_buffer_size(ctx, fill_mode, n, a.data, a.leading_dimension, tau.data)
        }

        fn orgtr(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            dorgtr(
                ctx,
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

    impl Sealed for Complex32 {
        fn orgtr_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            cungtr_buffer_size(ctx, fill_mode, n, a.data, a.leading_dimension, tau.data)
        }

        fn orgtr(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            cungtr(
                ctx,
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

    impl Sealed for Complex64 {
        fn orgtr_buffer_size(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            zungtr_buffer_size(ctx, fill_mode, n, a.data, a.leading_dimension, tau.data)
        }

        fn orgtr(
            ctx: &Context,
            fill_mode: FillMode,
            n: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            zungtr(
                ctx,
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

pub trait OrgtrScalar: orgtr_scalar::Sealed {}

impl OrgtrScalar for f32 {}
impl OrgtrScalar for f64 {}
impl OrgtrScalar for Complex32 {}
impl OrgtrScalar for Complex64 {}

mod ormtr_scalar {
    use super::*;

    pub trait Sealed {
        fn ormtr_buffer_size(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixRef<'_, Self>,
        ) -> Result<usize>
        where
            Self: Sized;

        fn ormtr(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixMut<'_, Self>,
            tau: VectorMut<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn ormtr_buffer_size(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixRef<'_, Self>,
        ) -> Result<usize> {
            sormtr_buffer_size(
                ctx,
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
            )
        }

        fn ormtr(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixMut<'_, Self>,
            tau: VectorMut<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            sormtr(
                ctx,
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

    impl Sealed for f64 {
        fn ormtr_buffer_size(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixRef<'_, Self>,
        ) -> Result<usize> {
            dormtr_buffer_size(
                ctx,
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
            )
        }

        fn ormtr(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixMut<'_, Self>,
            tau: VectorMut<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            dormtr(
                ctx,
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

    impl Sealed for Complex32 {
        fn ormtr_buffer_size(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixRef<'_, Self>,
        ) -> Result<usize> {
            cunmtr_buffer_size(
                ctx,
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
            )
        }

        fn ormtr(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixMut<'_, Self>,
            tau: VectorMut<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            cunmtr(
                ctx,
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

    impl Sealed for Complex64 {
        fn ormtr_buffer_size(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
            matrix: MatrixRef<'_, Self>,
        ) -> Result<usize> {
            zunmtr_buffer_size(
                ctx,
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
            )
        }

        fn ormtr(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            operation: Operation,
            m: usize,
            n: usize,
            reflectors: MatrixMut<'_, Self>,
            tau: VectorMut<'_, Self>,
            matrix: MatrixMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            zunmtr(
                ctx,
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

pub trait OrmtrScalar: ormtr_scalar::Sealed {}

impl OrmtrScalar for f32 {}
impl OrmtrScalar for f64 {}
impl OrmtrScalar for Complex32 {}
impl OrmtrScalar for Complex64 {}

mod gebrd_scalar {
    use super::*;

    pub trait Sealed {
        type Real;

        fn gebrd_buffer_size(ctx: &Context, m: usize, n: usize) -> Result<usize>;

        fn gebrd(
            ctx: &Context,
            m: usize,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorMut<'_, Self::Real>,
            e: VectorMut<'_, Self::Real>,
            tauq: VectorMut<'_, Self>,
            taup: VectorMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        type Real = f32;

        fn gebrd_buffer_size(ctx: &Context, m: usize, n: usize) -> Result<usize> {
            sgebrd_buffer_size(ctx, m, n)
        }

        fn gebrd(
            ctx: &Context,
            m: usize,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorMut<'_, Self::Real>,
            e: VectorMut<'_, Self::Real>,
            tauq: VectorMut<'_, Self>,
            taup: VectorMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            sgebrd(
                ctx,
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

    impl Sealed for f64 {
        type Real = f64;

        fn gebrd_buffer_size(ctx: &Context, m: usize, n: usize) -> Result<usize> {
            dgebrd_buffer_size(ctx, m, n)
        }

        fn gebrd(
            ctx: &Context,
            m: usize,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorMut<'_, Self::Real>,
            e: VectorMut<'_, Self::Real>,
            tauq: VectorMut<'_, Self>,
            taup: VectorMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            dgebrd(
                ctx,
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

    impl Sealed for Complex32 {
        type Real = f32;

        fn gebrd_buffer_size(ctx: &Context, m: usize, n: usize) -> Result<usize> {
            cgebrd_buffer_size(ctx, m, n)
        }

        fn gebrd(
            ctx: &Context,
            m: usize,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorMut<'_, Self::Real>,
            e: VectorMut<'_, Self::Real>,
            tauq: VectorMut<'_, Self>,
            taup: VectorMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            cgebrd(
                ctx,
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

    impl Sealed for Complex64 {
        type Real = f64;

        fn gebrd_buffer_size(ctx: &Context, m: usize, n: usize) -> Result<usize> {
            zgebrd_buffer_size(ctx, m, n)
        }

        fn gebrd(
            ctx: &Context,
            m: usize,
            n: usize,
            a: MatrixMut<'_, Self>,
            d: VectorMut<'_, Self::Real>,
            e: VectorMut<'_, Self::Real>,
            tauq: VectorMut<'_, Self>,
            taup: VectorMut<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            zgebrd(
                ctx,
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

pub trait GebrdScalar: gebrd_scalar::Sealed {}

impl GebrdScalar for f32 {}
impl GebrdScalar for f64 {}
impl GebrdScalar for Complex32 {}
impl GebrdScalar for Complex64 {}

mod orgbr_scalar {
    use super::*;

    pub trait Sealed {
        fn orgbr_buffer_size(
            ctx: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize>
        where
            Self: Sized;

        fn orgbr(
            ctx: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()>
        where
            Self: Sized;
    }

    impl Sealed for f32 {
        fn orgbr_buffer_size(
            ctx: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            sorgbr_buffer_size(ctx, side, m, n, k, a.data, a.leading_dimension, tau.data)
        }

        fn orgbr(
            ctx: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            sorgbr(
                ctx,
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

    impl Sealed for f64 {
        fn orgbr_buffer_size(
            ctx: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            dorgbr_buffer_size(ctx, side, m, n, k, a.data, a.leading_dimension, tau.data)
        }

        fn orgbr(
            ctx: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            dorgbr(
                ctx,
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

    impl Sealed for Complex32 {
        fn orgbr_buffer_size(
            ctx: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            cungbr_buffer_size(ctx, side, m, n, k, a.data, a.leading_dimension, tau.data)
        }

        fn orgbr(
            ctx: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            cungbr(
                ctx,
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

    impl Sealed for Complex64 {
        fn orgbr_buffer_size(
            ctx: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixRef<'_, Self>,
            tau: VectorRef<'_, Self>,
        ) -> Result<usize> {
            zungbr_buffer_size(ctx, side, m, n, k, a.data, a.leading_dimension, tau.data)
        }

        fn orgbr(
            ctx: &Context,
            side: SideMode,
            m: usize,
            n: usize,
            k: usize,
            a: MatrixMut<'_, Self>,
            tau: VectorRef<'_, Self>,
            workspace: &mut DeviceMemory<Self>,
            dev_info: &mut DeviceMemory<i32>,
        ) -> Result<()> {
            zungbr(
                ctx,
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

pub trait OrgbrScalar: orgbr_scalar::Sealed {}

impl OrgbrScalar for f32 {}
impl OrgbrScalar for f64 {}
impl OrgbrScalar for Complex32 {}
impl OrgbrScalar for Complex64 {}
