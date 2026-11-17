#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "BLAS and factorization operand names follow the standard mathematical API"
)]
use singe_cuda::{
    types::{Complex32, Complex64},
    view::DeviceOutputView,
};

use crate::{
    context::Context,
    data_type::private::Element,
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

// Per-precision entry points of the dense operations that have no generic native interface.
#[expect(
    unnameable_types,
    reason = "sealed supertrait of SolverElement; users cannot name or implement it"
)]
pub trait DenseDispatch: Element {
    unsafe fn potrf_batched(
        context: &Context,
        fill_mode: FillMode,
        n: usize,
        a: BatchedMatrixRef<'_, Self>,
        info: DeviceOutputView<'_, i32>,
    ) -> Result<()>;

    unsafe fn potrs_batched(
        context: &Context,
        fill_mode: FillMode,
        n: usize,
        a: BatchedMatrixRef<'_, Self>,
        b: BatchedVectorRef<'_, Self>,
        info: DeviceOutputView<'_, i32>,
    ) -> Result<()>;

    fn potri_buffer_size(
        context: &Context,
        fill_mode: FillMode,
        n: usize,
        factor: MatrixLayout,
    ) -> Result<usize>;

    unsafe fn potri(
        context: &Context,
        fill_mode: FillMode,
        n: usize,
        factor: MatrixMut<'_, Self>,
        workspace: DeviceOutputView<'_, Self>,
        dev_info: DeviceOutputView<'_, i32>,
    ) -> Result<()>;

    fn ormqr_buffer_size(
        context: &Context,
        side: SideMode,
        operation: Operation,
        m: usize,
        n: usize,
        k: usize,
        reflectors: MatrixLayout,
        matrix: MatrixLayout,
    ) -> Result<usize>;

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
    ) -> Result<()>;

    fn orgqr_buffer_size(
        context: &Context,
        m: usize,
        n: usize,
        k: usize,
        a: MatrixLayout,
    ) -> Result<usize>;

    unsafe fn orgqr(
        context: &Context,
        m: usize,
        n: usize,
        k: usize,
        a: MatrixMut<'_, Self>,
        tau: VectorRef<'_, Self>,
        workspace: DeviceOutputView<'_, Self>,
        dev_info: DeviceOutputView<'_, i32>,
    ) -> Result<()>;

    fn sytrd_buffer_size(
        context: &Context,
        fill_mode: FillMode,
        n: usize,
        a: MatrixLayout,
    ) -> Result<usize>;

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
    ) -> Result<()>;

    fn orgtr_buffer_size(
        context: &Context,
        fill_mode: FillMode,
        n: usize,
        a: MatrixLayout,
    ) -> Result<usize>;

    unsafe fn orgtr(
        context: &Context,
        fill_mode: FillMode,
        n: usize,
        a: MatrixMut<'_, Self>,
        tau: VectorRef<'_, Self>,
        workspace: DeviceOutputView<'_, Self>,
        dev_info: DeviceOutputView<'_, i32>,
    ) -> Result<()>;

    fn ormtr_buffer_size(
        context: &Context,
        side: SideMode,
        fill_mode: FillMode,
        operation: Operation,
        m: usize,
        n: usize,
        reflectors: MatrixLayout,
        matrix: MatrixLayout,
    ) -> Result<usize>;

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
    ) -> Result<()>;

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
    ) -> Result<()>;

    fn orgbr_buffer_size(
        context: &Context,
        side: SideMode,
        m: usize,
        n: usize,
        k: usize,
        a: MatrixLayout,
    ) -> Result<usize>;

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
    ) -> Result<()>;
}

macro_rules! dense_dispatch {
    (
        $ty:ty,
        potrf_batched: $potrf_batched:ident,
        potrs_batched: $potrs_batched:ident,
        potri: $potri:ident,
        potri_buffer_size: $potri_buffer_size:ident,
        ormqr: $ormqr:ident,
        ormqr_buffer_size: $ormqr_buffer_size:ident,
        orgqr: $orgqr:ident,
        orgqr_buffer_size: $orgqr_buffer_size:ident,
        sytrd: $sytrd:ident,
        sytrd_buffer_size: $sytrd_buffer_size:ident,
        orgtr: $orgtr:ident,
        orgtr_buffer_size: $orgtr_buffer_size:ident,
        ormtr: $ormtr:ident,
        ormtr_buffer_size: $ormtr_buffer_size:ident,
        gebrd: $gebrd:ident,
        gebrd_buffer_size: $gebrd_buffer_size:ident,
        orgbr: $orgbr:ident,
        orgbr_buffer_size: $orgbr_buffer_size:ident $(,)?
    ) => {
        impl DenseDispatch for $ty {
            unsafe fn potrf_batched(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
                a: BatchedMatrixRef<'_, Self>,
                info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                // SAFETY: forwards this method's raw execution contract unchanged; the per-precision function validates the operands itself.
                unsafe { $potrf_batched(context, fill_mode, n, a, info) }
            }

            unsafe fn potrs_batched(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
                a: BatchedMatrixRef<'_, Self>,
                b: BatchedVectorRef<'_, Self>,
                info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                // SAFETY: forwards this method's raw execution contract unchanged; the per-precision function validates the operands itself.
                unsafe { $potrs_batched(context, fill_mode, n, a, b, info) }
            }

            fn potri_buffer_size(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
                factor: MatrixLayout,
            ) -> Result<usize> {
                $potri_buffer_size(context, fill_mode, n, factor.leading_dimension)
            }

            unsafe fn potri(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
                factor: MatrixMut<'_, Self>,
                workspace: DeviceOutputView<'_, Self>,
                dev_info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                // SAFETY: forwards this method's raw execution contract unchanged; the per-precision function validates the operands itself.
                unsafe {
                    $potri(
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
                $ormqr_buffer_size(
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
                // SAFETY: forwards this method's raw execution contract unchanged; the per-precision function validates the operands itself.
                unsafe {
                    $ormqr(
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

            fn orgqr_buffer_size(
                context: &Context,
                m: usize,
                n: usize,
                k: usize,
                a: MatrixLayout,
            ) -> Result<usize> {
                $orgqr_buffer_size(context, m, n, k, a.leading_dimension)
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
                // SAFETY: forwards this method's raw execution contract unchanged; the per-precision function validates the operands itself.
                unsafe {
                    $orgqr(
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

            fn sytrd_buffer_size(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
                a: MatrixLayout,
            ) -> Result<usize> {
                $sytrd_buffer_size(context, fill_mode, n, a.leading_dimension)
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
                // SAFETY: forwards this method's raw execution contract unchanged; the per-precision function validates the operands itself.
                unsafe {
                    $sytrd(
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

            fn orgtr_buffer_size(
                context: &Context,
                fill_mode: FillMode,
                n: usize,
                a: MatrixLayout,
            ) -> Result<usize> {
                $orgtr_buffer_size(context, fill_mode, n, a.leading_dimension)
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
                // SAFETY: forwards this method's raw execution contract unchanged; the per-precision function validates the operands itself.
                unsafe {
                    $orgtr(
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
                $ormtr_buffer_size(
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
                // SAFETY: forwards this method's raw execution contract unchanged; the per-precision function validates the operands itself.
                unsafe {
                    $ormtr(
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

            fn gebrd_buffer_size(context: &Context, m: usize, n: usize) -> Result<usize> {
                $gebrd_buffer_size(context, m, n)
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
                // SAFETY: forwards this method's raw execution contract unchanged; the per-precision function validates the operands itself.
                unsafe {
                    $gebrd(
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

            fn orgbr_buffer_size(
                context: &Context,
                side: SideMode,
                m: usize,
                n: usize,
                k: usize,
                a: MatrixLayout,
            ) -> Result<usize> {
                $orgbr_buffer_size(context, side, m, n, k, a.leading_dimension)
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
                // SAFETY: forwards this method's raw execution contract unchanged; the per-precision function validates the operands itself.
                unsafe {
                    $orgbr(
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
    };
}

dense_dispatch!(
    f32,
    potrf_batched: spotrf_batched,
    potrs_batched: spotrs_batched,
    potri: spotri,
    potri_buffer_size: spotri_buffer_size,
    ormqr: sormqr,
    ormqr_buffer_size: sormqr_buffer_size,
    orgqr: sorgqr,
    orgqr_buffer_size: sorgqr_buffer_size,
    sytrd: ssytrd,
    sytrd_buffer_size: ssytrd_buffer_size,
    orgtr: sorgtr,
    orgtr_buffer_size: sorgtr_buffer_size,
    ormtr: sormtr,
    ormtr_buffer_size: sormtr_buffer_size,
    gebrd: sgebrd,
    gebrd_buffer_size: sgebrd_buffer_size,
    orgbr: sorgbr,
    orgbr_buffer_size: sorgbr_buffer_size,
);

dense_dispatch!(
    f64,
    potrf_batched: dpotrf_batched,
    potrs_batched: dpotrs_batched,
    potri: dpotri,
    potri_buffer_size: dpotri_buffer_size,
    ormqr: dormqr,
    ormqr_buffer_size: dormqr_buffer_size,
    orgqr: dorgqr,
    orgqr_buffer_size: dorgqr_buffer_size,
    sytrd: dsytrd,
    sytrd_buffer_size: dsytrd_buffer_size,
    orgtr: dorgtr,
    orgtr_buffer_size: dorgtr_buffer_size,
    ormtr: dormtr,
    ormtr_buffer_size: dormtr_buffer_size,
    gebrd: dgebrd,
    gebrd_buffer_size: dgebrd_buffer_size,
    orgbr: dorgbr,
    orgbr_buffer_size: dorgbr_buffer_size,
);

dense_dispatch!(
    Complex32,
    potrf_batched: cpotrf_batched,
    potrs_batched: cpotrs_batched,
    potri: cpotri,
    potri_buffer_size: cpotri_buffer_size,
    ormqr: cunmqr,
    ormqr_buffer_size: cunmqr_buffer_size,
    orgqr: cungqr,
    orgqr_buffer_size: cungqr_buffer_size,
    sytrd: chetrd,
    sytrd_buffer_size: chetrd_buffer_size,
    orgtr: cungtr,
    orgtr_buffer_size: cungtr_buffer_size,
    ormtr: cunmtr,
    ormtr_buffer_size: cunmtr_buffer_size,
    gebrd: cgebrd,
    gebrd_buffer_size: cgebrd_buffer_size,
    orgbr: cungbr,
    orgbr_buffer_size: cungbr_buffer_size,
);

dense_dispatch!(
    Complex64,
    potrf_batched: zpotrf_batched,
    potrs_batched: zpotrs_batched,
    potri: zpotri,
    potri_buffer_size: zpotri_buffer_size,
    ormqr: zunmqr,
    ormqr_buffer_size: zunmqr_buffer_size,
    orgqr: zungqr,
    orgqr_buffer_size: zungqr_buffer_size,
    sytrd: zhetrd,
    sytrd_buffer_size: zhetrd_buffer_size,
    orgtr: zungtr,
    orgtr_buffer_size: zungtr_buffer_size,
    ormtr: zunmtr,
    ormtr_buffer_size: zunmtr_buffer_size,
    gebrd: zgebrd,
    gebrd_buffer_size: zgebrd_buffer_size,
    orgbr: zungbr,
    orgbr_buffer_size: zungbr_buffer_size,
);
