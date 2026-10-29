use std::mem::MaybeUninit;

use singe_cuda::{
    types::{Complex32, Complex64},
    view::DeviceOutputView,
};

use crate::{
    context::Context,
    data_type::SolverElement,
    error::Result,
    layout::{
        MatrixMut, MatrixOutput, StridedBatchedMatrixOutput, StridedBatchedMatrixRef,
        StridedBatchedVectorOutput,
    },
    svd::{GesvdjInfo, StridedSvdLayout, SvdLayout, raw},
    types::EigenMode,
};

/// A supported matrix type for Jacobi and approximate native SVD dispatch.
pub trait SvdScalar: SolverElement + private::Sealed {}

type JacobiQuery =
    fn(&Context, EigenMode, bool, usize, usize, SvdLayout, &GesvdjInfo) -> Result<usize>;

type BatchedQuery =
    fn(&Context, EigenMode, usize, usize, SvdLayout, &GesvdjInfo, usize) -> Result<usize>;

type ApproximateQuery =
    fn(&Context, EigenMode, usize, usize, usize, StridedSvdLayout, usize) -> Result<usize>;

type Jacobi<T> = for<'a> unsafe fn(
    &Context,
    EigenMode,
    bool,
    usize,
    usize,
    MatrixMut<'a, T>,
    DeviceOutputView<'a, <T as SolverElement>::Real>,
    Option<MatrixOutput<'a, T>>,
    Option<MatrixOutput<'a, T>>,
    DeviceOutputView<'a, T>,
    DeviceOutputView<'a, i32>,
    &mut GesvdjInfo,
) -> Result<()>;

type Batched<T> = for<'a> unsafe fn(
    &Context,
    EigenMode,
    usize,
    usize,
    MatrixMut<'a, T>,
    DeviceOutputView<'a, <T as SolverElement>::Real>,
    Option<MatrixOutput<'a, T>>,
    Option<MatrixOutput<'a, T>>,
    DeviceOutputView<'a, T>,
    DeviceOutputView<'a, i32>,
    &mut GesvdjInfo,
    usize,
) -> Result<()>;

type Approximate<T> = for<'a> unsafe fn(
    &Context,
    EigenMode,
    usize,
    usize,
    usize,
    StridedBatchedMatrixRef<'a, T>,
    StridedBatchedVectorOutput<'a, <T as SolverElement>::Real>,
    Option<StridedBatchedMatrixOutput<'a, T>>,
    Option<StridedBatchedMatrixOutput<'a, T>>,
    DeviceOutputView<'a, T>,
    DeviceOutputView<'a, i32>,
    Option<&'a mut [MaybeUninit<f64>]>,
    usize,
) -> Result<()>;

mod private {
    use super::{
        Approximate, ApproximateQuery, Batched, BatchedQuery, Jacobi, JacobiQuery, SolverElement,
    };

    pub trait Sealed: SolverElement {
        const JACOBI_QUERY: JacobiQuery;
        const JACOBI: Jacobi<Self>;
        const BATCHED_QUERY: BatchedQuery;
        const BATCHED: Batched<Self>;
        const APPROXIMATE_QUERY: ApproximateQuery;
        const APPROXIMATE: Approximate<Self>;
    }
}

macro_rules! scalar {
    ($ty:ty, $jq:ident, $j:ident, $bq:ident, $b:ident, $aq:ident, $a:ident) => {
        impl private::Sealed for $ty {
            const JACOBI_QUERY: JacobiQuery = raw::$jq;
            const JACOBI: Jacobi<Self> = raw::$j;
            const BATCHED_QUERY: BatchedQuery = raw::$bq;
            const BATCHED: Batched<Self> = raw::$b;
            const APPROXIMATE_QUERY: ApproximateQuery = raw::$aq;
            const APPROXIMATE: Approximate<Self> = raw::$a;
        }
        impl SvdScalar for $ty {}
    };
}

scalar!(
    f32,
    sgesvdj_buffer_size,
    sgesvdj,
    sgesvdj_batched_buffer_size,
    sgesvdj_batched,
    sgesvda_strided_batched_buffer_size,
    sgesvda_strided_batched
);

scalar!(
    f64,
    dgesvdj_buffer_size,
    dgesvdj,
    dgesvdj_batched_buffer_size,
    dgesvdj_batched,
    dgesvda_strided_batched_buffer_size,
    dgesvda_strided_batched
);

scalar!(
    Complex32,
    cgesvdj_buffer_size,
    cgesvdj,
    cgesvdj_batched_buffer_size,
    cgesvdj_batched,
    cgesvda_strided_batched_buffer_size,
    cgesvda_strided_batched
);

scalar!(
    Complex64,
    zgesvdj_buffer_size,
    zgesvdj,
    zgesvdj_batched_buffer_size,
    zgesvdj_batched,
    zgesvda_strided_batched_buffer_size,
    zgesvda_strided_batched
);
