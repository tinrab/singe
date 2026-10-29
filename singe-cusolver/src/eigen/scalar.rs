use std::mem::MaybeUninit;

use singe_cuda::{
    types::{Complex32, Complex64},
    view::DeviceOutputView,
};

use crate::{
    context::Context,
    data_type::SolverElement,
    eigen::{EigenSelection, SyevjInfo, raw},
    error::Result,
    layout::{ByteWorkspaceMut, MatrixLayout, MatrixMut, MatrixOutput, WorkspaceSizes},
    params::Params,
    types::{EigenMode, EigenType, FillMode},
};

/// A supported scalar precision for native eigenvalue dispatch.
pub trait EigenScalar: SolverElement + private::Sealed {}

type JacobiQuery =
    fn(&Context, EigenMode, FillMode, usize, MatrixLayout, &SyevjInfo) -> Result<usize>;

type Jacobi<T> = for<'a> unsafe fn(
    &Context,
    EigenMode,
    FillMode,
    usize,
    MatrixMut<'a, T>,
    DeviceOutputView<'a, <T as SolverElement>::Real>,
    DeviceOutputView<'a, T>,
    DeviceOutputView<'a, i32>,
    &mut SyevjInfo,
) -> Result<()>;

type BatchedQuery =
    fn(&Context, EigenMode, FillMode, usize, MatrixLayout, &SyevjInfo, usize) -> Result<usize>;

type Batched<T> = for<'a> unsafe fn(
    &Context,
    EigenMode,
    FillMode,
    usize,
    MatrixMut<'a, T>,
    DeviceOutputView<'a, <T as SolverElement>::Real>,
    DeviceOutputView<'a, T>,
    DeviceOutputView<'a, i32>,
    &mut SyevjInfo,
    usize,
) -> Result<()>;

type GeneralizedJacobiQuery = fn(
    &Context,
    EigenType,
    EigenMode,
    FillMode,
    usize,
    MatrixLayout,
    MatrixLayout,
    &SyevjInfo,
) -> Result<usize>;

type GeneralizedJacobi<T> = for<'a> unsafe fn(
    &Context,
    EigenType,
    EigenMode,
    FillMode,
    usize,
    MatrixMut<'a, T>,
    MatrixMut<'a, T>,
    DeviceOutputView<'a, <T as SolverElement>::Real>,
    DeviceOutputView<'a, T>,
    DeviceOutputView<'a, i32>,
    &mut SyevjInfo,
) -> Result<()>;

type GeneralizedDivideQuery = fn(
    &Context,
    EigenType,
    EigenMode,
    FillMode,
    usize,
    MatrixLayout,
    MatrixLayout,
) -> Result<usize>;

type GeneralizedDivide<T> = for<'a> unsafe fn(
    &Context,
    EigenType,
    EigenMode,
    FillMode,
    usize,
    MatrixMut<'a, T>,
    MatrixMut<'a, T>,
    DeviceOutputView<'a, <T as SolverElement>::Real>,
    DeviceOutputView<'a, T>,
    DeviceOutputView<'a, i32>,
) -> Result<()>;

type GeneralizedSelectedQuery<T> = fn(
    &Context,
    EigenType,
    EigenMode,
    FillMode,
    &EigenSelection<<T as SolverElement>::Real>,
    usize,
    MatrixLayout,
    MatrixLayout,
) -> Result<usize>;

type GeneralizedSelected<T> = for<'a> unsafe fn(
    &Context,
    EigenType,
    EigenMode,
    FillMode,
    &EigenSelection<<T as SolverElement>::Real>,
    usize,
    MatrixMut<'a, T>,
    MatrixMut<'a, T>,
    DeviceOutputView<'a, <T as SolverElement>::Real>,
    DeviceOutputView<'a, T>,
    DeviceOutputView<'a, i32>,
    &mut MaybeUninit<i32>,
) -> Result<()>;

type GeneralQuery =
    fn(&Context, &Params, usize, MatrixLayout, Option<MatrixLayout>) -> Result<WorkspaceSizes>;

type General<T> = for<'a> unsafe fn(
    &Context,
    &Params,
    usize,
    MatrixMut<'a, T>,
    Option<MatrixOutput<'a, T>>,
    DeviceOutputView<'a, <T as SolverElement>::Complex>,
    ByteWorkspaceMut<'a>,
    DeviceOutputView<'a, i32>,
) -> Result<()>;

mod private {
    use super::{
        Batched, BatchedQuery, General, GeneralQuery, GeneralizedDivide, GeneralizedDivideQuery,
        GeneralizedJacobi, GeneralizedJacobiQuery, GeneralizedSelected, GeneralizedSelectedQuery,
        Jacobi, JacobiQuery, SolverElement,
    };

    pub trait Sealed: SolverElement {
        const JACOBI_QUERY: JacobiQuery;
        const JACOBI: Jacobi<Self>;
        const BATCHED_QUERY: BatchedQuery;
        const BATCHED: Batched<Self>;
        const GENERALIZED_JACOBI_QUERY: GeneralizedJacobiQuery;
        const GENERALIZED_JACOBI: GeneralizedJacobi<Self>;
        const GENERALIZED_DIVIDE_QUERY: GeneralizedDivideQuery;
        const GENERALIZED_DIVIDE: GeneralizedDivide<Self>;
        const GENERALIZED_SELECTED_QUERY: GeneralizedSelectedQuery<Self>;
        const GENERALIZED_SELECTED: GeneralizedSelected<Self>;
        const GENERAL_QUERY: GeneralQuery;
        const GENERAL: General<Self>;
    }
}

impl private::Sealed for f32 {
    const JACOBI_QUERY: JacobiQuery = raw::ssyevj_buffer_size;
    const JACOBI: Jacobi<Self> = raw::ssyevj;
    const BATCHED_QUERY: BatchedQuery = raw::ssyevj_batched_buffer_size;
    const BATCHED: Batched<Self> = raw::ssyevj_batched;
    const GENERALIZED_JACOBI_QUERY: GeneralizedJacobiQuery = raw::ssygvj_buffer_size;
    const GENERALIZED_JACOBI: GeneralizedJacobi<Self> = raw::ssygvj;
    const GENERALIZED_DIVIDE_QUERY: GeneralizedDivideQuery = raw::ssygvd_buffer_size;
    const GENERALIZED_DIVIDE: GeneralizedDivide<Self> = raw::ssygvd;

    const GENERALIZED_SELECTED_QUERY: GeneralizedSelectedQuery<Self> =
        raw::ssygvdx_selected_buffer_size;

    const GENERALIZED_SELECTED: GeneralizedSelected<Self> = raw::ssygvdx_selected;
    const GENERAL_QUERY: GeneralQuery = raw::xgeev_buffer_size::<Self, Complex32>;
    const GENERAL: General<Self> = raw::xgeev::<Self, Complex32>;
}

impl EigenScalar for f32 {}

impl private::Sealed for f64 {
    const JACOBI_QUERY: JacobiQuery = raw::dsyevj_buffer_size;
    const JACOBI: Jacobi<Self> = raw::dsyevj;
    const BATCHED_QUERY: BatchedQuery = raw::dsyevj_batched_buffer_size;
    const BATCHED: Batched<Self> = raw::dsyevj_batched;
    const GENERALIZED_JACOBI_QUERY: GeneralizedJacobiQuery = raw::dsygvj_buffer_size;
    const GENERALIZED_JACOBI: GeneralizedJacobi<Self> = raw::dsygvj;
    const GENERALIZED_DIVIDE_QUERY: GeneralizedDivideQuery = raw::dsygvd_buffer_size;
    const GENERALIZED_DIVIDE: GeneralizedDivide<Self> = raw::dsygvd;

    const GENERALIZED_SELECTED_QUERY: GeneralizedSelectedQuery<Self> =
        raw::dsygvdx_selected_buffer_size;

    const GENERALIZED_SELECTED: GeneralizedSelected<Self> = raw::dsygvdx_selected;
    const GENERAL_QUERY: GeneralQuery = raw::xgeev_buffer_size::<Self, Complex64>;
    const GENERAL: General<Self> = raw::xgeev::<Self, Complex64>;
}

impl EigenScalar for f64 {}

impl private::Sealed for Complex32 {
    const JACOBI_QUERY: JacobiQuery = raw::cheevj_buffer_size;
    const JACOBI: Jacobi<Self> = raw::cheevj;
    const BATCHED_QUERY: BatchedQuery = raw::cheevj_batched_buffer_size;
    const BATCHED: Batched<Self> = raw::cheevj_batched;
    const GENERALIZED_JACOBI_QUERY: GeneralizedJacobiQuery = raw::chegvj_buffer_size;
    const GENERALIZED_JACOBI: GeneralizedJacobi<Self> = raw::chegvj;
    const GENERALIZED_DIVIDE_QUERY: GeneralizedDivideQuery = raw::chegvd_buffer_size;
    const GENERALIZED_DIVIDE: GeneralizedDivide<Self> = raw::chegvd;

    const GENERALIZED_SELECTED_QUERY: GeneralizedSelectedQuery<Self> =
        raw::chegvdx_selected_buffer_size;

    const GENERALIZED_SELECTED: GeneralizedSelected<Self> = raw::chegvdx_selected;
    const GENERAL_QUERY: GeneralQuery = raw::xgeev_buffer_size::<Self, Self>;
    const GENERAL: General<Self> = raw::xgeev::<Self, Self>;
}

impl EigenScalar for Complex32 {}

impl private::Sealed for Complex64 {
    const JACOBI_QUERY: JacobiQuery = raw::zheevj_buffer_size;
    const JACOBI: Jacobi<Self> = raw::zheevj;
    const BATCHED_QUERY: BatchedQuery = raw::zheevj_batched_buffer_size;
    const BATCHED: Batched<Self> = raw::zheevj_batched;
    const GENERALIZED_JACOBI_QUERY: GeneralizedJacobiQuery = raw::zhegvj_buffer_size;
    const GENERALIZED_JACOBI: GeneralizedJacobi<Self> = raw::zhegvj;
    const GENERALIZED_DIVIDE_QUERY: GeneralizedDivideQuery = raw::zhegvd_buffer_size;
    const GENERALIZED_DIVIDE: GeneralizedDivide<Self> = raw::zhegvd;

    const GENERALIZED_SELECTED_QUERY: GeneralizedSelectedQuery<Self> =
        raw::zhegvdx_selected_buffer_size;

    const GENERALIZED_SELECTED: GeneralizedSelected<Self> = raw::zhegvdx_selected;
    const GENERAL_QUERY: GeneralQuery = raw::xgeev_buffer_size::<Self, Self>;
    const GENERAL: General<Self> = raw::xgeev::<Self, Self>;
}

impl EigenScalar for Complex64 {}
