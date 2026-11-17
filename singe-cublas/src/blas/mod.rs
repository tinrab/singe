//! BLAS operations, generic over the element type.
//!
//! Each operation is an `unsafe fn <op>_async`, which checks its operands and enqueues on the handle's stream, and a safe `fn <op>`, which also waits.
//! [`Element`] covers `f32`, `f64`, [`Complex32`], and [`Complex64`]; [`RealElement`] and [`ComplexElement`] select the routines that exist for only one of them.

// Declares the safe form of an `_async` operation, which waits for the handle's stream before the borrowed operands are released.
macro_rules! waiting {
    (
        $(#[$meta:meta])*
        $name:ident = $operation:ident<$element:ident: $bound:ident>($($arg:ident: $ty:ty),* $(,)?)
    ) => {
        $(#[$meta])*
        pub fn $name<$element: $bound>(
            context: &crate::context::Context,
            $($arg: $ty),*
        ) -> crate::error::Result<()> {
            // SAFETY: the scope waits for the handle's stream before the borrowed operands are released, on every path.
            context.sync_scope(|| unsafe { $operation(context, $($arg),*) })
        }
    };
}

mod dispatch;

/// Mixed-precision operations whose data types are chosen independently.
pub mod extension;

/// Vector operations, reductions, and rotations.
pub mod level1;

/// Matrix-vector products, rank updates, and triangular solves.
pub mod level2;

/// Matrix-matrix products, rank updates, triangular solves, and layout conversions.
pub mod level3;

pub(crate) mod shape;

use singe_cuda::{
    data_type::DataTypeLike,
    types::{Complex32, Complex64},
};

mod sealed {
    use singe_cuda::types::{Complex32, Complex64};

    #[expect(
        unnameable_types,
        reason = "sealed supertrait; users cannot name or implement it"
    )]
    pub trait Sealed {}

    impl Sealed for f32 {}

    impl Sealed for f64 {}

    impl Sealed for Complex32 {}

    impl Sealed for Complex64 {}
}

/// An element type of the typed BLAS routines: `f32`, `f64`, [`Complex32`], or [`Complex64`].
pub trait Element:
    DataTypeLike
    + sealed::Sealed
    + dispatch::Scal
    + dispatch::Axpy
    + dispatch::CopyVector
    + dispatch::Swap
    + dispatch::Dot
    + dispatch::Nrm2
    + dispatch::Asum
    + dispatch::Iamax
    + dispatch::Iamin
    + dispatch::Rot
    + dispatch::Rotg
    + dispatch::Gemv
    + dispatch::Gbmv
    + dispatch::Trmv
    + dispatch::Trsv
    + dispatch::Tbmv
    + dispatch::Tbsv
    + dispatch::Tpmv
    + dispatch::Tpsv
    + dispatch::Symv
    + dispatch::Ger
    + dispatch::Syr
    + dispatch::Syr2
    + dispatch::Gemm
    + dispatch::Symm
    + dispatch::Syrk
    + dispatch::Syr2k
    + dispatch::Syrkx
    + dispatch::Trmm
    + dispatch::Trsm
    + dispatch::Geam
    + dispatch::Dgmm
    + dispatch::Tpttr
    + dispatch::Trttp
    + dispatch::GemmBatched
    + dispatch::GemmStridedBatched
    + dispatch::TrsmBatched
{
    /// The real type of the element's components and norms, such as `f32` for [`Complex32`].
    type Real: DataTypeLike;
}

impl Element for f32 {
    type Real = Self;
}

impl Element for f64 {
    type Real = Self;
}

impl Element for Complex32 {
    type Real = f32;
}

impl Element for Complex64 {
    type Real = f64;
}

/// A real element type, `f32` or `f64`, for the routines cuBLAS provides only for real values.
pub trait RealElement:
    Element
    + dispatch::Rotm
    + dispatch::Rotmg
    + dispatch::Sbmv
    + dispatch::Spmv
    + dispatch::Spr
    + dispatch::Spr2
    + dispatch::GemmGroupedBatched
{
}

impl RealElement for f32 {}

impl RealElement for f64 {}

/// A complex element type, [`Complex32`] or [`Complex64`], for the conjugating and Hermitian routines.
pub trait ComplexElement:
    Element
    + dispatch::Dotc
    + dispatch::Gerc
    + dispatch::Hemv
    + dispatch::Hbmv
    + dispatch::Hpmv
    + dispatch::Her
    + dispatch::Hpr
    + dispatch::Her2
    + dispatch::Hpr2
    + dispatch::Gemm3m
    + dispatch::Hemm
    + dispatch::Herk
    + dispatch::Her2k
    + dispatch::Herkx
{
}

impl ComplexElement for Complex32 {}

impl ComplexElement for Complex64 {}
