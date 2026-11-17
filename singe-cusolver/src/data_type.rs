//! Supported matrix elements and their real, complex, and eigenvalue result types.

use singe_cuda::{
    data_type::DataTypeLike,
    types::{Complex32, Complex64},
};

/// A supported solver matrix element: `f32`, `f64`, `Complex32`, or `Complex64`.
///
/// The sealed trait carries the element's real and complex result types and dispatches every typed operation to its native per-precision entry point.
/// Associated types prevent unsupported precision combinations in matrix, vector, and scalar results.
pub trait SolverElement: private::Sealed {}

pub(crate) mod private {
    use singe_cuda::data_type::DataTypeLike;

    use crate::{
        data_type::SolverReal, dense::scalar::DenseDispatch, eigen::scalar::EigenDispatch,
        svd::scalar::SvdDispatch,
    };

    #[expect(
        unnameable_types,
        reason = "sealed supertrait of SolverElement; users cannot name or implement it"
    )]
    pub trait Element: DataTypeLike {
        /// Real result precision corresponding to this matrix element type.
        type Real: SolverReal;

        /// Complex result precision corresponding to this matrix element type.
        type Complex: DataTypeLike;
    }

    #[expect(
        unnameable_types,
        reason = "sealed supertrait of SolverElement; users cannot name or implement it"
    )]
    pub trait Sealed: Element + DenseDispatch + EigenDispatch + SvdDispatch {}

    #[expect(
        unnameable_types,
        reason = "sealed supertrait of SolverReal; users cannot name or implement it"
    )]
    pub trait Real {
        fn zero_ref() -> &'static Self;
    }

    #[expect(
        unnameable_types,
        reason = "sealed supertrait of EigenvalueElement; users cannot name or implement it"
    )]
    pub trait Eigenvalue<T> {}
}

/// A real scalar precision supported by native solver result and selection APIs.
pub trait SolverReal: DataTypeLike + PartialOrd + private::Real {}

impl private::Real for f32 {
    fn zero_ref() -> &'static Self {
        &0.0_f32
    }
}

impl private::Real for f64 {
    fn zero_ref() -> &'static Self {
        &0.0_f64
    }
}

impl SolverReal for f32 {}

impl SolverReal for f64 {}

/// A supported general-eigenvalue representation for an input matrix type.
/// Real storage uses two n-element blocks for real and imaginary parts.
/// Complex storage uses n elements.
pub trait EigenvalueElement<T: SolverElement>: DataTypeLike + private::Eigenvalue<T> {
    /// Storage elements per eigenvalue: two for separate real and imaginary blocks, or one for complex storage.
    const ELEMENTS_PER_VALUE: usize;
}

macro_rules! eigenvalue {
    ($input:ty, $output:ty, $width:expr) => {
        impl private::Eigenvalue<$input> for $output {}
        impl EigenvalueElement<$input> for $output {
            const ELEMENTS_PER_VALUE: usize = $width;
        }
    };
}

eigenvalue!(f32, f32, 2);

eigenvalue!(f32, Complex32, 1);

eigenvalue!(f64, f64, 2);

eigenvalue!(f64, Complex64, 1);

eigenvalue!(Complex32, Complex32, 1);

eigenvalue!(Complex64, Complex64, 1);

macro_rules! solver_element {
    ($matrix:ty, $real:ty, $complex:ty) => {
        impl private::Element for $matrix {
            type Real = $real;
            type Complex = $complex;
        }
        impl private::Sealed for $matrix {}
        impl SolverElement for $matrix {}
    };
}

solver_element!(f32, f32, Complex32);

solver_element!(f64, f64, Complex64);

solver_element!(Complex32, f32, Complex32);

solver_element!(Complex64, f64, Complex64);
