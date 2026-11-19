use singe_cuda::view::{DeviceOutput, DeviceRange};

use crate::{
    blas::{
        ComplexElement, Element, RealElement,
        shape::{ensure_matrix, ensure_packed, ensure_vector, stored},
    },
    context::Context,
    error::{Error, Result},
    scalar::Scalar,
    try_ffi,
    types::{DiagonalType, FillMode, Operation},
    utility::to_i64,
};

// Returns the lengths of `x` and `y` for a general `m` x `n` matrix-vector product.
fn product_lengths(trans: Operation, m: usize, n: usize) -> (usize, usize) {
    let (rows, cols) = stored(trans, m, n);
    (cols, rows)
}

fn band_rows(lower: usize, upper: usize) -> Result<usize> {
    lower
        .checked_add(upper)
        .and_then(|sum| sum.checked_add(1))
        .ok_or_else(|| Error::out_of_range("band rows"))
}

/// Enqueues `y = alpha * op(A) * x + beta * y` for an `m` x `n` matrix `A`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if an operand is too small, a leading dimension or increment is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn gemv_async<T: Element>(
    context: &Context,
    trans: Operation,
    m: usize,
    n: usize,
    alpha: Scalar<'_, T>,
    a: &(impl DeviceRange<T> + ?Sized),
    lda: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    beta: Scalar<'_, T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
) -> Result<()> {
    context.bind()?;
    ensure_matrix("a", a.len(), m, n, lda)?;
    let (x_len, y_len) = product_lengths(trans, m, n);
    ensure_vector("x", x.len(), x_len, incx)?;
    ensure_vector("y", y.len(), y_len, incy)?;
    context.prepare_operands(
        &[a.region(), x.region(), y.region()],
        &[alpha.region(), beta.region()],
    )?;
    context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
    // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::gemv(
            context.as_raw(),
            trans.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            alpha.ptr(),
            a.as_device_ptr(),
            to_i64(lda, "lda")?,
            x.as_device_ptr(),
            to_i64(incx, "incx")?,
            beta.ptr(),
            y.as_mut_ptr(),
            to_i64(incy, "incy")?,
        ))
    }
}

waiting! {
    /// Computes `y = alpha * op(A) * x + beta * y` and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`gemv_async`] lists.
    gemv = gemv_async<T: Element>(
        trans: Operation,
        m: usize,
        n: usize,
        alpha: Scalar<'_, T>,
        a: &(impl DeviceRange<T> + ?Sized),
        lda: usize,
        x: &(impl DeviceRange<T> + ?Sized),
        incx: usize,
        beta: Scalar<'_, T>,
        y: &mut (impl DeviceOutput<T> + ?Sized),
        incy: usize,
    )
}

/// Enqueues `y = alpha * op(A) * x + beta * y` for an `m` x `n` band matrix `A` with `kl` subdiagonals and `ku` superdiagonals.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if an operand is too small, a leading dimension or increment is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn gbmv_async<T: Element>(
    context: &Context,
    trans: Operation,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: Scalar<'_, T>,
    a: &(impl DeviceRange<T> + ?Sized),
    lda: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    beta: Scalar<'_, T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
) -> Result<()> {
    context.bind()?;
    ensure_matrix("a", a.len(), band_rows(kl, ku)?, n, lda)?;
    let (x_len, y_len) = product_lengths(trans, m, n);
    ensure_vector("x", x.len(), x_len, incx)?;
    ensure_vector("y", y.len(), y_len, incy)?;
    context.prepare_operands(
        &[a.region(), x.region(), y.region()],
        &[alpha.region(), beta.region()],
    )?;
    context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
    // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::gbmv(
            context.as_raw(),
            trans.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(kl, "kl")?,
            to_i64(ku, "ku")?,
            alpha.ptr(),
            a.as_device_ptr(),
            to_i64(lda, "lda")?,
            x.as_device_ptr(),
            to_i64(incx, "incx")?,
            beta.ptr(),
            y.as_mut_ptr(),
            to_i64(incy, "incy")?,
        ))
    }
}

waiting! {
    /// Computes the band matrix-vector product `y = alpha * op(A) * x + beta * y` and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`gbmv_async`] lists.
    gbmv = gbmv_async<T: Element>(
        trans: Operation,
        m: usize,
        n: usize,
        kl: usize,
        ku: usize,
        alpha: Scalar<'_, T>,
        a: &(impl DeviceRange<T> + ?Sized),
        lda: usize,
        x: &(impl DeviceRange<T> + ?Sized),
        incx: usize,
        beta: Scalar<'_, T>,
        y: &mut (impl DeviceOutput<T> + ?Sized),
        incy: usize,
    )
}

// Declares the triangular matrix-vector routines over a full `n` x `n` matrix.
macro_rules! triangular {
    ($operation:ident, $name:ident, $method:ident, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if an operand is too small, the leading dimension or increment is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: Element>(
            context: &Context,
            fill_mode: FillMode,
            trans: Operation,
            diagonal: DiagonalType,
            n: usize,
            a: &(impl DeviceRange<T> + ?Sized),
            lda: usize,
            x: &mut (impl DeviceOutput<T> + ?Sized),
            incx: usize,
        ) -> Result<()> {
            context.bind()?;
            ensure_matrix("a", a.len(), n, n, lda)?;
            ensure_vector("x", x.len(), n, incx)?;
            context.prepare_operands(&[a.region(), x.region()], &[])?;
            // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    trans.into(),
                    diagonal.into(),
                    to_i64(n, "n")?,
                    a.as_device_ptr(),
                    to_i64(lda, "lda")?,
                    x.as_mut_ptr(),
                    to_i64(incx, "incx")?,
                ))
            }
        }

        waiting! {
            #[doc = $waiting]
            ///
            /// # Errors
            ///
            /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
            $name = $operation<T: Element>(
                fill_mode: FillMode,
                trans: Operation,
                diagonal: DiagonalType,
                n: usize,
                a: &(impl DeviceRange<T> + ?Sized),
                lda: usize,
                x: &mut (impl DeviceOutput<T> + ?Sized),
                incx: usize,
            )
        }
    };
}

triangular!(
    trmv_async,
    trmv,
    trmv,
    "Enqueues `x = op(A) * x` for a triangular `n` x `n` matrix `A`.",
    "Computes `x = op(A) * x` for a triangular matrix and waits for it."
);

triangular!(
    trsv_async,
    trsv,
    trsv,
    "Enqueues the solution of `op(A) * x = b` for a triangular `n` x `n` matrix `A`, overwriting `b` in `x`.",
    "Solves `op(A) * x = b` for a triangular matrix in place and waits for it."
);

// Declares the triangular band matrix-vector routines over `k` off-diagonals.
macro_rules! triangular_band {
    ($operation:ident, $name:ident, $method:ident, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if an operand is too small, the leading dimension or increment is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: Element>(
            context: &Context,
            fill_mode: FillMode,
            trans: Operation,
            diagonal: DiagonalType,
            n: usize,
            k: usize,
            a: &(impl DeviceRange<T> + ?Sized),
            lda: usize,
            x: &mut (impl DeviceOutput<T> + ?Sized),
            incx: usize,
        ) -> Result<()> {
            context.bind()?;
            ensure_matrix("a", a.len(), band_rows(k, 0)?, n, lda)?;
            ensure_vector("x", x.len(), n, incx)?;
            context.prepare_operands(&[a.region(), x.region()], &[])?;
            // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    trans.into(),
                    diagonal.into(),
                    to_i64(n, "n")?,
                    to_i64(k, "k")?,
                    a.as_device_ptr(),
                    to_i64(lda, "lda")?,
                    x.as_mut_ptr(),
                    to_i64(incx, "incx")?,
                ))
            }
        }

        waiting! {
            #[doc = $waiting]
            ///
            /// # Errors
            ///
            /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
            $name = $operation<T: Element>(
                fill_mode: FillMode,
                trans: Operation,
                diagonal: DiagonalType,
                n: usize,
                k: usize,
                a: &(impl DeviceRange<T> + ?Sized),
                lda: usize,
                x: &mut (impl DeviceOutput<T> + ?Sized),
                incx: usize,
            )
        }
    };
}

triangular_band!(
    tbmv_async,
    tbmv,
    tbmv,
    "Enqueues `x = op(A) * x` for a triangular band matrix `A` with `k` off-diagonals.",
    "Computes `x = op(A) * x` for a triangular band matrix and waits for it."
);

triangular_band!(
    tbsv_async,
    tbsv,
    tbsv,
    "Enqueues the solution of `op(A) * x = b` for a triangular band matrix `A` with `k` off-diagonals, overwriting `b` in `x`.",
    "Solves `op(A) * x = b` for a triangular band matrix in place and waits for it."
);

// Declares the packed triangular matrix-vector routines.
macro_rules! triangular_packed {
    ($operation:ident, $name:ident, $method:ident, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if an operand is too small, the increment is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: Element>(
            context: &Context,
            fill_mode: FillMode,
            trans: Operation,
            diagonal: DiagonalType,
            n: usize,
            ap: &(impl DeviceRange<T> + ?Sized),
            x: &mut (impl DeviceOutput<T> + ?Sized),
            incx: usize,
        ) -> Result<()> {
            context.bind()?;
            ensure_packed("ap", ap.len(), n)?;
            ensure_vector("x", x.len(), n, incx)?;
            context.prepare_operands(&[ap.region(), x.region()], &[])?;
            // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    trans.into(),
                    diagonal.into(),
                    to_i64(n, "n")?,
                    ap.as_device_ptr(),
                    x.as_mut_ptr(),
                    to_i64(incx, "incx")?,
                ))
            }
        }

        waiting! {
            #[doc = $waiting]
            ///
            /// # Errors
            ///
            /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
            $name = $operation<T: Element>(
                fill_mode: FillMode,
                trans: Operation,
                diagonal: DiagonalType,
                n: usize,
                ap: &(impl DeviceRange<T> + ?Sized),
                x: &mut (impl DeviceOutput<T> + ?Sized),
                incx: usize,
            )
        }
    };
}

triangular_packed!(
    tpmv_async,
    tpmv,
    tpmv,
    "Enqueues `x = op(A) * x` for a packed triangular matrix `A`.",
    "Computes `x = op(A) * x` for a packed triangular matrix and waits for it."
);

triangular_packed!(
    tpsv_async,
    tpsv,
    tpsv,
    "Enqueues the solution of `op(A) * x = b` for a packed triangular matrix `A`, overwriting `b` in `x`.",
    "Solves `op(A) * x = b` for a packed triangular matrix in place and waits for it."
);

// Declares the symmetric or Hermitian matrix-vector products over a full `n` x `n` matrix.
macro_rules! symmetric_product {
    ($operation:ident, $name:ident, $method:ident, $bound:ident, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if an operand is too small, the leading dimension or an increment is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: $bound>(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            alpha: Scalar<'_, T>,
            a: &(impl DeviceRange<T> + ?Sized),
            lda: usize,
            x: &(impl DeviceRange<T> + ?Sized),
            incx: usize,
            beta: Scalar<'_, T>,
            y: &mut (impl DeviceOutput<T> + ?Sized),
            incy: usize,
        ) -> Result<()> {
            context.bind()?;
            ensure_matrix("a", a.len(), n, n, lda)?;
            ensure_vector("x", x.len(), n, incx)?;
            ensure_vector("y", y.len(), n, incy)?;
            context.prepare_operands(
                &[a.region(), x.region(), y.region()],
                &[alpha.region(), beta.region()],
            )?;
            context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
            // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    to_i64(n, "n")?,
                    alpha.ptr(),
                    a.as_device_ptr(),
                    to_i64(lda, "lda")?,
                    x.as_device_ptr(),
                    to_i64(incx, "incx")?,
                    beta.ptr(),
                    y.as_mut_ptr(),
                    to_i64(incy, "incy")?,
                ))
            }
        }

        waiting! {
            #[doc = $waiting]
            ///
            /// # Errors
            ///
            /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
            $name = $operation<T: $bound>(
                fill_mode: FillMode,
                n: usize,
                alpha: Scalar<'_, T>,
                a: &(impl DeviceRange<T> + ?Sized),
                lda: usize,
                x: &(impl DeviceRange<T> + ?Sized),
                incx: usize,
                beta: Scalar<'_, T>,
                y: &mut (impl DeviceOutput<T> + ?Sized),
                incy: usize,
            )
        }
    };
}

symmetric_product!(
    symv_async,
    symv,
    symv,
    Element,
    "Enqueues `y = alpha * A * x + beta * y` for a symmetric matrix `A` stored in the `fill_mode` triangle.",
    "Computes the symmetric matrix-vector product and waits for it."
);

symmetric_product!(
    hemv_async,
    hemv,
    hemv,
    ComplexElement,
    "Enqueues `y = alpha * A * x + beta * y` for a Hermitian matrix `A` stored in the `fill_mode` triangle.",
    "Computes the Hermitian matrix-vector product and waits for it."
);

// Declares the symmetric or Hermitian band matrix-vector products over `k` off-diagonals.
macro_rules! symmetric_band_product {
    ($operation:ident, $name:ident, $method:ident, $bound:ident, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if an operand is too small, the leading dimension or an increment is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: $bound>(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            k: usize,
            alpha: Scalar<'_, T>,
            a: &(impl DeviceRange<T> + ?Sized),
            lda: usize,
            x: &(impl DeviceRange<T> + ?Sized),
            incx: usize,
            beta: Scalar<'_, T>,
            y: &mut (impl DeviceOutput<T> + ?Sized),
            incy: usize,
        ) -> Result<()> {
            context.bind()?;
            ensure_matrix("a", a.len(), band_rows(k, 0)?, n, lda)?;
            ensure_vector("x", x.len(), n, incx)?;
            ensure_vector("y", y.len(), n, incy)?;
            context.prepare_operands(
                &[a.region(), x.region(), y.region()],
                &[alpha.region(), beta.region()],
            )?;
            context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
            // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    to_i64(n, "n")?,
                    to_i64(k, "k")?,
                    alpha.ptr(),
                    a.as_device_ptr(),
                    to_i64(lda, "lda")?,
                    x.as_device_ptr(),
                    to_i64(incx, "incx")?,
                    beta.ptr(),
                    y.as_mut_ptr(),
                    to_i64(incy, "incy")?,
                ))
            }
        }

        waiting! {
            #[doc = $waiting]
            ///
            /// # Errors
            ///
            /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
            $name = $operation<T: $bound>(
                fill_mode: FillMode,
                n: usize,
                k: usize,
                alpha: Scalar<'_, T>,
                a: &(impl DeviceRange<T> + ?Sized),
                lda: usize,
                x: &(impl DeviceRange<T> + ?Sized),
                incx: usize,
                beta: Scalar<'_, T>,
                y: &mut (impl DeviceOutput<T> + ?Sized),
                incy: usize,
            )
        }
    };
}

symmetric_band_product!(
    sbmv_async,
    sbmv,
    sbmv,
    RealElement,
    "Enqueues `y = alpha * A * x + beta * y` for a symmetric band matrix `A` with `k` off-diagonals.",
    "Computes the symmetric band matrix-vector product and waits for it."
);

symmetric_band_product!(
    hbmv_async,
    hbmv,
    hbmv,
    ComplexElement,
    "Enqueues `y = alpha * A * x + beta * y` for a Hermitian band matrix `A` with `k` off-diagonals.",
    "Computes the Hermitian band matrix-vector product and waits for it."
);

// Declares the packed symmetric or Hermitian matrix-vector products.
macro_rules! symmetric_packed_product {
    ($operation:ident, $name:ident, $method:ident, $bound:ident, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if an operand is too small, an increment is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: $bound>(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            alpha: Scalar<'_, T>,
            ap: &(impl DeviceRange<T> + ?Sized),
            x: &(impl DeviceRange<T> + ?Sized),
            incx: usize,
            beta: Scalar<'_, T>,
            y: &mut (impl DeviceOutput<T> + ?Sized),
            incy: usize,
        ) -> Result<()> {
            context.bind()?;
            ensure_packed("ap", ap.len(), n)?;
            ensure_vector("x", x.len(), n, incx)?;
            ensure_vector("y", y.len(), n, incy)?;
            context.prepare_operands(
                &[ap.region(), x.region(), y.region()],
                &[alpha.region(), beta.region()],
            )?;
            context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
            // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    to_i64(n, "n")?,
                    alpha.ptr(),
                    ap.as_device_ptr(),
                    x.as_device_ptr(),
                    to_i64(incx, "incx")?,
                    beta.ptr(),
                    y.as_mut_ptr(),
                    to_i64(incy, "incy")?,
                ))
            }
        }

        waiting! {
            #[doc = $waiting]
            ///
            /// # Errors
            ///
            /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
            $name = $operation<T: $bound>(
                fill_mode: FillMode,
                n: usize,
                alpha: Scalar<'_, T>,
                ap: &(impl DeviceRange<T> + ?Sized),
                x: &(impl DeviceRange<T> + ?Sized),
                incx: usize,
                beta: Scalar<'_, T>,
                y: &mut (impl DeviceOutput<T> + ?Sized),
                incy: usize,
            )
        }
    };
}

symmetric_packed_product!(
    spmv_async,
    spmv,
    spmv,
    RealElement,
    "Enqueues `y = alpha * A * x + beta * y` for a packed symmetric matrix `A`.",
    "Computes the packed symmetric matrix-vector product and waits for it."
);

symmetric_packed_product!(
    hpmv_async,
    hpmv,
    hpmv,
    ComplexElement,
    "Enqueues `y = alpha * A * x + beta * y` for a packed Hermitian matrix `A`.",
    "Computes the packed Hermitian matrix-vector product and waits for it."
);

// Declares the general rank-1 updates of an `m` x `n` matrix.
macro_rules! general_rank1 {
    ($operation:ident, $name:ident, $method:ident, $bound:ident, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if an operand is too small, the leading dimension or an increment is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: $bound>(
            context: &Context,
            m: usize,
            n: usize,
            alpha: Scalar<'_, T>,
            x: &(impl DeviceRange<T> + ?Sized),
            incx: usize,
            y: &(impl DeviceRange<T> + ?Sized),
            incy: usize,
            a: &mut (impl DeviceOutput<T> + ?Sized),
            lda: usize,
        ) -> Result<()> {
            context.bind()?;
            ensure_vector("x", x.len(), m, incx)?;
            ensure_vector("y", y.len(), n, incy)?;
            ensure_matrix("a", a.len(), m, n, lda)?;
            context.prepare_operands(&[x.region(), y.region(), a.region()], &[alpha.region()])?;
            context.use_pointer_mode(&[alpha.pointer_mode()])?;
            // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    to_i64(m, "m")?,
                    to_i64(n, "n")?,
                    alpha.ptr(),
                    x.as_device_ptr(),
                    to_i64(incx, "incx")?,
                    y.as_device_ptr(),
                    to_i64(incy, "incy")?,
                    a.as_mut_ptr(),
                    to_i64(lda, "lda")?,
                ))
            }
        }

        waiting! {
            #[doc = $waiting]
            ///
            /// # Errors
            ///
            /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
            $name = $operation<T: $bound>(
                m: usize,
                n: usize,
                alpha: Scalar<'_, T>,
                x: &(impl DeviceRange<T> + ?Sized),
                incx: usize,
                y: &(impl DeviceRange<T> + ?Sized),
                incy: usize,
                a: &mut (impl DeviceOutput<T> + ?Sized),
                lda: usize,
            )
        }
    };
}

general_rank1!(
    ger_async,
    ger,
    ger,
    Element,
    "Enqueues the unconjugated rank-1 update `A = alpha * x * y^T + A`.",
    "Computes the unconjugated rank-1 update and waits for it."
);

general_rank1!(
    gerc_async,
    gerc,
    gerc,
    ComplexElement,
    "Enqueues the conjugated rank-1 update `A = alpha * x * y^H + A`.",
    "Computes the conjugated rank-1 update and waits for it."
);

// Declares the symmetric or Hermitian rank-1 updates of a full `n` x `n` matrix.
macro_rules! symmetric_rank1 {
    ($operation:ident, $name:ident, $method:ident, $bound:ident, $alpha:ty, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if an operand is too small, the leading dimension or increment is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: $bound>(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            alpha: Scalar<'_, $alpha>,
            x: &(impl DeviceRange<T> + ?Sized),
            incx: usize,
            a: &mut (impl DeviceOutput<T> + ?Sized),
            lda: usize,
        ) -> Result<()> {
            context.bind()?;
            ensure_vector("x", x.len(), n, incx)?;
            ensure_matrix("a", a.len(), n, n, lda)?;
            context.prepare_operands(&[x.region(), a.region()], &[alpha.region()])?;
            context.use_pointer_mode(&[alpha.pointer_mode()])?;
            // SAFETY: the operands hold the checked extents in this context, ordered after their producers, and `alpha` has the type the routine reads; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    to_i64(n, "n")?,
                    alpha.ptr().cast(),
                    x.as_device_ptr(),
                    to_i64(incx, "incx")?,
                    a.as_mut_ptr(),
                    to_i64(lda, "lda")?,
                ))
            }
        }

        waiting! {
            #[doc = $waiting]
            ///
            /// # Errors
            ///
            /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
            $name = $operation<T: $bound>(
                fill_mode: FillMode,
                n: usize,
                alpha: Scalar<'_, $alpha>,
                x: &(impl DeviceRange<T> + ?Sized),
                incx: usize,
                a: &mut (impl DeviceOutput<T> + ?Sized),
                lda: usize,
            )
        }
    };
}

symmetric_rank1!(
    syr_async,
    syr,
    syr,
    Element,
    T,
    "Enqueues the symmetric rank-1 update `A = alpha * x * x^T + A` of the `fill_mode` triangle.",
    "Computes the symmetric rank-1 update and waits for it."
);

symmetric_rank1!(
    her_async,
    her,
    her,
    ComplexElement,
    T::Real,
    "Enqueues the Hermitian rank-1 update `A = alpha * x * x^H + A` of the `fill_mode` triangle, with a real `alpha`.",
    "Computes the Hermitian rank-1 update and waits for it."
);

// Declares the packed symmetric or Hermitian rank-1 updates.
macro_rules! symmetric_packed_rank1 {
    ($operation:ident, $name:ident, $method:ident, $bound:ident, $alpha:ty, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if an operand is too small, the increment is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: $bound>(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            alpha: Scalar<'_, $alpha>,
            x: &(impl DeviceRange<T> + ?Sized),
            incx: usize,
            ap: &mut (impl DeviceOutput<T> + ?Sized),
        ) -> Result<()> {
            context.bind()?;
            ensure_vector("x", x.len(), n, incx)?;
            ensure_packed("ap", ap.len(), n)?;
            context.prepare_operands(&[x.region(), ap.region()], &[alpha.region()])?;
            context.use_pointer_mode(&[alpha.pointer_mode()])?;
            // SAFETY: the operands hold the checked extents in this context, ordered after their producers, and `alpha` has the type the routine reads; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    to_i64(n, "n")?,
                    alpha.ptr().cast(),
                    x.as_device_ptr(),
                    to_i64(incx, "incx")?,
                    ap.as_mut_ptr(),
                ))
            }
        }

        waiting! {
            #[doc = $waiting]
            ///
            /// # Errors
            ///
            /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
            $name = $operation<T: $bound>(
                fill_mode: FillMode,
                n: usize,
                alpha: Scalar<'_, $alpha>,
                x: &(impl DeviceRange<T> + ?Sized),
                incx: usize,
                ap: &mut (impl DeviceOutput<T> + ?Sized),
            )
        }
    };
}

symmetric_packed_rank1!(
    spr_async,
    spr,
    spr,
    RealElement,
    T,
    "Enqueues the packed symmetric rank-1 update `A = alpha * x * x^T + A`.",
    "Computes the packed symmetric rank-1 update and waits for it."
);

symmetric_packed_rank1!(
    hpr_async,
    hpr,
    hpr,
    ComplexElement,
    T::Real,
    "Enqueues the packed Hermitian rank-1 update `A = alpha * x * x^H + A`, with a real `alpha`.",
    "Computes the packed Hermitian rank-1 update and waits for it."
);

// Declares the symmetric or Hermitian rank-2 updates of a full `n` x `n` matrix.
macro_rules! symmetric_rank2 {
    ($operation:ident, $name:ident, $method:ident, $bound:ident, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if an operand is too small, the leading dimension or an increment is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: $bound>(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            alpha: Scalar<'_, T>,
            x: &(impl DeviceRange<T> + ?Sized),
            incx: usize,
            y: &(impl DeviceRange<T> + ?Sized),
            incy: usize,
            a: &mut (impl DeviceOutput<T> + ?Sized),
            lda: usize,
        ) -> Result<()> {
            context.bind()?;
            ensure_vector("x", x.len(), n, incx)?;
            ensure_vector("y", y.len(), n, incy)?;
            ensure_matrix("a", a.len(), n, n, lda)?;
            context.prepare_operands(&[x.region(), y.region(), a.region()], &[alpha.region()])?;
            context.use_pointer_mode(&[alpha.pointer_mode()])?;
            // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    to_i64(n, "n")?,
                    alpha.ptr(),
                    x.as_device_ptr(),
                    to_i64(incx, "incx")?,
                    y.as_device_ptr(),
                    to_i64(incy, "incy")?,
                    a.as_mut_ptr(),
                    to_i64(lda, "lda")?,
                ))
            }
        }

        waiting! {
            #[doc = $waiting]
            ///
            /// # Errors
            ///
            /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
            $name = $operation<T: $bound>(
                fill_mode: FillMode,
                n: usize,
                alpha: Scalar<'_, T>,
                x: &(impl DeviceRange<T> + ?Sized),
                incx: usize,
                y: &(impl DeviceRange<T> + ?Sized),
                incy: usize,
                a: &mut (impl DeviceOutput<T> + ?Sized),
                lda: usize,
            )
        }
    };
}

symmetric_rank2!(
    syr2_async,
    syr2,
    syr2,
    Element,
    "Enqueues the symmetric rank-2 update `A = alpha * (x * y^T + y * x^T) + A` of the `fill_mode` triangle.",
    "Computes the symmetric rank-2 update and waits for it."
);

symmetric_rank2!(
    her2_async,
    her2,
    her2,
    ComplexElement,
    "Enqueues the Hermitian rank-2 update `A = alpha * x * y^H + conj(alpha) * y * x^H + A` of the `fill_mode` triangle.",
    "Computes the Hermitian rank-2 update and waits for it."
);

// Declares the packed symmetric or Hermitian rank-2 updates.
macro_rules! symmetric_packed_rank2 {
    ($operation:ident, $name:ident, $method:ident, $bound:ident, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if an operand is too small, an increment is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: $bound>(
            context: &Context,
            fill_mode: FillMode,
            n: usize,
            alpha: Scalar<'_, T>,
            x: &(impl DeviceRange<T> + ?Sized),
            incx: usize,
            y: &(impl DeviceRange<T> + ?Sized),
            incy: usize,
            ap: &mut (impl DeviceOutput<T> + ?Sized),
        ) -> Result<()> {
            context.bind()?;
            ensure_vector("x", x.len(), n, incx)?;
            ensure_vector("y", y.len(), n, incy)?;
            ensure_packed("ap", ap.len(), n)?;
            context.prepare_operands(&[x.region(), y.region(), ap.region()], &[alpha.region()])?;
            context.use_pointer_mode(&[alpha.pointer_mode()])?;
            // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    to_i64(n, "n")?,
                    alpha.ptr(),
                    x.as_device_ptr(),
                    to_i64(incx, "incx")?,
                    y.as_device_ptr(),
                    to_i64(incy, "incy")?,
                    ap.as_mut_ptr(),
                ))
            }
        }

        waiting! {
            #[doc = $waiting]
            ///
            /// # Errors
            ///
            /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
            $name = $operation<T: $bound>(
                fill_mode: FillMode,
                n: usize,
                alpha: Scalar<'_, T>,
                x: &(impl DeviceRange<T> + ?Sized),
                incx: usize,
                y: &(impl DeviceRange<T> + ?Sized),
                incy: usize,
                ap: &mut (impl DeviceOutput<T> + ?Sized),
            )
        }
    };
}

symmetric_packed_rank2!(
    spr2_async,
    spr2,
    spr2,
    RealElement,
    "Enqueues the packed symmetric rank-2 update `A = alpha * (x * y^T + y * x^T) + A`.",
    "Computes the packed symmetric rank-2 update and waits for it."
);

symmetric_packed_rank2!(
    hpr2_async,
    hpr2,
    hpr2,
    ComplexElement,
    "Enqueues the packed Hermitian rank-2 update `A = alpha * x * y^H + conj(alpha) * y * x^H + A`.",
    "Computes the packed Hermitian rank-2 update and waits for it."
);
