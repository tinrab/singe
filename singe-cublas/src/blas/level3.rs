#![allow(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    reason = "BLAS operations take each operand, dimension, and scalar separately, named as in BLAS"
)]

use singe_cuda::{
    memory::DeviceMemory,
    stream::AsStream,
    types::DevicePtr,
    view::{DeviceInputView, DeviceOutput, DeviceOutputView, DeviceRange, DeviceRegionProvider},
};

use crate::{
    blas::{
        ComplexElement, Element, RealElement,
        shape::{
            ensure_matrix, ensure_packed, ensure_strided, ensure_vector, matrix_span, side_order,
            stored,
        },
    },
    context::Context,
    error::{Error, Result},
    gemm::GemmShape,
    scalar::Scalar,
    sys, try_ffi,
    types::{DiagonalType, FillMode, Operation, PointerMode, SideMode},
    utility::{to_i32, to_i64, validate_min_len, validate_same_len},
};

// Checks the operands of `C = alpha * op(A) * op(B) + beta * C` against the shape.
fn ensure_gemm(
    shape: GemmShape,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    let (a_rows, a_cols) = stored(shape.transpose_a(), shape.m(), shape.k());
    let (b_rows, b_cols) = stored(shape.transpose_b(), shape.k(), shape.n());
    ensure_matrix("a", a_len, a_rows, a_cols, lda)?;
    ensure_matrix("b", b_len, b_rows, b_cols, ldb)?;
    ensure_matrix("c", c_len, shape.m(), shape.n(), ldc)
}

/// Enqueues `C = alpha * op(A) * op(B) + beta * C`, with `op(A)` `m` x `k`, `op(B)` `k` x `n`, and `C` `m` x `n`.
///
/// Matrices are column-major, and leading dimensions count elements.
/// `C` is read only when `beta` is nonzero.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a matrix is too small or a leading dimension is invalid for the shape, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn gemm_async<T: Element>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar<'_, T>,
    a: &(impl DeviceRange<T> + ?Sized),
    lda: usize,
    b: &(impl DeviceRange<T> + ?Sized),
    ldb: usize,
    beta: Scalar<'_, T>,
    c: &mut (impl DeviceOutput<T> + ?Sized),
    ldc: usize,
) -> Result<()> {
    context.bind()?;
    let shape = GemmShape::new(m, n, k)
        .with_transpose_a(transpose_a)
        .with_transpose_b(transpose_b);
    ensure_gemm(shape, a.len(), lda, b.len(), ldb, c.len(), ldc)?;
    context.prepare_operands(
        &[a.region(), b.region(), c.region()],
        &[alpha.region(), beta.region()],
    )?;
    context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
    // SAFETY: the matrices hold the checked shapes in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::gemm(
            context.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            alpha.ptr(),
            a.as_device_ptr(),
            to_i64(lda, "lda")?,
            b.as_device_ptr(),
            to_i64(ldb, "ldb")?,
            beta.ptr(),
            c.as_mut_ptr(),
            to_i64(ldc, "ldc")?,
        ))
    }
}

waiting! {
    /// Computes `C = alpha * op(A) * op(B) + beta * C` and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`gemm_async`] lists.
    gemm = gemm_async<T: Element>(
        transpose_a: Operation,
        transpose_b: Operation,
        m: usize,
        n: usize,
        k: usize,
        alpha: Scalar<'_, T>,
        a: &(impl DeviceRange<T> + ?Sized),
        lda: usize,
        b: &(impl DeviceRange<T> + ?Sized),
        ldb: usize,
        beta: Scalar<'_, T>,
        c: &mut (impl DeviceOutput<T> + ?Sized),
        ldc: usize,
    )
}

/// Enqueues the complex product `C = alpha * op(A) * op(B) + beta * C` with the Gauss three-multiplication method.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error for the reasons [`gemm_async`] lists.
pub unsafe fn gemm3m_async<T: ComplexElement>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar<'_, T>,
    a: &(impl DeviceRange<T> + ?Sized),
    lda: usize,
    b: &(impl DeviceRange<T> + ?Sized),
    ldb: usize,
    beta: Scalar<'_, T>,
    c: &mut (impl DeviceOutput<T> + ?Sized),
    ldc: usize,
) -> Result<()> {
    context.bind()?;
    let shape = GemmShape::new(m, n, k)
        .with_transpose_a(transpose_a)
        .with_transpose_b(transpose_b);
    ensure_gemm(shape, a.len(), lda, b.len(), ldb, c.len(), ldc)?;
    context.prepare_operands(
        &[a.region(), b.region(), c.region()],
        &[alpha.region(), beta.region()],
    )?;
    context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
    // SAFETY: the matrices hold the checked shapes in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::gemm3m(
            context.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            alpha.ptr(),
            a.as_device_ptr(),
            to_i64(lda, "lda")?,
            b.as_device_ptr(),
            to_i64(ldb, "ldb")?,
            beta.ptr(),
            c.as_mut_ptr(),
            to_i64(ldc, "ldc")?,
        ))
    }
}

waiting! {
    /// Computes the complex product with the Gauss three-multiplication method and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`gemm3m_async`] lists.
    gemm3m = gemm3m_async<T: ComplexElement>(
        transpose_a: Operation,
        transpose_b: Operation,
        m: usize,
        n: usize,
        k: usize,
        alpha: Scalar<'_, T>,
        a: &(impl DeviceRange<T> + ?Sized),
        lda: usize,
        b: &(impl DeviceRange<T> + ?Sized),
        ldb: usize,
        beta: Scalar<'_, T>,
        c: &mut (impl DeviceOutput<T> + ?Sized),
        ldc: usize,
    )
}

// Declares the symmetric or Hermitian matrix-matrix products.
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
        /// Returns an error if a matrix is too small or a leading dimension is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: $bound>(
            context: &Context,
            side: SideMode,
            fill_mode: FillMode,
            m: usize,
            n: usize,
            alpha: Scalar<'_, T>,
            a: &(impl DeviceRange<T> + ?Sized),
            lda: usize,
            b: &(impl DeviceRange<T> + ?Sized),
            ldb: usize,
            beta: Scalar<'_, T>,
            c: &mut (impl DeviceOutput<T> + ?Sized),
            ldc: usize,
        ) -> Result<()> {
            context.bind()?;
            let order = side_order(side, m, n);
            ensure_matrix("a", a.len(), order, order, lda)?;
            ensure_matrix("b", b.len(), m, n, ldb)?;
            ensure_matrix("c", c.len(), m, n, ldc)?;
            context.prepare_operands(
                &[a.region(), b.region(), c.region()],
                &[alpha.region(), beta.region()],
            )?;
            context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
            // SAFETY: the matrices hold the checked shapes in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    side.into(),
                    fill_mode.into(),
                    to_i64(m, "m")?,
                    to_i64(n, "n")?,
                    alpha.ptr(),
                    a.as_device_ptr(),
                    to_i64(lda, "lda")?,
                    b.as_device_ptr(),
                    to_i64(ldb, "ldb")?,
                    beta.ptr(),
                    c.as_mut_ptr(),
                    to_i64(ldc, "ldc")?,
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
                side: SideMode,
                fill_mode: FillMode,
                m: usize,
                n: usize,
                alpha: Scalar<'_, T>,
                a: &(impl DeviceRange<T> + ?Sized),
                lda: usize,
                b: &(impl DeviceRange<T> + ?Sized),
                ldb: usize,
                beta: Scalar<'_, T>,
                c: &mut (impl DeviceOutput<T> + ?Sized),
                ldc: usize,
            )
        }
    };
}

symmetric_product!(
    symm_async,
    symm,
    symm,
    Element,
    "Enqueues `C = alpha * A * B + beta * C` (left side) or `C = alpha * B * A + beta * C` (right side) for a symmetric matrix `A` stored in the `fill_mode` triangle.",
    "Computes the symmetric matrix-matrix product and waits for it."
);

symmetric_product!(
    hemm_async,
    hemm,
    hemm,
    ComplexElement,
    "Enqueues `C = alpha * A * B + beta * C` (left side) or `C = alpha * B * A + beta * C` (right side) for a Hermitian matrix `A` stored in the `fill_mode` triangle.",
    "Computes the Hermitian matrix-matrix product and waits for it."
);

// Declares the symmetric or Hermitian rank-k updates.
macro_rules! rank_k {
    ($operation:ident, $name:ident, $method:ident, $bound:ident, $scalar:ty, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if a matrix is too small or a leading dimension is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: $bound>(
            context: &Context,
            fill_mode: FillMode,
            trans: Operation,
            n: usize,
            k: usize,
            alpha: Scalar<'_, $scalar>,
            a: &(impl DeviceRange<T> + ?Sized),
            lda: usize,
            beta: Scalar<'_, $scalar>,
            c: &mut (impl DeviceOutput<T> + ?Sized),
            ldc: usize,
        ) -> Result<()> {
            context.bind()?;
            let (a_rows, a_cols) = stored(trans, n, k);
            ensure_matrix("a", a.len(), a_rows, a_cols, lda)?;
            ensure_matrix("c", c.len(), n, n, ldc)?;
            context
                .prepare_operands(&[a.region(), c.region()], &[alpha.region(), beta.region()])?;
            context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
            // SAFETY: the matrices hold the checked shapes in this context, ordered after their producers, and the scalars have the type the routine reads; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    trans.into(),
                    to_i64(n, "n")?,
                    to_i64(k, "k")?,
                    alpha.ptr().cast(),
                    a.as_device_ptr(),
                    to_i64(lda, "lda")?,
                    beta.ptr().cast(),
                    c.as_mut_ptr(),
                    to_i64(ldc, "ldc")?,
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
                trans: Operation,
                n: usize,
                k: usize,
                alpha: Scalar<'_, $scalar>,
                a: &(impl DeviceRange<T> + ?Sized),
                lda: usize,
                beta: Scalar<'_, $scalar>,
                c: &mut (impl DeviceOutput<T> + ?Sized),
                ldc: usize,
            )
        }
    };
}

rank_k!(
    syrk_async,
    syrk,
    syrk,
    Element,
    T,
    "Enqueues the symmetric rank-k update `C = alpha * op(A) * op(A)^T + beta * C` of the `fill_mode` triangle, with `op(A)` `n` x `k`.",
    "Computes the symmetric rank-k update and waits for it."
);

rank_k!(
    herk_async,
    herk,
    herk,
    ComplexElement,
    T::Real,
    "Enqueues the Hermitian rank-k update `C = alpha * op(A) * op(A)^H + beta * C` of the `fill_mode` triangle, with real scalars and `op(A)` `n` x `k`.",
    "Computes the Hermitian rank-k update and waits for it."
);

// Declares the symmetric or Hermitian rank-2k updates and their two-input rank-k variants.
macro_rules! rank_2k {
    ($operation:ident, $name:ident, $method:ident, $bound:ident, $beta:ty, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if a matrix is too small or a leading dimension is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
        pub unsafe fn $operation<T: $bound>(
            context: &Context,
            fill_mode: FillMode,
            trans: Operation,
            n: usize,
            k: usize,
            alpha: Scalar<'_, T>,
            a: &(impl DeviceRange<T> + ?Sized),
            lda: usize,
            b: &(impl DeviceRange<T> + ?Sized),
            ldb: usize,
            beta: Scalar<'_, $beta>,
            c: &mut (impl DeviceOutput<T> + ?Sized),
            ldc: usize,
        ) -> Result<()> {
            context.bind()?;
            let (rows, cols) = stored(trans, n, k);
            ensure_matrix("a", a.len(), rows, cols, lda)?;
            ensure_matrix("b", b.len(), rows, cols, ldb)?;
            ensure_matrix("c", c.len(), n, n, ldc)?;
            context.prepare_operands(
                &[a.region(), b.region(), c.region()],
                &[alpha.region(), beta.region()],
            )?;
            context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
            // SAFETY: the matrices hold the checked shapes in this context, ordered after their producers, and the scalars have the types the routine reads; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(T::$method(
                    context.as_raw(),
                    fill_mode.into(),
                    trans.into(),
                    to_i64(n, "n")?,
                    to_i64(k, "k")?,
                    alpha.ptr(),
                    a.as_device_ptr(),
                    to_i64(lda, "lda")?,
                    b.as_device_ptr(),
                    to_i64(ldb, "ldb")?,
                    beta.ptr().cast(),
                    c.as_mut_ptr(),
                    to_i64(ldc, "ldc")?,
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
                trans: Operation,
                n: usize,
                k: usize,
                alpha: Scalar<'_, T>,
                a: &(impl DeviceRange<T> + ?Sized),
                lda: usize,
                b: &(impl DeviceRange<T> + ?Sized),
                ldb: usize,
                beta: Scalar<'_, $beta>,
                c: &mut (impl DeviceOutput<T> + ?Sized),
                ldc: usize,
            )
        }
    };
}

rank_2k!(
    syr2k_async,
    syr2k,
    syr2k,
    Element,
    T,
    "Enqueues the symmetric rank-2k update `C = alpha * (op(A) * op(B)^T + op(B) * op(A)^T) + beta * C` of the `fill_mode` triangle.",
    "Computes the symmetric rank-2k update and waits for it."
);

rank_2k!(
    her2k_async,
    her2k,
    her2k,
    ComplexElement,
    T::Real,
    "Enqueues the Hermitian rank-2k update `C = alpha * op(A) * op(B)^H + conj(alpha) * op(B) * op(A)^H + beta * C` of the `fill_mode` triangle, with a real `beta`.",
    "Computes the Hermitian rank-2k update and waits for it."
);

rank_2k!(
    syrkx_async,
    syrkx,
    syrkx,
    Element,
    T,
    "Enqueues `C = alpha * op(A) * op(B)^T + beta * C` on the `fill_mode` triangle, for products known to be symmetric.",
    "Computes the symmetric rank-k update with two inputs and waits for it."
);

rank_2k!(
    herkx_async,
    herkx,
    herkx,
    ComplexElement,
    T::Real,
    "Enqueues `C = alpha * op(A) * op(B)^H + beta * C` on the `fill_mode` triangle, for products known to be Hermitian, with a real `beta`.",
    "Computes the Hermitian rank-k update with two inputs and waits for it."
);

/// Enqueues `C = alpha * op(A) * B` (left side) or `C = alpha * B * op(A)` (right side) for a triangular matrix `A`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a matrix is too small or a leading dimension is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn trmm_async<T: Element>(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    trans: Operation,
    diagonal: DiagonalType,
    m: usize,
    n: usize,
    alpha: Scalar<'_, T>,
    a: &(impl DeviceRange<T> + ?Sized),
    lda: usize,
    b: &(impl DeviceRange<T> + ?Sized),
    ldb: usize,
    c: &mut (impl DeviceOutput<T> + ?Sized),
    ldc: usize,
) -> Result<()> {
    context.bind()?;
    let order = side_order(side, m, n);
    ensure_matrix("a", a.len(), order, order, lda)?;
    ensure_matrix("b", b.len(), m, n, ldb)?;
    ensure_matrix("c", c.len(), m, n, ldc)?;
    context.prepare_operands(&[a.region(), b.region(), c.region()], &[alpha.region()])?;
    context.use_pointer_mode(&[alpha.pointer_mode()])?;
    // SAFETY: the matrices hold the checked shapes in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::trmm(
            context.as_raw(),
            side.into(),
            fill_mode.into(),
            trans.into(),
            diagonal.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            alpha.ptr(),
            a.as_device_ptr(),
            to_i64(lda, "lda")?,
            b.as_device_ptr(),
            to_i64(ldb, "ldb")?,
            c.as_mut_ptr(),
            to_i64(ldc, "ldc")?,
        ))
    }
}

waiting! {
    /// Computes the triangular matrix-matrix product and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`trmm_async`] lists.
    trmm = trmm_async<T: Element>(
        side: SideMode,
        fill_mode: FillMode,
        trans: Operation,
        diagonal: DiagonalType,
        m: usize,
        n: usize,
        alpha: Scalar<'_, T>,
        a: &(impl DeviceRange<T> + ?Sized),
        lda: usize,
        b: &(impl DeviceRange<T> + ?Sized),
        ldb: usize,
        c: &mut (impl DeviceOutput<T> + ?Sized),
        ldc: usize,
    )
}

/// Enqueues the solution of `op(A) * X = alpha * B` (left side) or `X * op(A) = alpha * B` (right side) for a triangular matrix `A`, overwriting `B` with `X`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a matrix is too small or a leading dimension is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn trsm_async<T: Element>(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    trans: Operation,
    diagonal: DiagonalType,
    m: usize,
    n: usize,
    alpha: Scalar<'_, T>,
    a: &(impl DeviceRange<T> + ?Sized),
    lda: usize,
    b: &mut (impl DeviceOutput<T> + ?Sized),
    ldb: usize,
) -> Result<()> {
    context.bind()?;
    let order = side_order(side, m, n);
    ensure_matrix("a", a.len(), order, order, lda)?;
    ensure_matrix("b", b.len(), m, n, ldb)?;
    context.prepare_operands(&[a.region(), b.region()], &[alpha.region()])?;
    context.use_pointer_mode(&[alpha.pointer_mode()])?;
    // SAFETY: the matrices hold the checked shapes in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::trsm(
            context.as_raw(),
            side.into(),
            fill_mode.into(),
            trans.into(),
            diagonal.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            alpha.ptr(),
            a.as_device_ptr(),
            to_i64(lda, "lda")?,
            b.as_mut_ptr(),
            to_i64(ldb, "ldb")?,
        ))
    }
}

waiting! {
    /// Solves the triangular system with multiple right-hand sides in place and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`trsm_async`] lists.
    trsm = trsm_async<T: Element>(
        side: SideMode,
        fill_mode: FillMode,
        trans: Operation,
        diagonal: DiagonalType,
        m: usize,
        n: usize,
        alpha: Scalar<'_, T>,
        a: &(impl DeviceRange<T> + ?Sized),
        lda: usize,
        b: &mut (impl DeviceOutput<T> + ?Sized),
        ldb: usize,
    )
}

/// Enqueues `C = alpha * op(A) + beta * op(B)` for `m` x `n` results.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a matrix is too small or a leading dimension is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn geam_async<T: Element>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    alpha: Scalar<'_, T>,
    a: &(impl DeviceRange<T> + ?Sized),
    lda: usize,
    beta: Scalar<'_, T>,
    b: &(impl DeviceRange<T> + ?Sized),
    ldb: usize,
    c: &mut (impl DeviceOutput<T> + ?Sized),
    ldc: usize,
) -> Result<()> {
    context.bind()?;
    let (a_rows, a_cols) = stored(transpose_a, m, n);
    let (b_rows, b_cols) = stored(transpose_b, m, n);
    ensure_matrix("a", a.len(), a_rows, a_cols, lda)?;
    ensure_matrix("b", b.len(), b_rows, b_cols, ldb)?;
    ensure_matrix("c", c.len(), m, n, ldc)?;
    context.prepare_operands(
        &[a.region(), b.region(), c.region()],
        &[alpha.region(), beta.region()],
    )?;
    context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
    // SAFETY: the matrices hold the checked shapes in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::geam(
            context.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            alpha.ptr(),
            a.as_device_ptr(),
            to_i64(lda, "lda")?,
            beta.ptr(),
            b.as_device_ptr(),
            to_i64(ldb, "ldb")?,
            c.as_mut_ptr(),
            to_i64(ldc, "ldc")?,
        ))
    }
}

waiting! {
    /// Computes `C = alpha * op(A) + beta * op(B)` and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`geam_async`] lists.
    geam = geam_async<T: Element>(
        transpose_a: Operation,
        transpose_b: Operation,
        m: usize,
        n: usize,
        alpha: Scalar<'_, T>,
        a: &(impl DeviceRange<T> + ?Sized),
        lda: usize,
        beta: Scalar<'_, T>,
        b: &(impl DeviceRange<T> + ?Sized),
        ldb: usize,
        c: &mut (impl DeviceOutput<T> + ?Sized),
        ldc: usize,
    )
}

/// Enqueues `C = A * diag(x)` (right side) or `C = diag(x) * A` (left side) for `m` x `n` matrices.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if an operand is too small, a leading dimension or the increment is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn dgmm_async<T: Element>(
    context: &Context,
    side: SideMode,
    m: usize,
    n: usize,
    a: &(impl DeviceRange<T> + ?Sized),
    lda: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    c: &mut (impl DeviceOutput<T> + ?Sized),
    ldc: usize,
) -> Result<()> {
    context.bind()?;
    ensure_matrix("a", a.len(), m, n, lda)?;
    ensure_vector("x", x.len(), side_order(side, m, n), incx)?;
    ensure_matrix("c", c.len(), m, n, ldc)?;
    context.prepare_operands(&[a.region(), x.region(), c.region()], &[])?;
    // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::dgmm(
            context.as_raw(),
            side.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a.as_device_ptr(),
            to_i64(lda, "lda")?,
            x.as_device_ptr(),
            to_i64(incx, "incx")?,
            c.as_mut_ptr(),
            to_i64(ldc, "ldc")?,
        ))
    }
}

waiting! {
    /// Scales the rows or columns of `A` by `x` into `C` and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`dgmm_async`] lists.
    dgmm = dgmm_async<T: Element>(
        side: SideMode,
        m: usize,
        n: usize,
        a: &(impl DeviceRange<T> + ?Sized),
        lda: usize,
        x: &(impl DeviceRange<T> + ?Sized),
        incx: usize,
        c: &mut (impl DeviceOutput<T> + ?Sized),
        ldc: usize,
    )
}

/// Enqueues the unpacking of the `fill_mode` triangle of packed `ap` into the full-storage matrix `a`.
///
/// Elements of `a` outside the triangle are left unchanged.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if an operand is too small, the leading dimension is invalid, a value exceeds `i32`, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn tpttr_async<T: Element>(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    ap: &(impl DeviceRange<T> + ?Sized),
    a: &mut (impl DeviceOutput<T> + ?Sized),
    lda: usize,
) -> Result<()> {
    context.bind()?;
    ensure_packed("ap", ap.len(), n)?;
    ensure_matrix("a", a.len(), n, n, lda)?;
    context.prepare_operands(&[ap.region(), a.region()], &[])?;
    // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::tpttr(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ap.as_device_ptr(),
            a.as_mut_ptr(),
            to_i32(lda, "lda")?,
        ))
    }
}

waiting! {
    /// Unpacks a packed triangle into full storage and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`tpttr_async`] lists.
    tpttr = tpttr_async<T: Element>(
        fill_mode: FillMode,
        n: usize,
        ap: &(impl DeviceRange<T> + ?Sized),
        a: &mut (impl DeviceOutput<T> + ?Sized),
        lda: usize,
    )
}

/// Enqueues the packing of the `fill_mode` triangle of the full-storage matrix `a` into `ap`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if an operand is too small, the leading dimension is invalid, a value exceeds `i32`, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn trttp_async<T: Element>(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &(impl DeviceRange<T> + ?Sized),
    lda: usize,
    ap: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    context.bind()?;
    ensure_matrix("a", a.len(), n, n, lda)?;
    ensure_packed("ap", ap.len(), n)?;
    context.prepare_operands(&[a.region(), ap.region()], &[])?;
    // SAFETY: the operands hold the checked extents in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::trttp(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_device_ptr(),
            to_i32(lda, "lda")?,
            ap.as_mut_ptr(),
        ))
    }
}

waiting! {
    /// Packs a triangle of full storage and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`trttp_async`] lists.
    trttp = trttp_async<T: Element>(
        fill_mode: FillMode,
        n: usize,
        a: &(impl DeviceRange<T> + ?Sized),
        lda: usize,
        ap: &mut (impl DeviceOutput<T> + ?Sized),
    )
}

/// Enqueues `batch_count` products `C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]` of matrices `stride_*` elements apart.
///
/// A zero input stride repeats one matrix for every product; output matrices may not overlap.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a buffer cannot hold its batch, a leading dimension or stride is invalid, output matrices overlap, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn gemm_strided_batched_async<T: Element>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar<'_, T>,
    a: &(impl DeviceRange<T> + ?Sized),
    lda: usize,
    stride_a: i64,
    b: &(impl DeviceRange<T> + ?Sized),
    ldb: usize,
    stride_b: i64,
    beta: Scalar<'_, T>,
    c: &mut (impl DeviceOutput<T> + ?Sized),
    ldc: usize,
    stride_c: i64,
    batch_count: usize,
) -> Result<()> {
    context.bind()?;
    let shape = GemmShape::new(m, n, k)
        .with_transpose_a(transpose_a)
        .with_transpose_b(transpose_b);
    ensure_strided_gemm(
        shape,
        (a.len(), lda, stride_a),
        (b.len(), ldb, stride_b),
        (c.len(), ldc, stride_c),
        batch_count,
    )?;
    context.prepare_operands(
        &[a.region(), b.region(), c.region()],
        &[alpha.region(), beta.region()],
    )?;
    context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
    // SAFETY: the buffers hold the checked batches in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::gemm_strided_batched(
            context.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            alpha.ptr(),
            a.as_device_ptr(),
            to_i64(lda, "lda")?,
            stride_a,
            b.as_device_ptr(),
            to_i64(ldb, "ldb")?,
            stride_b,
            beta.ptr(),
            c.as_mut_ptr(),
            to_i64(ldc, "ldc")?,
            stride_c,
            to_i64(batch_count, "batch count")?,
        ))
    }
}

waiting! {
    /// Computes a strided batch of matrix products and waits for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`gemm_strided_batched_async`] lists.
    gemm_strided_batched = gemm_strided_batched_async<T: Element>(
        transpose_a: Operation,
        transpose_b: Operation,
        m: usize,
        n: usize,
        k: usize,
        alpha: Scalar<'_, T>,
        a: &(impl DeviceRange<T> + ?Sized),
        lda: usize,
        stride_a: i64,
        b: &(impl DeviceRange<T> + ?Sized),
        ldb: usize,
        stride_b: i64,
        beta: Scalar<'_, T>,
        c: &mut (impl DeviceOutput<T> + ?Sized),
        ldc: usize,
        stride_c: i64,
        batch_count: usize,
    )
}

// Checks strided batches of `(length, leading dimension, stride)` against the shape.
pub(crate) fn ensure_strided_gemm(
    shape: GemmShape,
    a: (usize, usize, i64),
    b: (usize, usize, i64),
    c: (usize, usize, i64),
    batch_count: usize,
) -> Result<()> {
    let (a_rows, a_cols) = stored(shape.transpose_a(), shape.m(), shape.k());
    let (b_rows, b_cols) = stored(shape.transpose_b(), shape.k(), shape.n());
    ensure_matrix("a", a.0, a_rows, a_cols, a.1)?;
    ensure_matrix("b", b.0, b_rows, b_cols, b.1)?;
    ensure_matrix("c", c.0, shape.m(), shape.n(), c.1)?;
    let a_span = matrix_span(a_rows, a_cols, a.1)?;
    let b_span = matrix_span(b_rows, b_cols, b.1)?;
    let c_span = matrix_span(shape.m(), shape.n(), c.1)?;
    ensure_strided("a", a.0, a_span, a.2, batch_count, false)?;
    ensure_strided("b", b.0, b_span, b.2, batch_count, false)?;
    ensure_strided("c", c.0, c_span, c.2, batch_count, true)
}

// Copies the device addresses of the matrices into a device array that cuBLAS can read.
fn device_pointers(context: &Context, pointers: &[DevicePtr]) -> Result<DeviceMemory<DevicePtr>> {
    Ok(DeviceMemory::from_slice(context.cuda_context(), pointers)?)
}

fn input_pointers<T: Element>(matrices: &[DeviceInputView<'_, T>]) -> Vec<DevicePtr> {
    matrices
        .iter()
        // SAFETY: wrapping the address of a borrowed allocation does not dereference it.
        .map(|matrix| unsafe { DevicePtr::from_raw(matrix.as_ptr().cast_mut().cast()) })
        .collect()
}

fn output_pointers<T: Element>(matrices: &mut [DeviceOutputView<'_, T>]) -> Vec<DevicePtr> {
    matrices
        .iter_mut()
        // SAFETY: wrapping the address of a borrowed allocation does not dereference it.
        .map(|matrix| unsafe { DevicePtr::from_raw(matrix.as_mut_ptr().cast()) })
        .collect()
}

/// Enqueues `batch_count` products `C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]` whose matrices are listed in device arrays of pointers.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution) for the pointer arrays and for every matrix they address.
/// Each pointer must address a matrix of `T` in this context with the shape the dimensions and leading dimensions describe, and the `c` matrices must not overlap each other or the inputs.
///
/// # Errors
///
/// Returns an error if a pointer array holds fewer than `batch_count` entries, a leading dimension is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn gemm_batched_async<T: Element>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar<'_, T>,
    a: &(impl DeviceRange<DevicePtr> + ?Sized),
    lda: usize,
    b: &(impl DeviceRange<DevicePtr> + ?Sized),
    ldb: usize,
    beta: Scalar<'_, T>,
    c: &(impl DeviceRange<DevicePtr> + ?Sized),
    ldc: usize,
    batch_count: usize,
) -> Result<()> {
    context.bind()?;
    let (a_rows, a_cols) = stored(transpose_a, m, k);
    let (b_rows, b_cols) = stored(transpose_b, k, n);
    if lda < a_rows.max(1) || ldb < b_rows.max(1) || ldc < m.max(1) {
        return Err(Error::InvalidLeadingDimension);
    }

    matrix_span(a_rows, a_cols, lda)?;
    matrix_span(b_rows, b_cols, ldb)?;
    validate_min_len(batch_count, a.len(), "a")?;
    validate_min_len(batch_count, b.len(), "b")?;
    validate_min_len(batch_count, c.len(), "c")?;
    context.prepare_operands(
        &[a.region(), b.region(), c.region()],
        &[alpha.region(), beta.region()],
    )?;
    context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
    // SAFETY: the pointer arrays hold `batch_count` entries in this context, ordered after their producers; the caller upholds the matrix contract.
    unsafe {
        try_ffi!(T::gemm_batched(
            context.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            alpha.ptr(),
            a.as_device_ptr().cast(),
            to_i64(lda, "lda")?,
            b.as_device_ptr().cast(),
            to_i64(ldb, "ldb")?,
            beta.ptr(),
            c.as_device_ptr().cast(),
            to_i64(ldc, "ldc")?,
            to_i64(batch_count, "batch count")?,
        ))
    }
}

/// Computes the products `C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]` of corresponding matrices and waits for them.
///
/// It copies the matrix addresses to device pointer arrays, which live until the work completes.
///
/// # Errors
///
/// Returns an error if the batches have different lengths, a matrix is too small or a leading dimension is invalid, the handle's stream is capturing, copying the pointer arrays or waiting fails, or for the reasons [`gemm_batched_async`] lists.
pub fn gemm_batched<T: Element>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar<'_, T>,
    a: &[DeviceInputView<'_, T>],
    lda: usize,
    b: &[DeviceInputView<'_, T>],
    ldb: usize,
    beta: Scalar<'_, T>,
    c: &mut [DeviceOutputView<'_, T>],
    ldc: usize,
) -> Result<()> {
    let shape = GemmShape::new(m, n, k)
        .with_transpose_a(transpose_a)
        .with_transpose_b(transpose_b);
    validate_same_len(a.len(), b.len(), "b")?;
    validate_same_len(a.len(), c.len(), "c")?;
    if a.is_empty() {
        return Ok(());
    }

    for ((a, b), c) in a.iter().zip(b).zip(c.iter()) {
        ensure_gemm(shape, a.len(), lda, b.len(), ldb, c.len(), ldc)?;
    }

    let regions = a
        .iter()
        .map(DeviceInputView::region)
        .chain(b.iter().map(DeviceInputView::region))
        .chain(c.iter().map(DeviceRegionProvider::region))
        .collect::<Vec<_>>();
    context.bind()?;
    context.stream_binding().ensure_not_capturing()?;
    context.prepare_regions(&regions)?;
    let a_pointers = device_pointers(context, &input_pointers(a))?;
    let b_pointers = device_pointers(context, &input_pointers(b))?;
    let c_pointers = device_pointers(context, &output_pointers(c))?;
    // SAFETY: the pointers address the checked matrices, whose borrows and pointer arrays outlive the scope, which waits for the handle's stream; the `c` views are exclusive borrows, so they overlap neither each other nor the inputs.
    context.sync_scope(|| unsafe {
        gemm_batched_async(
            context,
            transpose_a,
            transpose_b,
            m,
            n,
            k,
            alpha,
            &a_pointers,
            lda,
            &b_pointers,
            ldb,
            beta,
            &c_pointers,
            ldc,
            c_pointers.len(),
        )
    })
}

/// Enqueues `batch_count` triangular solves `op(A[i]) * X[i] = alpha * B[i]` (left side) or `X[i] * op(A[i]) = alpha * B[i]` (right side) whose matrices are listed in device arrays of pointers.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution) for the pointer arrays and for every matrix they address.
/// Each pointer must address a matrix of `T` in this context with the shape the dimensions and leading dimensions describe, and the `b` matrices must not overlap each other or the `a` matrices.
///
/// # Errors
///
/// Returns an error if a pointer array holds fewer than `batch_count` entries, a leading dimension is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn trsm_batched_async<T: Element>(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    trans: Operation,
    diagonal: DiagonalType,
    m: usize,
    n: usize,
    alpha: Scalar<'_, T>,
    a: &(impl DeviceRange<DevicePtr> + ?Sized),
    lda: usize,
    b: &(impl DeviceRange<DevicePtr> + ?Sized),
    ldb: usize,
    batch_count: usize,
) -> Result<()> {
    context.bind()?;
    if lda < side_order(side, m, n).max(1) || ldb < m.max(1) {
        return Err(Error::InvalidLeadingDimension);
    }

    validate_min_len(batch_count, a.len(), "a")?;
    validate_min_len(batch_count, b.len(), "b")?;
    context.prepare_operands(&[a.region(), b.region()], &[alpha.region()])?;
    context.use_pointer_mode(&[alpha.pointer_mode()])?;
    // SAFETY: the pointer arrays hold `batch_count` entries in this context, ordered after their producers; the caller upholds the matrix contract.
    unsafe {
        try_ffi!(T::trsm_batched(
            context.as_raw(),
            side.into(),
            fill_mode.into(),
            trans.into(),
            diagonal.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            alpha.ptr(),
            a.as_device_ptr().cast(),
            to_i64(lda, "lda")?,
            b.as_device_ptr().cast(),
            to_i64(ldb, "ldb")?,
            to_i64(batch_count, "batch count")?,
        ))
    }
}

/// Solves corresponding triangular systems in place and waits for them.
///
/// It copies the matrix addresses to device pointer arrays, which live until the work completes.
///
/// # Errors
///
/// Returns an error if the batches have different lengths, a matrix is too small or a leading dimension is invalid, the handle's stream is capturing, copying the pointer arrays or waiting fails, or for the reasons [`trsm_batched_async`] lists.
pub fn trsm_batched<T: Element>(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    trans: Operation,
    diagonal: DiagonalType,
    m: usize,
    n: usize,
    alpha: Scalar<'_, T>,
    a: &[DeviceInputView<'_, T>],
    lda: usize,
    b: &mut [DeviceOutputView<'_, T>],
    ldb: usize,
) -> Result<()> {
    validate_same_len(a.len(), b.len(), "b")?;
    if a.is_empty() {
        return Ok(());
    }

    let order = side_order(side, m, n);
    for (a, b) in a.iter().zip(b.iter()) {
        ensure_matrix("a", a.len(), order, order, lda)?;
        ensure_matrix("b", b.len(), m, n, ldb)?;
    }

    let regions = a
        .iter()
        .map(DeviceInputView::region)
        .chain(b.iter().map(DeviceRegionProvider::region))
        .collect::<Vec<_>>();
    context.bind()?;
    context.stream_binding().ensure_not_capturing()?;
    context.prepare_regions(&regions)?;
    let a_pointers = device_pointers(context, &input_pointers(a))?;
    let b_pointers = device_pointers(context, &output_pointers(b))?;
    // SAFETY: the pointers address the checked matrices, whose borrows and pointer arrays outlive the scope, which waits for the handle's stream; the `b` views are exclusive borrows, so they overlap neither each other nor the `a` matrices.
    context.sync_scope(|| unsafe {
        trsm_batched_async(
            context,
            side,
            fill_mode,
            trans,
            diagonal,
            m,
            n,
            alpha,
            &a_pointers,
            lda,
            &b_pointers,
            ldb,
            b_pointers.len(),
        )
    })
}

/// One group of a grouped batched GEMM: `count` products of one shape, scalars, and leading dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmGroup<T> {
    /// Dimensions and transpositions of each product.
    pub shape: GemmShape,
    /// Scale of each product.
    pub alpha: T,
    /// Scale of each `C` matrix.
    pub beta: T,
    /// Leading dimension of each `A` matrix.
    pub lda: usize,
    /// Leading dimension of each `B` matrix.
    pub ldb: usize,
    /// Leading dimension of each `C` matrix.
    pub ldc: usize,
    /// Number of products in the group.
    pub count: usize,
}

/// Enqueues the products of every group, whose matrices are listed group by group in device arrays of pointers.
///
/// The group scalars are read from host memory.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution) for the pointer arrays and for every matrix they address.
/// Each pointer must address a matrix of `T` in this context with the shape of its group, and the `c` matrices must not overlap each other or the inputs.
///
/// # Errors
///
/// Returns an error if `groups` is empty, a pointer array holds fewer entries than the groups' total count, a leading dimension is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn gemm_grouped_batched_async<T: RealElement>(
    context: &Context,
    groups: &[GemmGroup<T>],
    a: &(impl DeviceRange<DevicePtr> + ?Sized),
    b: &(impl DeviceRange<DevicePtr> + ?Sized),
    c: &(impl DeviceRange<DevicePtr> + ?Sized),
) -> Result<()> {
    context.bind()?;
    let arrays = GroupArrays::create(groups)?;
    validate_min_len(arrays.total, a.len(), "a")?;
    validate_min_len(arrays.total, b.len(), "b")?;
    validate_min_len(arrays.total, c.len(), "c")?;
    context.prepare_operands(&[a.region(), b.region(), c.region()], &[])?;
    context.use_pointer_mode(&[PointerMode::Host])?;
    // SAFETY: the per-group host arrays hold one entry per group, and the pointer arrays hold the total count in this context, ordered after their producers; the caller upholds the matrix contract.
    unsafe {
        try_ffi!(T::gemm_grouped_batched(
            context.as_raw(),
            arrays.transpose_a.as_ptr(),
            arrays.transpose_b.as_ptr(),
            arrays.m.as_ptr(),
            arrays.n.as_ptr(),
            arrays.k.as_ptr(),
            arrays.alpha.as_ptr(),
            a.as_device_ptr().cast(),
            arrays.lda.as_ptr(),
            b.as_device_ptr().cast(),
            arrays.ldb.as_ptr(),
            arrays.beta.as_ptr(),
            c.as_device_ptr().cast(),
            arrays.ldc.as_ptr(),
            to_i64(groups.len(), "group count")?,
            arrays.count.as_ptr(),
        ))
    }
}

/// Computes the products of every group and waits for them.
///
/// `a`, `b`, and `c` list the matrices group by group.
/// It copies the matrix addresses to device pointer arrays, which live until the work completes.
///
/// # Errors
///
/// Returns an error if the matrix lists do not hold the groups' total count, a matrix is too small for its group, the handle's stream is capturing, copying the pointer arrays or waiting fails, or for the reasons [`gemm_grouped_batched_async`] lists.
pub fn gemm_grouped_batched<T: RealElement>(
    context: &Context,
    groups: &[GemmGroup<T>],
    a: &[DeviceInputView<'_, T>],
    b: &[DeviceInputView<'_, T>],
    c: &mut [DeviceOutputView<'_, T>],
) -> Result<()> {
    let total = GroupArrays::create(groups)?.total;
    validate_same_len(total, a.len(), "a")?;
    validate_same_len(total, b.len(), "b")?;
    validate_same_len(total, c.len(), "c")?;
    if total == 0 {
        return Ok(());
    }

    let mut index = 0;
    for group in groups {
        for _ in 0..group.count {
            ensure_gemm(
                group.shape,
                a[index].len(),
                group.lda,
                b[index].len(),
                group.ldb,
                c[index].len(),
                group.ldc,
            )?;
            index += 1;
        }
    }

    let regions = a
        .iter()
        .map(DeviceInputView::region)
        .chain(b.iter().map(DeviceInputView::region))
        .chain(c.iter().map(DeviceRegionProvider::region))
        .collect::<Vec<_>>();
    context.bind()?;
    context.stream_binding().ensure_not_capturing()?;
    context.prepare_regions(&regions)?;
    let a_pointers = device_pointers(context, &input_pointers(a))?;
    let b_pointers = device_pointers(context, &input_pointers(b))?;
    let c_pointers = device_pointers(context, &output_pointers(c))?;
    // SAFETY: the pointers address the checked matrices, whose borrows and pointer arrays outlive the scope, which waits for the handle's stream; the `c` views are exclusive borrows, so they overlap neither each other nor the inputs.
    context.sync_scope(|| unsafe {
        gemm_grouped_batched_async(context, groups, &a_pointers, &b_pointers, &c_pointers)
    })
}

// Per-group host arrays in the layout cuBLAS reads.
pub(crate) struct GroupArrays<T> {
    pub(crate) transpose_a: Vec<sys::cublasOperation_t>,
    pub(crate) transpose_b: Vec<sys::cublasOperation_t>,
    pub(crate) m: Vec<i64>,
    pub(crate) n: Vec<i64>,
    pub(crate) k: Vec<i64>,
    pub(crate) alpha: Vec<T>,
    pub(crate) beta: Vec<T>,
    pub(crate) lda: Vec<i64>,
    pub(crate) ldb: Vec<i64>,
    pub(crate) ldc: Vec<i64>,
    pub(crate) count: Vec<i64>,
    pub(crate) total: usize,
}

impl<T: Copy> GroupArrays<T> {
    pub(crate) fn create(groups: &[GemmGroup<T>]) -> Result<Self> {
        if groups.is_empty() {
            return Err(Error::LengthMismatch {
                name: "groups".into(),
                expected: 1,
                actual: 0,
            });
        }

        let mut arrays = Self {
            transpose_a: Vec::with_capacity(groups.len()),
            transpose_b: Vec::with_capacity(groups.len()),
            m: Vec::with_capacity(groups.len()),
            n: Vec::with_capacity(groups.len()),
            k: Vec::with_capacity(groups.len()),
            alpha: Vec::with_capacity(groups.len()),
            beta: Vec::with_capacity(groups.len()),
            lda: Vec::with_capacity(groups.len()),
            ldb: Vec::with_capacity(groups.len()),
            ldc: Vec::with_capacity(groups.len()),
            count: Vec::with_capacity(groups.len()),
            total: 0,
        };
        for group in groups {
            let shape = group.shape;
            let (a_rows, _) = stored(shape.transpose_a(), shape.m(), shape.k());
            let (b_rows, _) = stored(shape.transpose_b(), shape.k(), shape.n());
            if group.lda < a_rows.max(1)
                || group.ldb < b_rows.max(1)
                || group.ldc < shape.m().max(1)
            {
                return Err(Error::InvalidLeadingDimension);
            }

            arrays.transpose_a.push(shape.transpose_a().into());
            arrays.transpose_b.push(shape.transpose_b().into());
            arrays.m.push(to_i64(shape.m(), "m")?);
            arrays.n.push(to_i64(shape.n(), "n")?);
            arrays.k.push(to_i64(shape.k(), "k")?);
            arrays.alpha.push(group.alpha);
            arrays.beta.push(group.beta);
            arrays.lda.push(to_i64(group.lda, "lda")?);
            arrays.ldb.push(to_i64(group.ldb, "ldb")?);
            arrays.ldc.push(to_i64(group.ldc, "ldc")?);
            arrays.count.push(to_i64(group.count, "group size")?);
            arrays.total = arrays
                .total
                .checked_add(group.count)
                .ok_or_else(|| Error::out_of_range("grouped batch size"))?;
        }
        Ok(arrays)
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_gemm_matches_cpu_reference() -> Result<()> {
        let test = setup_context()?;
        let context = test.context();

        let a = DeviceMemory::from_slice(context.cuda_context(), &[1.0_f32, 2.0, 3.0, 4.0])?;
        let b = DeviceMemory::from_slice(context.cuda_context(), &[5.0_f32, 6.0, 7.0, 8.0])?;
        let mut c = DeviceMemory::<f32>::create_uninit(context.cuda_context(), 4)?;

        gemm(
            context,
            Operation::NonTranspose,
            Operation::NonTranspose,
            2,
            2,
            2,
            Scalar::host(&1.0),
            &a,
            2,
            &b,
            2,
            Scalar::host(&0.0),
            &mut c,
            2,
        )?;

        // SAFETY: beta is zero and the completed packed 2x2 GEMM wrote every element.
        let c = unsafe { c.assume_init() };
        let result = c.copy_to_host_vec()?;
        assert_eq!(result, vec![23.0, 34.0, 31.0, 46.0]);

        Ok(())
    }
}
