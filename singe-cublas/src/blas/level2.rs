#![expect(
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "BLAS and factorization operand names follow the standard mathematical API"
)]
use std::ptr;

use singe_cuda::{memory::DeviceMemory, types::Complex64};

use crate::{
    context::Context,
    error::{Error, Result},
    sys, try_ffi,
    types::{DiagonalType, FillMode, Operation},
    utility::{required_matrix_len, required_vector_len, to_i64},
};

/// Supports the 64-bit integer interface.
///
/// Performs the banded matrix-vector multiplication:
///
/// $$ \mathbf{y} = \alpha\text{ op}(A)\mathbf{x} + \beta\mathbf{y} $$
///
/// Here $A$ is a banded matrix with $kl$ subdiagonals and $ku$ superdiagonals, $\mathbf{x}$ and $\mathbf{y}$ are vectors, and $\alpha$ and $\beta$ are scalars.
/// For matrix $A$:
///
/// $$ \text{ op}(A) = \begin{cases} A & \text{ if } \texttt{trans = `Operation::NonTranspose`} \\\\ A^{T} & \text{ if } \texttt{trans = `Operation::Transpose`} \\\\ A^{H} & \text{ if } \texttt{trans = `Operation::ConjugateTranspose`} \\\\ \end{cases} $$
///
/// The banded matrix $A$ is stored column by column, with the main diagonal stored in row $ku + 1$ (starting in first position), the first superdiagonal stored in row $ku$ (starting in second position), the first subdiagonal stored in row $ku + 2$ (starting in first position), etc.
/// In general, the element $A\left( {i,j} \right)$ is stored in the memory location `A(ku+1+i-j,j)` for $j = 1,\ldots,n$ and $i \in \left\lbrack {\max\left( {1,j - ku} \right),\min\left( {m,j + kl} \right)} \right\rbrack$.
/// Elements in the array $A$ that do not conceptually correspond to the banded matrix (the top left $ku \times ku$ and bottom right $kl \times kl$ triangles) are not referenced.
///
/// See NETLIB documentation:
///
/// [sgbmv()](https://www.netlib.org/blas/sgbmv.f), [dgbmv()](https://www.netlib.org/blas/dgbmv.f), [cgbmv()](https://www.netlib.org/blas/cgbmv.f), [zgbmv()](https://www.netlib.org/blas/zgbmv.f).
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode,
/// `lda` is smaller than `kl + ku + 1`, either vector increment is zero or too
/// large for cuBLAS, `a`, `x`, or `y` is too small for the requested banded
/// matrix-vector operation, a dimension is too large for cuBLAS, or cuBLAS
/// rejects the operation.
#[allow(
    clippy::doc_markdown,
    reason = "upstream LaTeX formulas contain identifiers that are not Markdown code spans"
)]
pub fn sgbmv(
    context: &Context,
    trans: Operation,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: &f32,
    a: &DeviceMemory<f32>,
    lda: usize,
    x: &DeviceMemory<f32>,
    incx: usize,
    beta: &f32,
    y: &mut DeviceMemory<f32>,
    incy: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;

    if lda < kl + ku + 1 {
        return Err(Error::InvalidLeadingDimension);
    }
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }
    if a.len() < required_matrix_len(lda, n)? {
        return Err(Error::InvalidMatrixShape);
    }

    let (x_dim, y_dim) = match trans {
        Operation::NonTranspose => (n, m),
        _ => (m, n),
    };

    if !vector_fits(x.len(), x_dim, incx)? || !vector_fits(y.len(), y_dim, incy)? {
        return Err(Error::InvalidVectorShape);
    }

    let m = to_i64(m, "m")?;
    let n = to_i64(n, "n")?;
    let kl = to_i64(kl, "kl")?;
    let ku = to_i64(ku, "ku")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasSgbmv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasSgbmv_v2_64(
            context.as_raw(),
            trans.into(),
            m,
            n,
            kl,
            ku,
            alpha,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            beta,
            y.as_mut_ptr(),
            incy,
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Performs the banded matrix-vector multiplication:
///
/// $$ \mathbf{y} = \alpha\text{ op}(A)\mathbf{x} + \beta\mathbf{y} $$
///
/// Here $A$ is a banded matrix with $kl$ subdiagonals and $ku$ superdiagonals, $\mathbf{x}$ and $\mathbf{y}$ are vectors, and $\alpha$ and $\beta$ are scalars.
/// For matrix $A$:
///
/// $$ \text{ op}(A) = \begin{cases} A & \text{ if } \texttt{trans = `Operation::NonTranspose`} \\\\ A^{T} & \text{ if } \texttt{trans = `Operation::Transpose`} \\\\ A^{H} & \text{ if } \texttt{trans = `Operation::ConjugateTranspose`} \\\\ \end{cases} $$
///
/// The banded matrix $A$ is stored column by column, with the main diagonal stored in row $ku + 1$ (starting in first position), the first superdiagonal stored in row $ku$ (starting in second position), the first subdiagonal stored in row $ku + 2$ (starting in first position), etc.
/// In general, the element $A\left( {i,j} \right)$ is stored in the memory location `A(ku+1+i-j,j)` for $j = 1,\ldots,n$ and $i \in \left\lbrack {\max\left( {1,j - ku} \right),\min\left( {m,j + kl} \right)} \right\rbrack$.
/// Elements in the array $A$ that do not conceptually correspond to the banded matrix (the top left $ku \times ku$ and bottom right $kl \times kl$ triangles) are not referenced.
///
/// See NETLIB documentation:
///
/// [sgbmv()](https://www.netlib.org/blas/sgbmv.f), [dgbmv()](https://www.netlib.org/blas/dgbmv.f), [cgbmv()](https://www.netlib.org/blas/cgbmv.f), [zgbmv()](https://www.netlib.org/blas/zgbmv.f).
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode,
/// `lda` is smaller than `kl + ku + 1`, either vector increment is zero or too
/// large for cuBLAS, `a`, `x`, or `y` is too small for the requested banded
/// matrix-vector operation, a dimension is too large for cuBLAS, or cuBLAS
/// rejects the operation.
#[allow(
    clippy::doc_markdown,
    reason = "upstream LaTeX formulas contain identifiers that are not Markdown code spans"
)]
pub fn dgbmv(
    context: &Context,
    trans: Operation,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: &f64,
    a: &DeviceMemory<f64>,
    lda: usize,
    x: &DeviceMemory<f64>,
    incx: usize,
    beta: &f64,
    y: &mut DeviceMemory<f64>,
    incy: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;

    if lda < kl + ku + 1 {
        return Err(Error::InvalidLeadingDimension);
    }
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }
    if a.len() < required_matrix_len(lda, n)? {
        return Err(Error::InvalidMatrixShape);
    }

    let (x_dim, y_dim) = match trans {
        Operation::NonTranspose => (n, m),
        _ => (m, n),
    };

    if !vector_fits(x.len(), x_dim, incx)? || !vector_fits(y.len(), y_dim, incy)? {
        return Err(Error::InvalidVectorShape);
    }

    let m = to_i64(m, "m")?;
    let n = to_i64(n, "n")?;
    let kl = to_i64(kl, "kl")?;
    let ku = to_i64(ku, "ku")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasDgbmv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDgbmv_v2_64(
            context.as_raw(),
            trans.into(),
            m,
            n,
            kl,
            ku,
            alpha,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            beta,
            y.as_mut_ptr(),
            incy,
        ))?;
    }

    Ok(())
}

/// Computes `y = alpha * op(A) * x + beta * y` in double precision.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dgemv(
    context: &Context,
    trans: Operation,
    m: usize,
    n: usize,
    alpha: &f64,
    a: &DeviceMemory<f64>,
    lda: usize,
    x: &DeviceMemory<f64>,
    incx: usize,
    beta: &f64,
    y: &mut DeviceMemory<f64>,
    incy: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_general_matrix_vector(trans, m, n, a.len(), lda, x.len(), incx, y.len(), incy)?;

    let m = to_i64(m, "m")?;
    let n = to_i64(n, "n")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasDgemv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDgemv_v2_64(
            context.as_raw(),
            trans.into(),
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            beta,
            y.as_mut_ptr(),
            incy,
        ))?;
    }

    Ok(())
}

/// Updates a double-precision general matrix with `A = alpha * x * y^T + A`.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dger(
    context: &Context,
    m: usize,
    n: usize,
    alpha: &f64,
    x: &DeviceMemory<f64>,
    incx: usize,
    y: &DeviceMemory<f64>,
    incy: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_rank1_update(m, n, x.len(), incx, y.len(), incy, a.len(), lda)?;

    let m = to_i64(m, "m")?;
    let n = to_i64(n, "n")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasDger_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDger_v2_64(
            context.as_raw(),
            m,
            n,
            alpha,
            x.as_ptr(),
            incx,
            y.as_ptr(),
            incy,
            a.as_mut_ptr(),
            lda,
        ))?;
    }

    Ok(())
}

/// Multiplies a double-precision symmetric matrix by a vector and updates `y`.
///
/// Computes `y = alpha * A * x + beta * y` using the triangle selected by `fill_mode`.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dsymv(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &f64,
    a: &DeviceMemory<f64>,
    lda: usize,
    x: &DeviceMemory<f64>,
    incx: usize,
    beta: &f64,
    y: &mut DeviceMemory<f64>,
    incy: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_square_matrix_vector(n, a.len(), lda, x.len(), incx, y.len(), incy)?;

    let n = to_i64(n, "n")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasDsymv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDsymv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            alpha,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            beta,
            y.as_mut_ptr(),
            incy,
        ))?;
    }

    Ok(())
}

/// Multiplies a complex double-precision Hermitian matrix by a vector and updates `y`.
///
/// Computes `y = alpha * A * x + beta * y` using the triangle selected by `fill_mode`.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn zhemv(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &Complex64,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    x: &DeviceMemory<Complex64>,
    incx: usize,
    beta: &Complex64,
    y: &mut DeviceMemory<Complex64>,
    incy: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_square_matrix_vector(n, a.len(), lda, x.len(), incx, y.len(), incy)?;

    let n = to_i64(n, "n")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasZhemv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasZhemv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            ptr::from_ref(alpha).cast(),
            a.as_ptr().cast(),
            lda,
            x.as_ptr().cast(),
            incx,
            ptr::from_ref(beta).cast(),
            y.as_mut_ptr().cast(),
            incy,
        ))?;
    }

    Ok(())
}

/// Multiplies a double-precision symmetric band matrix by a vector and updates `y`.
///
/// Computes `y = alpha * A * x + beta * y` using the triangle selected by `fill_mode`.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dsbmv(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    k: usize,
    alpha: &f64,
    a: &DeviceMemory<f64>,
    lda: usize,
    x: &DeviceMemory<f64>,
    incx: usize,
    beta: &f64,
    y: &mut DeviceMemory<f64>,
    incy: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_symmetric_banded_matrix_vector(n, k, a.len(), lda, x.len(), incx, y.len(), incy)?;

    let n = to_i64(n, "n")?;
    let k = to_i64(k, "k")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasDsbmv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDsbmv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            beta,
            y.as_mut_ptr(),
            incy,
        ))?;
    }

    Ok(())
}

/// Multiplies a complex double-precision Hermitian band matrix by a vector and updates `y`.
///
/// Computes `y = alpha * A * x + beta * y` using the triangle selected by `fill_mode`.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn zhbmv(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    k: usize,
    alpha: &Complex64,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    x: &DeviceMemory<Complex64>,
    incx: usize,
    beta: &Complex64,
    y: &mut DeviceMemory<Complex64>,
    incy: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_symmetric_banded_matrix_vector(n, k, a.len(), lda, x.len(), incx, y.len(), incy)?;

    let n = to_i64(n, "n")?;
    let k = to_i64(k, "k")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasZhbmv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasZhbmv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            k,
            ptr::from_ref(alpha).cast(),
            a.as_ptr().cast(),
            lda,
            x.as_ptr().cast(),
            incx,
            ptr::from_ref(beta).cast(),
            y.as_mut_ptr().cast(),
            incy,
        ))?;
    }

    Ok(())
}

/// Multiplies a packed double-precision symmetric matrix by a vector and updates `y`.
///
/// Computes `y = alpha * A * x + beta * y` using the triangle selected by `fill_mode`.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dspmv(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &f64,
    ap: &DeviceMemory<f64>,
    x: &DeviceMemory<f64>,
    incx: usize,
    beta: &f64,
    y: &mut DeviceMemory<f64>,
    incy: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_packed_matrix_vector(n, ap.len(), x.len(), incx, y.len(), incy)?;

    let n = to_i64(n, "n")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasDspmv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDspmv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            alpha,
            ap.as_ptr(),
            x.as_ptr(),
            incx,
            beta,
            y.as_mut_ptr(),
            incy,
        ))?;
    }

    Ok(())
}

/// Multiplies a packed complex double-precision Hermitian matrix by a vector and updates `y`.
///
/// Computes `y = alpha * A * x + beta * y` using the triangle selected by `fill_mode`.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn zhpmv(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &Complex64,
    ap: &DeviceMemory<Complex64>,
    x: &DeviceMemory<Complex64>,
    incx: usize,
    beta: &Complex64,
    y: &mut DeviceMemory<Complex64>,
    incy: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_packed_matrix_vector(n, ap.len(), x.len(), incx, y.len(), incy)?;

    let n = to_i64(n, "n")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasZhpmv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasZhpmv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            ptr::from_ref(alpha).cast(),
            ap.as_ptr().cast(),
            x.as_ptr().cast(),
            incx,
            ptr::from_ref(beta).cast(),
            y.as_mut_ptr().cast(),
            incy,
        ))?;
    }

    Ok(())
}

/// Applies the symmetric rank-1 update `A = alpha * x * x^T + A` in double precision.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dsyr(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &f64,
    x: &DeviceMemory<f64>,
    incx: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_square_rank1_update(n, x.len(), incx, a.len(), lda)?;

    let n = to_i64(n, "n")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;

    // SAFETY: `cublasDsyr_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDsyr_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            alpha,
            x.as_ptr(),
            incx,
            a.as_mut_ptr(),
            lda,
        ))?;
    }

    Ok(())
}

/// Applies the Hermitian rank-1 update `A = alpha * x * x^H + A` in complex double precision.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn zher(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &f64,
    x: &DeviceMemory<Complex64>,
    incx: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_square_rank1_update(n, x.len(), incx, a.len(), lda)?;

    let n = to_i64(n, "n")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;

    // SAFETY: `cublasZher_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasZher_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            alpha,
            x.as_ptr().cast(),
            incx,
            a.as_mut_ptr().cast(),
            lda,
        ))?;
    }

    Ok(())
}

/// Applies a double-precision symmetric rank-1 update to a packed matrix.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dspr(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &f64,
    x: &DeviceMemory<f64>,
    incx: usize,
    ap: &mut DeviceMemory<f64>,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_packed_rank1_update(n, x.len(), incx, ap.len())?;

    let n = to_i64(n, "n")?;
    let incx = to_i64(incx, "incx")?;

    // SAFETY: `cublasDspr_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDspr_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            alpha,
            x.as_ptr(),
            incx,
            ap.as_mut_ptr(),
        ))?;
    }

    Ok(())
}

/// Applies a complex double-precision Hermitian rank-1 update to a packed matrix.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn zhpr(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &f64,
    x: &DeviceMemory<Complex64>,
    incx: usize,
    ap: &mut DeviceMemory<Complex64>,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_packed_rank1_update(n, x.len(), incx, ap.len())?;

    let n = to_i64(n, "n")?;
    let incx = to_i64(incx, "incx")?;

    // SAFETY: `cublasZhpr_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasZhpr_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            alpha,
            x.as_ptr().cast(),
            incx,
            ap.as_mut_ptr().cast(),
        ))?;
    }

    Ok(())
}

/// Applies `A = alpha * (x * y^T + y * x^T) + A` in double precision.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dsyr2(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &f64,
    x: &DeviceMemory<f64>,
    incx: usize,
    y: &DeviceMemory<f64>,
    incy: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_square_rank2_update(n, x.len(), incx, y.len(), incy, a.len(), lda)?;

    let n = to_i64(n, "n")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasDsyr2_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDsyr2_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            alpha,
            x.as_ptr(),
            incx,
            y.as_ptr(),
            incy,
            a.as_mut_ptr(),
            lda,
        ))?;
    }

    Ok(())
}

/// Applies `A = alpha * x * y^H + conj(alpha) * y * x^H + A` in complex double precision.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn zher2(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &Complex64,
    x: &DeviceMemory<Complex64>,
    incx: usize,
    y: &DeviceMemory<Complex64>,
    incy: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_square_rank2_update(n, x.len(), incx, y.len(), incy, a.len(), lda)?;

    let n = to_i64(n, "n")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasZher2_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasZher2_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            ptr::from_ref(alpha).cast(),
            x.as_ptr().cast(),
            incx,
            y.as_ptr().cast(),
            incy,
            a.as_mut_ptr().cast(),
            lda,
        ))?;
    }

    Ok(())
}

/// Applies a double-precision symmetric rank-2 update to a packed matrix.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dspr2(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &f64,
    x: &DeviceMemory<f64>,
    incx: usize,
    y: &DeviceMemory<f64>,
    incy: usize,
    ap: &mut DeviceMemory<f64>,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_packed_rank2_update(n, x.len(), incx, y.len(), incy, ap.len())?;

    let n = to_i64(n, "n")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasDspr2_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDspr2_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            alpha,
            x.as_ptr(),
            incx,
            y.as_ptr(),
            incy,
            ap.as_mut_ptr(),
        ))?;
    }

    Ok(())
}

/// Applies a complex double-precision Hermitian rank-2 update to a packed matrix.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// Returns an error if the handle is not in host pointer mode.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn zhpr2(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    alpha: &Complex64,
    x: &DeviceMemory<Complex64>,
    incx: usize,
    y: &DeviceMemory<Complex64>,
    incy: usize,
    ap: &mut DeviceMemory<Complex64>,
) -> Result<()> {
    context.bind()?;
    context.require_host_pointer_mode()?;
    validate_packed_rank2_update(n, x.len(), incx, y.len(), incy, ap.len())?;

    let n = to_i64(n, "n")?;
    let incx = to_i64(incx, "incx")?;
    let incy = to_i64(incy, "incy")?;

    // SAFETY: `cublasZhpr2_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasZhpr2_v2_64(
            context.as_raw(),
            fill_mode.into(),
            n,
            ptr::from_ref(alpha).cast(),
            x.as_ptr().cast(),
            incx,
            y.as_ptr().cast(),
            incy,
            ap.as_mut_ptr().cast(),
        ))?;
    }

    Ok(())
}

/// Replaces `x` with `op(A) * x` for a double-precision triangular matrix.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dtrmv(
    context: &Context,
    fill_mode: FillMode,
    trans: Operation,
    diag: DiagonalType,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    x: &mut DeviceMemory<f64>,
    incx: usize,
) -> Result<()> {
    context.bind()?;
    validate_triangular_matrix_vector(n, a.len(), lda, x.len(), incx)?;

    let n = to_i64(n, "n")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;

    // SAFETY: `cublasDtrmv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDtrmv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            trans.into(),
            diag.into(),
            n,
            a.as_ptr(),
            lda,
            x.as_mut_ptr(),
            incx,
        ))?;
    }

    Ok(())
}

/// Replaces `x` with `op(A) * x` for a double-precision triangular band matrix.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dtbmv(
    context: &Context,
    fill_mode: FillMode,
    trans: Operation,
    diag: DiagonalType,
    n: usize,
    k: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    x: &mut DeviceMemory<f64>,
    incx: usize,
) -> Result<()> {
    context.bind()?;
    validate_triangular_banded_matrix_vector(n, k, a.len(), lda, x.len(), incx)?;

    let n = to_i64(n, "n")?;
    let k = to_i64(k, "k")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;

    // SAFETY: `cublasDtbmv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDtbmv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            trans.into(),
            diag.into(),
            n,
            k,
            a.as_ptr(),
            lda,
            x.as_mut_ptr(),
            incx,
        ))?;
    }

    Ok(())
}

/// Replaces `x` with `op(A) * x` for a packed double-precision triangular matrix.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dtpmv(
    context: &Context,
    fill_mode: FillMode,
    trans: Operation,
    diag: DiagonalType,
    n: usize,
    ap: &DeviceMemory<f64>,
    x: &mut DeviceMemory<f64>,
    incx: usize,
) -> Result<()> {
    context.bind()?;
    validate_packed_triangular_matrix_vector(n, ap.len(), x.len(), incx)?;

    let n = to_i64(n, "n")?;
    let incx = to_i64(incx, "incx")?;

    // SAFETY: `cublasDtpmv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDtpmv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            trans.into(),
            diag.into(),
            n,
            ap.as_ptr(),
            x.as_mut_ptr(),
            incx,
        ))?;
    }

    Ok(())
}

/// Solves `op(A) * x = b` in place for a double-precision triangular matrix.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dtrsv(
    context: &Context,
    fill_mode: FillMode,
    trans: Operation,
    diag: DiagonalType,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    x: &mut DeviceMemory<f64>,
    incx: usize,
) -> Result<()> {
    context.bind()?;
    validate_triangular_matrix_vector(n, a.len(), lda, x.len(), incx)?;

    let n = to_i64(n, "n")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;

    // SAFETY: `cublasDtrsv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDtrsv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            trans.into(),
            diag.into(),
            n,
            a.as_ptr(),
            lda,
            x.as_mut_ptr(),
            incx,
        ))?;
    }

    Ok(())
}

/// Solves `op(A) * x = b` in place for a packed double-precision triangular matrix.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dtpsv(
    context: &Context,
    fill_mode: FillMode,
    trans: Operation,
    diag: DiagonalType,
    n: usize,
    ap: &DeviceMemory<f64>,
    x: &mut DeviceMemory<f64>,
    incx: usize,
) -> Result<()> {
    context.bind()?;
    validate_packed_triangular_matrix_vector(n, ap.len(), x.len(), incx)?;

    let n = to_i64(n, "n")?;
    let incx = to_i64(incx, "incx")?;

    // SAFETY: `cublasDtpsv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDtpsv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            trans.into(),
            diag.into(),
            n,
            ap.as_ptr(),
            x.as_mut_ptr(),
            incx,
        ))?;
    }

    Ok(())
}

/// Solves `op(A) * x = b` in place for a double-precision triangular band matrix.
///
/// Only the triangle selected by `fill_mode` is referenced.
///
/// Matrices use column-major storage; leading dimensions and vector increments are measured in elements.
///
/// # Errors
///
/// Returns an error if dimensions, leading dimensions, vector increments, or storage lengths are invalid, or a computed size exceeds the native integer range.
/// CUDA context binding and cuBLAS operation errors are propagated.
pub fn dtbsv(
    context: &Context,
    fill_mode: FillMode,
    trans: Operation,
    diag: DiagonalType,
    n: usize,
    k: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    x: &mut DeviceMemory<f64>,
    incx: usize,
) -> Result<()> {
    context.bind()?;
    validate_triangular_banded_matrix_vector(n, k, a.len(), lda, x.len(), incx)?;

    let n = to_i64(n, "n")?;
    let k = to_i64(k, "k")?;
    let lda = to_i64(lda, "lda")?;
    let incx = to_i64(incx, "incx")?;

    // SAFETY: `cublasDtbsv_v2_64` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cublasDtbsv_v2_64(
            context.as_raw(),
            fill_mode.into(),
            trans.into(),
            diag.into(),
            n,
            k,
            a.as_ptr(),
            lda,
            x.as_mut_ptr(),
            incx,
        ))?;
    }

    Ok(())
}

fn validate_general_matrix_vector(
    trans: Operation,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    incx: usize,
    y_len: usize,
    incy: usize,
) -> Result<()> {
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }
    if lda < m.max(1) {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, n)? {
        return Err(Error::InvalidMatrixShape);
    }

    let (x_dim, y_dim) = match trans {
        Operation::NonTranspose => (n, m),
        _ => (m, n),
    };
    if !vector_fits(x_len, x_dim, incx)? || !vector_fits(y_len, y_dim, incy)? {
        return Err(Error::InvalidVectorShape);
    }

    Ok(())
}

fn validate_rank1_update(
    m: usize,
    n: usize,
    x_len: usize,
    incx: usize,
    y_len: usize,
    incy: usize,
    a_len: usize,
    lda: usize,
) -> Result<()> {
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }
    if lda < m.max(1) {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, n)? {
        return Err(Error::InvalidMatrixShape);
    }
    if !vector_fits(x_len, m, incx)? || !vector_fits(y_len, n, incy)? {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn validate_square_matrix_vector(
    n: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    incx: usize,
    y_len: usize,
    incy: usize,
) -> Result<()> {
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }
    if lda < n.max(1) {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, n)? {
        return Err(Error::InvalidMatrixShape);
    }
    if !vector_fits(x_len, n, incx)? || !vector_fits(y_len, n, incy)? {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn validate_symmetric_banded_matrix_vector(
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    incx: usize,
    y_len: usize,
    incy: usize,
) -> Result<()> {
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }
    if lda < k + 1 {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, n)? {
        return Err(Error::InvalidMatrixShape);
    }
    if !vector_fits(x_len, n, incx)? || !vector_fits(y_len, n, incy)? {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn validate_packed_matrix_vector(
    n: usize,
    ap_len: usize,
    x_len: usize,
    incx: usize,
    y_len: usize,
    incy: usize,
) -> Result<()> {
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }
    if ap_len < required_packed_matrix_len(n)? {
        return Err(Error::InvalidMatrixShape);
    }
    if !vector_fits(x_len, n, incx)? || !vector_fits(y_len, n, incy)? {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn validate_square_rank1_update(
    n: usize,
    x_len: usize,
    incx: usize,
    a_len: usize,
    lda: usize,
) -> Result<()> {
    if incx == 0 {
        return Err(Error::InvalidIncrement);
    }
    if lda < n.max(1) {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, n)? {
        return Err(Error::InvalidMatrixShape);
    }
    if !vector_fits(x_len, n, incx)? {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn validate_packed_rank1_update(n: usize, x_len: usize, incx: usize, ap_len: usize) -> Result<()> {
    if incx == 0 {
        return Err(Error::InvalidIncrement);
    }
    if ap_len < required_packed_matrix_len(n)? {
        return Err(Error::InvalidMatrixShape);
    }
    if !vector_fits(x_len, n, incx)? {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn validate_square_rank2_update(
    n: usize,
    x_len: usize,
    incx: usize,
    y_len: usize,
    incy: usize,
    a_len: usize,
    lda: usize,
) -> Result<()> {
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }
    if lda < n.max(1) {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, n)? {
        return Err(Error::InvalidMatrixShape);
    }
    if !vector_fits(x_len, n, incx)? || !vector_fits(y_len, n, incy)? {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn validate_packed_rank2_update(
    n: usize,
    x_len: usize,
    incx: usize,
    y_len: usize,
    incy: usize,
    ap_len: usize,
) -> Result<()> {
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }
    if ap_len < required_packed_matrix_len(n)? {
        return Err(Error::InvalidMatrixShape);
    }
    if !vector_fits(x_len, n, incx)? || !vector_fits(y_len, n, incy)? {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn validate_triangular_matrix_vector(
    n: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    incx: usize,
) -> Result<()> {
    if incx == 0 {
        return Err(Error::InvalidIncrement);
    }
    if lda < n.max(1) {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, n)? {
        return Err(Error::InvalidMatrixShape);
    }
    if !vector_fits(x_len, n, incx)? {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn validate_triangular_banded_matrix_vector(
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    incx: usize,
) -> Result<()> {
    if incx == 0 {
        return Err(Error::InvalidIncrement);
    }
    if lda < k + 1 {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, n)? {
        return Err(Error::InvalidMatrixShape);
    }
    if !vector_fits(x_len, n, incx)? {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn validate_packed_triangular_matrix_vector(
    n: usize,
    ap_len: usize,
    x_len: usize,
    incx: usize,
) -> Result<()> {
    if incx == 0 {
        return Err(Error::InvalidIncrement);
    }
    if ap_len < required_packed_matrix_len(n)? {
        return Err(Error::InvalidMatrixShape);
    }
    if !vector_fits(x_len, n, incx)? {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

fn required_packed_matrix_len(n: usize) -> Result<usize> {
    n.checked_mul(n + 1)
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| Error::out_of_range("packed matrix length"))
}

fn vector_fits(length: usize, n: usize, inc: usize) -> Result<bool> {
    if n == 0 {
        return Ok(true);
    }

    required_vector_len(n, inc).map(|required| length >= required)
}
