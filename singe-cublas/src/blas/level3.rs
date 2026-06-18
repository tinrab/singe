use std::ptr;

use singe_cuda::{
    data_type::DataType,
    memory::DeviceMemory,
    types::{Complex32, Complex64, DevicePtr},
};

use crate::{
    context::Context,
    error::{Error, Result},
    scalar::Scalar,
    sys, try_ffi,
    types::{ComputeType, DiagonalType, FillMode, GemmAlgorithm, Operation, SideMode},
    utility::{required_matrix_len, required_vector_len, to_i32, to_usize},
};

/// Supports the 64-bit integer interface.
///
/// Performs the matrix-matrix multiplication:
///
/// $$ C = \alpha\text{op}(A)\text{op}(B) + \beta C $$
///
/// Here $\alpha$ and $\beta$ are scalars, and $A$, $B$ and $C$ are matrices stored in column-major format with dimensions $\text{op}(A)$ $m \times k$, $\text{op}(B)$ $k \times n$ and $C$ $m \times n$, respectively.
/// `a`, `b`, and `c` are device allocations; leading dimensions are element
/// strides in cuBLAS column-major layout. `alpha` and `beta` may be host
/// values or device scalar pointers, but both must use the same pointer mode.
/// For matrix $A$:
///
/// $$ \text{op}(A) = \left\\{ \begin{matrix} A & {\text{if } \texttt{transpose_a = Operation::NonTranspose}} \\\\ A^{T} & {\text{if } \texttt{transpose_a = Operation::Transpose}} \\\\ A^{H} & {\text{if } \texttt{transpose_a = Operation::ConjugateTranspose}} \\\\ \end{matrix} \right. $$
///
/// $\text{op}(B)$ is defined similarly for matrix $B$.
///
/// See NETLIB documentation:
///
/// [sgemm()](https://www.netlib.org/blas/sgemm.f), [dgemm()](https://www.netlib.org/blas/dgemm.f), [cgemm()](https://www.netlib.org/blas/cgemm.f), [zgemm()](https://www.netlib.org/blas/zgemm.f).
///
/// # Errors
///
/// Returns an error if the context is not usable, scalar pointer modes do not
/// match, dimensions or leading dimensions are invalid, device allocations are
/// too small for the requested matrix shapes, or cuBLAS rejects or fails the
/// operation.
pub fn sgemm<'alpha, 'beta>(
    ctx: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: impl Into<Scalar<'alpha, f32>>,
    a: &DeviceMemory<f32>,
    lda: usize,
    b: &DeviceMemory<f32>,
    ldb: usize,
    beta: impl Into<Scalar<'beta, f32>>,
    c: &mut DeviceMemory<f32>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    let alpha = alpha.into();
    let beta = beta.into();
    let pointer_mode = scalar_pointer_mode(&alpha, &beta)?;
    validate_gemm_parameters(transpose_a, transpose_b, m, n, k, lda, ldb, ldc)?;
    validate_gemm_shapes(
        transpose_a,
        transpose_b,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    ctx.with_pointer_mode(pointer_mode, || unsafe {
        try_ffi!(sys::cublasSgemm_v2(
            ctx.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            m,
            n,
            k,
            alpha.as_ptr(),
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta.as_ptr(),
            c.as_mut_ptr(),
            ldc,
        ))?;
        Ok(())
    })?;

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Performs the matrix-matrix multiplication:
///
/// $$ C = \alpha\text{op}(A)\text{op}(B) + \beta C $$
///
/// Here $\alpha$ and $\beta$ are scalars, and $A$, $B$ and $C$ are matrices stored in column-major format with dimensions $\text{op}(A)$ $m \times k$, $\text{op}(B)$ $k \times n$ and $C$ $m \times n$, respectively.
/// `a`, `b`, and `c` are device allocations; leading dimensions are element
/// strides in cuBLAS column-major layout. `alpha` and `beta` may be host
/// values or device scalar pointers, but both must use the same pointer mode.
/// For matrix $A$:
///
/// $$
/// \text{op}(A) = \left\\{ \begin{matrix} A & {\text{if } \texttt{transpose_a = Operation::NonTranspose}} \\\\ A^{T} & {\text{if } \texttt{transpose_a = Operation::Transpose}} \\\\ A^{H} & {\text{if } \texttt{transpose_a = Operation::ConjugateTranspose}} \\\\ \end{matrix} \right.
/// $$
///
/// $\text{op}(B)$ is defined similarly for matrix $B$.
///
/// See NETLIB documentation:
///
/// [sgemm()](https://www.netlib.org/blas/sgemm.f), [dgemm()](https://www.netlib.org/blas/dgemm.f), [cgemm()](https://www.netlib.org/blas/cgemm.f), [zgemm()](https://www.netlib.org/blas/zgemm.f).
///
/// # Errors
///
/// Returns an error if the context is not usable, scalar pointer modes do not
/// match, dimensions or leading dimensions are invalid, device allocations are
/// too small for the requested matrix shapes, or cuBLAS rejects or fails the
/// operation.
pub fn dgemm<'alpha, 'beta>(
    ctx: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: impl Into<Scalar<'alpha, f64>>,
    a: &DeviceMemory<f64>,
    lda: usize,
    b: &DeviceMemory<f64>,
    ldb: usize,
    beta: impl Into<Scalar<'beta, f64>>,
    c: &mut DeviceMemory<f64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    let alpha = alpha.into();
    let beta = beta.into();
    let pointer_mode = scalar_pointer_mode(&alpha, &beta)?;
    validate_gemm_parameters(transpose_a, transpose_b, m, n, k, lda, ldb, ldc)?;
    validate_gemm_shapes(
        transpose_a,
        transpose_b,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    ctx.with_pointer_mode(pointer_mode, || unsafe {
        try_ffi!(sys::cublasDgemm_v2(
            ctx.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            m,
            n,
            k,
            alpha.as_ptr(),
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta.as_ptr(),
            c.as_mut_ptr(),
            ldc,
        ))?;
        Ok(())
    })?;

    Ok(())
}

pub fn zgemm<'alpha, 'beta>(
    ctx: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: impl Into<Scalar<'alpha, Complex64>>,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    b: &DeviceMemory<Complex64>,
    ldb: usize,
    beta: impl Into<Scalar<'beta, Complex64>>,
    c: &mut DeviceMemory<Complex64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    let alpha = alpha.into();
    let beta = beta.into();
    let pointer_mode = scalar_pointer_mode(&alpha, &beta)?;
    validate_gemm_parameters(transpose_a, transpose_b, m, n, k, lda, ldb, ldc)?;
    validate_gemm_shapes(
        transpose_a,
        transpose_b,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    ctx.with_pointer_mode(pointer_mode, || unsafe {
        try_ffi!(sys::cublasZgemm_v2(
            ctx.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            m,
            n,
            k,
            alpha.as_ptr().cast(),
            a.as_ptr().cast(),
            lda,
            b.as_ptr().cast(),
            ldb,
            beta.as_ptr().cast(),
            c.as_mut_ptr().cast(),
            ldc,
        ))?;
        Ok(())
    })?;

    Ok(())
}

pub fn cgemm3m<'alpha, 'beta>(
    ctx: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: impl Into<Scalar<'alpha, Complex32>>,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    b: &DeviceMemory<Complex32>,
    ldb: usize,
    beta: impl Into<Scalar<'beta, Complex32>>,
    c: &mut DeviceMemory<Complex32>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    let alpha = alpha.into();
    let beta = beta.into();
    let pointer_mode = scalar_pointer_mode(&alpha, &beta)?;
    validate_gemm_parameters(transpose_a, transpose_b, m, n, k, lda, ldb, ldc)?;
    validate_gemm_shapes(
        transpose_a,
        transpose_b,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    ctx.with_pointer_mode(pointer_mode, || unsafe {
        try_ffi!(sys::cublasCgemm3m(
            ctx.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            m,
            n,
            k,
            alpha.as_ptr().cast(),
            a.as_ptr().cast(),
            lda,
            b.as_ptr().cast(),
            ldb,
            beta.as_ptr().cast(),
            c.as_mut_ptr().cast(),
            ldc,
        ))?;
        Ok(())
    })?;

    Ok(())
}

pub fn dsymm(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    m: usize,
    n: usize,
    alpha: &f64,
    a: &DeviceMemory<f64>,
    lda: usize,
    b: &DeviceMemory<f64>,
    ldb: usize,
    beta: &f64,
    c: &mut DeviceMemory<f64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_symmetric_matrix_multiply_shapes(
        side,
        m,
        n,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasDsymm_v2(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn zhemm(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    m: usize,
    n: usize,
    alpha: &Complex64,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    b: &DeviceMemory<Complex64>,
    ldb: usize,
    beta: &Complex64,
    c: &mut DeviceMemory<Complex64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_symmetric_matrix_multiply_shapes(
        side,
        m,
        n,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasZhemm_v2(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            m,
            n,
            ptr::from_ref(alpha).cast(),
            a.as_ptr().cast(),
            lda,
            b.as_ptr().cast(),
            ldb,
            ptr::from_ref(beta).cast(),
            c.as_mut_ptr().cast(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn dsyrk(
    ctx: &Context,
    fill_mode: FillMode,
    transpose_a: Operation,
    n: usize,
    k: usize,
    alpha: &f64,
    a: &DeviceMemory<f64>,
    lda: usize,
    beta: &f64,
    c: &mut DeviceMemory<f64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_syrk_shapes(transpose_a, n, k, a.len(), lda, c.len(), ldc)?;

    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasDsyrk_v2(
            ctx.as_raw(),
            fill_mode.into(),
            transpose_a.into(),
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            beta,
            c.as_mut_ptr(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn zherk(
    ctx: &Context,
    fill_mode: FillMode,
    transpose_a: Operation,
    n: usize,
    k: usize,
    alpha: &f64,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    beta: &f64,
    c: &mut DeviceMemory<Complex64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_syrk_shapes(transpose_a, n, k, a.len(), lda, c.len(), ldc)?;

    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasZherk_v2(
            ctx.as_raw(),
            fill_mode.into(),
            transpose_a.into(),
            n,
            k,
            alpha,
            a.as_ptr().cast(),
            lda,
            beta,
            c.as_mut_ptr().cast(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn dsyr2k(
    ctx: &Context,
    fill_mode: FillMode,
    transpose_a: Operation,
    n: usize,
    k: usize,
    alpha: &f64,
    a: &DeviceMemory<f64>,
    lda: usize,
    b: &DeviceMemory<f64>,
    ldb: usize,
    beta: &f64,
    c: &mut DeviceMemory<f64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_syr2k_shapes(transpose_a, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc)?;

    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasDsyr2k_v2(
            ctx.as_raw(),
            fill_mode.into(),
            transpose_a.into(),
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn zher2k(
    ctx: &Context,
    fill_mode: FillMode,
    transpose_a: Operation,
    n: usize,
    k: usize,
    alpha: &Complex64,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    b: &DeviceMemory<Complex64>,
    ldb: usize,
    beta: &f64,
    c: &mut DeviceMemory<Complex64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_syr2k_shapes(transpose_a, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc)?;

    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasZher2k_v2(
            ctx.as_raw(),
            fill_mode.into(),
            transpose_a.into(),
            n,
            k,
            ptr::from_ref(alpha).cast(),
            a.as_ptr().cast(),
            lda,
            b.as_ptr().cast(),
            ldb,
            beta,
            c.as_mut_ptr().cast(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn dsyrkx(
    ctx: &Context,
    fill_mode: FillMode,
    transpose_a: Operation,
    n: usize,
    k: usize,
    alpha: &f64,
    a: &DeviceMemory<f64>,
    lda: usize,
    b: &DeviceMemory<f64>,
    ldb: usize,
    beta: &f64,
    c: &mut DeviceMemory<f64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_syr2k_shapes(transpose_a, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc)?;

    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasDsyrkx(
            ctx.as_raw(),
            fill_mode.into(),
            transpose_a.into(),
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn zherkx(
    ctx: &Context,
    fill_mode: FillMode,
    transpose_a: Operation,
    n: usize,
    k: usize,
    alpha: &Complex64,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    b: &DeviceMemory<Complex64>,
    ldb: usize,
    beta: &f64,
    c: &mut DeviceMemory<Complex64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_syr2k_shapes(transpose_a, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc)?;

    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasZherkx(
            ctx.as_raw(),
            fill_mode.into(),
            transpose_a.into(),
            n,
            k,
            ptr::from_ref(alpha).cast(),
            a.as_ptr().cast(),
            lda,
            b.as_ptr().cast(),
            ldb,
            beta,
            c.as_mut_ptr().cast(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn dtrmm(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    transpose_a: Operation,
    diagonal_type: DiagonalType,
    m: usize,
    n: usize,
    alpha: &f64,
    a: &DeviceMemory<f64>,
    lda: usize,
    b: &DeviceMemory<f64>,
    ldb: usize,
    c: &mut DeviceMemory<f64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_triangular_matrix_multiply_shapes(
        side,
        m,
        n,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasDtrmm_v2(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            transpose_a.into(),
            diagonal_type.into(),
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            c.as_mut_ptr(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn dtrsm(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    transpose_a: Operation,
    diagonal_type: DiagonalType,
    m: usize,
    n: usize,
    alpha: &f64,
    a: &DeviceMemory<f64>,
    lda: usize,
    b: &mut DeviceMemory<f64>,
    ldb: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_triangular_solve_shapes(side, m, n, a.len(), lda, b.len(), ldb)?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;

    unsafe {
        try_ffi!(sys::cublasDtrsm_v2(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            transpose_a.into(),
            diagonal_type.into(),
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            b.as_mut_ptr(),
            ldb,
        ))?;
    }

    Ok(())
}

pub fn dgemm_grouped_batched(
    ctx: &Context,
    transpose_a: &[Operation],
    transpose_b: &[Operation],
    m: &[usize],
    n: &[usize],
    k: &[usize],
    alpha: &[f64],
    a: &[DevicePtr],
    lda: &[usize],
    b: &[DevicePtr],
    ldb: &[usize],
    beta: &[f64],
    c: &mut [DevicePtr],
    ldc: &[usize],
    group_size: &[usize],
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;

    let group_count = transpose_a.len();
    if group_count == 0
        || transpose_b.len() != group_count
        || m.len() != group_count
        || n.len() != group_count
        || k.len() != group_count
        || alpha.len() != group_count
        || lda.len() != group_count
        || b.len() != a.len()
        || ldb.len() != group_count
        || beta.len() != group_count
        || ldc.len() != group_count
        || group_size.len() != group_count
    {
        return Err(Error::InvalidMatrixShape);
    }

    let gemm_count = group_size
        .iter()
        .try_fold(0usize, |acc, &size| acc.checked_add(size))
        .ok_or(Error::OutOfRange {
            name: "grouped batch size".into(),
        })?;
    if a.len() != gemm_count || c.len() != gemm_count {
        return Err(Error::InvalidMatrixShape);
    }

    for index in 0..group_count {
        validate_gemm_parameters(
            transpose_a[index],
            transpose_b[index],
            m[index],
            n[index],
            k[index],
            lda[index],
            ldb[index],
            ldc[index],
        )?;
    }

    let transpose_a = transpose_a
        .iter()
        .copied()
        .map(Into::into)
        .collect::<Vec<_>>();
    let transpose_b = transpose_b
        .iter()
        .copied()
        .map(Into::into)
        .collect::<Vec<_>>();
    let m = m
        .iter()
        .copied()
        .map(|value| to_i32(value, "m"))
        .collect::<Result<Vec<_>>>()?;
    let n = n
        .iter()
        .copied()
        .map(|value| to_i32(value, "n"))
        .collect::<Result<Vec<_>>>()?;
    let k = k
        .iter()
        .copied()
        .map(|value| to_i32(value, "k"))
        .collect::<Result<Vec<_>>>()?;
    let lda = lda
        .iter()
        .copied()
        .map(|value| to_i32(value, "lda"))
        .collect::<Result<Vec<_>>>()?;
    let ldb = ldb
        .iter()
        .copied()
        .map(|value| to_i32(value, "ldb"))
        .collect::<Result<Vec<_>>>()?;
    let ldc = ldc
        .iter()
        .copied()
        .map(|value| to_i32(value, "ldc"))
        .collect::<Result<Vec<_>>>()?;
    let group_size = group_size
        .iter()
        .copied()
        .map(|value| to_i32(value, "group_size"))
        .collect::<Result<Vec<_>>>()?;
    let group_count = to_i32(group_count, "group_count")?;

    let a_device = DeviceMemory::from_slice(a)?;
    let b_device = DeviceMemory::from_slice(b)?;
    let c_device = DeviceMemory::from_slice(c)?;

    unsafe {
        try_ffi!(sys::cublasDgemmGroupedBatched(
            ctx.as_raw(),
            transpose_a.as_ptr(),
            transpose_b.as_ptr(),
            m.as_ptr(),
            n.as_ptr(),
            k.as_ptr(),
            alpha.as_ptr(),
            a_device.as_ptr().cast(),
            lda.as_ptr(),
            b_device.as_ptr().cast(),
            ldb.as_ptr(),
            beta.as_ptr(),
            c_device.as_ptr().cast(),
            ldc.as_ptr(),
            group_count,
            group_size.as_ptr(),
        ))?;
    }

    Ok(())
}

pub fn dtrsm_batched(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    transpose_a: Operation,
    diagonal_type: DiagonalType,
    m: usize,
    n: usize,
    alpha: &f64,
    a: &[DevicePtr],
    lda: usize,
    b: &mut [DevicePtr],
    ldb: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;

    if a.len() != b.len() {
        return Err(Error::InvalidMatrixShape);
    }
    validate_triangular_solve_parameters(side, m, n, lda, ldb)?;

    let batch_count = to_i32(a.len(), "batch_count")?;
    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;

    let a_device = DeviceMemory::from_slice(a)?;
    let b_device = DeviceMemory::from_slice(b)?;

    unsafe {
        try_ffi!(sys::cublasDtrsmBatched(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            transpose_a.into(),
            diagonal_type.into(),
            m,
            n,
            alpha,
            a_device.as_ptr().cast(),
            lda,
            b_device.as_ptr().cast(),
            ldb,
            batch_count,
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Extends the typed cuBLAS GEMM routines by allowing separate data types for the A, B, and C
/// matrices, the compute precision, and the GEMM algorithm.
/// This operation is only supported on devices with compute capability 5.0 or later.
///
/// $$ C = \alpha\text{op}(A)\text{op}(B) + \beta C $$
///
/// Here $\alpha$ and $\beta$ are scalars, and $A$, $B$ and $C$ are matrices stored in column-major format with dimensions $\text{op}(A)$ $m \times k$, $\text{op}(B)$ $k \times n$ and $C$ $m \times n$, respectively.
/// For matrix $A$:
///
/// $$ \text{op}(A) = \left\\{ \begin{matrix} A & {\text{if } \texttt{transpose_a = Operation::NonTranspose}} \\\\ A^{T} & {\text{if } \texttt{transpose_a = Operation::Transpose}} \\\\ A^{H} & {\text{if } \texttt{transpose_a = Operation::ConjugateTranspose}} \\\\ \end{matrix} \right. $$
///
/// $\text{op}(B)$ is defined similarly for matrix $B$.
///
/// Starting with cuBLAS 11.2, using typed functions instead of extension
/// functions such as `cublasGemmEx` helps reduce binary size when linking to
/// the static cuBLAS library.
///
/// See [sgemm()](https://www.netlib.org/blas/sgemm.f).
///
/// For more information about the numerical behavior of some GEMM algorithms, refer to the GEMM Algorithms Numerical Behavior section.
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode, the
/// device does not support GEMM Ex, dimensions or leading dimensions are
/// invalid, device allocations are too small, a data type or algorithm
/// combination is unsupported, or cuBLAS rejects or fails the operation.
pub fn gemm_ex<TAlpha, TA, TB, TBeta, TC>(
    ctx: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: &TAlpha,
    a: &DeviceMemory<TA>,
    a_type: DataType,
    lda: usize,
    b: &DeviceMemory<TB>,
    b_type: DataType,
    ldb: usize,
    beta: &TBeta,
    c: &mut DeviceMemory<TC>,
    c_type: DataType,
    ldc: usize,
    compute_type: ComputeType,
    algorithm: GemmAlgorithm,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_gemm_parameters(transpose_a, transpose_b, m, n, k, lda, ldb, ldc)?;
    validate_gemm_shapes(
        transpose_a,
        transpose_b,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasGemmEx(
            ctx.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            m,
            n,
            k,
            ptr::from_ref(alpha) as _,
            a.as_ptr() as _,
            a_type.into(),
            lda,
            b.as_ptr() as _,
            b_type.into(),
            ldb,
            ptr::from_ref(beta) as _,
            c.as_mut_ptr() as _,
            c_type.into(),
            ldc,
            compute_type.into(),
            algorithm.into(),
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Extends the typed cuBLAS batched GEMM routines by performing batched matrix multiplication
/// with separate data types for the A, B, and C matrix arrays, the compute precision,
/// and the GEMM algorithm.
/// Like the typed cuBLAS batched GEMM routines, the batch is uniform: all instances have the
/// same dimensions (`m`, `n`, `k`), leading dimensions (`lda`, `ldb`, `ldc`),
/// and transpositions (`transpose_a`, `transpose_b`) for their A, B, and C matrices.
/// `a`, `b`, and `c` provide one matrix pointer per batch instance.
/// $$ C\lbrack i\rbrack = \alpha\text{op}(A\lbrack i\rbrack)\text{op}(B\lbrack i\rbrack) + \beta C\lbrack i\rbrack,\text{ for i } \in \lbrack 0,\text{batch_count} - 1\rbrack $$
///
/// Here $\alpha$ and $\beta$ are scalars. `a`, `b`, and `c` are arrays of pointers
/// to column-major matrices with dimensions $\text{op}(A\lbrack i\rbrack)$
/// $m \times k$, $\text{op}(B\lbrack i\rbrack)$ $k \times n$, and
/// $C\lbrack i\rbrack$ $m \times n$, respectively.
/// For matrix $A$:
///
/// $$ \text{op}(A) = \left\\{ \begin{matrix} A & {\text{if } \texttt{transpose_a = Operation::NonTranspose}} \\\\ A^{T} & {\text{if } \texttt{transpose_a = Operation::Transpose}} \\\\ A^{H} & {\text{if } \texttt{transpose_a = Operation::ConjugateTranspose}} \\\\ \end{matrix} \right. $$
///
/// $\text{op}(B\lbrack i\rbrack)$ is defined similarly for matrix $B\lbrack i\rbrack$.
///
/// For some problem sizes, multiple typed cuBLAS GEMM calls in different CUDA
/// streams may be faster than this batched routine.
///
/// If `a_type` is [`DataType::F16`] or [`DataType::Bf16`], or `compute_type` is any of the `FAST` options, or when math mode or `algorithm` enable fast math modes, pointers (not the pointer arrays) placed in the GPU memory must be properly aligned to avoid misaligned memory access errors.
/// Ideally all matrix pointers are aligned to at least 16 bytes.
///
/// See [sgemm()](https://www.netlib.org/blas/sgemm.f).
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode, the
/// device does not support batched GEMM Ex, batch pointer slices have different
/// lengths, dimensions or leading dimensions are invalid, a data type or
/// algorithm combination is unsupported, or cuBLAS rejects or fails the
/// operation.
pub fn gemm_batched_ex<TAlpha, TBeta>(
    ctx: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: &TAlpha,
    a: &[DevicePtr],
    a_type: DataType,
    lda: usize,
    b: &[DevicePtr],
    b_type: DataType,
    ldb: usize,
    beta: &TBeta,
    c: &mut [DevicePtr],
    c_type: DataType,
    ldc: usize,
    compute_type: ComputeType,
    algorithm: GemmAlgorithm,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;

    if a.len() != b.len() || a.len() != c.len() {
        return Err(Error::InvalidMatrixShape);
    }
    validate_gemm_parameters(transpose_a, transpose_b, m, n, k, lda, ldb, ldc)?;

    let batch_count = to_i32(a.len(), "batch count")?;
    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    let a = a.iter().map(|ptr| ptr.as_const_ptr()).collect::<Vec<_>>();
    let b = b.iter().map(|ptr| ptr.as_const_ptr()).collect::<Vec<_>>();
    let mut c = c.iter().map(|ptr| ptr.as_ptr()).collect::<Vec<_>>();

    unsafe {
        try_ffi!(sys::cublasGemmBatchedEx(
            ctx.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            m,
            n,
            k,
            ptr::from_ref(alpha) as _,
            a.as_ptr() as _,
            a_type.into(),
            lda,
            b.as_ptr() as _,
            b_type.into(),
            ldb,
            ptr::from_ref(beta) as _,
            c.as_mut_ptr() as _,
            c_type.into(),
            ldc,
            batch_count,
            compute_type.into(),
            algorithm.into(),
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Extends the typed cuBLAS strided-batched GEMM routines by performing strided batched matrix
/// multiplication with separate data types for the A, B, and C matrices, the compute
/// precision, and the GEMM algorithm.
/// Like the typed cuBLAS strided-batched GEMM routines, the batch is uniform: all instances have
/// the same dimensions (`m`, `n`, `k`), leading dimensions (`lda`, `ldb`, `ldc`),
/// and transpositions (`transpose_a`, `transpose_b`) for their A, B, and C matrices.
/// Each batch instance is located at a fixed element offset from the previous instance.
/// The first A, B, and C matrices are passed with element strides `stride_a`,
/// `stride_b`, and `stride_c`, which determine the locations of later batch instances.
///
/// $$ C + i\cdot {\text{stride_c}} = \alpha\text{op}(A + i\cdot {\text{stride_a}})\text{op}(B + i\cdot {\text{stride_b}}) + \beta(C + i\cdot {\text{stride_c}}),\text{ for i } \in \lbrack 0,\text{batch_count} - 1\rbrack $$
///
/// Here $\alpha$ and $\beta$ are scalars. `a`, `b`, and `c` are strided
/// column-major matrix batches with dimensions $\text{op}(A\lbrack i\rbrack)$
/// $m \times k$, $\text{op}(B\lbrack i\rbrack)$ $k \times n$, and
/// $C\lbrack i\rbrack$ $m \times n$, respectively.
/// For matrix $A$:
///
/// $$ \text{op}(A) = \left\\{ \begin{matrix} A & {\text{if } \texttt{transpose_a = Operation::NonTranspose}} \\\\ A^{T} & {\text{if } \texttt{transpose_a = Operation::Transpose}} \\\\ A^{H} & {\text{if } \texttt{transpose_a = Operation::ConjugateTranspose}} \\\\ \end{matrix} \right. $$
///
/// $\text{op}(B\lbrack i\rbrack)$ is defined similarly for matrix $B\lbrack i\rbrack$.
///
/// For some problem sizes, multiple typed cuBLAS GEMM calls in different CUDA
/// streams may be faster than this batched routine.
///
/// See [sgemm()](https://www.netlib.org/blas/sgemm.f).
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode, the
/// device does not support strided-batched GEMM Ex, dimensions, leading
/// dimensions, or strides are invalid, device allocations are too small, a data
/// type or algorithm combination is unsupported, or cuBLAS rejects or fails the
/// operation.
pub fn gemm_strided_batched_ex<TAlpha, TA, TB, TBeta, TC>(
    ctx: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: &TAlpha,
    a: &DeviceMemory<TA>,
    a_type: DataType,
    lda: usize,
    stride_a: i64,
    b: &DeviceMemory<TB>,
    b_type: DataType,
    ldb: usize,
    stride_b: i64,
    beta: &TBeta,
    c: &mut DeviceMemory<TC>,
    c_type: DataType,
    ldc: usize,
    stride_c: i64,
    batch_count: usize,
    compute_type: ComputeType,
    algorithm: GemmAlgorithm,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_gemm_parameters(transpose_a, transpose_b, m, n, k, lda, ldb, ldc)?;
    validate_gemm_shapes(
        transpose_a,
        transpose_b,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;
    validate_strided_batched_gemm_shapes(
        transpose_a,
        transpose_b,
        m,
        n,
        k,
        a.len(),
        lda,
        stride_a,
        b.len(),
        ldb,
        stride_b,
        c.len(),
        ldc,
        stride_c,
        batch_count,
    )?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;
    let batch_count = to_i32(batch_count, "batch_count")?;

    unsafe {
        try_ffi!(sys::cublasGemmStridedBatchedEx(
            ctx.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            m,
            n,
            k,
            ptr::from_ref(alpha) as _,
            a.as_ptr() as _,
            a_type.into(),
            lda,
            stride_a,
            b.as_ptr() as _,
            b_type.into(),
            ldb,
            stride_b,
            ptr::from_ref(beta) as _,
            c.as_mut_ptr() as _,
            c_type.into(),
            ldc,
            stride_c,
            batch_count,
            compute_type.into(),
            algorithm.into(),
        ))?;
    }

    Ok(())
}

pub fn gemm_grouped_batched_ex<TAlpha, TBeta>(
    ctx: &Context,
    transpose_a: &[Operation],
    transpose_b: &[Operation],
    m: &[usize],
    n: &[usize],
    k: &[usize],
    alpha: &[TAlpha],
    a: &[DevicePtr],
    a_type: DataType,
    lda: &[usize],
    b: &[DevicePtr],
    b_type: DataType,
    ldb: &[usize],
    beta: &[TBeta],
    c: &mut [DevicePtr],
    c_type: DataType,
    ldc: &[usize],
    group_size: &[usize],
    compute_type: ComputeType,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;

    let group_count = transpose_a.len();
    if group_count == 0
        || transpose_b.len() != group_count
        || m.len() != group_count
        || n.len() != group_count
        || k.len() != group_count
        || alpha.len() != group_count
        || lda.len() != group_count
        || b.len() != a.len()
        || ldb.len() != group_count
        || beta.len() != group_count
        || ldc.len() != group_count
        || group_size.len() != group_count
    {
        return Err(Error::InvalidMatrixShape);
    }

    let gemm_count = group_size
        .iter()
        .try_fold(0usize, |acc, &size| acc.checked_add(size))
        .ok_or(Error::OutOfRange {
            name: "grouped batch size".into(),
        })?;
    if a.len() != gemm_count || c.len() != gemm_count {
        return Err(Error::InvalidMatrixShape);
    }

    for index in 0..group_count {
        validate_gemm_parameters(
            transpose_a[index],
            transpose_b[index],
            m[index],
            n[index],
            k[index],
            lda[index],
            ldb[index],
            ldc[index],
        )?;
    }

    let transpose_a = transpose_a
        .iter()
        .copied()
        .map(Into::into)
        .collect::<Vec<_>>();
    let transpose_b = transpose_b
        .iter()
        .copied()
        .map(Into::into)
        .collect::<Vec<_>>();
    let m = m
        .iter()
        .copied()
        .map(|value| to_i32(value, "m"))
        .collect::<Result<Vec<_>>>()?;
    let n = n
        .iter()
        .copied()
        .map(|value| to_i32(value, "n"))
        .collect::<Result<Vec<_>>>()?;
    let k = k
        .iter()
        .copied()
        .map(|value| to_i32(value, "k"))
        .collect::<Result<Vec<_>>>()?;
    let lda = lda
        .iter()
        .copied()
        .map(|value| to_i32(value, "lda"))
        .collect::<Result<Vec<_>>>()?;
    let ldb = ldb
        .iter()
        .copied()
        .map(|value| to_i32(value, "ldb"))
        .collect::<Result<Vec<_>>>()?;
    let ldc = ldc
        .iter()
        .copied()
        .map(|value| to_i32(value, "ldc"))
        .collect::<Result<Vec<_>>>()?;
    let group_size = group_size
        .iter()
        .copied()
        .map(|value| to_i32(value, "group_size"))
        .collect::<Result<Vec<_>>>()?;
    let group_count = to_i32(group_count, "group_count")?;

    let a_device = DeviceMemory::from_slice(a)?;
    let b_device = DeviceMemory::from_slice(b)?;
    let c_device = DeviceMemory::from_slice(c)?;

    unsafe {
        try_ffi!(sys::cublasGemmGroupedBatchedEx(
            ctx.as_raw(),
            transpose_a.as_ptr(),
            transpose_b.as_ptr(),
            m.as_ptr(),
            n.as_ptr(),
            k.as_ptr(),
            alpha.as_ptr().cast(),
            a_device.as_ptr().cast(),
            a_type.into(),
            lda.as_ptr(),
            b_device.as_ptr().cast(),
            b_type.into(),
            ldb.as_ptr(),
            beta.as_ptr().cast(),
            c_device.as_ptr().cast(),
            c_type.into(),
            ldc.as_ptr(),
            group_count,
            group_size.as_ptr(),
            compute_type.into(),
        ))?;
    }

    Ok(())
}

pub fn dgeam(
    ctx: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    alpha: &f64,
    a: &DeviceMemory<f64>,
    lda: usize,
    beta: &f64,
    b: &DeviceMemory<f64>,
    ldb: usize,
    c: &mut DeviceMemory<f64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_geam_shapes(
        transpose_a,
        transpose_b,
        m,
        n,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasDgeam(
            ctx.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            beta,
            b.as_ptr(),
            ldb,
            c.as_mut_ptr(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn ddgmm(
    ctx: &Context,
    side: SideMode,
    m: usize,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    x: &DeviceMemory<f64>,
    incx: usize,
    c: &mut DeviceMemory<f64>,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_dgmm_shapes(side, m, n, a.len(), lda, x.len(), incx, c.len(), ldc)?;

    let m = to_i32(m, "m")?;
    let n = to_i32(n, "n")?;
    let lda = to_i32(lda, "lda")?;
    let incx = to_i32(incx, "incx")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasDdgmm(
            ctx.as_raw(),
            side.into(),
            m,
            n,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            c.as_mut_ptr(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn dtpttr(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    ap: &DeviceMemory<f64>,
    a: &mut DeviceMemory<f64>,
    lda: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_tpttr_shapes(n, ap.len(), a.len(), lda)?;

    let n = to_i32(n, "n")?;
    let lda = to_i32(lda, "lda")?;

    unsafe {
        try_ffi!(sys::cublasDtpttr(
            ctx.as_raw(),
            fill_mode.into(),
            n,
            ap.as_ptr(),
            a.as_mut_ptr(),
            lda,
        ))?;
    }

    Ok(())
}

pub fn dtrttp(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    ap: &mut DeviceMemory<f64>,
) -> Result<()> {
    ctx.bind()?;
    validate_tpttr_shapes(n, ap.len(), a.len(), lda)?;

    let n = to_i32(n, "n")?;
    let lda = to_i32(lda, "lda")?;

    unsafe {
        try_ffi!(sys::cublasDtrttp(
            ctx.as_raw(),
            fill_mode.into(),
            n,
            a.as_ptr(),
            lda,
            ap.as_mut_ptr(),
        ))?;
    }

    Ok(())
}

pub fn csyrk_ex<TA, TC>(
    ctx: &Context,
    fill_mode: FillMode,
    transpose_a: Operation,
    n: usize,
    k: usize,
    alpha: &Complex32,
    a: &DeviceMemory<TA>,
    a_type: DataType,
    lda: usize,
    beta: &Complex32,
    c: &mut DeviceMemory<TC>,
    c_type: DataType,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_syrk_shapes(transpose_a, n, k, a.len(), lda, c.len(), ldc)?;

    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasCsyrkEx(
            ctx.as_raw(),
            fill_mode.into(),
            transpose_a.into(),
            n,
            k,
            ptr::from_ref(alpha).cast(),
            a.as_ptr().cast(),
            a_type.into(),
            lda,
            ptr::from_ref(beta).cast(),
            c.as_mut_ptr().cast(),
            c_type.into(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn csyrk3m_ex<TA, TC>(
    ctx: &Context,
    fill_mode: FillMode,
    transpose_a: Operation,
    n: usize,
    k: usize,
    alpha: &Complex32,
    a: &DeviceMemory<TA>,
    a_type: DataType,
    lda: usize,
    beta: &Complex32,
    c: &mut DeviceMemory<TC>,
    c_type: DataType,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_syrk_shapes(transpose_a, n, k, a.len(), lda, c.len(), ldc)?;

    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasCsyrk3mEx(
            ctx.as_raw(),
            fill_mode.into(),
            transpose_a.into(),
            n,
            k,
            ptr::from_ref(alpha).cast(),
            a.as_ptr().cast(),
            a_type.into(),
            lda,
            ptr::from_ref(beta).cast(),
            c.as_mut_ptr().cast(),
            c_type.into(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn cherk_ex<TA, TC>(
    ctx: &Context,
    fill_mode: FillMode,
    transpose_a: Operation,
    n: usize,
    k: usize,
    alpha: &f32,
    a: &DeviceMemory<TA>,
    a_type: DataType,
    lda: usize,
    beta: &f32,
    c: &mut DeviceMemory<TC>,
    c_type: DataType,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_syrk_shapes(transpose_a, n, k, a.len(), lda, c.len(), ldc)?;

    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasCherkEx(
            ctx.as_raw(),
            fill_mode.into(),
            transpose_a.into(),
            n,
            k,
            alpha,
            a.as_ptr().cast(),
            a_type.into(),
            lda,
            beta,
            c.as_mut_ptr().cast(),
            c_type.into(),
            ldc,
        ))?;
    }

    Ok(())
}

pub fn cherk3m_ex<TA, TC>(
    ctx: &Context,
    fill_mode: FillMode,
    transpose_a: Operation,
    n: usize,
    k: usize,
    alpha: &f32,
    a: &DeviceMemory<TA>,
    a_type: DataType,
    lda: usize,
    beta: &f32,
    c: &mut DeviceMemory<TC>,
    c_type: DataType,
    ldc: usize,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    validate_syrk_shapes(transpose_a, n, k, a.len(), lda, c.len(), ldc)?;

    let n = to_i32(n, "n")?;
    let k = to_i32(k, "k")?;
    let lda = to_i32(lda, "lda")?;
    let ldc = to_i32(ldc, "ldc")?;

    unsafe {
        try_ffi!(sys::cublasCherk3mEx(
            ctx.as_raw(),
            fill_mode.into(),
            transpose_a.into(),
            n,
            k,
            alpha,
            a.as_ptr().cast(),
            a_type.into(),
            lda,
            beta,
            c.as_mut_ptr().cast(),
            c_type.into(),
            ldc,
        ))?;
    }

    Ok(())
}

fn validate_gemm_shapes(
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    let a_cols = if transpose_a == Operation::NonTranspose {
        k
    } else {
        m
    };
    let b_cols = if transpose_b == Operation::NonTranspose {
        n
    } else {
        k
    };

    if a_len < required_matrix_len(lda, a_cols)? {
        return Err(Error::InvalidMatrixShape);
    }
    if b_len < required_matrix_len(ldb, b_cols)? {
        return Err(Error::InvalidMatrixShape);
    }
    if c_len < required_matrix_len(ldc, n)? {
        return Err(Error::InvalidMatrixShape);
    }

    Ok(())
}

fn validate_gemm_parameters(
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) -> Result<()> {
    let a_rows = if transpose_a == Operation::NonTranspose {
        m
    } else {
        k
    };
    let b_rows = if transpose_b == Operation::NonTranspose {
        k
    } else {
        n
    };

    if lda < a_rows || ldb < b_rows || ldc < m {
        return Err(Error::InvalidLeadingDimension);
    }

    Ok(())
}

fn validate_strided_batched_gemm_shapes(
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    stride_a: i64,
    b_len: usize,
    ldb: usize,
    stride_b: i64,
    c_len: usize,
    ldc: usize,
    stride_c: i64,
    batch_count: usize,
) -> Result<()> {
    if batch_count == 0 {
        return Ok(());
    }

    let a_cols = if transpose_a == Operation::NonTranspose {
        k
    } else {
        m
    };
    let b_cols = if transpose_b == Operation::NonTranspose {
        n
    } else {
        k
    };

    validate_strided_buffer_len(
        a_len,
        required_matrix_len(lda, a_cols)?,
        stride_a,
        batch_count,
    )?;
    validate_strided_buffer_len(
        b_len,
        required_matrix_len(ldb, b_cols)?,
        stride_b,
        batch_count,
    )?;
    validate_strided_buffer_len(c_len, required_matrix_len(ldc, n)?, stride_c, batch_count)?;

    Ok(())
}

fn validate_strided_buffer_len(
    length: usize,
    matrix_len: usize,
    stride: i64,
    batch_count: usize,
) -> Result<()> {
    let stride = to_usize(stride, "stride")?;
    if stride < matrix_len {
        return Err(Error::InvalidMatrixShape);
    }

    let total = if batch_count == 0 {
        0
    } else {
        batch_count
            .checked_sub(1)
            .and_then(|count| count.checked_mul(stride))
            .and_then(|offset| offset.checked_add(matrix_len))
            .ok_or(Error::OutOfRange {
                name: "strided batch length".into(),
            })?
    };

    if length < total {
        return Err(Error::InvalidMatrixShape);
    }

    Ok(())
}

fn validate_geam_shapes(
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    let a_rows = if transpose_a == Operation::NonTranspose {
        m
    } else {
        n
    };
    let a_cols = if transpose_a == Operation::NonTranspose {
        n
    } else {
        m
    };
    let b_rows = if transpose_b == Operation::NonTranspose {
        m
    } else {
        n
    };
    let b_cols = if transpose_b == Operation::NonTranspose {
        n
    } else {
        m
    };

    if lda < a_rows || ldb < b_rows || ldc < m {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, a_cols)?
        || b_len < required_matrix_len(ldb, b_cols)?
        || c_len < required_matrix_len(ldc, n)?
    {
        return Err(Error::InvalidMatrixShape);
    }

    Ok(())
}

fn validate_dgmm_shapes(
    side: SideMode,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    incx: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    if lda < m || ldc < m || incx == 0 {
        return Err(Error::InvalidIncrement);
    }
    let x_size = match side {
        SideMode::Left => m,
        SideMode::Right => n,
    };
    if a_len < required_matrix_len(lda, n)?
        || c_len < required_matrix_len(ldc, n)?
        || x_len < required_vector_len(x_size, incx)?
    {
        return Err(Error::InvalidMatrixShape);
    }

    Ok(())
}

fn validate_tpttr_shapes(n: usize, ap_len: usize, a_len: usize, lda: usize) -> Result<()> {
    if lda < n {
        return Err(Error::InvalidLeadingDimension);
    }
    let packed_len = required_packed_triangular_len(n)?;
    if ap_len < packed_len || a_len < required_matrix_len(lda, n)? {
        return Err(Error::InvalidMatrixShape);
    }

    Ok(())
}

fn required_packed_triangular_len(n: usize) -> Result<usize> {
    n.checked_mul(n.checked_add(1).ok_or(Error::OutOfRange {
        name: "packed triangular length".into(),
    })?)
    .and_then(|value| value.checked_div(2))
    .ok_or(Error::OutOfRange {
        name: "packed triangular length".into(),
    })
}

fn validate_symmetric_matrix_multiply_shapes(
    side: SideMode,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    let a_dimension = match side {
        SideMode::Left => m,
        SideMode::Right => n,
    };
    if lda < a_dimension || ldb < m || ldc < m {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, a_dimension)? {
        return Err(Error::InvalidMatrixShape);
    }
    if b_len < required_matrix_len(ldb, n)? || c_len < required_matrix_len(ldc, n)? {
        return Err(Error::InvalidMatrixShape);
    }

    Ok(())
}

fn validate_syrk_shapes(
    transpose_a: Operation,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    let a_rows = if transpose_a == Operation::NonTranspose {
        n
    } else {
        k
    };
    let a_cols = if transpose_a == Operation::NonTranspose {
        k
    } else {
        n
    };
    if lda < a_rows || ldc < n {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, a_cols)? || c_len < required_matrix_len(ldc, n)? {
        return Err(Error::InvalidMatrixShape);
    }

    Ok(())
}

fn validate_syr2k_shapes(
    transpose_a: Operation,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    validate_syrk_shapes(transpose_a, n, k, a_len, lda, c_len, ldc)?;

    let b_rows = if transpose_a == Operation::NonTranspose {
        n
    } else {
        k
    };
    let b_cols = if transpose_a == Operation::NonTranspose {
        k
    } else {
        n
    };
    if ldb < b_rows {
        return Err(Error::InvalidLeadingDimension);
    }
    if b_len < required_matrix_len(ldb, b_cols)? {
        return Err(Error::InvalidMatrixShape);
    }

    Ok(())
}

fn validate_triangular_matrix_multiply_shapes(
    side: SideMode,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    validate_triangular_solve_parameters(side, m, n, lda, ldb)?;
    if ldc < m {
        return Err(Error::InvalidLeadingDimension);
    }

    let a_dimension = match side {
        SideMode::Left => m,
        SideMode::Right => n,
    };
    if a_len < required_matrix_len(lda, a_dimension)?
        || b_len < required_matrix_len(ldb, n)?
        || c_len < required_matrix_len(ldc, n)?
    {
        return Err(Error::InvalidMatrixShape);
    }

    Ok(())
}

fn validate_triangular_solve_shapes(
    side: SideMode,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
) -> Result<()> {
    validate_triangular_solve_parameters(side, m, n, lda, ldb)?;

    let a_dimension = match side {
        SideMode::Left => m,
        SideMode::Right => n,
    };
    if a_len < required_matrix_len(lda, a_dimension)? || b_len < required_matrix_len(ldb, n)? {
        return Err(Error::InvalidMatrixShape);
    }

    Ok(())
}

fn validate_triangular_solve_parameters(
    side: SideMode,
    m: usize,
    n: usize,
    lda: usize,
    ldb: usize,
) -> Result<()> {
    let a_dimension = match side {
        SideMode::Left => m,
        SideMode::Right => n,
    };
    if lda < a_dimension || ldb < m {
        return Err(Error::InvalidLeadingDimension);
    }

    Ok(())
}

fn scalar_pointer_mode<T>(
    alpha: &Scalar<'_, T>,
    beta: &Scalar<'_, T>,
) -> Result<crate::types::PointerMode> {
    let pointer_mode = alpha.pointer_mode();
    if pointer_mode != beta.pointer_mode() {
        return Err(Error::ScalarPointerModeMismatch);
    }
    Ok(pointer_mode)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_sgemm_matches_cpu_reference() -> Result<()> {
        let ctx = setup_context()?;

        let a = DeviceMemory::from_slice(&[1.0_f32, 2.0, 3.0, 4.0])?;
        let b = DeviceMemory::from_slice(&[5.0_f32, 6.0, 7.0, 8.0])?;
        let mut c = DeviceMemory::create(4)?;

        sgemm(
            &ctx,
            Operation::NonTranspose,
            Operation::NonTranspose,
            2,
            2,
            2,
            &1.0,
            &a,
            2,
            &b,
            2,
            &0.0,
            &mut c,
            2,
        )?;

        let result = c.copy_to_host_vec()?;
        assert_eq!(result, vec![23.0, 34.0, 31.0, 46.0]);

        Ok(())
    }

    #[test]
    fn test_dgemm_matches_cpu_reference() -> Result<()> {
        let ctx = setup_context()?;

        let a = DeviceMemory::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])?;
        let b = DeviceMemory::from_slice(&[5.0_f64, 6.0, 7.0, 8.0])?;
        let mut c = DeviceMemory::create(4)?;

        dgemm(
            &ctx,
            Operation::NonTranspose,
            Operation::NonTranspose,
            2,
            2,
            2,
            &1.0,
            &a,
            2,
            &b,
            2,
            &0.0,
            &mut c,
            2,
        )?;

        let result = c.copy_to_host_vec()?;
        assert_eq!(result, vec![23.0, 34.0, 31.0, 46.0]);

        Ok(())
    }
}
