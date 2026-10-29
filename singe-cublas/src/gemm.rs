use singe_cuda::{data_type::DataType, memory::DeviceMemory, types::DevicePtr};

use crate::{
    blas::level3,
    context::Context,
    error::{Error, Result},
    scalar::Scalar,
    types::{ComputeType, GemmAlgorithm, Operation},
};

/// Matrix-matrix multiplication shape and layout.
///
/// Matrices use cuBLAS column-major storage. `m`, `n`, and `k` describe the
/// logical GEMM:
/// `C[m x n] = alpha * op(A)[m x k] * op(B)[k x n] + beta * C[m x n]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmConfig {
    m: usize,
    n: usize,
    k: usize,
    transpose_a: Operation,
    transpose_b: Operation,
}

impl GemmConfig {
    /// Creates a GEMM configuration with untransposed inputs.
    #[must_use]
    pub const fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            transpose_a: Operation::NonTranspose,
            transpose_b: Operation::NonTranspose,
        }
    }

    /// Sets the operation applied to matrix `A` before multiplication.
    #[must_use]
    pub fn with_transpose_a(mut self, transpose: Operation) -> Self {
        self.transpose_a = transpose;
        self
    }

    /// Sets the operation applied to matrix `B` before multiplication.
    #[must_use]
    pub fn with_transpose_b(mut self, transpose: Operation) -> Self {
        self.transpose_b = transpose;
        self
    }

    /// Returns the number of rows in the output matrix.
    #[must_use]
    pub const fn m(self) -> usize {
        self.m
    }

    /// Returns the number of columns in the output matrix.
    #[must_use]
    pub const fn n(self) -> usize {
        self.n
    }

    /// Returns the shared dimension of the operated input matrices.
    #[must_use]
    pub const fn k(self) -> usize {
        self.k
    }

    /// Returns the operation applied to matrix `A`.
    #[must_use]
    pub const fn transpose_a(self) -> Operation {
        self.transpose_a
    }

    /// Returns the operation applied to matrix `B`.
    #[must_use]
    pub const fn transpose_b(self) -> Operation {
        self.transpose_b
    }
}

/// A borrowed column-major device matrix.
#[derive(Debug, Clone, Copy)]
pub struct DeviceMatrix<'a, T> {
    memory: &'a DeviceMemory<T>,
    leading_dimension: usize,
}

impl<'a, T> DeviceMatrix<'a, T> {
    /// Borrows matrix storage with the given leading dimension.
    ///
    /// This constructor stores metadata without validating it against a matrix shape.
    #[must_use]
    pub const fn new(memory: &'a DeviceMemory<T>, leading_dimension: usize) -> Self {
        Self {
            memory,
            leading_dimension,
        }
    }

    /// Returns the underlying device storage.
    #[must_use]
    pub const fn memory(&self) -> &'a DeviceMemory<T> {
        self.memory
    }

    /// Returns the stride between adjacent columns, in elements.
    #[must_use]
    pub const fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }
}

/// An exclusively borrowed column-major device matrix.
#[derive(Debug)]
pub struct DeviceMatrixMut<'a, T> {
    memory: &'a mut DeviceMemory<T>,
    leading_dimension: usize,
}

impl<'a, T> DeviceMatrixMut<'a, T> {
    /// Borrows matrix storage with the given leading dimension.
    ///
    /// This constructor stores metadata without validating it against a matrix shape.
    pub const fn new(memory: &'a mut DeviceMemory<T>, leading_dimension: usize) -> Self {
        Self {
            memory,
            leading_dimension,
        }
    }

    /// Returns a mutable borrow of the underlying device storage.
    pub fn memory(&mut self) -> &mut DeviceMemory<T> {
        self.memory
    }

    /// Returns the stride between adjacent columns, in elements.
    #[must_use]
    pub const fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }
}

/// A borrowed batch of column-major device matrices with a fixed stride.
#[derive(Debug, Clone, Copy)]
pub struct StridedDeviceMatrices<'a, T> {
    memory: &'a DeviceMemory<T>,
    leading_dimension: usize,
    stride: i64,
    batch_count: usize,
}

impl<'a, T> StridedDeviceMatrices<'a, T> {
    /// Borrows a batch of matrices with the given layout metadata.
    ///
    /// This constructor stores metadata without validating it against a matrix shape.
    #[must_use]
    pub const fn new(
        memory: &'a DeviceMemory<T>,
        leading_dimension: usize,
        stride: i64,
        batch_count: usize,
    ) -> Self {
        Self {
            memory,
            leading_dimension,
            stride,
            batch_count,
        }
    }

    /// Returns the underlying device storage.
    #[must_use]
    pub const fn memory(&self) -> &'a DeviceMemory<T> {
        self.memory
    }

    /// Returns the stride between adjacent columns, in elements.
    #[must_use]
    pub const fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }

    /// Returns the offset between the starts of adjacent matrices, in elements.
    #[must_use]
    pub const fn stride(&self) -> i64 {
        self.stride
    }

    /// Returns the number of matrices in the batch.
    #[must_use]
    pub const fn batch_count(&self) -> usize {
        self.batch_count
    }
}

/// An exclusively borrowed batch of column-major device matrices with a fixed stride.
#[derive(Debug)]
pub struct StridedDeviceMatricesMut<'a, T> {
    memory: &'a mut DeviceMemory<T>,
    leading_dimension: usize,
    stride: i64,
    batch_count: usize,
}

impl<'a, T> StridedDeviceMatricesMut<'a, T> {
    /// Borrows a batch of matrices with the given layout metadata.
    ///
    /// This constructor stores metadata without validating it against a matrix shape.
    pub const fn new(
        memory: &'a mut DeviceMemory<T>,
        leading_dimension: usize,
        stride: i64,
        batch_count: usize,
    ) -> Self {
        Self {
            memory,
            leading_dimension,
            stride,
            batch_count,
        }
    }

    /// Returns a mutable borrow of the underlying device storage.
    pub fn memory(&mut self) -> &mut DeviceMemory<T> {
        self.memory
    }

    /// Returns the stride between adjacent columns, in elements.
    #[must_use]
    pub const fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }

    /// Returns the offset between the starts of adjacent matrices, in elements.
    #[must_use]
    pub const fn stride(&self) -> i64 {
        self.stride
    }

    /// Returns the number of matrices in the batch.
    #[must_use]
    pub const fn batch_count(&self) -> usize {
        self.batch_count
    }
}

/// An element type supported by the typed GEMM interface.
///
/// Implementations select the matching cuBLAS data and compute types.
pub trait GemmElement: private::Sealed {
    /// Submits `C = alpha * op(A) * op(B) + beta * C` for this element type.
    ///
    /// # Errors
    ///
    /// Returns an error if scalar pointer modes differ, matrix storage or leading dimensions are invalid, or dimensions exceed the native integer range.
    /// CUDA context binding, pointer-mode changes, and cuBLAS execution errors are propagated.
    ///
    /// # Safety
    ///
    /// Matrix storage and device scalars must remain live and correctly ordered until the context's stream completes.
    /// All input values read by the operation must be initialized, including `C` when `beta` is nonzero.
    /// Output writes must not conflict with other access.
    unsafe fn gemm(
        context: &Context,
        config: GemmConfig,
        alpha: Scalar<'_, Self>,
        a: DeviceMatrix<'_, Self>,
        b: DeviceMatrix<'_, Self>,
        beta: Scalar<'_, Self>,
        c: DeviceMatrixMut<'_, Self>,
    ) -> Result<()>
    where
        Self: Sized;

    /// Multiplies batches of matrices separated by fixed element strides.
    ///
    /// Uses `c.batch_count()` as the number of products.
    ///
    /// # Errors
    ///
    /// Returns an error if storage cannot hold the requested batch, strides or leading dimensions are invalid, or dimensions exceed the native integer range.
    /// Also returns an error if the handle does not use host pointer mode, CUDA context binding fails, or cuBLAS rejects the operation.
    fn strided_batched_gemm(
        context: &Context,
        config: GemmConfig,
        alpha: &Self,
        a: StridedDeviceMatrices<'_, Self>,
        b: StridedDeviceMatrices<'_, Self>,
        beta: &Self,
        c: StridedDeviceMatricesMut<'_, Self>,
    ) -> Result<()>
    where
        Self: Sized;

    /// Multiplies corresponding matrices from three batches.
    ///
    /// Each batch must use a uniform leading dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if batch lengths differ, leading dimensions are invalid or nonuniform within a matrix batch, or dimensions exceed the native integer range.
    /// Also returns an error if the handle does not use host pointer mode, allocating or copying the device pointer arrays fails, CUDA context binding fails, or cuBLAS rejects the operation.
    fn batched_gemm(
        context: &Context,
        config: GemmConfig,
        alpha: &Self,
        a: &[DeviceMatrix<'_, Self>],
        b: &[DeviceMatrix<'_, Self>],
        beta: &Self,
        c: &mut [DeviceMatrixMut<'_, Self>],
    ) -> Result<()>
    where
        Self: Sized;
}

impl GemmElement for f32 {
    unsafe fn gemm(
        context: &Context,
        config: GemmConfig,
        alpha: Scalar<'_, Self>,
        a: DeviceMatrix<'_, Self>,
        b: DeviceMatrix<'_, Self>,
        beta: Scalar<'_, Self>,
        mut c: DeviceMatrixMut<'_, Self>,
    ) -> Result<()> {
        let ldc = c.leading_dimension();
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            level3::sgemm(
                context,
                config.transpose_a(),
                config.transpose_b(),
                config.m(),
                config.n(),
                config.k(),
                alpha,
                a.memory(),
                a.leading_dimension(),
                b.memory(),
                b.leading_dimension(),
                beta,
                c.memory(),
                ldc,
            )
        }
    }

    fn strided_batched_gemm(
        context: &Context,
        config: GemmConfig,
        alpha: &Self,
        a: StridedDeviceMatrices<'_, Self>,
        b: StridedDeviceMatrices<'_, Self>,
        beta: &Self,
        mut c: StridedDeviceMatricesMut<'_, Self>,
    ) -> Result<()> {
        let ldc = c.leading_dimension();
        let stride_c = c.stride();
        let batch_count = c.batch_count();
        level3::gemm_strided_batched_ex(
            context,
            config.transpose_a(),
            config.transpose_b(),
            config.m(),
            config.n(),
            config.k(),
            alpha,
            a.memory(),
            DataType::F32,
            a.leading_dimension(),
            a.stride(),
            b.memory(),
            DataType::F32,
            b.leading_dimension(),
            b.stride(),
            beta,
            c.memory(),
            DataType::F32,
            ldc,
            stride_c,
            batch_count,
            ComputeType::F32,
            GemmAlgorithm::Default,
        )
    }

    fn batched_gemm(
        context: &Context,
        config: GemmConfig,
        alpha: &Self,
        a: &[DeviceMatrix<'_, Self>],
        b: &[DeviceMatrix<'_, Self>],
        beta: &Self,
        c: &mut [DeviceMatrixMut<'_, Self>],
    ) -> Result<()> {
        batched_gemm_ex(
            context,
            config,
            alpha,
            a,
            DataType::F32,
            b,
            DataType::F32,
            beta,
            c,
            DataType::F32,
            ComputeType::F32,
        )
    }
}

impl GemmElement for f64 {
    unsafe fn gemm(
        context: &Context,
        config: GemmConfig,
        alpha: Scalar<'_, Self>,
        a: DeviceMatrix<'_, Self>,
        b: DeviceMatrix<'_, Self>,
        beta: Scalar<'_, Self>,
        mut c: DeviceMatrixMut<'_, Self>,
    ) -> Result<()> {
        let ldc = c.leading_dimension();
        level3::dgemm(
            context,
            config.transpose_a(),
            config.transpose_b(),
            config.m(),
            config.n(),
            config.k(),
            alpha,
            a.memory(),
            a.leading_dimension(),
            b.memory(),
            b.leading_dimension(),
            beta,
            c.memory(),
            ldc,
        )
    }

    fn strided_batched_gemm(
        context: &Context,
        config: GemmConfig,
        alpha: &Self,
        a: StridedDeviceMatrices<'_, Self>,
        b: StridedDeviceMatrices<'_, Self>,
        beta: &Self,
        mut c: StridedDeviceMatricesMut<'_, Self>,
    ) -> Result<()> {
        let ldc = c.leading_dimension();
        let stride_c = c.stride();
        let batch_count = c.batch_count();
        level3::gemm_strided_batched_ex(
            context,
            config.transpose_a(),
            config.transpose_b(),
            config.m(),
            config.n(),
            config.k(),
            alpha,
            a.memory(),
            DataType::F64,
            a.leading_dimension(),
            a.stride(),
            b.memory(),
            DataType::F64,
            b.leading_dimension(),
            b.stride(),
            beta,
            c.memory(),
            DataType::F64,
            ldc,
            stride_c,
            batch_count,
            ComputeType::F64,
            GemmAlgorithm::Default,
        )
    }

    fn batched_gemm(
        context: &Context,
        config: GemmConfig,
        alpha: &Self,
        a: &[DeviceMatrix<'_, Self>],
        b: &[DeviceMatrix<'_, Self>],
        beta: &Self,
        c: &mut [DeviceMatrixMut<'_, Self>],
    ) -> Result<()> {
        batched_gemm_ex(
            context,
            config,
            alpha,
            a,
            DataType::F64,
            b,
            DataType::F64,
            beta,
            c,
            DataType::F64,
            ComputeType::F64,
        )
    }
}

/// Submits GEMM using borrowed matrices.
///
/// # Safety
///
/// Matrix storage and device scalars must remain live and correctly ordered until
/// the context's stream completes. Output writes must not conflict with other access.
///
/// # Errors
///
/// Returns an error if scalar pointer modes differ, matrix storage or leading dimensions are invalid, or dimensions exceed the native integer range.
/// CUDA context binding, pointer-mode changes, and cuBLAS execution errors are propagated.
pub unsafe fn gemm<'alpha, 'beta, T>(
    context: &Context,
    config: GemmConfig,
    alpha: impl Into<Scalar<'alpha, T>>,
    a: DeviceMatrix<'_, T>,
    b: DeviceMatrix<'_, T>,
    beta: impl Into<Scalar<'beta, T>>,
    c: DeviceMatrixMut<'_, T>,
) -> Result<()>
where
    T: GemmElement + 'static,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::gemm(context, config, alpha.into(), a, b, beta.into(), c) }
}

/// Multiplies batches of matrices separated by fixed element strides.
///
/// Uses `c.batch_count()` as the number of products.
///
/// # Errors
///
/// Returns an error if storage cannot hold the requested batch, strides or leading dimensions are invalid, or dimensions exceed the native integer range.
/// Also returns an error if the handle does not use host pointer mode, CUDA context binding fails, or cuBLAS rejects the operation.
pub fn strided_batched_gemm<T>(
    context: &Context,
    config: GemmConfig,
    alpha: &T,
    a: StridedDeviceMatrices<'_, T>,
    b: StridedDeviceMatrices<'_, T>,
    beta: &T,
    c: StridedDeviceMatricesMut<'_, T>,
) -> Result<()>
where
    T: GemmElement + 'static,
{
    T::strided_batched_gemm(context, config, alpha, a, b, beta, c)
}

/// Multiplies corresponding matrices from three batches.
///
/// Each batch must use a uniform leading dimension.
///
/// # Errors
///
/// Returns an error if batch lengths differ, leading dimensions are invalid or nonuniform within a matrix batch, or dimensions exceed the native integer range.
/// Also returns an error if the handle does not use host pointer mode, allocating or copying the device pointer arrays fails, CUDA context binding fails, or cuBLAS rejects the operation.
pub fn batched_gemm<T>(
    context: &Context,
    config: GemmConfig,
    alpha: &T,
    a: &[DeviceMatrix<'_, T>],
    b: &[DeviceMatrix<'_, T>],
    beta: &T,
    c: &mut [DeviceMatrixMut<'_, T>],
) -> Result<()>
where
    T: GemmElement + 'static,
{
    T::batched_gemm(context, config, alpha, a, b, beta, c)
}

fn batched_gemm_ex<T>(
    context: &Context,
    config: GemmConfig,
    alpha: &T,
    a: &[DeviceMatrix<'_, T>],
    a_type: DataType,
    b: &[DeviceMatrix<'_, T>],
    b_type: DataType,
    beta: &T,
    c: &mut [DeviceMatrixMut<'_, T>],
    c_type: DataType,
    compute_type: ComputeType,
) -> Result<()> {
    if a.is_empty() && b.is_empty() && c.is_empty() {
        return Ok(());
    }

    let Some(lda) = uniform_leading_dimension(a.iter().map(DeviceMatrix::leading_dimension)) else {
        return Err(Error::InvalidMatrixShape);
    };
    let Some(ldb) = uniform_leading_dimension(b.iter().map(DeviceMatrix::leading_dimension)) else {
        return Err(Error::InvalidMatrixShape);
    };
    let Some(ldc) = uniform_leading_dimension(c.iter().map(DeviceMatrixMut::leading_dimension))
    else {
        return Err(Error::InvalidMatrixShape);
    };

    let a = a
        .iter()
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        .map(|matrix| unsafe { DevicePtr::from_raw(matrix.memory().as_ptr().cast_mut().cast()) })
        .collect::<Vec<_>>();
    let b = b
        .iter()
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        .map(|matrix| unsafe { DevicePtr::from_raw(matrix.memory().as_ptr().cast_mut().cast()) })
        .collect::<Vec<_>>();
    let mut c = c
        .iter_mut()
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        .map(|matrix| unsafe { DevicePtr::from_raw(matrix.memory().as_mut_ptr().cast()) })
        .collect::<Vec<_>>();

    level3::gemm_batched_ex(
        context,
        config.transpose_a(),
        config.transpose_b(),
        config.m(),
        config.n(),
        config.k(),
        alpha,
        &a,
        a_type,
        lda,
        &b,
        b_type,
        ldb,
        beta,
        &mut c,
        c_type,
        ldc,
        compute_type,
        GemmAlgorithm::Default,
    )
}

fn uniform_leading_dimension(mut values: impl Iterator<Item = usize>) -> Option<usize> {
    let first = values.next()?;
    values.all(|value| value == first).then_some(first)
}

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}

    impl Sealed for f64 {}
}
