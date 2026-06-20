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
    pub const fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            transpose_a: Operation::NonTranspose,
            transpose_b: Operation::NonTranspose,
        }
    }

    pub fn with_transpose_a(mut self, transpose: Operation) -> Self {
        self.transpose_a = transpose;
        self
    }

    pub fn with_transpose_b(mut self, transpose: Operation) -> Self {
        self.transpose_b = transpose;
        self
    }

    pub const fn m(self) -> usize {
        self.m
    }

    pub const fn n(self) -> usize {
        self.n
    }

    pub const fn k(self) -> usize {
        self.k
    }

    pub const fn transpose_a(self) -> Operation {
        self.transpose_a
    }

    pub const fn transpose_b(self) -> Operation {
        self.transpose_b
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceMatrix<'a, T> {
    memory: &'a DeviceMemory<T>,
    leading_dimension: usize,
}

impl<'a, T> DeviceMatrix<'a, T> {
    pub const fn new(memory: &'a DeviceMemory<T>, leading_dimension: usize) -> Self {
        Self {
            memory,
            leading_dimension,
        }
    }

    pub const fn memory(&self) -> &'a DeviceMemory<T> {
        self.memory
    }

    pub const fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }
}

#[derive(Debug)]
pub struct DeviceMatrixMut<'a, T> {
    memory: &'a mut DeviceMemory<T>,
    leading_dimension: usize,
}

impl<'a, T> DeviceMatrixMut<'a, T> {
    pub const fn new(memory: &'a mut DeviceMemory<T>, leading_dimension: usize) -> Self {
        Self {
            memory,
            leading_dimension,
        }
    }

    pub fn memory(&mut self) -> &mut DeviceMemory<T> {
        self.memory
    }

    pub const fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StridedDeviceMatrices<'a, T> {
    memory: &'a DeviceMemory<T>,
    leading_dimension: usize,
    stride: i64,
    batch_count: usize,
}

impl<'a, T> StridedDeviceMatrices<'a, T> {
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

    pub const fn memory(&self) -> &'a DeviceMemory<T> {
        self.memory
    }

    pub const fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }

    pub const fn stride(&self) -> i64 {
        self.stride
    }

    pub const fn batch_count(&self) -> usize {
        self.batch_count
    }
}

#[derive(Debug)]
pub struct StridedDeviceMatricesMut<'a, T> {
    memory: &'a mut DeviceMemory<T>,
    leading_dimension: usize,
    stride: i64,
    batch_count: usize,
}

impl<'a, T> StridedDeviceMatricesMut<'a, T> {
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

    pub fn memory(&mut self) -> &mut DeviceMemory<T> {
        self.memory
    }

    pub const fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }

    pub const fn stride(&self) -> i64 {
        self.stride
    }

    pub const fn batch_count(&self) -> usize {
        self.batch_count
    }
}

pub trait GemmElement: private::Sealed {
    fn gemm(
        ctx: &Context,
        config: GemmConfig,
        alpha: Scalar<'_, Self>,
        a: DeviceMatrix<'_, Self>,
        b: DeviceMatrix<'_, Self>,
        beta: Scalar<'_, Self>,
        c: DeviceMatrixMut<'_, Self>,
    ) -> Result<()>
    where
        Self: Sized;

    fn strided_batched_gemm(
        ctx: &Context,
        config: GemmConfig,
        alpha: &Self,
        a: StridedDeviceMatrices<'_, Self>,
        b: StridedDeviceMatrices<'_, Self>,
        beta: &Self,
        c: StridedDeviceMatricesMut<'_, Self>,
    ) -> Result<()>
    where
        Self: Sized;

    fn batched_gemm(
        ctx: &Context,
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
    fn gemm(
        ctx: &Context,
        config: GemmConfig,
        alpha: Scalar<'_, Self>,
        a: DeviceMatrix<'_, Self>,
        b: DeviceMatrix<'_, Self>,
        beta: Scalar<'_, Self>,
        mut c: DeviceMatrixMut<'_, Self>,
    ) -> Result<()> {
        let ldc = c.leading_dimension();
        level3::sgemm(
            ctx,
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
        ctx: &Context,
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
            ctx,
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
        ctx: &Context,
        config: GemmConfig,
        alpha: &Self,
        a: &[DeviceMatrix<'_, Self>],
        b: &[DeviceMatrix<'_, Self>],
        beta: &Self,
        c: &mut [DeviceMatrixMut<'_, Self>],
    ) -> Result<()> {
        batched_gemm_ex(
            ctx,
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
    fn gemm(
        ctx: &Context,
        config: GemmConfig,
        alpha: Scalar<'_, Self>,
        a: DeviceMatrix<'_, Self>,
        b: DeviceMatrix<'_, Self>,
        beta: Scalar<'_, Self>,
        mut c: DeviceMatrixMut<'_, Self>,
    ) -> Result<()> {
        let ldc = c.leading_dimension();
        level3::dgemm(
            ctx,
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
        ctx: &Context,
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
            ctx,
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
        ctx: &Context,
        config: GemmConfig,
        alpha: &Self,
        a: &[DeviceMatrix<'_, Self>],
        b: &[DeviceMatrix<'_, Self>],
        beta: &Self,
        c: &mut [DeviceMatrixMut<'_, Self>],
    ) -> Result<()> {
        batched_gemm_ex(
            ctx,
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

pub fn gemm<'alpha, 'beta, T>(
    ctx: &Context,
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
    T::gemm(ctx, config, alpha.into(), a, b, beta.into(), c)
}

pub fn strided_batched_gemm<T>(
    ctx: &Context,
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
    T::strided_batched_gemm(ctx, config, alpha, a, b, beta, c)
}

pub fn batched_gemm<T>(
    ctx: &Context,
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
    T::batched_gemm(ctx, config, alpha, a, b, beta, c)
}

fn batched_gemm_ex<T>(
    ctx: &Context,
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

    let Some(lda) = uniform_leading_dimension(a.iter().map(|matrix| matrix.leading_dimension()))
    else {
        return Err(Error::InvalidMatrixShape);
    };
    let Some(ldb) = uniform_leading_dimension(b.iter().map(|matrix| matrix.leading_dimension()))
    else {
        return Err(Error::InvalidMatrixShape);
    };
    let Some(ldc) = uniform_leading_dimension(c.iter().map(|matrix| matrix.leading_dimension()))
    else {
        return Err(Error::InvalidMatrixShape);
    };

    let a = a
        .iter()
        .map(|matrix| unsafe { DevicePtr::from_raw(matrix.memory().as_ptr().cast_mut().cast()) })
        .collect::<Vec<_>>();
    let b = b
        .iter()
        .map(|matrix| unsafe { DevicePtr::from_raw(matrix.memory().as_ptr().cast_mut().cast()) })
        .collect::<Vec<_>>();
    let mut c = c
        .iter_mut()
        .map(|matrix| unsafe { DevicePtr::from_raw(matrix.memory().as_mut_ptr().cast()) })
        .collect::<Vec<_>>();

    level3::gemm_batched_ex(
        ctx,
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
