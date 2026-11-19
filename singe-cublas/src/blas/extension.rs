use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    memory::DeviceMemory,
    stream::AsStream,
    types::{Complex32, DevicePtr},
    view::{DeviceInputView, DeviceOutput, DeviceOutputView, DeviceRange, DeviceRegionProvider},
};

use crate::{
    blas::{
        level3::{GemmGroup, GroupArrays, ensure_strided_gemm},
        shape::{ensure_matrix, ensure_vector, stored},
    },
    context::Context,
    error::{Error, Result},
    gemm::GemmShape,
    scalar::{Scalar, ScalarMut},
    sys, try_ffi,
    types::{ComputeType, FillMode, GemmAlgorithm, Operation, PointerMode},
    utility::{ensure_data_type, to_i64, validate_min_len, validate_same_len},
};

// Returns the type cuBLAS reads `alpha` and `beta` as, which the GEMM Ex type table derives from the compute type and whether `C` is complex.
fn scale_type(compute_type: ComputeType, c_type: DataType) -> DataType {
    let complex = matches!(
        c_type,
        DataType::ComplexF16
            | DataType::ComplexBf16
            | DataType::ComplexF32
            | DataType::ComplexF64
            | DataType::ComplexI4
            | DataType::ComplexU4
            | DataType::ComplexI8
            | DataType::ComplexU8
            | DataType::ComplexI16
            | DataType::ComplexU16
            | DataType::ComplexI32
            | DataType::ComplexU32
            | DataType::ComplexI64
            | DataType::ComplexU64
    );
    match compute_type {
        ComputeType::F16 | ComputeType::F16Pedantic => DataType::F16,
        ComputeType::I32 | ComputeType::I32Pedantic => DataType::I32,
        ComputeType::F32
        | ComputeType::F32Pedantic
        | ComputeType::F32FastF16
        | ComputeType::F32FastBf16
        | ComputeType::F32FastTf32
        | ComputeType::F32EmulatedBf16x9 => {
            if complex {
                DataType::ComplexF32
            } else {
                DataType::F32
            }
        }
        ComputeType::F64 | ComputeType::F64Pedantic | ComputeType::F64EmulatedFixedPoint => {
            if complex {
                DataType::ComplexF64
            } else {
                DataType::F64
            }
        }
    }
}

fn ensure_scale_type<TS: DataTypeLike>(compute_type: ComputeType, c_type: DataType) -> Result<()> {
    ensure_data_type(
        "alpha and beta",
        scale_type(compute_type, c_type),
        TS::data_type(),
    )
}

/// Enqueues the Euclidean norm of `x` into `result`, computing in `execution_type`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if `x` is too small, `incx` is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
pub unsafe fn nrm2_ex_async<TX: DataTypeLike, TR: DataTypeLike>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<TX> + ?Sized),
    incx: usize,
    mut result: ScalarMut<'_, TR>,
    execution_type: DataType,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    context.prepare_operands(&[x.region()], &[result.region()])?;
    context.use_pointer_mode(&[result.pointer_mode()])?;
    // SAFETY: the data types are those of `x` and `result`, and `x` holds the strided values in this context, ordered after its producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(sys::cublasNrm2Ex_64(
            context.as_raw(),
            to_i64(n, "n")?,
            x.as_device_ptr().cast(),
            TX::data_type().into(),
            to_i64(incx, "incx")?,
            result.as_mut_ptr().cast(),
            TR::data_type().into(),
            execution_type.into(),
        ))
    }
}

/// Computes the Euclidean norm of `x` into `result` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`nrm2_ex_async`] lists.
pub fn nrm2_ex<TX: DataTypeLike, TR: DataTypeLike>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<TX> + ?Sized),
    incx: usize,
    result: ScalarMut<'_, TR>,
    execution_type: DataType,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { nrm2_ex_async(context, n, x, incx, result, execution_type) })
}

/// Enqueues the sum of the absolute values of `x` into `result`, computing in `execution_type`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if `x` is too small, `incx` is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
pub unsafe fn asum_ex_async<TX: DataTypeLike, TR: DataTypeLike>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<TX> + ?Sized),
    incx: usize,
    mut result: ScalarMut<'_, TR>,
    execution_type: DataType,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    context.prepare_operands(&[x.region()], &[result.region()])?;
    context.use_pointer_mode(&[result.pointer_mode()])?;
    // SAFETY: the data types are those of `x` and `result`, and `x` holds the strided values in this context, ordered after its producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(sys::cublasAsumEx_64(
            context.as_raw(),
            to_i64(n, "n")?,
            x.as_device_ptr().cast(),
            TX::data_type().into(),
            to_i64(incx, "incx")?,
            result.as_mut_ptr().cast(),
            TR::data_type().into(),
            execution_type.into(),
        ))
    }
}

/// Computes the sum of the absolute values of `x` into `result` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`asum_ex_async`] lists.
pub fn asum_ex<TX: DataTypeLike, TR: DataTypeLike>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<TX> + ?Sized),
    incx: usize,
    result: ScalarMut<'_, TR>,
    execution_type: DataType,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { asum_ex_async(context, n, x, incx, result, execution_type) })
}

// Declares the dot products of independently typed vectors.
macro_rules! dot_ex {
    ($operation:ident, $name:ident, $native:ident, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
        pub unsafe fn $operation<TX: DataTypeLike, TY: DataTypeLike, TR: DataTypeLike>(
            context: &Context,
            n: usize,
            x: &(impl DeviceRange<TX> + ?Sized),
            incx: usize,
            y: &(impl DeviceRange<TY> + ?Sized),
            incy: usize,
            mut result: ScalarMut<'_, TR>,
            execution_type: DataType,
        ) -> Result<()> {
            context.bind()?;
            ensure_vector("x", x.len(), n, incx)?;
            ensure_vector("y", y.len(), n, incy)?;
            context.prepare_operands(&[x.region(), y.region()], &[result.region()])?;
            context.use_pointer_mode(&[result.pointer_mode()])?;
            // SAFETY: the data types are those of the operands, and the vectors hold the strided values in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(sys::$native(
                    context.as_raw(),
                    to_i64(n, "n")?,
                    x.as_device_ptr().cast(),
                    TX::data_type().into(),
                    to_i64(incx, "incx")?,
                    y.as_device_ptr().cast(),
                    TY::data_type().into(),
                    to_i64(incy, "incy")?,
                    result.as_mut_ptr().cast(),
                    TR::data_type().into(),
                    execution_type.into(),
                ))
            }
        }

        #[doc = $waiting]
        ///
        /// # Errors
        ///
        /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
        pub fn $name<TX: DataTypeLike, TY: DataTypeLike, TR: DataTypeLike>(
            context: &Context,
            n: usize,
            x: &(impl DeviceRange<TX> + ?Sized),
            incx: usize,
            y: &(impl DeviceRange<TY> + ?Sized),
            incy: usize,
            result: ScalarMut<'_, TR>,
            execution_type: DataType,
        ) -> Result<()> {
            // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
            context.sync_scope(|| unsafe {
                $operation(context, n, x, incx, y, incy, result, execution_type)
            })
        }
    };
}

dot_ex!(
    dot_ex_async,
    dot_ex,
    cublasDotEx_64,
    "Enqueues the unconjugated dot product of `x` and `y` into `result`, computing in `execution_type`.",
    "Computes the unconjugated dot product of `x` and `y` into `result` and waits for it."
);

dot_ex!(
    dotc_ex_async,
    dotc_ex,
    cublasDotcEx_64,
    "Enqueues the dot product of conjugated `x` and `y` into `result`, computing in `execution_type`.",
    "Computes the dot product of conjugated `x` and `y` into `result` and waits for it."
);

/// Enqueues `x = alpha * x`, computing in `execution_type`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if `x` is too small, `incx` is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
pub unsafe fn scal_ex_async<TA: DataTypeLike, TX: DataTypeLike>(
    context: &Context,
    n: usize,
    alpha: Scalar<'_, TA>,
    x: &mut (impl DeviceOutput<TX> + ?Sized),
    incx: usize,
    execution_type: DataType,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    context.prepare_operands(&[x.region()], &[alpha.region()])?;
    context.use_pointer_mode(&[alpha.pointer_mode()])?;
    // SAFETY: the data types are those of the operands, and `x` holds the strided values in this context, ordered after its producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(sys::cublasScalEx_64(
            context.as_raw(),
            to_i64(n, "n")?,
            alpha.ptr().cast(),
            TA::data_type().into(),
            x.as_mut_ptr().cast(),
            TX::data_type().into(),
            to_i64(incx, "incx")?,
            execution_type.into(),
        ))
    }
}

/// Computes `x = alpha * x` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`scal_ex_async`] lists.
pub fn scal_ex<TA: DataTypeLike, TX: DataTypeLike>(
    context: &Context,
    n: usize,
    alpha: Scalar<'_, TA>,
    x: &mut (impl DeviceOutput<TX> + ?Sized),
    incx: usize,
    execution_type: DataType,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { scal_ex_async(context, n, alpha, x, incx, execution_type) })
}

/// Enqueues `y = alpha * x + y`, computing in `execution_type`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
pub unsafe fn axpy_ex_async<TA: DataTypeLike, TX: DataTypeLike, TY: DataTypeLike>(
    context: &Context,
    n: usize,
    alpha: Scalar<'_, TA>,
    x: &(impl DeviceRange<TX> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<TY> + ?Sized),
    incy: usize,
    execution_type: DataType,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    ensure_vector("y", y.len(), n, incy)?;
    context.prepare_operands(&[x.region(), y.region()], &[alpha.region()])?;
    context.use_pointer_mode(&[alpha.pointer_mode()])?;
    // SAFETY: the data types are those of the operands, and the vectors hold the strided values in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(sys::cublasAxpyEx_64(
            context.as_raw(),
            to_i64(n, "n")?,
            alpha.ptr().cast(),
            TA::data_type().into(),
            x.as_device_ptr().cast(),
            TX::data_type().into(),
            to_i64(incx, "incx")?,
            y.as_mut_ptr().cast(),
            TY::data_type().into(),
            to_i64(incy, "incy")?,
            execution_type.into(),
        ))
    }
}

/// Computes `y = alpha * x + y` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`axpy_ex_async`] lists.
pub fn axpy_ex<TA: DataTypeLike, TX: DataTypeLike, TY: DataTypeLike>(
    context: &Context,
    n: usize,
    alpha: Scalar<'_, TA>,
    x: &(impl DeviceRange<TX> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<TY> + ?Sized),
    incy: usize,
    execution_type: DataType,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe {
        axpy_ex_async(context, n, alpha, x, incx, y, incy, execution_type)
    })
}

/// Enqueues `y = x` between independently typed vectors.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
pub unsafe fn copy_ex_async<TX: DataTypeLike, TY: DataTypeLike>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<TX> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<TY> + ?Sized),
    incy: usize,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    ensure_vector("y", y.len(), n, incy)?;
    context.prepare_operands(&[x.region(), y.region()], &[])?;
    // SAFETY: the data types are those of the operands, and the vectors hold the strided values in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(sys::cublasCopyEx_64(
            context.as_raw(),
            to_i64(n, "n")?,
            x.as_device_ptr().cast(),
            TX::data_type().into(),
            to_i64(incx, "incx")?,
            y.as_mut_ptr().cast(),
            TY::data_type().into(),
            to_i64(incy, "incy")?,
        ))
    }
}

/// Copies `x` into `y` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`copy_ex_async`] lists.
pub fn copy_ex<TX: DataTypeLike, TY: DataTypeLike>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<TX> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<TY> + ?Sized),
    incy: usize,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { copy_ex_async(context, n, x, incx, y, incy) })
}

/// Enqueues an exchange of the values of independently typed vectors.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
pub unsafe fn swap_ex_async<TX: DataTypeLike, TY: DataTypeLike>(
    context: &Context,
    n: usize,
    x: &mut (impl DeviceOutput<TX> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<TY> + ?Sized),
    incy: usize,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    ensure_vector("y", y.len(), n, incy)?;
    context.prepare_operands(&[x.region(), y.region()], &[])?;
    // SAFETY: the data types are those of the operands, and the vectors hold the strided values in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(sys::cublasSwapEx_64(
            context.as_raw(),
            to_i64(n, "n")?,
            x.as_mut_ptr().cast(),
            TX::data_type().into(),
            to_i64(incx, "incx")?,
            y.as_mut_ptr().cast(),
            TY::data_type().into(),
            to_i64(incy, "incy")?,
        ))
    }
}

/// Exchanges the values of `x` and `y` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`swap_ex_async`] lists.
pub fn swap_ex<TX: DataTypeLike, TY: DataTypeLike>(
    context: &Context,
    n: usize,
    x: &mut (impl DeviceOutput<TX> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<TY> + ?Sized),
    incy: usize,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { swap_ex_async(context, n, x, incx, y, incy) })
}

// Declares the index searches over a typed vector.
macro_rules! index_ex {
    ($operation:ident, $name:ident, $native:ident, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// For complex values, the magnitude is the sum of the absolute real and imaginary parts.
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if `x` is too small, `incx` is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the type or the call.
        pub unsafe fn $operation<TX: DataTypeLike>(
            context: &Context,
            n: usize,
            x: &(impl DeviceRange<TX> + ?Sized),
            incx: usize,
            mut result: ScalarMut<'_, i64>,
        ) -> Result<()> {
            context.bind()?;
            ensure_vector("x", x.len(), n, incx)?;
            context.prepare_operands(&[x.region()], &[result.region()])?;
            context.use_pointer_mode(&[result.pointer_mode()])?;
            // SAFETY: the data type is that of `x`, which holds the strided values in this context, ordered after its producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(sys::$native(
                    context.as_raw(),
                    to_i64(n, "n")?,
                    x.as_device_ptr().cast(),
                    TX::data_type().into(),
                    to_i64(incx, "incx")?,
                    result.as_mut_ptr(),
                ))
            }
        }

        #[doc = $waiting]
        ///
        /// # Errors
        ///
        /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
        pub fn $name<TX: DataTypeLike>(
            context: &Context,
            n: usize,
            x: &(impl DeviceRange<TX> + ?Sized),
            incx: usize,
            result: ScalarMut<'_, i64>,
        ) -> Result<()> {
            // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
            context.sync_scope(|| unsafe { $operation(context, n, x, incx, result) })
        }
    };
}

index_ex!(
    iamax_ex_async,
    iamax_ex,
    cublasIamaxEx_64,
    "Enqueues the one-based index of the first element of `x` with the largest magnitude into `result`.",
    "Computes the one-based index of the first element of `x` with the largest magnitude and waits for it."
);

index_ex!(
    iamin_ex_async,
    iamin_ex,
    cublasIaminEx_64,
    "Enqueues the one-based index of the first element of `x` with the smallest magnitude into `result`.",
    "Computes the one-based index of the first element of `x` with the smallest magnitude and waits for it."
);

/// Enqueues the Givens rotation of independently typed vectors by `c` and `s`, computing in `execution_type`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
pub unsafe fn rot_ex_async<TX: DataTypeLike, TY: DataTypeLike, TCS: DataTypeLike>(
    context: &Context,
    n: usize,
    x: &mut (impl DeviceOutput<TX> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<TY> + ?Sized),
    incy: usize,
    c: Scalar<'_, TCS>,
    s: Scalar<'_, TCS>,
    execution_type: DataType,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    ensure_vector("y", y.len(), n, incy)?;
    context.prepare_operands(&[x.region(), y.region()], &[c.region(), s.region()])?;
    context.use_pointer_mode(&[c.pointer_mode(), s.pointer_mode()])?;
    // SAFETY: the data types are those of the operands, and the vectors hold the strided values in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(sys::cublasRotEx_64(
            context.as_raw(),
            to_i64(n, "n")?,
            x.as_mut_ptr().cast(),
            TX::data_type().into(),
            to_i64(incx, "incx")?,
            y.as_mut_ptr().cast(),
            TY::data_type().into(),
            to_i64(incy, "incy")?,
            c.ptr().cast(),
            s.ptr().cast(),
            TCS::data_type().into(),
            execution_type.into(),
        ))
    }
}

/// Applies the Givens rotation described by `c` and `s` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`rot_ex_async`] lists.
pub fn rot_ex<TX: DataTypeLike, TY: DataTypeLike, TCS: DataTypeLike>(
    context: &Context,
    n: usize,
    x: &mut (impl DeviceOutput<TX> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<TY> + ?Sized),
    incy: usize,
    c: Scalar<'_, TCS>,
    s: Scalar<'_, TCS>,
    execution_type: DataType,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context
        .sync_scope(|| unsafe { rot_ex_async(context, n, x, incx, y, incy, c, s, execution_type) })
}

/// Enqueues `C = alpha * op(A) * op(B) + beta * C` with the matrix types of `TA`, `TB`, and `TC` and scalars of the type `compute_type` selects.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if `TS` is not the scale type of `compute_type` and `TC`, a matrix is too small or a leading dimension is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
pub unsafe fn gemm_ex_async<TA, TB, TC, TS>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar<'_, TS>,
    a: &(impl DeviceRange<TA> + ?Sized),
    lda: usize,
    b: &(impl DeviceRange<TB> + ?Sized),
    ldb: usize,
    beta: Scalar<'_, TS>,
    c: &mut (impl DeviceOutput<TC> + ?Sized),
    ldc: usize,
    compute_type: ComputeType,
    algorithm: GemmAlgorithm,
) -> Result<()>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
    TC: DataTypeLike,
    TS: DataTypeLike,
{
    context.bind()?;
    ensure_scale_type::<TS>(compute_type, TC::data_type())?;
    let (a_rows, a_cols) = stored(transpose_a, m, k);
    let (b_rows, b_cols) = stored(transpose_b, k, n);
    ensure_matrix("a", a.len(), a_rows, a_cols, lda)?;
    ensure_matrix("b", b.len(), b_rows, b_cols, ldb)?;
    ensure_matrix("c", c.len(), m, n, ldc)?;
    context.prepare_operands(
        &[a.region(), b.region(), c.region()],
        &[alpha.region(), beta.region()],
    )?;
    context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
    // SAFETY: the data types are those of the operands, the matrices hold the checked shapes in this context, ordered after their producers, and the scalars have the scale type; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(sys::cublasGemmEx_64(
            context.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            alpha.ptr().cast(),
            a.as_device_ptr().cast(),
            TA::data_type().into(),
            to_i64(lda, "lda")?,
            b.as_device_ptr().cast(),
            TB::data_type().into(),
            to_i64(ldb, "ldb")?,
            beta.ptr().cast(),
            c.as_mut_ptr().cast(),
            TC::data_type().into(),
            to_i64(ldc, "ldc")?,
            compute_type.into(),
            algorithm.into(),
        ))
    }
}

/// Computes `C = alpha * op(A) * op(B) + beta * C` with independently typed matrices and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`gemm_ex_async`] lists.
pub fn gemm_ex<TA, TB, TC, TS>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar<'_, TS>,
    a: &(impl DeviceRange<TA> + ?Sized),
    lda: usize,
    b: &(impl DeviceRange<TB> + ?Sized),
    ldb: usize,
    beta: Scalar<'_, TS>,
    c: &mut (impl DeviceOutput<TC> + ?Sized),
    ldc: usize,
    compute_type: ComputeType,
    algorithm: GemmAlgorithm,
) -> Result<()>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
    TC: DataTypeLike,
    TS: DataTypeLike,
{
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe {
        gemm_ex_async(
            context,
            transpose_a,
            transpose_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
            compute_type,
            algorithm,
        )
    })
}

/// Enqueues `batch_count` products of independently typed matrices taken `stride_*` elements apart.
///
/// A zero input stride repeats one matrix for every product; output matrices may not overlap.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if `TS` is not the scale type of `compute_type` and `TC`, a buffer cannot hold its batch, a leading dimension or stride is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
pub unsafe fn gemm_strided_batched_ex_async<TA, TB, TC, TS>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar<'_, TS>,
    a: &(impl DeviceRange<TA> + ?Sized),
    lda: usize,
    stride_a: i64,
    b: &(impl DeviceRange<TB> + ?Sized),
    ldb: usize,
    stride_b: i64,
    beta: Scalar<'_, TS>,
    c: &mut (impl DeviceOutput<TC> + ?Sized),
    ldc: usize,
    stride_c: i64,
    batch_count: usize,
    compute_type: ComputeType,
    algorithm: GemmAlgorithm,
) -> Result<()>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
    TC: DataTypeLike,
    TS: DataTypeLike,
{
    context.bind()?;
    ensure_scale_type::<TS>(compute_type, TC::data_type())?;
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
    // SAFETY: the data types are those of the operands, the buffers hold the checked batches in this context, ordered after their producers, and the scalars have the scale type; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(sys::cublasGemmStridedBatchedEx_64(
            context.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            alpha.ptr().cast(),
            a.as_device_ptr().cast(),
            TA::data_type().into(),
            to_i64(lda, "lda")?,
            stride_a,
            b.as_device_ptr().cast(),
            TB::data_type().into(),
            to_i64(ldb, "ldb")?,
            stride_b,
            beta.ptr().cast(),
            c.as_mut_ptr().cast(),
            TC::data_type().into(),
            to_i64(ldc, "ldc")?,
            stride_c,
            to_i64(batch_count, "batch count")?,
            compute_type.into(),
            algorithm.into(),
        ))
    }
}

/// Computes a strided batch of independently typed matrix products and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`gemm_strided_batched_ex_async`] lists.
pub fn gemm_strided_batched_ex<TA, TB, TC, TS>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar<'_, TS>,
    a: &(impl DeviceRange<TA> + ?Sized),
    lda: usize,
    stride_a: i64,
    b: &(impl DeviceRange<TB> + ?Sized),
    ldb: usize,
    stride_b: i64,
    beta: Scalar<'_, TS>,
    c: &mut (impl DeviceOutput<TC> + ?Sized),
    ldc: usize,
    stride_c: i64,
    batch_count: usize,
    compute_type: ComputeType,
    algorithm: GemmAlgorithm,
) -> Result<()>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
    TC: DataTypeLike,
    TS: DataTypeLike,
{
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe {
        gemm_strided_batched_ex_async(
            context,
            transpose_a,
            transpose_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            stride_a,
            b,
            ldb,
            stride_b,
            beta,
            c,
            ldc,
            stride_c,
            batch_count,
            compute_type,
            algorithm,
        )
    })
}

/// Enqueues `batch_count` products of matrices of the given data types listed in device arrays of pointers.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution) for the pointer arrays and for every matrix they address.
/// Each pointer must address a matrix of its data type in this context with the shape the dimensions and leading dimensions describe, and the `c` matrices must not overlap each other or the inputs.
///
/// # Errors
///
/// Returns an error if `TS` is not the scale type of `compute_type` and `c_type`, a pointer array holds fewer than `batch_count` entries, a leading dimension is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
pub unsafe fn gemm_batched_ex_async<TS: DataTypeLike>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar<'_, TS>,
    a: &(impl DeviceRange<DevicePtr> + ?Sized),
    a_type: DataType,
    lda: usize,
    b: &(impl DeviceRange<DevicePtr> + ?Sized),
    b_type: DataType,
    ldb: usize,
    beta: Scalar<'_, TS>,
    c: &(impl DeviceRange<DevicePtr> + ?Sized),
    c_type: DataType,
    ldc: usize,
    batch_count: usize,
    compute_type: ComputeType,
    algorithm: GemmAlgorithm,
) -> Result<()> {
    context.bind()?;
    ensure_scale_type::<TS>(compute_type, c_type)?;
    let (a_rows, _) = stored(transpose_a, m, k);
    let (b_rows, _) = stored(transpose_b, k, n);
    if lda < a_rows.max(1) || ldb < b_rows.max(1) || ldc < m.max(1) {
        return Err(Error::InvalidLeadingDimension);
    }

    for (name, len) in [("a", a.len()), ("b", b.len()), ("c", c.len())] {
        validate_min_len(batch_count, len, name)?;
    }

    context.prepare_operands(
        &[a.region(), b.region(), c.region()],
        &[alpha.region(), beta.region()],
    )?;
    context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
    // SAFETY: the pointer arrays hold `batch_count` entries in this context, ordered after their producers, and the scalars have the scale type; the caller upholds the matrix contract.
    unsafe {
        try_ffi!(sys::cublasGemmBatchedEx_64(
            context.as_raw(),
            transpose_a.into(),
            transpose_b.into(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            alpha.ptr().cast(),
            a.as_device_ptr().cast(),
            a_type.into(),
            to_i64(lda, "lda")?,
            b.as_device_ptr().cast(),
            b_type.into(),
            to_i64(ldb, "ldb")?,
            beta.ptr().cast(),
            c.as_device_ptr().cast(),
            c_type.into(),
            to_i64(ldc, "ldc")?,
            to_i64(batch_count, "batch count")?,
            compute_type.into(),
            algorithm.into(),
        ))
    }
}

/// Computes the products of corresponding independently typed matrices and waits for them.
///
/// It copies the matrix addresses to device pointer arrays, which live until the work completes.
///
/// # Errors
///
/// Returns an error if the batches have different lengths, a matrix is too small or a leading dimension is invalid, the handle's stream is capturing, copying the pointer arrays or waiting fails, or for the reasons [`gemm_batched_ex_async`] lists.
pub fn gemm_batched_ex<TA, TB, TC, TS>(
    context: &Context,
    transpose_a: Operation,
    transpose_b: Operation,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar<'_, TS>,
    a: &[DeviceInputView<'_, TA>],
    lda: usize,
    b: &[DeviceInputView<'_, TB>],
    ldb: usize,
    beta: Scalar<'_, TS>,
    c: &mut [DeviceOutputView<'_, TC>],
    ldc: usize,
    compute_type: ComputeType,
    algorithm: GemmAlgorithm,
) -> Result<()>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
    TC: DataTypeLike,
    TS: DataTypeLike,
{
    validate_same_len(a.len(), b.len(), "b")?;
    validate_same_len(a.len(), c.len(), "c")?;
    if a.is_empty() {
        return Ok(());
    }

    let (a_rows, a_cols) = stored(transpose_a, m, k);
    let (b_rows, b_cols) = stored(transpose_b, k, n);
    for ((a, b), c) in a.iter().zip(b).zip(c.iter()) {
        ensure_matrix("a", a.len(), a_rows, a_cols, lda)?;
        ensure_matrix("b", b.len(), b_rows, b_cols, ldb)?;
        ensure_matrix("c", c.len(), m, n, ldc)?;
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
    let a_pointers = device_pointers(context, a.iter().map(|view| view.as_ptr().cast()))?;
    let b_pointers = device_pointers(context, b.iter().map(|view| view.as_ptr().cast()))?;
    let c_pointers = device_pointers(
        context,
        c.iter_mut()
            .map(|view| view.as_mut_ptr().cast_const().cast()),
    )?;
    // SAFETY: the pointers address the checked matrices of the declared types, whose borrows and pointer arrays outlive the scope, which waits for the handle's stream; the `c` views are exclusive borrows, so they overlap neither each other nor the inputs.
    context.sync_scope(|| unsafe {
        gemm_batched_ex_async(
            context,
            transpose_a,
            transpose_b,
            m,
            n,
            k,
            alpha,
            &a_pointers,
            TA::data_type(),
            lda,
            &b_pointers,
            TB::data_type(),
            ldb,
            beta,
            &c_pointers,
            TC::data_type(),
            ldc,
            c_pointers.len(),
            compute_type,
            algorithm,
        )
    })
}

/// Enqueues the products of every group, whose matrices of the given data types are listed group by group in device arrays of pointers.
///
/// The group scalars are read from host memory.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution) for the pointer arrays and for every matrix they address.
/// Each pointer must address a matrix of its data type in this context with the shape of its group, and the `c` matrices must not overlap each other or the inputs.
///
/// # Errors
///
/// Returns an error if `TS` is not the scale type of `compute_type` and `c_type`, `groups` is empty, a pointer array holds fewer entries than the groups' total count, a leading dimension is invalid, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
pub unsafe fn gemm_grouped_batched_ex_async<TS: DataTypeLike>(
    context: &Context,
    groups: &[GemmGroup<TS>],
    a: &(impl DeviceRange<DevicePtr> + ?Sized),
    a_type: DataType,
    b: &(impl DeviceRange<DevicePtr> + ?Sized),
    b_type: DataType,
    c: &(impl DeviceRange<DevicePtr> + ?Sized),
    c_type: DataType,
    compute_type: ComputeType,
) -> Result<()> {
    context.bind()?;
    ensure_scale_type::<TS>(compute_type, c_type)?;
    let arrays = GroupArrays::create(groups)?;
    for (name, len) in [("a", a.len()), ("b", b.len()), ("c", c.len())] {
        validate_min_len(arrays.total, len, name)?;
    }

    context.prepare_operands(&[a.region(), b.region(), c.region()], &[])?;
    context.use_pointer_mode(&[PointerMode::Host])?;
    // SAFETY: the per-group host arrays hold one entry per group with the scale type, and the pointer arrays hold the total count in this context, ordered after their producers; the caller upholds the matrix contract.
    unsafe {
        try_ffi!(sys::cublasGemmGroupedBatchedEx_64(
            context.as_raw(),
            arrays.transpose_a.as_ptr(),
            arrays.transpose_b.as_ptr(),
            arrays.m.as_ptr(),
            arrays.n.as_ptr(),
            arrays.k.as_ptr(),
            arrays.alpha.as_ptr().cast(),
            a.as_device_ptr().cast(),
            a_type.into(),
            arrays.lda.as_ptr(),
            b.as_device_ptr().cast(),
            b_type.into(),
            arrays.ldb.as_ptr(),
            arrays.beta.as_ptr().cast(),
            c.as_device_ptr().cast(),
            c_type.into(),
            arrays.ldc.as_ptr(),
            to_i64(groups.len(), "group count")?,
            arrays.count.as_ptr(),
            compute_type.into(),
        ))
    }
}

/// Computes the products of every group of independently typed matrices and waits for them.
///
/// `a`, `b`, and `c` list the matrices group by group.
/// It copies the matrix addresses to device pointer arrays, which live until the work completes.
///
/// # Errors
///
/// Returns an error if the matrix lists do not hold the groups' total count, a matrix is too small for its group, the handle's stream is capturing, copying the pointer arrays or waiting fails, or for the reasons [`gemm_grouped_batched_ex_async`] lists.
pub fn gemm_grouped_batched_ex<TA, TB, TC, TS>(
    context: &Context,
    groups: &[GemmGroup<TS>],
    a: &[DeviceInputView<'_, TA>],
    b: &[DeviceInputView<'_, TB>],
    c: &mut [DeviceOutputView<'_, TC>],
    compute_type: ComputeType,
) -> Result<()>
where
    TA: DataTypeLike,
    TB: DataTypeLike,
    TC: DataTypeLike,
    TS: DataTypeLike,
{
    let total = GroupArrays::create(groups)?.total;
    validate_same_len(total, a.len(), "a")?;
    validate_same_len(total, b.len(), "b")?;
    validate_same_len(total, c.len(), "c")?;
    if total == 0 {
        return Ok(());
    }

    let mut index = 0;
    for group in groups {
        let shape = group.shape;
        let (a_rows, a_cols) = stored(shape.transpose_a(), shape.m(), shape.k());
        let (b_rows, b_cols) = stored(shape.transpose_b(), shape.k(), shape.n());
        for _ in 0..group.count {
            ensure_matrix("a", a[index].len(), a_rows, a_cols, group.lda)?;
            ensure_matrix("b", b[index].len(), b_rows, b_cols, group.ldb)?;
            ensure_matrix("c", c[index].len(), shape.m(), shape.n(), group.ldc)?;
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
    let a_pointers = device_pointers(context, a.iter().map(|view| view.as_ptr().cast()))?;
    let b_pointers = device_pointers(context, b.iter().map(|view| view.as_ptr().cast()))?;
    let c_pointers = device_pointers(
        context,
        c.iter_mut()
            .map(|view| view.as_mut_ptr().cast_const().cast()),
    )?;
    // SAFETY: the pointers address the checked matrices of the declared types, whose borrows and pointer arrays outlive the scope, which waits for the handle's stream; the `c` views are exclusive borrows, so they overlap neither each other nor the inputs.
    context.sync_scope(|| unsafe {
        gemm_grouped_batched_ex_async(
            context,
            groups,
            &a_pointers,
            TA::data_type(),
            &b_pointers,
            TB::data_type(),
            &c_pointers,
            TC::data_type(),
            compute_type,
        )
    })
}

// Copies device addresses into a device array that cuBLAS can read.
fn device_pointers(
    context: &Context,
    pointers: impl Iterator<Item = *const ()>,
) -> Result<DeviceMemory<DevicePtr>> {
    let pointers = pointers
        // SAFETY: wrapping the address of a borrowed allocation does not dereference it.
        .map(|pointer| unsafe { DevicePtr::from_raw(pointer.cast_mut()) })
        .collect::<Vec<_>>();
    Ok(DeviceMemory::from_slice(context.cuda_context(), &pointers)?)
}

// Declares the complex single-precision rank-k updates with independently typed matrices.
macro_rules! rank_k_ex {
    ($operation:ident, $name:ident, $native:ident, $scalar:ty, $summary:literal, $waiting:literal) => {
        #[doc = $summary]
        ///
        /// # Safety
        ///
        /// The caller upholds the [raw execution contract](crate#raw-execution).
        ///
        /// # Errors
        ///
        /// Returns an error if a matrix is too small or a leading dimension is invalid, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the type combination or the call.
        pub unsafe fn $operation<TA: DataTypeLike, TC: DataTypeLike>(
            context: &Context,
            fill_mode: FillMode,
            trans: Operation,
            n: usize,
            k: usize,
            alpha: Scalar<'_, $scalar>,
            a: &(impl DeviceRange<TA> + ?Sized),
            lda: usize,
            beta: Scalar<'_, $scalar>,
            c: &mut (impl DeviceOutput<TC> + ?Sized),
            ldc: usize,
        ) -> Result<()> {
            context.bind()?;
            let (a_rows, a_cols) = stored(trans, n, k);
            ensure_matrix("a", a.len(), a_rows, a_cols, lda)?;
            ensure_matrix("c", c.len(), n, n, ldc)?;
            context
                .prepare_operands(&[a.region(), c.region()], &[alpha.region(), beta.region()])?;
            context.use_pointer_mode(&[alpha.pointer_mode(), beta.pointer_mode()])?;
            // SAFETY: the data types are those of the operands, and the matrices hold the checked shapes in this context, ordered after their producers; the caller upholds the raw execution contract.
            unsafe {
                try_ffi!(sys::$native(
                    context.as_raw(),
                    fill_mode.into(),
                    trans.into(),
                    to_i64(n, "n")?,
                    to_i64(k, "k")?,
                    alpha.ptr().cast(),
                    a.as_device_ptr().cast(),
                    TA::data_type().into(),
                    to_i64(lda, "lda")?,
                    beta.ptr().cast(),
                    c.as_mut_ptr().cast(),
                    TC::data_type().into(),
                    to_i64(ldc, "ldc")?,
                ))
            }
        }

        #[doc = $waiting]
        ///
        /// # Errors
        ///
        /// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons the `_async` form lists.
        pub fn $name<TA: DataTypeLike, TC: DataTypeLike>(
            context: &Context,
            fill_mode: FillMode,
            trans: Operation,
            n: usize,
            k: usize,
            alpha: Scalar<'_, $scalar>,
            a: &(impl DeviceRange<TA> + ?Sized),
            lda: usize,
            beta: Scalar<'_, $scalar>,
            c: &mut (impl DeviceOutput<TC> + ?Sized),
            ldc: usize,
        ) -> Result<()> {
            // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
            context.sync_scope(|| unsafe {
                $operation(context, fill_mode, trans, n, k, alpha, a, lda, beta, c, ldc)
            })
        }
    };
}

rank_k_ex!(
    csyrk_ex_async,
    csyrk_ex,
    cublasCsyrkEx_64,
    Complex32,
    "Enqueues the complex symmetric rank-k update `C = alpha * op(A) * op(A)^T + beta * C` with independently typed matrices.",
    "Computes the complex symmetric rank-k update and waits for it."
);

rank_k_ex!(
    csyrk3m_ex_async,
    csyrk3m_ex,
    cublasCsyrk3mEx_64,
    Complex32,
    "Enqueues the complex symmetric rank-k update with the Gauss three-multiplication method and independently typed matrices.",
    "Computes the complex symmetric rank-k update with the Gauss method and waits for it."
);

rank_k_ex!(
    cherk_ex_async,
    cherk_ex,
    cublasCherkEx_64,
    f32,
    "Enqueues the Hermitian rank-k update `C = alpha * op(A) * op(A)^H + beta * C` with real scalars and independently typed matrices.",
    "Computes the Hermitian rank-k update and waits for it."
);

rank_k_ex!(
    cherk3m_ex_async,
    cherk3m_ex,
    cublasCherk3mEx_64,
    f32,
    "Enqueues the Hermitian rank-k update with the Gauss three-multiplication method, real scalars, and independently typed matrices.",
    "Computes the Hermitian rank-k update with the Gauss method and waits for it."
);
