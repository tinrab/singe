use singe_cublas_sys as sys;
use singe_cuda::types::{Complex32, Complex64};

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{DiagonalType, FillMode, Operation, SideMode},
    utility::required_matrix_len,
    xt::context::Context,
};

trait XtScalar: Sized {
    type Raw;

    fn ptr(value: &Self) -> *const Self::Raw;
    fn slice_ptr(values: &[Self]) -> *const Self::Raw;
    fn slice_mut_ptr(values: &mut [Self]) -> *mut Self::Raw;
}

impl XtScalar for f32 {
    type Raw = f32;

    fn ptr(value: &Self) -> *const Self::Raw {
        value
    }

    fn slice_ptr(values: &[Self]) -> *const Self::Raw {
        values.as_ptr()
    }

    fn slice_mut_ptr(values: &mut [Self]) -> *mut Self::Raw {
        values.as_mut_ptr()
    }
}

impl XtScalar for f64 {
    type Raw = f64;

    fn ptr(value: &Self) -> *const Self::Raw {
        value
    }

    fn slice_ptr(values: &[Self]) -> *const Self::Raw {
        values.as_ptr()
    }

    fn slice_mut_ptr(values: &mut [Self]) -> *mut Self::Raw {
        values.as_mut_ptr()
    }
}

impl XtScalar for Complex32 {
    type Raw = sys::cuComplex;

    fn ptr(value: &Self) -> *const Self::Raw {
        (value as *const Self).cast()
    }

    fn slice_ptr(values: &[Self]) -> *const Self::Raw {
        values.as_ptr().cast()
    }

    fn slice_mut_ptr(values: &mut [Self]) -> *mut Self::Raw {
        values.as_mut_ptr().cast()
    }
}

impl XtScalar for Complex64 {
    type Raw = sys::cuDoubleComplex;

    fn ptr(value: &Self) -> *const Self::Raw {
        (value as *const Self).cast()
    }

    fn slice_ptr(values: &[Self]) -> *const Self::Raw {
        values.as_ptr().cast()
    }

    fn slice_mut_ptr(values: &mut [Self]) -> *mut Self::Raw {
        values.as_mut_ptr().cast()
    }
}

macro_rules! impl_gemm {
    ($name:ident, $ffi:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Matrices use cuBLAS column-major layout and host-accessible memory.

        pub fn $name(
            ctx: &Context,
            transpose_a: Operation,
            transpose_b: Operation,
            m: usize,
            n: usize,
            k: usize,
            alpha: &$ty,
            a: &[$ty],
            lda: usize,
            b: &[$ty],
            ldb: usize,
            beta: &$ty,
            c: &mut [$ty],
            ldc: usize,
        ) -> Result<()> {
            validate_gemm(
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
            unsafe {
                try_ffi!(sys::$ffi(
                    ctx.as_raw(),
                    transpose_a.into(),
                    transpose_b.into(),
                    m as _,
                    n as _,
                    k as _,
                    <$ty as XtScalar>::ptr(alpha),
                    <$ty as XtScalar>::slice_ptr(a),
                    lda as _,
                    <$ty as XtScalar>::slice_ptr(b),
                    ldb as _,
                    <$ty as XtScalar>::ptr(beta),
                    <$ty as XtScalar>::slice_mut_ptr(c),
                    ldc as _,
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_syrk {
    ($name:ident, $ffi:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Matrices use cuBLAS column-major layout and host-accessible memory.

        pub fn $name(
            ctx: &Context,
            fill_mode: FillMode,
            transpose_a: Operation,
            n: usize,
            k: usize,
            alpha: &$ty,
            a: &[$ty],
            lda: usize,
            beta: &$ty,
            c: &mut [$ty],
            ldc: usize,
        ) -> Result<()> {
            validate_syrk(transpose_a, n, k, a.len(), lda, c.len(), ldc)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    ctx.as_raw(),
                    fill_mode.into(),
                    transpose_a.into(),
                    n as _,
                    k as _,
                    <$ty as XtScalar>::ptr(alpha),
                    <$ty as XtScalar>::slice_ptr(a),
                    lda as _,
                    <$ty as XtScalar>::ptr(beta),
                    <$ty as XtScalar>::slice_mut_ptr(c),
                    ldc as _,
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_herk {
    ($name:ident, $ffi:ident, $ty:ty, $real:ty, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Matrices use cuBLAS column-major layout and host-accessible memory.

        pub fn $name(
            ctx: &Context,
            fill_mode: FillMode,
            transpose_a: Operation,
            n: usize,
            k: usize,
            alpha: &$real,
            a: &[$ty],
            lda: usize,
            beta: &$real,
            c: &mut [$ty],
            ldc: usize,
        ) -> Result<()> {
            validate_syrk(transpose_a, n, k, a.len(), lda, c.len(), ldc)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    ctx.as_raw(),
                    fill_mode.into(),
                    transpose_a.into(),
                    n as _,
                    k as _,
                    alpha,
                    <$ty as XtScalar>::slice_ptr(a),
                    lda as _,
                    beta,
                    <$ty as XtScalar>::slice_mut_ptr(c),
                    ldc as _,
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_syr2k_like {
    ($name:ident, $ffi:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Matrices use cuBLAS column-major layout and host-accessible memory.

        pub fn $name(
            ctx: &Context,
            fill_mode: FillMode,
            transpose_a: Operation,
            n: usize,
            k: usize,
            alpha: &$ty,
            a: &[$ty],
            lda: usize,
            b: &[$ty],
            ldb: usize,
            beta: &$ty,
            c: &mut [$ty],
            ldc: usize,
        ) -> Result<()> {
            validate_syr2k(transpose_a, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    ctx.as_raw(),
                    fill_mode.into(),
                    transpose_a.into(),
                    n as _,
                    k as _,
                    <$ty as XtScalar>::ptr(alpha),
                    <$ty as XtScalar>::slice_ptr(a),
                    lda as _,
                    <$ty as XtScalar>::slice_ptr(b),
                    ldb as _,
                    <$ty as XtScalar>::ptr(beta),
                    <$ty as XtScalar>::slice_mut_ptr(c),
                    ldc as _,
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_her2k_like {
    ($name:ident, $ffi:ident, $ty:ty, $real:ty, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Matrices use cuBLAS column-major layout and host-accessible memory.

        pub fn $name(
            ctx: &Context,
            fill_mode: FillMode,
            transpose_a: Operation,
            n: usize,
            k: usize,
            alpha: &$ty,
            a: &[$ty],
            lda: usize,
            b: &[$ty],
            ldb: usize,
            beta: &$real,
            c: &mut [$ty],
            ldc: usize,
        ) -> Result<()> {
            validate_syr2k(transpose_a, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    ctx.as_raw(),
                    fill_mode.into(),
                    transpose_a.into(),
                    n as _,
                    k as _,
                    <$ty as XtScalar>::ptr(alpha),
                    <$ty as XtScalar>::slice_ptr(a),
                    lda as _,
                    <$ty as XtScalar>::slice_ptr(b),
                    ldb as _,
                    beta,
                    <$ty as XtScalar>::slice_mut_ptr(c),
                    ldc as _,
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_symm {
    ($name:ident, $ffi:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Matrices use cuBLAS column-major layout and host-accessible memory.

        pub fn $name(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            m: usize,
            n: usize,
            alpha: &$ty,
            a: &[$ty],
            lda: usize,
            b: &[$ty],
            ldb: usize,
            beta: &$ty,
            c: &mut [$ty],
            ldc: usize,
        ) -> Result<()> {
            validate_symmetric_matrix_multiply(
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
            unsafe {
                try_ffi!(sys::$ffi(
                    ctx.as_raw(),
                    side.into(),
                    fill_mode.into(),
                    m as _,
                    n as _,
                    <$ty as XtScalar>::ptr(alpha),
                    <$ty as XtScalar>::slice_ptr(a),
                    lda as _,
                    <$ty as XtScalar>::slice_ptr(b),
                    ldb as _,
                    <$ty as XtScalar>::ptr(beta),
                    <$ty as XtScalar>::slice_mut_ptr(c),
                    ldc as _,
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_trsm {
    ($name:ident, $ffi:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Matrices use cuBLAS column-major layout and host-accessible memory.
        /// The solution overwrites `b`.

        pub fn $name(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            transpose_a: Operation,
            diagonal_type: DiagonalType,
            m: usize,
            n: usize,
            alpha: &$ty,
            a: &[$ty],
            lda: usize,
            b: &mut [$ty],
            ldb: usize,
        ) -> Result<()> {
            validate_triangular_solve(side, m, n, a.len(), lda, b.len(), ldb)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    ctx.as_raw(),
                    side.into(),
                    fill_mode.into(),
                    transpose_a.into(),
                    diagonal_type.into(),
                    m as _,
                    n as _,
                    <$ty as XtScalar>::ptr(alpha),
                    <$ty as XtScalar>::slice_ptr(a),
                    lda as _,
                    <$ty as XtScalar>::slice_mut_ptr(b),
                    ldb as _,
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_trmm {
    ($name:ident, $ffi:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Matrices use cuBLAS column-major layout and host-accessible memory.

        pub fn $name(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            transpose_a: Operation,
            diagonal_type: DiagonalType,
            m: usize,
            n: usize,
            alpha: &$ty,
            a: &[$ty],
            lda: usize,
            b: &[$ty],
            ldb: usize,
            c: &mut [$ty],
            ldc: usize,
        ) -> Result<()> {
            validate_triangular_matrix_multiply(
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
            unsafe {
                try_ffi!(sys::$ffi(
                    ctx.as_raw(),
                    side.into(),
                    fill_mode.into(),
                    transpose_a.into(),
                    diagonal_type.into(),
                    m as _,
                    n as _,
                    <$ty as XtScalar>::ptr(alpha),
                    <$ty as XtScalar>::slice_ptr(a),
                    lda as _,
                    <$ty as XtScalar>::slice_ptr(b),
                    ldb as _,
                    <$ty as XtScalar>::slice_mut_ptr(c),
                    ldc as _,
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_spmm {
    ($name:ident, $ffi:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Matrices use cuBLAS column-major layout and host-accessible memory.
        /// `ap` stores the symmetric matrix in packed format.

        pub fn $name(
            ctx: &Context,
            side: SideMode,
            fill_mode: FillMode,
            m: usize,
            n: usize,
            alpha: &$ty,
            ap: &[$ty],
            b: &[$ty],
            ldb: usize,
            beta: &$ty,
            c: &mut [$ty],
            ldc: usize,
        ) -> Result<()> {
            validate_packed_symmetric_matrix_multiply(
                side,
                m,
                n,
                ap.len(),
                b.len(),
                ldb,
                c.len(),
                ldc,
            )?;
            unsafe {
                try_ffi!(sys::$ffi(
                    ctx.as_raw(),
                    side.into(),
                    fill_mode.into(),
                    m as _,
                    n as _,
                    <$ty as XtScalar>::ptr(alpha),
                    <$ty as XtScalar>::slice_ptr(ap),
                    <$ty as XtScalar>::slice_ptr(b),
                    ldb as _,
                    <$ty as XtScalar>::ptr(beta),
                    <$ty as XtScalar>::slice_mut_ptr(c),
                    ldc as _,
                ))?;
            }
            Ok(())
        }
    };
}

impl_gemm!(
    sgemm,
    cublasXtSgemm,
    f32,
    "Performs single-precision host-memory GEMM through cuBLASXt."
);
impl_gemm!(
    dgemm,
    cublasXtDgemm,
    f64,
    "Performs double-precision host-memory GEMM through cuBLASXt."
);
impl_gemm!(
    cgemm,
    cublasXtCgemm,
    Complex32,
    "Performs single-precision complex host-memory GEMM through cuBLASXt."
);
impl_gemm!(
    zgemm,
    cublasXtZgemm,
    Complex64,
    "Performs double-precision complex host-memory GEMM through cuBLASXt."
);

impl_syrk!(
    ssyrk,
    cublasXtSsyrk,
    f32,
    "Performs single-precision host-memory SYRK through cuBLASXt."
);
impl_syrk!(
    dsyrk,
    cublasXtDsyrk,
    f64,
    "Performs double-precision host-memory SYRK through cuBLASXt."
);
impl_syrk!(
    csyrk,
    cublasXtCsyrk,
    Complex32,
    "Performs single-precision complex host-memory SYRK through cuBLASXt."
);
impl_syrk!(
    zsyrk,
    cublasXtZsyrk,
    Complex64,
    "Performs double-precision complex host-memory SYRK through cuBLASXt."
);

impl_herk!(
    cherk,
    cublasXtCherk,
    Complex32,
    f32,
    "Performs single-precision complex host-memory HERK through cuBLASXt."
);
impl_herk!(
    zherk,
    cublasXtZherk,
    Complex64,
    f64,
    "Performs double-precision complex host-memory HERK through cuBLASXt."
);

impl_syr2k_like!(
    ssyr2k,
    cublasXtSsyr2k,
    f32,
    "Performs single-precision host-memory SYR2K through cuBLASXt."
);
impl_syr2k_like!(
    dsyr2k,
    cublasXtDsyr2k,
    f64,
    "Performs double-precision host-memory SYR2K through cuBLASXt."
);
impl_syr2k_like!(
    csyr2k,
    cublasXtCsyr2k,
    Complex32,
    "Performs single-precision complex host-memory SYR2K through cuBLASXt."
);
impl_syr2k_like!(
    zsyr2k,
    cublasXtZsyr2k,
    Complex64,
    "Performs double-precision complex host-memory SYR2K through cuBLASXt."
);

impl_her2k_like!(
    cherkx,
    cublasXtCherkx,
    Complex32,
    f32,
    "Performs single-precision complex host-memory HERKX through cuBLASXt."
);
impl_her2k_like!(
    zherkx,
    cublasXtZherkx,
    Complex64,
    f64,
    "Performs double-precision complex host-memory HERKX through cuBLASXt."
);

impl_trsm!(
    strsm,
    cublasXtStrsm,
    f32,
    "Performs single-precision host-memory TRSM through cuBLASXt."
);
impl_trsm!(
    dtrsm,
    cublasXtDtrsm,
    f64,
    "Performs double-precision host-memory TRSM through cuBLASXt."
);
impl_trsm!(
    ctrsm,
    cublasXtCtrsm,
    Complex32,
    "Performs single-precision complex host-memory TRSM through cuBLASXt."
);
impl_trsm!(
    ztrsm,
    cublasXtZtrsm,
    Complex64,
    "Performs double-precision complex host-memory TRSM through cuBLASXt."
);

impl_symm!(
    ssymm,
    cublasXtSsymm,
    f32,
    "Performs single-precision host-memory SYMM through cuBLASXt."
);
impl_symm!(
    dsymm,
    cublasXtDsymm,
    f64,
    "Performs double-precision host-memory SYMM through cuBLASXt."
);
impl_symm!(
    csymm,
    cublasXtCsymm,
    Complex32,
    "Performs single-precision complex host-memory SYMM through cuBLASXt."
);
impl_symm!(
    zsymm,
    cublasXtZsymm,
    Complex64,
    "Performs double-precision complex host-memory SYMM through cuBLASXt."
);

impl_symm!(
    chemm,
    cublasXtChemm,
    Complex32,
    "Performs single-precision complex host-memory HEMM through cuBLASXt."
);
impl_symm!(
    zhemm,
    cublasXtZhemm,
    Complex64,
    "Performs double-precision complex host-memory HEMM through cuBLASXt."
);

impl_syr2k_like!(
    ssyrkx,
    cublasXtSsyrkx,
    f32,
    "Performs single-precision host-memory SYRKX through cuBLASXt."
);
impl_syr2k_like!(
    dsyrkx,
    cublasXtDsyrkx,
    f64,
    "Performs double-precision host-memory SYRKX through cuBLASXt."
);
impl_syr2k_like!(
    csyrkx,
    cublasXtCsyrkx,
    Complex32,
    "Performs single-precision complex host-memory SYRKX through cuBLASXt."
);
impl_syr2k_like!(
    zsyrkx,
    cublasXtZsyrkx,
    Complex64,
    "Performs double-precision complex host-memory SYRKX through cuBLASXt."
);

impl_her2k_like!(
    cher2k,
    cublasXtCher2k,
    Complex32,
    f32,
    "Performs single-precision complex host-memory HER2K through cuBLASXt."
);
impl_her2k_like!(
    zher2k,
    cublasXtZher2k,
    Complex64,
    f64,
    "Performs double-precision complex host-memory HER2K through cuBLASXt."
);

impl_spmm!(
    sspmm,
    cublasXtSspmm,
    f32,
    "Performs single-precision host-memory SPMM through cuBLASXt."
);
impl_spmm!(
    dspmm,
    cublasXtDspmm,
    f64,
    "Performs double-precision host-memory SPMM through cuBLASXt."
);
impl_spmm!(
    cspmm,
    cublasXtCspmm,
    Complex32,
    "Performs single-precision complex host-memory SPMM through cuBLASXt."
);
impl_spmm!(
    zspmm,
    cublasXtZspmm,
    Complex64,
    "Performs double-precision complex host-memory SPMM through cuBLASXt."
);

impl_trmm!(
    strmm,
    cublasXtStrmm,
    f32,
    "Performs single-precision host-memory TRMM through cuBLASXt."
);
impl_trmm!(
    dtrmm,
    cublasXtDtrmm,
    f64,
    "Performs double-precision host-memory TRMM through cuBLASXt."
);
impl_trmm!(
    ctrmm,
    cublasXtCtrmm,
    Complex32,
    "Performs single-precision complex host-memory TRMM through cuBLASXt."
);
impl_trmm!(
    ztrmm,
    cublasXtZtrmm,
    Complex64,
    "Performs double-precision complex host-memory TRMM through cuBLASXt."
);

fn validate_gemm(
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
    let a_rows = if transpose_a == Operation::NonTranspose {
        m
    } else {
        k
    };
    let a_cols = if transpose_a == Operation::NonTranspose {
        k
    } else {
        m
    };
    let b_rows = if transpose_b == Operation::NonTranspose {
        k
    } else {
        n
    };
    let b_cols = if transpose_b == Operation::NonTranspose {
        n
    } else {
        k
    };

    validate_three_matrices(
        a_rows, a_cols, a_len, lda, b_rows, b_cols, b_len, ldb, m, n, c_len, ldc,
    )
}

fn validate_syrk(
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

fn validate_syr2k(
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
    validate_syrk(transpose_a, n, k, a_len, lda, c_len, ldc)?;

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

fn validate_symmetric_matrix_multiply(
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
    let a_dimension = side_dimension(side, m, n);
    validate_three_matrices(
        a_dimension,
        a_dimension,
        a_len,
        lda,
        m,
        n,
        b_len,
        ldb,
        m,
        n,
        c_len,
        ldc,
    )
}

fn validate_triangular_matrix_multiply(
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
    let a_dimension = side_dimension(side, m, n);
    validate_three_matrices(
        a_dimension,
        a_dimension,
        a_len,
        lda,
        m,
        n,
        b_len,
        ldb,
        m,
        n,
        c_len,
        ldc,
    )
}

fn validate_triangular_solve(
    side: SideMode,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
) -> Result<()> {
    let a_dimension = side_dimension(side, m, n);
    if lda < a_dimension || ldb < m {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, a_dimension)? || b_len < required_matrix_len(ldb, n)? {
        return Err(Error::InvalidMatrixShape);
    }
    Ok(())
}

fn validate_packed_symmetric_matrix_multiply(
    side: SideMode,
    m: usize,
    n: usize,
    ap_len: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    let a_dimension = side_dimension(side, m, n);
    if ldb < m || ldc < m {
        return Err(Error::InvalidLeadingDimension);
    }
    if ap_len < required_packed_triangular_len(a_dimension)?
        || b_len < required_matrix_len(ldb, n)?
        || c_len < required_matrix_len(ldc, n)?
    {
        return Err(Error::InvalidMatrixShape);
    }
    Ok(())
}

fn validate_three_matrices(
    a_rows: usize,
    a_cols: usize,
    a_len: usize,
    lda: usize,
    b_rows: usize,
    b_cols: usize,
    b_len: usize,
    ldb: usize,
    c_rows: usize,
    c_cols: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    if lda < a_rows || ldb < b_rows || ldc < c_rows {
        return Err(Error::InvalidLeadingDimension);
    }
    if a_len < required_matrix_len(lda, a_cols)?
        || b_len < required_matrix_len(ldb, b_cols)?
        || c_len < required_matrix_len(ldc, c_cols)?
    {
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

fn side_dimension(side: SideMode, m: usize, n: usize) -> usize {
    match side {
        SideMode::Left => m,
        SideMode::Right => n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short_gemm_matrix() {
        let err = validate_gemm(
            Operation::NonTranspose,
            Operation::NonTranspose,
            2,
            2,
            2,
            3,
            2,
            4,
            2,
            4,
            2,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidMatrixShape));
    }

    #[test]
    fn accepts_transposed_gemm_shapes() -> Result<()> {
        validate_gemm(
            Operation::Transpose,
            Operation::NonTranspose,
            2,
            2,
            3,
            6,
            3,
            6,
            3,
            4,
            2,
        )
    }

    #[test]
    fn rejects_short_packed_matrix() {
        let err = validate_packed_symmetric_matrix_multiply(SideMode::Left, 3, 2, 5, 6, 3, 6, 3)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidMatrixShape));
    }
}
