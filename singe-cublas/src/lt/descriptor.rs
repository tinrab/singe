use std::{
    mem::{MaybeUninit, size_of, size_of_val},
    ptr,
};

use singe_cublas_sys as sys;
use singe_cuda::{
    data_type::DataType,
    memory::DeviceMemory,
    types::{EmulationMantissaControl, EmulationSpecialValuesSupport},
};
use singe_cuda_sys::library_types;

use crate::{
    error::{Error, Result},
    lt::{
        types::{
            BatchMode, EmulationDescAttribute, IntegerWidth, MatrixLayoutAttribute,
            MatrixTransformDescAttribute, Order, PointerMode,
        },
        utility::{read_attribute, set_attribute},
    },
    try_ffi,
    types::Operation,
    utility::{ensure_exact_size, to_i32},
};

/// Associates a matrix layout dimension type with its integer width for
/// grouped matrix descriptors.
pub trait GroupedMatrixLayoutValue: Copy + 'static {
    const INTEGER_WIDTH: IntegerWidth;
}

impl GroupedMatrixLayoutValue for i32 {
    const INTEGER_WIDTH: IntegerWidth = IntegerWidth::Bits32;
}

impl GroupedMatrixLayoutValue for u32 {
    const INTEGER_WIDTH: IntegerWidth = IntegerWidth::Bits32;
}

impl GroupedMatrixLayoutValue for i64 {
    const INTEGER_WIDTH: IntegerWidth = IntegerWidth::Bits64;
}

impl GroupedMatrixLayoutValue for u64 {
    const INTEGER_WIDTH: IntegerWidth = IntegerWidth::Bits64;
}

#[derive(Debug)]
pub struct MatrixLayout {
    raw: sys::cublasLtMatrixLayout_t,
}

impl MatrixLayout {
    /// Creates a matrix layout descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot allocate the descriptor or if it does not return
    /// a valid handle.
    pub fn create(data_type: DataType, rows: u64, cols: u64, ld: i64) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cublasLtMatrixLayoutCreate(
                &raw mut raw,
                data_type.into(),
                rows,
                cols,
                ld,
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Experimental: creates a grouped matrix layout descriptor.
    ///
    /// All device arrays must have the same length and remain valid while the
    /// grouped layout descriptor is used.
    ///
    /// # Errors
    ///
    /// Returns an error if the grouped dimension slices have mismatched lengths, if cuBLASLt
    /// cannot allocate the descriptor, or if it does not return a valid handle.
    pub fn create_grouped<T: GroupedMatrixLayoutValue>(
        data_type: DataType,
        rows: &DeviceMemory<T>,
        cols: &DeviceMemory<T>,
        ld: &DeviceMemory<T>,
    ) -> Result<Self> {
        if rows.len() != cols.len() || rows.len() != ld.len() {
            return Err(Error::LengthMismatch {
                name: "grouped matrix layout arrays".into(),
                expected: rows.len(),
                actual: cols.len().min(ld.len()),
            });
        }

        let group_count = to_i32(rows.len(), "group count")?;

        let mut raw = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cublasLtGroupedMatrixLayoutCreate(
                &raw mut raw,
                data_type.into(),
                group_count,
                rows.as_ptr().cast(),
                cols.as_ptr().cast(),
                ld.as_ptr().cast(),
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        let mut layout = Self { raw };
        layout.set_attribute(
            MatrixLayoutAttribute::GroupedRowsColsArrayIntegerWidth,
            &T::INTEGER_WIDTH,
        )?;
        layout.set_attribute(
            MatrixLayoutAttribute::GroupedLeadingDimensionArrayIntegerWidth,
            &T::INTEGER_WIDTH,
        )?;

        Ok(layout)
    }

    /// Sets a matrix layout attribute from a single value.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the
    /// attribute storage expected by cuBLASLt.
    pub fn set_attribute<T>(&mut self, attr: MatrixLayoutAttribute, value: &T) -> Result<()> {
        set_attribute(
            |value, size| unsafe {
                sys::cublasLtMatrixLayoutSetAttribute(self.raw, attr.into(), value, size)
            },
            (value as *const T).cast(),
            size_of::<T>(),
        )
    }

    /// Sets a matrix layout attribute from a slice.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the slice size does not match the
    /// attribute storage expected by cuBLASLt.
    pub fn set_attribute_slice<T>(
        &mut self,
        attr: MatrixLayoutAttribute,
        values: &[T],
    ) -> Result<()> {
        set_attribute(
            |value, size| unsafe {
                sys::cublasLtMatrixLayoutSetAttribute(self.raw, attr.into(), value, size)
            },
            values.as_ptr().cast(),
            size_of_val(values),
        )
    }

    /// Returns a matrix layout attribute value.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not
    /// match `T`.
    pub fn attribute<T: Copy>(&self, attr: MatrixLayoutAttribute) -> Result<T> {
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            |value, size, written| unsafe {
                sys::cublasLtMatrixLayoutGetAttribute(self.raw, attr.into(), value, size, written)
            },
            value.as_mut_ptr().cast(),
            size_of::<T>(),
            "matrix layout attribute",
        )?;
        ensure_exact_size(written, size_of::<T>())?;
        Ok(unsafe { value.assume_init() })
    }

    /// Sets the matrix data ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_order(&mut self, order: Order) -> Result<()> {
        self.set_attribute(
            MatrixLayoutAttribute::Order,
            &sys::cublasLtOrder_t::from(order),
        )
    }

    /// Returns the matrix data ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn order(&self) -> Result<Order> {
        Ok(self
            .attribute::<sys::cublasLtOrder_t>(MatrixLayoutAttribute::Order)?
            .into())
    }

    /// Sets the batch count for batched computation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_batch_count(&mut self, batch_count: i32) -> Result<()> {
        self.set_attribute(MatrixLayoutAttribute::BatchCount, &batch_count)
    }

    /// Returns the batch count.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn batch_count(&self) -> Result<i32> {
        self.attribute(MatrixLayoutAttribute::BatchCount)
    }

    /// Sets the batch mode for the matrix layout.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_batch_mode(&mut self, batch_mode: BatchMode) -> Result<()> {
        self.set_attribute(
            MatrixLayoutAttribute::BatchMode,
            &sys::cublasLtBatchMode_t::from(batch_mode),
        )
    }

    /// Returns the batch mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn batch_mode(&self) -> Result<BatchMode> {
        Ok(self
            .attribute::<sys::cublasLtBatchMode_t>(MatrixLayoutAttribute::BatchMode)?
            .into())
    }

    /// Sets the strided batch offset between consecutive matrices in a batch.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_strided_batch_offset(&mut self, offset: i64) -> Result<()> {
        self.set_attribute(MatrixLayoutAttribute::StridedBatchOffset, &offset)
    }

    /// Returns the strided batch offset.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn strided_batch_offset(&self) -> Result<i64> {
        self.attribute(MatrixLayoutAttribute::StridedBatchOffset)
    }

    /// Configures this layout as a strided batch of matrices.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_count` cannot fit in cuBLASLt's 32-bit batch
    /// count field, or if cuBLASLt rejects one of the batch attributes.
    pub fn set_strided_batch(&mut self, batch_count: usize, stride: i64) -> Result<()> {
        let batch_count = to_i32(batch_count, "matrix layout batch count")?;
        self.set_batch_mode(BatchMode::Strided)?;
        self.set_batch_count(batch_count)?;
        self.set_strided_batch_offset(stride)
    }

    /// Sets the plane offset in bytes for 3D matrix layouts.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_plane_offset(&mut self, offset_bytes: i64) -> Result<()> {
        self.set_attribute(MatrixLayoutAttribute::PlaneOffset, &offset_bytes)
    }

    /// Returns the raw cuBLASLt matrix layout handle.
    ///
    /// The returned handle is borrowed and remains valid only while the layout
    /// is alive.
    pub fn as_raw(&self) -> sys::cublasLtMatrixLayout_t {
        self.raw
    }

    /// Takes ownership of a raw cuBLASLt matrix layout handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid `cublasLtMatrixLayout_t` handle created by cuBLASLt.
    /// The returned wrapper takes ownership and will destroy it with
    /// `cublasLtMatrixLayoutDestroy`; no other owner may destroy or keep using it.
    pub unsafe fn from_raw(raw: sys::cublasLtMatrixLayout_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Releases ownership and returns the raw cuBLASLt matrix layout handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::cublasLtMatrixLayout_t {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cublasLtMatrixLayoutDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cublasLt matrix layout: {err}");
            }
        }
    }
}

#[derive(Debug)]
pub struct MatrixTransformDescriptor {
    raw: sys::cublasLtMatrixTransformDesc_t,
}

impl MatrixTransformDescriptor {
    /// Creates a matrix transform descriptor.
    ///
    /// The descriptor owns its cuBLASLt handle and destroys it when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot allocate the descriptor or if it does not return
    /// a valid handle.
    pub fn create(scale_type: DataType) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cublasLtMatrixTransformDescCreate(
                &raw mut raw,
                scale_type.into(),
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Sets a matrix transform descriptor attribute.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the
    /// attribute storage expected by cuBLASLt.
    pub fn set_attribute<T>(
        &mut self,
        attr: MatrixTransformDescAttribute,
        value: &T,
    ) -> Result<()> {
        set_attribute(
            |value, size| unsafe {
                sys::cublasLtMatrixTransformDescSetAttribute(self.raw, attr.into(), value, size)
            },
            (value as *const T).cast(),
            size_of::<T>(),
        )
    }

    /// Returns a matrix transform descriptor attribute value.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not
    /// match `T`.
    pub fn attribute<T: Copy>(&self, attr: MatrixTransformDescAttribute) -> Result<T> {
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            |value, size, written| unsafe {
                sys::cublasLtMatrixTransformDescGetAttribute(
                    self.raw,
                    attr.into(),
                    value,
                    size,
                    written,
                )
            },
            value.as_mut_ptr().cast(),
            size_of::<T>(),
            "matrix transform descriptor attribute",
        )?;
        ensure_exact_size(written, size_of::<T>())?;
        Ok(unsafe { value.assume_init() })
    }

    /// Sets the pointer mode for alpha and beta scalars.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_pointer_mode(&mut self, pointer_mode: PointerMode) -> Result<()> {
        self.set_attribute(
            MatrixTransformDescAttribute::PointerMode,
            &sys::cublasLtPointerMode_t::from(pointer_mode),
        )
    }

    /// Sets the transpose mode applied to input matrix A.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_transpose_a(&mut self, operation: Operation) -> Result<()> {
        self.set_attribute(
            MatrixTransformDescAttribute::TransposeA,
            &sys::cublasOperation_t::from(operation),
        )
    }

    /// Sets the transpose mode applied to input matrix B.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_transpose_b(&mut self, operation: Operation) -> Result<()> {
        self.set_attribute(
            MatrixTransformDescAttribute::TransposeB,
            &sys::cublasOperation_t::from(operation),
        )
    }

    /// Returns the raw cuBLASLt matrix transform descriptor handle.
    ///
    /// The returned handle is borrowed and remains valid only while the
    /// descriptor is alive.
    pub fn as_raw(&self) -> sys::cublasLtMatrixTransformDesc_t {
        self.raw
    }

    /// Takes ownership of a raw cuBLASLt matrix transform descriptor handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid `cublasLtMatrixTransformDesc_t` handle created by
    /// cuBLASLt. The returned wrapper takes ownership and will destroy it with
    /// `cublasLtMatrixTransformDescDestroy`; no other owner may destroy or keep
    /// using it.
    pub unsafe fn from_raw(raw: sys::cublasLtMatrixTransformDesc_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Releases ownership and returns the raw cuBLASLt matrix transform descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::cublasLtMatrixTransformDesc_t {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }
}

impl Drop for MatrixTransformDescriptor {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cublasLtMatrixTransformDescDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cublasLt matrix transform descriptor: {err}");
            }
        }
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn matrix_layout_sets_strided_batch_attributes() -> Result<()> {
        let mut layout = MatrixLayout::create(DataType::F32, 2, 3, 3)?;

        layout.set_strided_batch(4, 6)?;

        assert_eq!(layout.batch_mode()?, BatchMode::Strided);
        assert_eq!(layout.batch_count()?, 4);
        assert_eq!(layout.strided_batch_offset()?, 6);
        Ok(())
    }
}

#[derive(Debug)]
pub struct EmulationDescriptor {
    raw: sys::cublasLtEmulationDesc_t,
}

impl EmulationDescriptor {
    /// Creates an emulation descriptor.
    ///
    /// The descriptor owns its cuBLASLt handle and destroys it when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot allocate the descriptor or if it does not return
    /// a valid handle.
    pub fn create() -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cublasLtEmulationDescCreate(&raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Sets an emulation descriptor attribute.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the
    /// attribute storage expected by cuBLASLt.
    pub fn set_attribute<T>(&mut self, attr: EmulationDescAttribute, value: &T) -> Result<()> {
        set_attribute(
            |value, size| unsafe {
                sys::cublasLtEmulationDescSetAttribute(self.raw, attr.into(), value, size)
            },
            (value as *const T).cast(),
            size_of::<T>(),
        )
    }

    /// Returns an emulation descriptor attribute value.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not
    /// match `T`.
    pub fn attribute<T: Copy>(&self, attr: EmulationDescAttribute) -> Result<T> {
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            |value, size, written| unsafe {
                sys::cublasLtEmulationDescGetAttribute(self.raw, attr.into(), value, size, written)
            },
            value.as_mut_ptr().cast(),
            size_of::<T>(),
            "emulation descriptor attribute",
        )?;
        ensure_exact_size(written, size_of::<T>())?;
        Ok(unsafe { value.assume_init() })
    }

    /// Sets the floating-point emulation strategy.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_strategy(&mut self, strategy: singe_cuda::types::EmulationStrategy) -> Result<()> {
        self.set_attribute(
            EmulationDescAttribute::Strategy,
            &library_types::cudaEmulationStrategy::from(strategy),
        )
    }

    /// Sets the special values support for floating-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_special_values_support(
        &mut self,
        support: EmulationSpecialValuesSupport,
    ) -> Result<()> {
        let support = library_types::cudaEmulationSpecialValuesSupport::from(support);
        self.set_attribute(EmulationDescAttribute::SpecialValuesSupport, &support)
    }

    /// Sets the mantissa bit control for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_mantissa_control(&mut self, control: EmulationMantissaControl) -> Result<()> {
        self.set_attribute(
            EmulationDescAttribute::FixedPointMantissaControl,
            &library_types::cudaEmulationMantissaControl::from(control),
        )
    }

    /// Sets the maximum mantissa bit count for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_max_mantissa_bit_count(&mut self, count: i32) -> Result<()> {
        self.set_attribute(
            EmulationDescAttribute::FixedPointMaxMantissaBitCount,
            &count,
        )
    }

    /// Sets the mantissa bit offset for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_mantissa_bit_offset(&mut self, offset: i32) -> Result<()> {
        self.set_attribute(EmulationDescAttribute::FixedPointMantissaBitOffset, &offset)
    }

    /// Returns the raw cuBLASLt emulation descriptor handle.
    ///
    /// The returned handle is borrowed and remains valid only while the
    /// descriptor is alive.
    pub fn as_raw(&self) -> sys::cublasLtEmulationDesc_t {
        self.raw
    }

    /// Takes ownership of a raw cuBLASLt emulation descriptor handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid `cublasLtEmulationDesc_t` handle created by
    /// cuBLASLt. The returned wrapper takes ownership and will destroy it with
    /// `cublasLtEmulationDescDestroy`; no other owner may destroy or keep using it.
    pub unsafe fn from_raw(raw: sys::cublasLtEmulationDesc_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Releases ownership and returns the raw cuBLASLt emulation descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::cublasLtEmulationDesc_t {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }
}

impl Drop for EmulationDescriptor {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cublasLtEmulationDescDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cublasLt emulation descriptor: {err}");
            }
        }
    }
}
