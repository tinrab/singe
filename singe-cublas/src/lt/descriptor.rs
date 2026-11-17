use std::{
    mem,
    mem::{MaybeUninit, size_of, size_of_val},
    ptr,
};

use singe_core::report_cleanup_failure;
use singe_cublas_sys as sys;
use singe_cuda::{
    data_type::DataType,
    memory::DeviceMemory,
    types::{EmulationMantissaControl, EmulationSpecialValuesSupport, EmulationStrategy},
    view::{AnyBitPatternDeviceRepr, NoUninitDeviceRepr},
};
use singe_cuda_sys::library_types;

use crate::{
    error::{Error, Result, try_ffi},
    lt::{
        types::{
            BatchMode, EmulationDescAttribute, IntegerWidth, MatrixLayoutAttribute,
            MatrixTransformDescAttribute, Order, PointerMode,
        },
        utility::{attribute_enum, read_attribute, set_attribute},
    },
    types::Operation,
    utility::{ensure_exact_size, to_i32},
};

/// Associates a matrix layout dimension type with its integer width for grouped matrix descriptors.
///
/// # Safety
///
/// Values must use the native integer size and alignment selected by `INTEGER_WIDTH`, without padding or restricted integer bit patterns.
pub unsafe trait GroupedMatrixLayoutValue: NoUninitDeviceRepr {
    /// The native integer width used for arrays of this element type.
    const INTEGER_WIDTH: IntegerWidth;
}

// SAFETY: i32 has the native 32-bit integer representation.
unsafe impl GroupedMatrixLayoutValue for i32 {
    const INTEGER_WIDTH: IntegerWidth = IntegerWidth::Bits32;
}

// SAFETY: u32 has the native 32-bit integer representation.
unsafe impl GroupedMatrixLayoutValue for u32 {
    const INTEGER_WIDTH: IntegerWidth = IntegerWidth::Bits32;
}

// SAFETY: i64 has the native 64-bit integer representation.
unsafe impl GroupedMatrixLayoutValue for i64 {
    const INTEGER_WIDTH: IntegerWidth = IntegerWidth::Bits64;
}

// SAFETY: u64 has the native 64-bit integer representation.
unsafe impl GroupedMatrixLayoutValue for u64 {
    const INTEGER_WIDTH: IntegerWidth = IntegerWidth::Bits64;
}

/// An owned cuBLASLt matrix layout descriptor.
#[derive(Debug)]
pub struct MatrixLayout {
    raw: sys::cublasLtMatrixLayout_t,
}

// SAFETY: the descriptor is host memory owned by this value; cuBLASLt allows using and destroying it from any thread, one call at a time.
unsafe impl Send for MatrixLayout {}

impl MatrixLayout {
    /// Creates a matrix layout descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot allocate the descriptor or if it does not return a valid handle.
    pub fn create(data_type: DataType, rows: u64, cols: u64, ld: i64) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `cublasLtMatrixLayoutCreate` writes the new descriptor to `raw`, a live local, and takes its other arguments by value.
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
    /// The descriptor stores the device addresses of `rows`, `cols`, and `ld` without borrowing them.
    ///
    /// # Safety
    ///
    /// `rows`, `cols`, and `ld` must stay live, hold the per-group dimensions of matrices that the operation's matrix pointers address, and not be written by other work until every matmul using this layout completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the grouped dimension slices have mismatched lengths, if cuBLASLt cannot allocate the descriptor, or if it does not return a valid handle.
    pub unsafe fn create_grouped<T: GroupedMatrixLayoutValue>(
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
        // SAFETY: `group_count` matches the length of each array, and the caller keeps the stored device addresses valid for every use of the layout.
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
        // SAFETY: both attributes store the 32-bit integer-width enum supplied by the ABI trait.
        unsafe {
            layout.set_attribute(
                MatrixLayoutAttribute::GroupedRowsColsArrayIntegerWidth,
                &T::INTEGER_WIDTH,
            )?;
            layout.set_attribute(
                MatrixLayoutAttribute::GroupedLeadingDimensionArrayIntegerWidth,
                &T::INTEGER_WIDTH,
            )?;
        }

        Ok(layout)
    }

    /// Sets a matrix layout attribute from a single value.
    ///
    /// # Safety
    ///
    /// `T` must match the selected attribute's native representation, including its validity and initialized-byte requirements.
    /// Retain pointer targets for every native use.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the attribute storage expected by cuBLASLt.
    pub unsafe fn set_attribute<T>(
        &mut self,
        attr: MatrixLayoutAttribute,
        value: &T,
    ) -> Result<()> {
        set_attribute(
            // SAFETY: `self.raw` is the live descriptor this value owns, and `set_attribute` passes `value` with its exact byte size.
            |value, size| unsafe {
                sys::cublasLtMatrixLayoutSetAttribute(self.raw, attr.into(), value, size)
            },
            ptr::from_ref::<T>(value).cast(),
            size_of::<T>(),
        )
    }

    /// Sets a matrix layout attribute from a slice.
    ///
    /// # Safety
    ///
    /// `T` must match the selected attribute's native representation, including its validity and initialized-byte requirements.
    /// Retain pointer targets for every native use.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the slice size does not match the attribute storage expected by cuBLASLt.
    pub unsafe fn set_attribute_slice<T>(
        &mut self,
        attr: MatrixLayoutAttribute,
        values: &[T],
    ) -> Result<()> {
        set_attribute(
            // SAFETY: `self.raw` is the live descriptor this value owns, and `set_attribute` passes `value` with its exact byte size.
            |value, size| unsafe {
                sys::cublasLtMatrixLayoutSetAttribute(self.raw, attr.into(), value, size)
            },
            values.as_ptr().cast(),
            size_of_val(values),
        )
    }

    /// Returns a matrix layout attribute value.
    ///
    /// `T` must accept every bit pattern; enum attributes are read as integers and validated.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not match `T`.
    pub fn attribute<T: AnyBitPatternDeviceRepr>(&self, attr: MatrixLayoutAttribute) -> Result<T> {
        let () = T::LAYOUT;
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            // SAFETY: `self.raw` is the live descriptor this value owns, and `read_attribute` passes `value` with its byte size and a live `written` location.
            |value, size, written| unsafe {
                sys::cublasLtMatrixLayoutGetAttribute(self.raw, attr.into(), value, size, written)
            },
            value.as_mut_ptr().cast(),
            size_of::<T>(),
            "matrix layout attribute",
        )?;
        ensure_exact_size("matrix layout attribute", written, size_of::<T>())?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        Ok(unsafe { value.assume_init() })
    }

    /// Sets the matrix data ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_order(&mut self, order: Order) -> Result<()> {
        // SAFETY: `MatrixLayoutAttribute::Order` stores one `cublasLtOrder_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatrixLayoutAttribute::Order,
                &sys::cublasLtOrder_t::from(order),
            )
        }
    }

    /// Returns the matrix data ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn order(&self) -> Result<Order> {
        let value = self.attribute::<u32>(MatrixLayoutAttribute::Order)?;
        attribute_enum(value, "MatrixLayoutAttribute::Order")
    }

    /// Sets the batch count for batched computation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_batch_count(&mut self, batch_count: i32) -> Result<()> {
        // SAFETY: `MatrixLayoutAttribute::BatchCount` stores one `i32`, the type passed here.
        unsafe { self.set_attribute(MatrixLayoutAttribute::BatchCount, &batch_count) }
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
        // SAFETY: `MatrixLayoutAttribute::BatchMode` stores one `cublasLtBatchMode_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatrixLayoutAttribute::BatchMode,
                &sys::cublasLtBatchMode_t::from(batch_mode),
            )
        }
    }

    /// Returns the batch mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn batch_mode(&self) -> Result<BatchMode> {
        let value = self.attribute::<u32>(MatrixLayoutAttribute::BatchMode)?;
        attribute_enum(value, "MatrixLayoutAttribute::BatchMode")
    }

    /// Sets the strided batch offset between consecutive matrices in a batch.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_strided_batch_offset(&mut self, offset: i64) -> Result<()> {
        // SAFETY: `MatrixLayoutAttribute::StridedBatchOffset` stores one `i64`, the type passed here.
        unsafe { self.set_attribute(MatrixLayoutAttribute::StridedBatchOffset, &offset) }
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
    /// Returns an error if `batch_count` cannot fit in cuBLASLt's 32-bit batch count field, or if cuBLASLt rejects one of the batch attributes.
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
        // SAFETY: `MatrixLayoutAttribute::PlaneOffset` stores one `i64`, the type passed here.
        unsafe { self.set_attribute(MatrixLayoutAttribute::PlaneOffset, &offset_bytes) }
    }

    /// Returns the raw cuBLASLt matrix layout handle.
    ///
    /// The returned handle is borrowed and remains valid only while the layout is alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cublasLtMatrixLayout_t {
        self.raw
    }

    /// Takes ownership of a raw cuBLASLt matrix layout handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a live cuBLASLt matrix layout descriptor.
    /// The returned value takes ownership and destroys it on drop; no other owner may destroy or keep using it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw` is null.
    pub unsafe fn from_raw(raw: sys::cublasLtMatrixLayout_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Releases ownership and returns the raw cuBLASLt matrix layout handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> sys::cublasLtMatrixLayout_t {
        let raw = self.raw;
        mem::forget(self);
        raw
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        // SAFETY: this value owns `self.raw`, and `drop` destroys it exactly once.
        unsafe {
            if let Err(error) = try_ffi!(sys::cublasLtMatrixLayoutDestroy(self.raw)) {
                report_cleanup_failure("destroy", "cublaslt matrix layout", &error);
            }
        }
    }
}

/// An owned cuBLASLt descriptor for matrix scaling, transposition, and layout conversion.
#[derive(Debug)]
pub struct MatrixTransformDescriptor {
    raw: sys::cublasLtMatrixTransformDesc_t,
}

// SAFETY: the descriptor is host memory owned by this value; cuBLASLt allows using and destroying it from any thread, one call at a time.
unsafe impl Send for MatrixTransformDescriptor {}

impl MatrixTransformDescriptor {
    /// Creates a matrix transform descriptor.
    ///
    /// The descriptor owns its cuBLASLt handle and destroys it when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot allocate the descriptor or if it does not return a valid handle.
    pub fn create(scale_type: DataType) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `cublasLtMatrixTransformDescCreate` writes the new descriptor to `raw`, a live local, and takes its other arguments by value.
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
    /// # Safety
    ///
    /// `T` must match the selected attribute's native representation, including its validity and initialized-byte requirements.
    /// Retain pointer targets for every native use.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the attribute storage expected by cuBLASLt.
    pub unsafe fn set_attribute<T>(
        &mut self,
        attr: MatrixTransformDescAttribute,
        value: &T,
    ) -> Result<()> {
        set_attribute(
            // SAFETY: `self.raw` is the live descriptor this value owns, and `set_attribute` passes `value` with its exact byte size.
            |value, size| unsafe {
                sys::cublasLtMatrixTransformDescSetAttribute(self.raw, attr.into(), value, size)
            },
            ptr::from_ref::<T>(value).cast(),
            size_of::<T>(),
        )
    }

    /// Returns a matrix transform descriptor attribute value.
    ///
    /// `T` must accept every bit pattern; enum attributes are read as integers and validated.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not match `T`.
    pub fn attribute<T: AnyBitPatternDeviceRepr>(
        &self,
        attr: MatrixTransformDescAttribute,
    ) -> Result<T> {
        let () = T::LAYOUT;
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            // SAFETY: `self.raw` is the live descriptor this value owns, and `read_attribute` passes `value` with its byte size and a live `written` location.
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
        ensure_exact_size(
            "matrix transform descriptor attribute",
            written,
            size_of::<T>(),
        )?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        Ok(unsafe { value.assume_init() })
    }

    /// Sets the pointer mode for alpha and beta scalars.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_pointer_mode(&mut self, pointer_mode: PointerMode) -> Result<()> {
        // SAFETY: `MatrixTransformDescAttribute::PointerMode` stores one `cublasLtPointerMode_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatrixTransformDescAttribute::PointerMode,
                &sys::cublasLtPointerMode_t::from(pointer_mode),
            )
        }
    }

    /// Sets the transpose mode applied to input matrix A.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_transpose_a(&mut self, operation: Operation) -> Result<()> {
        // SAFETY: `MatrixTransformDescAttribute::TransposeA` stores one `cublasOperation_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatrixTransformDescAttribute::TransposeA,
                &sys::cublasOperation_t::from(operation),
            )
        }
    }

    /// Sets the transpose mode applied to input matrix B.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_transpose_b(&mut self, operation: Operation) -> Result<()> {
        // SAFETY: `MatrixTransformDescAttribute::TransposeB` stores one `cublasOperation_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatrixTransformDescAttribute::TransposeB,
                &sys::cublasOperation_t::from(operation),
            )
        }
    }

    /// Returns the raw cuBLASLt matrix transform descriptor handle.
    ///
    /// The returned handle is borrowed and remains valid only while the descriptor is alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cublasLtMatrixTransformDesc_t {
        self.raw
    }

    /// Takes ownership of a raw cuBLASLt matrix transform descriptor handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a live cuBLASLt matrix transform descriptor.
    /// The returned value takes ownership and destroys it on drop; no other owner may destroy or keep using it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw` is null.
    pub unsafe fn from_raw(raw: sys::cublasLtMatrixTransformDesc_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Releases ownership and returns the raw cuBLASLt matrix transform descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> sys::cublasLtMatrixTransformDesc_t {
        let raw = self.raw;
        mem::forget(self);
        raw
    }
}

impl Drop for MatrixTransformDescriptor {
    fn drop(&mut self) {
        // SAFETY: this value owns `self.raw`, and `drop` destroys it exactly once.
        unsafe {
            if let Err(error) = try_ffi!(sys::cublasLtMatrixTransformDescDestroy(self.raw)) {
                report_cleanup_failure("destroy", "cublaslt matrix transform descriptor", &error);
            }
        }
    }
}

/// An owned cuBLASLt descriptor controlling floating-point emulation.
#[derive(Debug)]
pub struct EmulationDescriptor {
    raw: sys::cublasLtEmulationDesc_t,
}

// SAFETY: the descriptor is host memory owned by this value; cuBLASLt allows using and destroying it from any thread, one call at a time.
unsafe impl Send for EmulationDescriptor {}

impl EmulationDescriptor {
    /// Creates an emulation descriptor.
    ///
    /// The descriptor owns its cuBLASLt handle and destroys it when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot allocate the descriptor or if it does not return a valid handle.
    pub fn create() -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `cublasLtEmulationDescCreate` writes the new descriptor to `raw`, a live local, and takes its other arguments by value.
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
    /// # Safety
    ///
    /// `T` must match the selected attribute's native representation, including its validity and initialized-byte requirements.
    /// Retain pointer targets for every native use.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the attribute storage expected by cuBLASLt.
    pub unsafe fn set_attribute<T>(
        &mut self,
        attr: EmulationDescAttribute,
        value: &T,
    ) -> Result<()> {
        set_attribute(
            // SAFETY: `self.raw` is the live descriptor this value owns, and `set_attribute` passes `value` with its exact byte size.
            |value, size| unsafe {
                sys::cublasLtEmulationDescSetAttribute(self.raw, attr.into(), value, size)
            },
            ptr::from_ref::<T>(value).cast(),
            size_of::<T>(),
        )
    }

    /// Returns an emulation descriptor attribute value.
    ///
    /// `T` must accept every bit pattern; enum attributes are read as integers and validated.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not match `T`.
    pub fn attribute<T: AnyBitPatternDeviceRepr>(&self, attr: EmulationDescAttribute) -> Result<T> {
        let () = T::LAYOUT;
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            // SAFETY: `self.raw` is the live descriptor this value owns, and `read_attribute` passes `value` with its byte size and a live `written` location.
            |value, size, written| unsafe {
                sys::cublasLtEmulationDescGetAttribute(self.raw, attr.into(), value, size, written)
            },
            value.as_mut_ptr().cast(),
            size_of::<T>(),
            "emulation descriptor attribute",
        )?;
        ensure_exact_size("emulation descriptor attribute", written, size_of::<T>())?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        Ok(unsafe { value.assume_init() })
    }

    /// Sets the floating-point emulation strategy.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_strategy(&mut self, strategy: EmulationStrategy) -> Result<()> {
        // SAFETY: `EmulationDescAttribute::Strategy` stores one `cudaEmulationStrategy`, the type passed here.
        unsafe {
            self.set_attribute(
                EmulationDescAttribute::Strategy,
                &library_types::cudaEmulationStrategy::from(strategy),
            )
        }
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
        // SAFETY: `EmulationDescAttribute::SpecialValuesSupport` stores one `cudaEmulationSpecialValuesSupport`, the type passed here.
        unsafe { self.set_attribute(EmulationDescAttribute::SpecialValuesSupport, &support) }
    }

    /// Sets the mantissa bit control for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_mantissa_control(&mut self, control: EmulationMantissaControl) -> Result<()> {
        // SAFETY: `EmulationDescAttribute::FixedPointMantissaControl` stores one `cudaEmulationMantissaControl`, the type passed here.
        unsafe {
            self.set_attribute(
                EmulationDescAttribute::FixedPointMantissaControl,
                &library_types::cudaEmulationMantissaControl::from(control),
            )
        }
    }

    /// Sets the maximum mantissa bit count for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_max_mantissa_bit_count(&mut self, count: i32) -> Result<()> {
        // SAFETY: `EmulationDescAttribute::FixedPointMaxMantissaBitCount` stores one `i32`, the type passed here.
        unsafe {
            self.set_attribute(
                EmulationDescAttribute::FixedPointMaxMantissaBitCount,
                &count,
            )
        }
    }

    /// Sets the mantissa bit offset for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_mantissa_bit_offset(&mut self, offset: i32) -> Result<()> {
        // SAFETY: `EmulationDescAttribute::FixedPointMantissaBitOffset` stores one `i32`, the type passed here.
        unsafe { self.set_attribute(EmulationDescAttribute::FixedPointMantissaBitOffset, &offset) }
    }

    /// Returns the raw cuBLASLt emulation descriptor handle.
    ///
    /// The returned handle is borrowed and remains valid only while the descriptor is alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cublasLtEmulationDesc_t {
        self.raw
    }

    /// Takes ownership of a raw cuBLASLt emulation descriptor handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a live cuBLASLt emulation descriptor.
    /// The returned value takes ownership and destroys it on drop; no other owner may destroy or keep using it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw` is null.
    pub unsafe fn from_raw(raw: sys::cublasLtEmulationDesc_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Releases ownership and returns the raw cuBLASLt emulation descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> sys::cublasLtEmulationDesc_t {
        let raw = self.raw;
        mem::forget(self);
        raw
    }
}

impl Drop for EmulationDescriptor {
    fn drop(&mut self) {
        // SAFETY: this value owns `self.raw`, and `drop` destroys it exactly once.
        unsafe {
            if let Err(error) = try_ffi!(sys::cublasLtEmulationDescDestroy(self.raw)) {
                report_cleanup_failure("destroy", "cublaslt emulation descriptor", &error);
            }
        }
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_layout_sets_strided_batch_attributes() -> Result<()> {
        let mut layout = MatrixLayout::create(DataType::F32, 2, 3, 3)?;

        layout.set_strided_batch(4, 6)?;

        assert_eq!(layout.batch_mode()?, BatchMode::Strided);
        assert_eq!(layout.batch_count()?, 4);
        assert_eq!(layout.strided_batch_offset()?, 6);
        Ok(())
    }
}
