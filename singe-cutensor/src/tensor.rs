use std::{mem::ManuallyDrop, ptr};

use singe_cuda::data_type::{DataType, DataTypeLike};

use crate::{
    context::{Context, ContextRef},
    error::{Error, Result},
    sys, try_ffi,
    utility::{to_i64_vec, to_u32},
};

/// Owns a dense tensor layout descriptor and retains its cuTENSOR context.
///
/// The descriptor describes sample storage without allocating tensor data.
#[derive(Debug)]
pub struct TensorDescriptor {
    handle: sys::cutensorTensorDescriptor_t,
    context: ContextRef,
    shape: Vec<u64>,
    strides: Option<Vec<u64>>,
    data_type: DataType,
    alignment_requirement: u32,
}

/// Transferred native ownership with its original metadata and retained library context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`TensorDescriptor::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawTensorDescriptor {
    /// Transferred native descriptor handle; destroy it before releasing the retained context.
    pub handle: sys::cutensorTensorDescriptor_t,
    context: ContextRef,
    shape: Vec<u64>,
    strides: Option<Vec<u64>>,
    data_type: DataType,
    alignment_requirement: u32,
}

impl TensorDescriptor {
    /// Creates a packed tensor descriptor using the native data type associated with `T`.
    ///
    /// Extents must be positive. Strides are measured in elements of the base data type; `None` selects packed storage with strides increasing from left to right. Alignment is a byte requirement on data pointers supplied during execution.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, rank, extents, or strides cannot be represented by the native API, stride and shape counts differ, cuTENSOR rejects the layout, type, or alignment, or the native descriptor is null.
    pub fn create_for<T: DataTypeLike>(
        context: &Context,
        shape: &[u64],
        alignment_requirement: u32,
    ) -> Result<Self> {
        Self::create(context, shape, T::data_type(), alignment_requirement)
    }

    /// Creates a dense tensor descriptor using packed generalized column-major storage.
    ///
    /// Extents must be positive. Strides are measured in elements of the base data type; `None` selects packed storage with strides increasing from left to right. Alignment is a byte requirement on data pointers supplied during execution.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, rank, extents, or strides cannot be represented by the native API, stride and shape counts differ, cuTENSOR rejects the layout, type, or alignment, or the native descriptor is null.
    pub fn create(
        context: &Context,
        shape: &[u64],
        data_type: DataType,
        alignment_requirement: u32,
    ) -> Result<Self> {
        Self::create_with_strides(context, shape, None, data_type, alignment_requirement)
    }

    /// Creates a tensor descriptor.
    ///
    /// This allocates a small amount of host memory.
    ///
    /// The descriptor frees the associated cuTENSOR resources when dropped.
    ///
    /// This non-blocking call is thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, `shape` and `strides`
    /// have different ranks, an extent or stride cannot be represented for
    /// cuTENSOR, the data type or descriptor layout is unsupported, or cuTENSOR
    /// returns a null descriptor handle.
    pub fn create_with_strides(
        context: &Context,
        shape: &[u64],
        strides: Option<&[u64]>,
        data_type: DataType,
        alignment_requirement: u32,
    ) -> Result<Self> {
        context.bind()?;

        let rank = to_u32(shape.len(), "tensor rank")?;
        if let Some(strides) = strides
            && shape.len() != strides.len()
        {
            return Err(Error::TensorStrideMismatch {
                rank,
                stride_length: strides.len(),
            });
        }

        let extent = to_i64_vec(shape, "shape")?;
        let strides_vec = strides.map(<[u64]>::to_vec);
        let stride_i64 = strides_vec
            .as_ref()
            .map(|values| to_i64_vec(values, "strides"))
            .transpose()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cutensorCreateTensorDescriptor` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorCreateTensorDescriptor(
                context.as_raw(),
                &raw mut handle,
                rank,
                extent.as_ptr(),
                stride_i64.as_ref().map_or(ptr::null(), Vec::as_ptr),
                data_type.into(),
                alignment_requirement,
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            shape: shape.to_vec(),
            strides: strides_vec,
            data_type,
            alignment_requirement,
        })
    }

    /// Creates a tensor descriptor for `T` with optional element strides.
    ///
    /// Extents must be positive. Strides are measured in elements of the base data type; `None` selects packed storage with strides increasing from left to right. Alignment is a byte requirement on data pointers supplied during execution.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, rank, extents, or strides cannot be represented by the native API, stride and shape counts differ, cuTENSOR rejects the layout, type, or alignment, or the native descriptor is null.
    pub fn create_with_strides_for<T: DataTypeLike>(
        context: &Context,
        shape: &[u64],
        strides: Option<&[u64]>,
        alignment_requirement: u32,
    ) -> Result<Self> {
        Self::create_with_strides(
            context,
            shape,
            strides,
            T::data_type(),
            alignment_requirement,
        )
    }

    /// Returns the extent of each tensor mode.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns explicit element strides, or `None` for packed generalized column-major storage.
    #[must_use]
    pub fn strides(&self) -> Option<&[u64]> {
        self.strides.as_deref()
    }

    /// Returns the number of tensor modes.
    ///
    /// # Panics
    ///
    /// Panics if the stored rank no longer fits `u32`. Checked construction excludes this; unsafe raw adoption must preserve the native rank invariant.
    #[must_use]
    pub fn rank(&self) -> u32 {
        u32::try_from(self.shape.len()).expect("tensor rank is validated on construction")
    }

    /// Returns the native tensor sample type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the required data-pointer alignment in bytes.
    #[must_use]
    pub fn alignment_requirement(&self) -> u32 {
        self.alignment_requirement
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

    /// Returns the borrowed native dense tensor descriptor.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorTensorDescriptor_t {
        self.handle
    }

    /// Wraps an existing cuTENSOR tensor descriptor and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cutensorTensorDescriptor_t` whose rank,
    /// extents, strides, data type, and alignment requirement match the
    /// metadata supplied to this function. Ownership of `handle` is transferred
    /// to the returned descriptor, and the handle must not be destroyed
    /// elsewhere after calling this function.
    pub unsafe fn from_raw_parts(
        handle: sys::cutensorTensorDescriptor_t,
        context: &Context,
        shape: Vec<u64>,
        strides: Option<Vec<u64>>,
        data_type: DataType,
        alignment_requirement: u32,
    ) -> Self {
        Self {
            handle,
            context: context.as_context_ref(),
            shape,
            strides,
            data_type,
            alignment_requirement,
        }
    }

    /// Consumes the descriptor and returns the raw cuTENSOR tensor descriptor
    /// handle without destroying it.
    ///
    /// The returned record retains the native library owner and descriptor metadata.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawTensorDescriptor {
        let owner = ManuallyDrop::new(self);
        // SAFETY: move each field exactly once while suppressing native destruction.
        unsafe {
            RawTensorDescriptor {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
                shape: ptr::read(&raw const owner.shape),
                strides: ptr::read(&raw const owner.strides),
                data_type: owner.data_type,
                alignment_requirement: owner.alignment_requirement,
            }
        }
    }

    /// Reclaims a transferred native resource with its original metadata and library owner.
    ///
    /// # Safety
    ///
    /// `handle` must remain valid and uniquely owned with its original descriptor metadata unchanged.
    /// Complete all borrowed queued accesses before replacing or releasing their resources.
    #[must_use]
    pub unsafe fn from_raw(raw: RawTensorDescriptor) -> Self {
        Self {
            handle: raw.handle,
            context: raw.context,
            shape: raw.shape,
            strides: raw.strides,
            data_type: raw.data_type,
            alignment_requirement: raw.alignment_requirement,
        }
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!(
                "failed to bind cutensor context before destroying tensor descriptor: {error}"
            );
        }

        // SAFETY: `cutensorDestroyTensorDescriptor` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorDestroyTensorDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensor tensor descriptor: {error}");
            }
        }
    }
}

/// Owns a block-sparse tensor layout and retains its cuTENSOR context.
#[derive(Debug)]
pub struct BlockSparseTensorDescriptor {
    handle: sys::cutensorBlockSparseTensorDescriptor_t,
    context: ContextRef,
    mode_count: u32,
    non_zero_block_count: u64,
    data_type: DataType,
}

/// Transferred native ownership with its original metadata and retained library context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`BlockSparseTensorDescriptor::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawBlockSparseTensorDescriptor {
    /// Transferred native descriptor handle; destroy it before releasing the retained context.
    pub handle: sys::cutensorBlockSparseTensorDescriptor_t,
    context: ContextRef,
    mode_count: u32,
    non_zero_block_count: u64,
    data_type: DataType,
}

impl BlockSparseTensorDescriptor {
    /// Creates a block-sparse tensor descriptor.
    ///
    /// A block-sparse tensor descriptor fully specifies the data layout of a block-sparse tensor (currently limited to up to 8 modes).
    ///
    /// Consider an example for a block-sparse tensor of order 2, namely a block-sparse matrix `A`.
    /// Its first mode (`rows`) is subdivided into 5 sections (with extents 4, 2, 3, 4, 5, respectively), and its second mode (`columns`) is subdivided into 3 sections (with extents 2, 3, 7).
    /// The matrix has 8 non-zero blocks:
    ///
    /// | 2 columns | 3 columns | 7 columns |
    /// | --- | --- | --- |
    /// | 4 x 2 | | 4 x 7 |
    /// | | 2 x 3 | |
    /// | | 3 x 3 | 3 x 7 |
    /// | 4 x 2 | | |
    /// | | 5 x 3 | 5 x 7 |
    ///
    /// Each subsection has the same extent across its entire mode: every block
    /// within the same section of a mode has the same extent.
    /// For example, in the table above every block on the right has 7 columns,
    /// and all blocks on the bottom have 5 rows.
    ///
    /// Only non-zero blocks are stored; zero blocks are left blank in the
    /// representation above.
    ///
    /// For example:
    ///
    /// * strides of block #0: 5, 1, 10, 20.
    ///   Sorted strides would be: 1, 5, 10, 20.
    ///   The permutation to sort the strides in this case is to swap the first two elements.
    /// * strides of block #1: 10, 1, 30, 60.
    ///   Applying the permutation would result in: 1, 10, 30, 60.
    ///   The result is sorted in ascending order, which is allowed.
    /// * strides of block #2: 1, 5, 50, 100.
    ///   Applying permutation would result in: 5, 1, 50, 100.
    ///   The result is *not* sorted in ascending order, this is *not* allowed.
    ///
    /// This non-blocking call is thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, the section extent,
    /// coordinate, or stride slices do not match the descriptor dimensions, a
    /// value cannot be represented for cuTENSOR, cuTENSOR rejects the
    /// descriptor, or cuTENSOR returns a null descriptor handle.
    pub fn create(
        context: &Context,
        section_count_per_mode: &[u32],
        section_extents: &[u64],
        non_zero_block_count: u64,
        non_zero_coordinates: &[i32],
        block_strides: Option<&[u64]>,
        data_type: DataType,
    ) -> Result<Self> {
        context.bind()?;

        let mode_count = to_u32(section_count_per_mode.len(), "block sparse mode count")?;
        let expected_section_extents = section_count_per_mode
            .iter()
            .try_fold(0usize, |sum, &value| sum.checked_add(value as usize))
            .ok_or_else(|| Error::out_of_range("section_extents"))?;
        if section_extents.len() != expected_section_extents {
            return Err(Error::LengthMismatch {
                name: "section_extents".into(),
                expected: expected_section_extents,
                actual: section_extents.len(),
            });
        }

        let expected_coordinates = mode_count
            .checked_mul(to_u32(non_zero_block_count, "nonzero block count")?)
            .ok_or_else(|| Error::out_of_range("non_zero_coordinates"))?;
        if non_zero_coordinates.len() != expected_coordinates as usize {
            return Err(Error::LengthMismatch {
                name: "non_zero_coordinates".into(),
                expected: expected_coordinates as usize,
                actual: non_zero_coordinates.len(),
            });
        }

        if let Some(block_strides) = block_strides
            && block_strides.len() != expected_coordinates as usize
        {
            return Err(Error::TensorStrideMismatch {
                rank: expected_coordinates,
                stride_length: block_strides.len(),
            });
        }

        let num_sections_per_mode = section_count_per_mode.to_vec();
        let section_extents = to_i64_vec(section_extents, "section_extents")?;
        let block_strides = block_strides
            .map(|block_strides| to_i64_vec(block_strides, "block_strides"))
            .transpose()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cutensorCreateBlockSparseTensorDescriptor` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorCreateBlockSparseTensorDescriptor(
                context.as_raw(),
                &raw mut handle,
                mode_count,
                non_zero_block_count,
                num_sections_per_mode.as_ptr(),
                section_extents.as_ptr(),
                non_zero_coordinates.as_ptr(),
                block_strides.as_ref().map_or(ptr::null(), Vec::as_ptr),
                data_type.into(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            mode_count,
            non_zero_block_count,
            data_type,
        })
    }

    /// Returns the number of block-sparse tensor modes.
    #[must_use]
    pub fn mode_count(&self) -> u32 {
        self.mode_count
    }

    /// Returns the number of explicitly stored nonzero blocks.
    #[must_use]
    pub fn non_zero_block_count(&self) -> u64 {
        self.non_zero_block_count
    }

    /// Returns the native sample data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

    /// Returns the borrowed native block-sparse tensor descriptor.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorBlockSparseTensorDescriptor_t {
        self.handle
    }

    /// Wraps an existing cuTENSOR block-sparse tensor descriptor and takes
    /// ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cutensorBlockSparseTensorDescriptor_t` whose
    /// mode count, non-zero block count, and data type match the metadata
    /// supplied to this function. Ownership of `handle` is transferred to the
    /// returned descriptor, and the handle must not be destroyed elsewhere after
    /// calling this function.
    pub unsafe fn from_raw_parts(
        handle: sys::cutensorBlockSparseTensorDescriptor_t,
        context: &Context,
        mode_count: u32,
        non_zero_block_count: u64,
        data_type: DataType,
    ) -> Self {
        Self {
            handle,
            context: context.as_context_ref(),
            mode_count,
            non_zero_block_count,
            data_type,
        }
    }

    /// Consumes the descriptor and returns the raw cuTENSOR block-sparse tensor
    /// descriptor handle without destroying it.
    ///
    /// The returned record retains the native library owner and descriptor metadata.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawBlockSparseTensorDescriptor {
        let owner = ManuallyDrop::new(self);
        // SAFETY: move each field exactly once while suppressing native destruction.
        unsafe {
            RawBlockSparseTensorDescriptor {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
                mode_count: owner.mode_count,
                non_zero_block_count: owner.non_zero_block_count,
                data_type: owner.data_type,
            }
        }
    }

    /// Reclaims a transferred native resource with its original metadata and library owner.
    ///
    /// # Safety
    ///
    /// `handle` must remain valid and uniquely owned with its original descriptor metadata unchanged.
    /// Complete all borrowed queued accesses before replacing or releasing their resources.
    #[must_use]
    pub unsafe fn from_raw(raw: RawBlockSparseTensorDescriptor) -> Self {
        Self {
            handle: raw.handle,
            context: raw.context,
            mode_count: raw.mode_count,
            non_zero_block_count: raw.non_zero_block_count,
            data_type: raw.data_type,
        }
    }
}

impl Drop for BlockSparseTensorDescriptor {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!(
                "failed to bind cutensor context before destroying block sparse tensor descriptor: {error}"
            );
        }

        // SAFETY: `cutensorDestroyBlockSparseTensorDescriptor` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) =
                try_ffi!(sys::cutensorDestroyBlockSparseTensorDescriptor(self.handle))
            {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensor block sparse tensor descriptor: {error}");
            }
        }
    }
}
