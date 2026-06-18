use std::{mem::ManuallyDrop, ptr};

use singe_cuda::data_type::{DataType, DataTypeLike};

use crate::{
    context::{Context, ContextRef},
    error::{Error, Result},
    sys, try_ffi,
    utility::to_i64_vec,
};

#[derive(Debug)]
pub struct TensorDescriptor {
    handle: sys::cutensorTensorDescriptor_t,
    context: ContextRef,
    shape: Vec<u64>,
    strides: Option<Vec<u64>>,
    data_type: DataType,
    alignment_requirement: u32,
}

impl TensorDescriptor {
    pub fn create_for<T: DataTypeLike>(
        ctx: &Context,
        shape: &[u64],
        alignment_requirement: u32,
    ) -> Result<Self> {
        Self::create(ctx, shape, T::data_type(), alignment_requirement)
    }

    pub fn create(
        ctx: &Context,
        shape: &[u64],
        data_type: DataType,
        alignment_requirement: u32,
    ) -> Result<Self> {
        Self::create_with_strides(ctx, shape, None, data_type, alignment_requirement)
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
        ctx: &Context,
        shape: &[u64],
        strides: Option<&[u64]>,
        data_type: DataType,
        alignment_requirement: u32,
    ) -> Result<Self> {
        ctx.bind()?;

        let rank = shape.len() as u32;
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
        unsafe {
            try_ffi!(sys::cutensorCreateTensorDescriptor(
                ctx.as_raw(),
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
            context: ctx.as_context_ref(),
            shape: shape.to_vec(),
            strides: strides_vec,
            data_type,
            alignment_requirement,
        })
    }

    pub fn create_with_strides_for<T: DataTypeLike>(
        ctx: &Context,
        shape: &[u64],
        strides: Option<&[u64]>,
        alignment_requirement: u32,
    ) -> Result<Self> {
        Self::create_with_strides(ctx, shape, strides, T::data_type(), alignment_requirement)
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn strides(&self) -> Option<&[u64]> {
        self.strides.as_deref()
    }

    pub fn rank(&self) -> u32 {
        self.shape.len() as u32
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn alignment_requirement(&self) -> u32 {
        self.alignment_requirement
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

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
    pub unsafe fn from_raw(
        handle: sys::cutensorTensorDescriptor_t,
        ctx: &Context,
        shape: Vec<u64>,
        strides: Option<Vec<u64>>,
        data_type: DataType,
        alignment_requirement: u32,
    ) -> Self {
        Self {
            handle,
            context: ctx.as_context_ref(),
            shape,
            strides,
            data_type,
            alignment_requirement,
        }
    }

    /// Consumes the descriptor and returns the raw cuTENSOR tensor descriptor
    /// handle without destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with cuTENSOR.
    pub fn into_raw(self) -> sys::cutensorTensorDescriptor_t {
        let descriptor = ManuallyDrop::new(self);
        descriptor.handle
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        if let Err(err) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cutensor context before destroying tensor descriptor: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorDestroyTensorDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensor tensor descriptor: {err}");
            }
        }
    }
}

#[derive(Debug)]
pub struct BlockSparseTensorDescriptor {
    handle: sys::cutensorBlockSparseTensorDescriptor_t,
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
        ctx: &Context,
        section_count_per_mode: &[u32],
        section_extents: &[u64],
        non_zero_block_count: u64,
        non_zero_coordinates: &[i32],
        block_strides: Option<&[u64]>,
        data_type: DataType,
    ) -> Result<Self> {
        ctx.bind()?;

        let mode_count = section_count_per_mode.len() as u32;
        let expected_section_extents = section_count_per_mode
            .iter()
            .try_fold(0usize, |sum, &value| sum.checked_add(value as usize))
            .ok_or(Error::OutOfRange {
                name: "section_extents".into(),
            })?;
        if section_extents.len() != expected_section_extents {
            return Err(Error::LengthMismatch {
                name: "section_extents".into(),
                expected: expected_section_extents,
                actual: section_extents.len(),
            });
        }

        let expected_coordinates =
            mode_count
                .checked_mul(non_zero_block_count as u32)
                .ok_or(Error::OutOfRange {
                    name: "non_zero_coordinates".into(),
                })?;
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
        unsafe {
            try_ffi!(sys::cutensorCreateBlockSparseTensorDescriptor(
                ctx.as_raw(),
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
            context: ctx.as_context_ref(),
            mode_count,
            non_zero_block_count,
            data_type,
        })
    }

    pub fn mode_count(&self) -> u32 {
        self.mode_count
    }

    pub fn non_zero_block_count(&self) -> u64 {
        self.non_zero_block_count
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

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
    pub unsafe fn from_raw(
        handle: sys::cutensorBlockSparseTensorDescriptor_t,
        ctx: &Context,
        mode_count: u32,
        non_zero_block_count: u64,
        data_type: DataType,
    ) -> Self {
        Self {
            handle,
            context: ctx.as_context_ref(),
            mode_count,
            non_zero_block_count,
            data_type,
        }
    }

    /// Consumes the descriptor and returns the raw cuTENSOR block-sparse tensor
    /// descriptor handle without destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with cuTENSOR.
    pub fn into_raw(self) -> sys::cutensorBlockSparseTensorDescriptor_t {
        let descriptor = ManuallyDrop::new(self);
        descriptor.handle
    }
}

impl Drop for BlockSparseTensorDescriptor {
    fn drop(&mut self) {
        if let Err(err) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!(
                "failed to bind cutensor context before destroying block sparse tensor descriptor: {err}"
            );
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorDestroyBlockSparseTensorDescriptor(self.handle))
            {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensor block sparse tensor descriptor: {err}");
            }
        }
    }
}
