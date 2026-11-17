//! Legacy cuDNN tensor descriptors and tensor fill.

use std::{marker::PhantomData, mem::replace, ptr};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_display, report_cleanup_failure};
use singe_cuda::view::DeviceOutput;
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    data_type::{DataTypeVectorization, TensorDataType, TensorElement},
    error::{Error, Result},
    try_ffi,
    utility::{check_range, checked_element_count, to_i32, to_usize},
};

singe_core::native_enum! {
    /// Predefined tensor layout used by [`TensorDescriptor`] constructors.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum TensorFormat: sys::cudnnTensorFormat_t {
        /// This tensor format specifies that the data is laid out in the following order: batch size, feature maps, rows, columns.
        /// The strides are implicitly defined in such a way that the data are contiguous in memory with no padding between images, feature maps, rows, and columns; the columns are the inner dimension and the images are the outermost dimension.
        Nchw = CUDNN_TENSOR_NCHW,
        /// This tensor format specifies that the data is laid out in the following order: batch size, rows, columns, feature maps.
        /// The strides are implicitly defined in such a way that the data are contiguous in memory with no padding between images, rows, columns, and feature maps; the feature maps are the inner dimension and the images are the outermost dimension.
        Nhwc = CUDNN_TENSOR_NHWC,
        /// This tensor format specifies that the data is laid out in the following order: batch size, feature maps, rows, columns.
        /// However, each element of the tensor is a vector of multiple feature maps.
        /// The length of the vector is carried by the data type of the tensor.
        /// The strides are implicitly defined in such a way that the data are contiguous in memory with no padding between images, feature maps, rows, and columns; the columns are the inner dimension and the images are the outermost dimension.
        /// This format is only supported with tensor data types [`TensorDataType::Int8x4`], [`TensorDataType::Int8x32`], and [`TensorDataType::Uint8x4`].
        ///
        /// [`TensorFormat::NchwVectC`] can also be interpreted as N x (C/32) x H x W x 32 for `INT8x32`, or N x (C/4) x H x W x 4 for `INT8x4`.
        /// In this layout, each W position stores a vector of 4 or 32 C values.
        NchwVectC = CUDNN_TENSOR_NCHW_VECT_C,
    }
}

impl_enum_display!(TensorFormat, {
    Self::Nchw => "CUDNN_TENSOR_NCHW",
    Self::Nhwc => "CUDNN_TENSOR_NHWC",
    Self::NchwVectC => "CUDNN_TENSOR_NCHW_VECT_C",
});

/// Context-independent legacy tensor descriptor with immutable configured attributes.
/// Descriptors can move between threads when the element type supports it, but are not `Sync`.
/// Concurrent cuDNN operations require independent context handles.
#[derive(Debug)]
pub struct TensorDescriptor<T> {
    handle: sys::cudnnTensorDescriptor_t,
    data_type: TensorDataType,
    dimensions: Vec<i32>,
    strides: Vec<i32>,
    _t: PhantomData<T>,
}

// SAFETY: the descriptor has no CUDA-context or thread affinity, and ownership transfers with the handle.
unsafe impl<T: Send> Send for TensorDescriptor<T> {}

impl<T: TensorElement> TensorDescriptor<T> {
    fn checked_contiguous_strides(dimensions: &[i32]) -> Result<Vec<i32>> {
        let mut strides = vec![0; dimensions.len()];
        let mut stride = 1i32;
        for (index, dimension) in dimensions.iter().enumerate().rev() {
            if *dimension <= 0 {
                return Err(Error::InvalidDataShape);
            }

            strides[index] = stride;
            if index != 0 {
                stride = stride
                    .checked_mul(*dimension)
                    .ok_or_else(|| Error::out_of_range("tensor strides"))?;
            }
        }
        Ok(strides)
    }

    /// Creates a legacy tensor descriptor for `T` with explicit native element strides.
    ///
    /// Dimensions and strides must have equal nonzero lengths.
    /// cuDNN validates the layout; this allocates only a descriptor, not tensor storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the rank or layout is invalid, a checked native integer conversion fails, cuDNN cannot create, configure, or query the descriptor, the native handle is null, or queried layout metadata or encoding is invalid.
    pub fn create_strided(dimensions: &[i32], strides: &[i32]) -> Result<Self> {
        if dimensions.is_empty() {
            return Err(Error::InvalidDataShape);
        }

        if dimensions.len() != strides.len() {
            return Err(Error::LengthMismatch {
                name: "tensor strides".into(),
                expected: dimensions.len(),
                actual: strides.len(),
            });
        }

        let rank = to_i32(dimensions.len(), "tensor rank")?;
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a writable local that receives the new descriptor.
        unsafe {
            try_ffi!(sys::cudnnCreateTensorDescriptor(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let mut descriptor = Self {
            handle,
            data_type: T::data_type().into(),
            dimensions: Vec::new(),
            strides: Vec::new(),
            _t: PhantomData,
        };
        // SAFETY: `descriptor` owns the new handle, and `dimensions` and `strides` were checked to hold `rank` elements each.
        unsafe {
            try_ffi!(sys::cudnnSetTensorNdDescriptor(
                descriptor.handle,
                T::data_type().into(),
                rank,
                dimensions.as_ptr(),
                strides.as_ptr(),
            ))?;
        }

        descriptor.read_layout()?;
        Ok(descriptor)
    }

    /// Creates a legacy tensor descriptor for `T` using packed row-major strides.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are empty or nonpositive, stride arithmetic exceeds `i32`, or explicit descriptor creation or metadata querying fails.
    pub fn create_contiguous(dimensions: &[i32]) -> Result<Self> {
        if dimensions.is_empty() {
            return Err(Error::InvalidDataShape);
        }

        let strides = Self::checked_contiguous_strides(dimensions)?;
        Self::create_strided(dimensions, &strides)
    }

    /// Creates a legacy channel-vectorized descriptor for `T` and vectorization `V`.
    ///
    /// Uses [`TensorFormat::NchwVectC`]; axis one is the channel dimension.
    /// This creates a descriptor without allocating tensor data.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than two dimensions are supplied, the type/vector encoding is unsupported, the channel extent is zero or not divisible by the vector width, or native rank conversion, descriptor creation, configuration, or metadata querying fails.
    ///
    /// # Panics
    ///
    /// A selected vector width must be nonzero and fit `i32`.
    /// A custom scalar/vector implementation violating this requirement can panic.
    pub fn create_vectorized<V: DataTypeVectorization>(dimensions: &[i32]) -> Result<Self> {
        if dimensions.len() < 2 {
            return Err(Error::InvalidDataShape);
        }

        let data_type = T::tensor_data_type::<V>().ok_or(Error::UnsupportedDataType)?;

        if dimensions[1] == 0
            || dimensions[1]
                % i32::try_from(V::WIDTH).expect("vector width fits the native tensor interface")
                != 0
        {
            return Err(Error::InvalidDataShape);
        }

        let native_data_type = data_type.into();
        let rank = to_i32(dimensions.len(), "tensor rank")?;
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a writable local that receives the new descriptor.
        unsafe {
            try_ffi!(sys::cudnnCreateTensorDescriptor(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let mut descriptor = Self {
            handle,
            data_type,
            dimensions: Vec::new(),
            strides: Vec::new(),
            _t: PhantomData,
        };
        // SAFETY: `descriptor` owns the new handle, and `dimensions` holds the `rank` elements the call reads.
        unsafe {
            try_ffi!(sys::cudnnSetTensorNdDescriptorEx(
                descriptor.handle,
                TensorFormat::NchwVectC.into(),
                native_data_type,
                rank,
                dimensions.as_ptr(),
            ))?;
        }

        descriptor.read_layout()?;
        Ok(descriptor)
    }
}

impl<T> TensorDescriptor<T> {
    fn read_layout(&mut self) -> Result<()> {
        // Receive integer bits before constructing a Rust enum with a closed set of discriminants.
        let mut data_type = 0_u32;
        let mut rank = 0;
        let mut dimensions = [0; sys::CUDNN_DIM_MAX as usize];
        let mut strides = [0; sys::CUDNN_DIM_MAX as usize];
        // SAFETY: `self.handle` is the owned descriptor, the scalar outputs are writable locals, and both arrays hold the `CUDNN_DIM_MAX` elements requested.
        unsafe {
            try_ffi!(sys::cudnnGetTensorNdDescriptor(
                self.handle,
                sys::CUDNN_DIM_MAX.cast_signed(),
                (&raw mut data_type).cast::<sys::cudnnDataType_t>(),
                &raw mut rank,
                dimensions.as_mut_ptr(),
                strides.as_mut_ptr(),
            ))?;
        }

        let rank = to_usize(rank, "tensor rank")?;
        if rank == 0 || rank > dimensions.len() {
            return Err(Error::InvalidDataShape);
        }

        self.data_type = sys::cudnnDataType_t(data_type).try_into()?;
        self.dimensions = dimensions[..rank].to_vec();
        self.strides = strides[..rank].to_vec();
        Ok(())
    }

    /// Native storage encoding, including vectorized INT8 and UINT8 formats.
    /// Retained from construction or raw-handle adoption, so this getter makes no native call.
    #[must_use]
    pub fn data_type(&self) -> TensorDataType {
        self.data_type
    }

    /// Native descriptor dimensions, including the conventions of its vectorized encoding.
    #[must_use]
    pub fn dimensions(&self) -> &[i32] {
        &self.dimensions
    }

    /// Native descriptor strides, queried after layout construction or raw-handle adoption.
    #[must_use]
    pub fn strides(&self) -> &[i32] {
        &self.strides
    }

    /// Returns the number of cached native descriptor dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the product of the cached native descriptor dimensions, excluding stride gaps.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is nonpositive or cannot be represented as `usize`, or the dimension product overflows `usize`.
    pub fn element_count(&self) -> Result<usize> {
        checked_element_count(&self.dimensions, "tensor shape")
    }

    /// Returns the byte length of the tensor in memory for this descriptor.
    /// Use this to determine the amount of GPU memory needed to hold the tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN cannot report the tensor size or if the value cannot be represented as `usize`.
    pub fn byte_len(&self) -> Result<usize> {
        let mut byte_len = 0;
        // SAFETY: `self.handle` is the owned descriptor and `byte_len` is a writable local.
        unsafe {
            try_ffi!(sys::cudnnGetTensorSizeInBytes(
                self.handle,
                &raw mut byte_len
            ))?;
        }
        to_usize(byte_len, "tensor byte length")
    }

    /// Returns the raw cuDNN tensor descriptor.
    ///
    /// The returned descriptor is borrowed and remains valid only while this wrapper is alive.
    /// Its attributes must remain unchanged while the wrapper owns it.
    #[must_use]
    pub fn as_raw(&self) -> sys::cudnnTensorDescriptor_t {
        self.handle
    }

    /// Takes ownership of a raw cuDNN tensor descriptor handle.
    /// Reads storage encoding, dimensions and strides from the initialized native descriptor.
    /// A failed metadata query destroys the transferred handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid [`sys::cudnnTensorDescriptor_t`] created by cuDNN.
    /// The returned wrapper takes ownership and destroys the descriptor on drop; no other owner may destroy or keep using it.
    /// The descriptor's scalar encoding must match `T`, including when its storage is vectorized.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null, cuDNN cannot query its metadata, or the returned encoding or rank is invalid or cannot be represented by the wrapper.
    pub unsafe fn from_raw(handle: sys::cudnnTensorDescriptor_t) -> Result<Self>
    where
        T: TensorElement,
    {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let mut descriptor = Self {
            handle,
            data_type: T::data_type().into(),
            dimensions: Vec::new(),
            strides: Vec::new(),
            _t: PhantomData,
        };
        descriptor.read_layout()?;
        Ok(descriptor)
    }

    /// Releases ownership and returns the raw cuDNN tensor descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(mut self) -> sys::cudnnTensorDescriptor_t {
        replace(&mut self.handle, ptr::null_mut())
    }
}

impl<T> Drop for TensorDescriptor<T> {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        // SAFETY: `self.handle` is non-null and owned by this wrapper, and nothing uses it after drop.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudnnDestroyTensorDescriptor(self.handle)) {
                report_cleanup_failure("destroy", "cudnn tensor descriptor", &error);
            }
        }
    }
}

/// Enqueues setting all elements of a tensor to a given value on the handle's stream.
///
/// # Safety
///
/// Retain the context, descriptor and destination through completion, failure and replay.
/// Order allocation readiness before use and keep writes exclusive.
/// Only descriptor elements are initialized; padding and tails remain unchanged.
///
/// # Errors
///
/// Returns an error if the context, tensor descriptor, tensor memory, or scalar value is invalid, if the operation fails to launch on the GPU, or if cuDNN does not support the provided configuration.
pub unsafe fn set_tensor_async<T: TensorElement>(
    context: &Context,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    value: &T,
) -> Result<()> {
    context.bind()?;
    check_range!(
        "set tensor storage span",
        y.region().byte_len() >= y_desc.byte_len()?
    )?;
    context.prepare_regions(&[y.region()])?;

    // SAFETY: `context` is bound, `y` was checked to span `y_desc`'s bytes and its producers are ordered, and `value` is a live `T`.
    // The caller retains `y` and keeps its writes exclusive through completion, per `# Safety`.
    unsafe {
        try_ffi!(sys::cudnnSetTensor(
            context.as_raw(),
            y_desc.as_raw(),
            y.as_mut_ptr().cast(),
            ptr::from_ref::<T>(value).cast(),
        ))?;
    }
    Ok(())
}

/// Sets all elements of a tensor to a given value and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`set_tensor_async`] fails, or the wait fails.
pub fn set_tensor<T: TensorElement>(
    context: &Context,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    value: &T,
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: the operation only writes `y`, and the borrows outlive the wait in `sync_scope`.
        unsafe { set_tensor_async(context, y_desc, y, value) }
    })
}
