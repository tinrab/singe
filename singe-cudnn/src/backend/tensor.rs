mod problem;

use std::{
    borrow::{Borrow, Cow},
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    mem::replace,
    ptr,
    result::Result as StdResult,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicI64, Ordering},
    },
};

use bomboni_serde::{self as serde_helpers, as_string};
use num_enum::{IntoPrimitive, TryFromPrimitive};
pub use problem::TensorProblemKey;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as DeError};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::{
    error::Error as CudaError,
    graph::GraphHostBuffer,
    types::{f4e2m1, f8e4m3, f8e5m2, f8ue8m0},
    view::DeviceOutput,
};
use singe_cudnn_sys as sys;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
    },
    context::Context,
    data_type::{DataType, DataTypeLike, DataTypeVectorization, TensorDataType},
    error::{Error, Result},
    frontend::error::Error as FrontendError,
    scalar::{ScalarStorage, ScalarValue},
    try_ffi,
    utility::{check_range, to_i32, to_i64, to_usize},
};

/// Predefined tensor layout used by [`TensorDescriptor`] constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum TensorFormat {
    /// This tensor format specifies that the data is laid out in the following order: batch size, feature maps, rows, columns.
    /// The strides are implicitly defined in such a way that the data are contiguous in memory with no padding between images, feature maps, rows, and columns; the columns are the inner dimension and the images are the outermost dimension.
    Nchw = sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW as _,
    /// This tensor format specifies that the data is laid out in the following order: batch size, rows, columns, feature maps.
    /// The strides are implicitly defined in such a way that the data are contiguous in memory with no padding between images, rows, columns, and feature maps; the feature maps are the inner dimension and the images are the outermost dimension.
    Nhwc = sys::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC as _,
    /// This tensor format specifies that the data is laid out in the following order: batch size, feature maps, rows, columns.
    /// However, each element of the tensor is a vector of multiple feature maps.
    /// The length of the vector is carried by the data type of the tensor.
    /// The strides are implicitly defined in such a way that the data are contiguous in memory with no padding between images, feature maps, rows, and columns; the columns are the inner dimension and the images are the outermost dimension.
    /// This format is only supported with tensor data types [`TensorDataType::Int8x4`], [`TensorDataType::Int8x32`], and [`TensorDataType::Uint8x4`].
    ///
    /// [`TensorFormat::NchwVectC`] can also be interpreted as N x (C/32) x H x W x 32
    /// for `INT8x32`, or N x (C/4) x H x W x 4 for `INT8x4`.
    /// In this layout, each W position stores a vector of 4 or 32 C values.
    NchwVectC = sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW_VECT_C as _,
}

impl_enum_conversion!(sys::cudnnTensorFormat_t, TensorFormat);

impl_enum_display!(TensorFormat, {
    Self::Nchw => "CUDNN_TENSOR_NCHW",
    Self::Nhwc => "CUDNN_TENSOR_NHWC",
    Self::NchwVectC => "CUDNN_TENSOR_NCHW_VECT_C",
});

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Tensor reordering property of a tensor descriptor.
/// This property is available on [`BackendDescriptorType::Tensor`] through
/// [`BackendDescriptor`] attribute methods.
#[repr(u32)]
#[non_exhaustive]
pub enum BackendTensorReordering {
    ///< No tensor reordering applied. @since cuDNN 9.0.0
    None = sys::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_NONE as _,
    ///< INT8 data reordered into 32-element vectors. @since cuDNN 9.0.0
    Int8x32 = sys::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_INT8x32 as _,
    ///< FP16 data reordered into 16-element vectors. @since cuDNN 9.0.0
    F16x16 = sys::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16 as _,
    ///< FP8 data reordered into 128x4 blocks. @since cuDNN 9.7.0
    F8_128x4 = sys::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F8_128x4 as _,
}

impl_enum_conversion!(sys::cudnnBackendTensorReordering_t, BackendTensorReordering);

impl_enum_display!(BackendTensorReordering, {
    Self::None => "CUDNN_TENSOR_REORDERING_NONE",
    Self::Int8x32 => "CUDNN_TENSOR_REORDERING_INT8x32",
    Self::F16x16 => "CUDNN_TENSOR_REORDERING_F16x16",
    Self::F8_128x4 => "CUDNN_TENSOR_REORDERING_F8_128x4",
});

/// Context-independent legacy tensor descriptor with immutable configured attributes.
/// Descriptors can move between threads or be shared when the element type supports it.
/// Concurrent cuDNN operations require independent context handles.
#[derive(Debug)]
#[deprecated]
pub struct TensorDescriptor<T> {
    handle: sys::cudnnTensorDescriptor_t,
    data_type: TensorDataType,
    dimensions: Vec<i32>,
    strides: Vec<i32>,
    _t: PhantomData<T>,
}

// SAFETY: the descriptor has no CUDA-context or thread affinity, and ownership transfers with the handle.
unsafe impl<T: Send> Send for TensorDescriptor<T> {}

// SAFETY: shared methods only read attributes; raw callers must keep configured attributes unchanged.
unsafe impl<T: Sync> Sync for TensorDescriptor<T> {}

/// Nonnegative tensor identity used by frontend graphs and their bindings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(i64);

pub(crate) const TENSOR_ID_SERDE_NAME: &str = "singe_cudnn::TensorId";

/// Native cuDNN tensor identity used by variant packs and native descriptor relations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendTensorUid(i64);

/// Descriptor extents and positive element strides, independent of storage encoding.
///
/// Logical element counts and physical storage spans are separate checked queries. Vector lanes are described by tensor metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Shape {
    dimensions: Vec<i64>,
    strides: Vec<i64>,
}

impl<'de> Deserialize<'de> for Shape {
    fn deserialize<D>(deserializer: D) -> StdResult<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ShapeFields {
            dimensions: Vec<i64>,
            strides: Vec<i64>,
        }

        let fields = ShapeFields::deserialize(deserializer)?;
        Self::with_strides(fields.dimensions, fields.strides).map_err(DeError::custom)
    }
}

/// Initialization performed before each execution or replay of a frontend workspace tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceInitializer {
    /// Copy immutable native-encoded bytes, retaining their host source through queued work and replay.
    Copy(#[serde(with = "workspace_data_serde")] Arc<GraphHostBuffer>),
    /// Fill the complete reserved allocation with a repeated byte, including layout padding and capacity.
    /// The byte count follows the resolved runtime layout rather than allocating a host source.
    Fill(u8),
}

/// Frontend tensor declaration containing storage encoding, layout, placement, and native descriptor options.
///
/// Fields can be edited directly. [`Self::validate`] checks local metadata; graph preparation resolves referenced tensor identities and relationships.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Storage encoding used for descriptor elements and scalar values.
    pub data_type: DataType,
    /// Descriptor dimensions and element strides. An active vectorized dimension counts vectors rather than individual lanes.
    pub shape: Shape,
    /// Optional diagnostic tensor name.
    pub name: Option<String>,
    /// Required binding-address alignment in bytes; must be a positive power of two.
    #[serde(with = "serde_helpers::as_string")]
    pub byte_alignment: i64,
    /// Device, virtual, workspace, runtime-scalar, or compile-time-constant storage semantics.
    pub placement: TensorPlacement,
    /// Optional copy or byte fill applied to frontend workspace before every execution or replay.
    #[serde(default)]
    pub workspace_initializer: Option<WorkspaceInitializer>,
    /// An omitted count or one uses ordinary scalar elements without vector lanes.
    #[serde(with = "serde_helpers::as_string_opt")]
    pub vector_count: Option<i64>,
    /// Ignored when the vector count is omitted or one, matching the native tensor descriptor.
    #[serde(with = "serde_helpers::as_string_opt")]
    pub vectorized_dimension: Option<i64>,
    /// Optional native storage reordering, separate from vector lane metadata.
    pub reordering: Option<BackendTensorReordering>,
    /// Frontend identity of the tensor containing ragged offsets; graph preparation resolves it to a native descriptor.
    pub ragged_offset: Option<TensorId>,
    /// Number of tensor elements per ragged-offset unit. `None` selects one; an active multiplier must be positive.
    pub ragged_offset_multiplier: Option<i64>,
    /// Physical storage span in bytes when ragged offsets are attached.
    /// Overrides capacity supplied to storage sizing and is enforced by graph-aware bindings.
    pub ragged_byte_capacity: Option<usize>,
}

/// Storage location and scalar semantics of a graph tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorPlacement {
    /// Caller-bound device storage.
    Device,
    /// An intermediate tensor whose storage is managed by the selected cuDNN execution plan.
    Virtual,
    /// Device scratch bound automatically inside the frontend workspace allocation.
    /// Packed layouts require an explicit capacity in bytes.
    Workspace {
        /// Optional explicit workspace capacity in bytes. Packed storage requires an explicit bound; other layouts can infer capacity where supported.
        byte_capacity: Option<usize>,
    },
    /// A scalar consumed by value at execution, with an optional default value.
    RuntimeScalar {
        /// Default scalar value used when execution supplies no replacement; `None` requires a runtime scalar binding.
        default: Option<ScalarValue>,
    },
    /// A scalar embedded in the native descriptor at compilation and retained by its owner.
    Constant(ScalarValue),
}

/// Finalized native tensor descriptor with retained layout metadata and optional compile-time scalar storage.
///
/// The descriptor describes caller bindings or virtual storage; it does not allocate ordinary tensor data.
#[derive(Debug)]
pub struct Tensor {
    descriptor: BackendDescriptor,
    id: BackendTensorUid,
    data_type: DataType,
    shape: Shape,
    byte_alignment: i64,
    placement: TensorPlacement,
    vector_count: Option<i64>,
    vectorized_dimension: Option<i64>,
    reordering: Option<BackendTensorReordering>,
    ragged_offset: Option<BackendTensorUid>,
    ragged_offset_multiplier: i64,
    ragged_byte_capacity: Option<usize>,
    constant: Option<Box<ScalarStorage>>,
}

impl TensorId {
    /// Creates a nonnegative identity, including in constant declarations.
    /// Use [`Self::try_new`] for runtime values that may be negative.
    ///
    /// # Panics
    /// Panics if `value` is negative.
    #[must_use]
    pub const fn new(value: i64) -> Self {
        assert!(value >= 0, "tensor id must be non-negative");
        Self(value)
    }

    /// Returns a nonnegative identity, or `None` for a negative value.
    #[must_use]
    pub const fn try_new(value: i64) -> Option<Self> {
        if value >= 0 { Some(Self(value)) } else { None }
    }

    /// Generates the next identity from a process-wide counter.
    ///
    /// # Panics
    /// Panics if the counter cannot advance without overflowing.
    pub fn generate() -> Self {
        static NEXT_ID: AtomicI64 = AtomicI64::new(1);
        let id = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .expect("tensor id space exhausted");
        Self(id)
    }

    /// Returns the nonnegative numeric frontend tensor identity.
    #[must_use]
    pub const fn as_i64(self) -> i64 {
        self.0
    }
}

impl BackendTensorUid {
    /// Wraps a raw signed cuDNN tensor identity.
    #[must_use]
    pub const fn new(value: i64) -> Self {
        Self(value)
    }

    /// Returns the raw signed cuDNN tensor identity.
    #[must_use]
    pub const fn as_i64(self) -> i64 {
        self.0
    }
}

impl TryFrom<i64> for TensorId {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        Self::try_new(value).ok_or(FrontendError::TensorIdInvalid { value }.into())
    }
}

impl From<i64> for BackendTensorUid {
    fn from(value: i64) -> Self {
        Self::new(value)
    }
}

impl From<TensorId> for BackendTensorUid {
    fn from(value: TensorId) -> Self {
        Self::new(value.as_i64())
    }
}

impl From<TensorId> for i64 {
    fn from(value: TensorId) -> Self {
        value.0
    }
}

impl From<BackendTensorUid> for i64 {
    fn from(value: BackendTensorUid) -> Self {
        value.0
    }
}

/// Checks the numeric UID without resolving a graph's frontend/native identity mapping.
impl TryFrom<BackendTensorUid> for TensorId {
    type Error = Error;

    fn try_from(value: BackendTensorUid) -> Result<Self> {
        Self::try_from(value.as_i64())
    }
}

impl Display for TensorId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)?;
        Ok(())
    }
}

impl Display for BackendTensorUid {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)?;
        Ok(())
    }
}

impl FromStr for TensorId {
    type Err = Error;

    fn from_str(s: &str) -> StdResult<Self, Self::Err> {
        Self::try_from(s.parse::<i64>()?)
    }
}

impl Serialize for TensorId {
    fn serialize<S>(&self, serializer: S) -> StdResult<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_newtype_struct(TENSOR_ID_SERDE_NAME, &self.0.to_string())
    }
}

impl<'de> Deserialize<'de> for TensorId {
    fn deserialize<D>(deserializer: D) -> StdResult<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: i64 = as_string::deserialize(deserializer)?;
        Self::try_from(value).map_err(DeError::custom)
    }
}

impl<T: DataTypeLike> TensorDescriptor<T> {
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
    /// Dimensions and strides must have equal nonzero lengths. cuDNN validates the layout; this allocates only a descriptor, not tensor storage.
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
        // SAFETY: `cudnnCreateTensorDescriptor` receives valid output locations.
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
        // SAFETY: `cudnnSetTensorNdDescriptor` receives pointers into buffers borrowed for the call.
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
    /// Uses [`TensorFormat::NchwVectC`]; axis one is the channel dimension. This creates a descriptor without allocating tensor data.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than two dimensions are supplied, the type/vector encoding is unsupported, the channel extent is zero or not divisible by the vector width, or native rank conversion, descriptor creation, configuration, or metadata querying fails.
    ///
    /// # Panics
    ///
    /// A selected vector width must be nonzero and fit `i32`. A custom scalar/vector implementation violating this requirement can panic.
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
        // SAFETY: `cudnnCreateTensorDescriptor` receives valid output locations.
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
        // SAFETY: `cudnnSetTensorNdDescriptorEx` receives pointers into buffers borrowed for the call.
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
        // SAFETY: `cudnnGetTensorNdDescriptor` receives live handles owned by the wrappers and valid output locations.
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
        self.data_type = data_type.try_into()?;
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
        checked_element_count_i32(&self.dimensions)
    }

    /// Returns the byte length of the tensor in memory for this descriptor.
    /// Use this to determine the amount of GPU memory needed to hold the tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN cannot report the tensor size or if the value
    /// cannot be represented as `usize`.
    pub fn byte_len(&self) -> Result<usize> {
        let mut byte_len = 0;
        // SAFETY: `cudnnGetTensorSizeInBytes` receives live handles owned by the wrappers and valid output locations.
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
    /// The returned descriptor is borrowed and remains valid only while this
    /// wrapper is alive.
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
    /// `handle` must be a valid `cudnnTensorDescriptor_t` created by cuDNN.
    /// The returned wrapper takes ownership and will destroy it with
    /// `cudnnDestroyTensorDescriptor`; no other owner may destroy or keep using
    /// it.
    /// The descriptor's scalar encoding must match `T`, including when its storage is vectorized.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null, cuDNN cannot query its metadata, or the returned encoding or rank is invalid or cannot be represented by the wrapper.
    pub unsafe fn from_raw(handle: sys::cudnnTensorDescriptor_t) -> Result<Self>
    where
        T: DataTypeLike,
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

        // SAFETY: `cudnnDestroyTensorDescriptor` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudnnDestroyTensorDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cudnn tensor descriptor: {error}");
            }
        }
    }
}

/// Sets all elements of a tensor to a given value.
///
/// # Errors
///
/// Returns an error if the context, tensor descriptor, tensor memory, or scalar
/// value is invalid, if the operation fails to launch on the GPU, or if cuDNN
/// does not support the provided configuration.
///
/// # Safety
/// Retain the context, descriptor and destination through completion, failure
/// and replay. Order allocation readiness before use and keep writes exclusive.
/// Only descriptor elements are initialized; padding and tails remain unchanged.
pub unsafe fn set_tensor<T: DataTypeLike>(
    context: &Context,
    y_desc: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    value: &T,
) -> Result<()> {
    context.bind()?;
    if y.region().context() != context.cuda_context() {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    check_range!(
        "set tensor storage span",
        y.region().byte_len() >= y_desc.byte_len()?
    )?;

    // SAFETY: `cudnnSetTensor` receives live handles owned by the wrappers and valid output locations.
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

impl Shape {
    /// Creates packed strides with axes ordered from fastest to slowest.
    /// Every axis must appear exactly once, independently of tensor storage encoding.
    /// For dimensions `[2, 3, 4, 5]`, order `[3, 0, 2, 1]` produces strides `[5, 40, 10, 1]`.
    /// Singleton axes retain a representable stride after packed pitch overflow.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions, missing, repeated or out-of-range axes, or an unrepresentable non-singleton stride.
    pub fn packed<I, T, O>(dimensions: I, minor_to_major: O) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Borrow<i64>,
        O: IntoIterator<Item = usize>,
    {
        let dimensions = dimensions
            .into_iter()
            .map(|value| *value.borrow())
            .collect::<Vec<_>>();
        let strides = checked_ordered_strides(&dimensions, minor_to_major)?;
        Ok(Self {
            dimensions,
            strides,
        })
    }

    /// Creates a dense row-major layout.
    /// Singleton axes use a representable positive stride when their packed pitch exceeds INT64.
    /// Element counts and physical storage spans remain separate checked queries.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are empty or nonpositive, or a non-singleton packed stride cannot be represented as `i64`.
    pub fn contiguous<I, T>(dimensions: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Borrow<i64>,
    {
        let dimensions: Vec<i64> = dimensions.into_iter().map(|x| *x.borrow()).collect();
        let strides = checked_ordered_strides(&dimensions, (0..dimensions.len()).rev())?;

        Ok(Self {
            dimensions,
            strides,
        })
    }

    /// Creates a packed channels-last layout while retaining logical `[N, C, spatial...]` dimensions.
    /// Channels have unit stride, followed by reversed spatial axes and then the batch axis.
    /// One-dimensional inputs use ordinary contiguous storage.
    /// Singleton axes use a representable positive stride when their packed pitch exceeds INT64.
    ///
    /// # Errors
    /// Returns an error for empty or nonpositive dimensions, or an unrepresentable non-singleton stride.
    pub fn channels_last<I, T>(dimensions: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Borrow<i64>,
    {
        let dimensions: Vec<i64> = dimensions.into_iter().map(|x| *x.borrow()).collect();
        let rank = dimensions.len();
        let order = (rank > 1)
            .then_some(1)
            .into_iter()
            .chain((2..rank).rev())
            .chain([0]);
        let strides = checked_ordered_strides(&dimensions, order)?;
        Ok(Self {
            dimensions,
            strides,
        })
    }

    /// Creates an explicit layout without calculating an unused contiguous layout.
    /// Dimensions and strides must be positive and have matching ranks.
    /// Logical counts and physical storage spans use separate checked queries.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are empty or nonpositive, dimension and stride counts differ, or a stride is nonpositive.
    pub fn with_strides<D, S, T, U>(dimensions: D, strides: S) -> Result<Self>
    where
        D: IntoIterator<Item = T>,
        S: IntoIterator<Item = U>,
        T: Borrow<i64>,
        U: Borrow<i64>,
    {
        let dimensions: Vec<i64> = dimensions.into_iter().map(|x| *x.borrow()).collect();
        let strides: Vec<i64> = strides.into_iter().map(|x| *x.borrow()).collect();
        validate_dimensions(&dimensions)?;
        if dimensions.len() != strides.len() {
            return Err(Error::LengthMismatch {
                name: "tensor strides".into(),
                expected: dimensions.len(),
                actual: strides.len(),
            });
        }
        validate_strides(&strides)?;

        Ok(Self {
            dimensions,
            strides,
        })
    }

    /// Returns the descriptor extent of each axis.
    #[must_use]
    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }

    /// Returns the element stride of each descriptor axis.
    #[must_use]
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    /// Packs new dimensions with this shape's stride order.
    ///
    /// # Errors
    /// Returns an error if the rank differs or the packed strides are unrepresentable.
    pub fn with_dimensions_preserving_stride_order(
        &self,
        dimensions: impl Into<Vec<i64>>,
    ) -> Result<Self> {
        let dimensions = dimensions.into();
        if self.strides.len() != dimensions.len() {
            return Err(Error::LengthMismatch {
                name: "tensor strides".into(),
                expected: dimensions.len(),
                actual: self.strides.len(),
            });
        }
        Self::packed(dimensions, self.stride_order())
    }

    /// Swaps the two innermost axes, including their strides.
    ///
    /// # Errors
    /// Returns an error for shapes with fewer than two axes.
    pub fn transpose_last_two(&self) -> Result<Self> {
        let rank = self.dimensions.len();
        if rank < 2 {
            return Err(Error::InvalidDataShape);
        }
        let mut dimensions = self.dimensions.clone();
        let mut strides = self.strides.clone();
        dimensions.swap(rank - 2, rank - 1);
        strides.swap(rank - 2, rank - 1);
        Self::with_strides(dimensions, strides)
    }

    /// Unit dimensions and strides with this shape's rank.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored shape has an empty or inconsistent rank.
    pub fn unit_like(&self) -> Result<Self> {
        Self::with_strides(
            vec![1_i64; self.dimensions.len()],
            vec![1_i64; self.strides.len()],
        )
    }

    /// Axes ordered by increasing descriptor stride, from fastest to slowest.
    /// Each axis appears once.
    /// For tied strides, singleton axes come first, then axis index breaks remaining ties.
    /// Pass this order to [`Self::packed`] to rebuild packed strides for new dimensions.
    #[must_use]
    pub fn stride_order(&self) -> Vec<usize> {
        let mut axes = (0..self.strides.len()).collect::<Vec<_>>();
        axes.sort_unstable_by_key(|&axis| (self.strides[axis], self.dimensions[axis] != 1, axis));
        axes
    }

    /// Whether descriptor elements use dense row-major storage.
    /// Singleton strides do not affect contiguity, including after the packed pitch exceeds INT64.
    /// Does not allocate, query cuDNN or require a representable total element count.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = Some(1_i64);
        for (&dimension, &stride) in self.dimensions.iter().zip(&self.strides).rev() {
            if dimension != 1 && expected_stride != Some(stride) {
                return false;
            }
            expected_stride = expected_stride.and_then(|pitch| pitch.checked_mul(dimension));
        }
        true
    }

    /// Returns the logical descriptor-element count, excluding stride gaps and tensor vector lanes.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or cannot be represented as `usize`, or their product overflows `usize`.
    pub fn element_count(&self) -> Result<usize> {
        checked_element_count(&self.dimensions)
    }

    /// Whether the descriptor shape contains exactly one element, independently of its strides.
    /// Checks dimensions directly without multiplying or requiring a representable total count.
    /// Vector lanes belong to tensor metadata; use [`TensorSpec::has_scalar_shape`] for logical scalars.
    #[must_use]
    pub fn has_single_element(&self) -> bool {
        self.dimensions.iter().all(|&dimension| dimension == 1)
    }

    /// Number of element positions up to the highest address, including layout gaps.
    /// This query requires the element-position span itself to fit `usize`.
    /// Tensor bit and byte sizing use wider arithmetic before converting their final result.
    ///
    /// # Errors
    ///
    /// Returns an error if layout metadata is invalid, physical element-position arithmetic overflows `u128`, or the final span does not fit `usize`.
    pub fn storage_span(&self) -> Result<usize> {
        to_usize(self.storage_span_wide()?, "tensor storage span")
    }

    fn validate_metadata(&self) -> Result<()> {
        validate_dimensions(&self.dimensions)?;
        validate_strides(&self.strides)?;
        if self.dimensions.len() != self.strides.len() {
            return Err(Error::InvalidDataStrides);
        }
        Ok(())
    }

    fn storage_span_wide(&self) -> Result<u128> {
        self.validate_metadata()?;
        self.dimensions
            .iter()
            .zip(&self.strides)
            .try_fold(1_u128, |span, (&dimension, &stride)| {
                let extent = (dimension - 1) as u128;
                extent
                    .checked_mul(stride as u128)
                    .and_then(|offset| span.checked_add(offset))
                    .ok_or_else(|| Error::out_of_range("tensor storage span"))
            })
    }

    /// Returns the number of descriptor axes.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }
}

fn validate_dimensions(dimensions: &[i64]) -> Result<()> {
    if dimensions.is_empty() || dimensions.iter().any(|&dimension| dimension <= 0) {
        return Err(Error::InvalidDataShape);
    }
    Ok(())
}

fn checked_element_count_i32(dimensions: &[i32]) -> Result<usize> {
    dimensions
        .iter()
        .copied()
        .try_fold(1usize, |count, dimension| {
            let dimension = usize::try_from(dimension).map_err(|_| Error::InvalidDataShape)?;
            count
                .checked_mul(dimension)
                .ok_or_else(|| Error::out_of_range("tensor shape"))
        })
}

fn checked_element_count(dimensions: &[i64]) -> Result<usize> {
    dimensions
        .iter()
        .copied()
        .try_fold(1usize, |count, dimension| {
            let dimension = usize::try_from(dimension).map_err(|_| Error::InvalidDataShape)?;
            count
                .checked_mul(dimension)
                .ok_or_else(|| Error::out_of_range("tensor shape"))
        })
}

fn tensor_logical_element_count(shape: &Shape, lanes: i64) -> Result<usize> {
    let lanes = to_usize(lanes, "tensor vector count")?;
    shape
        .element_count()?
        .checked_mul(lanes)
        .ok_or_else(|| Error::out_of_range("tensor logical element count"))
}

fn tensor_logical_element_counts_match(
    left: &Shape,
    left_lanes: i64,
    right: &Shape,
    right_lanes: i64,
) -> bool {
    // Shapes and resolved lane counts are positive. Cancel factors instead of forming a bounded product.
    let mut remaining = right
        .dimensions()
        .iter()
        .copied()
        .chain([right_lanes])
        .filter(|&factor| factor != 1)
        .collect::<Vec<_>>();
    for mut factor in left.dimensions().iter().copied().chain([left_lanes]) {
        for other in &mut remaining {
            let (mut divisor, mut remainder) = (factor, *other);
            while remainder != 0 {
                (divisor, remainder) = (remainder, divisor % remainder);
            }
            factor /= divisor;
            *other /= divisor;
            if factor == 1 {
                break;
            }
        }
        if factor != 1 {
            return false;
        }
    }
    remaining.into_iter().all(|factor| factor == 1)
}

fn validate_strides(strides: &[i64]) -> Result<()> {
    if strides.iter().any(|&stride| stride <= 0) {
        return Err(Error::InvalidDataStrides);
    }
    Ok(())
}

fn checked_ordered_strides(
    dimensions: &[i64],
    ordered_indices: impl IntoIterator<Item = usize>,
) -> Result<Vec<i64>> {
    validate_dimensions(dimensions)?;

    let mut strides = vec![0_i64; dimensions.len()];
    let mut next_stride = Some(1_i64);
    let mut last_stride = 1_i64;
    for index in ordered_indices {
        let destination = strides.get_mut(index).ok_or(Error::InvalidAxisOrder)?;
        if *destination != 0 {
            return Err(Error::InvalidAxisOrder);
        }
        let dimension = dimensions[index];
        let stride = match next_stride {
            Some(stride) => stride,
            // Singleton axes never advance and can reuse the last representable pitch.
            None if dimension == 1 => last_stride,
            None => {
                return Err(Error::out_of_range("tensor strides"));
            }
        };
        *destination = stride;
        last_stride = stride;
        next_stride = next_stride.and_then(|pitch| pitch.checked_mul(dimension));
    }
    if strides.contains(&0) {
        return Err(Error::InvalidAxisOrder);
    }

    Ok(strides)
}

impl TensorSpec {
    /// Projects a fused scalar's native unit layout without changing the declaration.
    pub(crate) fn shape_with_scalar_rank(&self, rank: usize) -> Result<Cow<'_, Shape>> {
        if self.is_by_value() {
            self.validate_scalar()?;
            Ok(Cow::Owned(Shape::contiguous(vec![1; rank])?))
        } else {
            Ok(Cow::Borrowed(&self.shape))
        }
    }

    /// Shared identities must describe the same storage independently of names, alignment and promoted unit axes.
    pub(crate) fn storage_metadata_matches(&self, other: &Self) -> Result<bool> {
        Ok(self.data_type == other.data_type
            && self.placement == other.placement
            && self.workspace_initializer == other.workspace_initializer
            && self.vectorization()? == other.vectorization()?
            && self.effective_reordering() == other.effective_reordering()
            && self.ragged_offset == other.ragged_offset
            && self.ragged_byte_capacity == other.ragged_byte_capacity
            && self.ragged_offset_multiplier.unwrap_or(1)
                == other.ragged_offset_multiplier.unwrap_or(1))
    }

    /// Returns whether this declaration represents virtual intermediate storage.
    #[must_use]
    pub fn is_virtual(&self) -> bool {
        matches!(self.placement, TensorPlacement::Virtual)
    }

    /// Returns whether this tensor is placed in frontend workspace.
    #[must_use]
    pub fn is_workspace(&self) -> bool {
        matches!(self.placement, TensorPlacement::Workspace { .. })
    }

    /// Explicit workspace capacity, independent of the physical span inferred from the layout.
    #[must_use]
    pub fn workspace_byte_capacity(&self) -> Option<usize> {
        match self.placement {
            TensorPlacement::Workspace { byte_capacity } => byte_capacity,
            _ => None,
        }
    }

    /// Whether native execution consumes scalar bytes by value.
    /// Includes runtime scalars and compile-time constants, matching [`Tensor::is_by_value`].
    #[must_use]
    pub fn is_by_value(&self) -> bool {
        self.is_runtime_scalar() || self.is_constant()
    }

    /// Whether this declaration accepts a runtime scalar binding or override.
    /// Compile-time constants are by value but cannot be replaced at execution.
    #[must_use]
    pub fn is_runtime_scalar(&self) -> bool {
        matches!(self.placement, TensorPlacement::RuntimeScalar { .. })
    }

    /// Returns whether this declaration embeds a compile-time scalar value.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        matches!(self.placement, TensorPlacement::Constant(_))
    }

    /// Returns the runtime default or compile-time scalar value, or `None` when neither is present.
    #[must_use]
    pub fn scalar_value(&self) -> Option<ScalarValue> {
        match self.placement {
            TensorPlacement::RuntimeScalar { default } => default,
            TensorPlacement::Constant(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the runtime scalar default, or `None` for another placement or a scalar without a default.
    #[must_use]
    pub fn runtime_scalar_value(&self) -> Option<ScalarValue> {
        match self.placement {
            TensorPlacement::RuntimeScalar { default } => default,
            _ => None,
        }
    }

    pub(crate) fn validate_vector_stride_alignment(
        &self,
        id: TensorId,
        shape: &Shape,
    ) -> Result<()> {
        if self.byte_alignment <= 1 {
            return Ok(());
        }
        let lanes = self.vectorization()?.map_or(1, |(count, _)| count) as u128;
        let alignment_bits = (to_usize(self.byte_alignment, "tensor alignment")? as u128)
            .checked_mul(8)
            .ok_or_else(|| Error::out_of_range("tensor alignment"))?;
        shape.validate_metadata()?;
        for (axis, (&extent, &stride)) in shape.dimensions().iter().zip(shape.strides()).enumerate()
        {
            if extent <= 1 || stride == 1 {
                continue;
            }
            let stride_bits = (stride as u128)
                .checked_mul(lanes)
                .and_then(|stride| stride.checked_mul(self.data_type.storage_bits() as u128))
                .ok_or_else(|| Error::out_of_range("attention byte stride"))?;
            if stride_bits % alignment_bits != 0 {
                return Err(FrontendError::TensorLayoutMismatch { tensor_id: id, operation: "native attention operand alignment", reason: format!(
                        "stride of dimension {axis} is not a multiple of the tensor's {}-byte alignment",
                        self.byte_alignment
                    ).into() }.into());
            }
        }
        Ok(())
    }

    /// Scalar dimensions after expanding the native vectorized dimension by its lane count.
    /// Descriptor dimensions count vectors on that axis; storage strides remain in vector units.
    ///
    /// # Errors
    ///
    /// Returns an error if an active vector count is nonpositive, its axis is missing or out of range, or rank/axis normalization cannot be represented by the native integer type. Also returns an error if dimensions are invalid or expanding the vectorized extent overflows `i64`.
    pub fn logical_dimensions(&self) -> Result<Vec<i64>> {
        let mut dimensions = self.shape.dimensions().to_vec();
        validate_dimensions(&dimensions)?;
        if let Some((count, axis)) = self.vectorization()? {
            let axis = to_usize(axis, "tensor vector dimension")?;
            dimensions[axis] = dimensions[axis]
                .checked_mul(count)
                .ok_or_else(|| Error::out_of_range("tensor logical dimension"))?;
        }
        Ok(dimensions)
    }

    /// Number of scalar values, including vector lanes and excluding stride gaps.
    /// Ragged dimensions describe the logical extent rather than an allocation capacity.
    /// Counts directly in `usize` without allocating expanded dimensions or requiring them to fit INT64.
    ///
    /// # Errors
    ///
    /// Returns an error if an active vector count is nonpositive, its axis is missing or out of range, or rank/axis normalization cannot be represented by the native integer type. Also returns an error if a descriptor extent is invalid or the scalar count cannot be represented as `usize`.
    pub fn logical_element_count(&self) -> Result<usize> {
        let lanes = self.vectorization()?.map_or(1, |(count, _)| count);
        tensor_logical_element_count(&self.shape, lanes)
    }

    /// Whether this declaration contains exactly one logical scalar, including vector lanes.
    /// Independent of storage encoding, placement and strides, without multiplying dimensions.
    /// Returns an error for invalid vector metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if an active vector count is nonpositive, its axis is missing or out of range, or rank/axis normalization cannot be represented by the native integer type.
    pub fn has_scalar_shape(&self) -> Result<bool> {
        let lanes = self.vectorization()?.map_or(1, |(count, _)| count);
        Ok(lanes == 1 && self.shape.has_single_element())
    }

    /// Compares logical scalar counts, including vector lanes, without forming either total count.
    /// Supports counts beyond `usize` and ignores storage encoding, strides and ragged byte capacity.
    /// Returns an error for invalid vector metadata on either declaration.
    ///
    /// # Errors
    ///
    /// Returns an error if either declaration has an invalid active vector count or axis, or rank/axis normalization fails.
    pub fn has_same_logical_element_count(&self, other: &Self) -> Result<bool> {
        let lanes = self.vectorization()?.map_or(1, |(count, _)| count);
        let other_lanes = other.vectorization()?.map_or(1, |(count, _)| count);
        Ok(tensor_logical_element_counts_match(
            &self.shape,
            lanes,
            &other.shape,
            other_lanes,
        ))
    }

    /// Physical bytes needed by a dense or strided tensor, including vector lanes.
    /// Packed sizing does not require the span in element positions to fit `usize`.
    /// Ragged layouts use their declared byte capacity, or the supplied capacity when none is declared.
    /// Nominal dimensions do not establish ragged allocation size.
    ///
    /// # Errors
    ///
    /// Returns an error if shape or vector metadata is invalid, the storage encoding is unsupported, physical-span arithmetic overflows or the byte count does not fit `usize`, or ragged storage has neither a declared capacity nor a supplied `ragged_capacity`.
    pub fn storage_byte_len(&self, ragged_capacity: Option<usize>) -> Result<usize> {
        let lanes = self.vectorization()?.map_or(1, |(count, _)| count) as u128;
        if self.ragged_offset.is_some() {
            self.shape.validate_metadata()?;
            if self.data_type == DataType::FastFloatForFp8 {
                return Err(Error::UnsupportedDataType);
            }
            return self.ragged_byte_capacity.or(ragged_capacity).ok_or(
                Error::DescriptorMismatch {
                    name: "ragged tensor capacity".into(),
                },
            );
        }
        let bits = tensor_storage_bit_len(&self.shape, self.data_type, lanes)?;
        to_usize(bits.div_ceil(8), "tensor storage bytes")
    }

    /// Dense physical bit span, or the declared ragged byte capacity expressed in bits.
    /// Dense packed tails are not rounded up, so views cannot consume their padding bits.
    /// Ragged storage requires a declared capacity because dimensions do not determine its span.
    ///
    /// # Errors
    ///
    /// Returns an error if shape or vector metadata is invalid, the storage encoding is unsupported, physical-span arithmetic overflows, or ragged storage has no required capacity declaration.
    pub fn storage_bit_len(&self) -> Result<u128> {
        if self.ragged_offset.is_some() {
            return Ok(self.storage_byte_len(None)? as u128 * 8);
        }
        let lanes = self.vectorization()?.map_or(1, |(count, _)| count) as u128;
        tensor_storage_bit_len(&self.shape, self.data_type, lanes)
    }

    /// Returns active native reordering, treating an omitted mode and explicit NONE identically.
    pub(crate) fn effective_reordering(&self) -> Option<BackendTensorReordering> {
        self.reordering
            .filter(|&mode| mode != BackendTensorReordering::None)
    }

    /// Resolves vector metadata once for inference, storage sizing, alignment and native construction.
    /// A default or unit count disables vectorization and ignores its inactive axis.
    /// Active negative axes follow the same normalization as the vectorization builder.
    pub(crate) fn vectorization(&self) -> Result<Option<(i64, i64)>> {
        match (self.vector_count, self.vectorized_dimension) {
            (None | Some(1), _) => Ok(None),
            (Some(count), Some(axis)) if count > 1 => {
                let rank = to_i64(self.shape.rank(), "tensor rank")?;
                let axis = if axis < 0 {
                    rank.checked_add(axis)
                } else {
                    Some(axis)
                }
                .filter(|&axis| axis >= 0 && axis < rank)
                .ok_or_else(|| Error::out_of_range("tensor vector dimension"))?;
                Ok(Some((count, axis)))
            }
            _ => Err(Error::DescriptorMismatch {
                name: "tensor vectorization".into(),
            }),
        }
    }

    /// Creates a device-storage declaration with 16-byte alignment and no vectorization, reordering, or ragged offsets.
    ///
    /// Records the supplied encoding and shape; [`Self::validate`] performs local validation.
    #[must_use]
    pub fn new(data_type: DataType, shape: Shape) -> Self {
        Self {
            data_type,
            shape,
            name: None,
            byte_alignment: 16,
            placement: TensorPlacement::Device,
            vector_count: None,
            vectorized_dimension: None,
            reordering: None,
            ragged_offset: None,
            ragged_offset_multiplier: None,
            ragged_byte_capacity: None,
            workspace_initializer: None,
        }
    }

    /// Sets the diagnostic tensor name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Sets the diagnostic tensor name and returns the updated declaration.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.set_name(name);
        self
    }

    /// Removes the diagnostic tensor name.
    pub fn clear_name(&mut self) {
        self.name = None;
    }

    /// Replaces the descriptor dimensions and strides while preserving the remaining declaration metadata.
    #[must_use]
    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }

    /// Sets the required binding-address alignment in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `byte_alignment` is nonpositive or is not a power of two.
    pub fn with_byte_alignment(mut self, byte_alignment: i64) -> Result<Self> {
        check_range!(
            "byte_alignment",
            byte_alignment > 0 && (byte_alignment as u64).is_power_of_two()
        )?;
        self.byte_alignment = byte_alignment;
        Ok(self)
    }

    /// Sets active vector lanes and their axis, normalizing negative axes against the current rank.
    /// A count of one clears vectorization and ignores the supplied axis, matching cuDNN.
    ///
    /// # Errors
    /// Returns an error for a nonpositive count or an invalid axis when more than one lane is selected.
    pub fn with_vectorization(
        mut self,
        vector_count: i64,
        vectorized_dimension: i64,
    ) -> Result<Self> {
        self.vector_count = Some(vector_count);
        self.vectorized_dimension = Some(vectorized_dimension);
        let vectorization = self.vectorization()?;
        self.vector_count = vectorization.map(|(count, _)| count);
        self.vectorized_dimension = vectorization.map(|(_, dimension)| dimension);
        Ok(self)
    }

    /// Sets native reordering, or clears it with `None` or [`BackendTensorReordering::None`].
    #[must_use]
    pub fn with_reordering(
        mut self,
        reordering: impl Into<Option<BackendTensorReordering>>,
    ) -> Self {
        self.reordering = reordering.into();
        self.reordering = self.effective_reordering();
        self
    }

    /// Replaces an identity reference without changing its storage or offset-unit metadata.
    /// Internal graph imports can detach the reference until all destination identities exist.
    pub(crate) fn with_ragged_offset_reference(mut self, ragged_offset: Option<TensorId>) -> Self {
        self.ragged_offset = ragged_offset;
        self
    }

    pub(crate) fn clear_ragged_offset(&mut self) {
        self.ragged_offset = None;
        self.ragged_byte_capacity = None;
        self.ragged_offset_multiplier = None;
    }

    /// Declares the physical span of ragged storage in bytes.
    /// Attach ragged offsets before creating a storage view or preparing the graph.
    /// The capacity is independent of nominal dimensions and offset units.
    /// Graph-aware execution rejects a binding whose remaining allocation is smaller.
    /// Without a declaration, ragged sizing requires a capacity from the caller's binding or workspace.
    #[must_use]
    pub fn with_ragged_byte_capacity(mut self, byte_capacity: usize) -> Self {
        self.ragged_byte_capacity = Some(byte_capacity);
        self
    }

    /// Uses the native 128-by-4 tile reordering for block-scale values.
    /// Dimensions and strides describe the caller's scale storage, including any tile padding.
    /// Reordering does not add vector lanes or change the declared dimensions.
    #[must_use]
    pub fn with_f8_128x4_reordering(self) -> Self {
        self.with_reordering(BackendTensorReordering::F8_128x4)
    }

    /// Creates one runtime by-value scalar with its Rust value's native encoding.
    /// The value fixes both the encoding and the unit shape, so declaration is infallible.
    /// Native allocation, alignment and engine support remain later fallible steps.
    /// Use [`Self::constant`] when the value belongs to the compiled graph instead.
    pub fn scalar(value: impl Into<ScalarValue>) -> Self {
        let value = value.into();
        Self {
            placement: TensorPlacement::RuntimeScalar {
                default: Some(value),
            },
            ..Self::new(
                value.data_type(),
                Shape {
                    dimensions: vec![1],
                    strides: vec![1],
                },
            )
        }
    }

    /// Creates one runtime by-value `f32` scalar with the supplied default value.
    #[must_use]
    pub fn scalar_f32(value: f32) -> Self {
        Self::scalar(value)
    }

    /// Creates one runtime by-value `i32` scalar with the supplied default value.
    #[must_use]
    pub fn scalar_i32(value: i32) -> Self {
        Self::scalar(value)
    }

    /// Creates one runtime by-value `f8e4m3` scalar with the supplied default value.
    #[must_use]
    pub fn scalar_f8e4m3(value: f8e4m3) -> Self {
        Self::scalar(value)
    }

    /// Creates one runtime by-value `f8e5m2` scalar with the supplied default value.
    #[must_use]
    pub fn scalar_f8e5m2(value: f8e5m2) -> Self {
        Self::scalar(value)
    }

    /// Creates one runtime by-value `f8ue8m0` scalar with the supplied default value.
    #[must_use]
    pub fn scalar_f8ue8m0(value: f8ue8m0) -> Self {
        Self::scalar(value)
    }

    /// Creates one runtime by-value `f4e2m1` scalar with the supplied default value.
    #[must_use]
    pub fn scalar_f4e2m1(value: f4e2m1) -> Self {
        Self::scalar(value)
    }

    /// Creates one compile-time constant with its value's native encoding and a unit shape.
    /// The declaration is infallible and requires no runtime binding.
    /// Use [`Self::with_constant_value`] to attach a constant to an existing typed declaration.
    pub fn constant(value: impl Into<ScalarValue>) -> Self {
        let value = value.into();
        Self {
            placement: TensorPlacement::Constant(value),
            ..Self::scalar(value)
        }
    }

    /// Marks the declaration as a virtual intermediate and clears any workspace initializer.
    #[must_use]
    pub fn virtual_tensor(mut self) -> Self {
        self.placement = TensorPlacement::Virtual;
        self.workspace_initializer = None;
        self
    }

    /// Places an internal tensor in the caller's frontend workspace.
    /// Dense capacity follows the physical layout, including strides and vector lanes.
    #[must_use]
    pub fn workspace_tensor(mut self) -> Self {
        self.placement = TensorPlacement::Workspace {
            byte_capacity: None,
        };
        self
    }

    /// Places a tensor in frontend workspace and copies its native storage before every execution.
    /// The data is retained through asynchronous copies and CUDA graph replay.
    /// An existing workspace capacity is preserved, including the bound required by ragged storage.
    ///
    /// # Errors
    ///
    /// Returns storage-sizing errors from [`Self::storage_byte_len`], or a length mismatch if `bytes` does not contain exactly the required physical byte capacity.
    pub fn with_workspace_data(mut self, bytes: Vec<u8>) -> Result<Self> {
        if !self.is_workspace() {
            self = self.workspace_tensor();
        }
        let expected = self.storage_byte_len(self.workspace_byte_capacity())?;
        if bytes.len() != expected {
            return Err(Error::LengthMismatch {
                name: "workspace tensor data".into(),
                expected,
                actual: bytes.len(),
            });
        }
        self.workspace_initializer = Some(WorkspaceInitializer::Copy(Arc::new(
            GraphHostBuffer::new(bytes),
        )));
        Ok(self)
    }

    /// Places a tensor in frontend workspace and fills its allocation before every execution or replay.
    /// Preserves an existing capacity bound and fills the complete resolved allocation, including padding.
    /// The value is a repeated byte pattern, independent of the tensor's numeric encoding.
    #[must_use]
    pub fn with_workspace_fill(mut self, value: u8) -> Self {
        if !self.is_workspace() {
            self = self.workspace_tensor();
        }
        self.workspace_initializer = Some(WorkspaceInitializer::Fill(value));
        self
    }

    /// Places an internal tensor in workspace with a capacity bound in bytes.
    /// Ragged layouts use this bound because their offsets are supplied at execution.
    #[must_use]
    pub fn with_workspace_byte_capacity(mut self, bytes: usize) -> Self {
        self.placement = TensorPlacement::Workspace {
            byte_capacity: Some(bytes),
        };
        self
    }

    /// Converts virtual or workspace placement to caller-bound device storage and clears any workspace initializer.
    ///
    /// Scalar placements retain their scalar semantics.
    #[must_use]
    pub fn output_tensor(mut self) -> Self {
        if self.is_virtual() || self.is_workspace() {
            self.placement = TensorPlacement::Device;
        }
        self.workspace_initializer = None;
        self
    }

    /// Selects runtime by-value scalar placement, retaining any existing scalar value as its default.
    ///
    /// Clears a workspace initializer. Scalar shape and encoding are checked by [`Self::validate`].
    #[must_use]
    pub fn pass_by_value(mut self) -> Self {
        self.placement = TensorPlacement::RuntimeScalar {
            default: self.scalar_value(),
        };
        self.workspace_initializer = None;
        self
    }

    /// Validates this declaration without creating a native descriptor or accessing a device.
    /// Checks layout metadata, storage encoding, alignment, scalar values,
    /// workspace initializer bytes and the local ragged-offset multiplier contract.
    /// Graph preparation resolves referenced tensor identities and their relationships.
    ///
    /// # Errors
    /// Returns an error for invalid metadata, mismatched scalar encodings or
    /// workspace data, or a nonunit ragged-offset multiplier without offsets.
    pub fn validate(&self) -> Result<()> {
        let multiplier = self.ragged_offset_multiplier()?;
        if let Some(initializer) = &self.workspace_initializer {
            if !self.is_workspace() {
                return Err(Error::DescriptorMismatch {
                    name: "workspace tensor initializer".into(),
                });
            }
            if let WorkspaceInitializer::Copy(data) = initializer
                && data.as_bytes().len() != self.storage_byte_len(self.workspace_byte_capacity())?
            {
                return Err(Error::DescriptorMismatch {
                    name: "workspace tensor data".into(),
                });
            }
        }
        self.vectorization()?;
        if self.data_type == DataType::FastFloatForFp8 {
            return Err(Error::UnsupportedDataType);
        }
        self.shape.validate_metadata()?;
        if self.byte_alignment <= 0 || !(self.byte_alignment as u64).is_power_of_two() {
            return Err(Error::out_of_range("tensor byte alignment"));
        }
        if multiplier != 1 && self.ragged_offset.is_none() {
            return Err(Error::DescriptorMismatch {
                name: "tensor ragged offset multiplier requires offsets".into(),
            });
        }
        self.validate_scalar()
    }

    /// Validates scalar storage before shape inference can promote its rank.
    pub(crate) fn validate_scalar(&self) -> Result<()> {
        if self.is_by_value() && !self.has_scalar_shape()? {
            return Err(Error::DescriptorMismatch {
                name: "tensor scalar shape".into(),
            });
        }
        if let Some(value) = self.scalar_value()
            && value.data_type() != self.data_type
        {
            return Err(Error::TensorScalarDataTypeMismatch {
                expected: self.data_type,
                actual: value.data_type(),
            });
        }
        Ok(())
    }

    /// Sets a runtime scalar default whose native encoding matches this declaration.
    ///
    /// Preserves the declared shape and clears workspace initialization. [`Self::validate`] checks that the resulting declaration contains one logical scalar.
    ///
    /// # Errors
    ///
    /// Returns an error if the supplied scalar encoding differs from `data_type`.
    pub fn with_scalar_value(mut self, scalar_value: impl Into<ScalarValue>) -> Result<Self> {
        let scalar_value = scalar_value.into();
        if scalar_value.data_type() != self.data_type {
            return Err(Error::TensorScalarDataTypeMismatch {
                expected: self.data_type,
                actual: scalar_value.data_type(),
            });
        }

        self.placement = TensorPlacement::RuntimeScalar {
            default: Some(scalar_value),
        };
        self.workspace_initializer = None;
        Ok(self)
    }

    /// Sets a compile-time value on this declaration, requiring its encoding to match.
    /// Preserves the declared shape and metadata; [`Self::constant`] infers a scalar declaration instead.
    ///
    /// # Errors
    ///
    /// Returns an error if the supplied scalar encoding differs from this declaration's data type.
    pub fn with_constant_value(mut self, value: impl Into<ScalarValue>) -> Result<Self> {
        let value = value.into();
        self = self.with_scalar_value(value)?;
        self.placement = TensorPlacement::Constant(value);
        Ok(self)
    }

    /// Sets the number of tensor elements represented by one ragged-offset unit.
    /// One restores the default. Other positive values require an offset tensor before lowering.
    ///
    /// # Errors
    ///
    /// Returns an error if `multiplier` is nonpositive.
    pub fn with_ragged_offset_multiplier(mut self, multiplier: i64) -> Result<Self> {
        self.ragged_offset_multiplier = Some(multiplier);
        let multiplier = self.ragged_offset_multiplier()?;
        self.ragged_offset_multiplier = (multiplier != 1).then_some(multiplier);
        Ok(self)
    }

    /// Effective positive multiplier for ragged offsets, defaulting to one.
    /// An explicit one has the same meaning as an omitted value, including without ragged storage.
    /// Checks directly assigned and deserialized values as well as builder inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if an explicitly stored multiplier is nonpositive.
    pub fn ragged_offset_multiplier(&self) -> Result<i64> {
        let multiplier = self.ragged_offset_multiplier.unwrap_or(1);
        check_range!("ragged offset multiplier", multiplier > 0)?;
        Ok(multiplier)
    }
}

impl Tensor {
    /// Creates a native tensor with the cuDNN UID used by variant packs and native tensor relations.
    /// Frontend identities must be converted explicitly at the lowering boundary.
    ///
    /// # Errors
    /// Returns an error for invalid tensor metadata, missing ragged descriptors, scalar allocation failure or native descriptor rejection.
    pub fn create_with_spec(
        id: BackendTensorUid,
        spec: &TensorSpec,
        ragged_offset: Option<&Self>,
    ) -> Result<Self> {
        spec.validate()?;
        let shape = &spec.shape;
        let ragged_offset_multiplier = spec.ragged_offset_multiplier()?;
        let reordering = spec.effective_reordering();
        let (vector_count, vectorized_dimension) = spec
            .vectorization()?
            .map_or((None, None), |(count, axis)| (Some(count), Some(axis)));
        if spec.ragged_offset.is_some() != ragged_offset.is_some() {
            return Err(Error::DescriptorMismatch {
                name: "tensor ragged offset descriptor".into(),
            });
        }

        let mut constant = match spec.placement {
            TensorPlacement::Constant(value) => Some(Box::new(ScalarStorage::create(
                value,
                to_usize(spec.byte_alignment, "tensor alignment")?,
            )?)),
            _ => None,
        };

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Tensor)?;
        descriptor.set_attribute_i64(BackendAttributeName::TensorUniqueId, id.as_i64())?;
        descriptor.set_attribute_enum(BackendAttributeName::TensorDataType, spec.data_type)?;
        descriptor
            .set_attribute_i64_slice(BackendAttributeName::TensorDimensions, &shape.dimensions)?;
        descriptor.set_attribute_i64_slice(BackendAttributeName::TensorStrides, &shape.strides)?;
        descriptor.set_attribute_i64(
            BackendAttributeName::TensorByteAlignment,
            spec.byte_alignment,
        )?;
        descriptor.set_attribute_bool(BackendAttributeName::TensorIsVirtual, spec.is_virtual())?;
        descriptor.set_attribute_bool(BackendAttributeName::TensorIsByValue, spec.is_by_value())?;
        if let Some(storage) = &mut constant {
            // SAFETY: the typed, aligned scalar bytes are retained by the tensor descriptor owner.
            unsafe {
                descriptor.set_attribute_void_ptr(
                    BackendAttributeName::TensorConstantValue,
                    storage.as_ptr().cast_mut().cast(),
                )?;
            }
        }
        if let Some(offset) = ragged_offset {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::TensorRaggedOffsetDesc,
                offset.descriptor(),
            )?;
        }
        if ragged_offset_multiplier != 1 {
            descriptor.set_attribute_i64(
                BackendAttributeName::TensorRaggedOffsetMultiplier,
                ragged_offset_multiplier,
            )?;
        }
        if let Some(vectorized_dimension) = vectorized_dimension {
            descriptor.set_attribute_i64(
                BackendAttributeName::TensorVectorizedDimension,
                vectorized_dimension,
            )?;
        }
        if let Some(vector_count) = vector_count {
            descriptor.set_attribute_i64(BackendAttributeName::TensorVectorCount, vector_count)?;
        }
        if let Some(reordering) = reordering {
            descriptor
                .set_attribute_enum(BackendAttributeName::TensorReorderingMode, reordering)?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            id,
            data_type: spec.data_type,
            shape: shape.clone(),
            byte_alignment: spec.byte_alignment,
            placement: spec.placement,
            vector_count,
            vectorized_dimension,
            reordering,
            ragged_offset: ragged_offset.map(Self::id),
            ragged_offset_multiplier,
            ragged_byte_capacity: spec.ragged_byte_capacity,
            constant,
        })
    }

    /// Creates a finalized device tensor descriptor with an automatically generated native identity.
    ///
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes. Ordinary tensor storage is supplied separately.
    ///
    /// # Errors
    ///
    /// Returns an error if layout metadata, storage encoding, or byte alignment is invalid, or native descriptor creation, attribute assignment, or finalization fails.
    ///
    /// # Panics
    ///
    /// Panics if the process-wide tensor identity counter is exhausted.
    pub fn create(
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
    ) -> Result<Self> {
        Self::create_with_spec(
            BackendTensorUid::from(TensorId::generate()),
            &TensorSpec::new(data_type, Shape::with_strides(dimensions, strides)?)
                .with_byte_alignment(byte_alignment)?,
            None,
        )
    }

    /// Creates a finalized device tensor descriptor with the supplied native identity.
    ///
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes. Ordinary tensor storage is supplied separately.
    ///
    /// # Errors
    ///
    /// Returns an error if layout metadata, storage encoding, or byte alignment is invalid, or native descriptor creation, attribute assignment, or finalization fails.
    pub fn create_with_id(
        id: BackendTensorUid,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
    ) -> Result<Self> {
        Self::create_with_spec(
            id,
            &TensorSpec::new(data_type, Shape::with_strides(dimensions, strides)?)
                .with_byte_alignment(byte_alignment)?,
            None,
        )
    }

    /// Creates a finalized device tensor descriptor with explicit native storage reordering.
    ///
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes. Ordinary tensor storage is supplied separately.
    ///
    /// # Errors
    ///
    /// Returns an error if layout metadata, storage encoding, or byte alignment is invalid, or native descriptor creation, attribute assignment, or finalization fails.
    pub fn create_with_reordering(
        id: BackendTensorUid,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
        reordering: BackendTensorReordering,
    ) -> Result<Self> {
        Self::create_with_spec(
            id,
            &TensorSpec::new(data_type, Shape::with_strides(dimensions, strides)?)
                .with_byte_alignment(byte_alignment)?
                .with_reordering(reordering),
            None,
        )
    }

    /// Creates a finalized tensor descriptor with explicit vector lanes, axis, and optional reordering.
    ///
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes. Ordinary tensor storage is supplied separately.
    ///
    /// # Errors
    ///
    /// Returns an error if layout metadata, storage encoding, or byte alignment is invalid, or native descriptor creation, attribute assignment, or finalization fails. Vector counts must be positive, and an active vector axis must resolve within the descriptor rank.
    pub fn create_vectorized(
        id: BackendTensorUid,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
        vectorized_dimension: i64,
        vector_count: i64,
        reordering: Option<BackendTensorReordering>,
    ) -> Result<Self> {
        Self::create_with_spec(
            id,
            &TensorSpec::new(data_type, Shape::with_strides(dimensions, strides)?)
                .with_byte_alignment(byte_alignment)?
                .with_vectorization(vector_count, vectorized_dimension)?
                .with_reordering(reordering),
            None,
        )
    }

    /// Creates a finalized virtual tensor descriptor with the supplied native identity.
    ///
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes. Ordinary tensor storage is supplied separately.
    ///
    /// # Errors
    ///
    /// Returns an error if layout metadata, storage encoding, or byte alignment is invalid, or native descriptor creation, attribute assignment, or finalization fails.
    pub fn create_virtual(
        id: BackendTensorUid,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
    ) -> Result<Self> {
        Self::create_with_spec(
            id,
            &TensorSpec::new(data_type, Shape::with_strides(dimensions, strides)?)
                .with_byte_alignment(byte_alignment)?
                .virtual_tensor(),
            None,
        )
    }

    /// Creates a finalized virtual tensor descriptor with native storage reordering.
    ///
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes. Ordinary tensor storage is supplied separately.
    ///
    /// # Errors
    ///
    /// Returns an error if layout metadata, storage encoding, or byte alignment is invalid, or native descriptor creation, attribute assignment, or finalization fails.
    pub fn create_virtual_with_reordering(
        id: BackendTensorUid,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
        reordering: BackendTensorReordering,
    ) -> Result<Self> {
        Self::create_with_spec(
            id,
            &TensorSpec::new(data_type, Shape::with_strides(dimensions, strides)?)
                .with_byte_alignment(byte_alignment)?
                .with_reordering(reordering)
                .virtual_tensor(),
            None,
        )
    }

    /// Creates a finalized runtime by-value scalar descriptor without a default scalar binding.
    ///
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes. Ordinary tensor storage is supplied separately.
    ///
    /// # Errors
    ///
    /// Returns an error if layout metadata, storage encoding, or byte alignment is invalid, or native descriptor creation, attribute assignment, or finalization fails. The declaration must contain exactly one logical scalar.
    pub fn create_by_value(
        id: BackendTensorUid,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
    ) -> Result<Self> {
        Self::create_with_spec(
            id,
            &TensorSpec::new(data_type, Shape::with_strides(dimensions, strides)?)
                .with_byte_alignment(byte_alignment)?
                .pass_by_value(),
            None,
        )
    }

    /// Returns the native tensor identity used by variant packs and descriptor relations.
    #[must_use]
    pub fn id(&self) -> BackendTensorUid {
        self.id
    }

    /// Returns the tensor's native storage encoding.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Borrows the retained descriptor dimensions and element strides.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Copies native declaration metadata for a complete graph's execution-plan cache key.
    /// Includes constant bytes and ragged relationships, without querying cuDNN or retaining owners.
    /// Runtime scalar values, device addresses and graph/target configuration are not included.
    #[must_use]
    pub fn problem_key(&self) -> TensorProblemKey {
        TensorProblemKey::from(self)
    }

    /// Native byte-alignment requirement recorded at descriptor creation.
    #[must_use]
    pub fn byte_alignment(&self) -> i64 {
        self.byte_alignment
    }

    /// Whether the descriptor represents a virtual intermediate.
    #[must_use]
    pub fn is_virtual(&self) -> bool {
        self.placement == TensorPlacement::Virtual
    }

    /// Whether the native descriptor consumes scalar bytes by value.
    /// Includes runtime scalars and compile-time constants.
    #[must_use]
    pub fn is_by_value(&self) -> bool {
        self.is_runtime_scalar() || self.is_constant()
    }

    /// Whether this descriptor consumes a runtime scalar binding rather than a compiled constant.
    #[must_use]
    pub fn is_runtime_scalar(&self) -> bool {
        matches!(self.placement, TensorPlacement::RuntimeScalar { .. })
    }

    /// Whether this descriptor retains a compile-time scalar constant.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        self.constant.is_some()
    }

    /// Retained compile-time scalar value, independent of runtime variant-pack bindings.
    #[must_use]
    pub fn constant_value(&self) -> Option<ScalarValue> {
        self.constant.as_ref().map(|storage| storage.value())
    }

    /// Dense physical bit span, or the declared ragged byte capacity expressed in bits.
    /// Includes dense vector lanes and stride gaps without rounding packed tails up.
    ///
    /// # Errors
    /// Returns an error for missing ragged capacity or overflowing shape and size arithmetic.
    pub fn storage_bit_len(&self) -> Result<u128> {
        if self.ragged_offset.is_some() {
            return Ok(self.storage_byte_len(None)? as u128 * 8);
        }
        tensor_storage_bit_len(
            &self.shape,
            self.data_type,
            self.vector_count.unwrap_or(1) as u128,
        )
    }

    /// Physical bytes required by this descriptor, rounding packed tails up once.
    /// Packed sizing does not require the span in element positions to fit `usize`.
    /// Ragged tensors use their retained byte capacity, or the supplied capacity when none was declared.
    /// Nominal dimensions and offset metadata do not establish ragged allocation size.
    ///
    /// # Errors
    /// Returns an error for missing ragged capacity or overflowing shape and byte counts.
    pub fn storage_byte_len(&self, ragged_capacity: Option<usize>) -> Result<usize> {
        if self.ragged_offset.is_some() {
            self.shape.validate_metadata()?;
            if self.data_type == DataType::FastFloatForFp8 {
                return Err(Error::UnsupportedDataType);
            }
            return self.ragged_byte_capacity.or(ragged_capacity).ok_or(
                Error::DescriptorMismatch {
                    name: "ragged tensor capacity".into(),
                },
            );
        }
        to_usize(self.storage_bit_len()?.div_ceil(8), "tensor storage bytes")
    }

    /// Number of scalar values, including vector lanes and excluding stride gaps.
    /// Ragged dimensions describe the logical extent rather than an allocation capacity.
    /// Uses the same checked count as [`TensorSpec::logical_element_count`].
    ///
    /// # Errors
    ///
    /// Returns an error if a descriptor extent or lane count is invalid, cannot be represented as `usize`, or causes the scalar-element count to overflow `usize`.
    pub fn logical_element_count(&self) -> Result<usize> {
        tensor_logical_element_count(&self.shape, self.vector_count.unwrap_or(1))
    }

    /// Whether this tensor contains exactly one logical scalar, including vector lanes.
    /// Uses the same shape predicate as [`TensorSpec::has_scalar_shape`] without multiplying dimensions.
    /// Independent of storage encoding, placement and strides.
    #[must_use]
    pub fn has_scalar_shape(&self) -> bool {
        self.vector_count.unwrap_or(1) == 1 && self.shape.has_single_element()
    }

    /// Compares logical scalar counts, including vector lanes, without forming either total count.
    /// Uses the same exact comparison as [`TensorSpec::has_same_logical_element_count`].
    /// Storage encoding, strides and ragged byte capacity do not affect this comparison.
    #[must_use]
    pub fn has_same_logical_element_count(&self, other: &Self) -> bool {
        tensor_logical_element_counts_match(
            &self.shape,
            self.vector_count.unwrap_or(1),
            &other.shape,
            other.vector_count.unwrap_or(1),
        )
    }

    /// Returns descriptor extents, counting vectors on an active vectorized axis.
    #[must_use]
    pub fn dimensions(&self) -> &[i64] {
        &self.shape.dimensions
    }

    /// Returns descriptor element strides; an active vectorized layout uses vector units.
    #[must_use]
    pub fn strides(&self) -> &[i64] {
        &self.shape.strides
    }

    /// Returns the number of descriptor axes.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Active lane count, or `None` for the native default of one scalar element.
    #[must_use]
    pub fn vector_count(&self) -> Option<i64> {
        self.vector_count
    }

    /// Normalized active vector axis, or `None` when the tensor has no vector lanes.
    #[must_use]
    pub fn vectorized_dimension(&self) -> Option<i64> {
        self.vectorized_dimension
    }

    /// Native storage reordering, separate from vectorization.
    /// Returns `None` when reordering is omitted or explicitly disabled.
    #[must_use]
    pub fn reordering(&self) -> Option<BackendTensorReordering> {
        self.reordering
    }

    /// Returns the native identity of the ragged-offset descriptor, or `None` for an ordinary layout.
    #[must_use]
    pub fn ragged_offset(&self) -> Option<BackendTensorUid> {
        self.ragged_offset
    }

    /// Elements represented by each ragged-offset unit, defaulting to one.
    #[must_use]
    pub fn ragged_offset_multiplier(&self) -> i64 {
        self.ragged_offset_multiplier
    }

    /// Retained physical storage span for ragged layouts, when declared before native construction.
    #[must_use]
    pub fn ragged_byte_capacity(&self) -> Option<usize> {
        self.ragged_byte_capacity
    }

    /// Borrows the finalized native tensor descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

fn tensor_storage_bit_len(shape: &Shape, data_type: DataType, lanes: u128) -> Result<u128> {
    if data_type == DataType::FastFloatForFp8 {
        return Err(Error::UnsupportedDataType);
    }
    let span = shape.storage_span_wide()?;
    span.checked_mul(lanes)
        .and_then(|span| span.checked_mul(data_type.storage_bits() as u128))
        .ok_or_else(|| Error::out_of_range("tensor storage bytes"))
}

mod workspace_data_serde {
    use super::{
        Arc, Deserialize, Deserializer, GraphHostBuffer, Serialize, Serializer, StdResult,
    };

    pub fn serialize<S: Serializer>(
        data: &Arc<GraphHostBuffer>,
        serializer: S,
    ) -> StdResult<S::Ok, S::Error> {
        data.as_bytes().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> StdResult<Arc<GraphHostBuffer>, D::Error> {
        Ok(Arc::new(GraphHostBuffer::new(Vec::<u8>::deserialize(
            deserializer,
        )?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_id_parse_preserves_integer_error_source() {
        use std::error::Error as StdError;

        let error = "not-an-integer".parse::<TensorId>().unwrap_err();
        assert!(matches!(error, Error::TensorIdParse(_)));
        assert!(StdError::source(&error).is_some());
        assert!(matches!(
            "-1".parse::<TensorId>(),
            Err(Error::Frontend(FrontendError::TensorIdInvalid {
                value: -1
            }))
        ));
    }

    #[test]
    fn test_transpose_last_two_swaps_dimensions_and_strides() -> Result<()> {
        let shape = Shape::contiguous([2, 3, 4])?;
        let transposed = shape.transpose_last_two()?;
        assert_eq!(transposed.dimensions(), [2, 4, 3]);
        assert_eq!(transposed.strides(), [12, 1, 4]);
        assert!(Shape::contiguous([5])?.transpose_last_two().is_err());
        Ok(())
    }

    #[test]
    fn test_unit_like_keeps_rank() -> Result<()> {
        let unit = Shape::contiguous([2, 3, 4])?.unit_like()?;
        assert_eq!(unit.dimensions(), [1, 1, 1]);
        assert_eq!(unit.strides(), [1, 1, 1]);
        Ok(())
    }

    #[test]
    fn test_preserving_stride_order_repacks_new_dimensions() -> Result<()> {
        let channels_last = Shape::packed([2, 3, 4, 5], [1, 3, 2, 0])?;
        let resized = channels_last.with_dimensions_preserving_stride_order(vec![2, 6, 4, 5])?;
        assert_eq!(resized.stride_order(), channels_last.stride_order());
        assert_eq!(resized.dimensions(), [2, 6, 4, 5]);
        assert!(
            channels_last
                .with_dimensions_preserving_stride_order(vec![2, 3])
                .is_err()
        );
        Ok(())
    }
}
