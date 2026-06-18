use std::{
    borrow::Borrow,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    num::ParseIntError,
    ptr,
    str::FromStr,
    sync::atomic::{AtomicI64, Ordering},
};

use bomboni_serde as serde_helpers;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::{memory::DeviceMemory, types::*};
use singe_cudnn_sys as sys;

use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    context::Context,
    data_type::{DataType, DataTypeLike, DataTypeVectorization},
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::{Error, Result},
    scalar::ScalarValue,
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
    /// This format is only supported with tensor data types [`TensorDataType::Int8x4`](crate::data_type::TensorDataType::Int8x4), [`TensorDataType::Int8x32`](crate::data_type::TensorDataType::Int8x32), and [`TensorDataType::Uint8x4`](crate::data_type::TensorDataType::Uint8x4).
    ///
    /// [`TensorFormat::NchwVectC`] can also be interpreted as N x (C/32) x H x W x 32
    /// for INT8x32, or N x (C/4) x H x W x 4 for INT8x4.
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
    None = sys::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_NONE as _,
    Int8x32 = sys::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_INT8x32 as _,
    F16x16 = sys::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16 as _,
    F8_128x4 = sys::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F8_128x4 as _,
}

impl_enum_conversion!(sys::cudnnBackendTensorReordering_t, BackendTensorReordering);

impl_enum_display!(BackendTensorReordering, {
    Self::None => "CUDNN_TENSOR_REORDERING_NONE",
    Self::Int8x32 => "CUDNN_TENSOR_REORDERING_INT8x32",
    Self::F16x16 => "CUDNN_TENSOR_REORDERING_F16x16",
    Self::F8_128x4 => "CUDNN_TENSOR_REORDERING_F8_128x4",
});

/// Legacy descriptor.
#[derive(Debug)]
#[deprecated]
pub struct TensorDescriptor<T> {
    handle: sys::cudnnTensorDescriptor_t,
    dimensions: Vec<i32>,
    strides: Vec<i32>,
    _t: PhantomData<T>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(i64);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendTensorUid(i64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    dimensions: Vec<i64>,
    strides: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub data_type: DataType,
    pub shape: Shape,
    pub id: Option<TensorId>,
    pub name: Option<String>,
    #[serde(with = "serde_helpers::as_string")]
    pub byte_alignment: i64,
    pub is_virtual: bool,
    pub is_by_value: bool,
    #[serde(with = "serde_helpers::as_string_opt")]
    pub vector_count: Option<i64>,
    #[serde(with = "serde_helpers::as_string_opt")]
    pub vectorized_dimension: Option<i64>,
    pub reordering: Option<BackendTensorReordering>,
    pub ragged_offset: Option<TensorId>,
    pub scalar_value: Option<ScalarValue>,
}

#[derive(Debug)]
pub struct Tensor {
    descriptor: BackendDescriptor,
    id: TensorId,
    data_type: DataType,
    shape: Shape,
    vector_count: Option<i64>,
    vectorized_dimension: Option<i64>,
    ragged_offset: Option<TensorId>,
}

impl TensorId {
    pub const fn new(value: i64) -> Self {
        Self(value)
    }

    pub fn generate() -> Self {
        static NEXT_ID: AtomicI64 = AtomicI64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }

    pub const fn as_i64(self) -> i64 {
        self.0
    }
}

impl BackendTensorUid {
    pub const fn new(value: i64) -> Self {
        Self(value)
    }

    pub const fn as_i64(self) -> i64 {
        self.0
    }

    pub const fn as_tensor_id(self) -> TensorId {
        TensorId::new(self.0)
    }
}

impl From<i64> for TensorId {
    fn from(value: i64) -> Self {
        Self::new(value)
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

impl From<BackendTensorUid> for TensorId {
    fn from(value: BackendTensorUid) -> Self {
        value.as_tensor_id()
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
    type Err = ParseIntError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self(s.parse()?))
    }
}

impl Serialize for TensorId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serde_helpers::as_string::serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for TensorId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self(serde_helpers::as_string::deserialize(deserializer)?))
    }
}

impl<T: DataTypeLike> TensorDescriptor<T> {
    fn checked_contiguous_strides(dimensions: &[i32]) -> Result<Vec<i32>> {
        let mut strides = vec![0; dimensions.len()];
        let mut stride = 1i32;
        for (index, dimension) in dimensions.iter().enumerate().rev() {
            if *dimension == 0 {
                return Err(Error::InvalidDataShape);
            }
            strides[index] = stride;
            stride = stride.checked_mul(*dimension).ok_or(Error::OutOfRange {
                name: "tensor shape".into(),
            })?;
        }
        Ok(strides)
    }

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

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cudnnCreateTensorDescriptor(&raw mut handle))?;
            try_ffi!(sys::cudnnSetTensorNdDescriptor(
                handle,
                T::data_type().into(),
                to_i32(dimensions.len(), "tensor rank")?,
                dimensions.as_ptr(),
                strides.as_ptr(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            dimensions: dimensions.to_vec(),
            strides: strides.to_vec(),
            _t: PhantomData,
        })
    }

    pub fn create_contiguous(dimensions: &[i32]) -> Result<Self> {
        if dimensions.is_empty() {
            return Err(Error::InvalidDataShape);
        }

        let strides = Self::checked_contiguous_strides(dimensions)?;
        Self::create_strided(dimensions, &strides)
    }

    pub fn create_vectorized<V: DataTypeVectorization>(dimensions: &[i32]) -> Result<Self> {
        if dimensions.len() < 2 {
            return Err(Error::InvalidDataShape);
        }

        let data_type = T::tensor_data_type::<V>().ok_or(Error::UnsupportedDataType)?;

        if dimensions[1] == 0 || dimensions[1] % V::WIDTH as i32 != 0 {
            return Err(Error::InvalidDataShape);
        }

        let strides = Self::checked_contiguous_strides(dimensions)?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cudnnCreateTensorDescriptor(&raw mut handle))?;
            try_ffi!(sys::cudnnSetTensorNdDescriptorEx(
                handle,
                TensorFormat::NchwVectC.into(),
                data_type.try_into()?,
                to_i32(dimensions.len(), "tensor rank")?,
                dimensions.as_ptr(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            dimensions: dimensions.to_vec(),
            strides,
            _t: PhantomData,
        })
    }
}

impl<T> TensorDescriptor<T> {
    pub fn dimensions(&self) -> &[i32] {
        &self.dimensions
    }

    pub fn strides(&self) -> &[i32] {
        &self.strides
    }

    pub fn rank(&self) -> i32 {
        self.dimensions.len() as i32
    }

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
    pub fn as_raw(&self) -> sys::cudnnTensorDescriptor_t {
        self.handle
    }

    /// Takes ownership of a raw cuDNN tensor descriptor handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cudnnTensorDescriptor_t` created by cuDNN.
    /// The returned wrapper takes ownership and will destroy it with
    /// `cudnnDestroyTensorDescriptor`; no other owner may destroy or keep using
    /// it. `dimensions` and `strides` must accurately describe the descriptor.
    pub unsafe fn from_raw(
        handle: sys::cudnnTensorDescriptor_t,
        dimensions: Vec<i32>,
        strides: Vec<i32>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        if dimensions.len() != strides.len() {
            return Err(Error::LengthMismatch {
                name: "strides".into(),
                expected: dimensions.len(),
                actual: strides.len(),
            });
        }

        Ok(Self {
            handle,
            dimensions,
            strides,
            _t: PhantomData,
        })
    }

    /// Releases ownership and returns the raw cuDNN tensor descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::cudnnTensorDescriptor_t {
        let handle = self.handle;
        std::mem::forget(self);
        handle
    }
}

impl<T> Drop for TensorDescriptor<T> {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cudnnDestroyTensorDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy tensor descriptor: {err}");
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
pub fn set_tensor<T: DataTypeLike>(
    ctx: &Context,
    y_desc: &TensorDescriptor<T>,
    y: &mut DeviceMemory<T>,
    value: &T,
) -> Result<()> {
    ctx.bind()?;

    unsafe {
        try_ffi!(sys::cudnnSetTensor(
            ctx.as_raw(),
            y_desc.as_raw(),
            y.as_mut_ptr() as _,
            value as *const T as _,
        ))?;
    }
    Ok(())
}

impl Shape {
    pub fn contiguous<I, T>(dimensions: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Borrow<i64>,
    {
        let dimensions: Vec<i64> = dimensions.into_iter().map(|x| *x.borrow()).collect();
        let strides = checked_contiguous_strides(&dimensions)?;

        Ok(Self {
            dimensions,
            strides,
        })
    }

    pub fn with_strides<I, T>(mut self, strides: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Borrow<i64>,
    {
        let strides: Vec<i64> = strides.into_iter().map(|x| *x.borrow()).collect();
        if self.dimensions.len() != strides.len() {
            return Err(Error::LengthMismatch {
                name: "tensor strides".into(),
                expected: self.dimensions.len(),
                actual: strides.len(),
            });
        }
        validate_strides(&strides)?;

        self.strides = strides;
        Ok(self)
    }

    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    pub fn element_count(&self) -> Result<usize> {
        checked_element_count(&self.dimensions)
    }

    pub fn rank(&self) -> i64 {
        self.dimensions.len() as i64
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
            count.checked_mul(dimension).ok_or(Error::OutOfRange {
                name: "tensor shape".into(),
            })
        })
}

fn checked_element_count(dimensions: &[i64]) -> Result<usize> {
    dimensions
        .iter()
        .copied()
        .try_fold(1usize, |count, dimension| {
            let dimension = usize::try_from(dimension).map_err(|_| Error::InvalidDataShape)?;
            count.checked_mul(dimension).ok_or(Error::OutOfRange {
                name: "tensor shape".into(),
            })
        })
}

fn validate_strides(strides: &[i64]) -> Result<()> {
    if strides.iter().any(|&stride| stride <= 0) {
        return Err(Error::InvalidDataStrides);
    }
    Ok(())
}

fn checked_contiguous_strides(dimensions: &[i64]) -> Result<Vec<i64>> {
    validate_dimensions(dimensions)?;

    let mut strides = vec![0_i64; dimensions.len()];
    let mut stride = 1_i64;
    for (index, dimension) in dimensions.iter().enumerate().rev() {
        strides[index] = stride;
        if index != 0 {
            stride = stride
                .checked_mul(*dimension)
                .ok_or_else(|| Error::OutOfRange {
                    name: "tensor strides".into(),
                })?;
        }
    }

    Ok(strides)
}

impl TensorSpec {
    pub fn new(data_type: DataType, shape: Shape) -> Self {
        Self {
            data_type,
            shape,
            name: None,
            id: None,
            byte_alignment: 16,
            is_virtual: false,
            is_by_value: false,
            vector_count: None,
            vectorized_dimension: None,
            reordering: None,
            ragged_offset: None,
            scalar_value: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<TensorId>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn id(&self) -> Option<TensorId> {
        self.id
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.set_name(name);
        self
    }

    pub fn clear_name(&mut self) {
        self.name = None;
    }

    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }

    pub fn with_byte_alignment(mut self, byte_alignment: i64) -> Result<Self> {
        check_range!("byte_alignment", byte_alignment > 0)?;
        self.byte_alignment = byte_alignment;
        Ok(self)
    }

    pub fn with_vectorization(
        mut self,
        vector_count: i64,
        vectorized_dimension: i64,
    ) -> Result<Self> {
        check_range!("vector_count", vector_count > 0)?;
        let rank = to_i64(self.shape.dimensions().len(), "tensor rank")?;
        let normalized_dimension = if vectorized_dimension < 0 {
            rank.checked_add(vectorized_dimension)
                .ok_or_else(|| Error::OutOfRange {
                    name: "vectorized_dimension".into(),
                })?
        } else {
            vectorized_dimension
        };
        check_range!(
            "vectorized_dimension",
            normalized_dimension >= 0 && normalized_dimension < rank
        )?;
        self.vector_count = Some(vector_count);
        self.vectorized_dimension = Some(normalized_dimension);
        Ok(self)
    }

    pub fn with_reordering(mut self, reordering: BackendTensorReordering) -> Self {
        self.reordering = Some(reordering);
        self
    }

    pub(crate) fn with_ragged_offset(mut self, ragged_offset: TensorId) -> Self {
        self.ragged_offset = Some(ragged_offset);
        self
    }

    pub(crate) fn clear_ragged_offset(mut self) -> Self {
        self.ragged_offset = None;
        self
    }

    pub fn with_f8_128x4_reordering(self) -> Self {
        self.with_reordering(BackendTensorReordering::F8_128x4)
    }

    pub fn with_vectorization_128x4(self) -> Result<Self> {
        self.with_vectorization(4, -1)
    }

    pub fn block_scale_tensor(data_type: DataType, shape: Shape) -> Self {
        let mut tensor = Self::new(data_type, shape).with_f8_128x4_reordering();
        tensor.vector_count = Some(4);
        tensor.vectorized_dimension = tensor
            .shape
            .dimensions()
            .len()
            .checked_sub(1)
            .map(|rank| rank as i64);
        tensor
    }

    pub fn scalar_f32(value: f32) -> Result<Self> {
        Self::new(DataType::F32, Shape::contiguous([1])?).with_scalar_value(ScalarValue::F32(value))
    }

    pub fn scalar_i32(value: i32) -> Result<Self> {
        Self::new(DataType::I32, Shape::contiguous([1])?).with_scalar_value(ScalarValue::I32(value))
    }

    pub fn scalar_f8e4m3(value: f8e4m3) -> Result<Self> {
        Self::new(DataType::F8E4M3, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::from_f8e4m3(value))
    }

    pub fn scalar_f8e5m2(value: f8e5m2) -> Result<Self> {
        Self::new(DataType::F8E5M2, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::from_f8e5m2(value))
    }

    pub fn scalar_f8ue8m0(value: f8ue8m0) -> Result<Self> {
        Self::new(DataType::F8E8M0, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::from_f8ue8m0(value))
    }

    pub fn scalar_f4e2m1(value: f4e2m1) -> Result<Self> {
        Self::new(DataType::F4E2M1, Shape::contiguous([1])?)
            .with_scalar_value(ScalarValue::from_f4e2m1(value))
    }

    pub fn virtual_tensor(mut self) -> Self {
        self.is_virtual = true;
        self
    }

    pub fn output_tensor(mut self) -> Self {
        self.is_virtual = false;
        self
    }

    pub fn pass_by_value(mut self) -> Self {
        self.is_by_value = true;
        self
    }

    pub fn with_scalar_value(mut self, scalar_value: ScalarValue) -> Result<Self> {
        if scalar_value.data_type() != self.data_type {
            return Err(Error::FrontendScalarTypeMismatch {
                tensor_id: self.id.unwrap_or_else(|| TensorId::new(0)),
                expected: self.data_type.type_name().into(),
                actual: scalar_value.type_name().into(),
            });
        }

        self.is_by_value = true;
        self.scalar_value = Some(scalar_value);
        Ok(self)
    }
}

impl Tensor {
    pub fn create_with_spec(spec: &TensorSpec) -> Result<Self> {
        let id = spec.id.ok_or_else(|| Error::DescriptorMismatch {
            name: "backend tensor id".into(),
        })?;

        Self::create_with_resolved_spec(id, spec, spec.vectorized_dimension, spec.vector_count)
    }

    pub(crate) fn create_with_resolved_spec(
        id: TensorId,
        spec: &TensorSpec,
        vectorized_dimension: Option<i64>,
        vector_count: Option<i64>,
    ) -> Result<Self> {
        let shape = &spec.shape;

        if let Some(vectorized_dimension) = vectorized_dimension
            && (vectorized_dimension < 0 || vectorized_dimension >= shape.rank())
        {
            return Err(Error::OutOfRange {
                name: "vectorized_dimension".into(),
            });
        }
        if let Some(vector_count) = vector_count
            && vector_count <= 0
        {
            return Err(Error::OutOfRange {
                name: "vector_count".into(),
            });
        }
        if vectorized_dimension.is_some() != vector_count.is_some() {
            return Err(Error::DescriptorMismatch {
                name: "backend tensor vectorization".into(),
            });
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Tensor)?;
        descriptor.set_attribute_i64(BackendAttributeName::TensorUniqueId, id.as_i64())?;
        descriptor.set_attribute_enum(
            BackendAttributeName::TensorDataType,
            BackendAttributeType::DataType,
            spec.data_type,
        )?;
        descriptor
            .set_attribute_i64_slice(BackendAttributeName::TensorDimensions, &shape.dimensions)?;
        descriptor.set_attribute_i64_slice(BackendAttributeName::TensorStrides, &shape.strides)?;
        descriptor.set_attribute_i64(
            BackendAttributeName::TensorByteAlignment,
            spec.byte_alignment,
        )?;
        descriptor.set_attribute_bool(BackendAttributeName::TensorIsVirtual, spec.is_virtual)?;
        descriptor.set_attribute_bool(BackendAttributeName::TensorIsByValue, spec.is_by_value)?;
        if let Some(vectorized_dimension) = vectorized_dimension {
            descriptor.set_attribute_i64(
                BackendAttributeName::TensorVectorizedDimension,
                vectorized_dimension,
            )?;
        }
        if let Some(vector_count) = vector_count {
            descriptor.set_attribute_i64(BackendAttributeName::TensorVectorCount, vector_count)?;
        }
        if let Some(reordering) = spec.reordering {
            descriptor.set_attribute_enum(
                BackendAttributeName::TensorReorderingMode,
                BackendAttributeType::TensorReorderingMode,
                reordering,
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            id,
            data_type: spec.data_type,
            shape: shape.clone(),
            vector_count,
            vectorized_dimension,
            ragged_offset: spec.ragged_offset,
        })
    }

    pub fn create(
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
    ) -> Result<Self> {
        Self::create_with_spec(&TensorSpec {
            data_type,
            shape: Shape::contiguous(dimensions)?.with_strides(strides)?,
            id: Some(TensorId::generate()),
            name: None,
            byte_alignment,
            is_virtual: false,
            is_by_value: false,
            vector_count: None,
            vectorized_dimension: None,
            reordering: None,
            ragged_offset: None,
            scalar_value: None,
        })
    }

    pub fn create_with_id(
        id: impl Into<TensorId>,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
    ) -> Result<Self> {
        Self::create_with_spec(&TensorSpec {
            data_type,
            shape: Shape::contiguous(dimensions)?.with_strides(strides)?,
            id: Some(id.into()),
            name: None,
            byte_alignment,
            is_virtual: false,
            is_by_value: false,
            vector_count: None,
            vectorized_dimension: None,
            reordering: None,
            ragged_offset: None,
            scalar_value: None,
        })
    }

    pub fn create_with_reordering(
        id: impl Into<TensorId>,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
        reordering: BackendTensorReordering,
    ) -> Result<Self> {
        Self::create_with_spec(&TensorSpec {
            data_type,
            shape: Shape::contiguous(dimensions)?.with_strides(strides)?,
            id: Some(id.into()),
            name: None,
            byte_alignment,
            is_virtual: false,
            is_by_value: false,
            vector_count: None,
            vectorized_dimension: None,
            reordering: Some(reordering),
            ragged_offset: None,
            scalar_value: None,
        })
    }

    pub fn create_vectorized(
        id: impl Into<TensorId>,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
        vectorized_dimension: i64,
        vector_count: i64,
        reordering: Option<BackendTensorReordering>,
    ) -> Result<Self> {
        Self::create_with_spec(&TensorSpec {
            data_type,
            shape: Shape::contiguous(dimensions)?.with_strides(strides)?,
            id: Some(id.into()),
            name: None,
            byte_alignment,
            is_virtual: false,
            is_by_value: false,
            vector_count: Some(vector_count),
            vectorized_dimension: Some(vectorized_dimension),
            reordering,
            ragged_offset: None,
            scalar_value: None,
        })
    }

    pub fn create_virtual(
        id: impl Into<TensorId>,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
    ) -> Result<Self> {
        Self::create_with_spec(&TensorSpec {
            data_type,
            shape: Shape::contiguous(dimensions)?.with_strides(strides)?,
            id: Some(id.into()),
            name: None,
            byte_alignment,
            is_virtual: true,
            is_by_value: false,
            vector_count: None,
            vectorized_dimension: None,
            reordering: None,
            ragged_offset: None,
            scalar_value: None,
        })
    }

    pub fn create_virtual_with_reordering(
        id: impl Into<TensorId>,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
        reordering: BackendTensorReordering,
    ) -> Result<Self> {
        Self::create_with_spec(&TensorSpec {
            data_type,
            shape: Shape::contiguous(dimensions)?.with_strides(strides)?,
            id: Some(id.into()),
            name: None,
            byte_alignment,
            is_virtual: true,
            is_by_value: false,
            vector_count: None,
            vectorized_dimension: None,
            reordering: Some(reordering),
            ragged_offset: None,
            scalar_value: None,
        })
    }

    pub fn create_by_value(
        id: impl Into<TensorId>,
        data_type: DataType,
        dimensions: &[i64],
        strides: &[i64],
        byte_alignment: i64,
    ) -> Result<Self> {
        Self::create_with_spec(&TensorSpec {
            data_type,
            shape: Shape::contiguous(dimensions)?.with_strides(strides)?,
            id: Some(id.into()),
            name: None,
            byte_alignment,
            is_virtual: false,
            is_by_value: true,
            vector_count: None,
            vectorized_dimension: None,
            reordering: None,
            ragged_offset: None,
            scalar_value: None,
        })
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dimensions(&self) -> &[i64] {
        &self.shape.dimensions
    }

    pub fn strides(&self) -> &[i64] {
        &self.shape.strides
    }

    pub fn rank(&self) -> i64 {
        self.shape.rank()
    }

    pub fn vector_count(&self) -> Option<i64> {
        self.vector_count
    }

    pub fn vectorized_dimension(&self) -> Option<i64> {
        self.vectorized_dimension
    }

    pub fn ragged_offset(&self) -> Option<TensorId> {
        self.ragged_offset
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
