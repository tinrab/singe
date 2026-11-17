//! Native tensor identities, reordering, and finalized tensor descriptors.

mod problem;

use std::fmt::{self, Display, Formatter};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::impl_enum_display;
use singe_cudnn_sys as sys;

pub use self::problem::TensorProblemKey;
use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
    },
    data_type::DataType,
    error::{Error, Result},
    scalar::{ScalarStorage, ScalarValue},
    tensor::{
        Shape, TensorId, TensorPlacement, TensorSpec, byte_len_bits, tensor_logical_element_count,
        tensor_logical_element_counts_match, tensor_storage_bit_len,
    },
    utility::to_usize,
};

singe_core::native_enum! {
    #[derive(
        Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
    )]
    /// Tensor reordering property of a tensor descriptor.
    /// This property is available on [`BackendDescriptorType::Tensor`] through [`BackendDescriptor`] attribute methods.
    #[repr(u32)]
    #[non_exhaustive]
    pub enum BackendTensorReordering: sys::cudnnBackendTensorReordering_t {
        /// No tensor reordering applied.
        None = CUDNN_TENSOR_REORDERING_NONE,
        /// INT8 data reordered into 32-element vectors.
        Int8x32 = CUDNN_TENSOR_REORDERING_INT8x32,
        /// FP16 data reordered into 16-element vectors.
        F16x16 = CUDNN_TENSOR_REORDERING_F16x16,
        /// FP8 data reordered into 128x4 blocks.
        F8_128x4 = CUDNN_TENSOR_REORDERING_F8_128x4,
    }
}

impl_enum_display!(BackendTensorReordering, {
    Self::None => "CUDNN_TENSOR_REORDERING_NONE",
    Self::Int8x32 => "CUDNN_TENSOR_REORDERING_INT8x32",
    Self::F16x16 => "CUDNN_TENSOR_REORDERING_F16x16",
    Self::F8_128x4 => "CUDNN_TENSOR_REORDERING_F8_128x4",
});

/// Native cuDNN tensor identity used by variant packs and native descriptor relations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendTensorUid(i64);

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

impl Display for BackendTensorUid {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)?;
        Ok(())
    }
}

impl Tensor {
    /// Creates a native tensor with the cuDNN UID used by variant packs and native tensor relations.
    /// Frontend identities must be converted explicitly at the lowering boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor metadata is invalid, a ragged descriptor is missing, scalar allocation fails, or cuDNN rejects the descriptor.
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
            .set_attribute_i64_slice(BackendAttributeName::TensorDimensions, shape.dimensions())?;
        descriptor.set_attribute_i64_slice(BackendAttributeName::TensorStrides, shape.strides())?;
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
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes.
    /// Ordinary tensor storage is supplied separately.
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
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes.
    /// Ordinary tensor storage is supplied separately.
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
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes.
    /// Ordinary tensor storage is supplied separately.
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
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes.
    /// Ordinary tensor storage is supplied separately.
    ///
    /// # Errors
    ///
    /// Returns an error if layout metadata, storage encoding, or byte alignment is invalid, or native descriptor creation, attribute assignment, or finalization fails.
    /// Vector counts must be positive, and an active vector axis must resolve within the descriptor rank.
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
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes.
    /// Ordinary tensor storage is supplied separately.
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
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes.
    /// Ordinary tensor storage is supplied separately.
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
    /// Dimensions and strides describe descriptor elements; alignment is measured in bytes.
    /// Ordinary tensor storage is supplied separately.
    ///
    /// # Errors
    ///
    /// Returns an error if layout metadata, storage encoding, or byte alignment is invalid, or native descriptor creation, attribute assignment, or finalization fails.
    /// The declaration must contain exactly one logical scalar.
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
    ///
    /// Returns an error if the ragged capacity is missing or shape and size arithmetic overflows.
    pub fn storage_bit_len(&self) -> Result<u128> {
        if self.ragged_offset.is_some() {
            return byte_len_bits(self.storage_byte_len(None)?);
        }

        let lanes = u128::try_from(self.vector_count.unwrap_or(1))
            .map_err(|_| Error::out_of_range("tensor vector count"))?;
        tensor_storage_bit_len(&self.shape, self.data_type, lanes)
    }

    /// Physical bytes required by this descriptor, rounding packed tails up once.
    /// Packed sizing does not require the span in element positions to fit `usize`.
    /// Ragged tensors use their retained byte capacity, or the supplied capacity when none was declared.
    /// Nominal dimensions and offset metadata do not establish ragged allocation size.
    ///
    /// # Errors
    ///
    /// Returns an error if the ragged capacity is missing or shape and byte counts overflow.
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
        self.shape.dimensions()
    }

    /// Returns descriptor element strides; an active vectorized layout uses vector units.
    #[must_use]
    pub fn strides(&self) -> &[i64] {
        self.shape.strides()
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
