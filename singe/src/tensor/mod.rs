mod algebra;
mod composite;
mod convolution;
mod creation;
mod data_type;
mod display;
mod elementwise;
mod fourier;
mod id;
mod io;
mod loss;
mod materialization;
mod mutation;
mod neural;
mod operators;
mod overloads;
mod promotion;
mod random;
mod reduction;
mod resample;
mod sampling;
mod spec;
mod views;

use std::{
    fmt::{self, Debug, Formatter},
    sync::{Arc, RwLock},
};

pub(crate) use self::data_type::{tensor_data_type_cases, validate_boolean_bytes};
pub use self::{
    algebra::BlockScaledFormat,
    elementwise::GeluApproximation,
    loss::LossReduction,
    neural::{AttentionOptions, Paging},
    reduction::NormOrder,
    resample::InterpolateMode,
    sampling::SamplingOptions,
};
pub use crate::tensor::{
    creation::{TensorLinspaceElement, TensorRangeElement},
    data_type::TensorDataType,
    id::TensorId,
    spec::TensorSpec,
};
use crate::{
    autograd::AutogradMeta, data_type::DataType, device::Device, error::Result, layout::Layout,
    shape::Shape, storage::Storage, value::ValueSpec,
};

#[derive(Debug)]
pub(crate) struct TensorInner {
    pub(crate) id: TensorId,
    pub(crate) storage: Arc<Storage>,
    pub(crate) data_type: DataType,
    pub(crate) layout: Layout,
    /// Gradient state when the tensor requires gradients, shared by its clones.
    pub(crate) autograd: RwLock<Option<Arc<AutogradMeta>>>,
}

/// A typed, shaped tensor backed by device storage.
///
/// Tensor methods execute eagerly.
/// When called inside [`crate::graph::Graph::capture`], the same methods also record graph nodes for the already-computed operation.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Arc<TensorInner>,
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl Tensor {
    /// Returns logical metadata without exposing storage or placement.
    #[must_use]
    pub fn value_spec(&self) -> ValueSpec {
        ValueSpec::new(self.data_type(), self.shape().clone())
    }

    /// Returns the tensor element data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.inner.data_type
    }

    /// Returns the tensor shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        self.inner.layout.shape()
    }

    /// Returns the physical strides and allocation-relative offset.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.inner.layout
    }

    /// Returns the device that owns the tensor storage.
    #[must_use]
    pub fn device(&self) -> Device {
        self.inner.storage.device()
    }

    /// Returns the number of logical tensor elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.inner.layout.element_count()
    }

    pub(crate) fn storage_offset(&self) -> usize {
        self.inner.layout.storage_offset()
    }

    /// Returns the number of bytes covered by this tensor's shape and data type.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape byte-size calculation overflows.
    pub fn byte_size(&self) -> Result<usize> {
        self.shape().byte_size(self.data_type().size_bytes())
    }

    /// Returns whether logical traversal follows contiguous physical strides.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.inner.layout.is_contiguous()
    }

    /// Returns the backing storage allocation.
    #[must_use]
    pub fn storage(&self) -> &Arc<Storage> {
        &self.inner.storage
    }

    /// Returns this tensor's stable identity.
    #[must_use]
    pub fn id(&self) -> TensorId {
        self.inner.id
    }
}
