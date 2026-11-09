mod range;

use std::sync::{Arc, RwLock};

pub use self::range::{TensorLinspaceElement, TensorRangeElement};

use crate::{
    backend::{
        from_bytes_on,
        registry::{Operation, OperationAttributes},
        zeros_on,
    },
    data_type::DataType,
    device::Device,
    dispatch::call_one,
    error::{Error, Result},
    layout::Layout,
    scalar::Scalar,
    shape::Shape,
    storage::{Storage, cpu::CpuStorage},
    tensor::{
        Tensor, TensorDataType, TensorId, TensorInner, TensorSpec,
        data_type::tensor_data_type_cases, validate_boolean_bytes,
    },
};

impl Tensor {
    /// Concatenates tensors along `dim`.
    ///
    /// This records a graph operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if `tensors` is empty, shapes/devices/data types are incompatible, `dim` is out of range, or backend execution fails.
    pub fn concat(tensors: &[Self], dim: i64) -> Result<Self> {
        if tensors.is_empty() {
            return Err(Error::InvalidArgument {
                name: "tensor concat inputs".into(),
            });
        }
        call_one(
            Operation::Concat,
            OperationAttributes::Concat(dim),
            &tensors.iter().collect::<Vec<_>>(),
        )
    }
}

impl Tensor {
    /// Creates a tensor filled with `1`.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation or device execution fails.
    pub fn ones<T: TensorDataType>(device: Device, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let shape = &shape;
        Self::full(device, shape, T::from_scalar(Scalar::Bool(true)))
    }

    /// Creates a rank-zero scalar tensor with shape `[]`.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation or device execution fails.
    pub fn scalar<T: TensorDataType>(device: Device, value: T) -> Result<Self> {
        Self::from_vec(device, Shape::scalar(), vec![value])
    }

    /// Creates a tensor from raw host bytes.
    ///
    /// The byte slice must contain exactly `shape.element_count() * data_type.size_bytes()` bytes.
    /// The bytes are copied into storage owned by `device` with contiguous layout.
    /// Use checked storage construction to wrap an existing strided allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` length does not match `shape` and `data_type`, the shape byte size overflows, the device/backend is unsupported, or allocation/copying fails.
    pub fn from_bytes(
        device: Device,
        data_type: DataType,
        shape: impl Into<Shape>,
        data: &[u8],
    ) -> Result<Self> {
        let shape = shape.into();
        validate_host_bytes("tensor from_bytes data", &shape, data_type, data.len())?;
        validate_boolean_bytes(data_type, data)?;
        from_bytes_on(device, data_type, &shape, data)
    }

    /// Creates a tensor from typed host data.
    ///
    /// The vector length must match `shape.element_count()`.
    /// The vector is interpreted using `T::data_type()` with contiguous layout.
    /// CPU storage takes ownership of the vector; CUDA storage receives an explicit copy.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len()` does not match `shape.element_count()`, the shape byte size overflows, the device/backend is unsupported, or allocation/copying fails.
    pub fn from_vec<T: TensorDataType>(
        device: Device,
        shape: impl Into<Shape>,
        data: Vec<T>,
    ) -> Result<Self> {
        let shape = shape.into();
        validate_host_elements("tensor from_vec data", &shape, data.len())?;
        if device.is_cpu() {
            let storage = CpuStorage::from_values(data);
            // SAFETY: the storage holds exactly the vector's `T` values, and validate_host_elements checked its length and byte size.
            return Ok(unsafe { Self::from_parts_unchecked(storage, T::data_type(), shape) });
        }
        from_bytes_on(device, T::data_type(), &shape, T::as_bytes(&data))
    }

    /// Creates a tensor filled with `value`.
    ///
    /// The data type is inferred from `T`, and the resulting tensor is allocated on `device` with contiguous layout.
    ///
    /// # Errors
    ///
    /// Returns an error if shape byte-size validation, allocation, or device execution fails.
    pub fn full<T: TensorDataType>(
        device: Device,
        shape: impl Into<Shape>,
        value: T,
    ) -> Result<Self> {
        let shape = shape.into();
        shape.byte_size(T::data_type().size_bytes())?;
        let element_count = shape.element_count();
        Self::from_vec(device, shape, vec![value; element_count])
    }
}

fn validate_host_bytes(
    name: &'static str,
    shape: &Shape,
    data_type: DataType,
    actual: usize,
) -> Result<()> {
    let expected = shape.byte_size(data_type.size_bytes())?;
    if actual != expected {
        return Err(Error::length_mismatch(name, expected, actual));
    }
    Ok(())
}

fn validate_host_elements(name: &'static str, shape: &Shape, actual: usize) -> Result<()> {
    let expected = shape.element_count();
    if actual != expected {
        return Err(Error::length_mismatch(name, expected, actual));
    }
    Ok(())
}

macro_rules! tensor_one_like_cases {
    ($tensor:expr; $($ty:ty => $variant:ident = $one:expr),+ $(,)?) => {
        match $tensor.data_type() {
            $(DataType::$variant => $tensor.full_like::<$ty>($one),)+
            DataType::F8E4M3 | DataType::F8E5M2 | DataType::F4E2M1 => {
                Err(Error::unsupported_data_type("tensor construction"))
            }
        }
    };
}

impl Tensor {
    fn contiguous_like_shape(&self) -> Result<Shape> {
        Shape::create(self.shape().dimensions().to_vec())
    }

    /// Creates a zero-filled tensor with this tensor's shape, data type, and device.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeroed(TensorSpec::create(
            self.device(),
            self.data_type(),
            self.contiguous_like_shape()?,
        )?)
    }

    /// Creates a tensor filled with `1` using this tensor's shape, data type, and device.
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is unsupported for a one value or allocation fails.
    pub fn ones_like(&self) -> Result<Self> {
        tensor_data_type_cases!(tensor_one_like_cases, self)
    }

    /// Creates a tensor filled with `value` using this tensor's shape and device.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` has a different data type from this tensor or allocation fails.
    pub fn full_like<T: TensorDataType>(&self, value: T) -> Result<Self> {
        if self.data_type() != T::data_type() {
            return Err(Error::unsupported_data_type("tensor construction"));
        }
        Self::full(self.device(), self.contiguous_like_shape()?, value)
    }
}

impl Tensor {
    /// Creates a tensor without validating storage, layout, data type, or device invariants.
    ///
    /// This is for backend fast paths that just allocated `storage` from the same checked layout and data type.
    /// Public construction should use [`Tensor::create`], [`Tensor::from_storage`], or one of the host-data constructors.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// - `storage` holds valid `data_type` values wherever `layout` addresses it.
    /// - `storage` is owned by the device intended for the tensor.
    /// - `layout` has already been validated for `data_type`.
    /// - `storage` contains at least `layout.storage_span_byte_size(data_type.size_bytes())` bytes.
    /// - No user-provided metadata can make the tensor view exceed the storage allocation.
    pub(crate) unsafe fn from_parts_unchecked(
        storage: impl Into<Storage>,
        data_type: DataType,
        layout: impl Into<Layout>,
    ) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                id: TensorId::generate(),
                storage: Arc::new(storage.into()),
                data_type,
                layout: layout.into(),
                autograd: RwLock::default(),
            }),
        }
    }

    /// Creates a tensor from owned storage and layout metadata.
    ///
    /// The tensor takes ownership of `storage` and validates that the allocation is large enough for `layout` with `data_type` elements.
    /// Use [`Tensor::from_storage`] when the expected device and data type should be checked against an explicit [`TensorSpec`].
    ///
    /// # Errors
    ///
    /// Returns an error if the layout byte span overflows or if the storage allocation is too small for the storage span implied by `layout` and `data_type`.
    pub fn create(
        storage: impl Into<Storage>,
        data_type: DataType,
        layout: impl Into<Layout>,
    ) -> Result<Self> {
        Self::create_from_storage(Arc::new(storage.into()), data_type, layout)
    }

    /// Creates a tensor from shared storage and layout metadata.
    ///
    /// This is the shared-storage form of [`Tensor::create`].
    /// The storage device defines the tensor device.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout byte span overflows or if the storage allocation is too small for the storage span implied by `layout` and `data_type`.
    pub fn create_from_storage(
        storage: Arc<Storage>,
        data_type: DataType,
        layout: impl Into<Layout>,
    ) -> Result<Self> {
        let layout = layout.into();
        Self::validate_storage_span(&storage, data_type, &layout)?;
        Ok(Self {
            inner: Arc::new(TensorInner {
                id: TensorId::generate(),
                storage,
                data_type,
                layout,
                autograd: RwLock::default(),
            }),
        })
    }

    /// Creates a tensor from storage that must match `spec`.
    ///
    /// This is the checked boundary for wrapping existing allocations.
    /// It rejects storage whose device or byte capacity cannot satisfy the requested tensor metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if storage device or byte capacity does not satisfy `spec`.
    pub fn from_storage(spec: TensorSpec, storage: Arc<Storage>) -> Result<Self> {
        if storage.device() != spec.device {
            return Err(Error::device_mismatch(
                "tensor storage device",
                spec.device,
                storage.device(),
            ));
        }
        Self::create_from_storage(storage, spec.data_type, spec.shape)
    }

    pub(crate) fn validate_storage_span(
        storage: &Storage,
        data_type: DataType,
        layout: &Layout,
    ) -> Result<()> {
        let required = layout.storage_span_byte_size(data_type.size_bytes())?;
        if required > storage.byte_size() {
            return Err(Error::length_mismatch(
                "tensor storage bytes",
                required,
                storage.byte_size(),
            ));
        }
        Ok(())
    }
}

impl Tensor {
    /// Creates a zero-filled tensor from validated tensor metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the device/backend is unsupported or allocation fails.
    pub fn zeroed(spec: TensorSpec) -> Result<Self> {
        Self::zeroed_on(spec.device, spec.data_type, spec.shape)
    }

    /// Creates a zero-filled tensor from separate metadata values.
    ///
    /// Use [`Tensor::zeroed`] when device, data type, and shape have already been grouped into a [`TensorSpec`].
    /// The result has contiguous layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the device/backend is unsupported, the shape byte size overflows, or allocation fails.
    pub fn zeroed_on(device: Device, data_type: DataType, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        zeros_on(device, data_type, &shape)
    }
}
