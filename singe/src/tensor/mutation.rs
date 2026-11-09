use crate::{
    backend::registry::{Operation, OperationAttributes},
    data_type::DataType,
    dispatch::call,
    error::{Error, Result},
    scalar::Scalar,
    tensor::Tensor,
};

impl Tensor {
    /// Returns the number of writes applied to this tensor's storage.
    ///
    /// Every tensor that shares the storage, including views, reports the same version.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.storage().version()
    }

    /// Writes `source` into the elements this tensor addresses.
    ///
    /// `source` broadcasts to this tensor's shape, converts to its data type, and moves to its device.
    /// The write is visible through every tensor that shares this storage.
    /// A source that overlaps this tensor's storage is copied before the write.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor addresses some element more than once, `source` does not broadcast to its shape, or conversion, transfer, or the backend write fails.
    pub fn assign(&self, source: &Self) -> Result<()> {
        let mut source = source.clone();
        if source.device() != self.device() {
            source = source.to_device(self.device())?;
        }

        if source.data_type() != self.data_type() {
            source = source.cast(self.data_type())?;
        }

        if source.shape() != self.shape() {
            source = source.expand(self.shape())?;
        }

        if source.storage().storage_identity() == self.storage().storage_identity()
            && source.layout().may_overlap(self.layout())
        {
            source = source.copy()?;
        }
        call(
            Operation::Assign,
            OperationAttributes::None,
            &[self, &source],
        )
        .map(drop)
    }

    /// Writes `value` into every element this tensor addresses.
    ///
    /// # Errors
    ///
    /// Returns an error if this tensor addresses some element more than once or the backend write fails.
    pub fn fill_(&self, value: impl Into<Scalar>) -> Result<()> {
        let value = value.into();
        let source = if self.data_type().is_lowp() {
            value.tensor(self.device(), DataType::F32)?
        } else {
            value.tensor(self.device(), self.data_type())?
        };
        self.assign(&source)
    }

    /// Adds `other` to this tensor in place.
    ///
    /// # Errors
    ///
    /// Returns an error if the operands do not broadcast to this tensor's shape, the promoted result cannot be stored in this tensor's data type, or the computation or write fails.
    pub fn add_(&self, other: &Self) -> Result<()> {
        self.store(&self.add(other)?)
    }

    /// Subtracts `other` from this tensor in place.
    ///
    /// # Errors
    ///
    /// Returns an error if the operands do not broadcast to this tensor's shape, the promoted result cannot be stored in this tensor's data type, or the computation or write fails.
    pub fn sub_(&self, other: &Self) -> Result<()> {
        self.store(&self.sub(other)?)
    }

    /// Multiplies this tensor by `other` in place.
    ///
    /// # Errors
    ///
    /// Returns an error if the operands do not broadcast to this tensor's shape, the promoted result cannot be stored in this tensor's data type, or the computation or write fails.
    pub fn mul_(&self, other: &Self) -> Result<()> {
        self.store(&self.mul(other)?)
    }

    /// Divides this tensor by `other` in place.
    ///
    /// # Errors
    ///
    /// Returns an error if the operands do not broadcast to this tensor's shape, the promoted result cannot be stored in this tensor's data type, or the computation or write fails.
    pub fn div_(&self, other: &Self) -> Result<()> {
        self.store(&self.div(other)?)
    }

    /// Clamps this tensor into `[min, max]` in place.
    ///
    /// # Errors
    ///
    /// Returns an error if the bounds are invalid or the computation or write fails.
    pub fn clamp_(&self, min: f32, max: f32) -> Result<()> {
        self.store(&self.clamp(min, max)?)
    }

    /// Writes `value` where `mask` is true.
    ///
    /// # Errors
    ///
    /// Returns an error if `mask` is not Boolean, does not broadcast to this tensor's shape, or the computation or write fails.
    pub fn masked_fill_(&self, mask: &Self, value: impl Into<Scalar>) -> Result<()> {
        let value = value.into().tensor(self.device(), self.data_type())?;
        self.store(&Self::where_select(mask, &value, self)?)
    }

    /// Writes an in-place result after checking that this tensor's data type can hold it.
    fn store(&self, result: &Self) -> Result<()> {
        if result.data_type().category() > self.data_type().category() {
            return Err(Error::data_type_mismatch(
                "in-place result",
                self.data_type(),
                result.data_type(),
            ));
        }
        self.assign(result)
    }
}
