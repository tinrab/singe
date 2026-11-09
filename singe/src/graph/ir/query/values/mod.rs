mod alias;
mod binding;
mod guards;
mod input;
mod outputs;
mod tensors;

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
};

#[cfg(any(test, feature = "cuda"))]
use crate::graph::ir::ValueAlias;
use crate::graph::ir::{Graph, Value, ValueId};

impl Graph {
    /// Returns all IR values.
    #[must_use]
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Returns one IR value by id.
    #[must_use]
    pub fn value(&self, id: ValueId) -> Option<&Value> {
        self.values.get(id.index())
    }

    /// Returns one required IR value by id.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` does not identify an IR value.
    pub fn required_value(&self, id: ValueId) -> Result<&Value> {
        self.value(id)
            .ok_or_else(|| Error::missing_node("required value"))
    }

    /// Returns one required IR value data type by value id.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` does not identify an IR value.
    pub fn required_value_data_type(&self, id: ValueId) -> Result<DataType> {
        Ok(self.required_value(id)?.data_type())
    }

    /// Returns one required IR value name by value id.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` does not identify an IR value.
    pub fn required_value_name(&self, id: ValueId) -> Result<Option<&str>> {
        Ok(self.required_value(id)?.name())
    }

    /// Returns one required IR value shape by value id.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` does not identify an IR value.
    pub fn required_value_shape(&self, id: ValueId) -> Result<&Shape> {
        Ok(self.required_value(id)?.shape())
    }

    /// Returns one required IR value layout by value id.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` does not identify an IR value.
    pub fn required_value_layout(&self, id: ValueId) -> Result<&Layout> {
        Ok(self.required_value(id)?.layout())
    }

    /// Returns the device of value `id`.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` does not identify an IR value.
    pub fn required_value_device(&self, id: ValueId) -> Result<Device> {
        Ok(self.required_value(id)?.device())
    }

    /// Returns one required IR value alias by value id.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` does not identify an IR value.
    #[cfg(any(test, feature = "cuda"))]
    pub fn required_value_alias(&self, id: ValueId) -> Result<Option<&ValueAlias>> {
        Ok(self.required_value(id)?.alias())
    }
}
