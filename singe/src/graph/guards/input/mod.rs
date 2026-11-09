#[cfg(test)]
mod constant;
#[cfg(test)]
mod static_storage;

use std::{collections::HashMap, result::Result as StdResult};

use crate::{
    data_type::DataType,
    device::Device,
    error::Result,
    graph::{
        InputDimensionGuard,
        guards::{
            GraphGuardFailure, InputGuard, StaticInputStorageGuard,
            storage::static_input_storage_guard, validation::check_tensor_metadata,
        },
    },
    layout::Layout,
    tensor::Tensor,
};

impl InputGuard {
    /// Returns the fallback initializer tensor for this input, when present.
    #[must_use]
    pub fn initializer(&self) -> Option<&Tensor> {
        self.initializer.as_ref()
    }

    /// Validates a tensor against this input guard.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor violates guarded metadata or storage capacity.
    pub fn check(&self, tensor: &Tensor) -> Result<()> {
        check_tensor_metadata(
            &self.name,
            self.data_type,
            &self.layout,
            self.device,
            &self.dimension_guards,
            &mut HashMap::new(),
            tensor,
        )
        .map_err(Into::into)
    }

    pub(crate) fn check_with_symbol_values(
        &self,
        tensor: &Tensor,
        symbol_values: &mut HashMap<String, i64>,
    ) -> StdResult<(), GraphGuardFailure> {
        check_tensor_metadata(
            &self.name,
            self.data_type,
            &self.layout,
            self.device,
            &self.dimension_guards,
            symbol_values,
            tensor,
        )
    }

    pub(crate) fn check_static_storage(&self, tensor: &Tensor) -> StdResult<(), GraphGuardFailure> {
        let Some(expected) = self.static_storage else {
            return Ok(());
        };
        let actual = static_input_storage_guard(tensor);
        if expected == actual {
            Ok(())
        } else {
            Err(GraphGuardFailure::StaticInputStorageMismatch {
                name: self.name.clone(),
            })
        }
    }
}

impl InputGuard {
    /// Returns the graph input name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the guarded input data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the guarded input device.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns the guarded input layout.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the required storage span in bytes, including layout offset.
    #[cfg(test)]
    #[must_use]
    pub fn storage_span_bytes(&self) -> usize {
        self.storage_span_bytes
    }

    /// Returns whether this input can be resolved from a graph initializer.
    #[must_use]
    pub fn has_initializer(&self) -> bool {
        self.initializer.is_some()
    }

    /// Returns whether this input is backed by a non-overridable static initializer.
    #[must_use]
    pub fn is_static_initializer(&self) -> bool {
        self.is_static_initializer
    }

    /// Returns process-local static storage identity material for this input.
    #[must_use]
    pub fn static_storage(&self) -> Option<StaticInputStorageGuard> {
        self.static_storage
    }

    /// Returns symbolic dimension guards attached to this input.
    #[must_use]
    pub fn dimension_guards(&self) -> &[InputDimensionGuard] {
        &self.dimension_guards
    }
}
