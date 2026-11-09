use crate::{
    error::Result,
    graph::ir::{Graph, ValueBinding, ValueBindingDetails, ValueBindingKind, ValueId},
};

impl Graph {
    /// Returns one required IR value binding category by value id.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` does not identify an IR value.
    pub fn required_value_binding_kind(&self, id: ValueId) -> Result<ValueBindingKind> {
        Ok(self.required_value_binding(id)?.kind())
    }

    /// Returns required IR value binding details by value id.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` does not identify an IR value.
    pub fn required_value_binding_details(&self, id: ValueId) -> Result<ValueBindingDetails<'_>> {
        Ok(ValueBindingDetails::new(self.required_value_binding(id)?))
    }

    /// Returns whether one IR value is an input binding.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn value_is_input(&self, id: ValueId) -> bool {
        self.value(id)
            .is_some_and(|value| value.binding().is_input())
    }

    /// Returns whether one IR value is a constant binding.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn value_is_constant(&self, id: ValueId) -> bool {
        self.value(id)
            .is_some_and(|value| value.binding().is_constant())
    }

    /// Returns one required IR value binding by value id.
    fn required_value_binding(&self, id: ValueId) -> Result<&ValueBinding> {
        Ok(self.required_value(id)?.binding())
    }
}
