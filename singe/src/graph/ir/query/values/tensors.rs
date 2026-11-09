use crate::{
    error::{Error, Result},
    graph::ir::{Graph, InitializerInputBinding, ValueId},
    tensor::Tensor,
};

impl Graph {
    /// Returns the embedded tensor payload for a constant value.
    pub fn constant_tensor(&self, id: ValueId) -> Option<&Tensor> {
        self.constant_tensors
            .get(id.index())
            .and_then(Option::as_ref)
    }

    /// Returns the required embedded tensor payload for a constant value.
    ///
    /// # Errors
    ///
    /// Returns a missing-input error if `id` has no embedded constant tensor payload.
    pub fn required_constant_tensor(&self, id: ValueId) -> Result<&Tensor> {
        self.constant_tensor(id)
            .ok_or_else(|| Error::missing_input("required constant tensor"))
    }

    /// Returns the initializer tensor payload for an initializer-backed input value.
    #[must_use]
    pub fn initializer_tensor(&self, id: ValueId) -> Option<&Tensor> {
        let binding = self.value(id)?.binding();
        if !binding.is_initializer_backed_input() {
            return None;
        }
        self.initializers.get(binding.input_name()?)
    }

    /// Returns the required initializer tensor payload for an initializer-backed value.
    ///
    /// # Errors
    ///
    /// Returns a missing-input error if `id` has no initializer tensor payload.
    pub fn required_initializer_tensor(&self, id: ValueId) -> Result<&Tensor> {
        self.initializer_tensor(id)
            .ok_or_else(|| Error::missing_input("required initializer tensor"))
    }

    /// Returns initializer-backed graph inputs projected into IR.
    ///
    /// # Errors
    ///
    /// Returns a missing-input error if an initializer-backed IR value lacks an input name or retained initializer tensor.
    pub fn initializer_input_bindings(&self) -> Result<Vec<InitializerInputBinding<'_>>> {
        self.values
            .iter()
            .filter(|value| value.binding().is_initializer_backed_input())
            .map(|value| {
                let binding = value.binding();
                let input = binding
                    .input_name()
                    .ok_or_else(|| Error::missing_input("initializer input bindings"))?;
                let tensor = self.required_initializer_tensor(value.id())?;
                Ok(InitializerInputBinding::new(
                    value.id(),
                    input,
                    tensor,
                    binding.is_static_input(),
                ))
            })
            .collect()
    }

    /// Returns static input names with their initializer tensors.
    ///
    /// # Errors
    ///
    /// Returns a missing-input error if an initializer-backed IR value lacks an input name or retained initializer tensor.
    #[cfg(feature = "cuda")]
    pub fn static_input_initializer_tensors(&self) -> Result<Vec<(&str, &Tensor)>> {
        Ok(self
            .initializer_input_bindings()?
            .into_iter()
            .filter(|binding| binding.is_static())
            .map(|binding| (binding.input(), binding.tensor()))
            .collect())
    }
}
