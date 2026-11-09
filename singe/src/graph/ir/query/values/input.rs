use std::collections::HashMap;

use crate::{
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    tensor::Tensor,
};

impl Graph {
    /// Returns each graph input value with its input name, in value order.
    pub fn input_values(&self) -> impl Iterator<Item = (ValueId, &str)> {
        self.values
            .iter()
            .filter_map(|value| Some((value.id(), value.binding().input_name()?)))
    }

    /// Binds tensors resolved by input name to the input values that read them.
    ///
    /// # Errors
    ///
    /// Returns a missing-input error if an input value has no resolved tensor.
    pub fn bind_inputs(
        &self,
        resolved: &HashMap<String, Tensor>,
    ) -> Result<HashMap<ValueId, Tensor>> {
        self.input_values()
            .map(|(value, name)| {
                Ok((
                    value,
                    resolved
                        .get(name)
                        .cloned()
                        .ok_or_else(|| Error::missing_input("bind inputs"))?,
                ))
            })
            .collect()
    }

    /// Returns one required graph input dimension by input name and axis.
    ///
    /// # Errors
    ///
    /// Returns a missing-input error if no IR input has this name, or an out-of-range error if `axis` exceeds its rank.
    #[cfg(feature = "cuda")]
    pub fn required_input_dimension(&self, input: &str, axis: usize) -> Result<i64> {
        let value = self
            .values
            .iter()
            .find(|value| value.binding().input_name() == Some(input))
            .ok_or_else(|| Error::missing_input("required input dimension"))?;
        value
            .shape()
            .dimensions()
            .get(axis)
            .copied()
            .ok_or_else(|| Error::out_of_range(format!("graph input `{input}` dimension {axis}")))
    }
}
