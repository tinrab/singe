use std::collections::HashMap;

use crate::{
    device::Device,
    onnx::error::{Error, Result},
};

/// Options controlling ONNX import into a Singe graph.
#[derive(Debug, Clone)]
pub struct OnnxOptions {
    /// Default execution target for `OnnxModel::run` and `compile`.
    /// Import always decodes and lowers on CPU without creating a CUDA context.
    pub device: Device,
    /// Concrete shape overrides keyed by ONNX graph input name.
    pub input_shapes: HashMap<String, Vec<i64>>,
    /// Representative values for ONNX symbolic dimension names.
    /// Symbols remain variable at runtime; these values prepare the initial specialization.
    pub symbolic_dimensions: HashMap<String, i64>,
    /// Inclusive runtime bounds for named symbols.
    /// Omitted bounds allow any nonnegative dimension; equal bounds explicitly fix a symbol.
    pub symbolic_dimension_bounds: HashMap<String, (Option<i64>, Option<i64>)>,
}

impl OnnxOptions {
    /// Creates default ONNX options for CPU execution.
    #[must_use]
    pub fn new() -> Self {
        Self {
            device: Device::Cpu,
            input_shapes: HashMap::new(),
            symbolic_dimensions: HashMap::new(),
            symbolic_dimension_bounds: HashMap::new(),
        }
    }

    /// Sets the default execution target.
    /// Placement occurs during execution preparation, after import succeeds.
    #[must_use]
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Adds a concrete shape override for an ONNX graph input.
    #[must_use]
    pub fn with_input_shape(mut self, name: impl Into<String>, shape: impl Into<Vec<i64>>) -> Self {
        self.input_shapes.insert(name.into(), shape.into());
        self
    }

    /// Adds a representative value for an ONNX symbolic dimension.
    #[must_use]
    pub fn with_symbolic_dimension(mut self, name: impl Into<String>, value: i64) -> Self {
        self.symbolic_dimensions.insert(name.into(), value);
        self
    }

    /// Declares a representative value and inclusive runtime bounds for a symbol.
    #[must_use]
    pub fn with_symbolic_dimension_range(
        mut self,
        name: impl Into<String>,
        representative: i64,
        min: Option<i64>,
        max: Option<i64>,
    ) -> Self {
        let name = name.into();
        self.symbolic_dimensions
            .insert(name.clone(), representative);
        self.symbolic_dimension_bounds.insert(name, (min, max));
        self
    }

    // The bounds declared for a symbolic dimension, where each side may be open.
    pub(crate) fn symbolic_bounds(&self, name: &str) -> (Option<i64>, Option<i64>) {
        self.symbolic_dimension_bounds
            .get(name)
            .copied()
            .unwrap_or_default()
    }

    pub(crate) fn validate(&self) -> Result<()> {
        for (name, value) in &self.symbolic_dimensions {
            let (min, max) = self.symbolic_bounds(name);
            if name.is_empty()
                || *value < 0
                || min.is_some_and(|min| min < 0 || *value < min)
                || max.is_some_and(|max| max < 0 || *value > max)
                || min.zip(max).is_some_and(|(min, max)| min > max)
            {
                return Err(Error::UnsupportedShape {
                    name: name.clone(),
                    reason: "invalid symbolic dimension declaration".into(),
                });
            }
        }

        for name in self.symbolic_dimension_bounds.keys() {
            if !self.symbolic_dimensions.contains_key(name) {
                return Err(Error::UnsupportedShape {
                    name: name.clone(),
                    reason: "symbol bounds require a representative value".into(),
                });
            }
        }
        Ok(())
    }
}

impl Default for OnnxOptions {
    fn default() -> Self {
        Self::new()
    }
}
