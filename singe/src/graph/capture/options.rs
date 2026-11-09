use crate::{graph::InputDimensionGuard, tensor::Tensor};

/// The observation contract retained with an eagerly captured program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureInfo {
    pub(crate) value_specialization: bool,
}

impl CaptureInfo {
    /// Whether capture permitted tracked tensor values to influence host code.
    /// Such values and the selected host path remain fixed during replay.
    #[must_use]
    pub fn allows_value_specialization(self) -> bool {
        self.value_specialization
    }
}

/// Eager capture runs kernels on the recording thread.
/// Allowing readback accepts specialization to the observed data and host control flow during that run.
#[derive(Debug, Clone, Default)]
pub struct CaptureOptions {
    pub(crate) allow_value_specialization: bool,
    pub(crate) dimensions: Vec<InputDimensionGuard>,
}

impl CaptureOptions {
    /// Creates eager capture options that prohibit value specialization.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether tracked tensor readback may specialize the captured program to observed values and host control flow.
    #[must_use]
    pub fn with_value_specialization(mut self, allow: bool) -> Self {
        self.allow_value_specialization = allow;
        self
    }
}

#[derive(Debug)]
pub enum CaptureBinding {
    Parameter(String, Tensor),
    Buffer(String, Tensor),
    Constant(String, Tensor),
}

/// Explicit closed-over tensors.
/// Parameters and buffers become replaceable named state.
/// Constants become immutable program literals and may be folded.
#[derive(Debug, Default)]
pub struct CaptureBindings {
    pub(crate) bindings: Vec<CaptureBinding>,
}

impl CaptureBindings {
    /// Creates an empty set of explicit closed-over tensor bindings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // Appends `other`'s bindings, such as a second model's state that one capture reads too.
    #[must_use]
    pub fn merged(mut self, other: Self) -> Self {
        self.bindings.extend(other.bindings);
        self
    }

    /// Appends a named parameter tensor as replaceable captured state.
    /// Names are retained without validation until capture processes the bindings.
    #[must_use]
    pub fn with_parameter(mut self, name: impl Into<String>, tensor: Tensor) -> Self {
        self.bindings
            .push(CaptureBinding::Parameter(name.into(), tensor));
        self
    }

    /// Appends a named buffer tensor as replaceable captured state.
    /// Names are retained without validation until capture processes the bindings.
    #[must_use]
    pub fn with_buffer(mut self, name: impl Into<String>, tensor: Tensor) -> Self {
        self.bindings
            .push(CaptureBinding::Buffer(name.into(), tensor));
        self
    }

    /// Appends a named immutable program literal that may be folded.
    /// Names are retained without validation until capture processes the bindings.
    #[must_use]
    pub fn with_constant(mut self, name: impl Into<String>, tensor: Tensor) -> Self {
        self.bindings
            .push(CaptureBinding::Constant(name.into(), tensor));
        self
    }
}
