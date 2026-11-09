use std::{
    fmt,
    hash::{Hash, Hasher},
    ptr,
    sync::Arc,
};

use crate::{
    backend::registry::{OperandRole, OutputRole},
    data_type::DataType,
    dispatch::{DispatchKey, Kernel, registry::Registry},
    error::{Error, Result},
    scalar::Scalar,
    value::ValueSpec,
};

type InferFunction = dyn Fn(&[ValueSpec], &[Scalar]) -> Result<Vec<ValueSpec>> + Send + Sync;

/// Definition of an operation registered outside the built-in catalogue.
///
/// Custom operations take tensor inputs and scalar attributes, and infer their results with a function supplied at registration.
pub struct CustomOperationDefinition {
    name: &'static str,
    operands: &'static [OperandRole],
    outputs: &'static [OutputRole],
    data_types: &'static [DataType],
    infer: Arc<InferFunction>,
}

impl fmt::Debug for CustomOperationDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomOperationDefinition")
            .field("name", &self.name)
            .field("inputs", &self.operands.len())
            .field("outputs", &self.outputs.len())
            .finish_non_exhaustive()
    }
}

/// Handle to a registered custom operation.
///
/// Handles compare equal only when they refer to the same registration.
#[derive(Debug, Clone, Copy)]
pub struct CustomOperation(&'static CustomOperationDefinition);

impl CustomOperation {
    /// Registers a custom operation named `name` with `inputs` tensor inputs and `outputs` results.
    ///
    /// `data_types` lists the dtypes its kernels accept, and `infer` returns the result specifications for given input specifications and scalar attributes.
    /// The definition lives for the rest of the process.
    /// Register kernels for it with [`register_kernel`](Self::register_kernel) or [`register_composite`](Self::register_composite).
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is empty, or `inputs` or `outputs` is 0.
    pub fn register(
        name: impl Into<String>,
        inputs: usize,
        outputs: usize,
        data_types: &[DataType],
        infer: impl Fn(&[ValueSpec], &[Scalar]) -> Result<Vec<ValueSpec>> + Send + Sync + 'static,
    ) -> Result<Self> {
        let name = name.into();
        if name.is_empty() || inputs == 0 || outputs == 0 {
            return Err(Error::invalid_argument("custom operation definition"));
        }

        let definition = CustomOperationDefinition {
            name: Box::leak(name.into_boxed_str()),
            operands: Box::leak(vec![OperandRole::Input; inputs].into_boxed_slice()),
            outputs: Box::leak(vec![OutputRole::Output; outputs].into_boxed_slice()),
            data_types: Box::leak(data_types.to_vec().into_boxed_slice()),
            infer: Arc::new(infer),
        };
        Ok(Self(Box::leak(Box::new(definition))))
    }

    /// Registers `kernel` for this operation at backend or frontend `key`.
    ///
    /// # Errors
    ///
    /// Returns an error if the registry cannot be updated.
    pub fn register_kernel(self, key: DispatchKey, kernel: Kernel) -> Result<Self> {
        Registry::get().register_custom(self, Some(key), kernel)?;
        Ok(self)
    }

    /// Registers a composite `kernel`, used on any backend without its own kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if the registry cannot be updated.
    pub fn register_composite(self, kernel: Kernel) -> Result<Self> {
        Registry::get().register_custom(self, None, kernel)?;
        Ok(self)
    }

    /// Returns the operation name.
    #[must_use]
    pub fn name(self) -> &'static str {
        self.0.name
    }

    /// Returns the roles of the tensor inputs.
    #[must_use]
    pub fn operands(self) -> &'static [OperandRole] {
        self.0.operands
    }

    /// Returns the roles of the results.
    #[must_use]
    pub fn outputs(self) -> &'static [OutputRole] {
        self.0.outputs
    }

    /// Returns the dtypes its kernels accept.
    #[must_use]
    pub fn data_types(self) -> &'static [DataType] {
        self.0.data_types
    }

    /// Infers the result specifications for `inputs` and `attributes`.
    ///
    /// # Errors
    ///
    /// Returns an error if the inference function rejects the inputs, or returns the wrong number of results.
    pub fn infer(self, inputs: &[ValueSpec], attributes: &[Scalar]) -> Result<Vec<ValueSpec>> {
        let outputs = (self.0.infer)(inputs, attributes)?;
        if outputs.len() != self.0.outputs.len() {
            return Err(Error::missing_output("infer"));
        }
        Ok(outputs)
    }
}

impl PartialEq for CustomOperation {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0, other.0)
    }
}

impl Eq for CustomOperation {}

impl Hash for CustomOperation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(self.0, state);
    }
}
