use std::{fmt, sync::Arc};

use crate::{
    backend::registry::{CheckedOperationRequest, Operation, OperationAttributes},
    dispatch::{DispatchKey, dispatch_checked},
    error::{Error, Result},
    tensor::Tensor,
};

type KernelFunction = dyn Fn(&KernelCall<'_>) -> Result<Vec<Tensor>> + Send + Sync;

/// Function that implements an operation at one dispatch key.
///
/// It returns the operation's results in output order.
#[derive(Clone)]
pub struct Kernel(Arc<KernelFunction>);

impl Kernel {
    /// Wraps `function` as a kernel.
    pub fn new(
        function: impl Fn(&KernelCall<'_>) -> Result<Vec<Tensor>> + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(function))
    }

    /// Wraps a kernel `function` that returns one result.
    pub fn single(
        function: impl Fn(&KernelCall<'_>) -> Result<Tensor> + Send + Sync + 'static,
    ) -> Self {
        Self::new(move |call| Ok(vec![function(call)?]))
    }

    pub(crate) fn call(&self, call: &KernelCall<'_>) -> Result<Vec<Tensor>> {
        (self.0)(call)
    }
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernel").finish_non_exhaustive()
    }
}

/// One operation call as seen by a kernel: the checked request, its inputs, and the running key.
#[derive(Debug)]
pub struct KernelCall<'a> {
    request: &'a CheckedOperationRequest,
    inputs: &'a [Tensor],
    key: DispatchKey,
}

impl<'a> KernelCall<'a> {
    pub(crate) const fn new(
        request: &'a CheckedOperationRequest,
        inputs: &'a [Tensor],
        key: DispatchKey,
    ) -> Self {
        Self {
            request,
            inputs,
            key,
        }
    }

    /// Returns the checked request, including its inferred results.
    #[must_use]
    pub const fn request(&self) -> &'a CheckedOperationRequest {
        self.request
    }

    /// Returns the operation.
    #[must_use]
    pub fn operation(&self) -> Operation {
        self.request.operation()
    }

    /// Returns the operation attributes.
    #[must_use]
    pub fn attributes(&self) -> &'a OperationAttributes {
        self.request.attributes()
    }

    /// Returns the input tensors in operand order.
    #[must_use]
    pub const fn inputs(&self) -> &'a [Tensor] {
        self.inputs
    }

    /// Returns the input at `index`.
    ///
    /// # Errors
    ///
    /// Returns an error if there is no input at `index`.
    pub fn input(&self, index: usize) -> Result<&'a Tensor> {
        self.inputs
            .get(index)
            .ok_or_else(|| Error::missing_input("input"))
    }

    /// Returns the input at `index` when `present` is true, and `None` otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if `present` is true and there is no input at `index`.
    pub fn optional_input(&self, index: usize, present: bool) -> Result<Option<&'a Tensor>> {
        present.then(|| self.input(index)).transpose()
    }

    /// Returns the dispatch key whose kernel is running.
    #[must_use]
    pub const fn key(&self) -> DispatchKey {
        self.key
    }

    /// Passes the call to the next dispatch key.
    ///
    /// The running key and every key above it stay excluded, so the call reaches the next key that has a kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if a lower kernel fails.
    pub fn redispatch(&self) -> Result<Vec<Tensor>> {
        dispatch_checked(self.request, self.inputs)
    }
}
