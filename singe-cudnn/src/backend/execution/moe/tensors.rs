use serde::{Deserialize, Serialize};

use crate::backend::tensor::TensorId;

/// Required forward operands, shared by frontend IDs, native references and native UID metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeGroupedMatmulInputs<T = TensorId> {
    /// Token activation tensor.
    pub token: T,
    /// Expert weight tensor.
    pub weight: T,
    /// INT32 or INT64 expert offsets in the format supported by the native runtime.
    pub first_token_offset: T,
}

impl<T> MoeGroupedMatmulInputs<T> {
    /// Records the required grouped-matmul operands.
    pub fn new(token: T, weight: T, first_token_offset: T) -> Self {
        Self {
            token,
            weight,
            first_token_offset,
        }
    }

    /// Borrows tensor operands while preserving their token, weight, gradient, and offset roles.
    pub fn as_ref(&self) -> MoeGroupedMatmulInputs<&T> {
        MoeGroupedMatmulInputs {
            token: &self.token,
            weight: &self.weight,
            first_token_offset: &self.first_token_offset,
        }
    }

    /// Maps tensor operands in declaration order while preserving their grouped-matmul roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> MoeGroupedMatmulInputs<U> {
        MoeGroupedMatmulInputs {
            token: map(self.token),
            weight: map(self.weight),
            first_token_offset: map(self.first_token_offset),
        }
    }

    /// Maps tensor operands in declaration order while preserving their grouped-matmul roles.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> Result<U, E>,
    ) -> Result<MoeGroupedMatmulInputs<U>, E> {
        Ok(MoeGroupedMatmulInputs {
            token: map(self.token)?,
            weight: map(self.weight)?,
            first_token_offset: map(self.first_token_offset)?,
        })
    }
}

/// Required forward operands and result, independent of optional gather/scatter routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeGroupedMatmulTensors<T = TensorId> {
    /// Required token, weight, and expert-offset operands.
    pub inputs: MoeGroupedMatmulInputs<T>,
    /// Grouped matmul output tensor.
    pub output: T,
}

impl<T> MoeGroupedMatmulTensors<T> {
    /// Records the required grouped-matmul operands and result.
    pub fn new(inputs: MoeGroupedMatmulInputs<T>, output: T) -> Self {
        Self { inputs, output }
    }

    /// Borrows tensor operands while preserving their token, weight, gradient, and offset roles.
    pub fn as_ref(&self) -> MoeGroupedMatmulTensors<&T> {
        MoeGroupedMatmulTensors {
            inputs: self.inputs.as_ref(),
            output: &self.output,
        }
    }

    /// Maps tensor operands in declaration order while preserving their grouped-matmul roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> MoeGroupedMatmulTensors<U> {
        MoeGroupedMatmulTensors {
            inputs: self.inputs.map(&mut map),
            output: map(self.output),
        }
    }

    /// Maps tensor operands in declaration order while preserving their grouped-matmul roles.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> Result<U, E>,
    ) -> Result<MoeGroupedMatmulTensors<U>, E> {
        Ok(MoeGroupedMatmulTensors {
            inputs: self.inputs.try_map(&mut map)?,
            output: map(self.output)?,
        })
    }
}

/// Required weight-gradient operands, retaining the roles of tokens and output gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeGroupedMatmulBackwardInputs<T = TensorId> {
    /// Gradient tensor for the forward output.
    pub output_gradient: T,
    /// Token activation tensor.
    pub token: T,
    /// INT32 or INT64 offsets; routing groups may reuse the same expert weights.
    pub first_token_offset: T,
}

impl<T> MoeGroupedMatmulBackwardInputs<T> {
    /// Records the required grouped-matmul weight-gradient operands.
    pub fn new(output_gradient: T, token: T, first_token_offset: T) -> Self {
        Self {
            output_gradient,
            token,
            first_token_offset,
        }
    }

    /// Borrows tensor operands while preserving their token, weight, gradient, and offset roles.
    pub fn as_ref(&self) -> MoeGroupedMatmulBackwardInputs<&T> {
        MoeGroupedMatmulBackwardInputs {
            output_gradient: &self.output_gradient,
            token: &self.token,
            first_token_offset: &self.first_token_offset,
        }
    }

    /// Maps tensor operands in declaration order while preserving their grouped-matmul roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> MoeGroupedMatmulBackwardInputs<U> {
        MoeGroupedMatmulBackwardInputs {
            output_gradient: map(self.output_gradient),
            token: map(self.token),
            first_token_offset: map(self.first_token_offset),
        }
    }

    /// Maps tensor operands in declaration order while preserving their grouped-matmul roles.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> Result<U, E>,
    ) -> Result<MoeGroupedMatmulBackwardInputs<U>, E> {
        Ok(MoeGroupedMatmulBackwardInputs {
            output_gradient: map(self.output_gradient)?,
            token: map(self.token)?,
            first_token_offset: map(self.first_token_offset)?,
        })
    }
}

/// Required operands and expert weight-gradient result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeGroupedMatmulBackwardTensors<T = TensorId> {
    /// Required output-gradient, token-activation, and expert-offset operands.
    pub inputs: MoeGroupedMatmulBackwardInputs<T>,
    /// Expert weight-gradient destination tensor.
    pub weight_gradient: T,
}

impl<T> MoeGroupedMatmulBackwardTensors<T> {
    /// Records the required grouped-matmul weight-gradient operands and result.
    pub fn new(inputs: MoeGroupedMatmulBackwardInputs<T>, weight_gradient: T) -> Self {
        Self {
            inputs,
            weight_gradient,
        }
    }

    /// Borrows tensor operands while preserving their token, weight, gradient, and offset roles.
    pub fn as_ref(&self) -> MoeGroupedMatmulBackwardTensors<&T> {
        MoeGroupedMatmulBackwardTensors {
            inputs: self.inputs.as_ref(),
            weight_gradient: &self.weight_gradient,
        }
    }

    /// Maps tensor operands in declaration order while preserving their grouped-matmul roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> MoeGroupedMatmulBackwardTensors<U> {
        MoeGroupedMatmulBackwardTensors {
            inputs: self.inputs.map(&mut map),
            weight_gradient: map(self.weight_gradient),
        }
    }

    /// Maps tensor operands in declaration order while preserving their grouped-matmul roles.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> Result<U, E>,
    ) -> Result<MoeGroupedMatmulBackwardTensors<U>, E> {
        Ok(MoeGroupedMatmulBackwardTensors {
            inputs: self.inputs.try_map(&mut map)?,
            weight_gradient: map(self.weight_gradient)?,
        })
    }
}
