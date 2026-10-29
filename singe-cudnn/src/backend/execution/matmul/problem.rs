use crate::{
    backend::{
        execution::matmul::{MatmulDescriptorConfig, MatmulOperationConfig},
        tensor::{Tensor, TensorProblemKey},
    },
    data_type::DataType,
};

/// Hashable native matmul declaration for execution-plan caching.
///
/// Includes A/B/C and supplied M/N/K override tensor snapshots, compute precision,
/// optional padding and irregular batch addressing. Padding compares by its exact
/// double-precision bits, distinguishing omission, signed zero and NaN payloads.
/// Device addresses and runtime scalar values do not participate in equality.
///
/// Scope the cache to a compatible device/context, complete operation graph and
/// compilation policy. Additional fused operations and runtime shape overrides
/// require additional key information. The key owns no native descriptors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatmulProblemKey {
    tensors: [TensorProblemKey; 3],
    config: MatmulOperationConfig<TensorProblemKey>,
    compute_type: DataType,
    padding_value: Option<u64>,
}

impl MatmulProblemKey {
    pub(crate) fn new(
        descriptor: MatmulDescriptorConfig,
        tensors: [&Tensor; 3],
        config: MatmulOperationConfig<&Tensor>,
    ) -> Self {
        Self {
            tensors: tensors.map(Tensor::problem_key),
            config: config.map(Tensor::problem_key),
            compute_type: descriptor.compute_type,
            padding_value: descriptor.padding_value.map(f64::to_bits),
        }
    }

    /// Borrowed tensor snapshots in native `[A, B, C]` operand order.
    #[must_use]
    pub fn tensors(&self) -> &[TensorProblemKey; 3] {
        &self.tensors
    }

    /// Supplied override tensor snapshots and irregular batch count.
    #[must_use]
    pub fn config(&self) -> &MatmulOperationConfig<TensorProblemKey> {
        &self.config
    }

    /// Retained math and padding attributes, preserving optionality and exact bits.
    pub fn descriptor_config(&self) -> MatmulDescriptorConfig {
        MatmulDescriptorConfig {
            compute_type: self.compute_type,
            padding_value: self.padding_value.map(f64::from_bits),
        }
    }
}
