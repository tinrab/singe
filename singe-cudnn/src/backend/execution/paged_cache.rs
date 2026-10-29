use std::result::Result as StdResult;

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::{BackendTensorUid, Tensor},
    },
    error::Result,
};

/// Named operands shared by frontend identities, native tensor references and native UID metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagedCacheLoadTensors<T> {
    /// Noncontiguous block container holding the key or value cache.
    pub container: T,
    /// Reconstructed virtual key or value tensor used by attention.
    pub output: T,
    /// Per-batch sequence extent selecting the requested cache prefix.
    pub sequence: T,
    /// Integer lookup table mapping logical cache pages to container blocks.
    pub page_table: T,
}

impl<T> PagedCacheLoadTensors<T> {
    /// Records the cache container, reconstructed output, sequence extents, and page table without validation.
    pub fn new(container: T, output: T, sequence: T, page_table: T) -> Self {
        Self {
            container,
            output,
            sequence,
            page_table,
        }
    }

    /// Borrows owned operands without cloning tensors or allocating an operand list.
    pub fn as_ref(&self) -> PagedCacheLoadTensors<&T> {
        PagedCacheLoadTensors {
            container: &self.container,
            output: &self.output,
            sequence: &self.sequence,
            page_table: &self.page_table,
        }
    }

    /// Maps identities without changing cache, result or addressing roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> PagedCacheLoadTensors<U> {
        PagedCacheLoadTensors {
            container: map(self.container),
            output: map(self.output),
            sequence: map(self.sequence),
            page_table: map(self.page_table),
        }
    }

    /// Maps every operand while preserving its role in the load operation.
    ///
    /// # Errors
    ///
    /// Returns the first mapping error, processing container, output, sequence, and page table in that order.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<PagedCacheLoadTensors<U>, E> {
        Ok(PagedCacheLoadTensors {
            container: map(self.container)?,
            output: map(self.output)?,
            sequence: map(self.sequence)?,
            page_table: map(self.page_table)?,
        })
    }
}

/// Finalized native paged-cache load with retained operand identities.
#[derive(Debug)]
pub struct PagedCacheLoadOperation {
    descriptor: BackendDescriptor,
    tensor_uids: PagedCacheLoadTensors<BackendTensorUid>,
}

impl PagedCacheLoadOperation {
    /// Creates a native load using the declared cache, sequence and page-table layouts.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects creation, operand descriptors or finalization.
    pub fn create(tensors: PagedCacheLoadTensors<&Tensor>) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationPagedCacheLoad)?;
        for (attribute, tensor) in [
            (
                BackendAttributeName::OperationPagedCacheLoadContainerDesc,
                tensors.container,
            ),
            (
                BackendAttributeName::OperationPagedCacheLoadYDesc,
                tensors.output,
            ),
            (
                BackendAttributeName::OperationPagedCacheLoadSequenceDesc,
                tensors.sequence,
            ),
            (
                BackendAttributeName::OperationPagedCacheLoadPageTableDesc,
                tensors.page_table,
            ),
        ] {
            descriptor.set_attribute_descriptor(attribute, tensor.descriptor())?;
        }
        descriptor.finalize()?;
        Ok(Self {
            descriptor,
            tensor_uids: tensors.map(Tensor::id),
        })
    }

    /// Actual native operand UIDs, independent of frontend identities and caller binding aliases.
    #[must_use]
    pub fn tensor_uids(&self) -> &PagedCacheLoadTensors<BackendTensorUid> {
        &self.tensor_uids
    }

    /// Borrows the finalized native paged-cache load descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
