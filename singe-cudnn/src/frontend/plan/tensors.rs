use std::{
    collections::{BTreeMap, btree_map},
    iter::FusedIterator,
};

use crate::{
    backend::tensor::{BackendTensorUid, TensorId, TensorSpec},
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{PreparedGraph, TensorRecord},
        operation::TensorAccess,
        plan::{BuiltPlanCandidates, CompiledGraph, PlanCandidates},
    },
};

/// Tensor metadata retained before or after compilation, including restored graphs.
/// This view does not expose a device pointer or grant permission to access storage.
#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "native capabilities and tensor properties are independent flags"
)]
pub struct TensorInfo<'a> {
    /// Frontend identity used to look up the retained tensor declaration.
    pub id: TensorId,
    /// Native identity after preparation when the tensor participates in a native operation or its addressing metadata.
    /// Validated declarations, unused declarations and storage-only alias sources have no native descriptor.
    pub backend_uid: Option<BackendTensorUid>,
    /// Borrowed tensor layout, encoding, storage placement, and addressing declaration.
    pub spec: &'a TensorSpec,
    /// Read/write access in the lowered execution, or `None` for declarations without an execution use.
    pub access: Option<TensorAccess>,
    /// Whether this tensor may read values supplied at graph entry.
    /// Virtual buffers and declarations without an execution use return false.
    pub reads_initial_state: bool,
    /// Whether tensor values control addresses or memory extents, such as offsets or sequence lengths.
    pub indexes_memory: bool,
    /// Whether runtime shapes must preserve vectorized dimension and stride alignment.
    pub requires_vector_alignment: bool,
    /// Forward-convolution filters retain the signed 32-bit kernel offset limit during shape overrides.
    /// The element count includes vector lanes, independently of stride gaps and physical capacity.
    pub requires_int32_filter_offsets: bool,
}

impl<'a> TensorInfo<'a> {
    pub(crate) fn from_record(id: TensorId, record: &'a TensorRecord) -> Self {
        Self {
            id,
            backend_uid: None,
            spec: &record.tensor,
            access: record.access,
            reads_initial_state: record.reads_initial_state,
            indexes_memory: record.indexes_memory,
            requires_vector_alignment: record.requires_vector_alignment,
            requires_int32_filter_offsets: record.requires_int32_filter_offsets,
        }
    }
}

/// Borrowed tensor metadata in frontend-ID order for validated or compiled graphs.
/// Iteration does not allocate, query cuDNN or compile kernels.
#[derive(Debug, Clone)]
pub struct TensorInfoIter<'a> {
    records: btree_map::Iter<'a, TensorId, TensorRecord>,
    native_uids: bool,
}

impl<'a> TensorInfoIter<'a> {
    pub(crate) fn native(records: &'a BTreeMap<TensorId, TensorRecord>) -> Self {
        Self {
            records: records.iter(),
            native_uids: true,
        }
    }

    pub(crate) fn declarations(records: &'a BTreeMap<TensorId, TensorRecord>) -> Self {
        Self {
            records: records.iter(),
            native_uids: false,
        }
    }

    fn info(&self, id: TensorId, record: &'a TensorRecord) -> TensorInfo<'a> {
        if self.native_uids {
            info(id, record)
        } else {
            TensorInfo::from_record(id, record)
        }
    }
}

impl<'a> Iterator for TensorInfoIter<'a> {
    type Item = TensorInfo<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (&id, record) = self.records.next()?;
        Some(self.info(id, record))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.records.size_hint()
    }
}

impl DoubleEndedIterator for TensorInfoIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let (&id, record) = self.records.next_back()?;
        Some(self.info(id, record))
    }
}

impl ExactSizeIterator for TensorInfoIter<'_> {}

impl FusedIterator for TensorInfoIter<'_> {}

fn info(id: TensorId, record: &TensorRecord) -> TensorInfo<'_> {
    TensorInfo {
        backend_uid: record.access.map(|_| record.id.into()),
        ..TensorInfo::from_record(id, record)
    }
}

fn by_uid(
    records: &BTreeMap<TensorId, TensorRecord>,
    uid: BackendTensorUid,
) -> Result<TensorInfo<'_>> {
    records
        .iter()
        .find(|(_, record)| record.access.is_some() && record.id.as_i64() == uid.as_i64())
        .map(|(&id, record)| info(id, record))
        .ok_or(FrontendError::BackendTensorNotFound(uid).into())
}

macro_rules! impl_native_tensor_queries {
    ($graph:ty, $($records:ident).+) => {
        impl $graph {
            /// All retained specifications, including lowered intermediates and score-subgraph tensors.
            /// Unused declarations and storage-only sources have no native UID.
            pub fn tensors(&self) -> TensorInfoIter<'_> {
                TensorInfoIter::native(&self.$($records).+)
            }

            /// Retained tensor attributes and access metadata by frontend identity.
            ///
            /// # Errors
            /// Returns an error if the frontend identity is absent from the retained tensor declarations.
            pub fn tensor_info(&self, id: TensorId) -> Result<TensorInfo<'_>> {
                self.$($records).+
                    .get(&id)
                    .map(|record| info(id, record))
                    .ok_or(FrontendError::TensorNotFound(id).into())
            }

            /// Retained attributes by native UID, matching the C++ graph lookup workflow.
            /// Does not query cuDNN, discover engines or build execution plans.
            ///
            /// # Errors
            /// Returns an error if no retained tensor participating in native execution or addressing metadata has this native UID.
            pub fn tensor_by_uid(&self, uid: BackendTensorUid) -> Result<TensorInfo<'_>> {
                by_uid(&self.$($records).+, uid)
            }
        }
    };
}

impl_native_tensor_queries!(PreparedGraph, lowered.layout.tensors);

impl_native_tensor_queries!(PlanCandidates, artifacts.tensors);

impl_native_tensor_queries!(BuiltPlanCandidates, artifacts.tensors);

impl_native_tensor_queries!(CompiledGraph, core.artifacts.tensors);
