use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::{
    backend::tensor::{TensorId, TensorSpec},
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, TensorRecord, validate_storage_view_byte_range},
        operation::{ConvolutionOperation, Operation, SdpaOperation, TensorAccess},
    },
};

impl Graph {
    /// Declarations that can supply caller storage or by-value arguments.
    /// Storage-view targets share their source allocation instead of owning another binding.
    pub(crate) fn caller_binding_tensors(&self) -> impl Iterator<Item = (TensorId, &TensorSpec)> {
        let views = self
            .storage_views
            .iter()
            .map(|binding| binding.target)
            .collect::<HashSet<_>>();
        self.tensors
            .iter()
            .filter(move |(id, tensor)| {
                !views.contains(*id)
                    && !tensor.is_virtual()
                    && !tensor.is_constant()
                    && !tensor.is_workspace()
            })
            .map(|(&id, tensor)| (id, tensor))
    }

    /// Caller allocations and virtual inputs read without a producer in this graph.
    /// A score graph can consume virtual values produced by its enclosing graph.
    pub(crate) fn captured_tensor_ids(&self) -> BTreeSet<TensorId> {
        let mut reads = BTreeSet::new();
        let mut writes = BTreeSet::new();
        for operation in &self.operations {
            let uses = operation.tensor_uses(self.config.data_type_policy);
            reads.extend(uses.reads);
            writes.extend(uses.writes);
        }
        let mut pending = reads.union(&writes).copied().collect::<Vec<_>>();
        let mut visited = HashSet::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Some(offset) = self
                .tensors
                .get(&id)
                .and_then(|tensor| tensor.ragged_offset)
            {
                reads.insert(offset);
                pending.push(offset);
            }
        }
        let mut captures = self
            .caller_binding_tensors()
            .map(|(id, _)| id)
            .collect::<BTreeSet<_>>();
        captures.extend(
            reads
                .difference(&writes)
                .copied()
                .filter(|id| self.tensors.get(id).is_none_or(TensorSpec::is_virtual)),
        );
        captures
    }

    /// Discovers operation uses and ragged dependencies without validating operation declarations.
    /// Score-subgraph capture uses this before its shared layouts have been reconciled.
    pub(crate) fn tensor_records(&self) -> Result<BTreeMap<TensorId, TensorRecord>> {
        let mut records = self
            .tensors
            .iter()
            .map(|(&id, tensor)| {
                (
                    id,
                    TensorRecord {
                        id,
                        tensor: tensor.clone(),
                        access: None,
                        reads_initial_state: false,
                        indexes_memory: false,
                        requires_vector_alignment: self.vector_aligned_tensors.contains(&id),
                        requires_int32_filter_offsets: false,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        for operation in &self.operations {
            if let Operation::Convolution(ConvolutionOperation::Forward { w, .. }) = operation {
                let filter = records
                    .get_mut(w)
                    .ok_or(FrontendError::TensorNotFound(*w))?;
                filter.requires_int32_filter_offsets |= matches!(filter.tensor.shape.rank(), 4 | 5);
            }
            let uses = match operation {
                Operation::Sdpa(operation) => operation.native_tensor_uses(),
                operation => operation.tensor_uses(self.config.data_type_policy),
            };
            for id in uses.reads {
                record_use(
                    &mut records,
                    id,
                    TensorAccess::Read,
                    uses.indexing.contains(&id),
                )?;
            }
            for id in uses.writes {
                record_use(
                    &mut records,
                    id,
                    TensorAccess::Write,
                    uses.indexing.contains(&id),
                )?;
            }
        }
        let used = records
            .iter()
            .filter_map(|(&id, record)| record.access.map(|_| id))
            .collect::<Vec<_>>();
        for id in used {
            let mut seen = HashSet::from([id]);
            let mut current = records[&id].tensor.ragged_offset;
            while let Some(offset) = current {
                if !seen.insert(offset) {
                    return Err(FrontendError::RaggedOffsetCycle { tensor_id: offset }.into());
                }
                record_use(&mut records, offset, TensorAccess::Read, true)?;
                current = records[&offset].tensor.ragged_offset;
            }
        }
        Ok(records)
    }

    /// Validates operation geometry, storage views and every used tensor before native allocation.
    /// Records actual operation uses, including the read dependencies of ragged layouts.
    /// Declared but unused tensors remain available for lookup without receiving native descriptors.
    pub(crate) fn validated_tensor_records(&self) -> Result<BTreeMap<TensorId, TensorRecord>> {
        for operation in &self.operations {
            match operation {
                Operation::Pointwise(operation) => operation.validate_graph(self)?,
                Operation::Tensor(operation) => operation.validate_graph(self)?,
                Operation::PagedCacheLoad(operation) => operation.validate_graph(self)?,
                Operation::BlockScale(operation) => operation.validate_graph(self)?,
                Operation::AttentionPrimitive(operation) => operation.validate_graph(self)?,
                Operation::Sdpa(operation) => operation.validate_graph(self)?,
                Operation::Matmul(operation) => operation.validate_graph(self)?,
                Operation::Resample(operation) => operation.validate_graph(self)?,
                Operation::RandomNumberGenerator(operation) => operation.validate_graph(self)?,
                Operation::GenStats(operation) => operation.validate_graph(self)?,
                Operation::MoeGroupedMatmul(operation) => operation.validate_graph(self)?,
                Operation::Rope(operation) => operation.validate_graph(self)?,
                Operation::Reduction(operation) => operation.validate_graph(self)?,
                Operation::Normalization(operation) => operation.validate_graph(self)?,
                Operation::Convolution(operation) => operation.validate_graph(self)?,
                _ => {}
            }
        }
        let mut alias_targets = HashSet::new();
        for alias in &self.storage_views {
            let source = self.tensor_spec(alias.source)?;
            let target = self.tensor_spec(alias.target)?;
            if !alias_targets.insert(alias.target) {
                return Err(Error::DescriptorMismatch {
                    name: "frontend storage view".into(),
                });
            }
            validate_storage_view_byte_range(
                alias.source,
                source,
                alias.target,
                target,
                alias.byte_offset,
            )?;
            let mut current = alias.target;
            let mut seen = HashSet::new();
            while let Some(parent) = self
                .storage_views
                .iter()
                .find(|binding| binding.target == current)
            {
                if !seen.insert(current) {
                    return Err(Error::DescriptorMismatch {
                        name: "frontend storage view cycle".into(),
                    });
                }
                current = parent.source;
            }
        }
        let mut records = self.tensor_records()?;
        for record in records.values().filter(|record| record.access.is_some()) {
            record.tensor.validate()?;
        }
        let mut aligned_operands = self
            .vector_aligned_tensors
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        for operation in &self.operations {
            match operation {
                Operation::Sdpa(SdpaOperation::Forward { tensors, .. }) => {
                    aligned_operands.extend([
                        tensors.query,
                        tensors.key,
                        tensors.value,
                        tensors.output,
                    ]);
                }
                Operation::Sdpa(SdpaOperation::Backward { tensors, .. }) => {
                    aligned_operands.extend(tensors.activations_and_gradients());
                }
                _ => {}
            }
        }
        for &id in &aligned_operands {
            let record = records
                .get_mut(&id)
                .ok_or(FrontendError::TensorNotFound(id))?;
            if record.access.is_some() {
                record
                    .tensor
                    .validate_vector_stride_alignment(id, &record.tensor.shape)?;
                record.requires_vector_alignment = true;
            }
        }

        Ok(records)
    }
}

fn record_use(
    records: &mut BTreeMap<TensorId, TensorRecord>,
    id: TensorId,
    access: TensorAccess,
    indexes_memory: bool,
) -> Result<()> {
    let record = records
        .get_mut(&id)
        .ok_or(FrontendError::TensorNotFound(id))?;
    record.access = Some(
        record
            .access
            .map_or(access, |previous| previous.merge(access)),
    );
    record.indexes_memory |= indexes_memory;
    Ok(())
}
