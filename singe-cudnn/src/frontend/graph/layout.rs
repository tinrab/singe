use std::collections::{BTreeMap, BTreeSet};

use crate::{
    backend::tensor::{TensorId, TensorSpec},
    error::Result,
    frontend::{
        bindings::resolve_binding_source,
        error::Error as FrontendError,
        graph::{Graph, TensorRecord, ValidatedGraph, subgraphs::ScoreSubgraph},
        plan::{AliasTensorKind, BindingReplacement, RequiredTensor, required_tensor_bindings},
        workspace::WorkspaceLayout,
    },
    scalar::ScalarValue,
};

/// Validated declarations and bindings, independent of native descriptors.
#[derive(Debug, Clone)]
pub struct GraphLayout {
    pub tensors: BTreeMap<TensorId, TensorRecord>,
    pub native_tensor_ids: BTreeSet<TensorId>,
    pub binding_template_ids: Vec<TensorId>,
    pub workspace_layout: WorkspaceLayout,
    pub scalar_bindings: Vec<(TensorId, ScalarValue)>,
    pub binding_replacements: Vec<BindingReplacement>,
    pub required_tensors: Vec<RequiredTensor>,
}

struct BindingCollector {
    tensors: BTreeMap<TensorId, TensorRecord>,
    binding_template_ids: Vec<TensorId>,
    scalar_bindings: Vec<(TensorId, ScalarValue)>,
    binding_replacements: Vec<BindingReplacement>,
}

impl GraphLayout {
    pub(crate) fn create(
        graph: &mut Graph,
        subgraphs: &BTreeMap<usize, ScoreSubgraph<ValidatedGraph>>,
    ) -> Result<Self> {
        let tensors = graph.validated_tensor_records()?;
        let native_tensor_ids = tensors
            .iter()
            .filter_map(|(&id, record)| record.access.is_some().then_some(id))
            .collect();
        let binding_template_ids = tensors
            .values()
            .filter_map(|record| {
                (!record.tensor.is_virtual()
                    && !record.tensor.is_constant()
                    && record.access.is_some())
                .then_some(record.id)
            })
            .collect();
        let mut bindings = BindingCollector {
            tensors,
            binding_template_ids,
            scalar_bindings: Vec::new(),
            binding_replacements: Vec::new(),
        };
        for subgraph in subgraphs.values() {
            bindings.import_score_subgraph(
                &subgraph.graph.layout,
                &subgraph.tensor_captures,
                &subgraph.captured_tensors,
            )?;
        }
        for subgraph in subgraphs.values() {
            for (&id, bounds) in &subgraph.graph.expanded.runtime_shape_constraints {
                let record = bindings.tensor_record(id)?;
                let bounds = match graph.runtime_shape_constraints.get(&id) {
                    Some(existing) => bounds.intersection(existing)?,
                    None => bounds.clone(),
                };
                bounds.validate_for(&record.tensor)?;
                graph.runtime_shape_constraints.insert(id, bounds);
            }
        }
        for &(id, _) in &bindings.scalar_bindings {
            if !bindings.binding_template_ids.contains(&id) {
                bindings.binding_template_ids.push(id);
            }
        }
        for binding in &graph.storage_views {
            let source_id = bindings.tensor_record(binding.source)?.id;
            let target_id = bindings.tensor_record(binding.target)?.id;
            bindings.binding_replacements.push(BindingReplacement {
                kind: AliasTensorKind::StorageView,
                source_id,
                target_id,
                byte_offset: binding.byte_offset,
            });
        }
        graph.annotate_initial_reads(&mut bindings.tensors, subgraphs);
        let workspace_layout =
            WorkspaceLayout::create(&bindings.tensors, &bindings.binding_template_ids, &[])?;
        let required_tensors = required_tensor_bindings(
            &bindings.tensors,
            &bindings.binding_replacements,
            &bindings.binding_template_ids,
        )?;
        let layout = Self {
            tensors: bindings.tensors,
            native_tensor_ids,
            binding_template_ids: bindings.binding_template_ids,
            workspace_layout,
            scalar_bindings: bindings.scalar_bindings,
            binding_replacements: bindings.binding_replacements,
            required_tensors,
        };
        graph.tensors = layout
            .tensors
            .iter()
            .map(|(&id, record)| (id, record.tensor.clone()))
            .collect();
        graph.vector_aligned_tensors.extend(
            layout
                .tensors
                .iter()
                .filter_map(|(&id, record)| record.requires_vector_alignment.then_some(id)),
        );
        Ok(layout)
    }

    pub(crate) fn tensor_record(&self, tensor: TensorId) -> Result<&TensorRecord> {
        self.tensors
            .get(&tensor)
            .ok_or(FrontendError::TensorNotFound(tensor).into())
    }
}

impl BindingCollector {
    fn tensor_record(&self, tensor: TensorId) -> Result<&TensorRecord> {
        self.tensors
            .get(&tensor)
            .ok_or(FrontendError::TensorNotFound(tensor).into())
    }

    fn import_score_subgraph(
        &mut self,
        subgraph: &GraphLayout,
        tensor_captures: &BTreeMap<TensorId, TensorId>,
        captured_tensors: &BTreeMap<TensorId, TensorSpec>,
    ) -> Result<()> {
        let sources = tensor_captures
            .iter()
            .map(|(&source, &captured)| (captured, source))
            .collect::<BTreeMap<_, _>>();
        let mut imported = subgraph
            .binding_template_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        for &id in &subgraph.binding_template_ids {
            imported.insert(resolve_binding_source(&subgraph.binding_replacements, id)?.0);
        }
        for id in imported {
            let record = subgraph.tensor_record(id)?;
            if let Some(&source) = sources.get(&id) {
                let source_tensor = captured_tensors
                    .get(&source)
                    .ok_or(FrontendError::TensorNotFound(source))?;
                // Child-only workspace captures retain automatically bound storage.
                if source != id
                    && (!source_tensor.is_workspace()
                        || self.binding_template_ids.contains(&source))
                {
                    self.tensors.entry(source).or_insert_with(|| {
                        let mut binding = record.clone();
                        binding.id = source;
                        binding.tensor = source_tensor.clone();
                        binding.access = None;
                        binding.indexes_memory = false;
                        binding.requires_int32_filter_offsets = false;
                        binding
                    });
                    let source_id = self.tensor_record(source)?.id;
                    self.binding_replacements.push(BindingReplacement {
                        kind: AliasTensorKind::CapturedTensor,
                        source_id,
                        target_id: record.id,
                        byte_offset: 0,
                    });
                }
            } else if record.access.is_some()
                && let Some(value) = record.tensor.runtime_scalar_value()
            {
                self.scalar_bindings.push((record.id, value));
            }
            if subgraph.binding_template_ids.contains(&record.id)
                && !self.binding_template_ids.contains(&record.id)
            {
                self.binding_template_ids.push(record.id);
            }
        }
        for (&id, record) in &subgraph.tensors {
            self.tensors
                .entry(id)
                .and_modify(|existing| {
                    if existing.access.is_none() && record.access.is_some() {
                        existing.tensor = record.tensor.clone();
                    }
                    existing.access = match (existing.access, record.access) {
                        (Some(parent), Some(child)) => Some(parent.merge(child)),
                        (None, access) | (access, None) => access,
                    };
                    existing.indexes_memory |= record.indexes_memory;
                    existing.requires_vector_alignment |= record.requires_vector_alignment;
                    existing.requires_int32_filter_offsets |= record.requires_int32_filter_offsets;
                })
                .or_insert_with(|| record.clone());
        }
        self.binding_replacements
            .extend(subgraph.binding_replacements.iter().copied());
        self.scalar_bindings
            .extend(subgraph.scalar_bindings.iter().copied());
        Ok(())
    }
}
