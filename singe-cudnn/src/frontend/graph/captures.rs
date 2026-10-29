use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::{
    backend::tensor::{TensorId, TensorSpec},
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, RuntimeShapeConstraints, StorageView},
        operation::Operation,
    },
};

/// A captured declaration and its ragged metadata in the destination graph's namespace.
#[derive(Debug, Clone)]
pub struct TensorCaptureTarget {
    id: TensorId,
    tensors: BTreeMap<TensorId, TensorSpec>,
}

impl TensorCaptureTarget {
    fn from_tensors(tensors: &BTreeMap<TensorId, TensorSpec>, id: TensorId) -> Result<Self> {
        let mut captured = BTreeMap::new();
        let mut current = Some(id);
        let mut seen = HashSet::new();
        while let Some(id) = current {
            if !seen.insert(id) {
                return Err(FrontendError::RaggedOffsetCycle { tensor_id: id }.into());
            }
            let tensor = tensors.get(&id).ok_or(FrontendError::TensorNotFound(id))?;
            current = tensor.ragged_offset;
            captured.insert(id, tensor.clone());
        }
        Ok(Self {
            id,
            tensors: captured,
        })
    }
}

impl Graph {
    pub(crate) fn capture_target(&self, id: TensorId) -> Result<TensorCaptureTarget> {
        TensorCaptureTarget::from_tensors(&self.tensors, id)
    }

    /// Finds the declaration that supplies an external capture, including captures owned only by a nested score graph.
    pub(crate) fn captured_tensor_owner(&self, id: TensorId) -> Option<&Self> {
        if self.tensors.contains_key(&id) {
            return Some(self);
        }
        self.operations
            .iter()
            .flat_map(Operation::score_subgraphs)
            .filter(|subgraph| subgraph.captured_tensor_ids().contains(&id))
            .find_map(|subgraph| subgraph.graph().captured_tensor_owner(id))
    }

    /// Rewrites external declarations while retaining local tensor scopes and process-local graph state.
    #[expect(
        clippy::too_many_lines,
        reason = "native operation construction keeps its resource ownership and descriptor ordering together"
    )]
    pub(crate) fn try_map_captures(
        &self,
        input: TensorId,
        remap: &dyn Fn(TensorId) -> Result<Option<TensorCaptureTarget>>,
    ) -> Result<(Self, BTreeMap<TensorId, TensorId>)> {
        let mut captures = self.captured_tensor_ids();
        captures.remove(&input);
        let targets = captures
            .iter()
            .filter_map(|&id| match remap(id) {
                Ok(Some(target)) => Some(Ok((id, target))),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        if targets.is_empty() {
            return Ok((self.clone(), BTreeMap::new()));
        }
        let mut target_tensors = BTreeMap::new();
        for target in targets.values() {
            for (&id, tensor) in &target.tensors {
                merge_tensor(&mut target_tensors, id, tensor.clone())?;
            }
        }
        let mut allocation = self.new_subgraph();
        allocation
            .runtime
            .reserved_tensor_ids
            .extend(target_tensors.keys().copied());
        let mut ids = BTreeMap::new();
        for (&id, tensor) in &self.tensors {
            let target = targets.get(&id).map_or_else(
                || {
                    if target_tensors.contains_key(&id) && !captures.contains(&id) {
                        allocation.tensor(tensor.clone())
                    } else {
                        id
                    }
                },
                |target| target.id,
            );
            ids.insert(id, target);
        }
        ids.extend(targets.iter().map(|(&id, target)| (id, target.id)));
        let map_id = |id: TensorId| ids.get(&id).copied().unwrap_or(id);
        let mut tensors = BTreeMap::new();
        for (&id, original) in &self.tensors {
            let target = map_id(id);
            let tensor = if captures.contains(&id) && target_tensors.contains_key(&target) {
                let mut tensor = target_tensors[&target].clone();
                if original.data_type != tensor.data_type {
                    return Err(FrontendError::TensorDataTypeMismatch {
                        tensor_id: target,
                        operation: "attention capture",
                        expected: tensor.data_type,
                        actual: original.data_type,
                    }
                    .into());
                }
                if Self::attention_index_layout_is_promotion(&tensor, original)? {
                    tensor.shape = original.shape.clone();
                } else if original.shape.dimensions() != tensor.shape.dimensions()
                    && !Self::attention_index_layout_is_promotion(original, &tensor)?
                {
                    return Err(FrontendError::TensorDimensionsMismatch {
                        tensor_id: target,
                        operation: "attention capture",
                        expected: tensor.shape.dimensions().to_vec(),
                        actual: original.shape.dimensions().to_vec(),
                    }
                    .into());
                }
                tensor.byte_alignment = tensor.byte_alignment.max(original.byte_alignment);
                tensor
            } else {
                let mut tensor = original.clone();
                tensor.ragged_offset = tensor.ragged_offset.map(map_id);
                tensor
            };
            merge_tensor(&mut tensors, target, tensor)?;
        }
        for (id, tensor) in target_tensors {
            merge_tensor(&mut tensors, id, tensor)?;
        }
        let operations = self
            .operations
            .iter()
            .map(|operation| {
                operation.try_map_tensor_ids(
                    &|id| {
                        self.tensor_spec(id)?;
                        Ok(map_id(id))
                    },
                    &|id| match ids.get(&id) {
                        Some(&target) => {
                            TensorCaptureTarget::from_tensors(&tensors, target).map(Some)
                        }
                        None => remap(id),
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut bounds: BTreeMap<TensorId, RuntimeShapeConstraints> = BTreeMap::new();
        for (&id, original) in &self.runtime_shape_constraints {
            let source = self.tensor_spec(id)?;
            original.validate_for(source)?;
            let target = map_id(id);
            let tensor = tensors
                .get(&target)
                .ok_or(FrontendError::TensorNotFound(target))?;
            let incoming = if Self::attention_index_layout_is_promotion(source, tensor)? {
                RuntimeShapeConstraints::create(
                    [original.min_dimensions[0], 1, 1, 1],
                    [original.max_dimensions[0], 1, 1, 1],
                )?
            } else {
                original.clone()
            };
            let incoming = match bounds.get(&target) {
                Some(existing) => incoming.intersection(existing)?,
                None => incoming,
            };
            incoming.validate_for(tensor)?;
            bounds.insert(target, incoming);
        }
        let mut mapped = self.clone();
        mapped.tensors = tensors;
        mapped.replace_operations(operations);
        mapped.runtime_shape_constraints = bounds;
        mapped.vector_aligned_tensors = self
            .vector_aligned_tensors
            .iter()
            .copied()
            .map(map_id)
            .collect::<BTreeSet<_>>();
        mapped.storage_views = self
            .storage_views
            .iter()
            .map(|binding| StorageView {
                source: map_id(binding.source),
                target: map_id(binding.target),
                byte_offset: binding.byte_offset,
            })
            .collect();
        mapped
            .runtime
            .reserved_tensor_ids
            .extend(mapped.tensors.keys().copied());
        Ok((mapped, ids))
    }
}

fn merge_tensor(
    tensors: &mut BTreeMap<TensorId, TensorSpec>,
    id: TensorId,
    incoming: TensorSpec,
) -> Result<()> {
    let Some(existing) = tensors.get_mut(&id) else {
        tensors.insert(id, incoming);
        return Ok(());
    };
    if existing.data_type != incoming.data_type {
        return Err(FrontendError::TensorDataTypeMismatch {
            tensor_id: id,
            operation: "attention capture merge",
            expected: existing.data_type,
            actual: incoming.data_type,
        }
        .into());
    }
    if !existing.storage_metadata_matches(&incoming)? {
        return Err(FrontendError::Compile(format!(
            "captured tensor `{id}` storage metadata mismatch"
        ))
        .into());
    }
    if Graph::attention_index_layout_is_promotion(existing, &incoming)? {
        existing.shape = incoming.shape.clone();
    } else if !Graph::attention_index_layout_is_promotion(&incoming, existing)?
        && (existing.shape.dimensions() != incoming.shape.dimensions()
            || existing.shape.strides() != incoming.shape.strides())
    {
        return Err(
            FrontendError::Compile(format!("captured tensor `{id}` layout mismatch")).into(),
        );
    }
    existing.byte_alignment = existing.byte_alignment.max(incoming.byte_alignment);
    Ok(())
}
