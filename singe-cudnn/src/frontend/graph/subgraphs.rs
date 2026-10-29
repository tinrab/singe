use std::collections::BTreeMap;

use crate::{
    backend::tensor::{TensorId, TensorSpec},
    context::Context,
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, RuntimeShapeConstraints, ValidatedGraph},
        operation::{
            AttentionDiagonalAlignment, AttentionDiagonalBand, AttentionMaskMode,
            DirectSdpaForwardConfig, Operation, SdpaOperation, SdpaScoreModifiers,
        },
    },
};

#[derive(Debug, Clone)]
pub struct ScoreSubgraph<G> {
    pub(crate) graph: G,
    pub(crate) input: TensorId,
    pub(crate) output: TensorId,
    pub(crate) tensor_captures: BTreeMap<TensorId, TensorId>,
    pub(crate) captured_tensors: BTreeMap<TensorId, TensorSpec>,
}

impl Graph {
    pub(crate) fn validate_sdpa_subgraphs(
        &self,
        context: Option<&Context>,
    ) -> Result<BTreeMap<usize, ScoreSubgraph<ValidatedGraph>>> {
        let mut validated = BTreeMap::new();
        for (index, operation) in self.operations.iter().enumerate() {
            let subgraph = match operation {
                Operation::Sdpa(SdpaOperation::Forward {
                    tensors,
                    config,
                    score_modifiers,
                    score_subgraph,
                    ..
                }) => self.build_sdpa_unified_subgraph(
                    tensors.query,
                    tensors.key,
                    tensors.value,
                    config,
                    *score_modifiers,
                    score_subgraph.as_deref(),
                )?,
                Operation::Sdpa(SdpaOperation::Backward { tensors, config })
                    if config.causal_mask() || config.score_subgraph().is_some() =>
                {
                    let mask_mode = match (config.causal_mask(), config.padding_mask()) {
                        (true, true) => AttentionMaskMode::DiagonalBand(
                            AttentionDiagonalBand::causal(AttentionDiagonalAlignment::TopLeft)
                                .with_padding(true),
                        ),
                        (false, true) => AttentionMaskMode::Padding,
                        (true, false) => AttentionMaskMode::DiagonalBand(
                            AttentionDiagonalBand::causal(AttentionDiagonalAlignment::TopLeft),
                        ),
                        (false, false) => AttentionMaskMode::None,
                    };
                    let modifiers = SdpaScoreModifiers::new()
                        .with_sequence_inputs(config.sequence_inputs())
                        .with_mask_mode(mask_mode);
                    let mut forward = DirectSdpaForwardConfig::new()
                        .with_sequence_inputs(config.sequence_inputs())
                        .with_padding_mask(config.padding_mask());
                    forward.inherit_data_type_policy(config.data_type_policy());
                    self.build_sdpa_unified_subgraph(
                        tensors.query,
                        tensors.key,
                        tensors.value,
                        &forward,
                        modifiers,
                        config.score_subgraph(),
                    )?
                }
                _ => None,
            };
            if let Some(subgraph) = subgraph {
                validated.insert(
                    index,
                    ScoreSubgraph {
                        graph: subgraph.graph.validate(context)?,
                        input: subgraph.input,
                        output: subgraph.output,
                        tensor_captures: subgraph.tensor_captures,
                        captured_tensors: subgraph.captured_tensors,
                    },
                );
            }
        }
        Ok(validated)
    }

    /// Resolves shared requirements before parent descriptors are constructed.
    /// Rank promotion, stronger alignment and intersected bounds only tighten the shared layout.
    pub(crate) fn merge_shared_sdpa_layouts(
        &mut self,
        subgraphs: &BTreeMap<usize, ScoreSubgraph<ValidatedGraph>>,
    ) -> Result<bool> {
        let mut changed = false;
        for subgraph in subgraphs.values() {
            for (&source, &captured) in &subgraph.tensor_captures {
                if source != captured {
                    continue;
                }
                let record = subgraph.graph.layout.tensor_record(captured)?;
                if record.access.is_none() {
                    continue;
                }
                let tensor = self.tensor_spec(source)?.clone();
                let child = &record.tensor;
                if !tensor.storage_metadata_matches(child)? {
                    return Err(FrontendError::Compile(format!(
                        "shared attention tensor {source} metadata mismatch"
                    ))
                    .into());
                }
                if Self::attention_index_layout_is_promotion(&tensor, child)? {
                    self.promote_index_tensors([source])?;
                    changed = true;
                } else if Self::attention_index_layout_is_promotion(child, &tensor)? {
                    // An earlier sibling promoted the parent after this child was validated.
                    changed = true;
                } else if tensor.shape.dimensions() != child.shape.dimensions()
                    || tensor.shape.strides() != child.shape.strides()
                {
                    return Err(FrontendError::Compile(format!(
                        "shared attention tensor {source} layout mismatch"
                    ))
                    .into());
                }
                let alignment = tensor.byte_alignment.max(child.byte_alignment);
                if alignment != tensor.byte_alignment {
                    self.tensor_spec_mut(source)?.byte_alignment = alignment;
                    changed = true;
                }
                if record.requires_vector_alignment && self.vector_aligned_tensors.insert(source) {
                    changed = true;
                }
                if let Some(bounds) = subgraph
                    .graph
                    .expanded
                    .runtime_shape_constraints
                    .get(&captured)
                {
                    let bounds = if Self::attention_index_layout_is_promotion(
                        child,
                        self.tensor_spec(source)?,
                    )? {
                        RuntimeShapeConstraints::create(
                            [bounds.min_dimensions[0], 1, 1, 1],
                            [bounds.max_dimensions[0], 1, 1, 1],
                        )?
                    } else {
                        bounds.clone()
                    };
                    let bounds = match self.runtime_shape_constraints.get(&source) {
                        Some(existing) => bounds.intersection(existing)?,
                        None => bounds,
                    };
                    bounds.validate_for(self.tensor_spec(source)?)?;
                    if self.runtime_shape_constraints.get(&source) != Some(&bounds) {
                        self.runtime_shape_constraints.insert(source, bounds);
                        changed = true;
                    }
                }
            }
        }
        Ok(changed)
    }

    pub(crate) fn attention_index_layout_is_promotion(
        vector: &TensorSpec,
        promoted: &TensorSpec,
    ) -> Result<bool> {
        Ok(vector.shape.rank() == 1
            && promoted.shape.dimensions() == [vector.shape.dimensions()[0], 1, 1, 1]
            && promoted.shape.strides() == [vector.shape.strides()[0], 1, 1, 1]
            && vector.vectorization()?.is_none()
            && promoted.vectorization()?.is_none())
    }
}
