use std::collections::BTreeMap;

use crate::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{TensorId, TensorSpec},
    },
    error::Result,
    frontend::{
        bindings::resolve_binding_source,
        error::Error as FrontendError,
        graph::Graph,
        operation::{
            AttentionBackwardRequest, AttentionPrimitiveOperation, Operation, PointwiseOperation,
            SdpaOperation, SoftmaxStatsBase, TensorOperation,
        },
        plan::{AliasTensorKind, BindingReplacement},
    },
};

#[derive(Clone, Copy)]
enum StatsSource {
    Generated(SoftmaxStatsBase),
    Copied(TensorId),
}

fn storage_view_preserves_stats_values(
    source: &TensorSpec,
    target: &TensorSpec,
    byte_offset: i64,
) -> Result<bool> {
    if source.data_type != target.data_type
        || source.ragged_offset.is_some()
        || target.ragged_offset.is_some()
        || source.effective_reordering().is_some()
        || target.effective_reordering().is_some()
    {
        return Ok(false);
    }
    if byte_offset == 0
        && source.shape == target.shape
        && source.vectorization()? == target.vectorization()?
    {
        return Ok(true);
    }
    // A checked view can include stride gaps in its source span; those bytes have no known statistics base.
    // A dense source covers every scalar in its span, independently of axis order and vector lanes.
    let mut packed_stride = 1_u128;
    for axis in source.shape.stride_order() {
        let dimension = source.shape.dimensions()[axis];
        if dimension == 1 {
            continue;
        }
        if source.shape.strides()[axis] as u128 != packed_stride {
            return Ok(false);
        }
        let Some(next) = packed_stride.checked_mul(dimension as u128) else {
            return Ok(false);
        };
        packed_stride = next;
    }
    Ok(true)
}

impl Graph {
    #[expect(
        clippy::too_many_lines,
        reason = "native operation construction keeps its resource ownership and descriptor ordering together"
    )]
    #[expect(
        clippy::float_cmp,
        reason = "statistics provenance recognizes exact copies with unit coefficients"
    )]
    pub(crate) fn validate_attention_statistics_bases(&self) -> Result<()> {
        let aliases = self
            .storage_views
            .iter()
            .map(|alias| BindingReplacement {
                kind: AliasTensorKind::StorageView,
                source_id: alias.source,
                target_id: alias.target,
                byte_offset: alias.byte_offset,
            })
            .collect::<Vec<_>>();
        let root = |id| resolve_binding_source(&aliases, id).map(|(root, _)| root);
        let mut writers = BTreeMap::<TensorId, usize>::new();
        for operation in &self.operations {
            for id in operation.tensor_uses(self.config.data_type_policy).writes {
                *writers.entry(root(id)?).or_default() += 1;
            }
            let subgraphs = match operation {
                Operation::AttentionForward(request) => [request.config.score_subgraph(), None],
                Operation::AttentionBackward(request) => [
                    request.config().score_subgraph(),
                    request.config().score_subgraph_bprop(),
                ],
                Operation::Sdpa(operation) => [operation.score_subgraph(), None],
                _ => [None, None],
            };
            // Captured storage may change inside a custom score graph; parent-only writers cannot establish its base.
            for subgraph in subgraphs.into_iter().flatten() {
                for id in subgraph.captured_tensor_ids() {
                    *writers.entry(root(id)?).or_default() += 1;
                }
            }
        }
        let mut sources = BTreeMap::new();
        for operation in &self.operations {
            let source = match operation {
                Operation::AttentionForward(request) => request
                    .outputs
                    .stats
                    .map(|id| (id, StatsSource::Generated(request.config.stats_base()))),
                Operation::Sdpa(SdpaOperation::Forward { stats, config, .. }) => {
                    stats.map(|id| (id, StatsSource::Generated(config.stats_base())))
                }
                Operation::AttentionPrimitive(AttentionPrimitiveOperation::Softmax {
                    stats,
                    stats_base,
                    ..
                }) => stats.map(|id| (id, StatsSource::Generated(*stats_base))),
                Operation::Pointwise(PointwiseOperation::Unary {
                    mode: PointwiseMode::Identity,
                    input,
                    output,
                    alpha1,
                    ..
                }) if *alpha1 == 1.0 => Some((*output, StatsSource::Copied(*input))),
                Operation::Tensor(
                    TensorOperation::Reshape { input, output, .. }
                    | TensorOperation::Transpose { input, output, .. }
                    | TensorOperation::Slice { input, output, .. },
                ) if self.tensor_spec(*input)?.data_type
                    == self.tensor_spec(*output)?.data_type =>
                {
                    Some((*output, StatsSource::Copied(*input)))
                }
                _ => None,
            };
            if let Some((id, source)) = source
                && writers.get(&root(id)?) == Some(&1)
            {
                sources.insert(id, source);
            }
        }
        for alias in &self.storage_views {
            if writers.get(&root(alias.target)?) == Some(&1)
                && storage_view_preserves_stats_values(
                    self.tensor_spec(alias.source)?,
                    self.tensor_spec(alias.target)?,
                    alias.byte_offset,
                )?
            {
                sources
                    .entry(alias.target)
                    .or_insert(StatsSource::Copied(alias.source));
            }
        }
        // Only uniquely written values establish a base; overwritten or externally supplied values remain caller contracts.
        let mut bases = BTreeMap::new();
        loop {
            let mut changed = false;
            for (&id, source) in &sources {
                if bases.contains_key(&id) {
                    continue;
                }
                let base = match source {
                    StatsSource::Generated(base) => Some(*base),
                    StatsSource::Copied(input) => bases.get(input).copied(),
                };
                if let Some(base) = base {
                    bases.insert(id, base);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for operation in &self.operations {
            let stats = match operation {
                Operation::AttentionBackward(AttentionBackwardRequest::Floating {
                    inputs, ..
                }) => Some(inputs.stats),
                Operation::AttentionBackward(AttentionBackwardRequest::Fp8 { inputs, .. }) => {
                    Some(inputs.stats)
                }
                Operation::AttentionBackward(AttentionBackwardRequest::Mxfp8 {
                    inputs, ..
                }) => Some(inputs.stats),
                Operation::Sdpa(SdpaOperation::Backward { tensors, .. }) => Some(tensors.stats),
                _ => None,
            };
            if let Some(actual) = stats.and_then(|id| bases.get(&id))
                && *actual != SoftmaxStatsBase::NaturalLog
            {
                return Err(
                    FrontendError::SdpaBackwardStatsBaseUnsupported { actual: *actual }.into(),
                );
            }
        }
        Ok(())
    }
}
