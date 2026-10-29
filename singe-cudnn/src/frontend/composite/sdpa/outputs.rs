use crate::{
    backend::tensor::TensorId,
    error::Result,
    frontend::{
        composite::sdpa::SdpaForwardOutputTensors, error::Error as FrontendError, graph::Graph,
    },
};

impl Graph {
    pub(crate) fn bind_attention_forward_outputs(
        &mut self,
        computed: SdpaForwardOutputTensors,
        declared: SdpaForwardOutputTensors,
    ) -> Result<()> {
        self.bind_sdpa_inferred_tensor(computed.output, declared.output, "attention output")?;
        for (computed, declared, name) in [
            (
                computed.logit_max,
                declared.logit_max,
                "attention logit maximum",
            ),
            (
                computed.score_sum_exp,
                declared.score_sum_exp,
                "attention exponential sum",
            ),
        ] {
            // C++ binds these summaries with their declared encodings rather than the inferred I/O default.
            match (computed, declared) {
                (Some(computed), Some(declared)) => {
                    self.bind_sdpa_declared_tensor(computed, declared, name)?;
                }
                (None, None) => {}
                _ => {
                    return Err(FrontendError::SdpaOptionalOutputMismatch { output: name }.into());
                }
            }
        }
        for (computed, declared, name) in [
            (computed.stats, declared.stats, "attention statistics"),
            (computed.rng_dump, declared.rng_dump, "attention rng dump"),
            (
                computed.absolute_max_probabilities,
                declared.absolute_max_probabilities,
                "attention probability amax",
            ),
            (
                computed.absolute_max_output,
                declared.absolute_max_output,
                "attention output amax",
            ),
        ] {
            self.bind_sdpa_inferred_optional_tensor(computed, declared, name)?;
        }
        Ok(())
    }

    pub(crate) fn bind_sdpa_inferred_tensor(
        &mut self,
        inferred: TensorId,
        output: TensorId,
        error_name: &'static str,
    ) -> Result<()> {
        let inferred_tensor = self.tensor_spec(inferred)?;
        let output_tensor = self.tensor_spec(output)?;
        if inferred_tensor.data_type != output_tensor.data_type {
            return Err(FrontendError::TensorDataTypeMismatch {
                tensor_id: output,
                operation: error_name,
                expected: inferred_tensor.data_type,
                actual: output_tensor.data_type,
            }
            .into());
        }
        self.bind_sdpa_declared_tensor(inferred, output, error_name)
    }

    fn bind_sdpa_declared_tensor(
        &mut self,
        inferred: TensorId,
        output: TensorId,
        error_name: &'static str,
    ) -> Result<()> {
        self.validate_tensor_dimensions(
            output,
            self.tensor_spec(inferred)?.shape.dimensions(),
            error_name,
        )?;
        self.retarget_attention_tensor(inferred, output)
    }

    pub(crate) fn bind_sdpa_inferred_optional_tensor(
        &mut self,
        inferred: Option<TensorId>,
        output: Option<TensorId>,
        error_name: &'static str,
    ) -> Result<()> {
        match (inferred, output) {
            (Some(inferred), Some(output)) => {
                self.bind_sdpa_inferred_tensor(inferred, output, error_name)
            }
            (None, None) => Ok(()),
            _ => Err(FrontendError::SdpaOptionalOutputMismatch { output: error_name }.into()),
        }
    }

    fn retarget_attention_tensor(&mut self, source: TensorId, target: TensorId) -> Result<()> {
        if source == target {
            return Ok(());
        }
        let constraints = self
            .runtime_shape_constraints
            .get(&source)
            .map(|bounds| -> Result<_> {
                bounds.validate_for(self.tensor_spec(source)?)?;
                let bounds = match self.runtime_shape_constraints.get(&target) {
                    Some(target_bounds) => bounds.intersection(target_bounds)?,
                    None => bounds.clone(),
                };
                bounds.validate_for(self.tensor_spec(target)?)?;
                Ok(bounds)
            })
            .transpose()?;
        let remap = |id: &mut TensorId| {
            if *id == source {
                *id = target;
            }
        };
        let optional = |id: &mut Option<TensorId>| {
            if let Some(id) = id {
                remap(id);
            }
        };
        let target_capture = self.capture_target(target)?;
        let operations = self
            .operations
            .iter()
            .map(|operation| {
                let uses = operation.tensor_uses(self.config.data_type_policy);
                if !uses.reads.contains(&source) && !uses.writes.contains(&source) {
                    return Ok(operation.clone());
                }
                operation
                    .try_map_tensor_ids(&|id| Ok(if id == source { target } else { id }), &|id| {
                        Ok((id == source).then(|| target_capture.clone()))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        self.replace_operations(operations);
        for tensor in self.tensors.values_mut() {
            optional(&mut tensor.ragged_offset);
        }
        for alias in &mut self.storage_views {
            remap(&mut alias.source);
            remap(&mut alias.target);
        }
        if let Some(constraints) = constraints {
            self.runtime_shape_constraints.insert(target, constraints);
        }
        self.runtime_shape_constraints.remove(&source);
        if self.vector_aligned_tensors.remove(&source) {
            self.vector_aligned_tensors.insert(target);
        }
        self.tensors.remove(&source);
        Ok(())
    }
}
