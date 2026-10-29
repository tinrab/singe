use std::borrow::Cow;

use crate::{
    backend::tensor::Shape,
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::{
            SdpaForwardOutputTensors, infer_sdpa_output_shape, infer_sdpa_scores_shape,
            support::SDPA_STATS_DATA_TYPE,
        },
        error::{Error as FrontendError, SdpaFeature, SdpaRequirement},
        graph::{DataTypePolicy, Graph, TensorOutputConfig, reduce_last_axis},
        operation::{AttentionConfig, AttentionForwardInputs, AttentionForwardRequest, Operation},
    },
};

impl Graph {
    pub(crate) fn normalized_attention_forward_config<'a>(
        &self,
        config: &'a AttentionConfig,
        outputs: SdpaForwardOutputTensors,
    ) -> Result<Cow<'a, AttentionConfig>> {
        let requested = outputs.aux_outputs();
        let mut config = if config.fused().aux_outputs() == requested {
            Cow::Borrowed(config)
        } else {
            Cow::Owned(config.clone().with_aux_outputs(requested))
        };
        if config
            .dropout()
            .is_some_and(|dropout| dropout.probability() == 0.0)
        {
            config.to_mut().remove_zero_probability_dropout();
        }
        let paged = config.paged();
        if paged.k().is_some()
            && paged.v().is_some()
            && paged.max_sequence_length_key_value().is_none()
            && config.modifiers().bias().is_none()
            && let Some(rng_dump) = outputs.rng_dump
        {
            self.validate_tensor_rank(rng_dump, 4, "attention rng dump")?;
            let length = self.shape(rng_dump)?.dimensions()[3];
            config = Cow::Owned(
                config
                    .into_owned()
                    .with_paged_cache(paged.with_max_sequence_length_key_value(length)),
            );
        }
        Ok(config)
    }

    pub(crate) fn record_explicit_attention_forward(
        &mut self,
        inputs: AttentionForwardInputs,
        outputs: SdpaForwardOutputTensors,
        config: &AttentionConfig,
    ) -> Result<()> {
        let config = self
            .normalized_attention_forward_config(config, outputs)?
            .into_owned();
        let policies = outputs.try_map(|id| -> Result<_> {
            let tensor = self.tensor_spec(id)?;
            let policy = TensorOutputConfig::new().with_data_type(tensor.data_type);
            Ok(if tensor.is_virtual() {
                policy
            } else {
                policy.output_tensor()
            })
        })?;
        let mut composed = self.clone();
        let computed = composed.record_attention_forward(inputs, policies, config)?;
        composed.bind_attention_forward_outputs(computed, outputs)?;
        *self = composed;
        Ok(())
    }

    pub(crate) fn record_attention_forward(
        &mut self,
        inputs: AttentionForwardInputs,
        outputs: SdpaForwardOutputTensors<TensorOutputConfig>,
        mut config: AttentionConfig,
    ) -> Result<SdpaForwardOutputTensors> {
        config = config.with_aux_outputs(outputs.aux_outputs());
        config.modifiers().sequence_inputs().validate_forward()?;
        config.remove_zero_probability_dropout();
        let [q, k, v] = inputs.query_key_value();
        let kv = self.attention_kv_shapes(k, v, &config)?;
        let scores = infer_sdpa_scores_shape(self.shape(q)?, &kv.key)?;
        let output_shape = infer_sdpa_output_shape(&scores, &kv.value)?;
        let aux_shape = reduce_last_axis(&scores)?;
        let request = config.fused().aux_outputs();
        if request.rng_dump() && config.dropout().is_none() {
            return Err(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::PhiloxDropout,
                operation: "sdpa rng dump",
            }
            .into());
        }
        if matches!(inputs, AttentionForwardInputs::Mxfp8(_)) && config.paged().is_enabled() {
            return Err(FrontendError::SdpaUnsupported {
                operation: "mxfp8 sdpa",
                feature: SdpaFeature::PagedCaches,
                sm_version: None,
            }
            .into());
        }
        let aux_type = self.tensor_spec(q)?.data_type;
        let output_type = match inputs {
            AttentionForwardInputs::Mxfp8(_) => DataType::BF16,
            _ => aux_type,
        };
        let scalar_shape = Shape::contiguous([1, 1, 1, 1])?;
        let specs = SdpaForwardOutputTensors {
            output: (
                outputs
                    .output
                    .into_spec(self.config.data_type_policy, output_type, output_shape),
                "O",
            ),
            stats: outputs.stats.map(|output| {
                (
                    output.into_spec(DataTypePolicy::default(), DataType::F32, aux_shape.clone()),
                    "Stats",
                )
            }),
            logit_max: outputs.logit_max.map(|output| {
                (
                    output.into_spec(self.config.data_type_policy, aux_type, aux_shape.clone()),
                    "Max",
                )
            }),
            score_sum_exp: outputs.score_sum_exp.map(|output| {
                (
                    output.into_spec(self.config.data_type_policy, aux_type, aux_shape),
                    "Sum_exp",
                )
            }),
            rng_dump: outputs.rng_dump.map(|output| {
                (
                    output.into_spec(DataTypePolicy::default(), DataType::F32, scores),
                    "RNG_DUMP",
                )
            }),
            absolute_max_probabilities: outputs.absolute_max_probabilities.map(|output| {
                (
                    output.into_spec(
                        DataTypePolicy::default(),
                        DataType::F32,
                        scalar_shape.clone(),
                    ),
                    "Amax_S",
                )
            }),
            absolute_max_output: outputs.absolute_max_output.map(|output| {
                (
                    output.into_spec(DataTypePolicy::default(), DataType::F32, scalar_shape),
                    "Amax_O",
                )
            }),
        };
        let scope = config.fused().name();
        let outputs = specs.map(|(spec, port)| self.inferred_tensor(spec, scope, port));
        self.operations
            .push(Operation::AttentionForward(AttentionForwardRequest {
                inputs,
                outputs,
                config,
            }));
        Ok(outputs)
    }

    pub(crate) fn expand_attention_requests(&mut self) -> Result<()> {
        if !self
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::AttentionForward(_)))
        {
            return Ok(());
        }
        if self.config.sm_version.is_none_or(|sm| sm <= 0) {
            return Err(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::TargetArchitecture,
                operation: "sdpa implementation selection",
            }
            .into());
        }
        for operation in self.take_operations() {
            let Operation::AttentionForward(mut request) = operation else {
                self.operations.push(operation);
                continue;
            };
            if let Cow::Owned(config) =
                self.normalized_attention_forward_config(&request.config, request.outputs)?
            {
                request.config = config;
            }
            if let Some(stats) = request.outputs.stats {
                self.validate_sdpa_tensor_data_type_match(
                    stats,
                    DataType::F32,
                    SDPA_STATS_DATA_TYPE,
                )?;
            }
            let output_data_type = self.tensor_spec(request.outputs.output)?.data_type;
            let implementation = self.resolve_attention_request(&request)?;
            request.config = request.config.with_implementation(implementation.into());
            self.promote_index_tensors(request.config.modifiers().sequence_inputs().tensors())?;
            let computed = match request.inputs {
                AttentionForwardInputs::Floating(inputs) => self.expand_floating_attention(
                    inputs,
                    output_data_type,
                    implementation,
                    &request.config,
                )?,
                AttentionForwardInputs::Fp8(inputs) => self.expand_fp8_attention(
                    inputs,
                    output_data_type,
                    implementation,
                    &request.config,
                )?,
                AttentionForwardInputs::Mxfp8(inputs) => self.expand_mxfp8_attention(
                    inputs,
                    output_data_type,
                    implementation,
                    &request.config,
                )?,
            };
            self.bind_attention_forward_outputs(computed, request.outputs)?;
        }
        Ok(())
    }
}
