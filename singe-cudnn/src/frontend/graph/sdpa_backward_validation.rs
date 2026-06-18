#[cfg(test)]
use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{SdpaBackwardGradientTensors, SdpaBackwardInputs, SdpaBackwardOutputs},
        graph::Graph,
        infer::{infer_sdpa_scores_shape, reduce_last_axis},
        operation::{DirectSdpaBackwardConfig, Operation},
        support,
    },
    tensor::{Shape, TensorSpec},
    utility::check_range,
    version,
};

#[cfg(test)]
#[derive(Debug, Clone)]
struct SdpaBackwardTensorSpecs {
    q: TensorSpec,
    k: TensorSpec,
    v: TensorSpec,
    o: TensorSpec,
    stats: TensorSpec,
    scale: TensorSpec,
    d_o: TensorSpec,
    d_q: TensorSpec,
    d_k: TensorSpec,
    d_v: TensorSpec,
}

#[cfg(test)]
impl SdpaBackwardTensorSpecs {
    fn resolve(
        graph: &Graph,
        inputs: SdpaBackwardInputs,
        gradients: SdpaBackwardGradientTensors,
    ) -> Result<Self> {
        Ok(Self {
            q: graph.tensor_config(inputs.query)?.clone(),
            k: graph.tensor_config(inputs.key)?.clone(),
            v: graph.tensor_config(inputs.value)?.clone(),
            o: graph.tensor_config(inputs.output)?.clone(),
            stats: graph.tensor_config(inputs.stats)?.clone(),
            scale: graph.tensor_config(inputs.scale)?.clone(),
            d_o: graph.tensor_config(inputs.output_gradient)?.clone(),
            d_q: graph.tensor_config(gradients.query_gradient)?.clone(),
            d_k: graph.tensor_config(gradients.key_gradient)?.clone(),
            d_v: graph.tensor_config(gradients.value_gradient)?.clone(),
        })
    }

    fn is_ragged(&self) -> bool {
        self.q.ragged_offset.is_some()
            || self.k.ragged_offset.is_some()
            || self.v.ragged_offset.is_some()
            || self.o.ragged_offset.is_some()
            || self.stats.ragged_offset.is_some()
            || self.d_o.ragged_offset.is_some()
            || self.d_q.ragged_offset.is_some()
            || self.d_k.ragged_offset.is_some()
            || self.d_v.ragged_offset.is_some()
    }
}

impl Graph {
    #[cfg(test)]
    pub(crate) fn normalize_sdpa_backward_max_total_seq_len_for_version(
        &self,
        config: DirectSdpaBackwardConfig,
        d_qk: i64,
        d_v: i64,
        support: support::SdpaBackwardSupport,
    ) -> DirectSdpaBackwardConfig {
        let has_max_total_seq_len = config.max_total_sequence_length_query().is_some()
            || config.max_total_sequence_length_key_value().is_some();
        if !has_max_total_seq_len {
            return config;
        }

        if support.should_clear_max_total_sequence_lengths(d_qk, d_v) {
            return config.clear_max_total_sequence_lengths();
        }

        config
    }

    #[cfg(test)]
    pub(crate) fn sdpa_backward_direct(
        &mut self,
        inputs: SdpaBackwardInputs,
        gradients: SdpaBackwardGradientTensors,
        config: DirectSdpaBackwardConfig,
    ) -> Result<()> {
        self.sdpa_backward_direct_for_version(version()?.raw(), inputs, gradients, config)
    }

    #[cfg(test)]
    pub(crate) fn sdpa_backward_direct_for_version(
        &mut self,
        cudnn_version: u64,
        inputs: SdpaBackwardInputs,
        gradients: SdpaBackwardGradientTensors,
        config: DirectSdpaBackwardConfig,
    ) -> Result<()> {
        let support = support::SdpaBackwardSupport::new(cudnn_version, self.sm_version());
        let config =
            self.validate_sdpa_backward_direct_for_version(support, inputs, gradients, config)?;
        let SdpaBackwardInputs {
            query: q,
            key: k,
            value: v,
            output: o,
            output_gradient: d_o,
            stats,
            scale,
        } = inputs;
        let SdpaBackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            ..
        } = gradients;

        self.operations.push(Operation::SdpaBackward {
            q,
            k,
            v,
            o,
            stats,
            scale,
            d_o,
            d_q,
            d_k,
            d_v,
            sink_gradient: config.sink_gradient(),
            config: Box::new(config),
        });
        Ok(())
    }

    #[cfg(test)]
    fn validate_sdpa_backward_direct_for_version(
        &self,
        support: support::SdpaBackwardSupport,
        inputs: SdpaBackwardInputs,
        gradients: SdpaBackwardGradientTensors,
        config: DirectSdpaBackwardConfig,
    ) -> Result<DirectSdpaBackwardConfig> {
        let SdpaBackwardGradientTensors {
            bias_gradient,
            rng_dump,
            sink_token_gradient,
            ..
        } = gradients;
        if bias_gradient.is_some() || rng_dump.is_some() || sink_token_gradient.is_some() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward direct auxiliary gradients".into(),
            });
        }

        let config = config;
        let tensors = SdpaBackwardTensorSpecs::resolve(self, inputs, gradients)?;

        self.validate_sdpa_backward_tensor_invariants(inputs, gradients, &tensors, support)?;
        self.validate_sdpa_backward_optional_tensors(&config, &tensors)?;
        self.validate_sdpa_backward_ragged_config(&config, &tensors, support)?;

        Ok(self.normalize_sdpa_backward_max_total_seq_len_for_version(
            config,
            tensors.q.shape.dimensions()[3],
            tensors.v.shape.dimensions()[3],
            support,
        ))
    }

    #[cfg(test)]
    fn validate_sdpa_backward_tensor_invariants(
        &self,
        inputs: SdpaBackwardInputs,
        gradients: SdpaBackwardGradientTensors,
        tensors: &SdpaBackwardTensorSpecs,
        support: support::SdpaBackwardSupport,
    ) -> Result<()> {
        let SdpaBackwardInputs {
            query: q,
            key: k,
            value: v,
            output: o,
            output_gradient: d_o,
            stats,
            scale: _,
        } = inputs;
        let SdpaBackwardGradientTensors {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            ..
        } = gradients;

        for (tensor, operation) in [
            (k, "sdpa backward k data type"),
            (v, "sdpa backward v data type"),
            (o, "sdpa backward output data type"),
            (d_o, "sdpa backward d_o data type"),
            (d_q, "sdpa backward d_q data type"),
            (d_k, "sdpa backward d_k data type"),
            (d_v, "sdpa backward d_v data type"),
        ] {
            self.validate_tensor_data_type(tensor, tensors.q.data_type, operation)?;
        }
        let aux_data_type = self.effective_compute_data_type(DataType::F32);
        self.validate_tensor_data_type(stats, aux_data_type, "sdpa backward stats data type")?;
        self.validate_tensor_rank(q, 4, "sdpa backward q rank")?;
        self.validate_tensor_rank(k, 4, "sdpa backward k rank")?;
        self.validate_tensor_rank(v, 4, "sdpa backward v rank")?;
        self.validate_tensor_rank(o, 4, "sdpa backward output rank")?;
        self.validate_tensor_rank(stats, 4, "sdpa backward stats rank")?;

        let q_dims = tensors.q.shape.dimensions();
        let k_dims = tensors.k.shape.dimensions();
        let v_dims = tensors.v.shape.dimensions();
        if q_dims[2] == 1 && k_dims[2] == 1 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward sequence lengths".into(),
            });
        }
        if q_dims[1] % k_dims[1] != 0 || q_dims[1] % v_dims[1] != 0 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward gqa heads".into(),
            });
        }

        support.require_data_type(tensors.q.data_type)?;
        support.require_architecture(tensors.q.data_type, q_dims[3], v_dims[3])?;
        self.validate_sdpa_layout_last_stride_one(q, "sdpa backward q layout")?;
        self.validate_sdpa_layout_last_stride_one(k, "sdpa backward k layout")?;
        self.validate_sdpa_layout_last_stride_one(v, "sdpa backward v layout")?;
        self.validate_sdpa_layout_last_stride_one(o, "sdpa backward o layout")?;
        self.validate_sdpa_layout_last_stride_one(stats, "sdpa backward stats layout")?;
        self.validate_sdpa_layout_last_stride_one(d_o, "sdpa backward d_o layout")?;
        self.validate_sdpa_layout_last_stride_one(d_q, "sdpa backward d_q layout")?;
        self.validate_sdpa_layout_last_stride_one(d_k, "sdpa backward d_k layout")?;
        self.validate_sdpa_layout_last_stride_one(d_v, "sdpa backward d_v layout")?;
        self.validate_tensor_dimensions(
            d_o,
            tensors.o.shape.dimensions(),
            "sdpa backward d_o shape",
        )?;
        self.validate_tensor_strides(d_o, tensors.o.shape.strides(), "sdpa backward d_o strides")?;
        self.validate_tensor_dimensions(
            d_q,
            tensors.q.shape.dimensions(),
            "sdpa backward d_q shape",
        )?;
        self.validate_tensor_strides(d_q, tensors.q.shape.strides(), "sdpa backward d_q strides")?;
        self.validate_tensor_dimensions(
            d_k,
            tensors.k.shape.dimensions(),
            "sdpa backward d_k shape",
        )?;
        self.validate_tensor_strides(d_k, tensors.k.shape.strides(), "sdpa backward d_k strides")?;
        self.validate_tensor_dimensions(
            d_v,
            tensors.v.shape.dimensions(),
            "sdpa backward d_v shape",
        )?;
        self.validate_tensor_strides(d_v, tensors.v.shape.strides(), "sdpa backward d_v strides")?;

        if tensors.scale.shape.element_count()? != 1 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward scale".into(),
            });
        }
        let expected_stats_shape = reduce_last_axis(&infer_sdpa_scores_shape(
            &tensors.q.shape,
            &tensors.k.shape,
        )?)?;
        self.validate_tensor_dimensions(
            stats,
            expected_stats_shape.dimensions(),
            "sdpa backward stats shape",
        )?;
        self.validate_tensor_strides(
            stats,
            expected_stats_shape.strides(),
            "sdpa backward stats strides",
        )?;

        Ok(())
    }

    #[cfg(test)]
    fn validate_sdpa_backward_optional_tensors(
        &self,
        config: &DirectSdpaBackwardConfig,
        tensors: &SdpaBackwardTensorSpecs,
    ) -> Result<()> {
        match (
            config.sequence_length_query(),
            config.sequence_length_key_value(),
        ) {
            (Some(sequence_length_query), Some(sequence_length_key_value)) => {
                self.validate_sdpa_sequence_lengths(
                    sequence_length_query,
                    sequence_length_key_value,
                    tensors.q.shape.dimensions()[0],
                    "sdpa backward seq len data type",
                )?;
            }
            (None, None) => {}
            _ => {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward seq lens".into(),
                });
            }
        }
        if let Some(sink) = config.sink() {
            let expected_sink_shape =
                Shape::contiguous([1, tensors.q.shape.dimensions()[1], 1, 1])?;
            self.validate_tensor_data_type(sink, DataType::F32, "sdpa backward sink data type")?;
            self.validate_tensor_dimensions(
                sink,
                expected_sink_shape.dimensions(),
                "sdpa backward sink shape",
            )?;
            self.validate_tensor_strides(
                sink,
                expected_sink_shape.strides(),
                "sdpa backward sink strides",
            )?;
        }
        if let Some(sink_gradient) = config.sink_gradient() {
            if config.sink().is_none() {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward sink_gradient requires sink".into(),
                });
            }
            let expected_sink_gradient_shape =
                Shape::contiguous([1, tensors.q.shape.dimensions()[1], 1, 1])?;
            self.validate_tensor_data_type(
                sink_gradient,
                DataType::F32,
                "sdpa backward sink_gradient data type",
            )?;
            self.validate_tensor_dimensions(
                sink_gradient,
                expected_sink_gradient_shape.dimensions(),
                "sdpa backward sink_gradient shape",
            )?;
            self.validate_tensor_strides(
                sink_gradient,
                expected_sink_gradient_shape.strides(),
                "sdpa backward sink_gradient strides",
            )?;
        }

        Ok(())
    }

    #[cfg(test)]
    fn validate_sdpa_backward_ragged_config(
        &self,
        config: &DirectSdpaBackwardConfig,
        tensors: &SdpaBackwardTensorSpecs,
        support: support::SdpaBackwardSupport,
    ) -> Result<()> {
        let is_ragged = tensors.is_ragged();
        if let Some(max_total_sequence_length_query) = config.max_total_sequence_length_query()
            && max_total_sequence_length_query <= 0
        {
            check_range!("sdpa backward max_total_sequence_length_query", false)?;
        }
        if let Some(max_total_sequence_length_key_value) =
            config.max_total_sequence_length_key_value()
            && max_total_sequence_length_key_value <= 0
        {
            check_range!("sdpa backward max_total_sequence_length_key_value", false)?;
        }
        if (config.max_total_sequence_length_query().is_some()
            || config.max_total_sequence_length_key_value().is_some())
            && !is_ragged
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa backward ragged max_total_seq_len".into(),
            });
        }
        support.require_ragged_gqa(
            is_ragged,
            tensors.q.shape.dimensions()[1],
            tensors.k.shape.dimensions()[1],
            tensors.v.shape.dimensions()[1],
        )?;
        support.require_ragged_stats(tensors.stats.ragged_offset.is_some())?;

        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn sdpa_backward_direct_infer(
        &mut self,
        inputs: SdpaBackwardInputs,
        config: DirectSdpaBackwardConfig,
    ) -> Result<SdpaBackwardOutputs> {
        let SdpaBackwardInputs {
            query: q,
            key: k,
            value: v,
            output: _,
            output_gradient: _,
            stats: _,
            scale: _,
        } = inputs;
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();

        let d_q = self.tensor(TensorSpec::new(q_tensor.data_type, q_tensor.shape.clone()));
        let d_k = self.tensor(TensorSpec::new(k_tensor.data_type, k_tensor.shape.clone()));
        let d_v = self.tensor(TensorSpec::new(v_tensor.data_type, v_tensor.shape.clone()));
        if let Err(error) = self.sdpa_backward_direct(
            inputs,
            SdpaBackwardGradientTensors {
                query_gradient: d_q,
                key_gradient: d_k,
                value_gradient: d_v,
                bias_gradient: None,
                rng_dump: None,
                sink_token_gradient: None,
            },
            config,
        ) {
            self.tensors.remove(&d_q);
            self.tensors.remove(&d_k);
            self.tensors.remove(&d_v);
            return Err(error);
        }
        Ok(SdpaBackwardOutputs {
            query_gradient: d_q,
            key_gradient: d_k,
            value_gradient: d_v,
            bias_gradient: None,
        })
    }
}
