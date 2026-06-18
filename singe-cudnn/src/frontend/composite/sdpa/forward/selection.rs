use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{composite::sdpa::SdpaAuxOutputs, graph::Graph, infer::*, operation::*, support},
    scalar::ScalarValue,
    tensor::{Shape, TensorId, TensorSpec},
    utility::check_range,
    version,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::frontend::composite::sdpa) struct DirectSdpaKvShapes {
    pub key: Shape,
    pub value: Shape,
}

impl DirectSdpaKvShapes {
    pub fn new(key: Shape, value: Shape) -> Self {
        Self { key, value }
    }
}

impl Graph {
    pub(in crate::frontend::composite::sdpa) fn can_lower_attention_as_unified_for_version(
        &self,
        cudnn_version: u64,
        config: AttentionConfig,
    ) -> bool {
        let config = self.normalize_attention_config_for_forward_version(cudnn_version, config);
        support::UnifiedSdpaForwardSupport::new(cudnn_version).supports_config(&config)
    }

    pub(in crate::frontend::composite::sdpa) fn can_lower_attention_as_unified_with_tensors_for_version(
        &self,
        cudnn_version: u64,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: AttentionConfig,
    ) -> Result<bool> {
        if self.dynamic_shape_enabled() || self.override_shape_enabled() {
            return Ok(false);
        }

        if !self.can_lower_attention_as_unified_for_version(cudnn_version, config.clone()) {
            return Ok(false);
        }

        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let q_data_type = q_tensor.data_type;
        let k_data_type = k_tensor.data_type;
        let v_data_type = self.tensor_config(v)?.data_type;

        if !matches!(q_data_type, DataType::F16 | DataType::BF16)
            || q_data_type != k_data_type
            || q_data_type != v_data_type
            || config.softmax().compute_type() != DataType::F32
        {
            return Ok(false);
        }

        if let Some(cache) = config.paged_k()
            && self
                .tensor_config(cache.page_table)?
                .ragged_offset
                .is_some()
        {
            return Ok(false);
        }
        if let Some(cache) = config.paged_v()
            && self
                .tensor_config(cache.page_table)?
                .ragged_offset
                .is_some()
        {
            return Ok(false);
        }
        Ok(q_tensor.shape.dimensions()[3] % 8 == 0 && k_tensor.shape.dimensions()[3] % 8 == 0)
    }

    pub fn resolve_attention_implementation(
        &self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: AttentionConfig,
    ) -> Result<AttentionImplementation> {
        self.resolve_attention_implementation_for_version(version()?.raw(), q, k, v, config)
    }

    pub(crate) fn resolve_attention_implementation_for_version(
        &self,
        cudnn_version: u64,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        config: AttentionConfig,
    ) -> Result<AttentionImplementation> {
        let config = self.normalize_attention_config_for_forward_version(cudnn_version, config);
        match config.implementation() {
            AttentionImplementation::Auto => {
                if self.can_lower_attention_as_unified_with_tensors_for_version(
                    cudnn_version,
                    q,
                    k,
                    v,
                    config.clone(),
                )? || (self.override_shape_enabled()
                    && !self.dynamic_shape_enabled()
                    && self.can_lower_attention_as_unified_for_version(cudnn_version, config))
                {
                    Ok(AttentionImplementation::Unified)
                } else {
                    Ok(AttentionImplementation::Composite)
                }
            }
            AttentionImplementation::Unified => {
                if self.can_lower_attention_as_unified_with_tensors_for_version(
                    cudnn_version,
                    q,
                    k,
                    v,
                    config,
                )? {
                    Ok(AttentionImplementation::Unified)
                } else {
                    Err(Error::DescriptorMismatch {
                        name: "sdpa unified implementation".into(),
                    })
                }
            }
            AttentionImplementation::Composite => Ok(AttentionImplementation::Composite),
        }
    }

    pub(in crate::frontend::composite::sdpa) fn infer_direct_sdpa_kv_shapes(
        &self,
        k: TensorId,
        v: TensorId,
        direct_config: &DirectSdpaForwardConfig,
        max_sequence_length_key_value: Option<i64>,
    ) -> Result<DirectSdpaKvShapes> {
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();

        let mut k_shape = if let Some(page_table_k) = direct_config.page_table_k() {
            let sequence_length_key_value =
                direct_config
                    .sequence_length_key_value()
                    .ok_or(Error::DescriptorMismatch {
                        name: "sdpa paged seq lens".into(),
                    })?;
            let seq_len_kv_tensor = self.tensor_config(sequence_length_key_value)?.clone();
            let page_table_k_tensor = self.tensor_config(page_table_k)?.clone();
            infer_paged_cache_output(
                &k_tensor.shape,
                &seq_len_kv_tensor.shape,
                &page_table_k_tensor.shape,
                true,
            )?
        } else {
            k_tensor.shape.clone()
        };

        let mut v_shape = if let Some(page_table_v) = direct_config.page_table_v() {
            let sequence_length_key_value =
                direct_config
                    .sequence_length_key_value()
                    .ok_or(Error::DescriptorMismatch {
                        name: "sdpa paged seq lens".into(),
                    })?;
            let seq_len_kv_tensor = self.tensor_config(sequence_length_key_value)?.clone();
            let page_table_v_tensor = self.tensor_config(page_table_v)?.clone();
            infer_paged_cache_output(
                &v_tensor.shape,
                &seq_len_kv_tensor.shape,
                &page_table_v_tensor.shape,
                false,
            )?
        } else {
            v_tensor.shape.clone()
        };

        if let Some(max_sequence_length_key_value) = max_sequence_length_key_value {
            if let Some(page_table_k) = direct_config.page_table_k() {
                let _ = page_table_k;
                let k_dims = k_shape.dimensions();
                k_shape = infer_slice_output(
                    &k_shape,
                    &[
                        (0, k_dims[0]),
                        (0, k_dims[1]),
                        (0, k_dims[2]),
                        (0, max_sequence_length_key_value),
                    ],
                )?;
            }
            if let Some(page_table_v) = direct_config.page_table_v() {
                let _ = page_table_v;
                let v_dims = v_shape.dimensions();
                v_shape = infer_slice_output(
                    &v_shape,
                    &[
                        (0, v_dims[0]),
                        (0, v_dims[1]),
                        (0, max_sequence_length_key_value),
                        (0, v_dims[3]),
                    ],
                )?;
            }
        }

        Ok(DirectSdpaKvShapes::new(k_shape, v_shape))
    }

    pub(in crate::frontend::composite::sdpa) fn sdpa_unified_infer(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        scale: TensorId,
        config: AttentionConfig,
    ) -> Result<SdpaAuxOutputs> {
        let config = self.normalize_attention_config_for_forward_version(version()?.raw(), config);
        let fused = config.fused();
        let mut direct_config = DirectSdpaForwardConfig::new();
        if config.unfuse_fma() {
            direct_config = direct_config.with_unfused_fma();
        }
        if let (Some(sequence_length_query), Some(sequence_length_key_value)) = (
            config.modifiers().sequence_length_query,
            config.modifiers().sequence_length_key_value,
        ) {
            direct_config = direct_config
                .with_sequence_lengths(sequence_length_query, sequence_length_key_value);
            if config.modifiers().padding_mask {
                direct_config = direct_config.with_padding_mask();
            }
        }
        if let Some(cache) = config.paged_k() {
            direct_config = direct_config.with_page_table_k(cache.page_table);
        }
        if let Some(cache) = config.paged_v() {
            direct_config = direct_config.with_page_table_v(cache.page_table);
        }
        if let Some(block_mask) = config.block_mask() {
            direct_config = direct_config.with_block_mask(block_mask);
        }
        let q_tensor = self.tensor_config(q)?.clone();
        let kv_shapes = self.infer_direct_sdpa_kv_shapes(
            k,
            v,
            &direct_config,
            config.max_sequence_length_key_value(),
        )?;
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &kv_shapes.key)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &kv_shapes.value)?;
        let output = self.tensor(TensorSpec::new(q_tensor.data_type, output_shape));
        let aux_data_type = self.sdpa_aux_data_type();
        let stats = if fused.has_stats() {
            let stats_shape = reduce_last_axis(&scores_shape)?;
            Some(self.tensor(TensorSpec::new(aux_data_type, stats_shape)))
        } else {
            None
        };
        let logit_max = if fused.has_logit_max() {
            let aux_shape = reduce_last_axis(&scores_shape)?;
            Some(self.tensor(TensorSpec::new(aux_data_type, aux_shape)))
        } else {
            None
        };
        let score_sum_exp = if fused.has_score_sum_exp() {
            let aux_shape = reduce_last_axis(&scores_shape)?;
            Some(self.tensor(TensorSpec::new(aux_data_type, aux_shape)))
        } else {
            None
        };
        let sink = config.modifiers().sink_token;
        let (softmax_p, softmax_s) = if version()? >= 92100
            && (stats.is_some() || logit_max.is_some() || score_sum_exp.is_some() || sink.is_some())
        {
            let softmax_shape = Shape::contiguous(scores_shape.dimensions().to_vec())?;
            (
                Some(self.tensor(
                    TensorSpec::new(aux_data_type, softmax_shape.clone()).virtual_tensor(),
                )),
                Some(self.tensor(TensorSpec::new(aux_data_type, softmax_shape).virtual_tensor())),
            )
        } else {
            (None, None)
        };
        let rng_dump = if config.dropout().is_some() {
            Some(self.tensor(TensorSpec::new(DataType::F32, scores_shape.clone())))
        } else {
            None
        };
        if let Some(dropout) = config.dropout() {
            let seed = if let Some(seed) = dropout.seed_tensor() {
                seed
            } else {
                let seed_value = dropout.seed().ok_or(Error::DescriptorMismatch {
                    name: "sdpa dropout seed".into(),
                })?;
                self.tensor(
                    TensorSpec::new(DataType::I64, Shape::contiguous([1, 1, 1, 1])?)
                        .with_scalar_value(ScalarValue::I64(seed_value))?,
                )
            };
            direct_config = direct_config.with_dropout(dropout).with_dropout_seed(seed);
            if let Some(rng_dump) = rng_dump {
                direct_config = direct_config.with_random_number_generator_dump(rng_dump);
            }
        }
        self.sdpa_direct(
            q,
            k,
            v,
            scale,
            output,
            stats,
            logit_max,
            score_sum_exp,
            softmax_p,
            softmax_s,
            sink,
            *config.modifiers(),
            config.score_subgraph().cloned(),
            fused,
            direct_config,
            config.max_sequence_length_key_value(),
        )?;
        Ok(SdpaAuxOutputs::with_aux(
            output,
            stats,
            logit_max,
            score_sum_exp,
            rng_dump,
        ))
    }

    pub(in crate::frontend::composite::sdpa) fn sdpa_direct(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        scale: TensorId,
        o: TensorId,
        stats: Option<TensorId>,
        logit_max: Option<TensorId>,
        score_sum_exp: Option<TensorId>,
        softmax_p: Option<TensorId>,
        softmax_s: Option<TensorId>,
        sink: Option<TensorId>,
        score_modifiers: SdpaScoreModifiers,
        score_subgraph: Option<SdpaScoreSubgraph>,
        config: &SdpaConfig,
        direct_config: DirectSdpaForwardConfig,
        max_sequence_length_key_value: Option<i64>,
    ) -> Result<()> {
        let scale = self.sdpa_effective_scale(scale, config.attention_scale())?;

        let q_tensor = self.tensor_config(q)?.clone();
        let o_tensor = self.tensor_config(o)?.clone();
        let scale_tensor = self.tensor_config(scale)?.clone();
        let kv_shapes =
            self.infer_direct_sdpa_kv_shapes(k, v, &direct_config, max_sequence_length_key_value)?;

        if q_tensor.data_type != o_tensor.data_type {
            return Err(Error::DescriptorMismatch {
                name: "sdpa data type".into(),
            });
        }
        self.validate_sdpa_tensor_layout(q, "sdpa q layout")?;
        self.validate_sdpa_k_layout(k)?;
        self.validate_sdpa_k_layout(v)?;
        self.validate_sdpa_tensor_layout(o, "sdpa output layout")?;
        if q_tensor.shape.dimensions()[1] % kv_shapes.key.dimensions()[1] != 0
            || q_tensor.shape.dimensions()[1] % kv_shapes.value.dimensions()[1] != 0
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa attention heads".into(),
            });
        }
        self.validate_sdpa_direct_ragged_support_for_version(
            version()?.raw(),
            q,
            k,
            v,
            o,
            &direct_config,
            score_subgraph.is_some(),
        )?;
        if scale_tensor.shape.element_count()? != 1 {
            return Err(Error::DescriptorMismatch {
                name: "sdpa scale".into(),
            });
        }
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &kv_shapes.key)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &kv_shapes.value)?;
        let aux_data_type = self.sdpa_aux_data_type();
        let expected_softmax_shape = Shape::contiguous(scores_shape.dimensions().to_vec())?;
        if o_tensor.shape.dimensions() != output_shape.dimensions() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa output shape".into(),
            });
        }
        if let Some(stats) = stats {
            let stats_tensor = self.tensor_config(stats)?.clone();
            let expected_stats_shape = reduce_last_axis(&scores_shape)?;
            if stats_tensor.shape.dimensions() != expected_stats_shape.dimensions() {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa stats shape".into(),
                });
            }
            if stats_tensor.data_type != aux_data_type {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa stats data type".into(),
                });
            }
        } else if config.has_stats() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa stats".into(),
            });
        }
        if let Some(logit_max) = logit_max {
            let logit_max_tensor = self.tensor_config(logit_max)?.clone();
            let expected_shape = reduce_last_axis(&scores_shape)?;
            if logit_max_tensor.shape.dimensions() != expected_shape.dimensions() {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa logit max shape".into(),
                });
            }
            if logit_max_tensor.data_type != aux_data_type {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa logit max data type".into(),
                });
            }
        } else if config.has_logit_max() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa logit max".into(),
            });
        }
        if let Some(score_sum_exp) = score_sum_exp {
            let score_sum_exp_tensor = self.tensor_config(score_sum_exp)?.clone();
            let expected_shape = reduce_last_axis(&scores_shape)?;
            if score_sum_exp_tensor.shape.dimensions() != expected_shape.dimensions() {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa score sum exp shape".into(),
                });
            }
            if score_sum_exp_tensor.data_type != aux_data_type {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa score sum exp data type".into(),
                });
            }
        } else if config.has_score_sum_exp() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa score sum exp".into(),
            });
        }
        if let Some(dropout) = direct_config.dropout() {
            check_range!(
                "sdpa dropout probability",
                (0.0..1.0).contains(&dropout.probability())
            )?;
            if let Some(dropout_seed) = direct_config.dropout_seed() {
                let dropout_seed_tensor = self.tensor_config(dropout_seed)?.clone();
                let expected_scalar_shape = Shape::contiguous([1, 1, 1, 1])?;
                if !matches!(dropout_seed_tensor.data_type, DataType::I32 | DataType::I64)
                    || dropout_seed_tensor.shape.dimensions() != expected_scalar_shape.dimensions()
                {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa dropout seed".into(),
                    });
                }
            }
            if let Some(dropout_offset) = dropout.offset() {
                let dropout_offset_tensor = self.tensor_config(dropout_offset)?.clone();
                let expected_scalar_shape = Shape::contiguous([1, 1, 1, 1])?;
                if !matches!(
                    dropout_offset_tensor.data_type,
                    DataType::I32 | DataType::I64
                ) || dropout_offset_tensor.shape.dimensions()
                    != expected_scalar_shape.dimensions()
                {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa dropout offset".into(),
                    });
                }
            }
            if let Some(rng_dump) = direct_config.random_number_generator_dump() {
                let rng_dump_tensor = self.tensor_config(rng_dump)?.clone();
                if rng_dump_tensor.data_type != DataType::F32
                    || rng_dump_tensor.shape.dimensions() != scores_shape.dimensions()
                {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa rng dump".into(),
                    });
                }
            }
        } else if direct_config.dropout_seed().is_some()
            || direct_config
                .dropout()
                .and_then(|dropout| dropout.offset())
                .is_some()
            || direct_config.random_number_generator_dump().is_some()
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa dropout".into(),
            });
        }
        match (softmax_p, softmax_s) {
            (Some(softmax_p), Some(softmax_s)) => {
                let softmax_p_tensor = self.tensor_config(softmax_p)?.clone();
                let softmax_s_tensor = self.tensor_config(softmax_s)?.clone();
                if softmax_p_tensor.data_type != aux_data_type
                    || softmax_p_tensor.shape.dimensions() != expected_softmax_shape.dimensions()
                    || softmax_p_tensor.shape.strides() != expected_softmax_shape.strides()
                {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa softmax p".into(),
                    });
                }
                if softmax_s_tensor.data_type != aux_data_type
                    || softmax_s_tensor.shape.dimensions() != expected_softmax_shape.dimensions()
                    || softmax_s_tensor.shape.strides() != expected_softmax_shape.strides()
                {
                    return Err(Error::DescriptorMismatch {
                        name: "sdpa softmax s".into(),
                    });
                }
            }
            (None, None) => {}
            _ => {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa unified softmax descriptor".into(),
                });
            }
        }
        if let Some(sink) = sink {
            let sink_tensor = self.tensor_config(sink)?.clone();
            let expected_sink_shape = Shape::contiguous([1, q_tensor.shape.dimensions()[1], 1, 1])?;
            if softmax_p.is_none() || softmax_s.is_none() {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa unified softmax descriptor".into(),
                });
            }
            if sink_tensor.data_type != DataType::F32
                || sink_tensor.shape.dimensions() != expected_sink_shape.dimensions()
                || sink_tensor.shape.strides() != expected_sink_shape.strides()
            {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa sink token shape".into(),
                });
            }
        }

        self.operations.push(Operation::SdpaForward {
            q,
            k,
            v,
            o,
            scale,
            stats,
            logit_max,
            score_sum_exp,
            softmax_p,
            softmax_s,
            sink,
            score_modifiers,
            score_subgraph: Box::new(score_subgraph),
            config: Box::new(direct_config),
        });

        Ok(())
    }
}
