use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::{
            SdpaAuxOutputs, SdpaInputs, SdpaOutputTensors, SdpaOutputs,
            support::{sdpa_mask_scalar, sdpa_scalar_tensor},
        },
        graph::Graph,
        infer::*,
        operation::*,
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    tensor::{Shape, TensorId, TensorSpec},
    utility::check_range,
    version,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::frontend::composite::sdpa) struct PreparedAttentionKv {
    pub key: TensorId,
    pub value: TensorId,
}

impl PreparedAttentionKv {
    pub fn new(key: TensorId, value: TensorId) -> Self {
        Self { key, value }
    }
}

impl Graph {
    pub(in crate::frontend::composite::sdpa) fn transpose_last_two(
        &mut self,
        input: TensorId,
    ) -> Result<TensorId> {
        let input_tensor = self.tensor_config(input)?.clone();
        if input_tensor.is_virtual {
            return self.transpose_last_two_reshape(input);
        }
        let mut dimensions = input_tensor.shape.dimensions().to_vec();
        let mut strides = input_tensor.shape.strides().to_vec();
        let rank = dimensions.len();
        if rank < 2 {
            return Err(Error::InvalidDataShape);
        }
        dimensions.swap(rank - 2, rank - 1);
        strides.swap(rank - 2, rank - 1);
        let output = self.alias_tensor(
            input,
            TensorSpec::new(
                input_tensor.data_type,
                Shape::contiguous(dimensions)?.with_strides(strides)?,
            ),
            0,
        )?;
        Ok(output)
    }

    pub(in crate::frontend::composite::sdpa) fn transpose_last_two_reshape(
        &mut self,
        input: TensorId,
    ) -> Result<TensorId> {
        let input_tensor = self.tensor_config(input)?.clone();
        let mut dimensions = input_tensor.shape.dimensions().to_vec();
        let mut strides = input_tensor.shape.strides().to_vec();
        let rank = dimensions.len();
        if rank < 2 {
            return Err(Error::InvalidDataShape);
        }
        dimensions.swap(rank - 2, rank - 1);
        strides.swap(rank - 2, rank - 1);
        let output = self.tensor(
            TensorSpec::new(
                input_tensor.data_type,
                Shape::contiguous(dimensions)?.with_strides(strides)?,
            )
            .virtual_tensor(),
        );
        self.reshape(input, output)?;
        Ok(output)
    }

    pub(in crate::frontend::composite::sdpa) fn prepare_attention_kv(
        &mut self,
        k: TensorId,
        v: TensorId,
        config: AttentionConfig,
    ) -> Result<PreparedAttentionKv> {
        let max_sequence_length_key_value = config.max_sequence_length_key_value();
        let k_loaded = if let Some(cache) = config.paged_k() {
            self.paged_cache_load_attention_infer(
                k,
                cache.sequence,
                cache.page_table,
                true,
                max_sequence_length_key_value,
            )?
        } else {
            self.transpose_last_two(k)?
        };
        let v_loaded = if let Some(cache) = config.paged_v() {
            self.paged_cache_load_attention_infer(
                v,
                cache.sequence,
                cache.page_table,
                false,
                max_sequence_length_key_value,
            )?
        } else {
            v
        };

        Ok(PreparedAttentionKv::new(k_loaded, v_loaded))
    }

    pub(in crate::frontend::composite::sdpa) fn paged_cache_load_attention_infer(
        &mut self,
        container: TensorId,
        sequence: TensorId,
        page_table: TensorId,
        transposed: bool,
        max_sequence_length_key_value: Option<i64>,
    ) -> Result<TensorId> {
        let loaded = self.paged_cache_load_infer(container, sequence, page_table, transposed)?;
        if let Some(max_sequence_length_key_value) = max_sequence_length_key_value {
            let loaded_tensor = self.tensor_config(loaded)?.clone();
            let sequence_axis = if transposed { 3 } else { 2 };
            if max_sequence_length_key_value > loaded_tensor.shape.dimensions()[sequence_axis] {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa max_sequence_length_key_value".into(),
                });
            }
            let mut slices = loaded_tensor
                .shape
                .dimensions()
                .iter()
                .copied()
                .map(|dimension| 0..dimension)
                .collect::<Vec<_>>();
            slices[sequence_axis].end = max_sequence_length_key_value;
            self.slice_infer(loaded, &slices)
        } else {
            Ok(loaded)
        }
    }

    pub(in crate::frontend::composite::sdpa) fn diagonal_band_mask_composite(
        &mut self,
        scores: TensorId,
        lower_bandwidth: i64,
        upper_bandwidth: Option<i64>,
    ) -> Result<TensorId> {
        let scores_tensor = self.tensor_config(scores)?.clone();
        let index_shape = scores_tensor.shape.clone();

        let compare =
            self.diagonal_band_compare(scores, &index_shape, lower_bandwidth, upper_bandwidth)?;

        let negative_inf = self.tensor(sdpa_mask_scalar(
            scores_tensor.data_type,
            f32::NEG_INFINITY,
        )?);
        let zero = self.tensor(sdpa_mask_scalar(scores_tensor.data_type, 0.0)?);
        let mask =
            self.tensor(TensorSpec::new(scores_tensor.data_type, index_shape).virtual_tensor());
        self.pointwise(PointwiseOperation::Ternary {
            mode: PointwiseMode::BinarySelect,
            x: compare,
            b: negative_inf,
            t: zero,
            output: mask,
            compute_type: scores_tensor.data_type,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });

        Ok(mask)
    }

    pub(in crate::frontend::composite::sdpa) fn apply_diagonal_band_mask(
        &mut self,
        scores: TensorId,
        lower_bandwidth: i64,
        upper_bandwidth: Option<i64>,
    ) -> Result<TensorId> {
        let scores_tensor = self.tensor_config(scores)?.clone();
        let compare = self.diagonal_band_compare(
            scores,
            &scores_tensor.shape,
            lower_bandwidth,
            upper_bandwidth,
        )?;

        let negative_inf = self.tensor(sdpa_mask_scalar(
            scores_tensor.data_type,
            f32::NEG_INFINITY,
        )?);
        let masked = self.tensor(
            TensorSpec::new(scores_tensor.data_type, scores_tensor.shape.clone()).virtual_tensor(),
        );
        self.pointwise(PointwiseOperation::Ternary {
            mode: PointwiseMode::BinarySelect,
            x: scores,
            b: negative_inf,
            t: compare,
            output: masked,
            compute_type: scores_tensor.data_type,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });

        Ok(masked)
    }

    pub(in crate::frontend::composite::sdpa) fn diagonal_band_compare(
        &mut self,
        scores: TensorId,
        index_shape: &Shape,
        lower_bandwidth: i64,
        upper_bandwidth: Option<i64>,
    ) -> Result<TensorId> {
        let row = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        let col = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: row,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            axis: Some(2),
        });
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: col,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            axis: Some(3),
        });

        let left_compare_rhs = if lower_bandwidth == 0 {
            col
        } else {
            let bandwidth = self.tensor(TensorSpec::scalar_i32(lower_bandwidth as i32)?);
            self.pointwise_binary_infer(col, bandwidth, PointwiseMode::Add, DataType::I32)?
        };

        let left_compare =
            self.tensor(TensorSpec::new(DataType::Boolean, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::CmpGe,
            lhs: left_compare_rhs,
            rhs: row,
            output: left_compare,
            compute_type: DataType::Boolean,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });

        if let Some(upper_bandwidth) = upper_bandwidth {
            if lower_bandwidth == 0 && upper_bandwidth == 0 {
                return Ok(left_compare);
            }
            let right_compare_rhs = if upper_bandwidth == 0 {
                row
            } else {
                let bandwidth = self.tensor(TensorSpec::scalar_i32(upper_bandwidth as i32)?);
                self.pointwise_binary_infer(row, bandwidth, PointwiseMode::Add, DataType::I32)?
            };

            let right_compare = self
                .tensor(TensorSpec::new(DataType::Boolean, index_shape.clone()).virtual_tensor());
            self.pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::CmpGt,
                lhs: col,
                rhs: right_compare_rhs,
                output: right_compare,
                compute_type: DataType::Boolean,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });

            self.pointwise_binary_infer(
                left_compare,
                right_compare,
                PointwiseMode::LogicalOr,
                DataType::Boolean,
            )
        } else {
            Ok(left_compare)
        }
    }

    pub(in crate::frontend::composite::sdpa) fn padding_mask(
        &mut self,
        scores: TensorId,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Result<TensorId> {
        let scores_tensor = self.tensor_config(scores)?.clone();
        let seq_len_q_tensor = self.tensor_config(sequence_length_query)?.clone();
        let seq_len_kv_tensor = self.tensor_config(sequence_length_key_value)?.clone();

        if seq_len_q_tensor.data_type != DataType::I32
            || seq_len_kv_tensor.data_type != DataType::I32
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa padding mask seq len data type".into(),
            });
        }

        let index_shape = scores_tensor.shape.clone();
        let row = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        let col = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: row,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            axis: Some(2),
        });
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: col,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: Some(3),
        });

        let row_mask =
            self.tensor(TensorSpec::new(DataType::U8, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::CmpGe,
            lhs: row,
            rhs: sequence_length_query,
            output: row_mask,
            compute_type: DataType::U8,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });

        let col_mask =
            self.tensor(TensorSpec::new(DataType::U8, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::CmpGe,
            lhs: col,
            rhs: sequence_length_key_value,
            output: col_mask,
            compute_type: DataType::U8,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });

        let compare = self.pointwise_binary_virtual_infer_as(
            row_mask,
            col_mask,
            PointwiseMode::LogicalOr,
            DataType::U8,
            DataType::U8,
        )?;

        let negative_inf = self.tensor(sdpa_mask_scalar(
            scores_tensor.data_type,
            f32::NEG_INFINITY,
        )?);
        let zero = self.tensor(sdpa_mask_scalar(scores_tensor.data_type, 0.0)?);
        let mask =
            self.tensor(TensorSpec::new(scores_tensor.data_type, index_shape).virtual_tensor());
        self.pointwise(PointwiseOperation::Ternary {
            mode: PointwiseMode::BinarySelect,
            x: compare,
            b: negative_inf,
            t: zero,
            output: mask,
            compute_type: scores_tensor.data_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });

        Ok(mask)
    }

    pub(in crate::frontend::composite::sdpa) fn padding_mask_select(
        &mut self,
        scores: TensorId,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Result<TensorId> {
        let scores_tensor = self.tensor_config(scores)?.clone();
        let seq_len_q_tensor = self.tensor_config(sequence_length_query)?.clone();
        let seq_len_kv_tensor = self.tensor_config(sequence_length_key_value)?.clone();

        if seq_len_q_tensor.data_type != DataType::I32
            || seq_len_kv_tensor.data_type != DataType::I32
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa padding mask seq len data type".into(),
            });
        }

        let index_shape = scores_tensor.shape.clone();
        let row = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        let col = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: row,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            axis: Some(2),
        });
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: col,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: Some(3),
        });

        let row_valid =
            self.tensor(TensorSpec::new(DataType::Boolean, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::CmpLt,
            lhs: row,
            rhs: sequence_length_query,
            output: row_valid,
            compute_type: DataType::Boolean,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });

        let col_valid =
            self.tensor(TensorSpec::new(DataType::Boolean, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::CmpLt,
            lhs: col,
            rhs: sequence_length_key_value,
            output: col_valid,
            compute_type: DataType::Boolean,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });

        let valid = self.pointwise_binary_virtual_infer_as(
            row_valid,
            col_valid,
            PointwiseMode::LogicalAnd,
            DataType::Boolean,
            DataType::Boolean,
        )?;

        let negative_inf = self.tensor(sdpa_mask_scalar(
            scores_tensor.data_type,
            f32::NEG_INFINITY,
        )?);
        let masked =
            self.tensor(TensorSpec::new(scores_tensor.data_type, index_shape).virtual_tensor());
        self.pointwise(PointwiseOperation::Ternary {
            mode: PointwiseMode::BinarySelect,
            x: scores,
            b: negative_inf,
            t: valid,
            output: masked,
            compute_type: scores_tensor.data_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });

        Ok(masked)
    }

    pub(in crate::frontend::composite::sdpa) fn causal_bottom_right_mask(
        &mut self,
        scores: TensorId,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Result<TensorId> {
        let scores_tensor = self.tensor_config(scores)?.clone();
        let seq_len_q_tensor = self.tensor_config(sequence_length_query)?.clone();
        let seq_len_kv_tensor = self.tensor_config(sequence_length_key_value)?.clone();

        if seq_len_q_tensor.data_type != DataType::I32
            || seq_len_kv_tensor.data_type != DataType::I32
        {
            return Err(Error::DescriptorMismatch {
                name: "sdpa bottom right seq len data type".into(),
            });
        }

        let index_shape = scores_tensor.shape.clone();
        let row = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        let col = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: row,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: Some(2),
        });
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: col,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: Some(3),
        });

        let shifted_row = self.pointwise_binary_virtual_infer_as(
            row,
            sequence_length_key_value,
            PointwiseMode::Add,
            DataType::I32,
            DataType::I32,
        )?;
        let shifted_row = self.pointwise_binary_virtual_infer_as(
            shifted_row,
            sequence_length_query,
            PointwiseMode::Sub,
            DataType::I32,
            DataType::I32,
        )?;

        let compare =
            self.tensor(TensorSpec::new(DataType::U8, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Binary {
            mode: PointwiseMode::CmpGt,
            lhs: col,
            rhs: shifted_row,
            output: compare,
            compute_type: DataType::U8,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });

        let negative_inf = self.tensor(sdpa_mask_scalar(
            scores_tensor.data_type,
            f32::NEG_INFINITY,
        )?);
        let zero = self.tensor(sdpa_mask_scalar(scores_tensor.data_type, 0.0)?);
        let mask =
            self.tensor(TensorSpec::new(scores_tensor.data_type, index_shape).virtual_tensor());
        self.pointwise(PointwiseOperation::Ternary {
            mode: PointwiseMode::BinarySelect,
            x: compare,
            b: negative_inf,
            t: zero,
            output: mask,
            compute_type: scores_tensor.data_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });

        Ok(mask)
    }

    pub(in crate::frontend::composite::sdpa) fn alibi_bias(
        &mut self,
        scores: TensorId,
        alibi_slopes: TensorId,
    ) -> Result<TensorId> {
        let scores_tensor = self.tensor_config(scores)?.clone();
        let index_shape = scores_tensor.shape.clone();

        let row = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        let col = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: row,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: Some(2),
        });
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: scores,
            output: col,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: Some(3),
        });

        let relative_position =
            self.pointwise_binary_infer(col, row, PointwiseMode::Sub, DataType::I32)?;

        let relative_position_f =
            self.tensor(TensorSpec::new(scores_tensor.data_type, index_shape));
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::Identity,
            input: relative_position,
            output: relative_position_f,
            compute_type: scores_tensor.data_type,
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: None,
        });

        self.pointwise_binary_infer(
            relative_position_f,
            alibi_slopes,
            PointwiseMode::Mul,
            scores_tensor.data_type,
        )
    }

    pub fn sdpa_causal_mask_infer(&mut self, q: TensorId, k: TensorId) -> Result<TensorId> {
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let scores = self.tensor(TensorSpec::new(
            q_tensor.data_type,
            infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?,
        ));

        self.diagonal_band_mask_composite(scores, 0, Some(0))
    }

    pub fn sdpa_sliding_window_mask_infer(
        &mut self,
        q: TensorId,
        k: TensorId,
        left_window: i64,
        right_window: i64,
    ) -> Result<TensorId> {
        check_range!("left_window", left_window >= 0)?;
        check_range!("right_window", right_window >= 0)?;

        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let scores = self.tensor(TensorSpec::new(
            q_tensor.data_type,
            infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?,
        ));

        self.diagonal_band_mask_composite(scores, left_window, Some(right_window))
    }

    /// Adds scaled dot product attention with explicit output and optional stats.
    ///
    /// SDPA computes `softmax(QK^T / sqrt(d))V` and may generate stats for
    /// backward training computation.
    pub fn sdpa(
        &mut self,
        inputs: SdpaInputs,
        outputs: SdpaOutputTensors,
        config: &SdpaConfig,
    ) -> Result<()> {
        let SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        } = inputs;
        if config.has_causal_mask() || config.has_causal_bottom_right() {
            return self.sdpa_with_aux(inputs, outputs, config);
        }
        self.sdpa_direct(
            q,
            k,
            v,
            scale,
            outputs.output,
            outputs.stats,
            outputs.logit_max,
            outputs.score_sum_exp,
            None,
            None,
            None,
            SdpaScoreModifiers::new(),
            None,
            config,
            DirectSdpaForwardConfig::new(),
            None,
        )
    }

    /// Adds SDPA with explicit auxiliary outputs for stats, logit max, and sum-exp.
    pub fn sdpa_with_aux(
        &mut self,
        inputs: SdpaInputs,
        outputs: SdpaOutputTensors,
        config: &SdpaConfig,
    ) -> Result<()> {
        let SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        } = inputs;
        if config.has_stats() != outputs.stats.is_some() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa stats".into(),
            });
        }
        if config.has_logit_max() != outputs.logit_max.is_some() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa logit max".into(),
            });
        }
        if config.has_score_sum_exp() != outputs.score_sum_exp.is_some() {
            return Err(Error::DescriptorMismatch {
                name: "sdpa score sum exp".into(),
            });
        }

        let modifiers = if config.has_causal_bottom_right() {
            SdpaScoreModifiers::new().with_causal_bottom_right(
                config
                    .sequence_length_query()
                    .ok_or(Error::DescriptorMismatch {
                        name: "sdpa bottom right seq len".into(),
                    })?,
                config
                    .sequence_length_key_value()
                    .ok_or(Error::DescriptorMismatch {
                        name: "sdpa bottom right seq len".into(),
                    })?,
            )
        } else if config.has_causal_mask() {
            let causal_mask = self.sdpa_causal_mask_infer(q, k)?;
            SdpaScoreModifiers::new().with_additive_mask(causal_mask)
        } else {
            SdpaScoreModifiers::new()
        };
        let scale = self.sdpa_effective_scale(scale, config.attention_scale())?;

        self.sdpa_with_modifiers(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale,
            },
            outputs,
            modifiers,
            SoftmaxConfig::new(self.sdpa_aux_data_type()),
        )
    }

    /// Adds SDPA and creates output and optional stats tensors.
    pub fn sdpa_infer(&mut self, inputs: SdpaInputs, config: &SdpaConfig) -> Result<SdpaOutputs> {
        let outputs = self.sdpa_infer_with_aux(inputs, config)?;
        Ok(outputs.into())
    }

    /// Adds SDPA and creates output plus requested auxiliary tensors.
    pub fn sdpa_infer_with_aux(
        &mut self,
        inputs: SdpaInputs,
        config: &SdpaConfig,
    ) -> Result<SdpaAuxOutputs> {
        let SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: _,
        } = inputs;
        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &v_tensor.shape)?;
        let output = self.tensor(TensorSpec::new(q_tensor.data_type, output_shape));
        let aux_data_type = self.sdpa_aux_data_type();

        let stats = if config.has_stats() {
            let stats_shape = reduce_last_axis(&q_tensor.shape)?;
            Some(self.tensor(TensorSpec::new(aux_data_type, stats_shape)))
        } else {
            None
        };
        let logit_max = if config.has_logit_max() {
            let stats_shape = reduce_last_axis(&q_tensor.shape)?;
            Some(self.tensor(TensorSpec::new(aux_data_type, stats_shape)))
        } else {
            None
        };
        let score_sum_exp = if config.has_score_sum_exp() {
            let stats_shape = reduce_last_axis(&q_tensor.shape)?;
            Some(self.tensor(TensorSpec::new(aux_data_type, stats_shape)))
        } else {
            None
        };

        self.sdpa_with_aux(
            inputs,
            SdpaOutputTensors {
                output,
                stats,
                logit_max,
                score_sum_exp,
            },
            config,
        )?;

        Ok(SdpaAuxOutputs::with_aux(
            output,
            stats,
            logit_max,
            score_sum_exp,
            None,
        ))
    }

    pub fn sdpa_auto_infer(
        &mut self,
        inputs: SdpaInputs,
        config: AttentionConfig,
    ) -> Result<SdpaOutputs> {
        let outputs = self.sdpa_auto_infer_with_aux(inputs, config)?;
        Ok(outputs.into())
    }

    pub fn sdpa_auto_infer_with_aux(
        &mut self,
        inputs: SdpaInputs,
        config: AttentionConfig,
    ) -> Result<SdpaAuxOutputs> {
        let SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        } = inputs;
        let config = self.normalize_attention_config_for_forward_version(version()?.raw(), config);
        self.validate_attention_config(q, k, v, config.clone())?;
        let implementation = self.resolve_attention_implementation(q, k, v, config.clone())?;

        if implementation == AttentionImplementation::Unified {
            return self.sdpa_unified_infer(q, k, v, scale, config);
        }

        let scale = self.sdpa_effective_scale(scale, config.fused().attention_scale())?;

        let prepared_kv = self.prepare_attention_kv(k, v, config.clone())?;
        let k = prepared_kv.key;
        let v = prepared_kv.value;

        let q_tensor = self.tensor_config(q)?.clone();
        let k_tensor = self.tensor_config(k)?.clone();
        let v_tensor = self.tensor_config(v)?.clone();
        let scores_shape = infer_sdpa_scores_shape(&q_tensor.shape, &k_tensor.shape)?;
        let output_shape = infer_sdpa_output_shape(&scores_shape, &v_tensor.shape)?;
        let output = self.tensor(TensorSpec::new(q_tensor.data_type, output_shape));
        let aux_data_type = self.sdpa_aux_data_type();

        let stats = if config.fused().has_stats() {
            let stats_shape = reduce_last_axis(&scores_shape)?;
            Some(self.tensor(TensorSpec::new(aux_data_type, stats_shape)))
        } else {
            None
        };
        let logit_max = if config.fused().has_logit_max() {
            let aux_shape = reduce_last_axis(&scores_shape)?;
            Some(self.tensor(TensorSpec::new(aux_data_type, aux_shape)))
        } else {
            None
        };
        let score_sum_exp = if config.fused().has_score_sum_exp() {
            let aux_shape = reduce_last_axis(&scores_shape)?;
            Some(self.tensor(TensorSpec::new(aux_data_type, aux_shape)))
        } else {
            None
        };
        let rng_dump = if let Some(dropout) = config.dropout() {
            let rng_config = self.attention_dropout_random_number_generator_config(
                dropout.probability(),
                dropout.seed_source(),
                config.dropout_offset(),
            )?;
            let rng_dump = self.tensor(TensorSpec::new(DataType::F32, scores_shape.clone()));
            self.random_number_generator(rng_dump, rng_config)?;
            Some(rng_dump)
        } else {
            None
        };
        let modifiers = if let Some(dropout) = config.dropout() {
            let dropout_scale = self.tensor(sdpa_scalar_tensor(
                q_tensor.data_type,
                1.0 / (1.0 - dropout.probability()),
            )?);
            let rng_dump = rng_dump.ok_or_else(|| Error::DescriptorMismatch {
                name: "sdpa rng dump".into(),
            })?;
            config.modifiers().with_dropout(rng_dump, dropout_scale)
        } else {
            *config.modifiers()
        };

        self.sdpa_with_modifiers_internal(
            q,
            k,
            v,
            scale,
            output,
            stats,
            logit_max,
            score_sum_exp,
            modifiers,
            config.score_subgraph(),
            *config.softmax(),
        )?;

        Ok(SdpaAuxOutputs::with_aux(
            output,
            stats,
            logit_max,
            score_sum_exp,
            rng_dump,
        ))
    }

    pub fn sdpa_bias(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        stats: Option<TensorId>,
        config: SoftmaxConfig,
    ) -> Result<()> {
        self.sdpa_with_modifiers(
            SdpaInputs {
                query: q,
                key: k,
                value: v,
                scale,
            },
            SdpaOutputTensors {
                output,
                stats,
                logit_max: None,
                score_sum_exp: None,
            },
            SdpaScoreModifiers::new().with_bias(bias),
            config,
        )
    }
}
