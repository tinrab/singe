use std::fmt;

use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    data_type::DataType,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::{Error, Result},
    execution::OperationGraph,
    pointwise::PointwiseMode,
    tensor::{Tensor, TensorId},
    utility::check_range,
    version,
};
use serde::{Deserialize, Serialize};

use singe_cudnn_sys as sys;

#[derive(Debug)]
pub struct PagedCacheLoadOperation {
    descriptor: BackendDescriptor,
}

impl PagedCacheLoadOperation {
    pub fn create(
        container: &Tensor,
        output: &Tensor,
        sequence: &Tensor,
        page_table: &Tensor,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationPagedCacheLoad)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPagedCacheLoadContainerDesc,
            container.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPagedCacheLoadYDesc,
            output.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPagedCacheLoadSequenceDesc,
            sequence.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationPagedCacheLoadPageTableDesc,
            page_table.descriptor(),
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct BlockScaleQuantizeOperation {
    descriptor: BackendDescriptor,
}

impl BlockScaleQuantizeOperation {
    pub fn create(
        x: &Tensor,
        y: &Tensor,
        scale: &Tensor,
        compute_type: DataType,
        block_size: i32,
    ) -> Result<Self> {
        check_range!("block_size", block_size > 0)?;

        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationBlockScaleQuantize)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleQuantizeXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleQuantizeYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleQuantizeScaleDesc,
            scale.descriptor(),
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationBlockScaleQuantizeMathPrec,
            BackendAttributeType::DataType,
            compute_type,
        )?;
        descriptor.set_attribute_i32(
            BackendAttributeName::OperationBlockScaleQuantizeBlockSize,
            block_size,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct BlockScaleDequantizeOperation {
    descriptor: BackendDescriptor,
}

impl BlockScaleDequantizeOperation {
    pub fn create(
        x: &Tensor,
        scale: &Tensor,
        y: &Tensor,
        compute_type: Option<DataType>,
        block_sizes: &[i32],
        is_negative_scale: bool,
    ) -> Result<Self> {
        check_range!(
            "block_sizes",
            !block_sizes.is_empty() && block_sizes.iter().all(|&block_size| block_size > 0)
        )?;

        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationBlockScaleDequantize)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleDequantizeXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleDequantizeScaleDesc,
            scale.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleDequantizeYDesc,
            y.descriptor(),
        )?;
        if let Some(compute_type) = compute_type {
            descriptor.set_attribute_enum(
                BackendAttributeName::OperationBlockScaleDequantizeMathPrec,
                BackendAttributeType::DataType,
                compute_type,
            )?;
        }
        descriptor.set_attribute_i32_slice(
            BackendAttributeName::OperationBlockScaleDequantizeBlockSize,
            block_sizes,
        )?;
        if version()? >= 91400 {
            descriptor.set_attribute_bool(
                BackendAttributeName::OperationBlockScaleDequantizeNegScale,
                is_negative_scale,
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct ExpandBandMatrixOperation {
    descriptor: BackendDescriptor,
}

impl ExpandBandMatrixOperation {
    pub fn create(
        x: &Tensor,
        y: &Tensor,
        lower_bandwidth: i64,
        upper_bandwidth: i64,
        axis: i64,
        pad_value: f32,
        kv_token_offset: Option<&Tensor>,
        speculative_mask: Option<&Tensor>,
    ) -> Result<Self> {
        check_range!("lower_bandwidth", lower_bandwidth >= 0)?;
        check_range!("upper_bandwidth", lower_bandwidth >= 0)?;

        let rank = x.rank();
        check_range!("axis", axis >= 0 && axis < rank)?;

        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationExpandBandMatrix)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationExpandBandMatrixXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationExpandBandMatrixYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationExpandBandMatrixLowerBandwidth,
            lower_bandwidth,
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationExpandBandMatrixUpperBandwidth,
            upper_bandwidth,
        )?;
        descriptor.set_attribute_i64(BackendAttributeName::OperationExpandBandMatrixAxis, axis)?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationExpandBandMatrixPadValue,
            pad_value,
        )?;
        if let Some(kv_token_offset) = kv_token_offset {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationExpandBandMatrixKvTokenOffsetDesc,
                kv_token_offset.descriptor(),
            )?;
        }
        if let Some(speculative_mask) = speculative_mask {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationExpandBandMatrixSpeculativeMaskDesc,
                speculative_mask.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct ContractBandMatrixOperation {
    descriptor: BackendDescriptor,
}

impl ContractBandMatrixOperation {
    pub fn create(
        x: &Tensor,
        y: &Tensor,
        lower_bandwidth: i64,
        upper_bandwidth: i64,
        axis: i64,
        pad_value: f32,
        max_token_value: i64,
    ) -> Result<Self> {
        if lower_bandwidth < 0 {
            return Err(Error::OutOfRange {
                name: "lower_bandwidth".into(),
            });
        }
        if upper_bandwidth < 0 {
            return Err(Error::OutOfRange {
                name: "upper_bandwidth".into(),
            });
        }
        if max_token_value < 0 {
            return Err(Error::OutOfRange {
                name: "max_token_value".into(),
            });
        }
        let rank = x.rank();
        if axis < 0 || axis >= rank {
            return Err(Error::OutOfRange {
                name: "axis".into(),
            });
        }

        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationContractBandMatrix)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationContractBandMatrixXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationContractBandMatrixYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationContractBandMatrixLowerBandwidth,
            lower_bandwidth,
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationContractBandMatrixUpperBandwidth,
            upper_bandwidth,
        )?;
        descriptor
            .set_attribute_i64(BackendAttributeName::OperationContractBandMatrixAxis, axis)?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationContractBandMatrixPadValue,
            pad_value,
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationContractBandMaxTokenValue,
            max_token_value,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct MatmulDescriptor {
    descriptor: BackendDescriptor,
}

impl MatmulDescriptor {
    pub fn create(compute_type: DataType) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Matmul)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::MatmulCompType,
            BackendAttributeType::DataType,
            compute_type,
        )?;
        descriptor.set_attribute_f64(BackendAttributeName::MatmulPaddingValue, 0.0)?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct MatmulOperation {
    descriptor: BackendDescriptor,
}

impl MatmulOperation {
    pub fn create(matmul: &MatmulDescriptor, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Self> {
        Self::create_with_overrides(matmul, a, b, c, None, None, None)
    }

    pub fn create_with_overrides(
        matmul: &MatmulDescriptor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        m_override: Option<&Tensor>,
        n_override: Option<&Tensor>,
        k_override: Option<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationMatmul)?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationMatmulADesc, a.descriptor())?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationMatmulBDesc, b.descriptor())?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationMatmulCDesc, c.descriptor())?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMatmulDesc,
            matmul.descriptor(),
        )?;
        if let Some(m_override) = m_override {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationMatmulGemmMOverrideDesc,
                m_override.descriptor(),
            )?;
        }
        if let Some(n_override) = n_override {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationMatmulGemmNOverrideDesc,
                n_override.descriptor(),
            )?;
        }
        if let Some(k_override) = k_override {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationMatmulGemmKOverrideDesc,
                k_override.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn create_irregularly_strided_batched(
        matmul: &MatmulDescriptor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        irregularly_strided_batch_count: i64,
    ) -> Result<Self> {
        if irregularly_strided_batch_count <= 0 {
            return Err(Error::OutOfRange {
                name: "irregularly_strided_batch_count".into(),
            });
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationMatmul)?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationMatmulADesc, a.descriptor())?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationMatmulBDesc, b.descriptor())?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationMatmulCDesc, c.descriptor())?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMatmulDesc,
            matmul.descriptor(),
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationMatmulIrregularlyStridedBatchCount,
            irregularly_strided_batch_count,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct SdpaForwardOperation {
    descriptor: BackendDescriptor,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SdpaForwardConfig<'a> {
    pub sequence_length_query: Option<&'a Tensor>,
    pub sequence_length_key_value: Option<&'a Tensor>,
    pub page_table_k: Option<&'a Tensor>,
    pub page_table_v: Option<&'a Tensor>,
    pub block_mask: Option<&'a Tensor>,
    pub softmax: Option<&'a SoftmaxOperation>,
    pub subgraph: Option<&'a OperationGraph>,
    pub subgraph_input_id: Option<TensorId>,
    pub subgraph_output_id: Option<TensorId>,
    pub dropout_seed: Option<&'a Tensor>,
    pub dropout_offset: Option<&'a Tensor>,
    pub dropout_random_number_generator_dump: Option<&'a Tensor>,
    pub dropout_probability: Option<f32>,
    pub unfuse_fma: bool,
}

impl SdpaForwardOperation {
    pub fn create(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &Tensor,
        scale: &Tensor,
        stats: Option<&Tensor>,
        config: SdpaForwardConfig<'_>,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationSdpaForward)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaFwdQDesc,
            q.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaFwdKDesc,
            k.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaFwdVDesc,
            v.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaFwdODesc,
            o.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaFwdScaleDesc,
            scale.descriptor(),
        )?;
        if let Some(softmax) = config.softmax {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdSoftmaxDesc,
                softmax.descriptor(),
            )?;
        } else if let Some(stats) = stats {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdStatsDesc,
                stats.descriptor(),
            )?;
        }
        if let Some(sequence_length_query) = config.sequence_length_query {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdSeqLenQDesc,
                sequence_length_query.descriptor(),
            )?;
        }
        if let Some(sequence_length_key_value) = config.sequence_length_key_value {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdSeqLenKvDesc,
                sequence_length_key_value.descriptor(),
            )?;
        }
        if let Some(page_table_k) = config.page_table_k {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdPageTableKDesc,
                page_table_k.descriptor(),
            )?;
        }
        if let Some(page_table_v) = config.page_table_v {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdPageTableVDesc,
                page_table_v.descriptor(),
            )?;
        }
        if let Some(block_mask) = config.block_mask {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdBlockMaskDesc,
                block_mask.descriptor(),
            )?;
        }
        match (
            config.subgraph,
            config.subgraph_input_id,
            config.subgraph_output_id,
        ) {
            (Some(subgraph), Some(subgraph_input_id), Some(subgraph_output_id)) => {
                descriptor.set_attribute_descriptor(
                    BackendAttributeName::OperationSdpaFwdSubgraph,
                    subgraph.descriptor(),
                )?;
                descriptor.set_attribute_i64(
                    BackendAttributeName::OperationSdpaFwdSubgraphInputId,
                    subgraph_input_id.as_i64(),
                )?;
                descriptor.set_attribute_i64(
                    BackendAttributeName::OperationSdpaFwdSubgraphOutputId,
                    subgraph_output_id.as_i64(),
                )?;
            }
            (None, None, None) => {}
            _ => {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa subgraph descriptor".into(),
                });
            }
        }
        if let Some(dropout_seed) = config.dropout_seed {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdDropoutSeedDesc,
                dropout_seed.descriptor(),
            )?;
        }
        if let Some(dropout_offset) = config.dropout_offset {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdDropoutOffsetDesc,
                dropout_offset.descriptor(),
            )?;
        }
        if let Some(dropout_random_number_generator_dump) =
            config.dropout_random_number_generator_dump
        {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaFwdDropoutRngDumpDesc,
                dropout_random_number_generator_dump.descriptor(),
            )?;
        }
        if let Some(dropout_probability) = config.dropout_probability {
            descriptor.set_attribute_f32(
                BackendAttributeName::OperationSdpaFwdDropoutProbability,
                dropout_probability,
            )?;
        }
        if config.unfuse_fma {
            descriptor.set_attribute_bool(BackendAttributeName::OperationSdpaFwdUnfuseFma, true)?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct SdpaBackwardOperation {
    descriptor: BackendDescriptor,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SdpaBackwardConfig<'a> {
    pub sequence_length_query: Option<&'a Tensor>,
    pub sequence_length_key_value: Option<&'a Tensor>,
    pub subgraph: Option<&'a OperationGraph>,
    pub subgraph_input_id: Option<TensorId>,
    pub subgraph_output_id: Option<TensorId>,
    pub sink: Option<&'a Tensor>,
    pub sink_gradient: Option<&'a Tensor>,
    pub max_total_sequence_length_query: Option<i64>,
    pub max_total_sequence_length_key_value: Option<i64>,
}

impl SdpaBackwardOperation {
    pub fn create(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &Tensor,
        stats: &Tensor,
        scale: &Tensor,
        d_o: &Tensor,
        d_q: &Tensor,
        d_k: &Tensor,
        d_v: &Tensor,
        config: SdpaBackwardConfig<'_>,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationSdpaBackward)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdQDesc,
            q.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdKDesc,
            k.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdVDesc,
            v.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdODesc,
            o.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdStatsDesc,
            stats.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdScaleDesc,
            scale.descriptor(),
        )?;
        if let Some(sequence_length_query) = config.sequence_length_query {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaBwdSeqLenQDesc,
                sequence_length_query.descriptor(),
            )?;
        }
        if let Some(sequence_length_key_value) = config.sequence_length_key_value {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaBwdSeqLenKvDesc,
                sequence_length_key_value.descriptor(),
            )?;
        }
        match (
            config.subgraph,
            config.subgraph_input_id,
            config.subgraph_output_id,
        ) {
            (Some(subgraph), Some(subgraph_input_id), Some(subgraph_output_id)) => {
                descriptor.set_attribute_descriptor(
                    BackendAttributeName::OperationSdpaBwdSubgraph,
                    subgraph.descriptor(),
                )?;
                descriptor.set_attribute_i64(
                    BackendAttributeName::OperationSdpaBwdSubgraphInputId,
                    subgraph_input_id.as_i64(),
                )?;
                descriptor.set_attribute_i64(
                    BackendAttributeName::OperationSdpaBwdSubgraphOutputId,
                    subgraph_output_id.as_i64(),
                )?;
            }
            (None, None, None) => {}
            _ => {
                return Err(Error::DescriptorMismatch {
                    name: "sdpa backward subgraph descriptor".into(),
                });
            }
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdDqDesc,
            d_q.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdDkDesc,
            d_k.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdDvDesc,
            d_v.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSdpaBwdDoDesc,
            d_o.descriptor(),
        )?;
        if let Some(sink) = config.sink {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaBwdSinkDesc,
                sink.descriptor(),
            )?;
        }
        if let Some(sink_gradient) = config.sink_gradient {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSdpaBwdDsinkDesc,
                sink_gradient.descriptor(),
            )?;
        }
        if let Some(max_total_sequence_length_query) = config.max_total_sequence_length_query {
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationSdpaBwdMaxTotalSeqLenQ,
                max_total_sequence_length_query,
            )?;
        }
        if let Some(max_total_sequence_length_key_value) =
            config.max_total_sequence_length_key_value
        {
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationSdpaBwdMaxTotalSeqLenKv,
                max_total_sequence_length_key_value,
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct SoftmaxOperation {
    descriptor: BackendDescriptor,
}

impl SoftmaxOperation {
    pub fn create(
        x: &Tensor,
        y: &Tensor,
        stats: Option<&Tensor>,
        max: Option<&Tensor>,
        sum_exp: Option<&Tensor>,
        sink: Option<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationSoftmax)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSoftmaxXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSoftmaxYDesc,
            y.descriptor(),
        )?;
        if let Some(stats) = stats {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSoftmaxStatsDesc,
                stats.descriptor(),
            )?;
        }
        if let Some(max) = max {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSoftmaxMaxDesc,
                max.descriptor(),
            )?;
        }
        if let Some(sum_exp) = sum_exp {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSoftmaxSumExpDesc,
                sum_exp.descriptor(),
            )?;
        }
        if let Some(sink) = sink {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSoftmaxSinkDesc,
                sink.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct DiagonalBandMaskOperation {
    descriptor: BackendDescriptor,
}

impl DiagonalBandMaskOperation {
    pub fn create(
        x: &Tensor,
        b: &Tensor,
        y: &Tensor,
        comparison_mode: PointwiseMode,
        sequence_length_query: Option<&Tensor>,
        sequence_length_key_value: Option<&Tensor>,
        left_bound: Option<&Tensor>,
        shift_right_bound: Option<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationDiagonalBandMask)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationDiagonalBandMaskXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationDiagonalBandMaskBDesc,
            b.descriptor(),
        )?;
        if let Some(sequence_length_query) = sequence_length_query {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationDiagonalBandMaskSeqLenQDesc,
                sequence_length_query.descriptor(),
            )?;
        }
        if let Some(sequence_length_key_value) = sequence_length_key_value {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationDiagonalBandMaskSeqLenKvDesc,
                sequence_length_key_value.descriptor(),
            )?;
        }
        if let Some(left_bound) = left_bound {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationDiagonalBandMaskLeftBoundDesc,
                left_bound.descriptor(),
            )?;
        }
        if let Some(shift_right_bound) = shift_right_bound {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationDiagonalBandMaskShiftRightBoundDesc,
                shift_right_bound.descriptor(),
            )?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationDiagonalBandMaskYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationDiagonalBandMaskComparisonMode,
            BackendAttributeType::PointwiseMode,
            comparison_mode,
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    num_enum::TryFromPrimitive,
    num_enum::IntoPrimitive,
    Serialize,
    Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum MoeGroupedMatmulMode {
    None = sys::cudnnMoeGroupedMatmulMode_t::CUDNN_MOE_GROUPED_MATMUL_MODE_NONE as _,
    Gather = sys::cudnnMoeGroupedMatmulMode_t::CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER as _,
    Scatter = sys::cudnnMoeGroupedMatmulMode_t::CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER as _,
}

impl From<MoeGroupedMatmulMode> for sys::cudnnMoeGroupedMatmulMode_t {
    fn from(value: MoeGroupedMatmulMode) -> Self {
        match value {
            MoeGroupedMatmulMode::None => Self::CUDNN_MOE_GROUPED_MATMUL_MODE_NONE,
            MoeGroupedMatmulMode::Gather => Self::CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER,
            MoeGroupedMatmulMode::Scatter => Self::CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER,
        }
    }
}

impl fmt::Display for MoeGroupedMatmulMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "CUDNN_MOE_GROUPED_MATMUL_MODE_NONE"),
            Self::Gather => write!(f, "CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER"),
            Self::Scatter => write!(f, "CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER"),
        }
    }
}

#[derive(Debug)]
pub struct MoeGroupedMatmulOperation {
    descriptor: BackendDescriptor,
}

impl MoeGroupedMatmulOperation {
    pub fn create(
        mode: MoeGroupedMatmulMode,
        compute_type: DataType,
        token: &Tensor,
        weight: &Tensor,
        first_token_offset: &Tensor,
        output: &Tensor,
        token_index: Option<&Tensor>,
        token_ks: Option<&Tensor>,
        top_k: Option<i32>,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationMoeGroupedMatmul)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationMoeGroupedMatmulMode,
            BackendAttributeType::MoeGroupedMatmulMode,
            mode,
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationMoeGroupedMatmulMathPrec,
            BackendAttributeType::DataType,
            compute_type,
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulTokenDesc,
            token.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulWeightDesc,
            weight.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulFirstTokenOffsetDesc,
            first_token_offset.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulOutputDesc,
            output.descriptor(),
        )?;
        if let Some(token_index) = token_index {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationMoeGroupedMatmulTokenIndexDesc,
                token_index.descriptor(),
            )?;
        }
        if let Some(token_ks) = token_ks {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationMoeGroupedMatmulTokenKsDesc,
                token_ks.descriptor(),
            )?;
        }
        if matches!(mode, MoeGroupedMatmulMode::Scatter)
            && let Some(top_k) = top_k
        {
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationMoeGroupedMatmulTopK,
                i64::from(top_k),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
