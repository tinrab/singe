use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::operation::{AttentionConfig, ReductionConfig},
};

pub(crate) const CUDNN_8_9_0_2: u64 = 8902;
pub(crate) const CUDNN_9_1: u64 = 90100;
pub(crate) const CUDNN_9_4: u64 = 90400;
pub(crate) const CUDNN_9_5: u64 = 90500;
pub(crate) const CUDNN_9_6: u64 = 90600;
pub(crate) const CUDNN_9_7: u64 = 90700;
pub(crate) const CUDNN_9_8: u64 = 90800;
pub(crate) const CUDNN_9_9: u64 = 90900;
pub(crate) const CUDNN_9_10: u64 = 91000;
pub(crate) const CUDNN_9_10_2: u64 = 91002;
pub(crate) const CUDNN_9_11: u64 = 91100;
pub(crate) const CUDNN_9_13: u64 = 91300;
pub(crate) const CUDNN_9_13_1: u64 = 91301;
pub(crate) const CUDNN_9_14: u64 = 91400;
pub(crate) const CUDNN_9_15: u64 = 91500;
pub(crate) const CUDNN_9_18: u64 = 91800;
pub(crate) const CUDNN_9_18_1: u64 = 91801;
pub(crate) const CUDNN_9_19: u64 = 91900;
pub(crate) const CUDNN_9_21: u64 = 92100;
pub(crate) const CUDNN_9_22: u64 = 92200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrontendFeatureSupport {
    pub(crate) feature: &'static str,
    pub(crate) min_version: u64,
    pub(crate) min_version_label: &'static str,
}

impl FrontendFeatureSupport {
    pub(crate) const fn new(
        feature: &'static str,
        min_version: u64,
        min_version_label: &'static str,
    ) -> Self {
        Self {
            feature,
            min_version,
            min_version_label,
        }
    }

    pub(crate) fn is_supported(self, cudnn_version: u64) -> bool {
        cudnn_version >= self.min_version
    }

    pub(crate) fn require_frontend_feature(self, cudnn_version: u64) -> Result<()> {
        if !self.is_supported(cudnn_version) {
            return Err(Error::FrontendFeatureRequiresVersion {
                feature: self.feature.into(),
                min_version: self.min_version_label.into(),
                actual_version: cudnn_version,
            });
        }
        Ok(())
    }
}

pub(crate) const DYNAMIC_SHAPE_OR_KERNEL_CACHE: FrontendFeatureSupport =
    FrontendFeatureSupport::new("dynamic shape or kernel cache", CUDNN_9_4, "9.4");
pub(crate) const OVERRIDE_SHAPE: FrontendFeatureSupport =
    FrontendFeatureSupport::new("override shape", CUDNN_9_21, "9.21");
pub(crate) const DEVICE_PROPERTIES: FrontendFeatureSupport =
    FrontendFeatureSupport::new("device properties", CUDNN_9_8, "9.8");
pub(crate) const KERNEL_CACHE_SERIALIZATION: FrontendFeatureSupport =
    FrontendFeatureSupport::new("kernel cache serialization", CUDNN_9_10, "9.10");

pub(crate) const CONCAT: FrontendFeatureSupport =
    FrontendFeatureSupport::new("concat", CUDNN_9_7, "9.7");
pub(crate) const BLOCK_SCALE_QUANTIZE: FrontendFeatureSupport =
    FrontendFeatureSupport::new("block scale quantize", CUDNN_9_7, "9.7");
pub(crate) const BLOCK_SCALE_DEQUANTIZE: FrontendFeatureSupport =
    FrontendFeatureSupport::new("block scale dequantize", CUDNN_9_7, "9.7");
pub(crate) const PAGED_CACHE_LOAD: FrontendFeatureSupport =
    FrontendFeatureSupport::new("paged cache load", CUDNN_9_5, "9.5");
pub(crate) const ADAPTIVE_LAYER_NORM: FrontendFeatureSupport =
    FrontendFeatureSupport::new("adaptive layer norm", CUDNN_9_9, "9.9");
pub(crate) const MOE_GROUPED_MATMUL: FrontendFeatureSupport =
    FrontendFeatureSupport::new("moe grouped matmul", CUDNN_9_15, "9.15");
pub(crate) const MOE_GROUPED_MATMUL_BACKWARD: FrontendFeatureSupport =
    FrontendFeatureSupport::new("moe grouped matmul backward", CUDNN_9_22, "9.22");
pub(crate) const DETERMINISTIC_REDUCTION: FrontendFeatureSupport =
    FrontendFeatureSupport::new("reduction deterministic", CUDNN_9_11, "9.11");

pub(crate) const SDPA_SINK_TOKEN: FrontendFeatureSupport =
    FrontendFeatureSupport::new("sdpa sink token", CUDNN_9_13, "9.13");
pub(crate) const SDPA_PAGED_ATTENTION: FrontendFeatureSupport =
    FrontendFeatureSupport::new("sdpa paged attention", CUDNN_9_5, "9.5");
pub(crate) const SDPA_UNIFIED_AUX_SOFTMAX: FrontendFeatureSupport =
    FrontendFeatureSupport::new("sdpa unified auxiliary softmax", CUDNN_9_21, "9.21");

pub(crate) fn runtime_requires_legacy_attention_primitives(cudnn_version: u64) -> bool {
    cudnn_version < CUDNN_9_21
}

pub(crate) fn runtime_requires_composite_sdpa_forward(cudnn_version: u64) -> bool {
    cudnn_version < CUDNN_9_13_1
}

pub(crate) const SDPA_SM_VERSION: &str = "sdpa sm version";
pub(crate) const SDPA_HEAD_DIMENSION: &str = "sdpa head dimension";
pub(crate) const SDPA_DECODE_HEAD_DIMENSION: &str = "sdpa decode head dimension";
pub(crate) const SDPA_PAGED_ATTENTION_HEAD_DIMENSION: &str = "sdpa paged attention head dimension";
pub(crate) const SDPA_PAGED_ATTENTION_RAGGED_OFFSETS: &str = "sdpa paged attention ragged offsets";

pub(crate) const BATCH_NORM_PEER_STAT_SHAPE: &str = "batch norm peer_stat shape";
pub(crate) const BLOCK_SCALE_RANK: &str = "block scale rank";
pub(crate) const CONCAT_RANKS: &str = "concat ranks";
pub(crate) const CONCAT_SHAPE: &str = "concat shape";
pub(crate) const CONVOLUTION_CHANNELS: &str = "convolution channels";
pub(crate) const CONVOLUTION_RANK: &str = "convolution rank";
pub(crate) const LAYER_NORM_RANK: &str = "layer norm rank";
pub(crate) const PAGED_CACHE_LOAD_RANK: &str = "paged cache load rank";
pub(crate) const PAGED_CACHE_LOAD_SHAPE: &str = "paged cache load shape";
pub(crate) const RESAMPLE_RANK: &str = "resample rank";
pub(crate) const RNG_OFFSET: &str = "rng offset";
pub(crate) const RNG_SEED: &str = "rng seed";
pub(crate) const SDPA_RANK: &str = "sdpa rank";
pub(crate) const SOFTMAX_OUTPUT_DATA_TYPE: &str = "softmax output data type";
pub(crate) const SOFTMAX_RANK: &str = "softmax rank";
pub(crate) const SOFTMAX_REDUCTION_TENSORS: &str = "softmax reduction tensors";
pub(crate) const TRANSPOSE_PERMUTATION: &str = "transpose permutation";
pub(crate) const SDPA_PAGED_K_PACKED_PAGE_TABLE_VERSION: &str =
    "sdpa paged k packed page table version";
pub(crate) const SDPA_PAGED_V_PACKED_PAGE_TABLE_VERSION: &str =
    "sdpa paged v packed page table version";
pub(crate) const SDPA_BACKWARD_SCALE: &str = "sdpa backward scale";
pub(crate) const SDPA_BACKWARD_CAUSAL_BOTTOM_RIGHT_VERSION: &str =
    "sdpa backward causal bottom right version";
pub(crate) const SDPA_BACKWARD_SINK_TOKEN_VERSION: &str = "sdpa backward sink token version";
pub(crate) const SDPA_BACKWARD_D_BIAS_VERSION: &str = "sdpa backward d_bias version";
pub(crate) const SDPA_BACKWARD_D_BIAS_SEQUENCE_LENGTHS: &str =
    "sdpa backward d_bias sequence lengths";
pub(crate) const SDPA_BACKWARD_DETERMINISTIC_ALGORITHM_VERSION: &str =
    "sdpa backward deterministic algorithm version";
pub(crate) const SDPA_MXFP8_BACKWARD_VERSION: &str = "sdpa mxfp8 backward version";
pub(crate) const SDPA_FP8_BACKWARD_VERSION: &str = "sdpa fp8 backward version";
pub(crate) const SDPA_FP8_BACKWARD_OUTPUT_DATA_TYPE: &str = "sdpa fp8 backward output data type";
pub(crate) const SDPA_FP8_BACKWARD_DETERMINISTIC_ALGORITHM_VERSION: &str =
    "sdpa fp8 backward deterministic algorithm version";

fn sdpa_architecture_unsupported(
    operation: &str,
    reason: &str,
    cudnn_version: u64,
    sm_version: Option<i32>,
    data_type: DataType,
    d_qk: i64,
    d_v: i64,
) -> Error {
    Error::FrontendSdpaArchitectureUnsupported {
        operation: operation.into(),
        reason: reason.into(),
        cudnn_version,
        sm_version,
        data_type,
        d_qk,
        d_v,
    }
}

#[derive(Debug, Clone, Copy)]
struct SdpaArchitectureSupportContext {
    operation: &'static str,
    cudnn_version: u64,
    sm_version: i32,
    data_type: DataType,
    d_qk: i64,
    d_v: i64,
}

impl SdpaArchitectureSupportContext {
    fn new(
        operation: &'static str,
        cudnn_version: u64,
        sm_version: i32,
        data_type: DataType,
        d_qk: i64,
        d_v: i64,
    ) -> Self {
        Self {
            operation,
            cudnn_version,
            sm_version,
            data_type,
            d_qk,
            d_v,
        }
    }

    fn prop_major(self) -> i32 {
        self.sm_version / 10
    }

    fn unsupported(self, reason: &str) -> Error {
        sdpa_architecture_unsupported(
            self.operation,
            reason,
            self.cudnn_version,
            Some(self.sm_version),
            self.data_type,
            self.d_qk,
            self.d_v,
        )
    }
}

fn require_cudnn_version(
    feature: &str,
    min_version: u64,
    min_version_label: &str,
    actual_version: u64,
    required: bool,
) -> Result<()> {
    if required && actual_version < min_version {
        return Err(Error::FrontendFeatureRequiresVersion {
            feature: feature.into(),
            min_version: min_version_label.into(),
            actual_version,
        });
    }
    Ok(())
}

fn require_range(name: &str, in_range: bool) -> Result<()> {
    if !in_range {
        return Err(Error::OutOfRange { name: name.into() });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SdpaSupportContext {
    cudnn_version: u64,
    sm_version: Option<i32>,
}

impl SdpaSupportContext {
    pub(crate) const fn new(cudnn_version: u64, sm_version: Option<i32>) -> Self {
        Self {
            cudnn_version,
            sm_version,
        }
    }

    pub(crate) const fn forward(self) -> SdpaForwardSupport {
        SdpaForwardSupport::new(self.cudnn_version, self.sm_version)
    }

    pub(crate) const fn backward(self) -> SdpaBackwardSupport {
        SdpaBackwardSupport::new(self.cudnn_version, self.sm_version)
    }

    pub(crate) const fn quantized(self) -> QuantizedSdpaSupport {
        QuantizedSdpaSupport::new(self.cudnn_version, self.sm_version)
    }

    pub(crate) fn require_forward_architecture(
        self,
        data_type: DataType,
        d_qk: i64,
        d_v: i64,
        is_decode_only: bool,
        is_paged: bool,
    ) -> Result<()> {
        require_sdpa_forward_architecture_support(
            self.cudnn_version,
            self.sm_version,
            data_type,
            d_qk,
            d_v,
            is_decode_only,
            is_paged,
        )
    }

    pub(crate) fn require_ragged_offsets_arch(self, is_ragged: bool) -> Result<()> {
        require_sdpa_ragged_offsets_arch_support(self.cudnn_version, self.sm_version, is_ragged)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UnifiedSdpaForwardSupport {
    cudnn_version: u64,
}

impl UnifiedSdpaForwardSupport {
    pub(crate) const fn new(cudnn_version: u64) -> Self {
        Self { cudnn_version }
    }

    pub(crate) fn supports_config(self, config: &AttentionConfig) -> bool {
        let modifiers = config.modifiers();
        let fused = config.fused();
        let aux_outputs = fused.aux_outputs();

        let uses_sequence_lengths = modifiers.sequence_lengths().is_some();
        let uses_pre_softmax_subgraph =
            config.score_subgraph().is_some() || modifiers.has_pre_softmax_subgraph_modifier();
        let uses_paged_cache = config.paged().has_any_request();
        self.supports_base_unified()
            && self.supports_score_subgraph(uses_pre_softmax_subgraph)
            && self.supports_sequence_lengths(uses_sequence_lengths)
            && self.supports_paged_cache(uses_paged_cache)
            && self.supports_block_mask(config.block_mask().is_some())
            && self.supports_native_dropout(config.dropout().is_some())
            && self.supports_unfused_fma(config.unfuse_fma())
            && self.supports_sink_token(modifiers.sink_token().is_some())
            && !modifiers.has_custom_dropout()
            && self.supports_extended_aux_outputs(aux_outputs.requires_unified_cudnn_921())
            && !aux_outputs.has_direct_quantized_request()
    }

    const fn supports_base_unified(self) -> bool {
        self.cudnn_version >= CUDNN_9_13_1
    }

    const fn supports_block_mask(self, feature_used: bool) -> bool {
        self.cudnn_version >= CUDNN_9_14 || !feature_used
    }

    const fn supports_sequence_lengths(self, feature_used: bool) -> bool {
        self.cudnn_version >= CUDNN_9_15 || !feature_used
    }

    const fn supports_paged_cache(self, feature_used: bool) -> bool {
        self.supports_sequence_lengths(feature_used)
    }

    const fn supports_score_subgraph(self, feature_used: bool) -> bool {
        self.supports_cudnn_921_feature(feature_used)
    }

    const fn supports_native_dropout(self, feature_used: bool) -> bool {
        self.supports_cudnn_921_feature(feature_used)
    }

    const fn supports_unfused_fma(self, feature_used: bool) -> bool {
        self.supports_cudnn_921_feature(feature_used)
    }

    const fn supports_sink_token(self, feature_used: bool) -> bool {
        self.supports_cudnn_921_feature(feature_used)
    }

    const fn supports_extended_aux_outputs(self, feature_used: bool) -> bool {
        self.supports_cudnn_921_feature(feature_used)
    }

    const fn supports_cudnn_921_feature(self, feature_used: bool) -> bool {
        self.cudnn_version >= CUDNN_9_21 || !feature_used
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SdpaForwardSupport {
    cudnn_version: u64,
    sm_version: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SdpaPagedTable {
    Key,
    Value,
}

impl SdpaPagedTable {
    const fn packed_page_table_error_name(self) -> &'static str {
        match self {
            Self::Key => SDPA_PAGED_K_PACKED_PAGE_TABLE_VERSION,
            Self::Value => SDPA_PAGED_V_PACKED_PAGE_TABLE_VERSION,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SdpaSlidingWindowSupport {
    pub(crate) left_window: i64,
    pub(crate) query_sequence_length: i64,
    pub(crate) key_sequence_length: i64,
    pub(crate) has_padding_mask: bool,
    pub(crate) has_causal_like_masking: bool,
    pub(crate) has_dropout: bool,
    pub(crate) has_bias: bool,
    pub(crate) has_causal_bottom_right: bool,
    pub(crate) is_decode_only: bool,
}

impl SdpaForwardSupport {
    pub(crate) const fn new(cudnn_version: u64, sm_version: Option<i32>) -> Self {
        Self {
            cudnn_version,
            sm_version,
        }
    }

    pub(crate) fn require_sliding_window(self, window: SdpaSlidingWindowSupport) -> Result<()> {
        require_range(
            "left_window",
            self.cudnn_version >= CUDNN_9_10 || window.left_window > 0,
        )?;
        if self.cudnn_version < CUDNN_9_10_2
            && (!window.has_causal_like_masking || window.has_dropout || window.has_bias)
        {
            return Err(Error::FrontendSdpaSlidingWindowModifierUnsupported {
                reason: sdpa_sliding_window_modifier_reason(window).into(),
                actual_version: self.cudnn_version,
            });
        }
        if self.cudnn_version <= CUDNN_9_9
            && self
                .sm_version
                .is_some_and(|sm_version| sm_version / 10 == 9)
            && window.has_causal_bottom_right
            && window.query_sequence_length * window.left_window
                == window.key_sequence_length * window.left_window
        {
            return Err(Error::FrontendSdpaSlidingWindowAlignmentUnsupported);
        }
        if !window.has_padding_mask && window.query_sequence_length > window.key_sequence_length {
            return Err(Error::FrontendSdpaSlidingWindowSequenceLengthUnsupported {
                query_sequence_length: window.query_sequence_length,
                key_sequence_length: window.key_sequence_length,
            });
        }
        if window.is_decode_only && self.cudnn_version <= CUDNN_9_9 {
            return Err(Error::FrontendSdpaDecodeMaskingUnsupported {
                actual_version: self.cudnn_version,
            });
        }
        if self.cudnn_version == CUDNN_9_14
            && window.key_sequence_length > 1024
            && !window.has_causal_like_masking
        {
            return Err(Error::FrontendSdpaCudnn914LongSequenceRequiresCausalMask {
                key_sequence_length: window.key_sequence_length,
            });
        }
        Ok(())
    }

    pub(crate) fn require_causal_bottom_right(
        self,
        enabled: bool,
        has_padding_mask: bool,
        has_sliding_window: bool,
        query_sequence_length: i64,
        key_sequence_length: i64,
    ) -> Result<()> {
        if !enabled {
            return Ok(());
        }
        if !has_padding_mask && query_sequence_length > key_sequence_length {
            return Err(
                Error::FrontendSdpaCausalBottomRightSequenceLengthUnsupported {
                    query_sequence_length,
                    key_sequence_length,
                },
            );
        }
        if self.cudnn_version < CUDNN_9_6
            && (query_sequence_length % 64 != 0 || key_sequence_length % 64 != 0)
        {
            return Err(
                Error::FrontendSdpaCausalBottomRightSequenceLengthAlignmentUnsupported {
                    query_sequence_length,
                    key_sequence_length,
                    actual_version: self.cudnn_version,
                },
            );
        }
        if has_sliding_window
            && self.cudnn_version < CUDNN_9_6
            && query_sequence_length != key_sequence_length
        {
            return Err(
                Error::FrontendSdpaSlidingWindowCausalBottomRightAlignmentUnsupported {
                    query_sequence_length,
                    key_sequence_length,
                    actual_version: self.cudnn_version,
                },
            );
        }
        Ok(())
    }

    pub(crate) fn require_paged_ragged_offsets(
        self,
        is_paged: bool,
        is_ragged: bool,
    ) -> Result<()> {
        require_cudnn_version(
            SDPA_PAGED_ATTENTION_RAGGED_OFFSETS,
            CUDNN_9_7,
            "9.7",
            self.cudnn_version,
            is_paged && is_ragged,
        )
    }

    pub(crate) fn require_packed_page_table(
        self,
        table: SdpaPagedTable,
        has_ragged_offset: bool,
    ) -> Result<()> {
        require_cudnn_version(
            table.packed_page_table_error_name(),
            CUDNN_9_10_2,
            "9.10.2",
            self.cudnn_version,
            has_ragged_offset,
        )
    }
}

fn sdpa_sliding_window_modifier_reason(window: SdpaSlidingWindowSupport) -> &'static str {
    if window.has_dropout {
        "dropout"
    } else if window.has_bias {
        "bias"
    } else if !window.has_causal_like_masking {
        "non-causal masking"
    } else {
        "modifier combination"
    }
}

pub(crate) fn requires_sdpa_ragged_mask_or_subgraph(
    is_ragged: bool,
    has_padding_mask: bool,
    has_additive_mask: bool,
    has_score_subgraph: bool,
) -> bool {
    is_ragged && !has_padding_mask && !has_additive_mask && !has_score_subgraph
}

pub(crate) fn require_sdpa_ragged_mask_or_subgraph(
    is_ragged: bool,
    has_padding_mask: bool,
    has_additive_mask: bool,
    has_score_subgraph: bool,
    operation: &str,
) -> Result<()> {
    if requires_sdpa_ragged_mask_or_subgraph(
        is_ragged,
        has_padding_mask,
        has_additive_mask,
        has_score_subgraph,
    ) {
        return Err(Error::FrontendSdpaRaggedOffsetsRequireMaskOrSubgraph {
            operation: operation.into(),
        });
    }
    Ok(())
}

pub(crate) fn requires_sdpa_sequence_lengths_mask_or_subgraph(
    has_sequence_lengths: bool,
    has_padding_mask: bool,
    has_causal_bottom_right: bool,
    has_score_subgraph: bool,
) -> bool {
    has_sequence_lengths && !has_padding_mask && !has_causal_bottom_right && !has_score_subgraph
}

pub(crate) fn require_sdpa_sequence_lengths_mask_or_subgraph(
    has_sequence_lengths: bool,
    has_padding_mask: bool,
    has_causal_bottom_right: bool,
    has_score_subgraph: bool,
    operation: &str,
) -> Result<()> {
    if requires_sdpa_sequence_lengths_mask_or_subgraph(
        has_sequence_lengths,
        has_padding_mask,
        has_causal_bottom_right,
        has_score_subgraph,
    ) {
        return Err(Error::FrontendSdpaSequenceLengthsRequireMaskOrSubgraph {
            operation: operation.into(),
        });
    }
    Ok(())
}

pub(crate) fn rejects_sdpa_ragged_offsets_for_arch(
    cudnn_version: u64,
    sm_version: Option<i32>,
    is_ragged: bool,
) -> bool {
    is_ragged
        && sm_version.is_some_and(|sm_version| sm_version < 90)
        && cudnn_version < CUDNN_9_18_1
}

pub(crate) fn require_sdpa_ragged_offsets_arch_support(
    cudnn_version: u64,
    sm_version: Option<i32>,
    is_ragged: bool,
) -> Result<()> {
    if rejects_sdpa_ragged_offsets_for_arch(cudnn_version, sm_version, is_ragged) {
        return Err(Error::FrontendSdpaRaggedOffsetsArchitectureUnsupported {
            cudnn_version,
            sm_version,
        });
    }
    Ok(())
}

pub(crate) fn require_sdpa_forward_data_type_support(
    cudnn_version: u64,
    data_type: DataType,
) -> Result<()> {
    if matches!(data_type, DataType::F16 | DataType::BF16)
        && (CUDNN_9_10..CUDNN_9_10_2).contains(&cudnn_version)
    {
        return Err(Error::FrontendSdpaDataTypeRequiresCudnnVersion {
            operation: String::from("sdpa"),
            data_type,
            min_version: String::from("9.10.2"),
            actual_version: cudnn_version,
        });
    }
    Ok(())
}

pub(crate) fn require_sdpa_forward_architecture_support(
    cudnn_version: u64,
    sm_version: Option<i32>,
    data_type: DataType,
    d_qk: i64,
    d_v: i64,
    is_decode_only: bool,
    is_paged: bool,
) -> Result<()> {
    let Some(sm_version) = sm_version else {
        return Ok(());
    };
    let support = SdpaArchitectureSupportContext::new(
        "sdpa",
        cudnn_version,
        sm_version,
        data_type,
        d_qk,
        d_v,
    );
    let prop_major = support.prop_major();

    if matches!(data_type, DataType::F16 | DataType::BF16) && sm_version < 80 {
        return Err(support.unsupported(SDPA_SM_VERSION));
    }

    if prop_major == 8 && cudnn_version <= CUDNN_9_9 && (d_qk > 128 || d_v > 128) {
        return Err(support.unsupported(SDPA_HEAD_DIMENSION));
    }
    if prop_major == 9 && cudnn_version <= CUDNN_9_9 && (d_qk > 256 || d_v > 256) {
        return Err(support.unsupported(SDPA_HEAD_DIMENSION));
    }
    if prop_major == 10 && cudnn_version < CUDNN_9_9 && (d_qk > 128 || d_v > 128) {
        return Err(support.unsupported(SDPA_HEAD_DIMENSION));
    }
    if is_decode_only && prop_major == 10 && cudnn_version <= CUDNN_9_9 && (d_qk > 128 || d_v > 128)
    {
        return Err(support.unsupported(SDPA_DECODE_HEAD_DIMENSION));
    }
    if is_paged && cudnn_version <= CUDNN_9_9 && (d_qk > 128 || d_v > 128) {
        return Err(support.unsupported(SDPA_PAGED_ATTENTION_HEAD_DIMENSION));
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SdpaBackwardSupport {
    pub(crate) cudnn_version: u64,
    pub(crate) sm_version: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SdpaBackwardCausalBottomRightSupport {
    pub(crate) enabled: bool,
    pub(crate) query_sequence_length: i64,
    pub(crate) key_sequence_length: i64,
    pub(crate) has_dropout: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SdpaBackwardBiasGradientSupport {
    pub(crate) enabled: bool,
    pub(crate) has_padding_mask: bool,
    pub(crate) query_sequence_length: i64,
    pub(crate) key_sequence_length: i64,
    pub(crate) is_ragged: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SdpaBackwardDeterministicSupport {
    pub(crate) enabled: bool,
    pub(crate) has_bias_gradient: bool,
    pub(crate) has_dropout: bool,
    pub(crate) has_alibi: bool,
    pub(crate) is_ragged: bool,
}

impl SdpaBackwardSupport {
    pub(crate) const fn new(cudnn_version: u64, sm_version: Option<i32>) -> Self {
        Self {
            cudnn_version,
            sm_version,
        }
    }

    pub(crate) fn rejects_ragged_stats(self, has_ragged_stats: bool) -> bool {
        has_ragged_stats
            && self.cudnn_version >= CUDNN_9_18_1
            && self
                .sm_version
                .is_some_and(|sm_version| matches!(sm_version / 10, 8 | 12))
    }

    pub(crate) fn require_causal_bottom_right(
        self,
        rule: SdpaBackwardCausalBottomRightSupport,
    ) -> Result<()> {
        if !rule.enabled {
            return Ok(());
        }
        require_cudnn_version(
            SDPA_BACKWARD_CAUSAL_BOTTOM_RIGHT_VERSION,
            CUDNN_9_7,
            "9.7",
            self.cudnn_version,
            true,
        )?;
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 < 10)
        {
            return Err(
                Error::FrontendSdpaBackwardCausalBottomRightArchitectureUnsupported {
                    sm_version: self.sm_version,
                },
            );
        }
        if rule.query_sequence_length > rule.key_sequence_length {
            return Err(
                Error::FrontendSdpaBackwardCausalBottomRightSequenceLengthUnsupported {
                    query_sequence_length: rule.query_sequence_length,
                    key_sequence_length: rule.key_sequence_length,
                },
            );
        }
        if rule.has_dropout {
            return Err(Error::FrontendSdpaBackwardCausalBottomRightDropoutUnsupported);
        }
        if rule.query_sequence_length % 64 != 0 || rule.key_sequence_length % 64 != 0 {
            return Err(
                Error::FrontendSdpaBackwardCausalBottomRightSequenceLengthAlignmentUnsupported {
                    query_sequence_length: rule.query_sequence_length,
                    key_sequence_length: rule.key_sequence_length,
                },
            );
        }
        Ok(())
    }

    pub(crate) fn require_sink_token(self, has_sink_token: bool) -> Result<()> {
        require_cudnn_version(
            SDPA_BACKWARD_SINK_TOKEN_VERSION,
            CUDNN_9_13,
            "9.13",
            self.cudnn_version,
            has_sink_token,
        )
    }

    pub(crate) fn require_bias_gradient(self, rule: SdpaBackwardBiasGradientSupport) -> Result<()> {
        if !rule.enabled {
            return Ok(());
        }
        require_cudnn_version(
            SDPA_BACKWARD_D_BIAS_VERSION,
            CUDNN_9_5,
            "9.5",
            self.cudnn_version,
            rule.has_padding_mask,
        )?;
        require_cudnn_version(
            SDPA_BACKWARD_D_BIAS_SEQUENCE_LENGTHS,
            CUDNN_9_5,
            "9.5",
            self.cudnn_version,
            rule.query_sequence_length % 64 != 0 || rule.key_sequence_length % 64 != 0,
        )?;
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 == 10)
            && rule.is_ragged
        {
            return Err(Error::FrontendSdpaBackwardBiasGradientRaggedUnsupported {
                sm_version: self.sm_version,
            });
        }
        Ok(())
    }

    pub(crate) fn require_deterministic_algorithm(
        self,
        rule: SdpaBackwardDeterministicSupport,
    ) -> Result<()> {
        if !rule.enabled {
            return Ok(());
        }
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 == 10)
        {
            require_cudnn_version(
                SDPA_BACKWARD_DETERMINISTIC_ALGORITHM_VERSION,
                CUDNN_9_18,
                "9.18",
                self.cudnn_version,
                true,
            )?;
            if rule.has_bias_gradient || rule.has_dropout || rule.has_alibi {
                return Err(
                    Error::FrontendSdpaBackwardDeterministicAlgorithmModifierUnsupported {
                        sm_version: self.sm_version,
                    },
                );
            }
        }
        if rule.is_ragged && self.rejects_ragged_stats(true) {
            return Err(
                Error::FrontendSdpaBackwardDeterministicAlgorithmRaggedUnsupported {
                    sm_version: self.sm_version,
                    cudnn_version: self.cudnn_version,
                },
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuantizedSdpaSupport {
    cudnn_version: u64,
    sm_version: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MxFp8BackwardSupport {
    pub(crate) output_io_type: DataType,
    pub(crate) is_ragged: bool,
    pub(crate) has_dropout: bool,
    pub(crate) has_random_number_generator_dump: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Fp8BackwardSupport {
    pub(crate) output_io_type: DataType,
    pub(crate) d_qk: i64,
    pub(crate) d_v: i64,
    pub(crate) is_ragged: bool,
    pub(crate) has_dropout: bool,
    pub(crate) has_random_number_generator_dump: bool,
}

impl QuantizedSdpaSupport {
    pub(crate) const fn new(cudnn_version: u64, sm_version: Option<i32>) -> Self {
        Self {
            cudnn_version,
            sm_version,
        }
    }

    pub(crate) fn require_mxfp8_backward(self, rule: MxFp8BackwardSupport) -> Result<()> {
        self.require_quantized_backward_architecture("mxfp8 sdpa backward")?;
        self.require_quantized_backward_ragged("mxfp8 sdpa backward", rule.is_ragged)?;
        require_cudnn_version(
            SDPA_MXFP8_BACKWARD_VERSION,
            CUDNN_9_21,
            "9.21",
            self.cudnn_version,
            true,
        )?;
        self.require_fp8_output_io_type("mxfp8 sdpa backward", rule.output_io_type)?;
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 == 10)
            && rule.has_dropout
        {
            return Err(Error::FrontendSdpaQuantizedBackwardDropoutUnsupported {
                operation: "mxfp8 sdpa backward".into(),
                sm_version: self.sm_version,
            });
        }
        self.require_quantized_backward_no_rng_dump(
            "mxfp8 sdpa backward",
            rule.has_random_number_generator_dump,
        )
    }

    pub(crate) fn require_fp8_backward(self, rule: Fp8BackwardSupport) -> Result<()> {
        require_cudnn_version(
            SDPA_FP8_BACKWARD_VERSION,
            CUDNN_9_1,
            "9.1",
            self.cudnn_version,
            true,
        )?;
        if self.cudnn_version == CUDNN_9_10 {
            return Err(Error::FrontendSdpaFp8BackwardCudnn910Unsupported);
        }
        self.require_quantized_backward_architecture("fp8 sdpa backward")?;
        self.require_quantized_backward_ragged("fp8 sdpa backward", rule.is_ragged)?;
        self.require_fp8_output_io_type("fp8 sdpa backward", rule.output_io_type)?;

        let requires_deterministic_algorithm = self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 == 10)
            && rule.d_qk <= 192
            && rule.d_v <= 128;
        if requires_deterministic_algorithm {
            require_cudnn_version(
                SDPA_FP8_BACKWARD_DETERMINISTIC_ALGORITHM_VERSION,
                CUDNN_9_19,
                "9.19",
                self.cudnn_version,
                true,
            )?;
            if rule.has_dropout {
                return Err(Error::FrontendSdpaQuantizedBackwardDropoutUnsupported {
                    operation: "fp8 sdpa backward".into(),
                    sm_version: self.sm_version,
                });
            }
        }
        self.require_quantized_backward_no_rng_dump(
            "fp8 sdpa backward",
            rule.has_random_number_generator_dump,
        )
    }

    pub(crate) fn require_fp8_forward_output_type(self, output_io_type: DataType) -> Result<()> {
        self.require_fp8_output_io_type("fp8 sdpa", output_io_type)
    }

    pub(crate) fn require_quantized_backward_head_dimensions(
        self,
        d_qk: i64,
        d_v: i64,
        operation: &str,
    ) -> Result<()> {
        let Some(sm_version) = self.sm_version else {
            return Ok(());
        };
        let supported = if sm_version / 10 >= 10 {
            ((d_qk <= 128) && (d_qk % 16 == 0) && (d_v <= 128) && (d_v % 16 == 0))
                || (d_qk == 192 && d_v == 128)
        } else {
            d_qk == 128 && d_v == 128 && d_qk % 16 == 0 && d_v % 16 == 0
        };
        if !supported {
            return Err(
                Error::FrontendSdpaQuantizedBackwardHeadDimensionUnsupported {
                    operation: operation.into(),
                    d_qk,
                    d_v,
                    sm_version: self.sm_version,
                },
            );
        }
        Ok(())
    }

    pub(crate) fn require_quantized_backward_output_ragged_support(
        self,
        has_ragged_output: bool,
        operation: &str,
    ) -> Result<()> {
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 == 9)
            && has_ragged_output
        {
            return Err(
                Error::FrontendSdpaQuantizedBackwardRaggedOutputUnsupported {
                    operation: operation.into(),
                    sm_version: self.sm_version,
                },
            );
        }
        Ok(())
    }

    fn require_fp8_output_io_type(self, operation: &str, output_io_type: DataType) -> Result<()> {
        if matches!(output_io_type, DataType::F16 | DataType::BF16)
            && (self.cudnn_version < CUDNN_9_13
                || self
                    .sm_version
                    .is_some_and(|sm_version| sm_version / 10 < 10))
        {
            return Err(
                Error::FrontendSdpaQuantizedOutputDataTypeRequiresCudnn913Sm100 {
                    operation: operation.into(),
                    output_io_type,
                    cudnn_version: self.cudnn_version,
                    sm_version: self.sm_version,
                },
            );
        }
        Ok(())
    }

    fn require_quantized_backward_architecture(self, operation: &str) -> Result<()> {
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 < 9)
        {
            return Err(
                Error::FrontendSdpaQuantizedBackwardArchitectureUnsupported {
                    operation: operation.into(),
                    sm_version: self.sm_version,
                },
            );
        }
        Ok(())
    }

    fn require_quantized_backward_ragged(self, operation: &str, is_ragged: bool) -> Result<()> {
        if self
            .sm_version
            .is_some_and(|sm_version| sm_version / 10 == 9)
            && is_ragged
        {
            return Err(Error::FrontendSdpaQuantizedBackwardRaggedUnsupported {
                operation: operation.into(),
                sm_version: self.sm_version,
            });
        }
        Ok(())
    }

    fn require_quantized_backward_no_rng_dump(
        self,
        operation: &str,
        has_random_number_generator_dump: bool,
    ) -> Result<()> {
        if has_random_number_generator_dump {
            return Err(Error::FrontendSdpaQuantizedBackwardRngDumpUnsupported {
                operation: operation.into(),
            });
        }
        Ok(())
    }
}

pub(crate) fn require_reduction_support(cudnn_version: u64, config: ReductionConfig) -> Result<()> {
    if config.is_deterministic() {
        DETERMINISTIC_REDUCTION.require_frontend_feature(cudnn_version)?;
    }
    Ok(())
}
