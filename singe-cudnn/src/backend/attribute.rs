use std::fmt;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;

use crate::{
    backend::{
        behavior::{BackendBehaviorNote, BackendNumericalNote},
        convolution::ConvolutionMode,
        execution::{
            moe::MoeGroupedMatmulMode,
            normalization::{BatchNormalizationFinalizeStatsMode, GenStatsMode},
            resample::{PaddingMode, ResampleMode},
            rng::RngDistribution,
            tensor_ops::{ReshapeMode, SignalMode},
        },
        heuristic::BackendHeuristicMode,
        knob::BackendKnobType,
        layout::BackendLayoutType,
        normalization::{BackendNormalizationForwardPhase, BackendNormalizationMode},
        pointwise::PointwiseMode,
        reduction::ReduceTensorOperator,
        tensor::BackendTensorReordering,
    },
    data_type::DataType,
    math::NanPropagation,
};

/// Backend descriptor attribute name that can be set or retrieved with
/// [`BackendDescriptor`](crate::backend::descriptor::BackendDescriptor) attribute methods.
/// The backend descriptor to which an attribute belongs is identified by the prefix of the attribute name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum BackendAttributeName {
    /// Descriptor of the group-offset tensor for a grouped reduction operation.
    OperationReductionGroupOffsetDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_GROUP_OFFSET_DESC as _,
    ///< Backward math precision for `MoE` grouped matmul. @since cuDNN 9.22.0
    OperationMoeGroupedMatmulBackwardMathPrec = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC as _,
    ///< Backward token descriptor for `MoE` grouped matmul. @since cuDNN 9.22.0
    OperationMoeGroupedMatmulBackwardTokenDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC as _,
    ///< Backward weight descriptor for `MoE` grouped matmul. @since cuDNN 9.22.0
    OperationMoeGroupedMatmulBackwardWeightGradientDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC as _,
    ///< Backward first token offset descriptor for `MoE` grouped matmul. @since cuDNN 9.22.0
    OperationMoeGroupedMatmulBackwardFirstTokenOffsetDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC as _,
    ///< Backward output descriptor for `MoE` grouped matmul. @since cuDNN 9.22.0
    OperationMoeGroupedMatmulBackwardOutputGradientDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC as _,

    /// `RoPE` input tensor descriptor with shape `[B, H, S, D]`. Available since cuDNN 9.24.0.
    OperationRopeForwardInputDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_ROPE_FWD_XDESC as _,
    /// `RoPE` output tensor descriptor with shape `[B, H, S, D]`. Available since cuDNN 9.24.0.
    OperationRopeForwardOutputDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_ROPE_FWD_YDESC as _,
    /// `RoPE` frequency-angle tensor with shape `[S, 1, 1, D]`. The kernel computes `sincosf` internally. Available since cuDNN 9.24.0.
    OperationRopeForwardFrequenciesDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC as _,
    ///< `RoPE` host-side fp32 scalar multiplied into cos/sin (default 1.0). Used to fold YARN mscale and/or SDPA `attn_scale`. @since cuDNN 9.24.0
    OperationRopeForwardOutputScale = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE as _,
    ///< `RoPE` rotation width in last dim (int64, default = `head_dim`). When < `head_dim`, the last `rope_dim` slots get rotated; the first (`head_dim` - `rope_dim`) are scaled-pass-through (used for DSv3-style MLA). @since cuDNN 9.24.0
    OperationRopeForwardRotationDimension = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM as _,
    /// `RoPE` input-gradient (`dY`) tensor descriptor with shape `[B, H, S, D]`. Available since cuDNN 9.25.0.
    OperationRopeBackwardInputDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_ROPE_BWD_DYDESC as _,
    /// `RoPE` output-gradient (`dX`) tensor descriptor with shape `[B, H, S, D]`. Available since cuDNN 9.25.0.
    OperationRopeBackwardOutputDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_ROPE_BWD_DXDESC as _,
    /// `RoPE` backward frequency-angle tensor with shape `[S, 1, 1, D]`, using the same semantics as forward execution. Available since cuDNN 9.25.0.
    OperationRopeBackwardFrequenciesDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_ROPE_BWD_FREQSDESC as _,
    ///< `RoPE` bwd host fp32 scalar (default 1.0). Symmetric to fwd: cos/sin pre-scaled, passthrough scaled. @since cuDNN 9.25.0
    OperationRopeBackwardOutputScale = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_ROPE_BWD_OUTPUT_SCALE as _,
    ///< `RoPE` bwd rotation width in last dim (int64, default = `head_dim`). @since cuDNN 9.25.0
    OperationRopeBackwardRotationDimension = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_ROPE_BWD_ROPE_DIM as _,

    ///< Compile-time constant value for by-value tensors. @since cuDNN 9.22.0
    TensorConstantValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_CONSTANT_VALUE as _,
    ///< Ragged tensor offset unit size. @since cuDNN 9.24.0
    TensorRaggedOffsetMultiplier = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_RAGGED_OFFSET_MULTIPLIER as _,
    ///< Reshape mode (view-only or logical). @since cuDNN 9.22.0
    OperationReshapeMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESHAPE_MODE as _,
    ///< Slice input tensor descriptor. @since cuDNN 9.22.0
    OperationSliceXDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SLICE_XDESC as _,
    ///< Slice output tensor descriptor. @since cuDNN 9.22.0
    OperationSliceYDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SLICE_YDESC as _,
    ///< Slice start indices. @since cuDNN 9.22.0
    OperationSliceStartIndices = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SLICE_START_INDICES as _,
    ///< Slice limit indices. @since cuDNN 9.22.0
    OperationSliceLimitIndices = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES as _,
    ///< Slice strides. @since cuDNN 9.22.0
    OperationSliceStrides = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SLICE_STRIDES as _,
    ///< Transpose input tensor descriptor. @since cuDNN 9.22.0
    OperationTransposeXDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_TRANSPOSE_XDESC as _,
    ///< Transpose output tensor descriptor. @since cuDNN 9.22.0
    OperationTransposeYDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_TRANSPOSE_YDESC as _,
    ///< Transpose permutation array. @since cuDNN 9.22.0
    OperationTransposePermutation = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION as _,
    /// Shared memory limit metadata propagated to heuristic results.
    EngineHeurSharedMemoryLimit = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_SHARED_MEMORY_LIMIT as _,
    /// Dynamic shared memory budget, in bytes.
    EngineSharedMemoryLimit = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_SHARED_MEMORY_LIMIT as _,
    /// Counter that shows if the kernel cache changed.
    KernelCacheRevision = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KERNEL_CACHE_REVISION as _,
    /// Number of entries in the kernel cache.
    KernelCacheEntryCount = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KERNEL_CACHE_ENTRY_COUNT as _,
    /// Emit statistics as natural-log LSE multiplied by log2(e). SDPA backward expects natural-log statistics.
    OperationSoftmaxStatsLog2 = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2 as _,
    /// Indices of the input dimensions to reduce, ascending and distinct.
    OperationReductionDims = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_DIMS as _,
    /// Accumulator identity values, one pointer per output; set the output descriptors first.
    OperationReductionInitValues = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_INIT_VALUES as _,
    /// Finalized operation graph describing the reducer body combining accumulator and current element.
    OperationReductionSubgraph = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH as _,
    /// Subgraph tensor IDs holding accumulator values, one per output.
    OperationReductionSubgraphAccumulatorUids = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_ACCUMULATOR_UIDS as _,
    /// Subgraph tensor IDs holding current elements, one per output.
    OperationReductionSubgraphCurrentUids = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_CURRENT_UIDS as _,
    /// Subgraph tensor IDs holding output values, one per output.
    OperationReductionSubgraphOutputUids = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_OUTPUT_UIDS as _,
    /// Optional compute type the reducer body evaluates in, one per output; defaults to output element type.
    OperationReductionSubgraphComputeTypes = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_COMPUTE_TYPES as _,

    ///< Pointwise operation type. @since cuDNN 9.0.0
    PointwiseMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_MODE as _,
    ///< Computation precision for pointwise ops. @since cuDNN 9.0.0
    PointwiseMathPrec = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_MATH_PREC as _,
    ///< NaN handling behavior. @deprecated @since cuDNN 9.0.0
    PointwiseNanPropagation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_NAN_PROPAGATION as _,
    ///< Lower clipping threshold for `ReLU`. @since cuDNN 9.0.0
    PointwiseReluLowerClip =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP as _,
    ///< Upper clipping threshold for `ReLU`. @since cuDNN 9.0.0
    PointwiseReluUpperClip =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP as _,
    ///< Slope below lower clip for leaky `ReLU`. @since cuDNN 9.0.0
    PointwiseReluLowerClipSlope =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE as _,
    ///< Alpha parameter for ELU activation. @since cuDNN 9.0.0
    PointwiseEluAlpha = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_ELU_ALPHA as _,
    ///< Beta parameter for softplus function. @since cuDNN 9.0.0
    PointwiseSoftplusBeta = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA as _,
    ///< Beta parameter for swish activation. @since cuDNN 9.0.0
    PointwiseSwishBeta = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_SWISH_BETA as _,
    ///< Axis for axis-dependent pointwise operations. @since cuDNN 9.0.0
    PointwiseAxis = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_AXIS as _,

    ///< Computation data type for convolution. @since cuDNN 9.0.0
    ConvolutionCompType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_COMP_TYPE as _,
    ///< Convolution vs cross-correlation mode. @since cuDNN 9.0.0
    ConvolutionConvMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_CONV_MODE as _,
    ///< Dilation factors per spatial dimension. @since cuDNN 9.0.0
    ConvolutionDilations = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_DILATIONS as _,
    ///< Filter stride values per spatial dimension. @since cuDNN 9.0.0
    ConvolutionFilterStrides =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES as _,
    ///< Post-paddings per spatial dimension. @since cuDNN 9.0.0
    ConvolutionPostPaddings =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_POST_PADDINGS as _,
    ///< Pre-paddings per spatial dimension. @since cuDNN 9.0.0
    ConvolutionPrePaddings =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS as _,
    ///< Number of spatial dimensions. @since cuDNN 9.0.0
    ConvolutionSpatialDims =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS as _,

    ///< Heuristic algorithm selection mode. @since cuDNN 9.0.0
    EngineHeurMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_MODE as _,
    ///< Associated operation graph descriptor. @since cuDNN 9.0.0
    EngineHeurOperationGraph =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH as _,
    ///< Array of resulting engine configurations. @since cuDNN 9.0.0
    EngineHeurResults = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_RESULTS as _,
    ///< Target streaming multiprocessor count. @since cuDNN 9.0.0
    EngineHeurSmCountTarget =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET as _,
    ///< Device properties for heuristic query. @since cuDNN 9.8.0
    EngineHeurDeviceProp = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_DEVICEPROP as _,

    ///< Selected engine from heuristic results. @since cuDNN 9.0.0
    EngineCfgEngine = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_ENGINE as _,
    ///< Intermediate tensor information. @since cuDNN 9.0.0
    EngineCfgIntermediateInfo =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO as _,
    ///< Performance tuning knob selections. @since cuDNN 9.0.0
    EngineCfgKnobChoices = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_KNOB_CHOICES as _,
    ///< Required workspace memory size. @since cuDNN 9.2.0
    EngineCfgWorkspaceSize =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE as _,
    /// Shared-memory requirement reported by an engine configuration, in bytes.
    EngineCfgSharedMemoryUsed = 304,
    ///< Associated cuDNN handle. @deprecated @since cuDNN 9.0.0
    ExecutionPlanHandle = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_HANDLE as _,
    ///< Engine configuration to execute. @since cuDNN 9.0.0
    ExecutionPlanEngineConfig =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG as _,
    ///< Total workspace size requirement. @since cuDNN 9.0.0
    ExecutionPlanWorkspaceSize =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE as _,
    ///< UIDs of computed intermediates. @since cuDNN 9.0.0
    ExecutionPlanComputedIntermediateIds =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS as _,
    ///< Run-only intermediate UIDs. @since cuDNN 9.0.0
    ExecutionPlanRunOnlyIntermediateIds =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS as _,
    ///< Human-readable execution plan JSON. @since cuDNN 9.0.0
    ExecutionPlanJsonRepresentation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION as _,
    ///< Compiled kernel cache descriptor. @since cuDNN 9.4.0
    ExecutionPlanKernelCache =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE as _,
    ///< Device properties for execution plan. @since cuDNN 9.8.0
    ExecutionPlanDeviceProp =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP as _,

    ///< Unique identifier for intermediate tensor. @since cuDNN 9.0.0
    IntermediateInfoUniqueId =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID as _,
    ///< Memory size requirement in bytes. @since cuDNN 9.0.0
    IntermediateInfoSize = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_INTERMEDIATE_INFO_SIZE as _,
    ///< Data dependency UIDs. @since cuDNN 9.0.0
    IntermediateInfoDependentTensorUids =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS as _,
    ///< Attribute dependencies. @since cuDNN 9.0.0
    IntermediateInfoDependentAttributes =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES as _,

    ///< Type of performance tuning knob. @since cuDNN 9.0.0
    KnobChoiceKnobType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE as _,
    ///< Selected value for the knob. @since cuDNN 9.0.0
    KnobChoiceKnobValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE as _,

    ///< Forward convolution scaling factor alpha. @since cuDNN 9.0.0
    OperationConvolutionForwardAlpha =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA as _,
    ///< Forward convolution scaling factor beta. @since cuDNN 9.0.0
    OperationConvolutionForwardBeta =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA as _,
    ///< Forward convolution descriptor. @since cuDNN 9.0.0
    OperationConvolutionForwardConvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC as _,
    ///< Forward convolution filter weight tensor. @since cuDNN 9.0.0
    OperationConvolutionForwardW =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W as _,
    ///< Forward convolution input tensor. @since cuDNN 9.0.0
    OperationConvolutionForwardX =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X as _,
    ///< Forward convolution output tensor. @since cuDNN 9.0.0
    OperationConvolutionForwardY =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y as _,
    ///< Backward data scaling factor alpha. @since cuDNN 9.0.0
    OperationConvolutionBwdDataAlpha =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA as _,
    ///< Backward data scaling factor beta. @since cuDNN 9.0.0
    OperationConvolutionBwdDataBeta =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA as _,
    ///< Backward data convolution descriptor. @since cuDNN 9.0.0
    OperationConvolutionBwdDataConvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC as _,
    ///< Backward data filter weights. @since cuDNN 9.0.0
    OperationConvolutionBwdDataW =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W as _,
    ///< Backward data input gradient output. @since cuDNN 9.0.0
    OperationConvolutionBwdDataDx =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX as _,
    ///< Backward data output gradient input. @since cuDNN 9.0.0
    OperationConvolutionBwdDataDy =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY as _,
    ///< Backward filter scaling factor alpha. @since cuDNN 9.0.0
    OperationConvolutionBwdFilterAlpha =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA as _,
    ///< Backward filter scaling factor beta. @since cuDNN 9.0.0
    OperationConvolutionBwdFilterBeta =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA as _,
    ///< Backward filter convolution descriptor. @since cuDNN 9.0.0
    OperationConvolutionBwdFilterConvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC as _,
    ///< Backward filter gradient output. @since cuDNN 9.0.0
    OperationConvolutionBwdFilterDw =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW as _,
    ///< Backward filter input feature maps. @since cuDNN 9.0.0
    OperationConvolutionBwdFilterX =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X as _,
    ///< Backward filter output gradients. @since cuDNN 9.0.0
    OperationConvolutionBwdFilterDy =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY as _,

    ///< Pointwise descriptor reference. @since cuDNN 9.0.0
    OperationPointwisePwDescriptor =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR as _,
    ///< First input tensor descriptor. @since cuDNN 9.0.0
    OperationPointwiseXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_XDESC as _,
    ///< Bias or second operand descriptor. @since cuDNN 9.0.0
    OperationPointwiseBDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_BDESC as _,
    ///< Output tensor descriptor. @since cuDNN 9.0.0
    OperationPointwiseYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_YDESC as _,
    ///< First scaling constant. @since cuDNN 9.0.0
    OperationPointwiseAlpha1 =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1 as _,
    ///< Second scaling constant. @since cuDNN 9.0.0
    OperationPointwiseAlpha2 =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2 as _,
    ///< Input gradient descriptor. @since cuDNN 9.0.0
    OperationPointwiseDxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_DXDESC as _,
    ///< Output gradient descriptor. @since cuDNN 9.0.0
    OperationPointwiseDyDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_DYDESC as _,
    ///< Intermediate tensor descriptor. @since cuDNN 9.0.0
    OperationPointwiseTDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_TDESC as _,

    ///< Statistics computation mode. @since cuDNN 9.0.0
    OperationGenStatsMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_GENSTATS_MODE as _,
    ///< Computation precision for statistics. @since cuDNN 9.0.0
    OperationGenStatsMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC as _,
    ///< Input data tensor descriptor. @since cuDNN 9.0.0
    OperationGenStatsXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_GENSTATS_XDESC as _,
    ///< Sum output descriptor. @since cuDNN 9.0.0
    OperationGenStatsSumDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC as _,
    ///< Sum of squares output descriptor. @since cuDNN 9.0.0
    OperationGenStatsSqSumDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC as _,

    ///< Training vs inference mode. @since cuDNN 9.0.0
    OperationBnFinalizeStatsMode =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE as _,
    ///< Computation precision. @since cuDNN 9.0.0
    OperationBnFinalizeMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC as _,
    ///< Sum of batch norm outputs. @since cuDNN 9.0.0
    OperationBnFinalizeYSumDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC as _,
    ///< Sum of squared outputs. @since cuDNN 9.0.0
    OperationBnFinalizeYSqSumDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC as _,
    ///< Batch norm scale parameter. @since cuDNN 9.0.0
    OperationBnFinalizeScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC as _,
    ///< Batch norm bias parameter. @since cuDNN 9.0.0
    OperationBnFinalizeBiasDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC as _,
    ///< Previous running mean. @since cuDNN 9.0.0
    OperationBnFinalizePrevRunningMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC as _,
    ///< Previous running variance. @since cuDNN 9.0.0
    OperationBnFinalizePrevRunningVarDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC as _,
    ///< Updated running mean output. @since cuDNN 9.0.0
    OperationBnFinalizeUpdatedRunningMeanDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC as _,
    ///< Updated running variance output. @since cuDNN 9.0.0
    OperationBnFinalizeUpdatedRunningVarDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC as _,
    ///< Cached mean for backward pass. @since cuDNN 9.0.0
    OperationBnFinalizeSavedMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC as _,
    ///< Cached inverse std dev for backward. @since cuDNN 9.0.0
    OperationBnFinalizeSavedInvStdDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC as _,
    ///< Equivalent scale for fused inference. @since cuDNN 9.0.0
    OperationBnFinalizeEqScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC as _,
    ///< Equivalent bias for fused inference. @since cuDNN 9.0.0
    OperationBnFinalizeEqBiasDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC as _,
    ///< Accumulation sample counter. @since cuDNN 9.0.0
    OperationBnFinalizeAccumCountDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC as _,
    ///< Numerical stability epsilon. @since cuDNN 9.0.0
    OperationBnFinalizeEpsilonDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC as _,
    ///< Exponential averaging factor. @since cuDNN 9.0.0
    OperationBnFinalizeExpAverateFactorDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC as _,

    ///< Associated cuDNN handle. @deprecated @since cuDNN 9.0.0
    OperationGraphHandle = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_HANDLE as _,
    ///< Array of operation descriptors in the graph. @since cuDNN 9.0.0
    OperationGraphOps = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_OPS as _,
    ///< Total number of engines available. @since cuDNN 9.0.0
    OperationGraphEngineGlobalCount =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT as _,
    ///< Dynamic shape support flag. @since cuDNN 9.4.0
    OperationGraphIsDynamicShapeEnabled =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED as _,
    ///< Same topology reuse flag. @since cuDNN 9.6.0
    OperationGraphIsSameTopology =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_IS_SAME_TOPOLOGY as _,
    ///< Dynamic shape support with execute time override @since cuDNN 9.21.0
    OperationGraphIsOverrideShapeEnabled =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED as _,
    ///< Memory alignment requirement in bytes. @since cuDNN 9.0.0
    TensorByteAlignment = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT as _,
    ///< Element data type. @since cuDNN 9.0.0
    TensorDataType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DATA_TYPE as _,
    ///< Dimension sizes array. @since cuDNN 9.0.0
    TensorDimensions = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DIMENSIONS as _,
    ///< Memory strides per dimension. @since cuDNN 9.0.0
    TensorStrides = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_STRIDES as _,
    ///< Vectorization element count. @since cuDNN 9.0.0
    TensorVectorCount = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_VECTOR_COUNT as _,
    ///< Which dimension is vectorized. @since cuDNN 9.0.0
    TensorVectorizedDimension =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION as _,
    ///< Unique identifier for graph connectivity. @since cuDNN 9.0.0
    TensorUniqueId = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_UNIQUE_ID as _,
    ///< Virtual (intermediate) vs I/O tensor flag. @since cuDNN 9.0.0
    TensorIsVirtual = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_IS_VIRTUAL as _,
    ///< Constant scalar vs device pointer flag. @since cuDNN 9.0.0
    TensorIsByValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_IS_BY_VALUE as _,
    ///< Memory layout transformation type. @since cuDNN 9.0.0
    TensorReorderingMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_REORDERING_MODE as _,
    ///< Ragged tensor offset descriptor. @since cuDNN 9.0.0
    TensorRaggedOffsetDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC as _,

    ///< Tensor UIDs in this variant pack. @since cuDNN 9.0.0
    VariantPackUniqueIds = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS as _,
    ///< GPU memory data pointers array. @since cuDNN 9.0.0
    VariantPackDataPointers =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS as _,
    ///< Intermediate tensor pointers. @since cuDNN 9.0.0
    VariantPackIntermediates =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES as _,
    ///< Workspace memory pointer. @since cuDNN 9.0.0
    VariantPackWorkspace = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_WORKSPACE as _,
    ///< Override tensor UIDs for dynamic shapes. @since cuDNN 9.18.0
    VariantPackOverrideUniqueIds =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS as _,
    ///< Override shapes for dynamic shapes. @since cuDNN 9.18.0
    VariantPackOverrideShapes =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES as _,
    ///< Override strides for dynamic shapes. @since cuDNN 9.18.0
    VariantPackOverrideStrides =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES as _,

    ///< Associated tensor identifier. @since cuDNN 9.0.0
    LayoutInfoTensorUid = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID as _,
    ///< Available memory layout types. @since cuDNN 9.0.0
    LayoutInfoTypes = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_LAYOUT_INFO_TYPES as _,

    ///< Knob type being described. @since cuDNN 9.0.0
    KnobInfoType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_INFO_TYPE as _,
    ///< Upper bound for knob value. @since cuDNN 9.0.0
    KnobInfoMaximumValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE as _,
    ///< Lower bound for knob value. @since cuDNN 9.0.0
    KnobInfoMinimumValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE as _,
    ///< Valid increment between knob values. @since cuDNN 9.0.0
    KnobInfoStride = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_INFO_STRIDE as _,

    ///< Operation graph this engine processes. @since cuDNN 9.0.0
    EngineOperationGraph = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_OPERATION_GRAPH as _,
    ///< Engine index in the global engine list. @since cuDNN 9.0.0
    EngineGlobalIndex = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_GLOBAL_INDEX as _,
    ///< Available knob configuration options. @since cuDNN 9.0.0
    EngineKnobInfo = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_KNOB_INFO as _,
    ///< Numerical properties (tensor cores, precision). @since cuDNN 9.0.0
    EngineNumericalNote = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_NUMERICAL_NOTE as _,
    ///< Preferred tensor memory layouts. @since cuDNN 9.0.0
    EngineLayoutInfo = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_LAYOUT_INFO as _,
    ///< Runtime behavior characteristics. @since cuDNN 9.0.0
    EngineBehaviorNote = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE as _,
    ///< Streaming multiprocessor target count. @since cuDNN 9.0.0
    EngineSmCountTarget = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_SM_COUNT_TARGET as _,
    ///< Device properties descriptor. @since cuDNN 9.8.0
    EngineDeviceProp = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_DEVICEPROP as _,

    ///< Computation precision type for matmul. @since cuDNN 9.0.0
    MatmulCompType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_MATMUL_COMP_TYPE as _,
    ///< Padding value for incomplete blocks. @since cuDNN 9.0.0
    MatmulPaddingValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_MATMUL_PADDING_VALUE as _,

    ///< First input matrix (A) tensor descriptor. @since cuDNN 9.0.0
    OperationMatmulADesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_ADESC as _,
    ///< Second input matrix (B) tensor descriptor. @since cuDNN 9.0.0
    OperationMatmulBDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_BDESC as _,
    ///< Output matrix (C) tensor descriptor. @since cuDNN 9.0.0
    OperationMatmulCDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_CDESC as _,
    ///< `MatMul` operation configuration descriptor. @since cuDNN 9.0.0
    OperationMatmulDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_DESC as _,
    ///< Irregular batch count. @deprecated @since cuDNN 9.0.0
    OperationMatmulIrregularlyStridedBatchCount = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT as _,
    ///< Override for output rows (M). @since cuDNN 9.0.0
    OperationMatmulGemmMOverrideDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC as _,
    ///< Override for output columns (N). @since cuDNN 9.0.0
    OperationMatmulGemmNOverrideDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC as _,
    ///< Override for contraction dim (K). @since cuDNN 9.0.0
    OperationMatmulGemmKOverrideDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC as _,

    ///< Reduction operation type (ADD, MUL, MIN, etc.). @since cuDNN 9.0.0
    ReductionOperator = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_OPERATOR as _,
    ///< Computation data type for reduction. @since cuDNN 9.0.0
    ReductionCompType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_COMP_TYPE as _,
    ///< Whether reduction must be deterministic. @since cuDNN 9.11.0
    ReductionIsDeterministic =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC as _,

    ///< Reduction input tensor descriptor. @since cuDNN 9.0.0
    OperationReductionXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_XDESC as _,
    ///< Reduction output tensor descriptor. @since cuDNN 9.0.0
    OperationReductionYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_YDESC as _,
    ///< Reduction descriptor reference. @since cuDNN 9.0.0
    OperationReductionDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_DESC as _,

    ///< Computation precision. @since cuDNN 9.0.0
    OperationBnBwdWeightsMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC as _,
    ///< Cached batch mean from forward. @since cuDNN 9.0.0
    OperationBnBwdWeightsMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC as _,
    ///< Cached inverse std dev from forward. @since cuDNN 9.0.0
    OperationBnBwdWeightsInvStdDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC as _,
    ///< Batch norm scale parameter. @since cuDNN 9.0.0
    OperationBnBwdWeightsBnScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC as _,
    ///< Forward input tensor. @since cuDNN 9.0.0
    OperationBnBwdWeightsXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC as _,
    ///< Output gradient tensor. @since cuDNN 9.0.0
    OperationBnBwdWeightsDyDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC as _,
    ///< Scale gradient output. @since cuDNN 9.0.0
    OperationBnBwdWeightsDbnScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC as _,
    ///< Bias gradient output. @since cuDNN 9.0.0
    OperationBnBwdWeightsDbnBiasDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC as _,
    ///< Equivalent output gradient scale. @since cuDNN 9.0.0
    OperationBnBwdWeightsEqDyScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC as _,
    ///< Equivalent input scale. @since cuDNN 9.0.0
    OperationBnBwdWeightsEqXScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC as _,
    ///< Equivalent bias value. @since cuDNN 9.0.0
    OperationBnBwdWeightsEqBias =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS as _,

    ///< Resampling interpolation method. @since cuDNN 9.0.0
    ResampleMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_MODE as _,
    ///< Computation precision for resampling. @since cuDNN 9.0.0
    ResampleCompType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_COMP_TYPE as _,
    ///< Number of spatial dimensions. @since cuDNN 9.0.0
    ResampleSpatialDims = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS as _,
    ///< Post-paddings per spatial dimension. @since cuDNN 9.0.0
    ResamplePostPaddings = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_POST_PADDINGS as _,
    ///< Pre-paddings per spatial dimension. @since cuDNN 9.0.0
    ResamplePrePaddings = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_PRE_PADDINGS as _,
    ///< Stride factors per spatial dimension. @since cuDNN 9.0.0
    ResampleStrides = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_STRIDES as _,
    ///< Filter window sizes per dimension. @since cuDNN 9.0.0
    ResampleWindowDims = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_WINDOW_DIMS as _,
    ///< NaN handling behavior. @since cuDNN 9.0.0
    ResampleNanPropagation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION as _,
    ///< Padding strategy (zero, `neg_inf`, edge). @since cuDNN 9.0.0
    ResamplePaddingMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_PADDING_MODE as _,

    ///< Forward resample input tensor. @since cuDNN 9.0.0
    OperationResampleFwdXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC as _,
    ///< Forward resample output tensor. @since cuDNN 9.0.0
    OperationResampleFwdYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC as _,
    ///< Max pooling index tensor. @since cuDNN 9.0.0
    OperationResampleFwdIdxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC as _,
    ///< Output scaling factor. @deprecated @since cuDNN 9.0.0
    OperationResampleFwdAlpha =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA as _,
    ///< Accumulation scaling factor. @deprecated @since cuDNN 9.0.0
    OperationResampleFwdBeta =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA as _,
    ///< Resample descriptor reference. @since cuDNN 9.0.0
    OperationResampleFwdDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC as _,

    ///< Backward resample input gradient. @since cuDNN 9.0.0
    OperationResampleBwdDxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC as _,
    ///< Backward resample output gradient. @since cuDNN 9.0.0
    OperationResampleBwdDyDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC as _,
    ///< Index tensor from forward pass. @since cuDNN 9.0.0
    OperationResampleBwdIdxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC as _,
    ///< Gradient scaling factor. @deprecated @since cuDNN 9.0.0
    OperationResampleBwdAlpha =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA as _,
    ///< Accumulation scaling. @deprecated @since cuDNN 9.0.0
    OperationResampleBwdBeta =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA as _,
    ///< Resample descriptor reference. @since cuDNN 9.0.0
    OperationResampleBwdDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC as _,
    ///< Forward input reference. @since cuDNN 9.0.0
    OperationResampleBwdXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC as _,
    ///< Forward output reference. @since cuDNN 9.0.0
    OperationResampleBwdYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC as _,

    ///< Concatenation axis dimension. @since cuDNN 9.0.0
    OperationConcatAxis = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONCAT_AXIS as _,
    ///< Array of input tensor descriptors. @since cuDNN 9.0.0
    OperationConcatInputDescs =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS as _,
    ///< In-place output tensor selection index. @since cuDNN 9.0.0
    OperationConcatInplaceIndex =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX as _,
    ///< Concatenated output descriptor. @since cuDNN 9.0.0
    OperationConcatOutputDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC as _,

    ///< Signal set vs wait mode. @since cuDNN 9.0.0
    OperationSignalMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SIGNAL_MODE as _,
    ///< Flag variable tensor descriptor. @since cuDNN 9.0.0
    OperationSignalFlagDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC as _,
    ///< Signal value for comparison. @since cuDNN 9.0.0
    OperationSignalValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SIGNAL_VALUE as _,
    ///< Input tensor for signal set. @since cuDNN 9.0.0
    OperationSignalXDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SIGNAL_XDESC as _,
    ///< Output tensor descriptor. @since cuDNN 9.0.0
    OperationSignalYDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SIGNAL_YDESC as _,

    ///< Cache container descriptor. @since cuDNN 9.4.0
    OperationPagedCacheLoadContainerDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC as _,
    ///< Output tensor descriptor. @since cuDNN 9.4.0
    OperationPagedCacheLoadYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC as _,
    ///< Load sequence specification. @since cuDNN 9.4.0
    OperationPagedCacheLoadSequenceDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC as _,
    ///< Page table mapping descriptor. @since cuDNN 9.4.0
    OperationPagedCacheLoadPageTableDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC as _,

    ///< Normalization algorithm type. @since cuDNN 9.0.0
    OperationNormFwdMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_MODE as _,
    ///< Training vs inference phase. @since cuDNN 9.0.0
    OperationNormFwdPhase = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_PHASE as _,
    ///< Input tensor descriptor. @since cuDNN 9.0.0
    OperationNormFwdXDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_XDESC as _,
    ///< Computed or cached mean tensor. @since cuDNN 9.0.0
    OperationNormFwdMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC as _,
    ///< Computed or cached inverse variance. @since cuDNN 9.0.0
    OperationNormFwdInvVarianceDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC as _,
    ///< Learnable scale parameter (gamma). @since cuDNN 9.0.0
    OperationNormFwdScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC as _,
    ///< Learnable bias parameter (beta). @since cuDNN 9.0.0
    OperationNormFwdBiasDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC as _,
    ///< Numerical stability constant. @since cuDNN 9.0.0
    OperationNormFwdEpsilonDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC as _,
    ///< Momentum for running statistics. @since cuDNN 9.0.0
    OperationNormFwdExpAvgFactorDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC as _,
    ///< Input running mean. @since cuDNN 9.0.0
    OperationNormFwdInputRunningMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC as _,
    ///< Input running variance. @since cuDNN 9.0.0
    OperationNormFwdInputRunningVarDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC as _,
    ///< Updated running mean output. @since cuDNN 9.0.0
    OperationNormFwdOutputRunningMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC as _,
    ///< Updated running variance output. @since cuDNN 9.0.0
    OperationNormFwdOutputRunningVarDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC as _,
    ///< Output tensor descriptor. @since cuDNN 9.0.0
    OperationNormFwdYDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_YDESC as _,
    ///< Peer statistics for multi-GPU sync. @since cuDNN 9.0.0
    OperationNormFwdPeerStatDescs =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS as _,

    ///< Normalization algorithm type. @since cuDNN 9.0.0
    OperationNormBwdMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_MODE as _,
    ///< Forward input tensor reference. @since cuDNN 9.0.0
    OperationNormBwdXDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_XDESC as _,
    ///< Cached mean from forward pass. @since cuDNN 9.0.0
    OperationNormBwdMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC as _,
    ///< Cached inverse std dev. @since cuDNN 9.0.0
    OperationNormBwdInvVarianceDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC as _,
    ///< Output gradient tensor. @since cuDNN 9.0.0
    OperationNormBwdDyDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC as _,
    ///< Forward scale parameter. @since cuDNN 9.0.0
    OperationNormBwdScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC as _,
    ///< Numerical stability epsilon. @since cuDNN 9.0.0
    OperationNormBwdEpsilonDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC as _,
    ///< Scale gradient output. @since cuDNN 9.0.0
    OperationNormBwdDScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC as _,
    ///< Bias gradient output. @since cuDNN 9.0.0
    OperationNormBwdDBiasDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC as _,
    ///< Input gradient output tensor. @since cuDNN 9.0.0
    OperationNormBwdDxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC as _,
    ///< Peer gradient statistics for multi-GPU. @since cuDNN 9.0.0
    OperationNormBwdPeerStatDescs =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS as _,

    ///< Reshape input tensor descriptor. @since cuDNN 9.0.0
    OperationReshapeXDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESHAPE_XDESC as _,
    ///< Reshape output tensor descriptor. @since cuDNN 9.0.0
    OperationReshapeYDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESHAPE_YDESC as _,

    ///< Band matrix input tensor. @since cuDNN 9.10.0
    OperationExpandBandMatrixXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_XDESC as _,
    ///< Expanded output tensor. @since cuDNN 9.10.0
    OperationExpandBandMatrixYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_YDESC as _,
    ///< Lower bandwidth of the band. @since cuDNN 9.10.0
    OperationExpandBandMatrixLowerBandwidth =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_LOWER_BANDWIDTH as _,
    ///< Upper bandwidth of the band. @since cuDNN 9.10.0
    OperationExpandBandMatrixUpperBandwidth =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_UPPER_BANDWIDTH as _,
    ///< Axis along which to expand. @since cuDNN 9.10.0
    OperationExpandBandMatrixAxis =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_AXIS as _,
    ///< Padding value outside the band. @since cuDNN 9.10.0
    OperationExpandBandMatrixPadValue =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_PAD_VALUE as _,
    ///< KV token offset descriptor. @since cuDNN 9.10.0
    OperationExpandBandMatrixKvTokenOffsetDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_KV_TOKEN_OFFSET_DESC as _,
    ///< Speculative decoding mask. @since cuDNN 9.13.0
    OperationExpandBandMatrixSpeculativeMaskDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_SPECULATIVE_MASK_DESC as _,

    ///< Full matrix input tensor. @since cuDNN 9.10.0
    OperationContractBandMatrixXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_XDESC as _,
    ///< Contracted band output tensor. @since cuDNN 9.10.0
    OperationContractBandMatrixYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_YDESC as _,
    ///< Lower bandwidth. @since cuDNN 9.10.0
    OperationContractBandMatrixLowerBandwidth =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_LOWER_BANDWIDTH as _,
    ///< Upper bandwidth. @since cuDNN 9.10.0
    OperationContractBandMatrixUpperBandwidth =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_UPPER_BANDWIDTH as _,
    ///< Axis along which to contract. @since cuDNN 9.10.0
    OperationContractBandMatrixAxis =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_AXIS as _,
    ///< Padding value. @since cuDNN 9.10.0
    OperationContractBandMatrixPadValue =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_PAD_VALUE as _,
    ///< Maximum token value for contraction. @since cuDNN 9.10.0
    OperationContractBandMaxTokenValue =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MAX_TOKEN_VALUE as _,

    ///< Random distribution type selection. @since cuDNN 9.0.0
    RngDistribution = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_DISTRIBUTION as _,
    ///< Normal distribution mean parameter. @since cuDNN 9.0.0
    RngNormalDistMean = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_NORMAL_DIST_MEAN as _,
    ///< Normal distribution std deviation. @since cuDNN 9.0.0
    RngNormalDistStandardDeviation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION as _,
    ///< Uniform distribution upper bound. @since cuDNN 9.0.0
    RngUniformDistMaximum = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM as _,
    ///< Uniform distribution lower bound. @since cuDNN 9.0.0
    RngUniformDistMinimum = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM as _,
    ///< Bernoulli probability of 1. @since cuDNN 9.0.0
    RngBernoulliDistProbability =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY as _,

    ///< RNG output tensor descriptor. @since cuDNN 9.0.0
    OperationRngYDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RNG_YDESC as _,
    ///< RNG seed value. @since cuDNN 9.0.0
    OperationRngSeed = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RNG_SEED as _,
    ///< RNG descriptor reference. @since cuDNN 9.0.0
    OperationRngDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RNG_DESC as _,
    ///< RNG offset/state descriptor. @since cuDNN 9.0.0
    OperationRngOffsetDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC as _,

    ///< Whether kernel is cached. @since cuDNN 9.4.0
    KernelCacheIsEngineCfgKernelCached =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED as _,
    ///< Operation graph for kernel cache. @since cuDNN 9.5.0
    KernelCacheOperationGraph =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KERNEL_CACHE_OPERATION_GRAPH as _,
    ///< Kernel cache JSON serialization. @since cuDNN 9.10.0
    KernelCacheJsonRepresentation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION as _,
    ///< Input float tensor to quantize. @since cuDNN 9.7.0
    OperationBlockScaleQuantizeXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC as _,
    ///< Quantized output tensor. @since cuDNN 9.7.0
    OperationBlockScaleQuantizeYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC as _,
    ///< Per-block scaling factors output. @since cuDNN 9.7.0
    OperationBlockScaleQuantizeScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC as _,
    ///< Computation precision. @since cuDNN 9.7.0
    OperationBlockScaleQuantizeMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC as _,
    ///< Quantization block size. @since cuDNN 9.7.0
    OperationBlockScaleQuantizeBlockSize =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE as _,

    ///< Quantized input tensor. @since cuDNN 9.7.0
    OperationBlockScaleDequantizeXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC as _,
    ///< Per-block scale factors. @since cuDNN 9.7.0
    OperationBlockScaleDequantizeScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC as _,
    ///< Dequantized output tensor. @since cuDNN 9.7.0
    OperationBlockScaleDequantizeYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC as _,
    ///< Computation precision. @since cuDNN 9.7.0
    OperationBlockScaleDequantizeMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC as _,
    ///< Dequantization block size. @since cuDNN 9.7.0
    OperationBlockScaleDequantizeBlockSize =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE as _,
    ///< Negative scale handling. @since cuDNN 9.13.0
    OperationBlockScaleDequantizeNegScale =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE as _,

    ///< Query cumulative sequence length tensor. @since cuDNN 9.24.0
    OperationSdpaFwdCuSeqLenQDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_QDESC as _,
    ///< Key-value cumulative sequence length tensor. @since cuDNN 9.24.0
    OperationSdpaFwdCuSeqLenKvDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_KVDESC as _,
    ///< Query cumulative sequence length tensor. @since cuDNN 9.24.0
    OperationDiagonalBandMaskCuSeqLenQDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_QDESC as _,
    ///< Key-value cumulative sequence length tensor. @since cuDNN 9.24.0
    OperationDiagonalBandMaskCuSeqLenKvDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_KVDESC as _,
    ///< FP8 descale factor for Q. @since cuDNN 9.25.0
    OperationSdpaFwdDescaleQDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_QDESC as _,
    ///< FP8 descale factor for K. @since cuDNN 9.25.0
    OperationSdpaFwdDescaleKDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_KDESC as _,
    ///< FP8 descale factor for V. @since cuDNN 9.25.0
    OperationSdpaFwdDescaleVDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_VDESC as _,
    ///< FP8 descale factor for S. @since cuDNN 9.25.0
    OperationSdpaFwdDescaleSDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_SDESC as _,
    ///< FP8 scale factor for S. @since cuDNN 9.25.0
    OperationSdpaFwdScaleSDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_SDESC as _,
    ///< FP8 scale factor for O. @since cuDNN 9.25.0
    OperationSdpaFwdScaleODesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_ODESC as _,
    ///< FP8 absolute maximum of S. @since cuDNN 9.25.0
    OperationSdpaFwdAmaxSDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_SDESC as _,
    ///< FP8 absolute maximum of O. @since cuDNN 9.25.0
    OperationSdpaFwdAmaxODesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_ODESC as _,
    ///< Query tensor descriptor. @since cuDNN 9.13.0
    OperationSdpaFwdQDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC as _,
    ///< Key tensor descriptor. @since cuDNN 9.13.0
    OperationSdpaFwdKDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC as _,
    ///< Value tensor descriptor. @since cuDNN 9.13.0
    OperationSdpaFwdVDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC as _,
    ///< Output tensor descriptor. @since cuDNN 9.13.0
    OperationSdpaFwdODesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC as _,
    ///< Statistics output descriptor. @since cuDNN 9.13.0
    OperationSdpaFwdStatsDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC as _,
    ///< Attention scaling factor descriptor. @since cuDNN 9.13.0
    OperationSdpaFwdScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC as _,
    ///< Block-sparse attention mask. @since cuDNN 9.14.0
    OperationSdpaFwdBlockMaskDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC as _,
    ///< Paged attention key page table. @since cuDNN 9.15.0
    OperationSdpaFwdPageTableKDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC as _,
    ///< Paged attention value page table. @since cuDNN 9.15.0
    OperationSdpaFwdPageTableVDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC as _,
    ///< Query sequence length tensor. @since cuDNN 9.15.0
    OperationSdpaFwdSeqLenQDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC as _,
    ///< Key-value sequence length tensor. @since cuDNN 9.15.0
    OperationSdpaFwdSeqLenKvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC as _,
    ///< Softmax descriptor. @since cuDNN 9.21.0
    OperationSdpaFwdSoftmaxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SOFTMAX_DESC as _,
    ///< Forward SDPA subgraph. @since cuDNN 9.21.0
    OperationSdpaFwdSubgraph =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH as _,
    ///< Subgraph input tensor UID. @since cuDNN 9.21.0
    OperationSdpaFwdSubgraphInputId =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID as _,
    ///< Subgraph output tensor UID. @since cuDNN 9.21.0
    OperationSdpaFwdSubgraphOutputId =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID as _,
    ///< Dropout seed tensor descriptor. @since cuDNN 9.21.0
    OperationSdpaFwdDropoutSeedDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SEED_DESC as _,
    ///< Dropout offset tensor descriptor. @since cuDNN 9.21.0
    OperationSdpaFwdDropoutOffsetDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_OFFSET_DESC as _,
    ///< Dropout RNG dump tensor descriptor. @since cuDNN 9.21.0
    OperationSdpaFwdDropoutRngDumpDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_RNG_DUMP_DESC as _,
    ///< Dropout probability. @since cuDNN 9.21.0
    OperationSdpaFwdDropoutProbability =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_PROBABILITY as _,
    ///< Unfuse FMA in softmax for SM100. @since cuDNN 9.22.0
    OperationSdpaFwdUnfuseFma =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_UNFUSE_FMA as _,
    ///< Query tensor descriptor. @since cuDNN 9.17.0
    OperationSdpaBwdQDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC as _,
    ///< Key tensor descriptor. @since cuDNN 9.17.0
    OperationSdpaBwdKDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC as _,
    ///< Value tensor descriptor. @since cuDNN 9.17.0
    OperationSdpaBwdVDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC as _,
    ///< Forward output tensor descriptor. @since cuDNN 9.17.0
    OperationSdpaBwdODesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC as _,
    ///< Forward statistics descriptor. @since cuDNN 9.17.0
    OperationSdpaBwdStatsDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC as _,
    ///< Attention scaling factor. @since cuDNN 9.17.0
    OperationSdpaBwdScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC as _,
    ///< Query sequence length tensor. @since cuDNN 9.17.0
    OperationSdpaBwdSeqLenQDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC as _,
    ///< Key-value sequence length tensor. @since cuDNN 9.17.0
    OperationSdpaBwdSeqLenKvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC as _,
    ///< Query gradient output tensor. @since cuDNN 9.17.0
    OperationSdpaBwdDqDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC as _,
    ///< Key gradient output tensor. @since cuDNN 9.17.0
    OperationSdpaBwdDkDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC as _,
    ///< Value gradient output tensor. @since cuDNN 9.17.0
    OperationSdpaBwdDvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC as _,
    ///< Output gradient input tensor. @since cuDNN 9.17.0
    OperationSdpaBwdDoDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC as _,
    ///< Backward sink descriptor. @since UNPUBLISHED
    OperationSdpaBwdSinkDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SINK_DESC as _,
    ///< Backward sink gradient descriptor. @since UNPUBLISHED
    OperationSdpaBwdDsinkDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_DSINK_DESC as _,
    ///< Max total query sequence length. @since UNPUBLISHED
    OperationSdpaBwdMaxTotalSeqLenQ =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_Q as _,
    ///< Max total KV sequence length. @since UNPUBLISHED
    OperationSdpaBwdMaxTotalSeqLenKv =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_KV as _,
    ///< Backward SDPA subgraph. @since UNPUBLISHED
    OperationSdpaBwdSubgraph =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH as _,
    ///< Subgraph input tensor UID. @since UNPUBLISHED
    OperationSdpaBwdSubgraphInputId =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_INPUT_UID as _,
    ///< Subgraph output tensor UID. @since UNPUBLISHED
    OperationSdpaBwdSubgraphOutputId =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_OUTPUT_UID as _,
    ///< Gather/scatter mode for `MoE`. @since cuDNN 9.15.0
    OperationMoeGroupedMatmulMode =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE as _,
    ///< Computation precision. @since cuDNN 9.15.0
    OperationMoeGroupedMatmulMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC as _,
    ///< Token tensor descriptor. @since cuDNN 9.15.0
    OperationMoeGroupedMatmulTokenDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC as _,
    ///< Expert weight tensor descriptor. @since cuDNN 9.15.0
    OperationMoeGroupedMatmulWeightDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC as _,
    ///< First token offset per expert. @since cuDNN 9.15.0
    OperationMoeGroupedMatmulFirstTokenOffsetDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC as _,
    ///< Output tensor descriptor. @since cuDNN 9.15.0
    OperationMoeGroupedMatmulOutputDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC as _,
    ///< Token routing index descriptor. @since cuDNN 9.15.0
    OperationMoeGroupedMatmulTokenIndexDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC as _,
    ///< Token routing weights descriptor. @since cuDNN 9.15.0
    OperationMoeGroupedMatmulTokenKsDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC as _,
    ///< Top-K experts per token. @since cuDNN 9.15.0
    OperationMoeGroupedMatmulTopK =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K as _,
    ///< Input tensor descriptor. @since cuDNN 9.20.0
    OperationDiagonalBandMaskXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC as _,
    ///< KV sequence length tensor. @since cuDNN 9.20.0
    OperationDiagonalBandMaskSeqLenKvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC as _,
    ///< Query sequence length tensor. @since cuDNN 9.20.0
    OperationDiagonalBandMaskSeqLenQDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC as _,
    ///< Left bound offset descriptor. @since cuDNN 9.20.0
    OperationDiagonalBandMaskLeftBoundDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC as _,
    ///< Right bound shift descriptor. @since cuDNN 9.20.0
    OperationDiagonalBandMaskShiftRightBoundDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC as _,
    ///< Band descriptor. @since cuDNN 9.20.0
    OperationDiagonalBandMaskBDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC as _,
    ///< Output mask tensor descriptor. @since cuDNN 9.20.0
    OperationDiagonalBandMaskYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC as _,
    ///< Mask comparison mode. @since cuDNN 9.20.0
    OperationDiagonalBandMaskComparisonMode =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE as _,
    ///< Softmax input tensor descriptor. @since cuDNN 9.20.0
    OperationSoftmaxXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_XDESC as _,
    ///< Softmax output tensor descriptor. @since cuDNN 9.20.0
    OperationSoftmaxYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_YDESC as _,
    ///< Softmax statistics output. @since cuDNN 9.20.0
    OperationSoftmaxStatsDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC as _,
    ///< Row-wise max values descriptor. @since cuDNN 9.20.0
    OperationSoftmaxMaxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_MAX_DESC as _,
    ///< Row-wise sum of exponentials. @since cuDNN 9.20.0
    OperationSoftmaxSumExpDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_SUM_EXP_DESC as _,
    ///< Softmax sink descriptor. @since cuDNN 9.20.0
    OperationSoftmaxSinkDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_SINK_DESC as _,

    ///< CUDA device identifier. @since cuDNN 9.8.0
    DevicePropDeviceId = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_DEVICEPROP_DEVICE_ID as _,
    ///< Associated cuDNN handle. @since cuDNN 9.8.0
    DevicePropHandle = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_DEVICEPROP_HANDLE as _,
    ///< Device properties JSON. @since cuDNN 9.8.0
    DevicePropJsonRepresentation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION as _,
}

impl_enum_conversion!(sys::cudnnBackendAttributeName_t, BackendAttributeName);

/// Native attribute ID reported by runtime metadata, including IDs absent from this crate's headers.
/// Descriptor attribute keys still use the checked [`BackendAttributeName`] enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct BackendAttributeId(u32);

impl BackendAttributeId {
    /// Native numeric ID without requiring a matching generated Rust enum variant.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl From<u32> for BackendAttributeId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<BackendAttributeName> for BackendAttributeId {
    fn from(value: BackendAttributeName) -> Self {
        Self(value.into())
    }
}

impl From<BackendAttributeId> for u32 {
    fn from(value: BackendAttributeId) -> Self {
        value.raw()
    }
}

impl TryFrom<BackendAttributeId> for BackendAttributeName {
    type Error = BackendAttributeId;

    fn try_from(value: BackendAttributeId) -> Result<Self, Self::Error> {
        Self::try_from(value.raw()).map_err(|_| value)
    }
}

impl fmt::Display for BackendAttributeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match BackendAttributeName::try_from(*self) {
            Ok(name) => fmt::Display::fmt(&name, formatter),
            Err(id) => write!(formatter, "backend attribute {}", id.raw()),
        }
    }
}

impl_enum_display!(BackendAttributeName, {
    Self::OperationReductionGroupOffsetDesc => "CUDNN_ATTR_OPERATION_REDUCTION_GROUP_OFFSET_DESC",
    Self::OperationMoeGroupedMatmulBackwardMathPrec => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC",
    Self::OperationMoeGroupedMatmulBackwardTokenDesc => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC",
    Self::OperationMoeGroupedMatmulBackwardWeightGradientDesc => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC",
    Self::OperationMoeGroupedMatmulBackwardFirstTokenOffsetDesc => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC",
    Self::OperationMoeGroupedMatmulBackwardOutputGradientDesc => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC",

    Self::OperationRopeForwardInputDesc => "CUDNN_ATTR_OPERATION_ROPE_FWD_XDESC",
    Self::OperationRopeForwardOutputDesc => "CUDNN_ATTR_OPERATION_ROPE_FWD_YDESC",
    Self::OperationRopeForwardFrequenciesDesc => "CUDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC",
    Self::OperationRopeForwardOutputScale => "CUDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE",
    Self::OperationRopeForwardRotationDimension => "CUDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM",
    Self::OperationRopeBackwardInputDesc => "CUDNN_ATTR_OPERATION_ROPE_BWD_DYDESC",
    Self::OperationRopeBackwardOutputDesc => "CUDNN_ATTR_OPERATION_ROPE_BWD_DXDESC",
    Self::OperationRopeBackwardFrequenciesDesc => "CUDNN_ATTR_OPERATION_ROPE_BWD_FREQSDESC",
    Self::OperationRopeBackwardOutputScale => "CUDNN_ATTR_OPERATION_ROPE_BWD_OUTPUT_SCALE",
    Self::OperationRopeBackwardRotationDimension => "CUDNN_ATTR_OPERATION_ROPE_BWD_ROPE_DIM",

    Self::TensorConstantValue => "CUDNN_ATTR_TENSOR_CONSTANT_VALUE",
    Self::TensorRaggedOffsetMultiplier => "CUDNN_ATTR_TENSOR_RAGGED_OFFSET_MULTIPLIER",
    Self::OperationReshapeMode => "CUDNN_ATTR_OPERATION_RESHAPE_MODE",
    Self::OperationSliceXDesc => "CUDNN_ATTR_OPERATION_SLICE_XDESC",
    Self::OperationSliceYDesc => "CUDNN_ATTR_OPERATION_SLICE_YDESC",
    Self::OperationSliceStartIndices => "CUDNN_ATTR_OPERATION_SLICE_START_INDICES",
    Self::OperationSliceLimitIndices => "CUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES",
    Self::OperationSliceStrides => "CUDNN_ATTR_OPERATION_SLICE_STRIDES",
    Self::OperationTransposeXDesc => "CUDNN_ATTR_OPERATION_TRANSPOSE_XDESC",
    Self::OperationTransposeYDesc => "CUDNN_ATTR_OPERATION_TRANSPOSE_YDESC",
    Self::OperationTransposePermutation => "CUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION",
    Self::EngineHeurSharedMemoryLimit => "CUDNN_ATTR_ENGINEHEUR_SHARED_MEMORY_LIMIT",
    Self::EngineSharedMemoryLimit => "CUDNN_ATTR_ENGINE_SHARED_MEMORY_LIMIT",
    Self::KernelCacheRevision => "CUDNN_ATTR_KERNEL_CACHE_REVISION",
    Self::KernelCacheEntryCount => "CUDNN_ATTR_KERNEL_CACHE_ENTRY_COUNT",
    Self::OperationSoftmaxStatsLog2 => "CUDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2",
    Self::OperationReductionDims => "CUDNN_ATTR_OPERATION_REDUCTION_DIMS",
    Self::OperationReductionInitValues => "CUDNN_ATTR_OPERATION_REDUCTION_INIT_VALUES",
    Self::OperationReductionSubgraph => "CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH",
    Self::OperationReductionSubgraphAccumulatorUids => "CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_ACCUMULATOR_UIDS",
    Self::OperationReductionSubgraphCurrentUids => "CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_CURRENT_UIDS",
    Self::OperationReductionSubgraphOutputUids => "CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_OUTPUT_UIDS",
    Self::OperationReductionSubgraphComputeTypes => "CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_COMPUTE_TYPES",

    Self::PointwiseMode => "CUDNN_ATTR_POINTWISE_MODE",
    Self::PointwiseMathPrec => "CUDNN_ATTR_POINTWISE_MATH_PREC",
    Self::PointwiseNanPropagation => "CUDNN_ATTR_POINTWISE_NAN_PROPAGATION",
    Self::PointwiseReluLowerClip => "CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP",
    Self::PointwiseReluUpperClip => "CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP",
    Self::PointwiseReluLowerClipSlope => "CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE",
    Self::PointwiseEluAlpha => "CUDNN_ATTR_POINTWISE_ELU_ALPHA",
    Self::PointwiseSoftplusBeta => "CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA",
    Self::PointwiseSwishBeta => "CUDNN_ATTR_POINTWISE_SWISH_BETA",
    Self::PointwiseAxis => "CUDNN_ATTR_POINTWISE_AXIS",
    Self::ConvolutionCompType => "CUDNN_ATTR_CONVOLUTION_COMP_TYPE",
    Self::ConvolutionConvMode => "CUDNN_ATTR_CONVOLUTION_CONV_MODE",
    Self::ConvolutionDilations => "CUDNN_ATTR_CONVOLUTION_DILATIONS",
    Self::ConvolutionFilterStrides => "CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES",
    Self::ConvolutionPostPaddings => "CUDNN_ATTR_CONVOLUTION_POST_PADDINGS",
    Self::ConvolutionPrePaddings => "CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS",
    Self::ConvolutionSpatialDims => "CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS",
    Self::EngineHeurMode => "CUDNN_ATTR_ENGINEHEUR_MODE",
    Self::EngineHeurOperationGraph => "CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH",
    Self::EngineHeurResults => "CUDNN_ATTR_ENGINEHEUR_RESULTS",
    Self::EngineHeurSmCountTarget => "CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET",
    Self::EngineHeurDeviceProp => "CUDNN_ATTR_ENGINEHEUR_DEVICEPROP",
    Self::EngineCfgEngine => "CUDNN_ATTR_ENGINECFG_ENGINE",
    Self::EngineCfgIntermediateInfo => "CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO",
    Self::EngineCfgKnobChoices => "CUDNN_ATTR_ENGINECFG_KNOB_CHOICES",
    Self::EngineCfgWorkspaceSize => "CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE",
    Self::EngineCfgSharedMemoryUsed => "CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED",
    Self::ExecutionPlanHandle => "CUDNN_ATTR_EXECUTION_PLAN_HANDLE",
    Self::ExecutionPlanEngineConfig => "CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG",
    Self::ExecutionPlanWorkspaceSize => "CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE",
    Self::ExecutionPlanComputedIntermediateIds => "CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS",
    Self::ExecutionPlanRunOnlyIntermediateIds => "CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS",
    Self::ExecutionPlanJsonRepresentation => "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION",
    Self::ExecutionPlanKernelCache => "CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE",
    Self::ExecutionPlanDeviceProp => "CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP",
    Self::IntermediateInfoUniqueId => "CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID",
    Self::IntermediateInfoSize => "CUDNN_ATTR_INTERMEDIATE_INFO_SIZE",
    Self::IntermediateInfoDependentTensorUids => "CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS",
    Self::IntermediateInfoDependentAttributes => "CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES",
    Self::KnobChoiceKnobType => "CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE",
    Self::KnobChoiceKnobValue => "CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE",
    Self::OperationConvolutionForwardAlpha => "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
    Self::OperationConvolutionForwardBeta => "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
    Self::OperationConvolutionForwardConvDesc => "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
    Self::OperationConvolutionForwardW => "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
    Self::OperationConvolutionForwardX => "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
    Self::OperationConvolutionForwardY => "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
    Self::OperationConvolutionBwdDataAlpha => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
    Self::OperationConvolutionBwdDataBeta => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
    Self::OperationConvolutionBwdDataConvDesc => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
    Self::OperationConvolutionBwdDataW => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
    Self::OperationConvolutionBwdDataDx => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
    Self::OperationConvolutionBwdDataDy => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
    Self::OperationConvolutionBwdFilterAlpha => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
    Self::OperationConvolutionBwdFilterBeta => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
    Self::OperationConvolutionBwdFilterConvDesc => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
    Self::OperationConvolutionBwdFilterDw => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
    Self::OperationConvolutionBwdFilterX => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
    Self::OperationConvolutionBwdFilterDy => "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
    Self::OperationPointwisePwDescriptor => "CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR",
    Self::OperationPointwiseXDesc => "CUDNN_ATTR_OPERATION_POINTWISE_XDESC",
    Self::OperationPointwiseBDesc => "CUDNN_ATTR_OPERATION_POINTWISE_BDESC",
    Self::OperationPointwiseYDesc => "CUDNN_ATTR_OPERATION_POINTWISE_YDESC",
    Self::OperationPointwiseAlpha1 => "CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1",
    Self::OperationPointwiseAlpha2 => "CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2",
    Self::OperationPointwiseDxDesc => "CUDNN_ATTR_OPERATION_POINTWISE_DXDESC",
    Self::OperationPointwiseDyDesc => "CUDNN_ATTR_OPERATION_POINTWISE_DYDESC",
    Self::OperationPointwiseTDesc => "CUDNN_ATTR_OPERATION_POINTWISE_TDESC",
    Self::OperationGenStatsMode => "CUDNN_ATTR_OPERATION_GENSTATS_MODE",
    Self::OperationGenStatsMathPrec => "CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC",
    Self::OperationGenStatsXDesc => "CUDNN_ATTR_OPERATION_GENSTATS_XDESC",
    Self::OperationGenStatsSumDesc => "CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC",
    Self::OperationGenStatsSqSumDesc => "CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC",
    Self::OperationBnFinalizeStatsMode => "CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE",
    Self::OperationBnFinalizeMathPrec => "CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC",
    Self::OperationBnFinalizeYSumDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC",
    Self::OperationBnFinalizeYSqSumDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC",
    Self::OperationBnFinalizeScaleDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC",
    Self::OperationBnFinalizeBiasDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC",
    Self::OperationBnFinalizePrevRunningMeanDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC",
    Self::OperationBnFinalizePrevRunningVarDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC",
    Self::OperationBnFinalizeUpdatedRunningMeanDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC",
    Self::OperationBnFinalizeUpdatedRunningVarDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC",
    Self::OperationBnFinalizeSavedMeanDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC",
    Self::OperationBnFinalizeSavedInvStdDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC",
    Self::OperationBnFinalizeEqScaleDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC",
    Self::OperationBnFinalizeEqBiasDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC",
    Self::OperationBnFinalizeAccumCountDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC",
    Self::OperationBnFinalizeEpsilonDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC",
    Self::OperationBnFinalizeExpAverateFactorDesc => "CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC",
    Self::OperationGraphHandle => "CUDNN_ATTR_OPERATIONGRAPH_HANDLE",
    Self::OperationGraphOps => "CUDNN_ATTR_OPERATIONGRAPH_OPS",
    Self::OperationGraphEngineGlobalCount => "CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT",
    Self::OperationGraphIsDynamicShapeEnabled => "CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED",
    Self::OperationGraphIsSameTopology => "CUDNN_ATTR_OPERATIONGRAPH_IS_SAME_TOPOLOGY",
    Self::OperationGraphIsOverrideShapeEnabled => "CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED",
    Self::TensorByteAlignment => "CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT",
    Self::TensorDataType => "CUDNN_ATTR_TENSOR_DATA_TYPE",
    Self::TensorDimensions => "CUDNN_ATTR_TENSOR_DIMENSIONS",
    Self::TensorStrides => "CUDNN_ATTR_TENSOR_STRIDES",
    Self::TensorVectorCount => "CUDNN_ATTR_TENSOR_VECTOR_COUNT",
    Self::TensorVectorizedDimension => "CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION",
    Self::TensorUniqueId => "CUDNN_ATTR_TENSOR_UNIQUE_ID",
    Self::TensorIsVirtual => "CUDNN_ATTR_TENSOR_IS_VIRTUAL",
    Self::TensorIsByValue => "CUDNN_ATTR_TENSOR_IS_BY_VALUE",
    Self::TensorReorderingMode => "CUDNN_ATTR_TENSOR_REORDERING_MODE",
    Self::TensorRaggedOffsetDesc => "CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC",
    Self::VariantPackUniqueIds => "CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS",
    Self::VariantPackDataPointers => "CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS",
    Self::VariantPackIntermediates => "CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES",
    Self::VariantPackWorkspace => "CUDNN_ATTR_VARIANT_PACK_WORKSPACE",
    Self::VariantPackOverrideUniqueIds => "CUDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS",
    Self::VariantPackOverrideShapes => "CUDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES",
    Self::VariantPackOverrideStrides => "CUDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES",
    Self::LayoutInfoTensorUid => "CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID",
    Self::LayoutInfoTypes => "CUDNN_ATTR_LAYOUT_INFO_TYPES",
    Self::KnobInfoType => "CUDNN_ATTR_KNOB_INFO_TYPE",
    Self::KnobInfoMaximumValue => "CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE",
    Self::KnobInfoMinimumValue => "CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE",
    Self::KnobInfoStride => "CUDNN_ATTR_KNOB_INFO_STRIDE",
    Self::EngineOperationGraph => "CUDNN_ATTR_ENGINE_OPERATION_GRAPH",
    Self::EngineGlobalIndex => "CUDNN_ATTR_ENGINE_GLOBAL_INDEX",
    Self::EngineKnobInfo => "CUDNN_ATTR_ENGINE_KNOB_INFO",
    Self::EngineNumericalNote => "CUDNN_ATTR_ENGINE_NUMERICAL_NOTE",
    Self::EngineLayoutInfo => "CUDNN_ATTR_ENGINE_LAYOUT_INFO",
    Self::EngineBehaviorNote => "CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE",
    Self::EngineSmCountTarget => "CUDNN_ATTR_ENGINE_SM_COUNT_TARGET",
    Self::EngineDeviceProp => "CUDNN_ATTR_ENGINE_DEVICEPROP",
    Self::MatmulCompType => "CUDNN_ATTR_MATMUL_COMP_TYPE",
    Self::MatmulPaddingValue => "CUDNN_ATTR_MATMUL_PADDING_VALUE",
    Self::OperationMatmulADesc => "CUDNN_ATTR_OPERATION_MATMUL_ADESC",
    Self::OperationMatmulBDesc => "CUDNN_ATTR_OPERATION_MATMUL_BDESC",
    Self::OperationMatmulCDesc => "CUDNN_ATTR_OPERATION_MATMUL_CDESC",
    Self::OperationMatmulDesc => "CUDNN_ATTR_OPERATION_MATMUL_DESC",
    Self::OperationMatmulIrregularlyStridedBatchCount => "CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT",
    Self::OperationMatmulGemmMOverrideDesc => "CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC",
    Self::OperationMatmulGemmNOverrideDesc => "CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC",
    Self::OperationMatmulGemmKOverrideDesc => "CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC",
    Self::ReductionOperator => "CUDNN_ATTR_REDUCTION_OPERATOR",
    Self::ReductionCompType => "CUDNN_ATTR_REDUCTION_COMP_TYPE",
    Self::ReductionIsDeterministic => "CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC",
    Self::OperationReductionXDesc => "CUDNN_ATTR_OPERATION_REDUCTION_XDESC",
    Self::OperationReductionYDesc => "CUDNN_ATTR_OPERATION_REDUCTION_YDESC",
    Self::OperationReductionDesc => "CUDNN_ATTR_OPERATION_REDUCTION_DESC",
    Self::OperationBnBwdWeightsMathPrec => "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC",
    Self::OperationBnBwdWeightsMeanDesc => "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC",
    Self::OperationBnBwdWeightsInvStdDesc => "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC",
    Self::OperationBnBwdWeightsBnScaleDesc => "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC",
    Self::OperationBnBwdWeightsXDesc => "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC",
    Self::OperationBnBwdWeightsDyDesc => "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC",
    Self::OperationBnBwdWeightsDbnScaleDesc => "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC",
    Self::OperationBnBwdWeightsDbnBiasDesc => "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC",
    Self::OperationBnBwdWeightsEqDyScaleDesc => "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC",
    Self::OperationBnBwdWeightsEqXScaleDesc => "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC",
    Self::OperationBnBwdWeightsEqBias => "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS",
    Self::ResampleMode => "CUDNN_ATTR_RESAMPLE_MODE",
    Self::ResampleCompType => "CUDNN_ATTR_RESAMPLE_COMP_TYPE",
    Self::ResampleSpatialDims => "CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS",
    Self::ResamplePostPaddings => "CUDNN_ATTR_RESAMPLE_POST_PADDINGS",
    Self::ResamplePrePaddings => "CUDNN_ATTR_RESAMPLE_PRE_PADDINGS",
    Self::ResampleStrides => "CUDNN_ATTR_RESAMPLE_STRIDES",
    Self::ResampleWindowDims => "CUDNN_ATTR_RESAMPLE_WINDOW_DIMS",
    Self::ResampleNanPropagation => "CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION",
    Self::ResamplePaddingMode => "CUDNN_ATTR_RESAMPLE_PADDING_MODE",
    Self::OperationResampleFwdXDesc => "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC",
    Self::OperationResampleFwdYDesc => "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC",
    Self::OperationResampleFwdIdxDesc => "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC",
    Self::OperationResampleFwdAlpha => "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA",
    Self::OperationResampleFwdBeta => "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA",
    Self::OperationResampleFwdDesc => "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC",
    Self::OperationResampleBwdDxDesc => "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC",
    Self::OperationResampleBwdDyDesc => "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC",
    Self::OperationResampleBwdIdxDesc => "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC",
    Self::OperationResampleBwdAlpha => "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA",
    Self::OperationResampleBwdBeta => "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA",
    Self::OperationResampleBwdDesc => "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC",
    Self::OperationResampleBwdXDesc => "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC",
    Self::OperationResampleBwdYDesc => "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC",
    Self::OperationConcatAxis => "CUDNN_ATTR_OPERATION_CONCAT_AXIS",
    Self::OperationConcatInputDescs => "CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS",
    Self::OperationConcatInplaceIndex => "CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX",
    Self::OperationConcatOutputDesc => "CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC",
    Self::OperationSignalMode => "CUDNN_ATTR_OPERATION_SIGNAL_MODE",
    Self::OperationSignalFlagDesc => "CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC",
    Self::OperationSignalValue => "CUDNN_ATTR_OPERATION_SIGNAL_VALUE",
    Self::OperationSignalXDesc => "CUDNN_ATTR_OPERATION_SIGNAL_XDESC",
    Self::OperationSignalYDesc => "CUDNN_ATTR_OPERATION_SIGNAL_YDESC",
    Self::OperationPagedCacheLoadContainerDesc => "CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC",
    Self::OperationPagedCacheLoadYDesc => "CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC",
    Self::OperationPagedCacheLoadSequenceDesc => "CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC",
    Self::OperationPagedCacheLoadPageTableDesc => "CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC",
    Self::OperationNormFwdMode => "CUDNN_ATTR_OPERATION_NORM_FWD_MODE",
    Self::OperationNormFwdPhase => "CUDNN_ATTR_OPERATION_NORM_FWD_PHASE",
    Self::OperationNormFwdXDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_XDESC",
    Self::OperationNormFwdMeanDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC",
    Self::OperationNormFwdInvVarianceDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC",
    Self::OperationNormFwdScaleDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC",
    Self::OperationNormFwdBiasDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC",
    Self::OperationNormFwdEpsilonDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC",
    Self::OperationNormFwdExpAvgFactorDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC",
    Self::OperationNormFwdInputRunningMeanDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC",
    Self::OperationNormFwdInputRunningVarDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC",
    Self::OperationNormFwdOutputRunningMeanDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC",
    Self::OperationNormFwdOutputRunningVarDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC",
    Self::OperationNormFwdYDesc => "CUDNN_ATTR_OPERATION_NORM_FWD_YDESC",
    Self::OperationNormFwdPeerStatDescs => "CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS",
    Self::OperationNormBwdMode => "CUDNN_ATTR_OPERATION_NORM_BWD_MODE",
    Self::OperationNormBwdXDesc => "CUDNN_ATTR_OPERATION_NORM_BWD_XDESC",
    Self::OperationNormBwdMeanDesc => "CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC",
    Self::OperationNormBwdInvVarianceDesc => "CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC",
    Self::OperationNormBwdDyDesc => "CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC",
    Self::OperationNormBwdScaleDesc => "CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC",
    Self::OperationNormBwdEpsilonDesc => "CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC",
    Self::OperationNormBwdDScaleDesc => "CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC",
    Self::OperationNormBwdDBiasDesc => "CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC",
    Self::OperationNormBwdDxDesc => "CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC",
    Self::OperationNormBwdPeerStatDescs => "CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS",
    Self::OperationReshapeXDesc => "CUDNN_ATTR_OPERATION_RESHAPE_XDESC",
    Self::OperationReshapeYDesc => "CUDNN_ATTR_OPERATION_RESHAPE_YDESC",
    Self::OperationExpandBandMatrixXDesc => "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_XDESC",
    Self::OperationExpandBandMatrixYDesc => "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_YDESC",
    Self::OperationExpandBandMatrixLowerBandwidth => "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_LOWER_BANDWIDTH",
    Self::OperationExpandBandMatrixUpperBandwidth => "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_UPPER_BANDWIDTH",
    Self::OperationExpandBandMatrixAxis => "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_AXIS",
    Self::OperationExpandBandMatrixPadValue => "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_PAD_VALUE",
    Self::OperationExpandBandMatrixKvTokenOffsetDesc => "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_KV_TOKEN_OFFSET_DESC",
    Self::OperationExpandBandMatrixSpeculativeMaskDesc => "CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_SPECULATIVE_MASK_DESC",
    Self::OperationContractBandMatrixXDesc => "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_XDESC",
    Self::OperationContractBandMatrixYDesc => "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_YDESC",
    Self::OperationContractBandMatrixLowerBandwidth => "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_LOWER_BANDWIDTH",
    Self::OperationContractBandMatrixUpperBandwidth => "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_UPPER_BANDWIDTH",
    Self::OperationContractBandMatrixAxis => "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_AXIS",
    Self::OperationContractBandMatrixPadValue => "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_PAD_VALUE",
    Self::OperationContractBandMaxTokenValue => "CUDNN_ATTR_OPERATION_CONTRACT_BAND_MAX_TOKEN_VALUE",
    Self::RngDistribution => "CUDNN_ATTR_RNG_DISTRIBUTION",
    Self::RngNormalDistMean => "CUDNN_ATTR_RNG_NORMAL_DIST_MEAN",
    Self::RngNormalDistStandardDeviation => "CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION",
    Self::RngUniformDistMaximum => "CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM",
    Self::RngUniformDistMinimum => "CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM",
    Self::RngBernoulliDistProbability => "CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY",
    Self::OperationRngYDesc => "CUDNN_ATTR_OPERATION_RNG_YDESC",
    Self::OperationRngSeed => "CUDNN_ATTR_OPERATION_RNG_SEED",
    Self::OperationRngDesc => "CUDNN_ATTR_OPERATION_RNG_DESC",
    Self::OperationRngOffsetDesc => "CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC",
    Self::KernelCacheIsEngineCfgKernelCached => "CUDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED",
    Self::KernelCacheOperationGraph => "CUDNN_ATTR_KERNEL_CACHE_OPERATION_GRAPH",
    Self::KernelCacheJsonRepresentation => "CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION",
    Self::OperationBlockScaleQuantizeXDesc => "CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC",
    Self::OperationBlockScaleQuantizeYDesc => "CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC",
    Self::OperationBlockScaleQuantizeScaleDesc => "CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC",
    Self::OperationBlockScaleQuantizeMathPrec => "CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC",
    Self::OperationBlockScaleQuantizeBlockSize => "CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE",
    Self::OperationBlockScaleDequantizeXDesc => "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC",
    Self::OperationBlockScaleDequantizeScaleDesc => "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC",
    Self::OperationBlockScaleDequantizeYDesc => "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC",
    Self::OperationBlockScaleDequantizeMathPrec => "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC",
    Self::OperationBlockScaleDequantizeBlockSize => "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE",
    Self::OperationBlockScaleDequantizeNegScale => "CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE",
    Self::OperationSdpaFwdCuSeqLenQDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_QDESC",
    Self::OperationSdpaFwdCuSeqLenKvDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_KVDESC",
    Self::OperationDiagonalBandMaskCuSeqLenQDesc => "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_QDESC",
    Self::OperationDiagonalBandMaskCuSeqLenKvDesc => "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_KVDESC",
    Self::OperationSdpaFwdDescaleQDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_QDESC",
    Self::OperationSdpaFwdDescaleKDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_KDESC",
    Self::OperationSdpaFwdDescaleVDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_VDESC",
    Self::OperationSdpaFwdDescaleSDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_SDESC",
    Self::OperationSdpaFwdScaleSDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_SDESC",
    Self::OperationSdpaFwdScaleODesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_ODESC",
    Self::OperationSdpaFwdAmaxSDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_SDESC",
    Self::OperationSdpaFwdAmaxODesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_ODESC",
    Self::OperationSdpaFwdQDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC",
    Self::OperationSdpaFwdKDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC",
    Self::OperationSdpaFwdVDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC",
    Self::OperationSdpaFwdODesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC",
    Self::OperationSdpaFwdStatsDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC",
    Self::OperationSdpaFwdScaleDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC",
    Self::OperationSdpaFwdBlockMaskDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC",
    Self::OperationSdpaFwdPageTableKDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC",
    Self::OperationSdpaFwdPageTableVDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC",
    Self::OperationSdpaFwdSeqLenQDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC",
    Self::OperationSdpaFwdSeqLenKvDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC",
    Self::OperationSdpaFwdSoftmaxDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_SOFTMAX_DESC",
    Self::OperationSdpaFwdSubgraph => "CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH",
    Self::OperationSdpaFwdSubgraphInputId => "CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID",
    Self::OperationSdpaFwdSubgraphOutputId => "CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID",
    Self::OperationSdpaFwdDropoutSeedDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SEED_DESC",
    Self::OperationSdpaFwdDropoutOffsetDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_OFFSET_DESC",
    Self::OperationSdpaFwdDropoutRngDumpDesc => "CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_RNG_DUMP_DESC",
    Self::OperationSdpaFwdDropoutProbability => "CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_PROBABILITY",
    Self::OperationSdpaFwdUnfuseFma => "CUDNN_ATTR_OPERATION_SDPA_FWD_UNFUSE_FMA",
    Self::OperationSdpaBwdQDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC",
    Self::OperationSdpaBwdKDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC",
    Self::OperationSdpaBwdVDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC",
    Self::OperationSdpaBwdODesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC",
    Self::OperationSdpaBwdStatsDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC",
    Self::OperationSdpaBwdScaleDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC",
    Self::OperationSdpaBwdSeqLenQDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC",
    Self::OperationSdpaBwdSeqLenKvDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC",
    Self::OperationSdpaBwdDqDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC",
    Self::OperationSdpaBwdDkDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC",
    Self::OperationSdpaBwdDvDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC",
    Self::OperationSdpaBwdDoDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC",
    Self::OperationSdpaBwdSinkDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_SINK_DESC",
    Self::OperationSdpaBwdDsinkDesc => "CUDNN_ATTR_OPERATION_SDPA_BWD_DSINK_DESC",
    Self::OperationSdpaBwdMaxTotalSeqLenQ => "CUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_Q",
    Self::OperationSdpaBwdMaxTotalSeqLenKv => "CUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_KV",
    Self::OperationSdpaBwdSubgraph => "CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH",
    Self::OperationSdpaBwdSubgraphInputId => "CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_INPUT_UID",
    Self::OperationSdpaBwdSubgraphOutputId => "CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_OUTPUT_UID",
    Self::OperationMoeGroupedMatmulMode => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE",
    Self::OperationMoeGroupedMatmulMathPrec => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC",
    Self::OperationMoeGroupedMatmulTokenDesc => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC",
    Self::OperationMoeGroupedMatmulWeightDesc => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC",
    Self::OperationMoeGroupedMatmulFirstTokenOffsetDesc => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC",
    Self::OperationMoeGroupedMatmulOutputDesc => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC",
    Self::OperationMoeGroupedMatmulTokenIndexDesc => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC",
    Self::OperationMoeGroupedMatmulTokenKsDesc => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC",
    Self::OperationMoeGroupedMatmulTopK => "CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K",
    Self::OperationDiagonalBandMaskXDesc => "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC",
    Self::OperationDiagonalBandMaskSeqLenKvDesc => "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC",
    Self::OperationDiagonalBandMaskSeqLenQDesc => "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC",
    Self::OperationDiagonalBandMaskLeftBoundDesc => "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC",
    Self::OperationDiagonalBandMaskShiftRightBoundDesc => "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC",
    Self::OperationDiagonalBandMaskBDesc => "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC",
    Self::OperationDiagonalBandMaskYDesc => "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC",
    Self::OperationDiagonalBandMaskComparisonMode => "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE",
    Self::OperationSoftmaxXDesc => "CUDNN_ATTR_OPERATION_SOFTMAX_XDESC",
    Self::OperationSoftmaxYDesc => "CUDNN_ATTR_OPERATION_SOFTMAX_YDESC",
    Self::OperationSoftmaxStatsDesc => "CUDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC",
    Self::OperationSoftmaxMaxDesc => "CUDNN_ATTR_OPERATION_SOFTMAX_MAX_DESC",
    Self::OperationSoftmaxSumExpDesc => "CUDNN_ATTR_OPERATION_SOFTMAX_SUM_EXP_DESC",
    Self::OperationSoftmaxSinkDesc => "CUDNN_ATTR_OPERATION_SOFTMAX_SINK_DESC",
    Self::DevicePropDeviceId => "CUDNN_ATTR_DEVICEPROP_DEVICE_ID",
    Self::DevicePropHandle => "CUDNN_ATTR_DEVICEPROP_HANDLE",
    Self::DevicePropJsonRepresentation => "CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION",
});

/// [`BackendAttributeType`] specifies the data type of an attribute of a cuDNN backend
/// descriptor.
/// [`BackendDescriptor`](crate::backend::descriptor::BackendDescriptor) attribute methods use this enum to pass the matching cuDNN attribute type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BackendAttributeType {
    ///< `cudnnBackendReshapeMode_t` value. @since cuDNN 9.22.0
    ReshapeMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_RESHAPE_MODE as _,
    ///< `cudnnHandle_t` value. @since cuDNN 9.0.0
    Handle = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE as _,
    ///< `cudnnDataType_t` value. @since cuDNN 9.0.0
    DataType = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE as _,
    ///< Boolean value. @since cuDNN 9.0.0
    Boolean = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BOOLEAN as _,
    ///< 64-bit signed integer value. @since cuDNN 9.0.0
    Int64 = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64 as _,
    ///< 32-bit float value. @since cuDNN 9.0.0
    Float = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT as _,
    ///< 64-bit double value. @since cuDNN 9.0.0
    Double = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE as _,
    ///< Void pointer value. @since cuDNN 9.0.0
    VoidPtr = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR as _,
    ///< `cudnnConvolutionMode_t` value. @since cuDNN 9.0.0
    ConvolutionMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_CONVOLUTION_MODE as _,
    ///< `cudnnBackendHeurMode_t` value. @since cuDNN 9.0.0
    HeurMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_HEUR_MODE as _,
    ///< `cudnnBackendKnobType_t` value. @since cuDNN 9.0.0
    KnobType = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_KNOB_TYPE as _,
    ///< `cudnnNanPropagation_t` value. @deprecated @since cuDNN 9.0.0
    NanPropagation = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_NAN_PROPOGATION as _, // Note: FFI has typo "PROPOGATION".
    ///< `cudnnBackendNumericalNote_t` value. @since cuDNN 9.0.0
    NumericalNote = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_NUMERICAL_NOTE as _,
    ///< `cudnnBackendLayoutType_t` value. @since cuDNN 9.0.0
    LayoutType = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_LAYOUT_TYPE as _,
    ///< `cudnnBackendAttributeName_t` value. @since cuDNN 9.0.0
    AttribName = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_ATTRIB_NAME as _,
    ///< `cudnnPointwiseMode_t` value. @since cuDNN 9.0.0
    PointwiseMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_POINTWISE_MODE as _,
    ///< `cudnnBackendDescriptor_t` value. @since cuDNN 9.0.0
    BackendDescriptor = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR as _,
    ///< `cudnnGenStatsMode_t` value. @since cuDNN 9.0.0
    GenStatsMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_GENSTATS_MODE as _,
    ///< `cudnnBnFinalizeStatsMode_t` value. @since cuDNN 9.0.0
    BnFinalizeStatsMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BN_FINALIZE_STATS_MODE as _,
    ///< `cudnnReduceTensorOp_t` value. @since cuDNN 9.0.0
    ReductionOperatorType =
        sys::cudnnBackendAttributeType_t::CUDNN_TYPE_REDUCTION_OPERATOR_TYPE as _,
    ///< `cudnnBackendBehaviorNote_t` value. @since cuDNN 9.0.0
    BehaviorNote = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BEHAVIOR_NOTE as _,
    ///< `cudnnBackendTensorReordering_t` value. @since cuDNN 9.0.0
    TensorReorderingMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_TENSOR_REORDERING_MODE as _,
    ///< `cudnnResampleMode_t` value. @since cuDNN 9.0.0
    ResampleMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_RESAMPLE_MODE as _,
    ///< `cudnnPaddingMode_t` value. @since cuDNN 9.0.0
    PaddingMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_PADDING_MODE as _,
    ///< 32-bit signed integer value. @since cuDNN 9.0.0
    Int32 = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT32 as _,
    ///< Character value. @since cuDNN 9.0.0
    Char = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_CHAR as _,
    ///< `cudnnSignalMode_t` value. @since cuDNN 9.0.0
    SignalMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_SIGNAL_MODE as _,
    ///< `cudnnFraction_t` value. @since cuDNN 9.0.0
    Fraction = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FRACTION as _,
    ///< `cudnnBackendNormMode_t` value. @since cuDNN 9.0.0
    NormMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_NORM_MODE as _,
    ///< `cudnnBackendNormFwdPhase_t` value. @since cuDNN 9.0.0
    NormFwdPhase = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_NORM_FWD_PHASE as _,
    ///< `cudnnRngDistribution_t` value. @since cuDNN 9.0.0
    RngDistribution = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_RNG_DISTRIBUTION as _,
    ///< `cudnnMoeGroupedMatmulMode_t` value. @since cuDNN 9.15.0
    MoeGroupedMatmulMode =
        sys::cudnnBackendAttributeType_t::CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE as _,
}

impl_enum_conversion!(sys::cudnnBackendAttributeType_t, BackendAttributeType,);

impl_enum_display!(BackendAttributeType, {
    Self::ReshapeMode => "CUDNN_TYPE_RESHAPE_MODE",
    Self::Handle => "CUDNN_TYPE_HANDLE",
    Self::DataType => "CUDNN_TYPE_DATA_TYPE",
    Self::Boolean => "CUDNN_TYPE_BOOLEAN",
    Self::Int64 => "CUDNN_TYPE_INT64",
    Self::Float => "CUDNN_TYPE_FLOAT",
    Self::Double => "CUDNN_TYPE_DOUBLE",
    Self::VoidPtr => "CUDNN_TYPE_VOID_PTR",
    Self::ConvolutionMode => "CUDNN_TYPE_CONVOLUTION_MODE",
    Self::HeurMode => "CUDNN_TYPE_HEUR_MODE",
    Self::KnobType => "CUDNN_TYPE_KNOB_TYPE",
    Self::NanPropagation => "CUDNN_TYPE_NAN_PROPOGATION",
    Self::NumericalNote => "CUDNN_TYPE_NUMERICAL_NOTE",
    Self::LayoutType => "CUDNN_TYPE_LAYOUT_TYPE",
    Self::AttribName => "CUDNN_TYPE_ATTRIB_NAME",
    Self::PointwiseMode => "CUDNN_TYPE_POINTWISE_MODE",
    Self::BackendDescriptor => "CUDNN_TYPE_BACKEND_DESCRIPTOR",
    Self::GenStatsMode => "CUDNN_TYPE_GENSTATS_MODE",
    Self::BnFinalizeStatsMode => "CUDNN_TYPE_BN_FINALIZE_STATS_MODE",
    Self::ReductionOperatorType => "CUDNN_TYPE_REDUCTION_OPERATOR_TYPE",
    Self::BehaviorNote => "CUDNN_TYPE_BEHAVIOR_NOTE",
    Self::TensorReorderingMode => "CUDNN_TYPE_TENSOR_REORDERING_MODE",
    Self::ResampleMode => "CUDNN_TYPE_RESAMPLE_MODE",
    Self::PaddingMode => "CUDNN_TYPE_PADDING_MODE",
    Self::Int32 => "CUDNN_TYPE_INT32",
    Self::Char => "CUDNN_TYPE_CHAR",
    Self::SignalMode => "CUDNN_TYPE_SIGNAL_MODE",
    Self::Fraction => "CUDNN_TYPE_FRACTION",
    Self::NormMode => "CUDNN_TYPE_NORM_MODE",
    Self::NormFwdPhase => "CUDNN_TYPE_NORM_FWD_PHASE",
    Self::RngDistribution => "CUDNN_TYPE_RNG_DISTRIBUTION",
    Self::MoeGroupedMatmulMode => "CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE",
});

/// A backend enum value with its native attribute payload category.
/// Safe scalar and slice setters convert values to u32, and getters decode u32 through the enum's conversion.
/// Engine metadata preserves unknown native IDs; enums with a closed set of values reject them.
///
/// # Safety
/// `ATTRIBUTE_TYPE` must be a native enum category represented by one u32.
/// It must match the meaning of this type's `Into<u32>` and `TryFrom<u32>` conversions.
/// Categories with other payload sizes, pointers or retained objects are invalid.
/// Use the public unsafe attribute methods for other native payload ABIs.
pub unsafe trait BackendEnum: Copy + Into<u32> + TryFrom<u32> {
    /// Native attribute payload category matching this enum's 32-bit integer representation.
    const ATTRIBUTE_TYPE: BackendAttributeType;
}

macro_rules! backend_enum {
    ($value:ty, $native:ty, $category:ident) => {
        const _: () = assert!(size_of::<$native>() == size_of::<u32>());
        // SAFETY: this cuDNN category stores the corresponding native 32-bit enum;
        // the Rust wrapper supplies checked decoding and matching integer values.
        unsafe impl BackendEnum for $value {
            const ATTRIBUTE_TYPE: BackendAttributeType = BackendAttributeType::$category;
        }
    };
}

backend_enum!(DataType, sys::cudnnDataType_t, DataType);

backend_enum!(ReshapeMode, sys::cudnnBackendReshapeMode_t, ReshapeMode);

backend_enum!(
    ConvolutionMode,
    sys::cudnnConvolutionMode_t,
    ConvolutionMode
);

backend_enum!(BackendHeuristicMode, sys::cudnnBackendHeurMode_t, HeurMode);

backend_enum!(BackendKnobType, sys::cudnnBackendKnobType_t, KnobType);

backend_enum!(NanPropagation, sys::cudnnNanPropagation_t, NanPropagation);

backend_enum!(
    BackendNumericalNote,
    sys::cudnnBackendNumericalNote_t,
    NumericalNote
);

backend_enum!(BackendLayoutType, sys::cudnnBackendLayoutType_t, LayoutType);

backend_enum!(
    BackendAttributeName,
    sys::cudnnBackendAttributeName_t,
    AttribName
);

backend_enum!(
    BackendAttributeId,
    sys::cudnnBackendAttributeName_t,
    AttribName
);

backend_enum!(PointwiseMode, sys::cudnnPointwiseMode_t, PointwiseMode);

backend_enum!(GenStatsMode, sys::cudnnGenStatsMode_t, GenStatsMode);

backend_enum!(
    BatchNormalizationFinalizeStatsMode,
    sys::cudnnBnFinalizeStatsMode_t,
    BnFinalizeStatsMode
);

backend_enum!(
    ReduceTensorOperator,
    sys::cudnnReduceTensorOp_t,
    ReductionOperatorType
);

backend_enum!(
    BackendBehaviorNote,
    sys::cudnnBackendBehaviorNote_t,
    BehaviorNote
);

backend_enum!(
    BackendTensorReordering,
    sys::cudnnBackendTensorReordering_t,
    TensorReorderingMode
);

backend_enum!(ResampleMode, sys::cudnnResampleMode_t, ResampleMode);

backend_enum!(PaddingMode, sys::cudnnPaddingMode_t, PaddingMode);

backend_enum!(SignalMode, sys::cudnnSignalMode_t, SignalMode);

backend_enum!(
    BackendNormalizationMode,
    sys::cudnnBackendNormMode_t,
    NormMode
);

backend_enum!(
    BackendNormalizationForwardPhase,
    sys::cudnnBackendNormFwdPhase_t,
    NormFwdPhase
);

backend_enum!(
    RngDistribution,
    sys::cudnnRngDistribution_t,
    RngDistribution
);

backend_enum!(
    MoeGroupedMatmulMode,
    sys::cudnnMoeGroupedMatmulMode_t,
    MoeGroupedMatmulMode
);
