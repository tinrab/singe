//! Typed backend attribute names, native representations, and attribute-value conversion.

use std::fmt;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::impl_enum_display;
use singe_cudnn_sys as sys;

use crate::{
    backend::{
        behavior::{BackendBehaviorNote, BackendNumericalNote},
        heuristic::BackendHeuristicMode,
        knob::BackendKnobType,
        layout::BackendLayoutType,
        operation::{
            convolution::ConvolutionMode,
            moe::MoeGroupedMatmulMode,
            normalization::{
                BackendNormalizationForwardPhase, BackendNormalizationMode,
                BatchNormalizationFinalizeStatsMode, GenStatsMode,
            },
            pointwise::PointwiseMode,
            reduction::ReduceTensorOperator,
            resample::{PaddingMode, ResampleMode},
            rng::RngDistribution,
            tensor_ops::{ReshapeMode, SignalMode},
        },
        tensor::BackendTensorReordering,
    },
    data_type::DataType,
    math::NanPropagation,
};

singe_core::native_enum! {
    /// Backend descriptor attribute name that can be set or retrieved with [`BackendDescriptor`](crate::backend::descriptor::BackendDescriptor) attribute methods.
    /// The backend descriptor to which an attribute belongs is identified by the prefix of the attribute name.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum BackendAttributeName: sys::cudnnBackendAttributeName_t {
        /// Descriptor of the group-offset tensor for a grouped reduction operation.
        OperationReductionGroupOffsetDesc = CUDNN_ATTR_OPERATION_REDUCTION_GROUP_OFFSET_DESC,
        /// Backward math precision for `MoE` grouped matmul.
        OperationMoeGroupedMatmulBackwardMathPrec = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
        /// Backward token descriptor for `MoE` grouped matmul.
        OperationMoeGroupedMatmulBackwardTokenDesc = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC,
        /// Backward weight descriptor for `MoE` grouped matmul.
        OperationMoeGroupedMatmulBackwardWeightGradientDesc = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
        /// Backward first token offset descriptor for `MoE` grouped matmul.
        OperationMoeGroupedMatmulBackwardFirstTokenOffsetDesc = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
        /// Backward output descriptor for `MoE` grouped matmul.
        OperationMoeGroupedMatmulBackwardOutputGradientDesc = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
        /// `RoPE` input tensor descriptor with shape `[B, H, S, D]`.
        /// Available since cuDNN 9.24.0.
        OperationRopeForwardInputDesc = CUDNN_ATTR_OPERATION_ROPE_FWD_XDESC,
        /// `RoPE` output tensor descriptor with shape `[B, H, S, D]`.
        /// Available since cuDNN 9.24.0.
        OperationRopeForwardOutputDesc = CUDNN_ATTR_OPERATION_ROPE_FWD_YDESC,
        /// `RoPE` frequency-angle tensor with shape `[S, 1, 1, D]`.
        /// The kernel computes `sincosf` internally.
        /// Available since cuDNN 9.24.0.
        OperationRopeForwardFrequenciesDesc = CUDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC,
        /// `RoPE` host-side fp32 scalar multiplied into cos/sin (default 1.0).
        /// Used to fold YARN mscale and/or SDPA `attn_scale`.
        OperationRopeForwardOutputScale = CUDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE,
        /// `RoPE` rotation width in last dim (int64, default = `head_dim`).
        /// When < `head_dim`, the last `rope_dim` slots get rotated; the first (`head_dim` - `rope_dim`) are scaled-pass-through (used for DSv3-style MLA).
        OperationRopeForwardRotationDimension = CUDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM,
        /// `RoPE` input-gradient (`dY`) tensor descriptor with shape `[B, H, S, D]`.
        /// Available since cuDNN 9.25.0.
        OperationRopeBackwardInputDesc = CUDNN_ATTR_OPERATION_ROPE_BWD_DYDESC,
        /// `RoPE` output-gradient (`dX`) tensor descriptor with shape `[B, H, S, D]`.
        /// Available since cuDNN 9.25.0.
        OperationRopeBackwardOutputDesc = CUDNN_ATTR_OPERATION_ROPE_BWD_DXDESC,
        /// `RoPE` backward frequency-angle tensor with shape `[S, 1, 1, D]`, using the same semantics as forward execution.
        /// Available since cuDNN 9.25.0.
        OperationRopeBackwardFrequenciesDesc = CUDNN_ATTR_OPERATION_ROPE_BWD_FREQSDESC,
        /// `RoPE` bwd host fp32 scalar (default 1.0).
        /// Symmetric to fwd: cos/sin pre-scaled, passthrough scaled.
        OperationRopeBackwardOutputScale = CUDNN_ATTR_OPERATION_ROPE_BWD_OUTPUT_SCALE,
        /// `RoPE` bwd rotation width in last dim (int64, default = `head_dim`).
        OperationRopeBackwardRotationDimension = CUDNN_ATTR_OPERATION_ROPE_BWD_ROPE_DIM,
        /// Compile-time constant value for by-value tensors.
        TensorConstantValue = CUDNN_ATTR_TENSOR_CONSTANT_VALUE,
        /// Ragged tensor offset unit size.
        TensorRaggedOffsetMultiplier = CUDNN_ATTR_TENSOR_RAGGED_OFFSET_MULTIPLIER,
        /// Reshape mode (view-only or logical).
        OperationReshapeMode = CUDNN_ATTR_OPERATION_RESHAPE_MODE,
        /// Slice input tensor descriptor.
        OperationSliceXDesc = CUDNN_ATTR_OPERATION_SLICE_XDESC,
        /// Slice output tensor descriptor.
        OperationSliceYDesc = CUDNN_ATTR_OPERATION_SLICE_YDESC,
        /// Slice start indices.
        OperationSliceStartIndices = CUDNN_ATTR_OPERATION_SLICE_START_INDICES,
        /// Slice limit indices.
        OperationSliceLimitIndices = CUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES,
        /// Slice strides.
        OperationSliceStrides = CUDNN_ATTR_OPERATION_SLICE_STRIDES,
        /// Transpose input tensor descriptor.
        OperationTransposeXDesc = CUDNN_ATTR_OPERATION_TRANSPOSE_XDESC,
        /// Transpose output tensor descriptor.
        OperationTransposeYDesc = CUDNN_ATTR_OPERATION_TRANSPOSE_YDESC,
        /// Transpose permutation array.
        OperationTransposePermutation = CUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION,
        /// Shared memory limit metadata propagated to heuristic results.
        EngineHeurSharedMemoryLimit = CUDNN_ATTR_ENGINEHEUR_SHARED_MEMORY_LIMIT,
        /// Dynamic shared memory budget, in bytes.
        EngineSharedMemoryLimit = CUDNN_ATTR_ENGINE_SHARED_MEMORY_LIMIT,
        /// Counter that shows if the kernel cache changed.
        KernelCacheRevision = CUDNN_ATTR_KERNEL_CACHE_REVISION,
        /// Number of entries in the kernel cache.
        KernelCacheEntryCount = CUDNN_ATTR_KERNEL_CACHE_ENTRY_COUNT,
        /// Emit statistics as natural-log LSE multiplied by log2(e).
        /// SDPA backward expects natural-log statistics.
        OperationSoftmaxStatsLog2 = CUDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2,
        /// Indices of the input dimensions to reduce, ascending and distinct.
        OperationReductionDims = CUDNN_ATTR_OPERATION_REDUCTION_DIMS,
        /// Accumulator identity values, one pointer per output; set the output descriptors first.
        OperationReductionInitValues = CUDNN_ATTR_OPERATION_REDUCTION_INIT_VALUES,
        /// Finalized operation graph describing the reducer body combining accumulator and current element.
        OperationReductionSubgraph = CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH,
        /// Subgraph tensor IDs holding accumulator values, one per output.
        OperationReductionSubgraphAccumulatorUids = CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_ACCUMULATOR_UIDS,
        /// Subgraph tensor IDs holding current elements, one per output.
        OperationReductionSubgraphCurrentUids = CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_CURRENT_UIDS,
        /// Subgraph tensor IDs holding output values, one per output.
        OperationReductionSubgraphOutputUids = CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_OUTPUT_UIDS,
        /// Optional compute type the reducer body evaluates in, one per output; defaults to output element type.
        OperationReductionSubgraphComputeTypes = CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_COMPUTE_TYPES,
        /// Pointwise operation type.
        PointwiseMode = CUDNN_ATTR_POINTWISE_MODE,
        /// Computation precision for pointwise ops.
        PointwiseMathPrec = CUDNN_ATTR_POINTWISE_MATH_PREC,
        /// NaN handling behavior.
        /// Deprecated by cuDNN.
        PointwiseNanPropagation = CUDNN_ATTR_POINTWISE_NAN_PROPAGATION,
        /// Lower clipping threshold for `ReLU`.
        PointwiseReluLowerClip = CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
        /// Upper clipping threshold for `ReLU`.
        PointwiseReluUpperClip = CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
        /// Slope below lower clip for leaky `ReLU`.
        PointwiseReluLowerClipSlope = CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
        /// Alpha parameter for ELU activation.
        PointwiseEluAlpha = CUDNN_ATTR_POINTWISE_ELU_ALPHA,
        /// Beta parameter for softplus function.
        PointwiseSoftplusBeta = CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
        /// Beta parameter for swish activation.
        PointwiseSwishBeta = CUDNN_ATTR_POINTWISE_SWISH_BETA,
        /// Axis for axis-dependent pointwise operations.
        PointwiseAxis = CUDNN_ATTR_POINTWISE_AXIS,
        /// Computation data type for convolution.
        ConvolutionCompType = CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
        /// Convolution vs cross-correlation mode.
        ConvolutionConvMode = CUDNN_ATTR_CONVOLUTION_CONV_MODE,
        /// Dilation factors per spatial dimension.
        ConvolutionDilations = CUDNN_ATTR_CONVOLUTION_DILATIONS,
        /// Filter stride values per spatial dimension.
        ConvolutionFilterStrides = CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
        /// Post-paddings per spatial dimension.
        ConvolutionPostPaddings = CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
        /// Pre-paddings per spatial dimension.
        ConvolutionPrePaddings = CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
        /// Number of spatial dimensions.
        ConvolutionSpatialDims = CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
        /// Heuristic algorithm selection mode.
        EngineHeurMode = CUDNN_ATTR_ENGINEHEUR_MODE,
        /// Associated operation graph descriptor.
        EngineHeurOperationGraph = CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
        /// Array of resulting engine configurations.
        EngineHeurResults = CUDNN_ATTR_ENGINEHEUR_RESULTS,
        /// Target streaming multiprocessor count.
        EngineHeurSmCountTarget = CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET,
        /// Device properties for heuristic query.
        EngineHeurDeviceProp = CUDNN_ATTR_ENGINEHEUR_DEVICEPROP,
        /// Selected engine from heuristic results.
        EngineCfgEngine = CUDNN_ATTR_ENGINECFG_ENGINE,
        /// Intermediate tensor information.
        EngineCfgIntermediateInfo = CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO,
        /// Performance tuning knob selections.
        EngineCfgKnobChoices = CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
        /// Required workspace memory size.
        EngineCfgWorkspaceSize = CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
        /// Shared-memory requirement reported by an engine configuration, in bytes.
        EngineCfgSharedMemoryUsed = CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED,
        /// Associated cuDNN handle.
        /// Deprecated by cuDNN.
        ExecutionPlanHandle = CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
        /// Engine configuration to execute.
        ExecutionPlanEngineConfig = CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
        /// Total workspace size requirement.
        ExecutionPlanWorkspaceSize = CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
        /// UIDs of computed intermediates.
        ExecutionPlanComputedIntermediateIds = CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS,
        /// Run-only intermediate UIDs.
        ExecutionPlanRunOnlyIntermediateIds = CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS,
        /// Human-readable execution plan JSON.
        ExecutionPlanJsonRepresentation = CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
        /// Compiled kernel cache descriptor.
        ExecutionPlanKernelCache = CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE,
        /// Device properties for execution plan.
        ExecutionPlanDeviceProp = CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP,
        /// Unique identifier for intermediate tensor.
        IntermediateInfoUniqueId = CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID,
        /// Memory size requirement in bytes.
        IntermediateInfoSize = CUDNN_ATTR_INTERMEDIATE_INFO_SIZE,
        /// Data dependency UIDs.
        IntermediateInfoDependentTensorUids = CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS,
        /// Attribute dependencies.
        IntermediateInfoDependentAttributes = CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES,
        /// Type of performance tuning knob.
        KnobChoiceKnobType = CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
        /// Selected value for the knob.
        KnobChoiceKnobValue = CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
        /// Forward convolution scaling factor alpha.
        OperationConvolutionForwardAlpha = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
        /// Forward convolution scaling factor beta.
        OperationConvolutionForwardBeta = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
        /// Forward convolution descriptor.
        OperationConvolutionForwardConvDesc = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
        /// Forward convolution filter weight tensor.
        OperationConvolutionForwardW = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
        /// Forward convolution input tensor.
        OperationConvolutionForwardX = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
        /// Forward convolution output tensor.
        OperationConvolutionForwardY = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
        /// Backward data scaling factor alpha.
        OperationConvolutionBwdDataAlpha = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
        /// Backward data scaling factor beta.
        OperationConvolutionBwdDataBeta = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
        /// Backward data convolution descriptor.
        OperationConvolutionBwdDataConvDesc = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
        /// Backward data filter weights.
        OperationConvolutionBwdDataW = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
        /// Backward data input gradient output.
        OperationConvolutionBwdDataDx = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
        /// Backward data output gradient input.
        OperationConvolutionBwdDataDy = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
        /// Backward filter scaling factor alpha.
        OperationConvolutionBwdFilterAlpha = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
        /// Backward filter scaling factor beta.
        OperationConvolutionBwdFilterBeta = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
        /// Backward filter convolution descriptor.
        OperationConvolutionBwdFilterConvDesc = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
        /// Backward filter gradient output.
        OperationConvolutionBwdFilterDw = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
        /// Backward filter input feature maps.
        OperationConvolutionBwdFilterX = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
        /// Backward filter output gradients.
        OperationConvolutionBwdFilterDy = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
        /// Pointwise descriptor reference.
        OperationPointwisePwDescriptor = CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
        /// First input tensor descriptor.
        OperationPointwiseXDesc = CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
        /// Bias or second operand descriptor.
        OperationPointwiseBDesc = CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
        /// Output tensor descriptor.
        OperationPointwiseYDesc = CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
        /// First scaling constant.
        OperationPointwiseAlpha1 = CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1,
        /// Second scaling constant.
        OperationPointwiseAlpha2 = CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2,
        /// Input gradient descriptor.
        OperationPointwiseDxDesc = CUDNN_ATTR_OPERATION_POINTWISE_DXDESC,
        /// Output gradient descriptor.
        OperationPointwiseDyDesc = CUDNN_ATTR_OPERATION_POINTWISE_DYDESC,
        /// Intermediate tensor descriptor.
        OperationPointwiseTDesc = CUDNN_ATTR_OPERATION_POINTWISE_TDESC,
        /// Statistics computation mode.
        OperationGenStatsMode = CUDNN_ATTR_OPERATION_GENSTATS_MODE,
        /// Computation precision for statistics.
        OperationGenStatsMathPrec = CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC,
        /// Input data tensor descriptor.
        OperationGenStatsXDesc = CUDNN_ATTR_OPERATION_GENSTATS_XDESC,
        /// Sum output descriptor.
        OperationGenStatsSumDesc = CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC,
        /// Sum of squares output descriptor.
        OperationGenStatsSqSumDesc = CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC,
        /// Training vs inference mode.
        OperationBnFinalizeStatsMode = CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE,
        /// Computation precision.
        OperationBnFinalizeMathPrec = CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC,
        /// Sum of batch norm outputs.
        OperationBnFinalizeYSumDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC,
        /// Sum of squared outputs.
        OperationBnFinalizeYSqSumDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC,
        /// Batch norm scale parameter.
        OperationBnFinalizeScaleDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC,
        /// Batch norm bias parameter.
        OperationBnFinalizeBiasDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC,
        /// Previous running mean.
        OperationBnFinalizePrevRunningMeanDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC,
        /// Previous running variance.
        OperationBnFinalizePrevRunningVarDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC,
        /// Updated running mean output.
        OperationBnFinalizeUpdatedRunningMeanDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC,
        /// Updated running variance output.
        OperationBnFinalizeUpdatedRunningVarDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC,
        /// Cached mean for backward pass.
        OperationBnFinalizeSavedMeanDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC,
        /// Cached inverse std dev for backward.
        OperationBnFinalizeSavedInvStdDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC,
        /// Equivalent scale for fused inference.
        OperationBnFinalizeEqScaleDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC,
        /// Equivalent bias for fused inference.
        OperationBnFinalizeEqBiasDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC,
        /// Accumulation sample counter.
        OperationBnFinalizeAccumCountDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC,
        /// Numerical stability epsilon.
        OperationBnFinalizeEpsilonDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC,
        /// Exponential averaging factor.
        OperationBnFinalizeExpAverateFactorDesc = CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC,
        /// Associated cuDNN handle.
        /// Deprecated by cuDNN.
        OperationGraphHandle = CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
        /// Array of operation descriptors in the graph.
        OperationGraphOps = CUDNN_ATTR_OPERATIONGRAPH_OPS,
        /// Total number of engines available.
        OperationGraphEngineGlobalCount = CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
        /// Dynamic shape support flag.
        OperationGraphIsDynamicShapeEnabled = CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED,
        /// Same topology reuse flag.
        OperationGraphIsSameTopology = CUDNN_ATTR_OPERATIONGRAPH_IS_SAME_TOPOLOGY,
        /// Dynamic shape support with execute time override
        OperationGraphIsOverrideShapeEnabled = CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED,
        /// Memory alignment requirement in bytes.
        TensorByteAlignment = CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
        /// Element data type.
        TensorDataType = CUDNN_ATTR_TENSOR_DATA_TYPE,
        /// Dimension sizes array.
        TensorDimensions = CUDNN_ATTR_TENSOR_DIMENSIONS,
        /// Memory strides per dimension.
        TensorStrides = CUDNN_ATTR_TENSOR_STRIDES,
        /// Vectorization element count.
        TensorVectorCount = CUDNN_ATTR_TENSOR_VECTOR_COUNT,
        /// Which dimension is vectorized.
        TensorVectorizedDimension = CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION,
        /// Unique identifier for graph connectivity.
        TensorUniqueId = CUDNN_ATTR_TENSOR_UNIQUE_ID,
        /// Virtual (intermediate) vs I/O tensor flag.
        TensorIsVirtual = CUDNN_ATTR_TENSOR_IS_VIRTUAL,
        /// Constant scalar vs device pointer flag.
        TensorIsByValue = CUDNN_ATTR_TENSOR_IS_BY_VALUE,
        /// Memory layout transformation type.
        TensorReorderingMode = CUDNN_ATTR_TENSOR_REORDERING_MODE,
        /// Ragged tensor offset descriptor.
        TensorRaggedOffsetDesc = CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC,
        /// Tensor UIDs in this variant pack.
        VariantPackUniqueIds = CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
        /// GPU memory data pointers array.
        VariantPackDataPointers = CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
        /// Intermediate tensor pointers.
        VariantPackIntermediates = CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES,
        /// Workspace memory pointer.
        VariantPackWorkspace = CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
        /// Override tensor UIDs for dynamic shapes.
        VariantPackOverrideUniqueIds = CUDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS,
        /// Override shapes for dynamic shapes.
        VariantPackOverrideShapes = CUDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES,
        /// Override strides for dynamic shapes.
        VariantPackOverrideStrides = CUDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES,
        /// Associated tensor identifier.
        LayoutInfoTensorUid = CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID,
        /// Available memory layout types.
        LayoutInfoTypes = CUDNN_ATTR_LAYOUT_INFO_TYPES,
        /// Knob type being described.
        KnobInfoType = CUDNN_ATTR_KNOB_INFO_TYPE,
        /// Upper bound for knob value.
        KnobInfoMaximumValue = CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE,
        /// Lower bound for knob value.
        KnobInfoMinimumValue = CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE,
        /// Valid increment between knob values.
        KnobInfoStride = CUDNN_ATTR_KNOB_INFO_STRIDE,
        /// Operation graph this engine processes.
        EngineOperationGraph = CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
        /// Engine index in the global engine list.
        EngineGlobalIndex = CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
        /// Available knob configuration options.
        EngineKnobInfo = CUDNN_ATTR_ENGINE_KNOB_INFO,
        /// Numerical properties (tensor cores, precision).
        EngineNumericalNote = CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
        /// Preferred tensor memory layouts.
        EngineLayoutInfo = CUDNN_ATTR_ENGINE_LAYOUT_INFO,
        /// Runtime behavior characteristics.
        EngineBehaviorNote = CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
        /// Streaming multiprocessor target count.
        EngineSmCountTarget = CUDNN_ATTR_ENGINE_SM_COUNT_TARGET,
        /// Device properties descriptor.
        EngineDeviceProp = CUDNN_ATTR_ENGINE_DEVICEPROP,
        /// Computation precision type for matmul.
        MatmulCompType = CUDNN_ATTR_MATMUL_COMP_TYPE,
        /// Padding value for incomplete blocks.
        MatmulPaddingValue = CUDNN_ATTR_MATMUL_PADDING_VALUE,
        /// First input matrix (A) tensor descriptor.
        OperationMatmulADesc = CUDNN_ATTR_OPERATION_MATMUL_ADESC,
        /// Second input matrix (B) tensor descriptor.
        OperationMatmulBDesc = CUDNN_ATTR_OPERATION_MATMUL_BDESC,
        /// Output matrix (C) tensor descriptor.
        OperationMatmulCDesc = CUDNN_ATTR_OPERATION_MATMUL_CDESC,
        /// `MatMul` operation configuration descriptor.
        OperationMatmulDesc = CUDNN_ATTR_OPERATION_MATMUL_DESC,
        /// Irregular batch count.
        /// Deprecated by cuDNN.
        OperationMatmulIrregularlyStridedBatchCount = CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT,
        /// Override for output rows (M).
        OperationMatmulGemmMOverrideDesc = CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC,
        /// Override for output columns (N).
        OperationMatmulGemmNOverrideDesc = CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC,
        /// Override for contraction dim (K).
        OperationMatmulGemmKOverrideDesc = CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC,
        /// Reduction operation type (ADD, MUL, MIN, etc.).
        ReductionOperator = CUDNN_ATTR_REDUCTION_OPERATOR,
        /// Computation data type for reduction.
        ReductionCompType = CUDNN_ATTR_REDUCTION_COMP_TYPE,
        /// Whether reduction must be deterministic.
        ReductionIsDeterministic = CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC,
        /// Reduction input tensor descriptor.
        OperationReductionXDesc = CUDNN_ATTR_OPERATION_REDUCTION_XDESC,
        /// Reduction output tensor descriptor.
        OperationReductionYDesc = CUDNN_ATTR_OPERATION_REDUCTION_YDESC,
        /// Reduction descriptor reference.
        OperationReductionDesc = CUDNN_ATTR_OPERATION_REDUCTION_DESC,
        /// Computation precision.
        OperationBnBwdWeightsMathPrec = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC,
        /// Cached batch mean from forward.
        OperationBnBwdWeightsMeanDesc = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC,
        /// Cached inverse std dev from forward.
        OperationBnBwdWeightsInvStdDesc = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC,
        /// Batch norm scale parameter.
        OperationBnBwdWeightsBnScaleDesc = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC,
        /// Forward input tensor.
        OperationBnBwdWeightsXDesc = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC,
        /// Output gradient tensor.
        OperationBnBwdWeightsDyDesc = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC,
        /// Scale gradient output.
        OperationBnBwdWeightsDbnScaleDesc = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC,
        /// Bias gradient output.
        OperationBnBwdWeightsDbnBiasDesc = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC,
        /// Equivalent output gradient scale.
        OperationBnBwdWeightsEqDyScaleDesc = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC,
        /// Equivalent input scale.
        OperationBnBwdWeightsEqXScaleDesc = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC,
        /// Equivalent bias value.
        OperationBnBwdWeightsEqBias = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS,
        /// Resampling interpolation method.
        ResampleMode = CUDNN_ATTR_RESAMPLE_MODE,
        /// Computation precision for resampling.
        ResampleCompType = CUDNN_ATTR_RESAMPLE_COMP_TYPE,
        /// Number of spatial dimensions.
        ResampleSpatialDims = CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS,
        /// Post-paddings per spatial dimension.
        ResamplePostPaddings = CUDNN_ATTR_RESAMPLE_POST_PADDINGS,
        /// Pre-paddings per spatial dimension.
        ResamplePrePaddings = CUDNN_ATTR_RESAMPLE_PRE_PADDINGS,
        /// Stride factors per spatial dimension.
        ResampleStrides = CUDNN_ATTR_RESAMPLE_STRIDES,
        /// Filter window sizes per dimension.
        ResampleWindowDims = CUDNN_ATTR_RESAMPLE_WINDOW_DIMS,
        /// NaN handling behavior.
        ResampleNanPropagation = CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION,
        /// Padding strategy (zero, `neg_inf`, edge).
        ResamplePaddingMode = CUDNN_ATTR_RESAMPLE_PADDING_MODE,
        /// Forward resample input tensor.
        OperationResampleFwdXDesc = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
        /// Forward resample output tensor.
        OperationResampleFwdYDesc = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC,
        /// Max pooling index tensor.
        OperationResampleFwdIdxDesc = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
        /// Output scaling factor.
        /// Deprecated by cuDNN.
        OperationResampleFwdAlpha = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA,
        /// Accumulation scaling factor.
        /// Deprecated by cuDNN.
        OperationResampleFwdBeta = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA,
        /// Resample descriptor reference.
        OperationResampleFwdDesc = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC,
        /// Backward resample input gradient.
        OperationResampleBwdDxDesc = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC,
        /// Backward resample output gradient.
        OperationResampleBwdDyDesc = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC,
        /// Index tensor from forward pass.
        OperationResampleBwdIdxDesc = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC,
        /// Gradient scaling factor.
        /// Deprecated by cuDNN.
        OperationResampleBwdAlpha = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA,
        /// Accumulation scaling.
        /// Deprecated by cuDNN.
        OperationResampleBwdBeta = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA,
        /// Resample descriptor reference.
        OperationResampleBwdDesc = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC,
        /// Forward input reference.
        OperationResampleBwdXDesc = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC,
        /// Forward output reference.
        OperationResampleBwdYDesc = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC,
        /// Concatenation axis dimension.
        OperationConcatAxis = CUDNN_ATTR_OPERATION_CONCAT_AXIS,
        /// Array of input tensor descriptors.
        OperationConcatInputDescs = CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS,
        /// In-place output tensor selection index.
        OperationConcatInplaceIndex = CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX,
        /// Concatenated output descriptor.
        OperationConcatOutputDesc = CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC,
        /// Signal set vs wait mode.
        OperationSignalMode = CUDNN_ATTR_OPERATION_SIGNAL_MODE,
        /// Flag variable tensor descriptor.
        OperationSignalFlagDesc = CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC,
        /// Signal value for comparison.
        OperationSignalValue = CUDNN_ATTR_OPERATION_SIGNAL_VALUE,
        /// Input tensor for signal set.
        OperationSignalXDesc = CUDNN_ATTR_OPERATION_SIGNAL_XDESC,
        /// Output tensor descriptor.
        OperationSignalYDesc = CUDNN_ATTR_OPERATION_SIGNAL_YDESC,
        /// Cache container descriptor.
        OperationPagedCacheLoadContainerDesc = CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC,
        /// Output tensor descriptor.
        OperationPagedCacheLoadYDesc = CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC,
        /// Load sequence specification.
        OperationPagedCacheLoadSequenceDesc = CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC,
        /// Page table mapping descriptor.
        OperationPagedCacheLoadPageTableDesc = CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC,
        /// Normalization algorithm type.
        OperationNormFwdMode = CUDNN_ATTR_OPERATION_NORM_FWD_MODE,
        /// Training vs inference phase.
        OperationNormFwdPhase = CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
        /// Input tensor descriptor.
        OperationNormFwdXDesc = CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
        /// Computed or cached mean tensor.
        OperationNormFwdMeanDesc = CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC,
        /// Computed or cached inverse variance.
        OperationNormFwdInvVarianceDesc = CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
        /// Learnable scale parameter (gamma).
        OperationNormFwdScaleDesc = CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
        /// Learnable bias parameter (beta).
        OperationNormFwdBiasDesc = CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
        /// Numerical stability constant.
        OperationNormFwdEpsilonDesc = CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
        /// Momentum for running statistics.
        OperationNormFwdExpAvgFactorDesc = CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
        /// Input running mean.
        OperationNormFwdInputRunningMeanDesc = CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC,
        /// Input running variance.
        OperationNormFwdInputRunningVarDesc = CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC,
        /// Updated running mean output.
        OperationNormFwdOutputRunningMeanDesc = CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
        /// Updated running variance output.
        OperationNormFwdOutputRunningVarDesc = CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC,
        /// Output tensor descriptor.
        OperationNormFwdYDesc = CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
        /// Peer statistics for multi-GPU sync.
        OperationNormFwdPeerStatDescs = CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS,
        /// Normalization algorithm type.
        OperationNormBwdMode = CUDNN_ATTR_OPERATION_NORM_BWD_MODE,
        /// Forward input tensor reference.
        OperationNormBwdXDesc = CUDNN_ATTR_OPERATION_NORM_BWD_XDESC,
        /// Cached mean from forward pass.
        OperationNormBwdMeanDesc = CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC,
        /// Cached inverse std dev.
        OperationNormBwdInvVarianceDesc = CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC,
        /// Output gradient tensor.
        OperationNormBwdDyDesc = CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC,
        /// Forward scale parameter.
        OperationNormBwdScaleDesc = CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC,
        /// Numerical stability epsilon.
        OperationNormBwdEpsilonDesc = CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC,
        /// Scale gradient output.
        OperationNormBwdDScaleDesc = CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC,
        /// Bias gradient output.
        OperationNormBwdDBiasDesc = CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC,
        /// Input gradient output tensor.
        OperationNormBwdDxDesc = CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC,
        /// Peer gradient statistics for multi-GPU.
        OperationNormBwdPeerStatDescs = CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS,
        /// Reshape input tensor descriptor.
        OperationReshapeXDesc = CUDNN_ATTR_OPERATION_RESHAPE_XDESC,
        /// Reshape output tensor descriptor.
        OperationReshapeYDesc = CUDNN_ATTR_OPERATION_RESHAPE_YDESC,
        /// Band matrix input tensor.
        OperationExpandBandMatrixXDesc = CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_XDESC,
        /// Expanded output tensor.
        OperationExpandBandMatrixYDesc = CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_YDESC,
        /// Lower bandwidth of the band.
        OperationExpandBandMatrixLowerBandwidth = CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_LOWER_BANDWIDTH,
        /// Upper bandwidth of the band.
        OperationExpandBandMatrixUpperBandwidth = CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_UPPER_BANDWIDTH,
        /// Axis along which to expand.
        OperationExpandBandMatrixAxis = CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_AXIS,
        /// Padding value outside the band.
        OperationExpandBandMatrixPadValue = CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_PAD_VALUE,
        /// KV token offset descriptor.
        OperationExpandBandMatrixKvTokenOffsetDesc = CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_KV_TOKEN_OFFSET_DESC,
        /// Speculative decoding mask.
        OperationExpandBandMatrixSpeculativeMaskDesc = CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_SPECULATIVE_MASK_DESC,
        /// Full matrix input tensor.
        OperationContractBandMatrixXDesc = CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_XDESC,
        /// Contracted band output tensor.
        OperationContractBandMatrixYDesc = CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_YDESC,
        /// Lower bandwidth.
        OperationContractBandMatrixLowerBandwidth = CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_LOWER_BANDWIDTH,
        /// Upper bandwidth.
        OperationContractBandMatrixUpperBandwidth = CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_UPPER_BANDWIDTH,
        /// Axis along which to contract.
        OperationContractBandMatrixAxis = CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_AXIS,
        /// Padding value.
        OperationContractBandMatrixPadValue = CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_PAD_VALUE,
        /// Maximum token value for contraction.
        OperationContractBandMaxTokenValue = CUDNN_ATTR_OPERATION_CONTRACT_BAND_MAX_TOKEN_VALUE,
        /// Random distribution type selection.
        RngDistribution = CUDNN_ATTR_RNG_DISTRIBUTION,
        /// Normal distribution mean parameter.
        RngNormalDistMean = CUDNN_ATTR_RNG_NORMAL_DIST_MEAN,
        /// Normal distribution std deviation.
        RngNormalDistStandardDeviation = CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION,
        /// Uniform distribution upper bound.
        RngUniformDistMaximum = CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM,
        /// Uniform distribution lower bound.
        RngUniformDistMinimum = CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM,
        /// Bernoulli probability of 1.
        RngBernoulliDistProbability = CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY,
        /// RNG output tensor descriptor.
        OperationRngYDesc = CUDNN_ATTR_OPERATION_RNG_YDESC,
        /// RNG seed value.
        OperationRngSeed = CUDNN_ATTR_OPERATION_RNG_SEED,
        /// RNG descriptor reference.
        OperationRngDesc = CUDNN_ATTR_OPERATION_RNG_DESC,
        /// RNG offset/state descriptor.
        OperationRngOffsetDesc = CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC,
        /// Whether kernel is cached.
        KernelCacheIsEngineCfgKernelCached = CUDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED,
        /// Operation graph for kernel cache.
        KernelCacheOperationGraph = CUDNN_ATTR_KERNEL_CACHE_OPERATION_GRAPH,
        /// Kernel cache JSON serialization.
        KernelCacheJsonRepresentation = CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION,
        /// Input float tensor to quantize.
        OperationBlockScaleQuantizeXDesc = CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
        /// Quantized output tensor.
        OperationBlockScaleQuantizeYDesc = CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC,
        /// Per-block scaling factors output.
        OperationBlockScaleQuantizeScaleDesc = CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC,
        /// Computation precision.
        OperationBlockScaleQuantizeMathPrec = CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
        /// Quantization block size.
        OperationBlockScaleQuantizeBlockSize = CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE,
        /// Quantized input tensor.
        OperationBlockScaleDequantizeXDesc = CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
        /// Per-block scale factors.
        OperationBlockScaleDequantizeScaleDesc = CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
        /// Dequantized output tensor.
        OperationBlockScaleDequantizeYDesc = CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC,
        /// Computation precision.
        OperationBlockScaleDequantizeMathPrec = CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
        /// Dequantization block size.
        OperationBlockScaleDequantizeBlockSize = CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
        /// Negative scale handling.
        OperationBlockScaleDequantizeNegScale = CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
        /// Query cumulative sequence length tensor.
        OperationSdpaFwdCuSeqLenQDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_QDESC,
        /// Key-value cumulative sequence length tensor.
        OperationSdpaFwdCuSeqLenKvDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_KVDESC,
        /// Query cumulative sequence length tensor.
        OperationDiagonalBandMaskCuSeqLenQDesc = CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_QDESC,
        /// Key-value cumulative sequence length tensor.
        OperationDiagonalBandMaskCuSeqLenKvDesc = CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_KVDESC,
        /// FP8 descale factor for Q.
        OperationSdpaFwdDescaleQDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_QDESC,
        /// FP8 descale factor for K.
        OperationSdpaFwdDescaleKDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_KDESC,
        /// FP8 descale factor for V.
        OperationSdpaFwdDescaleVDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_VDESC,
        /// FP8 descale factor for S.
        OperationSdpaFwdDescaleSDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_SDESC,
        /// FP8 scale factor for S.
        OperationSdpaFwdScaleSDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_SDESC,
        /// FP8 scale factor for O.
        OperationSdpaFwdScaleODesc = CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_ODESC,
        /// FP8 absolute maximum of S.
        OperationSdpaFwdAmaxSDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_SDESC,
        /// FP8 absolute maximum of O.
        OperationSdpaFwdAmaxODesc = CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_ODESC,
        /// Query tensor descriptor.
        OperationSdpaFwdQDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC,
        /// Key tensor descriptor.
        OperationSdpaFwdKDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC,
        /// Value tensor descriptor.
        OperationSdpaFwdVDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC,
        /// Output tensor descriptor.
        OperationSdpaFwdODesc = CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC,
        /// Statistics output descriptor.
        OperationSdpaFwdStatsDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC,
        /// Attention scaling factor descriptor.
        OperationSdpaFwdScaleDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
        /// Block-sparse attention mask.
        OperationSdpaFwdBlockMaskDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC,
        /// Paged attention key page table.
        OperationSdpaFwdPageTableKDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC,
        /// Paged attention value page table.
        OperationSdpaFwdPageTableVDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC,
        /// Query sequence length tensor.
        OperationSdpaFwdSeqLenQDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC,
        /// Key-value sequence length tensor.
        OperationSdpaFwdSeqLenKvDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC,
        /// Softmax descriptor.
        OperationSdpaFwdSoftmaxDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_SOFTMAX_DESC,
        /// Forward SDPA subgraph.
        OperationSdpaFwdSubgraph = CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH,
        /// Subgraph input tensor UID.
        OperationSdpaFwdSubgraphInputId = CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID,
        /// Subgraph output tensor UID.
        OperationSdpaFwdSubgraphOutputId = CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID,
        /// Dropout seed tensor descriptor.
        OperationSdpaFwdDropoutSeedDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SEED_DESC,
        /// Dropout offset tensor descriptor.
        OperationSdpaFwdDropoutOffsetDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_OFFSET_DESC,
        /// Dropout RNG dump tensor descriptor.
        OperationSdpaFwdDropoutRngDumpDesc = CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_RNG_DUMP_DESC,
        /// Dropout probability.
        OperationSdpaFwdDropoutProbability = CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_PROBABILITY,
        /// Unfuse FMA in softmax for SM100.
        OperationSdpaFwdUnfuseFma = CUDNN_ATTR_OPERATION_SDPA_FWD_UNFUSE_FMA,
        /// Query tensor descriptor.
        OperationSdpaBwdQDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC,
        /// Key tensor descriptor.
        OperationSdpaBwdKDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC,
        /// Value tensor descriptor.
        OperationSdpaBwdVDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC,
        /// Forward output tensor descriptor.
        OperationSdpaBwdODesc = CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC,
        /// Forward statistics descriptor.
        OperationSdpaBwdStatsDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC,
        /// Attention scaling factor.
        OperationSdpaBwdScaleDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC,
        /// Query sequence length tensor.
        OperationSdpaBwdSeqLenQDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC,
        /// Key-value sequence length tensor.
        OperationSdpaBwdSeqLenKvDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC,
        /// Query gradient output tensor.
        OperationSdpaBwdDqDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC,
        /// Key gradient output tensor.
        OperationSdpaBwdDkDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC,
        /// Value gradient output tensor.
        OperationSdpaBwdDvDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC,
        /// Output gradient input tensor.
        OperationSdpaBwdDoDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC,
        /// Backward sink descriptor.
        OperationSdpaBwdSinkDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_SINK_DESC,
        /// Backward sink gradient descriptor.
        OperationSdpaBwdDsinkDesc = CUDNN_ATTR_OPERATION_SDPA_BWD_DSINK_DESC,
        /// Max total query sequence length.
        OperationSdpaBwdMaxTotalSeqLenQ = CUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_Q,
        /// Max total KV sequence length.
        OperationSdpaBwdMaxTotalSeqLenKv = CUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_KV,
        /// Backward SDPA subgraph.
        OperationSdpaBwdSubgraph = CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH,
        /// Subgraph input tensor UID.
        OperationSdpaBwdSubgraphInputId = CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_INPUT_UID,
        /// Subgraph output tensor UID.
        OperationSdpaBwdSubgraphOutputId = CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_OUTPUT_UID,
        /// Gather/scatter mode for `MoE`.
        OperationMoeGroupedMatmulMode = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
        /// Computation precision.
        OperationMoeGroupedMatmulMathPrec = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
        /// Token tensor descriptor.
        OperationMoeGroupedMatmulTokenDesc = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
        /// Expert weight tensor descriptor.
        OperationMoeGroupedMatmulWeightDesc = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
        /// First token offset per expert.
        OperationMoeGroupedMatmulFirstTokenOffsetDesc = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
        /// Output tensor descriptor.
        OperationMoeGroupedMatmulOutputDesc = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
        /// Token routing index descriptor.
        OperationMoeGroupedMatmulTokenIndexDesc = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
        /// Token routing weights descriptor.
        OperationMoeGroupedMatmulTokenKsDesc = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
        /// Top-K experts per token.
        OperationMoeGroupedMatmulTopK = CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K,
        /// Input tensor descriptor.
        OperationDiagonalBandMaskXDesc = CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC,
        /// KV sequence length tensor.
        OperationDiagonalBandMaskSeqLenKvDesc = CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC,
        /// Query sequence length tensor.
        OperationDiagonalBandMaskSeqLenQDesc = CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC,
        /// Left bound offset descriptor.
        OperationDiagonalBandMaskLeftBoundDesc = CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC,
        /// Right bound shift descriptor.
        OperationDiagonalBandMaskShiftRightBoundDesc = CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC,
        /// Band descriptor.
        OperationDiagonalBandMaskBDesc = CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC,
        /// Output mask tensor descriptor.
        OperationDiagonalBandMaskYDesc = CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC,
        /// Mask comparison mode.
        OperationDiagonalBandMaskComparisonMode = CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE,
        /// Softmax input tensor descriptor.
        OperationSoftmaxXDesc = CUDNN_ATTR_OPERATION_SOFTMAX_XDESC,
        /// Softmax output tensor descriptor.
        OperationSoftmaxYDesc = CUDNN_ATTR_OPERATION_SOFTMAX_YDESC,
        /// Softmax statistics output.
        OperationSoftmaxStatsDesc = CUDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC,
        /// Row-wise max values descriptor.
        OperationSoftmaxMaxDesc = CUDNN_ATTR_OPERATION_SOFTMAX_MAX_DESC,
        /// Row-wise sum of exponentials.
        OperationSoftmaxSumExpDesc = CUDNN_ATTR_OPERATION_SOFTMAX_SUM_EXP_DESC,
        /// Softmax sink descriptor.
        OperationSoftmaxSinkDesc = CUDNN_ATTR_OPERATION_SOFTMAX_SINK_DESC,
        /// CUDA device identifier.
        DevicePropDeviceId = CUDNN_ATTR_DEVICEPROP_DEVICE_ID,
        /// Associated cuDNN handle.
        DevicePropHandle = CUDNN_ATTR_DEVICEPROP_HANDLE,
        /// Device properties JSON.
        DevicePropJsonRepresentation = CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION,
    }
}

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

singe_core::native_enum! {
    /// [`BackendAttributeType`] specifies the data type of an attribute of a cuDNN backend descriptor.
    /// [`BackendDescriptor`](crate::backend::descriptor::BackendDescriptor) attribute methods use this enum to pass the matching cuDNN attribute type.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum BackendAttributeType: sys::cudnnBackendAttributeType_t {
        /// [`ReshapeMode`] value.
        ReshapeMode = CUDNN_TYPE_RESHAPE_MODE,
        /// [`Context`](crate::context::Context) handle value.
        Handle = CUDNN_TYPE_HANDLE,
        /// [`DataType`] value.
        DataType = CUDNN_TYPE_DATA_TYPE,
        /// Boolean value.
        Boolean = CUDNN_TYPE_BOOLEAN,
        /// 64-bit signed integer value.
        Int64 = CUDNN_TYPE_INT64,
        /// 32-bit float value.
        Float = CUDNN_TYPE_FLOAT,
        /// 64-bit double value.
        Double = CUDNN_TYPE_DOUBLE,
        /// Void pointer value.
        VoidPtr = CUDNN_TYPE_VOID_PTR,
        /// [`ConvolutionMode`] value.
        ConvolutionMode = CUDNN_TYPE_CONVOLUTION_MODE,
        /// [`BackendHeuristicMode`] value.
        HeurMode = CUDNN_TYPE_HEUR_MODE,
        /// [`BackendKnobType`] value.
        KnobType = CUDNN_TYPE_KNOB_TYPE,
        /// [`NanPropagation`] value.
        /// Deprecated by cuDNN.
        // Note: FFI has typo "PROPOGATION".
        NanPropagation = CUDNN_TYPE_NAN_PROPOGATION,
        /// [`BackendNumericalNote`] value.
        NumericalNote = CUDNN_TYPE_NUMERICAL_NOTE,
        /// [`BackendLayoutType`] value.
        LayoutType = CUDNN_TYPE_LAYOUT_TYPE,
        /// [`BackendAttributeName`] value.
        AttribName = CUDNN_TYPE_ATTRIB_NAME,
        /// [`PointwiseMode`] value.
        PointwiseMode = CUDNN_TYPE_POINTWISE_MODE,
        /// [`BackendDescriptor`](crate::backend::descriptor::BackendDescriptor) handle value.
        BackendDescriptor = CUDNN_TYPE_BACKEND_DESCRIPTOR,
        /// [`GenStatsMode`] value.
        GenStatsMode = CUDNN_TYPE_GENSTATS_MODE,
        /// [`BatchNormalizationFinalizeStatsMode`] value.
        BnFinalizeStatsMode = CUDNN_TYPE_BN_FINALIZE_STATS_MODE,
        /// [`ReduceTensorOperator`] value.
        ReductionOperatorType = CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
        /// [`BackendBehaviorNote`] value.
        BehaviorNote = CUDNN_TYPE_BEHAVIOR_NOTE,
        /// [`BackendTensorReordering`] value.
        TensorReorderingMode = CUDNN_TYPE_TENSOR_REORDERING_MODE,
        /// [`ResampleMode`] value.
        ResampleMode = CUDNN_TYPE_RESAMPLE_MODE,
        /// [`PaddingMode`] value.
        PaddingMode = CUDNN_TYPE_PADDING_MODE,
        /// 32-bit signed integer value.
        Int32 = CUDNN_TYPE_INT32,
        /// Character value.
        Char = CUDNN_TYPE_CHAR,
        /// [`SignalMode`] value.
        SignalMode = CUDNN_TYPE_SIGNAL_MODE,
        /// [`Fraction`](crate::math::Fraction) value.
        Fraction = CUDNN_TYPE_FRACTION,
        /// [`BackendNormalizationMode`] value.
        NormMode = CUDNN_TYPE_NORM_MODE,
        /// [`BackendNormalizationForwardPhase`] value.
        NormFwdPhase = CUDNN_TYPE_NORM_FWD_PHASE,
        /// [`RngDistribution`] value.
        RngDistribution = CUDNN_TYPE_RNG_DISTRIBUTION,
        /// [`MoeGroupedMatmulMode`] value.
        MoeGroupedMatmulMode = CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
    }
}

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
///
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
        // SAFETY: the assertion above checks that the native enum is 32 bits wide, and `$category` is the attribute type cuDNN uses for it.
        // The wrapper's `Into<u32>` and `TryFrom<u32>` use the native discriminants.
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
