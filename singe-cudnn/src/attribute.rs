use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_cudnn_sys as sys;

use singe_core::{impl_enum_conversion, impl_enum_display};

/// Backend descriptor attribute name that can be set or retrieved with
/// [`BackendDescriptor`](crate::descriptor::BackendDescriptor) attribute methods.
/// The backend descriptor to which an attribute belongs is identified by the prefix of the attribute name.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum BackendAttributeName {
    PointwiseMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_MODE as _,
    PointwiseMathPrec = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_MATH_PREC as _,
    PointwiseNanPropagation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_NAN_PROPAGATION as _,
    PointwiseReluLowerClip =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP as _,
    PointwiseReluUpperClip =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP as _,
    PointwiseReluLowerClipSlope =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE as _,
    PointwiseEluAlpha = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_ELU_ALPHA as _,
    PointwiseSoftplusBeta = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA as _,
    PointwiseSwishBeta = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_SWISH_BETA as _,
    PointwiseAxis = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_AXIS as _,

    ConvolutionCompType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_COMP_TYPE as _,
    ConvolutionConvMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_CONV_MODE as _,
    ConvolutionDilations = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_DILATIONS as _,
    ConvolutionFilterStrides =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES as _,
    ConvolutionPostPaddings =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_POST_PADDINGS as _,
    ConvolutionPrePaddings =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS as _,
    ConvolutionSpatialDims =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS as _,

    EngineHeurMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_MODE as _,
    EngineHeurOperationGraph =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH as _,
    EngineHeurResults = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_RESULTS as _,
    EngineHeurSmCountTarget =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET as _,
    EngineHeurDeviceProp = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_DEVICEPROP as _,

    EngineCfgEngine = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_ENGINE as _,
    EngineCfgIntermediateInfo =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO as _,
    EngineCfgKnobChoices = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_KNOB_CHOICES as _,
    EngineCfgWorkspaceSize =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE as _,
    EngineCfgSharedMemoryUsed = 304,
    ExecutionPlanHandle = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_HANDLE as _,
    ExecutionPlanEngineConfig =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG as _,
    ExecutionPlanWorkspaceSize =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE as _,
    ExecutionPlanComputedIntermediateIds =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS as _,
    ExecutionPlanRunOnlyIntermediateIds =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS as _,
    ExecutionPlanJsonRepresentation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION as _,
    ExecutionPlanKernelCache =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE as _,
    ExecutionPlanDeviceProp =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP as _,

    IntermediateInfoUniqueId =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID as _,
    IntermediateInfoSize = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_INTERMEDIATE_INFO_SIZE as _,
    IntermediateInfoDependentDataIds =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS as _,
    IntermediateInfoDependentAttributes =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES as _,

    KnobChoiceKnobType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE as _,
    KnobChoiceKnobValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE as _,

    OperationConvolutionForwardAlpha =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA as _,
    OperationConvolutionForwardBeta =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA as _,
    OperationConvolutionForwardConvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC as _,
    OperationConvolutionForwardW =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W as _,
    OperationConvolutionForwardX =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X as _,
    OperationConvolutionForwardY =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y as _,
    OperationConvolutionBwdDataAlpha =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA as _,
    OperationConvolutionBwdDataBeta =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA as _,
    OperationConvolutionBwdDataConvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC as _,
    OperationConvolutionBwdDataW =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W as _,
    OperationConvolutionBwdDataDx =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX as _,
    OperationConvolutionBwdDataDy =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY as _,
    OperationConvolutionBwdFilterAlpha =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA as _,
    OperationConvolutionBwdFilterBeta =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA as _,
    OperationConvolutionBwdFilterConvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC as _,
    OperationConvolutionBwdFilterDw =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW as _,
    OperationConvolutionBwdFilterX =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X as _,
    OperationConvolutionBwdFilterDy =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY as _,

    OperationPointwisePwDescriptor =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR as _,
    OperationPointwiseXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_XDESC as _,
    OperationPointwiseBDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_BDESC as _,
    OperationPointwiseYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_YDESC as _,
    OperationPointwiseAlpha1 =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1 as _,
    OperationPointwiseAlpha2 =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2 as _,
    OperationPointwiseDxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_DXDESC as _,
    OperationPointwiseDyDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_DYDESC as _,
    OperationPointwiseTDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_TDESC as _,

    OperationGenStatsMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_GENSTATS_MODE as _,
    OperationGenStatsMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC as _,
    OperationGenStatsXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_GENSTATS_XDESC as _,
    OperationGenStatsSumDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC as _,
    OperationGenStatsSqSumDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC as _,

    OperationBnFinalizeStatsMode =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE as _,
    OperationBnFinalizeMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC as _,
    OperationBnFinalizeYSumDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC as _,
    OperationBnFinalizeYSqSumDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC as _,
    OperationBnFinalizeScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC as _,
    OperationBnFinalizeBiasDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC as _,
    OperationBnFinalizePrevRunningMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC as _,
    OperationBnFinalizePrevRunningVarDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC as _,
    OperationBnFinalizeUpdatedRunningMeanDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC as _,
    OperationBnFinalizeUpdatedRunningVarDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC as _,
    OperationBnFinalizeSavedMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC as _,
    OperationBnFinalizeSavedInvStdDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC as _,
    OperationBnFinalizeEqScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC as _,
    OperationBnFinalizeEqBiasDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC as _,
    OperationBnFinalizeAccumCountDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC as _,
    OperationBnFinalizeEpsilonDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC as _,
    OperationBnFinalizeExpAverateFactorDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC as _,

    OperationGraphHandle = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_HANDLE as _,
    OperationGraphOps = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_OPS as _,
    OperationGraphEngineGlobalCount =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT as _,
    OperationGraphIsDynamicShapeEnabled =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED as _,
    OperationGraphIsSameTopology =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_IS_SAME_TOPOLOGY as _,
    OperationGraphIsOverrideShapeEnabled =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED as _,
    TensorByteAlignment = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT as _,
    TensorDataType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DATA_TYPE as _,
    TensorDimensions = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DIMENSIONS as _,
    TensorStrides = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_STRIDES as _,
    TensorVectorCount = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_VECTOR_COUNT as _,
    TensorVectorizedDimension =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION as _,
    TensorUniqueId = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_UNIQUE_ID as _,
    TensorIsVirtual = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_IS_VIRTUAL as _,
    TensorIsByValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_IS_BY_VALUE as _,
    TensorReorderingMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_REORDERING_MODE as _,
    TensorRaggedOffsetDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC as _,

    VariantPackUniqueIds = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS as _,
    VariantPackDataPointers =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS as _,
    VariantPackIntermediates =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES as _,
    VariantPackWorkspace = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_WORKSPACE as _,
    VariantPackOverrideUniqueIds =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS as _,
    VariantPackOverrideShapes =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES as _,
    VariantPackOverrideStrides =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES as _,

    LayoutInfoTensorId = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID as _,
    LayoutInfoTypes = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_LAYOUT_INFO_TYPES as _,

    KnobInfoType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_INFO_TYPE as _,
    KnobInfoMaximumValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE as _,
    KnobInfoMinimumValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE as _,
    KnobInfoStride = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_INFO_STRIDE as _,

    EngineOperationGraph = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_OPERATION_GRAPH as _,
    EngineGlobalIndex = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_GLOBAL_INDEX as _,
    EngineKnobInfo = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_KNOB_INFO as _,
    EngineNumericalNote = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_NUMERICAL_NOTE as _,
    EngineLayoutInfo = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_LAYOUT_INFO as _,
    EngineBehaviorNote = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE as _,
    EngineSmCountTarget = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_SM_COUNT_TARGET as _,
    EngineDeviceProp = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_DEVICEPROP as _,

    MatmulCompType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_MATMUL_COMP_TYPE as _,
    MatmulPaddingValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_MATMUL_PADDING_VALUE as _,

    OperationMatmulADesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_ADESC as _,
    OperationMatmulBDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_BDESC as _,
    OperationMatmulCDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_CDESC as _,
    OperationMatmulDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_DESC as _,
    OperationMatmulIrregularlyStridedBatchCount = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT as _,
    OperationMatmulGemmMOverrideDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC as _,
    OperationMatmulGemmNOverrideDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC as _,
    OperationMatmulGemmKOverrideDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC as _,

    ReductionOperator = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_OPERATOR as _,
    ReductionCompType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_COMP_TYPE as _,
    ReductionIsDeterministic =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC as _,

    OperationReductionXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_XDESC as _,
    OperationReductionYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_YDESC as _,
    OperationReductionDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_REDUCTION_DESC as _,

    OperationBnBwdWeightsMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC as _,
    OperationBnBwdWeightsMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC as _,
    OperationBnBwdWeightsInvStdDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC as _,
    OperationBnBwdWeightsBnScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC as _,
    OperationBnBwdWeightsXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC as _,
    OperationBnBwdWeightsDyDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC as _,
    OperationBnBwdWeightsDbnScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC as _,
    OperationBnBwdWeightsDbnBiasDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC as _,
    OperationBnBwdWeightsEqDyScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC as _,
    OperationBnBwdWeightsEqXScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC as _,
    OperationBnBwdWeightsEqBias =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS as _,

    ResampleMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_MODE as _,
    ResampleCompType = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_COMP_TYPE as _,
    ResampleSpatialDims = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS as _,
    ResamplePostPaddings = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_POST_PADDINGS as _,
    ResamplePrePaddings = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_PRE_PADDINGS as _,
    ResampleStrides = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_STRIDES as _,
    ResampleWindowDims = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_WINDOW_DIMS as _,
    ResampleNanPropagation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION as _,
    ResamplePaddingMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RESAMPLE_PADDING_MODE as _,

    OperationResampleFwdXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC as _,
    OperationResampleFwdYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC as _,
    OperationResampleFwdIdxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC as _,
    OperationResampleFwdAlpha =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA as _,
    OperationResampleFwdBeta =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA as _,
    OperationResampleFwdDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC as _,

    OperationResampleBwdDxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC as _,
    OperationResampleBwdDyDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC as _,
    OperationResampleBwdIdxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC as _,
    OperationResampleBwdAlpha =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA as _,
    OperationResampleBwdBeta =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA as _,
    OperationResampleBwdDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC as _,
    OperationResampleBwdXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC as _,
    OperationResampleBwdYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC as _,

    OperationConcatAxis = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONCAT_AXIS as _,
    OperationConcatInputDescs =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS as _,
    OperationConcatInplaceIndex =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX as _,
    OperationConcatOutputDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC as _,

    OperationSignalMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SIGNAL_MODE as _,
    OperationSignalFlagDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC as _,
    OperationSignalValue = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SIGNAL_VALUE as _,
    OperationSignalXDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SIGNAL_XDESC as _,
    OperationSignalYDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SIGNAL_YDESC as _,

    OperationPagedCacheLoadContainerDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC as _,
    OperationPagedCacheLoadYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC as _,
    OperationPagedCacheLoadSequenceDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC as _,
    OperationPagedCacheLoadPageTableDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC as _,

    OperationNormFwdMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_MODE as _,
    OperationNormFwdPhase = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_PHASE as _,
    OperationNormFwdXDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_XDESC as _,
    OperationNormFwdMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC as _,
    OperationNormFwdInvVarianceDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC as _,
    OperationNormFwdScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC as _,
    OperationNormFwdBiasDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC as _,
    OperationNormFwdEpsilonDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC as _,
    OperationNormFwdExpAvgFactorDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC as _,
    OperationNormFwdInputRunningMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC as _,
    OperationNormFwdInputRunningVarDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC as _,
    OperationNormFwdOutputRunningMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC as _,
    OperationNormFwdOutputRunningVarDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC as _,
    OperationNormFwdYDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_YDESC as _,
    OperationNormFwdPeerStatDescs =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS as _,

    OperationNormBwdMode = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_MODE as _,
    OperationNormBwdXDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_XDESC as _,
    OperationNormBwdMeanDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC as _,
    OperationNormBwdInvVarianceDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC as _,
    OperationNormBwdDyDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC as _,
    OperationNormBwdScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC as _,
    OperationNormBwdEpsilonDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC as _,
    OperationNormBwdDScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC as _,
    OperationNormBwdDBiasDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC as _,
    OperationNormBwdDxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC as _,
    OperationNormBwdPeerStatDescs =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS as _,

    OperationReshapeXDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESHAPE_XDESC as _,
    OperationReshapeYDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RESHAPE_YDESC as _,

    OperationExpandBandMatrixXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_XDESC as _,
    OperationExpandBandMatrixYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_YDESC as _,
    OperationExpandBandMatrixLowerBandwidth =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_LOWER_BANDWIDTH as _,
    OperationExpandBandMatrixUpperBandwidth =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_UPPER_BANDWIDTH as _,
    OperationExpandBandMatrixAxis =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_AXIS as _,
    OperationExpandBandMatrixPadValue =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_PAD_VALUE as _,
    OperationExpandBandMatrixKvTokenOffsetDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_KV_TOKEN_OFFSET_DESC as _,
    OperationExpandBandMatrixSpeculativeMaskDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_SPECULATIVE_MASK_DESC as _,

    OperationContractBandMatrixXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_XDESC as _,
    OperationContractBandMatrixYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_YDESC as _,
    OperationContractBandMatrixLowerBandwidth =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_LOWER_BANDWIDTH as _,
    OperationContractBandMatrixUpperBandwidth =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_UPPER_BANDWIDTH as _,
    OperationContractBandMatrixAxis =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_AXIS as _,
    OperationContractBandMatrixPadValue =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_PAD_VALUE as _,
    OperationContractBandMaxTokenValue =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONTRACT_BAND_MAX_TOKEN_VALUE as _,

    RngDistribution = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_DISTRIBUTION as _,
    RngNormalDistMean = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_NORMAL_DIST_MEAN as _,
    RngNormalDistStandardDeviation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION as _,
    RngUniformDistMaximum = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM as _,
    RngUniformDistMinimum = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM as _,
    RngBernoulliDistProbability =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY as _,

    OperationRngYDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RNG_YDESC as _,
    OperationRngSeed = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RNG_SEED as _,
    OperationRngDesc = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RNG_DESC as _,
    OperationRngOffsetDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC as _,

    KernelCacheIsEngineCfgKernelCached =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED as _,
    KernelCacheOperationGraph =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KERNEL_CACHE_OPERATION_GRAPH as _,
    KernelCacheJsonRepresentation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION as _,
    OperationBlockScaleQuantizeXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC as _,
    OperationBlockScaleQuantizeYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC as _,
    OperationBlockScaleQuantizeScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC as _,
    OperationBlockScaleQuantizeMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC as _,
    OperationBlockScaleQuantizeBlockSize =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE as _,

    OperationBlockScaleDequantizeXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC as _,
    OperationBlockScaleDequantizeScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC as _,
    OperationBlockScaleDequantizeYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC as _,
    OperationBlockScaleDequantizeMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC as _,
    OperationBlockScaleDequantizeBlockSize =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE as _,
    OperationBlockScaleDequantizeNegScale =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE as _,

    OperationSdpaFwdQDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC as _,
    OperationSdpaFwdKDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC as _,
    OperationSdpaFwdVDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC as _,
    OperationSdpaFwdODesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC as _,
    OperationSdpaFwdStatsDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC as _,
    OperationSdpaFwdScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC as _,
    OperationSdpaFwdBlockMaskDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC as _,
    OperationSdpaFwdPageTableKDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC as _,
    OperationSdpaFwdPageTableVDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC as _,
    OperationSdpaFwdSeqLenQDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC as _,
    OperationSdpaFwdSeqLenKvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC as _,
    OperationSdpaFwdSoftmaxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SOFTMAX_DESC as _,
    OperationSdpaFwdSubgraph =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH as _,
    OperationSdpaFwdSubgraphInputId =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID as _,
    OperationSdpaFwdSubgraphOutputId =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID as _,
    OperationSdpaFwdDropoutSeedDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SEED_DESC as _,
    OperationSdpaFwdDropoutOffsetDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_OFFSET_DESC as _,
    OperationSdpaFwdDropoutRngDumpDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_RNG_DUMP_DESC as _,
    OperationSdpaFwdDropoutProbability =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_PROBABILITY as _,
    OperationSdpaFwdUnfuseFma =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_FWD_UNFUSE_FMA as _,
    OperationSdpaBwdQDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC as _,
    OperationSdpaBwdKDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC as _,
    OperationSdpaBwdVDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC as _,
    OperationSdpaBwdODesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC as _,
    OperationSdpaBwdStatsDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC as _,
    OperationSdpaBwdScaleDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC as _,
    OperationSdpaBwdSeqLenQDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC as _,
    OperationSdpaBwdSeqLenKvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC as _,
    OperationSdpaBwdDqDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC as _,
    OperationSdpaBwdDkDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC as _,
    OperationSdpaBwdDvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC as _,
    OperationSdpaBwdDoDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC as _,
    OperationSdpaBwdSinkDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SINK_DESC as _,
    OperationSdpaBwdDsinkDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_DSINK_DESC as _,
    OperationSdpaBwdMaxTotalSeqLenQ =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_Q as _,
    OperationSdpaBwdMaxTotalSeqLenKv =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_KV as _,
    OperationSdpaBwdSubgraph =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH as _,
    OperationSdpaBwdSubgraphInputId =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_INPUT_UID as _,
    OperationSdpaBwdSubgraphOutputId =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_OUTPUT_UID as _,
    OperationMoeGroupedMatmulMode =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE as _,
    OperationMoeGroupedMatmulMathPrec =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC as _,
    OperationMoeGroupedMatmulTokenDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC as _,
    OperationMoeGroupedMatmulWeightDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC as _,
    OperationMoeGroupedMatmulFirstTokenOffsetDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC as _,
    OperationMoeGroupedMatmulOutputDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC as _,
    OperationMoeGroupedMatmulTokenIndexDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC as _,
    OperationMoeGroupedMatmulTokenKsDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC as _,
    OperationMoeGroupedMatmulTopK =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K as _,
    OperationDiagonalBandMaskXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC as _,
    OperationDiagonalBandMaskSeqLenKvDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC as _,
    OperationDiagonalBandMaskSeqLenQDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC as _,
    OperationDiagonalBandMaskLeftBoundDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC as _,
    OperationDiagonalBandMaskShiftRightBoundDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC as _,
    OperationDiagonalBandMaskBDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC as _,
    OperationDiagonalBandMaskYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC as _,
    OperationDiagonalBandMaskComparisonMode =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE as _,
    OperationSoftmaxXDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_XDESC as _,
    OperationSoftmaxYDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_YDESC as _,
    OperationSoftmaxStatsDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC as _,
    OperationSoftmaxMaxDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_MAX_DESC as _,
    OperationSoftmaxSumExpDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_SUM_EXP_DESC as _,
    OperationSoftmaxSinkDesc =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_SOFTMAX_SINK_DESC as _,

    DevicePropDeviceId = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_DEVICEPROP_DEVICE_ID as _,
    DevicePropHandle = sys::cudnnBackendAttributeName_t::CUDNN_ATTR_DEVICEPROP_HANDLE as _,
    DevicePropJsonRepresentation =
        sys::cudnnBackendAttributeName_t::CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION as _,
}

impl_enum_conversion!(sys::cudnnBackendAttributeName_t, BackendAttributeName);

impl_enum_display!(BackendAttributeName, {
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
    Self::IntermediateInfoDependentDataIds => "CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS",
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
    Self::LayoutInfoTensorId => "CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID",
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
/// [`BackendDescriptor`](crate::descriptor::BackendDescriptor) attribute methods use this enum to pass the matching cuDNN attribute type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BackendAttributeType {
    Handle = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE as _,
    DataType = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE as _,
    Boolean = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BOOLEAN as _,
    Int64 = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64 as _,
    Float = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT as _,
    Double = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE as _,
    VoidPtr = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR as _,
    ConvolutionMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_CONVOLUTION_MODE as _,
    HeurMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_HEUR_MODE as _,
    KnobType = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_KNOB_TYPE as _,
    NanPropagation = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_NAN_PROPOGATION as _, // Note: FFI has typo "PROPOGATION".
    NumericalNote = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_NUMERICAL_NOTE as _,
    LayoutType = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_LAYOUT_TYPE as _,
    AttribName = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_ATTRIB_NAME as _,
    PointwiseMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_POINTWISE_MODE as _,
    BackendDescriptor = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR as _,
    GenStatsMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_GENSTATS_MODE as _,
    BnFinalizeStatsMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BN_FINALIZE_STATS_MODE as _,
    ReductionOperatorType =
        sys::cudnnBackendAttributeType_t::CUDNN_TYPE_REDUCTION_OPERATOR_TYPE as _,
    BehaviorNote = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BEHAVIOR_NOTE as _,
    TensorReorderingMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_TENSOR_REORDERING_MODE as _,
    ResampleMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_RESAMPLE_MODE as _,
    PaddingMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_PADDING_MODE as _,
    Int32 = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT32 as _,
    Char = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_CHAR as _,
    SignalMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_SIGNAL_MODE as _,
    Fraction = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FRACTION as _,
    NormMode = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_NORM_MODE as _,
    NormFwdPhase = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_NORM_FWD_PHASE as _,
    RngDistribution = sys::cudnnBackendAttributeType_t::CUDNN_TYPE_RNG_DISTRIBUTION as _,
    MoeGroupedMatmulMode =
        sys::cudnnBackendAttributeType_t::CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE as _,
}

impl_enum_conversion!(sys::cudnnBackendAttributeType_t, BackendAttributeType,);

impl_enum_display!(BackendAttributeType, {
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
