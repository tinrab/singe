use std::{
    borrow::Cow,
    fmt::{self, Write as _},
    io::Error as IoError,
    sync::Arc,
};

use thiserror::Error;

use crate::{
    backend::{
        engine::EngineIndex,
        execution::{EngineChoice, attention::SoftmaxStatsBase},
        knob::BackendKnobType,
        tensor::{BackendTensorUid, TensorId, TensorPlacement},
    },
    data_type::DataType,
    error::Error as CudnnError,
    frontend::plan::{PlanBuildReport, PlanFailure, PlanId},
};

/// Errors raised while building, preparing, compiling or executing frontend graphs.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// An output has neither an explicit storage encoding nor a graph data-type policy for its placement.
    #[error(
        "`{output}` requires an explicit encoding or a graph data-type policy for its placement"
    )]
    OutputDataTypeRequired {
        /// Output requiring an explicit storage encoding or graph data-type policy.
        output: &'static str,
    },

    /// The frontend logging-level environment setting is not an integer.
    #[error("invalid CUDNN_FRONTEND_LOG_INFO `{value}`: expected an integer")]
    LoggingLevelInvalid {
        /// Unparseable `CUDNN_FRONTEND_LOG_INFO` environment value.
        value: String,
    },

    /// Writing frontend log output failed.
    #[error("logging i/o failed: {source}")]
    LoggingIo {
        /// Underlying error that caused this failure.
        source: IoError,
    },

    /// A frontend logging mutex is poisoned.
    #[error("logging `{name}` lock is poisoned")]
    LoggingLockPoisoned {
        /// Name of the poisoned logging mutex.
        name: &'static str,
    },

    /// The attention-planning workspace limit is neither `-1` nor a nonnegative byte count.
    #[error(
        "invalid CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT `{value}`: expected -1 or a non-negative byte count"
    )]
    SdpaWorkspaceLimitInvalid {
        /// Invalid `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT` environment value.
        value: String,
    },

    /// An SDPA request reached a stage that requires its graph expansion to be complete.
    #[error("sdpa request must be expanded during graph preparation")]
    SdpaRequestNotExpanded,

    /// An SDPA implementation does not support a requested capability.
    #[error("{operation} does not support {feature}{}", on_sm(*.sm_version))]
    SdpaUnsupported {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Requested feature that is unavailable or requires a newer runtime.
        feature: SdpaFeature,
        /// Target SM architecture version, or `None` when unavailable.
        sm_version: Option<i32>,
    },

    /// An SDPA operation is missing a required input, capability, or target setting.
    #[error("{operation} requires {requirement}")]
    SdpaRequires {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Missing SDPA input, capability, or target requirement.
        requirement: SdpaRequirement,
    },

    /// Input, filter, and output channel counts cannot form valid convolution groups.
    #[error(
        "convolution channels cannot form groups: input={input_channels}, filter={filter_channels}, output={output_channels}"
    )]
    ConvolutionGroupChannelsMismatch {
        /// Number of input channels.
        input_channels: i64,
        /// Number of input channels per convolution filter.
        filter_channels: i64,
        /// Number of output channels.
        output_channels: i64,
    },

    /// A convolution filter exceeds the kernel's signed 32-bit element-offset limit.
    #[error(
        "convolution filter `{tensor_id}` has {elements} elements, exceeding the int32 kernel offset limit"
    )]
    ConvolutionFilterTooLarge {
        /// Identifier of the oversized convolution filter.
        tensor_id: TensorId,
        /// Number of elements in the convolution filter.
        elements: usize,
    },

    /// Accumulation was inferred without a bindable, initialized output tensor.
    #[error(
        "inferred {operation} accumulation requires a bindable result with initialized storage"
    )]
    AccumulationRequiresOutput {
        /// Operation whose requirements failed validation.
        operation: &'static str,
    },

    /// Restoring a native execution plan selected different engine or knob settings.
    #[error("restored native plan choice differs: expected {expected:?}, got {actual:?}")]
    NativePlanChoiceMismatch {
        /// Engine and knob configuration recorded by the serialized plan.
        expected: EngineChoice,
        /// Engine and knob configuration selected during restoration.
        actual: EngineChoice,
    },

    /// A serialized native execution plan could not be restored by the current cuDNN runtime.
    #[error(
        "failed to restore native plan {position} from cudnn {serialized_cudnn_version} with runtime {runtime_cudnn_version}: {source}"
    )]
    NativePlanRestoreFailed {
        /// Zero-based position of the native plan being restored.
        position: usize,
        /// cuDNN version recorded in the serialized artifact.
        serialized_cudnn_version: u64,
        /// cuDNN version of the runtime performing restoration.
        runtime_cudnn_version: u64,
        /// Underlying error that caused this failure.
        source: Box<CudnnError>,
    },

    /// A plan identifier does not belong to the collection or no longer names an available plan.
    #[error("plan `{plan}` does not belong to this collection or is no longer available")]
    PlanNotFound {
        /// Identifier of the unavailable or duplicated plan.
        plan: PlanId,
    },

    /// A requested plan order removes every plan.
    #[error("plan order must retain at least one plan")]
    PlanOrderEmpty,

    /// A requested plan order includes the same plan more than once.
    #[error("plan order contains plan `{plan}` more than once")]
    PlanOrderDuplicate {
        /// Identifier of the unavailable or duplicated plan.
        plan: PlanId,
    },

    /// A requested execution plan is unavailable; its build report records the outcome.
    #[error("execution plan is unavailable: {report}")]
    PlanUnavailable {
        /// Build report for the unavailable execution plan.
        report: Box<PlanBuildReport>,
    },

    /// No candidate execution plan is available after building the batch.
    #[error(
        "no candidate execution plan is available{}",
        plan_report_details(.reports)
    )]
    NoAvailablePlans {
        /// Build reports retained for the candidate execution plans.
        reports: Vec<PlanBuildReport>,
    },

    /// Autotuning could not obtain a timing for any candidate execution plan.
    #[error("no candidate execution plan could be timed")]
    AutotuneFailed {
        /// Failures recorded while compiling or timing the candidate plans.
        failures: Vec<PlanFailure>,
    },

    /// A plan cannot be timed with the supplied tuning layouts.
    #[error("plan `{plan}` is unavailable for these tuning layouts")]
    AutotunePlanUnavailable {
        /// Identifier of the unavailable or duplicated plan.
        plan: PlanId,
    },

    /// Custom tuning produced more than one timing for the same plan.
    #[error("custom tuning returned more than one timing for plan `{plan}`")]
    AutotuneDuplicatePlan {
        /// Identifier of the unavailable or duplicated plan.
        plan: PlanId,
    },

    /// An error aborted the autotuning session.
    #[error("autotune session aborted: {source}")]
    AutotuneAborted {
        /// Underlying error that caused this failure.
        source: Arc<CudnnError>,
    },

    /// Autotuning cannot determine the byte capacity of a ragged tensor binding.
    #[error(
        "autotune requires the byte capacity of ragged binding `{tensor_id}`; declare its ragged byte capacity or use a typed binding, encoded bytes or set_raw_buffer"
    )]
    AutotuneRaggedCapacityRequired {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
    },

    /// A normalization operation does not support the selected axes.
    #[error("unsupported normalization axes: {reason}")]
    NormalizationAxes {
        /// Explanation of the unsupported or invalid configuration.
        reason: Cow<'static, str>,
    },

    /// Requested affine gradients do not match the supplied normalization scale and bias-gradient configuration.
    #[error(
        "`{operation}` affine gradients must match the supplied scale and requested bias gradient"
    )]
    NormalizationAffineGradients {
        /// Operation whose requirements failed validation.
        operation: &'static str,
    },

    /// A tensor binding does not satisfy its required byte alignment.
    #[error("binding for tensor `{tensor_id}` does not meet its {required}-byte alignment")]
    BindingAlignmentMismatch {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Required binding-address alignment in bytes.
        required: usize,
    },

    /// Native SDPA does not support the requested output data type.
    #[error("native sdpa does not support output data type `{data_type}`")]
    SdpaOutputDataTypeUnsupported {
        /// Requested tensor or scalar storage data type.
        data_type: DataType,
    },

    /// SDPA backward received softmax statistics in a base other than the natural logarithm.
    #[error("sdpa backward requires natural-log softmax statistics, got {actual:?}")]
    SdpaBackwardStatsBaseUnsupported {
        /// Logarithm base used by the supplied softmax statistics.
        actual: SoftmaxStatsBase,
    },

    /// A frontend graph contains no operations.
    #[error("graph has no operations")]
    GraphEmpty,

    /// Related graph or operation settings do not agree.
    #[error("configuration mismatch for `{name}`")]
    ConfigurationMismatch {
        /// Name of the invalid configuration, binding, or internal state.
        name: &'static str,
    },

    /// A tensor identifier is negative.
    #[error("tensor id {value} must be non-negative")]
    TensorIdInvalid {
        /// Negative tensor identifier supplied by the caller.
        value: i64,
    },

    /// A requested frontend tensor is absent.
    #[error("tensor `{0}` was not found")]
    TensorNotFound(TensorId),

    /// A requested native tensor descriptor is absent.
    #[error("native tensor `{0}` was not found")]
    BackendTensorNotFound(BackendTensorUid),

    /// A runtime override targets a tensor without a native layout.
    #[error("tensor `{0}` has no native layout to override")]
    RuntimeOverrideNotNative(TensorId),

    /// A tensor required by execution has no binding.
    #[error("tensor `{0}` was not bound")]
    TensorNotBound(TensorId),

    /// A replacement binding refers to a source tensor that has no bound storage.
    #[error(
        "binding replacement source tensor `{source_id}` was not bound while preparing target `{target_id}` with byte offset {byte_offset}"
    )]
    BindingReplacementSourceNotBound {
        /// Identifier of the source tensor providing storage.
        source_id: TensorId,
        /// Identifier of the target tensor whose storage view is being prepared.
        target_id: TensorId,
        /// Requested signed byte offset into the source storage.
        byte_offset: i64,
    },

    /// A frontend tensor identifier is used more than once.
    #[error("tensor id `{0}` is duplicated")]
    TensorIdConflict(TensorId),

    /// Ragged-offset tensor references form a cycle.
    #[error("tensor `{tensor_id}` participates in a ragged offset cycle")]
    RaggedOffsetCycle {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
    },

    /// A tensor specifies only part of its required vectorization metadata.
    #[error("tensor `{tensor_id}` has incomplete vectorization metadata")]
    TensorVectorizationIncomplete {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
    },

    /// A scalar binding has a different scalar type than the tensor requires.
    #[error("tensor `{tensor_id}` has incompatible scalar type `{actual}` (expected `{expected}`)")]
    ScalarTypeMismatch {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Scalar type required by the tensor.
        expected: &'static str,
        /// Scalar type of the supplied value.
        actual: &'static str,
    },

    /// A binding uses a different Rust storage type than the tensor requires.
    #[error(
        "tensor `{tensor_id}` has incompatible binding type `{actual}` (expected `{expected}`)"
    )]
    BindingTypeMismatch {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Rust storage type required by the tensor.
        expected: &'static str,
        /// Rust storage type of the supplied binding.
        actual: &'static str,
    },

    /// Packed tensor storage requires an encoded-byte binding.
    #[error("tensor `{tensor_id}` uses packed `{data_type}` storage; bind its encoded bytes")]
    PackedBindingRequiresBytes {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Requested tensor or scalar storage data type.
        data_type: DataType,
    },

    /// A tensor binding has an incompatible host or device placement.
    #[error("tensor `{tensor_id}` binding requires {expected}, got {actual:?}")]
    BindingPlacementMismatch {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Required host or device placement.
        expected: &'static str,
        /// Placement of the supplied binding.
        actual: TensorPlacement,
    },

    /// A native tensor writes a binding that was supplied as read-only.
    #[error("binding `{binding_id}` is read-only, but native tensor `{tensor_id}` writes it")]
    BindingReadOnly {
        /// Identifier of the caller-supplied storage binding.
        binding_id: TensorId,
        /// Tensor whose native operation writes the binding.
        tensor_id: TensorId,
    },

    /// A binding has insufficient byte capacity, including the tensor's storage-view offset.
    #[error(
        "binding `{binding_id}` has {actual} bytes, but tensor `{tensor_id}` requires {required} bytes including its storage-view offset"
    )]
    BindingCapacityMismatch {
        /// Identifier of the caller-supplied storage binding.
        binding_id: TensorId,
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Required binding capacity in bytes, including the view offset.
        required: usize,
        /// Supplied binding capacity in bytes.
        actual: usize,
    },

    /// A tensor has a different data type than the operation requires.
    #[error(
        "tensor `{tensor_id}` has incompatible data type `{actual}` for {operation} (expected `{expected}`)"
    )]
    TensorDataTypeMismatch {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Storage data type required by the operation.
        expected: DataType,
        /// Data type of the supplied tensor.
        actual: DataType,
    },

    /// An operation does not support the tensor's data type.
    #[error(
        "tensor `{tensor_id}` has unsupported data type `{actual}` for {operation} (supported {supported:?})"
    )]
    TensorDataTypeUnsupported {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Storage data types supported by the operation.
        supported: Vec<DataType>,
        /// Unsupported data type of the supplied tensor.
        actual: DataType,
    },

    /// A tensor's extents differ from those required by the operation.
    #[error(
        "tensor `{tensor_id}` has incompatible dimensions for {operation} (expected {expected:?}, got {actual:?})"
    )]
    TensorDimensionsMismatch {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Tensor extents required by the operation.
        expected: Vec<i64>,
        /// Supplied tensor extents.
        actual: Vec<i64>,
    },

    /// A tensor's rank does not meet the operation's rank requirement.
    #[error(
        "tensor `{tensor_id}` has incompatible rank for {operation} (expected {expected}, got {actual})"
    )]
    TensorRankMismatch {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Exact or minimum tensor-rank requirement.
        expected: RankExpectation,
        /// Number of axes in the supplied tensor.
        actual: usize,
    },

    /// A tensor's element strides differ from those required by the operation.
    #[error(
        "tensor `{tensor_id}` has incompatible strides for {operation} (expected {expected:?}, got {actual:?})"
    )]
    TensorStridesMismatch {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Required strides in storage elements.
        expected: Vec<i64>,
        /// Supplied strides in storage elements.
        actual: Vec<i64>,
    },

    /// A tensor layout violates an operation-specific requirement.
    #[error("tensor `{tensor_id}` has incompatible layout for {operation}: {reason}")]
    TensorLayoutMismatch {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Explanation of the unsupported or invalid configuration.
        reason: Cow<'static, str>,
    },

    /// A tensor's final element stride differs from the required value.
    #[error(
        "tensor `{tensor_id}` has incompatible last stride for {operation} (expected {expected}, got {actual})"
    )]
    TensorLastStrideMismatch {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Required final-axis stride in storage elements.
        expected: i64,
        /// Supplied final-axis stride in storage elements.
        actual: i64,
    },

    /// An operation requires a tensor containing exactly one logical scalar.
    #[error("tensor `{tensor_id}` must contain one logical scalar for {operation}")]
    TensorScalarShapeMismatch {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Operation whose requirements failed validation.
        operation: &'static str,
    },

    /// A reshape changes the logical element count.
    #[error(
        "reshape input `{input_id}` and output `{output_id}` have different logical element counts"
    )]
    ReshapeElementCountMismatch {
        /// Identifier of the reshape input tensor.
        input_id: TensorId,
        /// Identifier of the reshape output tensor.
        output_id: TensorId,
    },

    /// A storage-view byte offset is not aligned to its data type's element size.
    #[error(
        "storage view `{target_id}` byte offset {byte_offset} is not aligned to element size {element_size} of source `{source_id}`"
    )]
    StorageViewByteOffsetUnaligned {
        /// Identifier of the source tensor providing storage.
        source_id: TensorId,
        /// Identifier of the target tensor whose storage view is being prepared.
        target_id: TensorId,
        /// Requested signed byte offset into the source storage.
        byte_offset: i64,
        /// Storage element size shared by source and target, in bytes.
        element_size: i64,
    },

    /// A storage view requests a byte range outside the source storage span.
    #[error(
        "storage view `{target_id}` byte range [{start}, {end}) is outside source `{source_id}` byte length {source_byte_len}"
    )]
    StorageViewByteRangeOutOfBounds {
        /// Identifier of the source tensor providing storage.
        source_id: TensorId,
        /// Identifier of the target tensor whose storage view is being prepared.
        target_id: TensorId,
        /// Reported byte-range start, inclusive.
        start: usize,
        /// Reported byte-range end, exclusive.
        end: usize,
        /// Available source storage span in bytes.
        source_byte_len: usize,
    },

    /// A storage view requests a bit range outside the source storage span.
    #[error(
        "storage view `{target_id}` bit range [{start_bits}, {end_bits}) is outside source `{source_id}` bit span {source_bits}"
    )]
    StorageViewBitRangeOutOfBounds {
        /// Identifier of the source tensor providing storage.
        source_id: TensorId,
        /// Identifier of the target tensor whose storage view is being prepared.
        target_id: TensorId,
        /// Reported bit-range start, inclusive.
        start_bits: u128,
        /// Reported bit-range end, exclusive.
        end_bits: u128,
        /// Available source storage span in bits.
        source_bits: u128,
    },

    /// A storage view refers to a source tensor that cannot supply bound storage.
    #[error("storage view source tensor `{source_id}` is not bindable: {reason}")]
    StorageViewSourceNotBindable {
        /// Identifier of the source tensor providing storage.
        source_id: TensorId,
        /// Explanation of the unsupported or invalid configuration.
        reason: &'static str,
    },

    /// A storage view or its source uses placement other than device memory.
    #[error(
        "storage view `{target_id}` and its source `{source_id}` must both use device placement"
    )]
    StorageViewPlacementMismatch {
        /// Identifier of the source tensor providing storage.
        source_id: TensorId,
        /// Identifier of the target tensor whose storage view is being prepared.
        target_id: TensorId,
    },

    /// Runtime shape bounds have incompatible ranks or invalid minimum and maximum extents.
    #[error(
        "runtime shape bounds are invalid (min dimensions {min_dimensions:?}, max dimensions {max_dimensions:?})"
    )]
    RuntimeShapeBoundsInvalid {
        /// Supplied per-axis minimum extents.
        min_dimensions: Vec<i64>,
        /// Supplied per-axis maximum extents.
        max_dimensions: Vec<i64>,
    },

    /// A fused scalar tensor is used at inconsistent ranks.
    #[error(
        "fused scalar tensor `{tensor_id}` is used with conflicting ranks ({first_rank} and {second_rank})"
    )]
    FusedScalarRankConflict {
        /// Identifier of the tensor whose layout, type, or binding is invalid.
        tensor_id: TensorId,
        /// Rank required by the first use of the fused scalar.
        first_rank: usize,
        /// Conflicting rank required by another use of the fused scalar.
        second_rank: usize,
    },

    /// Pointwise input shapes cannot broadcast together.
    #[error("pointwise inputs cannot broadcast together (lhs {lhs:?}, rhs {rhs:?})")]
    PointwiseBroadcastShapeMismatch {
        /// Left input tensor extents.
        lhs: Vec<i64>,
        /// Right input tensor extents.
        rhs: Vec<i64>,
    },

    /// Matrix-multiplication input shapes are incompatible.
    #[error(
        "matmul inputs `{lhs_id}` and `{rhs_id}` have incompatible dimensions for {operation} (lhs {lhs:?}, rhs {rhs:?})"
    )]
    MatmulShapeMismatch {
        /// Identifier of the left matrix-multiplication input.
        lhs_id: TensorId,
        /// Identifier of the right matrix-multiplication input.
        rhs_id: TensorId,
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Left input tensor extents.
        lhs: Vec<i64>,
        /// Right input tensor extents.
        rhs: Vec<i64>,
    },

    /// A pointwise output shape is incompatible with the input broadcast shape.
    #[error(
        "pointwise {operation} broadcast shape {expected:?} is incompatible with output shape {actual:?}"
    )]
    PointwiseOutputShapeMismatch {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Shape obtained by broadcasting the pointwise inputs.
        expected: Vec<i64>,
        /// Supplied or declared pointwise output shape.
        actual: Vec<i64>,
    },

    /// SDPA query and key shapes are incompatible.
    #[error("sdpa query/key shapes are incompatible (query {query:?}, key {key:?})")]
    SdpaQueryKeyShapeMismatch {
        /// Query tensor extents.
        query: Vec<i64>,
        /// Key tensor extents.
        key: Vec<i64>,
    },

    /// SDPA attention-score and value shapes are incompatible.
    #[error("sdpa scores/value shapes are incompatible (scores {scores:?}, value {value:?})")]
    SdpaScoresValueShapeMismatch {
        /// Attention-score tensor extents.
        scores: Vec<i64>,
        /// Value tensor extents.
        value: Vec<i64>,
    },

    /// SDPA head counts are nonpositive or query heads are not divisible by key and value heads.
    #[error(
        "{operation} head counts must be positive and query heads divisible by key/value heads (query {query_heads}, key {key_heads}, value {value_heads})"
    )]
    SdpaGroupedQueryAttentionHeadsMismatch {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Number of query attention heads.
        query_heads: i64,
        /// Number of key attention heads.
        key_heads: i64,
        /// Number of value attention heads.
        value_heads: i64,
    },

    /// An SDPA operation is missing a requested output.
    #[error("{operation} requested `{output}` output is missing")]
    SdpaOutputMissing {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Name of the requested output.
        output: &'static str,
    },

    /// Automatic SDPA planning received conflicting explicit engine choices.
    #[error("automatic sdpa planning has conflicting engine overrides {first} and {second}")]
    SdpaHeuristicEngineConflict {
        /// First explicit engine choice.
        first: EngineIndex,
        /// Conflicting explicit engine choice.
        second: EngineIndex,
    },

    /// Automatic SDPA planning received conflicting values for one engine knob.
    #[error(
        "automatic sdpa planning has conflicting values {first} and {second} for engine {engine_index} knob `{knob_type}`"
    )]
    SdpaHeuristicKnobConflict {
        /// Engine for which incompatible knob settings were supplied.
        engine_index: EngineIndex,
        /// Engine knob with conflicting values.
        knob_type: BackendKnobType,
        /// First requested value for the engine knob.
        first: i64,
        /// Conflicting requested value for the same engine knob.
        second: i64,
    },

    /// An SDPA scalar uses an unsupported data type.
    #[error("sdpa scalar data type `{data_type}` is unsupported")]
    SdpaScalarDataTypeUnsupported {
        /// Requested tensor or scalar storage data type.
        data_type: DataType,
    },

    /// An optional SDPA output does not match the requested output configuration.
    #[error("sdpa optional output `{output}` request mismatch")]
    SdpaOptionalOutputMismatch {
        /// Name of the requested output.
        output: &'static str,
    },

    /// The unified SDPA implementation cannot handle the requested configuration.
    #[error("unified sdpa is unavailable for this configuration")]
    SdpaUnifiedImplementationUnavailable,

    /// A quantized SDPA operation does not support the requested auxiliary output.
    #[error("sdpa quantized output `{output}` is unsupported for {operation}")]
    SdpaQuantizedAuxOutputUnsupported {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Name of the requested output.
        output: &'static str,
    },

    /// Causal bottom-right masking conflicts with another attention modifier.
    #[error("sdpa causal bottom-right mask cannot be combined with {modifier}")]
    SdpaCausalBottomRightModifierConflict {
        /// Attention modifier that conflicts with the selected operation or mask.
        modifier: &'static str,
    },

    /// Causal bottom-right masking has a query sequence longer than the key/value sequence without a padding mask.
    #[error(
        "sdpa causal bottom-right query sequence length {query_sequence_length} exceeds key/value sequence length {key_sequence_length} without a padding mask"
    )]
    SdpaCausalBottomRightSequenceLengthUnsupported {
        /// Number of tokens in the query sequence.
        query_sequence_length: i64,
        /// Number of tokens in the key/value sequence.
        key_sequence_length: i64,
    },

    /// The configured key/value sequence limit differs from the bias tensor's key/value extent.
    #[error(
        "sdpa max sequence length key/value {max_sequence_length_key_value} does not match bias key/value dimension {bias_key_value_dimension}"
    )]
    SdpaMaxSequenceLengthBiasMismatch {
        /// Configured maximum key/value sequence length in tokens.
        max_sequence_length_key_value: i64,
        /// Key/value extent of the attention-bias tensor.
        bias_key_value_dimension: i64,
    },

    /// Internal dropout and a custom dropout mask were requested together.
    #[error("sdpa cannot combine internal dropout with custom dropout")]
    SdpaDropoutModeConflict,

    /// Sliding-window attention has a query sequence longer than the key/value sequence without a padding mask.
    #[error(
        "sdpa sliding window query sequence length {query_sequence_length} exceeds key/value sequence length {key_sequence_length} without a padding mask"
    )]
    SdpaSlidingWindowSequenceLengthUnsupported {
        /// Number of tokens in the query sequence.
        query_sequence_length: i64,
        /// Number of tokens in the key/value sequence.
        key_sequence_length: i64,
    },

    /// Sink-token attention was requested for a query sequence of length at most one.
    #[error(
        "sdpa sink token requires query sequence length greater than one (got {query_sequence_length})"
    )]
    SdpaSinkTokenRequiresNonDecode {
        /// Number of tokens in the query sequence.
        query_sequence_length: i64,
    },

    /// A custom attention-score subgraph conflicts with another modifier.
    #[error("{operation} score subgraph cannot be combined with {modifier}")]
    SdpaScoreSubgraphModifierConflict {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Attention modifier that conflicts with the selected operation or mask.
        modifier: &'static str,
    },

    /// SDPA backward causal bottom-right masking has a query sequence longer than the key/value sequence.
    #[error(
        "sdpa backward causal bottom-right query sequence length {query_sequence_length} exceeds key/value sequence length {key_sequence_length}"
    )]
    SdpaBackwardCausalBottomRightSequenceLengthUnsupported {
        /// Number of tokens in the query sequence.
        query_sequence_length: i64,
        /// Number of tokens in the key/value sequence.
        key_sequence_length: i64,
    },

    /// SDPA backward causal bottom-right masking requires query and key/value sequence lengths divisible by 64.
    #[error(
        "sdpa backward causal bottom-right requires query and key/value sequence lengths to be multiples of 64 (query {query_sequence_length}, key/value {key_sequence_length})"
    )]
    SdpaBackwardCausalBottomRightSequenceLengthAlignmentUnsupported {
        /// Number of tokens in the query sequence.
        query_sequence_length: i64,
        /// Number of tokens in the key/value sequence.
        key_sequence_length: i64,
    },

    /// A quantized SDPA backward operation does not support the requested head dimensions.
    #[error(
        "{operation} head dimensions are unsupported{} (d_qk {d_qk}, d_v {d_v})",
        on_sm(*.sm_version)
    )]
    SdpaQuantizedBackwardHeadDimensionUnsupported {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Query/key feature extent per attention head, in elements.
        d_qk: i64,
        /// Value feature extent per attention head, in elements.
        d_v: i64,
        /// Target SM architecture version, or `None` when unavailable.
        sm_version: Option<i32>,
    },

    /// The requested quantized SDPA output data type requires SM 100 or later.
    #[error("{operation} output data type `{output_io_type}` requires SM 100 or newer{}", on_sm(*.sm_version))]
    SdpaQuantizedOutputDataTypeRequiresSm100 {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Requested quantized attention-output storage data type.
        output_io_type: DataType,
        /// Target SM architecture version, or `None` when unavailable.
        sm_version: Option<i32>,
    },

    /// The selected SDPA configuration is unsupported for the runtime, target architecture, or head dimensions.
    #[error(
        "{operation} is unsupported for {reason} (cudnn {cudnn_version}{}, data type {data_type}, d_qk {d_qk}, d_v {d_v})",
        on_sm(*.sm_version)
    )]
    SdpaArchitectureUnsupported {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Explanation of the unsupported or invalid configuration.
        reason: &'static str,
        /// Detected cuDNN runtime version.
        cudnn_version: u64,
        /// Target SM architecture version, or `None` when unavailable.
        sm_version: Option<i32>,
        /// Requested tensor or scalar storage data type.
        data_type: DataType,
        /// Query/key feature extent per attention head, in elements.
        d_qk: i64,
        /// Value feature extent per attention head, in elements.
        d_v: i64,
    },

    /// Runtime layout overrides require a graph that enables dynamic shapes or shape overrides.
    #[error("runtime overrides require a dynamic-shape or override-shape graph")]
    RuntimeOverridesRequireShapeEnabledGraph,

    /// Native CUDA graph population requires an empty destination graph.
    #[error("cuda graph provided to populate must be empty")]
    CudaGraphPopulateRequiresEmptyGraph,

    /// A kernel cache was requested for a graph without dynamic shapes.
    #[error("kernel cache requires a dynamic-shape graph")]
    KernelCacheRequiresDynamicShape,

    /// Compilation without a live context requires explicit target-device properties.
    #[error("deviceless compilation requires device properties")]
    DevicelessRequiresDeviceProperties,

    /// An operation requires bindings associated with a compiled graph template.
    #[error("graph-aware bindings require a compiled graph template")]
    GraphAwareBindingsRequired,

    /// Legacy softmax received auxiliary outputs other than statistics alone or maximum plus exponential sum.
    #[error("legacy softmax supports no aux outputs, stats only, or max plus sum_exp")]
    SoftmaxLegacyOutputsUnsupported,

    /// Resampling indices were requested outside max-pool or nearest-neighbor mode.
    #[error("resample indices require max-pool or nearest-neighbor mode")]
    ResampleIndicesUnsupportedMode,

    /// Batch-normalization running-statistic inputs and outputs were not supplied together.
    #[error("batch norm running-stat inputs and outputs must be provided together")]
    BatchNormRunningStatsIncomplete,

    /// A required operation field is absent.
    #[error("{operation} requires `{field}`")]
    OperationFieldMissing {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Name of the missing operation field.
        field: &'static str,
    },

    /// A requested feature requires a newer cuDNN runtime.
    #[error(
        "feature `{feature}` requires cudnn >= {min_version} (detected version {actual_version})"
    )]
    FeatureRequiresVersion {
        /// Requested feature that is unavailable or requires a newer runtime.
        feature: &'static str,
        /// Minimum cuDNN version required by the feature.
        min_version: &'static str,
        /// Detected cuDNN runtime version.
        actual_version: u64,
    },

    /// A requested feature is unavailable for the supplied configuration.
    #[error("feature `{feature}` is unavailable: {reason}")]
    FeatureUnavailable {
        /// Requested feature that is unavailable or requires a newer runtime.
        feature: &'static str,
        /// Explanation of the unsupported or invalid configuration.
        reason: Cow<'static, str>,
    },

    /// The frontend kernel-cache mutex is poisoned.
    #[error("kernel cache lock is poisoned")]
    KernelCacheLockPoisoned,

    /// The frontend plan-cache mutex is poisoned.
    #[error("plan cache lock is poisoned")]
    PlanCacheLockPoisoned,

    /// An operation has no required output tensor.
    #[error("operation output is missing")]
    OperationMissingOutput,

    /// A required internal graph or planning state is absent.
    #[error("internal state `{name}` is missing")]
    InternalStateMissing {
        /// Name of the invalid configuration, binding, or internal state.
        name: &'static str,
    },

    /// A serialized graph or native artifact uses an incompatible schema version.
    #[error("graph schema version mismatch: {actual} (expected {expected})")]
    GraphSchemaVersionMismatch {
        /// Schema version required by the current frontend decoder.
        expected: String,
        /// Incompatible schema version read from the serialized artifact.
        actual: String,
    },

    /// A native compiled artifact contains no execution plans.
    #[error("native compiled artifact has no plans")]
    NativeArtifactMissingPlans,

    /// A native artifact has no graph structure available for serialization.
    #[error("native compiled artifact has no graph structure to serialize")]
    NativeArtifactMissingStructure,

    /// A native artifact selects a plan index outside its stored plan list.
    #[error("native artifact selects plan {selected}, but contains {plans} plans")]
    NativeArtifactSelectionOutOfRange {
        /// Zero-based plan index selected by the artifact.
        selected: usize,
        /// Number of plans stored in the artifact.
        plans: usize,
    },

    /// Frontend compilation failed with a diagnostic message.
    #[error("compilation failed: {0}")]
    Compile(String),
}

/// Expected tensor rank in rank-mismatch errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankExpectation {
    /// Requires exactly the contained number of tensor axes.
    Exact(usize),
    /// Requires at least the contained number of tensor axes.
    AtLeast(usize),
}

impl fmt::Display for RankExpectation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(rank) => write!(f, "{rank}"),
            Self::AtLeast(rank) => write!(f, ">= {rank}"),
        }
    }
}

/// SDPA capability rejected by [`Error::SdpaUnsupported`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SdpaFeature {
    /// Auxiliary attention gradients.
    AuxiliaryGradients,
    /// An attention-bias gradient.
    BiasGradient,
    /// Bias gradients, dropout, or attention with linear biases (`ALiBi`).
    BiasDropoutOrAlibi,
    /// Causal masking aligned to the bottom-right corner of the attention matrix.
    CausalBottomRight,
    /// Cumulative sequence-length tensors.
    CumulativeSequenceLengths,
    /// Caller-supplied dropout mask and scale tensors.
    CustomDropout,
    /// Attention dropout.
    Dropout,
    /// Dropout combined with generated softmax statistics.
    DropoutWithGeneratedStats,
    /// Dynamic-shape compilation rather than runtime shape overrides.
    DynamicShape,
    /// Internally generated dropout.
    GeneratedDropout,
    /// A bias tensor with a nonnumeric data type.
    NonNumericBias,
    /// Paged key/value caches.
    PagedCaches,
    /// Ragged attention outputs.
    RaggedOutputs,
    /// Ragged tensor storage.
    RaggedTensors,
    /// Dumping generated random numbers.
    RngDump,
    /// A sequence containing one token.
    SingleTokenSequence,
    /// The selected target architecture.
    TargetArchitecture,
    /// `ALiBi` slopes without causal top-left or right-aligned sliding-window masking.
    UnalignedAlibi,
}

impl fmt::Display for SdpaFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::AuxiliaryGradients => "auxiliary gradients",
            Self::BiasGradient => "bias gradients",
            Self::BiasDropoutOrAlibi => "bias gradients, dropout, or alibi",
            Self::CausalBottomRight => "causal bottom-right masking",
            Self::CumulativeSequenceLengths => "cumulative sequence lengths",
            Self::CustomDropout => "custom dropout mask and scale tensors",
            Self::Dropout => "dropout",
            Self::DropoutWithGeneratedStats => "dropout with generated stats",
            Self::DynamicShape => "dynamic-shape compilation; use runtime shape overrides",
            Self::GeneratedDropout => "generated dropout",
            Self::NonNumericBias => "a bias tensor with a non-numeric data type",
            Self::PagedCaches => "paged key/value caches",
            Self::RaggedOutputs => "ragged outputs",
            Self::RaggedTensors => "ragged tensors",
            Self::RngDump => "rng dump",
            Self::SingleTokenSequence => "sequence length one",
            Self::TargetArchitecture => "this target architecture",
            Self::UnalignedAlibi => {
                "alibi slopes without causal top-left or right-aligned sliding-window masking"
            }
        })
    }
}

/// SDPA prerequisite reported by [`Error::SdpaRequires`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SdpaRequirement {
    /// An explicit attention scale.
    AttentionScale,
    /// Dropout configuration.
    Dropout,
    /// A padding mask, additive mask, or score subgraph when ragged offsets are used.
    MaskOrScoreSubgraphForRaggedOffsets,
    /// A padding mask, causal bottom-right mask, or score subgraph when sequence lengths are supplied.
    MaskOrScoreSubgraphForSequenceLengths,
    /// Paged attention.
    PagedAttention,
    /// Philox-generated dropout.
    PhiloxDropout,
    /// Both attention-probability and sum descriptors.
    ProbabilityAndSumDescriptors,
    /// Ragged tensor storage.
    RaggedTensors,
    /// An attention-score subgraph.
    ScoreSubgraph,
    /// Both query and key/value sequence-length tensors.
    SequenceLengths,
    /// A sink-token tensor.
    SinkToken,
    /// An explicit SM version or a live CUDA context for target resolution.
    TargetArchitecture,
    /// The unified SDPA implementation.
    UnifiedImplementation,
}

impl fmt::Display for SdpaRequirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::AttentionScale => "an attention scale",
            Self::Dropout => "dropout",
            Self::MaskOrScoreSubgraphForRaggedOffsets => {
                "a padding mask, additive mask, or score subgraph for ragged offsets"
            }
            Self::MaskOrScoreSubgraphForSequenceLengths => {
                "a padding mask, causal bottom-right mask, or score subgraph for sequence lengths"
            }
            Self::PagedAttention => "paged attention",
            Self::PhiloxDropout => "Philox dropout",
            Self::ProbabilityAndSumDescriptors => "both probability and sum descriptors",
            Self::RaggedTensors => "ragged tensors",
            Self::ScoreSubgraph => "a score subgraph",
            Self::SequenceLengths => "query and key/value sequence lengths",
            Self::SinkToken => "a sink token tensor",
            Self::TargetArchitecture => "a target sm version or a live context",
            Self::UnifiedImplementation => "the unified implementation",
        })
    }
}

fn on_sm(sm_version: Option<i32>) -> String {
    sm_version
        .map(|sm_version| format!(" on SM {sm_version}"))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_expectation_display() {
        assert_eq!(RankExpectation::Exact(4).to_string(), "4");
        assert_eq!(RankExpectation::AtLeast(2).to_string(), ">= 2");
    }

    #[test]
    fn test_sdpa_unsupported_mentions_architecture_only_when_known() {
        let with_sm = Error::SdpaUnsupported {
            operation: "sdpa backward",
            feature: SdpaFeature::RaggedTensors,
            sm_version: Some(90),
        };
        assert_eq!(
            with_sm.to_string(),
            "sdpa backward does not support ragged tensors on SM 90"
        );
        let without_sm = Error::SdpaUnsupported {
            operation: "mxfp8 sdpa",
            feature: SdpaFeature::PagedCaches,
            sm_version: None,
        };
        assert_eq!(
            without_sm.to_string(),
            "mxfp8 sdpa does not support paged key/value caches"
        );
    }

    #[test]
    fn test_sdpa_requires_message() {
        let error = Error::SdpaRequires {
            operation: "sdpa paged attention",
            requirement: SdpaRequirement::SequenceLengths,
        };
        assert_eq!(
            error.to_string(),
            "sdpa paged attention requires query and key/value sequence lengths"
        );
    }
}

fn plan_report_details(reports: &[PlanBuildReport]) -> String {
    let mut output = String::new();
    for report in reports {
        write!(output, "\n{report}").expect("formatting into a string cannot fail");
    }
    output
}
