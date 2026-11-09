use std::{borrow::Cow, iter};

use singe_cudnn::{
    backend::{
        behavior::{BackendBehaviorNote, BackendNumericalNote},
        engine::{Engine as CudnnEngine, EngineIndex},
        execution::{
            EngineChoice, EngineConfig as CudnnEngineConfig,
            EngineHeuristics as CudnnEngineHeuristics, EngineKnobInfo,
            ExecutionPlan as CudnnExecutionPlan, KnobChoiceInfo,
            OperationGraph as CudnnOperationGraph,
        },
        heuristic::BackendHeuristicMode as CudnnBackendHeuristicMode,
        tensor::{BackendTensorUid as CudnnTensorUid, Tensor as CudnnBackendTensor},
    },
    context::Context as CudnnContext,
    data_type::DataType as CudnnDataType,
};

use crate::{
    backend::cuda::cudnn_backend::data_type::{compute_type, storage_type},
    data_type::DataType,
    error::{Error, Result},
    graph::{ComputePrecision, Determinism, NumericalPolicy},
    layout::Layout,
    shape::Shape,
};

/// Native tensor ID reserved for the first input in cuDNN graph templates.
pub const CUDNN_GRAPH_INPUT0_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(1);

/// Native tensor ID reserved for the second input in cuDNN graph templates.
pub const CUDNN_GRAPH_INPUT1_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(2);

/// Native tensor ID reserved for the output in cuDNN graph templates.
pub const CUDNN_GRAPH_OUTPUT_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(3);

/// Native tensor ID reserved for the bias in cuDNN graph templates.
pub const CUDNN_GRAPH_BIAS_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(4);

/// Native tensor ID reserved for the first virtual intermediate in cuDNN graph templates.
pub const CUDNN_GRAPH_VIRTUAL0_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(5);

pub const CUDNN_GRAPH_VIRTUAL1_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(6);

/// Scalar encoding and physical layout used by a native cuDNN graph template.
/// Floating-point templates use FP32 compute for FP16/BF16/FP32 storage and FP64 compute for FP64 storage.
#[derive(Debug, Clone)]
pub struct CudnnBackendTensorMetadata {
    data_type: DataType,
    dimensions: Vec<i64>,
    strides: Vec<i64>,
    byte_alignment: i64,
}

impl CudnnBackendTensorMetadata {
    /// Copies storage type, dimensions, and element strides for a native tensor descriptor.
    /// Alignment derives from the layout offset and a CUDA allocation base alignment of at least sixteen bytes.
    /// The constructor does not validate that `shape` and `layout` describe the same tensor.
    #[must_use]
    pub fn new(data_type: DataType, shape: &Shape, layout: &Layout) -> Self {
        Self {
            data_type,
            dimensions: shape.dimensions().to_vec(),
            strides: layout.strides().to_vec(),
            byte_alignment: Self::byte_alignment_for_layout(data_type, layout),
        }
    }

    /// Declared storage encoding, independent of inferred compute precision.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    // A tensor laid out as given, for operands whose axes cuDNN names differently.
    #[must_use]
    pub(crate) const fn strided(
        data_type: DataType,
        dimensions: Vec<i64>,
        strides: Vec<i64>,
        byte_alignment: i64,
    ) -> Self {
        Self {
            data_type,
            dimensions,
            strides,
            byte_alignment,
        }
    }

    pub(crate) fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }

    pub(crate) fn strides(&self) -> &[i64] {
        &self.strides
    }

    pub(crate) const fn byte_alignment(&self) -> i64 {
        self.byte_alignment
    }

    pub(crate) fn compute_type(&self) -> Result<CudnnDataType> {
        compute_type(self.data_type)
    }

    pub(crate) fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Adds leading unit axes without changing the addresses of existing elements.
    pub(crate) fn promote_rank(&self, rank: usize) -> Result<Cow<'_, Self>> {
        let leading_axes = rank
            .checked_sub(self.rank())
            .ok_or_else(|| Error::invalid_argument("cudnn tensor rank promotion"))?;
        if leading_axes == 0 {
            return Ok(Cow::Borrowed(self));
        }

        let mut dimensions = vec![1; leading_axes];
        dimensions.extend_from_slice(&self.dimensions);
        let mut strides = vec![self.unit_axis_stride(); leading_axes];
        strides.extend_from_slice(&self.strides);
        Ok(Cow::Owned(Self {
            data_type: self.data_type,
            dimensions,
            strides,
            byte_alignment: self.byte_alignment,
        }))
    }

    /// Packed virtual storage in the operation's accumulation precision.
    pub(crate) fn compute_intermediate(&self) -> Result<Self> {
        let data_type = if self.compute_type()? == CudnnDataType::F64 {
            DataType::F64
        } else {
            DataType::F32
        };
        self.contiguous_with_data_type(data_type)
    }

    pub(crate) fn contiguous_with_data_type(&self, data_type: DataType) -> Result<Self> {
        let shape = Shape::create(self.dimensions.clone())?;
        Ok(Self::new(data_type, &shape, &Layout::from_shape(&shape)))
    }

    // The per-channel bias of a convolution whose output has `rank` axes, broadcast over the others.
    pub(crate) fn convolution_bias(&self, rank: usize) -> Result<Self> {
        let ([channels], [channel_stride]) = (self.dimensions.as_slice(), self.strides.as_slice())
        else {
            return Err(Error::invalid_argument("cudnn convolution bias layout"));
        };
        let span = self.unit_axis_stride();
        Ok(Self {
            data_type: self.data_type,
            dimensions: [1, *channels]
                .into_iter()
                .chain(iter::repeat_n(1, rank.saturating_sub(2)))
                .collect(),
            strides: [span, *channel_stride]
                .into_iter()
                .chain(iter::repeat_n(span, rank.saturating_sub(2)))
                .collect(),
            byte_alignment: self.byte_alignment,
        })
    }

    /// Preserves conventional unit-axis pitches without requiring the full span to fit INT64.
    fn unit_axis_stride(&self) -> i64 {
        self.dimensions
            .iter()
            .zip(&self.strides)
            .try_fold(1_i64, |span, (&dimension, &stride)| {
                span.checked_add(dimension.checked_sub(1)?.checked_mul(stride)?)
            })
            .map_or_else(
                || self.strides.iter().copied().max().unwrap_or(1).max(1),
                |span| span.max(1),
            )
    }

    pub(crate) fn byte_alignment_for_layout(data_type: DataType, layout: &Layout) -> i64 {
        // CUDA allocation bases provide at least sixteen-byte alignment.
        // Reduce the offset and element width first, since only their low four bits matter.
        let byte_offset = (layout.storage_offset() % 16) * (data_type.size_bytes() % 16) % 16;
        if byte_offset == 0 {
            16
        } else {
            1_i64 << byte_offset.trailing_zeros()
        }
    }
}

pub fn cudnn_backend_tensor(
    id: CudnnTensorUid,
    metadata: &CudnnBackendTensorMetadata,
) -> Result<CudnnBackendTensor> {
    if metadata.strides.contains(&0) {
        return Err(Error::unsupported_operation(
            "cudnn zero-stride tensor layout",
        ));
    }
    Ok(CudnnBackendTensor::create_with_id(
        id,
        storage_type(metadata.data_type)?,
        &metadata.dimensions,
        &metadata.strides,
        metadata.byte_alignment,
    )?)
}

pub fn cudnn_backend_virtual_tensor(
    id: CudnnTensorUid,
    metadata: &CudnnBackendTensorMetadata,
) -> Result<CudnnBackendTensor> {
    Ok(CudnnBackendTensor::create_virtual(
        id,
        storage_type(metadata.data_type)?,
        &metadata.dimensions,
        &metadata.strides,
        metadata.byte_alignment,
    )?)
}

/// Selects a cuDNN execution plan under the current precision and determinism policy.
/// Tries instant, B, and fallback heuristics in order.
/// Prefers engines with native CUDA graph support and retains a stream-capture fallback when available.
///
/// # Errors
///
/// Returns an error if no compatible engine can produce a plan, all candidates violate the numerical policy, or cuDNN heuristic, engine metadata, or plan creation calls fail without a usable fallback.
pub fn cudnn_native_graph_execution_plan(
    cudnn: &CudnnContext,
    operation_graph: &CudnnOperationGraph,
    storage_data_type: DataType,
) -> Result<CudnnExecutionPlan> {
    cudnn_execution_plan(cudnn, operation_graph, storage_data_type, false)
}

pub fn cudnn_matmul_graph_execution_plan(
    cudnn: &CudnnContext,
    operation_graph: &CudnnOperationGraph,
    storage_data_type: DataType,
) -> Result<CudnnExecutionPlan> {
    cudnn_execution_plan(cudnn, operation_graph, storage_data_type, true)
}

const HEURISTIC_MODES: [CudnnBackendHeuristicMode; 3] = [
    CudnnBackendHeuristicMode::Instant,
    CudnnBackendHeuristicMode::B,
    CudnnBackendHeuristicMode::Fallback,
];

// Why an engine may not run a graph of this type under the current numerical policy, whether a heuristic or a record offered it.
fn engine_refusal(
    config: &CudnnEngineConfig,
    storage_data_type: DataType,
    matmul: bool,
) -> Result<Option<Error>> {
    let engine = config.engine()?;
    if !accepts_numerical_notes(
        NumericalPolicy::current(),
        storage_data_type,
        engine.numerical_notes()?,
    ) {
        return Ok(Some(Error::unsupported_operation(
            "cuDNN engine violates precision or determinism policy",
        )));
    }

    // Legacy BLAS matmul engine 0 can use TF32 without reporting numerical notes.
    if matmul
        && storage_data_type == DataType::F32
        && engine.index() == EngineIndex::new(0)
        && NumericalPolicy::current().precision == ComputePrecision::Full
    {
        return Ok(Some(Error::unsupported_operation(
            "cuDNN legacy matmul engine does not guarantee FP32 input precision",
        )));
    }
    Ok(None)
}

// What cuDNN's heuristics offer for a graph, from the quickest to choose to the most general.
fn heuristic_configs(
    cudnn: &CudnnContext,
    operation_graph: &CudnnOperationGraph,
) -> Vec<CudnnEngineConfig> {
    let mut configs = Vec::new();
    for mode in HEURISTIC_MODES {
        if let Ok(heuristics) = CudnnEngineHeuristics::create(cudnn, operation_graph, mode, None)
            && let Ok(offered) = heuristics.engine_configs(None)
        {
            configs.extend(offered);
        }
    }
    configs
}

// The plan of an engine and knobs a record states, if cuDNN takes them for this graph and they keep the numerical policy.
// An engine with no knobs stated runs as the heuristics configure it, since an engine such as attention's has no default of its own to build with.
//
// # Errors
//
// Returns an error if cuDNN refuses them or the policy does.
pub fn cudnn_forced_execution_plan(
    cudnn: &CudnnContext,
    operation_graph: &CudnnOperationGraph,
    storage_data_type: DataType,
    matmul: bool,
    choice: &EngineChoice,
) -> Result<CudnnExecutionPlan> {
    let offered = if choice.knob_choices.is_empty() {
        heuristic_configs(cudnn, operation_graph)
            .into_iter()
            .find(|config| {
                config
                    .engine()
                    .is_ok_and(|engine| engine.index() == choice.engine_index)
            })
    } else {
        None
    };
    let config = match offered {
        Some(config) => config,
        None => CudnnEngineConfig::create(cudnn, operation_graph, choice)?,
    };
    if let Some(refusal) = engine_refusal(&config, storage_data_type, matmul)? {
        return Err(refusal);
    }
    Ok(CudnnExecutionPlan::create(cudnn, &config)?)
}

// How many of one graph's engine and knob choices tuning times, a bound the budget also limits.
const LISTED_ENGINE_CHOICES: usize = 48;

// How many of those one engine may take, so another engine is reached.
const LISTED_CHOICES_OF_ENGINE: usize = 24;

// How many values of one knob are listed, so another knob is reached.
const LISTED_VALUES_OF_KNOB: usize = 16;

// What cuDNN could run a graph with: each engine that may run it, and each of those with one knob set at a time, since their product is too large to walk.
//
// # Errors
//
// Returns an error if cuDNN cannot list its engines.
pub fn cudnn_engine_choices(
    cudnn: &CudnnContext,
    operation_graph: &CudnnOperationGraph,
    storage_data_type: DataType,
    matmul: bool,
) -> Result<Vec<EngineChoice>> {
    let mut engines = Vec::<(EngineIndex, Vec<EngineKnobInfo>)>::new();
    let offered = heuristic_configs(cudnn, operation_graph);
    let built = CudnnEngine::engine_indices_for_operation_graph(operation_graph)?
        .into_iter()
        .filter_map(|index| {
            let base = EngineChoice::new(index, Vec::new());
            CudnnEngineConfig::create(cudnn, operation_graph, &base).ok()
        });
    for config in offered.into_iter().chain(built) {
        let index = config.engine()?.index();
        if engines.iter().all(|(engine, _)| *engine != index)
            && engine_refusal(&config, storage_data_type, matmul)?.is_none()
        {
            engines.push((index, config.knob_infos()?.to_vec()));
        }
    }

    let mut choices = engines
        .iter()
        .map(|(index, _)| EngineChoice::new(*index, Vec::new()))
        .collect::<Vec<_>>();
    for (index, knobs) in engines {
        let mut of_engine = 1;
        for knob in knobs {
            if knob.stride <= 0 {
                continue;
            }

            let mut of_knob = 0;
            let mut value = knob.minimum_value;
            while value <= knob.maximum_value
                && of_knob < LISTED_VALUES_OF_KNOB
                && of_engine < LISTED_CHOICES_OF_ENGINE
            {
                let choice =
                    EngineChoice::new(index, vec![KnobChoiceInfo::new(knob.knob_type, value)]);
                // cuDNN refuses many values, and only trying one says which.
                if let Ok(config) = CudnnEngineConfig::create(cudnn, operation_graph, &choice)
                    && engine_refusal(&config, storage_data_type, matmul)?.is_none()
                {
                    choices.push(choice);
                    of_knob += 1;
                    of_engine += 1;
                }

                value += knob.stride;
            }
        }
    }

    choices.truncate(LISTED_ENGINE_CHOICES);
    Ok(choices)
}

fn cudnn_execution_plan(
    cudnn: &CudnnContext,
    operation_graph: &CudnnOperationGraph,
    storage_data_type: DataType,
    matmul: bool,
) -> Result<CudnnExecutionPlan> {
    let mut last_error = None;
    // Engines without the native CUDA graph API are recorded through stream capture instead.
    let mut captured_fallback = None;

    for mode in HEURISTIC_MODES {
        let heuristics = match CudnnEngineHeuristics::create(cudnn, operation_graph, mode, None) {
            Ok(heuristics) => heuristics,
            Err(error) => {
                last_error = Some(Error::from(error));
                continue;
            }
        };
        let configs = match heuristics.engine_configs(None) {
            Ok(configs) => configs,
            Err(error) => {
                last_error = Some(error.into());
                continue;
            }
        };
        for config in configs {
            let notes = match engine_refusal(&config, storage_data_type, matmul) {
                Ok(None) => config
                    .engine()
                    .and_then(|engine| Ok(engine.behavior_notes()?.to_vec())),
                Ok(Some(refusal)) => {
                    last_error = Some(refusal);
                    continue;
                }
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            let notes = match notes {
                Ok(notes) => notes,
                Err(error) => {
                    last_error = Some(error.into());
                    continue;
                }
            };

            let native = notes.contains(&BackendBehaviorNote::SupportsCudaGraphNativeApi);
            if !native && captured_fallback.is_some() {
                continue;
            }
            match CudnnExecutionPlan::create(cudnn, &config) {
                Ok(plan) if native => return Ok(plan),
                Ok(plan) => captured_fallback = Some(plan),
                Err(error) => last_error = Some(Error::from(error)),
            }
        }
    }

    captured_fallback.ok_or_else(|| {
        last_error.unwrap_or_else(|| {
            Error::library_refused(
                "cuDNN",
                Error::unsupported_operation("cuda graph cuDNN native graph execution plan"),
            )
        })
    })
}

fn accepts_numerical_notes(
    policy: NumericalPolicy,
    storage_data_type: DataType,
    notes: &[BackendNumericalNote],
) -> bool {
    !notes.iter().any(|note| {
        matches!(
            note,
            BackendNumericalNote::DownConvertInputs
                | BackendNumericalNote::ReducedPrecisionReduction
        ) || (storage_data_type == DataType::F32
            && *note == BackendNumericalNote::TensorCore
            && policy.precision == ComputePrecision::Full)
            || (*note == BackendNumericalNote::Nondeterministic
                && policy.determinism == Determinism::Require)
    })
}

#[cfg(test)]
mod tests;
