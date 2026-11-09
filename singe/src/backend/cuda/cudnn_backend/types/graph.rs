use std::sync::{Arc, Mutex};

use singe_cudnn::backend::{
    execution::{
        EngineChoice, ExecutionPlan as CudnnExecutionPlan, OperationGraph as CudnnOperationGraph,
    },
    pointwise::PointwiseMode as CudnnPointwiseMode,
    reduction::ReduceTensorOperator as CudnnReductionMode,
    tensor::{BackendTensorUid as CudnnTensorUid, Tensor as CudnnBackendTensor},
};

use crate::{
    backend::{cuda::cudnn_backend::types::CudnnBackendOperation, registry::Operation},
    data_type::DataType,
    graph::NumericalPolicy,
};

/// Operation or fused-operation pattern represented by a cached cuDNN backend graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudnnBackendGraphKind {
    /// Two-dimensional convolution.
    Conv2d,
    // Three-dimensional convolution.
    Conv3d,
    /// Convolution followed by bias addition and rectified linear activation.
    Conv2dBiasRelu,
    // Attention over the graph's own keys and values.
    Attention,
    // Attention over a paged cache.
    PagedAttention,
    /// Matrix multiplication.
    Matmul,
    /// Matrix multiplication followed by addition.
    MatmulAdd,
    /// Matrix multiplication followed by addition and rectified linear activation.
    MatmulAddRelu,
    /// Pointwise computation selected by the native mode.
    Pointwise(CudnnPointwiseMode),
    /// Sigmoid-weighted gated linear computation.
    SwiGlu,
    /// Scalar-tensor computation identified by its registry operation.
    ScalarPointwise(Operation),
    /// Normalization identified by its registry operation.
    Normalization(Operation),
    /// Tensor reduction selected by its native reduction mode.
    Reduction(CudnnReductionMode),
}

/// Numerical policy, graph pattern, operation parameters, and tensor layouts identifying a cuDNN graph template.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudnnBackendGraphKey {
    numerics: NumericalPolicy,
    kind: CudnnBackendGraphKind,
    parameters: Vec<i64>,
    input_shapes: Vec<CudnnBackendTensorKey>,
    output_shape: CudnnBackendTensorKey,
}

/// Element type, dimensions, element strides, and byte alignment used in cuDNN graph cache keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudnnBackendTensorKey {
    data_type: DataType,
    dimensions: Vec<i64>,
    strides: Vec<i64>,
    byte_alignment: i64,
}

/// A retained supported graph template or a memoized unsupported-operation reason.
#[derive(Debug, Clone)]
pub enum CudnnBackendGraphCacheEntry {
    /// A shared graph template with its recorded workspace requirement.
    Supported {
        /// Shared native graph template protected by a mutex.
        template: Arc<Mutex<CudnnBackendGraphTemplate>>,
        /// Reported required workspace in bytes, rather than an allocated buffer.
        workspace_size: usize,
    },
    /// A cached failure to prepare a supported graph template.
    Unsupported {
        /// Explanation of why preparation was unsupported.
        reason: String,
    },
}

/// Retains a cuDNN execution plan, binding tensors, source operation graph, and workspace requirement.
#[derive(Debug)]
pub struct CudnnBackendGraphTemplate {
    plan: CudnnExecutionPlan,
    workspace_size: usize,
    operation: CudnnBackendOperation,
    graph: CudnnOperationGraph,
}

// What a forced engine adds to a graph's key, so one forced choice is never served another's plan.
#[must_use]
pub fn cudnn_engine_key_parameters(engine: Option<&EngineChoice>) -> Vec<i64> {
    engine.map_or_else(Vec::new, |choice| {
        let mut parameters = vec![choice.engine_index.as_i64() + 1];
        for knob in &choice.knob_choices {
            parameters.push(i64::from(knob.knob_type.raw()));
            parameters.push(knob.value);
        }
        parameters
    })
}

impl CudnnBackendGraphKey {
    pub(crate) fn accounted_bytes(&self) -> usize {
        let base = size_of::<Self>().saturating_add(
            self.input_shapes
                .capacity()
                .saturating_mul(size_of::<CudnnBackendTensorKey>()),
        );
        self.input_shapes
            .iter()
            .chain([&self.output_shape])
            .flat_map(|key| [key.dimensions.capacity(), key.strides.capacity()])
            .chain([self.parameters.capacity()])
            .fold(base, |bytes, capacity| {
                bytes.saturating_add(capacity.saturating_mul(size_of::<i64>()))
            })
    }

    /// Stores graph pattern, operation-specific parameters, and tensor keys using the current numerical policy.
    ///
    /// This does not validate tensor metadata or build a native graph.
    #[must_use]
    pub fn new(
        kind: CudnnBackendGraphKind,
        parameters: Vec<i64>,
        input_shapes: Vec<CudnnBackendTensorKey>,
        output_shape: CudnnBackendTensorKey,
    ) -> Self {
        Self {
            kind,
            numerics: NumericalPolicy::current(),
            parameters,
            input_shapes,
            output_shape,
        }
    }
}

impl CudnnBackendTensorKey {
    /// Stores element type, dimensions, element strides, and byte alignment without validation.
    #[must_use]
    pub fn new(
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
}

impl CudnnBackendGraphTemplate {
    /// Retains a native execution plan, operation bindings, source graph, and supplied workspace byte requirement.
    ///
    /// This does not allocate workspace or cross-check the supplied requirement.
    #[must_use]
    pub fn new(
        plan: CudnnExecutionPlan,
        workspace_size: usize,
        operation: CudnnBackendOperation,
        graph: CudnnOperationGraph,
    ) -> Self {
        Self {
            plan,
            workspace_size,
            operation,
            graph,
        }
    }

    // The native graph this plan runs, which also lists what else could run it.
    pub const fn operation_graph(&self) -> &CudnnOperationGraph {
        &self.graph
    }

    /// Returns the retained native cuDNN execution plan.
    #[must_use]
    pub fn plan(&self) -> &CudnnExecutionPlan {
        &self.plan
    }

    /// Returns the retained binding tensor with native UID `id`, or `None` if absent.
    #[must_use]
    pub fn binding_tensor(&self, id: CudnnTensorUid) -> Option<&CudnnBackendTensor> {
        self.operation.binding_tensor(id)
    }

    /// Returns the stored workspace byte requirement.
    #[must_use]
    pub fn workspace_size(&self) -> usize {
        self.workspace_size
    }
}
