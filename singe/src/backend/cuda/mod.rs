//! The CUDA backend: eager kernels the dispatcher calls, and compilation of captured graphs into CUDA graphs.
//!
//! Eager calls choose a kernel per operation (`provider_selection`): a generated kernel (`kernel_plan`), or an NVIDIA library such as cuBLAS, cuDNN, or cuTENSOR.
//! Graph compilation plans a provider per node, plans memory, and lowers the graph into one native CUDA graph (`compiled::{plan, build, executable}`).

/// CUDA storage allocation strategies and allocator statistics.
pub(crate) mod allocator;

mod codegen;

/// CUDA execution context helpers.
pub(crate) mod context;

/// Compilation of captured graphs into native CUDA graphs.
pub(crate) mod compiled;

/// cuDNN graph templates, execution plans, and allocation bindings.
pub(crate) mod cudnn_backend;

mod cutensor;

/// CUDA graph lowering reports and provider failure classification.
pub(crate) mod diagnostics;

mod device_api;
mod device_views;
pub(crate) mod eager;
mod event;
mod gemm;
mod graph_binding;
pub(crate) mod graph_value;
mod kernel_plan;
pub mod kernel_tuning;
pub(crate) mod kernels;
mod nvtx;
mod provider_selection;

/// Shared CUDA library handles and bounded provider caches.
pub(crate) mod runtime;

pub mod tuning;

// Whole graphs as CUDA C++ programs.
pub mod export;
mod stream;
pub(crate) mod tensor_region;
#[cfg(all(test, feature = "testing"))]
mod test_support;
mod transfer;
mod utility;

#[cfg(feature = "onnx")]
pub use self::compiled::build::compile_onnx_model;
pub use self::{
    codegen::{
        BufferDescription, KernelAbi, KernelArgument, KernelArtifact, KernelCode,
        KernelDescription, KernelLanguage, KernelLaunch, KernelOutput, KernelRecorder,
        PlanningContext, disable_kernel_cache, enable_kernel_cache,
    },
    compiled::{
        build::compile_graph,
        executable::{CudaCompiledGraph, CudaCompiledGraphStats, GraphReplayStats},
        plan::{
            cache_key::{DynamicShapeMode, DynamicShapePolicy},
            cublas::{CublasLtMatmulTuningOptions, CublasLtMatmulTuningResult},
            lowering::{
                CublasFeature, CublasLtFeature, CudaGraphProvider, CudnnFeature, CufftFeature,
                CutensorFeature, LibraryFeature, LoweringPolicy, ModuleKernel,
            },
        },
    },
    device_api::{
        allocator_stats, autotune_cache_stats, clear_cache, cublaslt_matmul_algorithm_cache_stats,
        current_stream, device, device_count, eager_execution_family, execution_context,
        is_available, operation_execution_family, provider_cache_stats, reset_peak_memory,
        set_current_stream, set_memory_limit, supports_operation, supports_operation_input,
        supports_operation_output, synchronize, tune_cublaslt_matmul,
    },
    diagnostics::GraphFailureCategory,
    event::{CudaEvent, EventTiming},
    runtime::{
        AutotuneCacheStats, CublasLtMatmulAlgorithmCacheStats, CudaRuntimeCacheStats,
        ProviderCacheStats,
    },
    stream::{CudaStream, StreamHold},
    transfer::CudaTransfer,
};
pub(crate) use self::{compiled::build::compile, eager::storage::pinned_tensor};
use crate::backend::cuda::{
    graph_binding::{validate_graph_matmul, validate_graph_matmul_add},
    runtime::CudaRuntime,
};
