mod bindings;
mod cache;
mod data_type;
mod eager;
mod executable;
mod templates;
mod types;

#[cfg(test)]
pub use self::executable::{CAPTURED_GRAPHS, FORCE_CAPTURED_GRAPHS};
pub use self::{
    bindings::cudnn_backend_tensor_binding,
    cache::{
        cached_cudnn_backend_graph_template, cudnn_backend_template_workspace_size,
        cudnn_backend_tensor_metadata_key, stable_rejection as stable_cudnn_rejection,
    },
    eager::execute_cudnn_template,
    executable::{CudnnBackendExecutable, CudnnGraphBinding, CudnnGraphBindingSource},
    templates::{
        CUDNN_ATTENTION_KEY_LENGTHS_TENSOR_ID, CUDNN_ATTENTION_KEY_PAGES_TENSOR_ID,
        CUDNN_ATTENTION_KEY_TENSOR_ID, CUDNN_ATTENTION_OFFSETS_TENSOR_ID,
        CUDNN_ATTENTION_OUTPUT_TENSOR_ID, CUDNN_ATTENTION_QUERY_LENGTHS_TENSOR_ID,
        CUDNN_ATTENTION_QUERY_TENSOR_ID, CUDNN_ATTENTION_STATS_TENSOR_ID,
        CUDNN_ATTENTION_VALUE_PAGES_TENSOR_ID, CUDNN_ATTENTION_VALUE_TENSOR_ID,
        CUDNN_GRAPH_BIAS_TENSOR_ID, CUDNN_GRAPH_INPUT0_TENSOR_ID, CUDNN_GRAPH_INPUT1_TENSOR_ID,
        CUDNN_GRAPH_OUTPUT_TENSOR_ID, CudnnAttentionProblem, CudnnBackendTensorMetadata,
        CudnnPaging, build_cudnn_attention_template, build_cudnn_conv2d_bias_relu_template,
        build_cudnn_convolution_template, build_cudnn_matmul_add_relu_template,
        build_cudnn_matmul_add_template, build_cudnn_matmul_template, cudnn_attention_scalars,
        cudnn_attention_scale, cudnn_convolution_geometry, cudnn_engine_choices,
        cudnn_reads_key_lengths, cudnn_takes_attention, cudnn_takes_matmul,
    },
    types::{
        CudnnBackendGraphCacheEntry, CudnnBackendGraphKey, CudnnBackendGraphKind,
        CudnnBackendGraphTemplate, CudnnBackendTensorKey, cudnn_engine_key_parameters,
    },
};

#[cfg(all(test, feature = "testing"))]
mod tests;
