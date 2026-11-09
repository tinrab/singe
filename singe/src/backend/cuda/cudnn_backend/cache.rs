use std::sync::{Arc, Mutex};

use singe_core::ErrorChain;
use singe_cudnn::{context::Context as CudnnContext, error::Status as CudnnStatus};

use crate::{
    backend::cuda::{
        cudnn_backend::{
            CudnnBackendTensorMetadata,
            types::{
                CudnnBackendGraphCacheEntry, CudnnBackendGraphKey, CudnnBackendGraphTemplate,
                CudnnBackendTensorKey,
            },
        },
        runtime::CudaRuntime,
    },
    data_type::DataType,
    error::{Error, Result, UnsupportedTarget},
    layout::Layout,
    shape::Shape,
    utility::Locked,
};

/// Returns a cached cuDNN graph template or builds one under the runtime's cuDNN handle lock.
/// Stable capability rejections are cached; resource and transient failures are not.
/// A concurrent cache insertion takes precedence over a newly built result.
///
/// # Errors
///
/// Returns an error if a cache or cuDNN handle mutex is poisoned, a cached entry records an unsupported graph, or `build` fails.
pub fn cached_cudnn_backend_graph_template(
    runtime: &Arc<CudaRuntime>,
    key: CudnnBackendGraphKey,
    build: impl FnOnce(&CudnnContext) -> Result<CudnnBackendGraphTemplate>,
) -> Result<Arc<Mutex<CudnnBackendGraphTemplate>>> {
    {
        let cache = runtime
            .cudnn_backend_graphs()
            .locked("cuda runtime cache")?;
        if let Some(entry) = cache.get(&key) {
            return cudnn_backend_graph_cache_entry(entry);
        }
    }

    let build_result = {
        let cudnn = runtime.cudnn().locked("cuda runtime cache")?;
        build(&cudnn)
    };

    let mut cache = runtime
        .cudnn_backend_graphs()
        .locked("cuda runtime cache")?;
    if let Some(entry) = cache.get(&key) {
        return cudnn_backend_graph_cache_entry(entry);
    }

    let result = match build_result {
        Ok(template) => {
            let workspace_size = template.workspace_size();
            let template = Arc::new(Mutex::new(template));
            cache.insert(
                key,
                CudnnBackendGraphCacheEntry::Supported {
                    template: Arc::clone(&template),
                    workspace_size,
                },
            );
            Ok(template)
        }
        Err(error) => {
            if stable_rejection(&error) {
                cache.insert(
                    key,
                    CudnnBackendGraphCacheEntry::Unsupported {
                        reason: ErrorChain(&error).to_string(),
                    },
                );
            }
            Err(error)
        }
    };
    drop(cache);
    result
}

fn cudnn_backend_graph_cache_entry(
    entry: &CudnnBackendGraphCacheEntry,
) -> Result<Arc<Mutex<CudnnBackendGraphTemplate>>> {
    match entry {
        CudnnBackendGraphCacheEntry::Supported { template, .. } => Ok(Arc::clone(template)),
        CudnnBackendGraphCacheEntry::Unsupported { reason } => Err(Error::library_refused(
            "cuDNN",
            Error::unsupported_operation_with_details(
                "cuda graph cuDNN backend plan",
                reason.clone(),
            ),
        )),
    }
}

/// Returns whether an error is a stable cuDNN graph capability rejection suitable for negative caching.
/// Nested graph-source errors are classified through their underlying error.
/// Resource, driver, and internal failures return false.
pub fn stable_rejection(error: &Error) -> bool {
    match error {
        Error::Unsupported {
            target: UnsupportedTarget::Operation,
            ..
        }
        | Error::LibraryRefused { .. } => true,
        Error::GraphSource { source, .. } => stable_rejection(source),
        Error::Cudnn(error) => matches!(
            error.status(),
            Some(
                CudnnStatus::NotSupported
                    | CudnnStatus::NotSupportedGraphPattern
                    | CudnnStatus::NotSupportedShape
                    | CudnnStatus::NotSupportedDataType
                    | CudnnStatus::NotSupportedLayout
                    | CudnnStatus::NotSupportedArchitectureMismatch
                    | CudnnStatus::NotSupportedSharedMemoryInsufficient
                    | CudnnStatus::NotSupportedPadding
                    | CudnnStatus::NotSupportedCudaGraphNativeApi
            )
        ),
        _ => false,
    }
}

/// Copies storage data type, dimensions, element strides, and offset-derived alignment into a cuDNN tensor cache key.
/// The constructor records metadata without validating shape and layout compatibility.
#[must_use]
pub fn cudnn_backend_tensor_metadata_key(
    data_type: DataType,
    shape: &Shape,
    layout: &Layout,
) -> CudnnBackendTensorKey {
    CudnnBackendTensorKey::new(
        data_type,
        shape.dimensions().to_vec(),
        layout.strides().to_vec(),
        CudnnBackendTensorMetadata::byte_alignment_for_layout(data_type, layout),
    )
}

/// Returns the retained template's workspace requirement in bytes.
///
/// # Errors
///
/// Returns an error if a lock is poisoned.
pub fn cudnn_backend_template_workspace_size(
    template: &Arc<Mutex<CudnnBackendGraphTemplate>>,
) -> Result<usize> {
    Ok(template.locked("cuda runtime cache")?.workspace_size())
}
