use singe_core::ErrorChain;
use singe_cublas::error::{Error as CublasError, Status as CublasStatus};
use singe_cuda::error::Error as CudaError;
use singe_cutensor::error::{Error as CutensorError, Status as CutensorStatus};

use crate::{
    backend::cuda::{
        compiled::plan::lowering::CudaGraphProvider, cudnn_backend::stable_cudnn_rejection,
    },
    error::{Error, Result},
};

/// Latest capability rejection message for each attempted CUDA lowering provider.
/// Entries retain the order in which providers first failed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphProviderFailures {
    failures: Vec<(CudaGraphProvider, String)>,
}

impl GraphProviderFailures {
    /// Stores the error message for a provider, replacing its previous message when already present.
    pub fn record(&mut self, provider: CudaGraphProvider, error: &Error) {
        let message = ErrorChain(error).to_string();
        if let Some((_, details)) = self
            .failures
            .iter_mut()
            .find(|(candidate, _)| *candidate == provider)
        {
            *details = message;
        } else {
            self.failures.push((provider, message));
        }
    }

    /// Returns whether no provider failures have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.failures.is_empty()
    }

    /// Returns providers and rejection messages in first-recorded order.
    #[must_use]
    pub fn as_slice(&self) -> &[(CudaGraphProvider, String)] {
        &self.failures
    }
}

// Records a rejection by Singe's own checks of what a provider can run, so the next provider can be tried.
// A rejection the library itself reports is an error, since planning decided that library runs the step and nothing else then does.
//
// # Errors
//
// Returns `Error::LibraryRefused` for a library's own rejection, and the original error unchanged when it is not a rejection at all, such as a resource, driver, or internal failure.
pub fn record_cuda_graph_provider_failure(
    failures: &mut GraphProviderFailures,
    provider: CudaGraphProvider,
    error: Error,
) -> Result<()> {
    match rejection(&error) {
        Some(Rejection::Checked) => {
            failures.record(provider, &error);
            Ok(())
        }
        Some(Rejection::Library) => Err(Error::LibraryRefused {
            library: format!("{provider:?}"),
            source: Box::new(error),
        }),
        None => Err(error),
    }
}

// Who rejected what a provider was asked to run.
enum Rejection {
    // Singe's own checks, which ask no library.
    Checked,
    // The library, which planning decided runs it.
    Library,
}

// Driver, allocation, compilation, and internal errors are no rejection, and keep their original type.
fn rejection(error: &Error) -> Option<Rejection> {
    match error {
        Error::GraphSource { source, .. } => rejection(source),
        Error::Cudnn(_) => stable_cudnn_rejection(error).then_some(Rejection::Library),
        Error::Unsupported { .. }
        | Error::ContiguousRequired
        | Error::Cuda(
            CudaError::UnsupportedArchitecture { .. } | CudaError::NoCompatibleArtifact { .. },
        ) => Some(Rejection::Checked),
        Error::Cublas(CublasError::Cublas {
            code: CublasStatus::NotSupported,
            ..
        })
        | Error::Cutensor(CutensorError::Cutensor {
            code: CutensorStatus::NotSupported,
            ..
        }) => Some(Rejection::Library),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
