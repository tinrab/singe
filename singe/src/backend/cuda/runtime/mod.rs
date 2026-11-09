mod autotune;
mod cache;
mod cublaslt;
mod policy;
mod retention;
mod stats;
mod weight;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use singe_cufft::{plan::Plan, types::TransformType};

#[cfg(test)]
pub use self::autotune::AutotuneTensorBucket;
use singe_cublas::{context::Context as CublasContext, lt::context::Context as CublasLtContext};
use singe_cuda::{
    architecture::GpuArchitecture, context::Context as CudaContext, module::KernelFunction,
};
use singe_cudnn::context::Context as CudnnContext;
use singe_cutensor::context::Context as CutensorContext;

pub use self::{
    autotune::{AutotuneCacheEntry, AutotuneCacheKey},
    cublaslt::{
        CublasLtMatmulAlgorithmEntry, CublasLtMatmulAlgorithmFingerprint,
        CublasLtMatmulAlgorithmKey, CublasLtMatmulKeyParts,
    },
    policy::{CudaRuntimeCacheLimits, CudaRuntimeCacheStats},
    retention::{ProviderCache, ProviderCacheLimits, ProviderCacheStats},
    stats::{AutotuneCacheStats, CublasLtMatmulAlgorithmCacheStats},
};
use crate::{
    backend::cuda::{
        cudnn_backend::{CudnnBackendGraphCacheEntry, CudnnBackendGraphKey},
        cutensor::{CachedCutensorContractionPlan, CutensorContractionPlanKey},
    },
    device::Device,
    error::{Error, Result},
    storage::cuda::{cuda_context_for_device, cuda_device_id},
    utility::Locked,
};

/// Shared CUDA context, library handles, and bounded provider caches for one device.
/// Library handles and caches have separate mutexes.
#[derive(Debug)]
pub struct CudaRuntime {
    context: Arc<CudaContext>,
    /// The real architecture generated kernels compile for, such as `sm_120`.
    architecture: GpuArchitecture,
    cublas: Mutex<CublasContext>,
    cublaslt: Mutex<CublasLtContext>,
    cudnn: Mutex<CudnnContext>,
    cutensor: Mutex<CutensorContext>,
    cudnn_backend_graphs: Mutex<ProviderCache<CudnnBackendGraphKey, CudnnBackendGraphCacheEntry>>,
    cutensor_contraction_plans:
        Mutex<ProviderCache<CutensorContractionPlanKey, CachedCutensorContractionPlan>>,
    cublaslt_matmul_algorithms:
        Mutex<ProviderCache<CublasLtMatmulAlgorithmKey, CublasLtMatmulAlgorithmEntry>>,
    unsupported_cublaslt_matmul_algorithms: Mutex<ProviderCache<CublasLtMatmulAlgorithmKey, ()>>,
    autotune_cache: Mutex<ProviderCache<AutotuneCacheKey, AutotuneCacheEntry>>,
    fft_plans: Mutex<ProviderCache<FftPlanKey, Plan>>,
    generated_kernels: Mutex<ProviderCache<String, KernelFunction>>,
}

// The transform, signal length, and batch a cuFFT plan runs.
pub type FftPlanKey = (TransformType, usize, usize);

impl CudaRuntime {
    fn create(device_id: i32) -> Result<Self> {
        let context = cuda_context_for_device(device_id)?;
        context.bind()?;
        let cublas = CublasContext::create(&context)?;
        let cublaslt = CublasLtContext::create(&context)?;
        let cudnn = CudnnContext::create(&context)?;
        let cutensor = CutensorContext::create(&context)?;
        let properties = context.device().properties()?;
        let architecture = format!("sm_{}{}", properties.major, properties.minor).parse()?;
        Ok(Self {
            context,
            architecture,
            cublas: Mutex::new(cublas),
            cublaslt: Mutex::new(cublaslt),
            cudnn: Mutex::new(cudnn),
            cutensor: Mutex::new(cutensor),
            cudnn_backend_graphs: Mutex::new(ProviderCache::new(
                ProviderCacheLimits::default(),
                weight::cudnn,
            )),
            cutensor_contraction_plans: Mutex::new(ProviderCache::new(
                ProviderCacheLimits::default(),
                weight::cutensor,
            )),
            cublaslt_matmul_algorithms: Mutex::new(ProviderCache::new(
                ProviderCacheLimits::default(),
                weight::cublaslt,
            )),
            unsupported_cublaslt_matmul_algorithms: Mutex::new(ProviderCache::new(
                ProviderCacheLimits::default(),
                weight::unsupported_cublaslt,
            )),
            autotune_cache: Mutex::new(ProviderCache::new(
                ProviderCacheLimits::default(),
                weight::autotune,
            )),
            fft_plans: Mutex::new(ProviderCache::new(
                ProviderCacheLimits::default(),
                weight::fft,
            )),
            generated_kernels: Mutex::new(ProviderCache::new(
                CudaRuntimeCacheLimits::default().generated_kernels,
                weight::compiled_kernel,
            )),
        })
    }

    // Returns the bounded cache of cuFFT plans eager transforms reuse, since planning takes far longer than a transform.
    pub(crate) fn fft_plans(&self) -> &Mutex<ProviderCache<FftPlanKey, Plan>> {
        &self.fft_plans
    }

    /// Returns the bounded cache of compiled generated kernels, keyed by their compile options and code.
    pub(crate) fn generated_kernels(&self) -> &Mutex<ProviderCache<String, KernelFunction>> {
        &self.generated_kernels
    }

    /// Returns the real architecture generated kernels compile for, such as `sm_120`.
    pub(crate) const fn architecture(&self) -> GpuArchitecture {
        self.architecture
    }

    /// Returns the actual CUDA context retained by this runtime.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Returns the mutex-protected cuBLAS handle associated with this runtime's CUDA context.
    pub fn cublas(&self) -> &Mutex<CublasContext> {
        &self.cublas
    }

    /// Returns the mutex-protected cuBLASLt handle associated with this runtime's CUDA context.
    pub fn cublaslt(&self) -> &Mutex<CublasLtContext> {
        &self.cublaslt
    }

    /// Returns the mutex-protected cuDNN handle associated with this runtime's CUDA context.
    pub fn cudnn(&self) -> &Mutex<CudnnContext> {
        &self.cudnn
    }

    /// Returns the mutex-protected cuTENSOR handle associated with this runtime's CUDA context.
    pub fn cutensor(&self) -> &Mutex<CutensorContext> {
        &self.cutensor
    }
}

/// Returns the process-wide shared runtime for a CUDA device, creating it on first use.
/// The registry retains the runtime and its caches for the process lifetime.
///
/// # Errors
///
/// Returns an error if `device` is not CUDA, its ordinal cannot be represented, the runtime registry mutex is poisoned, or CUDA context binding or library handle creation fails.
pub fn runtime_for_device(device: Device) -> Result<Arc<CudaRuntime>> {
    static RUNTIMES: OnceLock<Mutex<HashMap<i32, Arc<CudaRuntime>>>> = OnceLock::new();

    if !device.is_cuda() {
        return Err(Error::device_mismatch(
            "cuda runtime device",
            Device::Cuda(0),
            device,
        ));
    }

    let device_id = cuda_device_id(device)?;
    let runtimes = RUNTIMES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut runtimes = runtimes.locked("cuda runtime cache")?;
    if let Some(runtime) = runtimes.get(&device_id) {
        return Ok(Arc::clone(runtime));
    }

    let runtime = Arc::new(CudaRuntime::create(device_id)?);
    runtimes.insert(device_id, Arc::clone(&runtime));
    drop(runtimes);
    Ok(runtime)
}

#[cfg(all(test, feature = "testing"))]
mod tests;
