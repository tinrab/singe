mod autotune;
mod cache;
mod cublaslt;
mod policy;
mod retention;
mod stats;
mod weight;

use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};

use singe_cufft::{plan::Plan, types::TransformType};

use singe_cublas::{context::Context as CublasContext, lt::context::Context as CublasLtContext};
use singe_cuda::{
    architecture::GpuArchitecture,
    context::Context as CudaContext,
    module::KernelFunction,
    stream::{
        Stream, StreamCaptureMode, StreamCaptureStatus, StreamOptions, exchange_capture_mode,
    },
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
        device_count, kernel_plan,
        nvtx::CudaNvtxRange,
        stream::CudaStream,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    shape::Shape,
    storage::cuda::{cuda_context_for_device, cuda_device_id},
    tensor::Tensor,
    utility::Locked,
};

/// Shared CUDA context, library handles, and bounded provider caches for one device.
/// Library handles and caches have separate mutexes.
#[derive(Debug)]
pub struct CudaRuntime {
    context: Arc<CudaContext>,
    /// The real architecture generated kernels compile for, such as `sm_120`.
    architecture: GpuArchitecture,
    // Each library's handle, created on first use: creating them all costs about half a second, mostly cuTENSOR's, which most models never call.
    cublas: OnceLock<Mutex<CublasContext>>,
    cublaslt: OnceLock<Mutex<CublasLtContext>>,
    cudnn: OnceLock<Mutex<CudnnContext>>,
    cutensor: OnceLock<Mutex<CutensorContext>>,
    cudnn_backend_graphs: Mutex<ProviderCache<CudnnBackendGraphKey, CudnnBackendGraphCacheEntry>>,
    cutensor_contraction_plans:
        Mutex<ProviderCache<CutensorContractionPlanKey, CachedCutensorContractionPlan>>,
    cublaslt_matmul_algorithms:
        Mutex<ProviderCache<CublasLtMatmulAlgorithmKey, CublasLtMatmulAlgorithmEntry>>,
    unsupported_cublaslt_matmul_algorithms: Mutex<ProviderCache<CublasLtMatmulAlgorithmKey, ()>>,
    autotune_cache: Mutex<ProviderCache<AutotuneCacheKey, AutotuneCacheEntry>>,
    fft_plans: Mutex<ProviderCache<FftPlanKey, Plan>>,
    generated_kernels: Mutex<ProviderCache<String, KernelFunction>>,
    // The stream eager kernels launch on when their context names none, created once rather than for each launch.
    eager_stream: OnceLock<CudaStream>,
    // The status word eager checking kernels share, and whether one ran since it was last read.
    // Eager work reads it at the next readback or synchronization, as compiled replay reads its own, rather than draining the stream after each checking kernel.
    eager_status: OnceLock<Tensor>,
    eager_status_pending: AtomicBool,
}

// The transform, signal length, and batch a cuFFT plan runs.
pub type FftPlanKey = (TransformType, usize, usize);

impl CudaRuntime {
    fn create(device_id: i32) -> Result<Self> {
        let _range = CudaNvtxRange::new(|| "singe cuda runtime create");
        let context = {
            let _range = CudaNvtxRange::new(|| "singe cuda context");
            cuda_context_for_device(device_id)?
        };
        context.bind()?;
        let properties = context.device().properties()?;
        let architecture = format!("sm_{}{}", properties.major, properties.minor).parse()?;
        Ok(Self {
            context,
            architecture,
            cublas: OnceLock::new(),
            cublaslt: OnceLock::new(),
            cudnn: OnceLock::new(),
            cutensor: OnceLock::new(),
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
            eager_stream: OnceLock::new(),
            eager_status: OnceLock::new(),
            eager_status_pending: AtomicBool::new(false),
        })
    }

    // Returns the stream eager work runs on when its context names none: kernels, library calls, and copies, none of which wait for it.
    // It is a blocking stream, so a host copy on the legacy default stream waits for what it holds.
    pub(crate) fn eager_stream(&self) -> Result<&CudaStream> {
        if let Some(stream) = self.eager_stream.get() {
            return Ok(stream);
        }

        let stream = self.context.create_stream(StreamOptions::default())?;
        Ok(self
            .eager_stream
            .get_or_init(|| CudaStream::new(Arc::new(stream))))
    }

    // Returns the status word an eager checking kernel about to launch writes, zeroed when created, and notes that it needs reading.
    pub(crate) fn eager_status_for_write(&self) -> Result<&Tensor> {
        let status = if let Some(status) = self.eager_status.get() {
            status
        } else {
            let device = Device::Cuda(self.context.device().id());
            let status = Tensor::zeroed_on(device, DataType::I32, &Shape::create([1])?)?;
            self.eager_status.get_or_init(|| status)
        };
        self.eager_status_pending.store(true, Ordering::Release);
        Ok(status)
    }

    // Reads and clears what eager checking kernels reported since the last read, and returns the error it records.
    // The caller has waited for the eager stream, or reads through a copy ordered after it.
    pub(crate) fn take_eager_status(&self) -> Result<()> {
        if !self.eager_status_pending.swap(false, Ordering::AcqRel) {
            return Ok(());
        }

        self.eager_status
            .get()
            .map_or(Ok(()), kernel_plan::take_status)
    }

    // Orders the work enqueued on `stream` from now on after what the eager stream holds, as a stream not ordered with it does before reading what eager work wrote.
    // A capturing stream is left alone, since a capture cannot wait on work outside it, and eager work refuses to run under capture.
    pub(crate) fn order_after_eager(&self, stream: &Stream) -> Result<()> {
        let eager = self.eager_stream()?;
        if eager.raw() == stream || stream.capture_status()? != StreamCaptureStatus::None {
            return Ok(());
        }

        stream.join(eager.raw())?;
        Ok(())
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

    /// Returns the mutex-protected cuBLAS handle associated with this runtime's CUDA context, creating it on first use.
    ///
    /// # Errors
    ///
    /// Returns an error if binding the context or creating the handle fails.
    pub fn cublas(&self) -> Result<&Mutex<CublasContext>> {
        self.handle(&self.cublas, "singe cublas create", |context| {
            Ok(CublasContext::create(context)?)
        })
    }

    /// Returns the mutex-protected cuBLASLt handle associated with this runtime's CUDA context, creating it on first use.
    ///
    /// # Errors
    ///
    /// Returns an error if binding the context or creating the handle fails.
    pub fn cublaslt(&self) -> Result<&Mutex<CublasLtContext>> {
        self.handle(&self.cublaslt, "singe cublaslt create", |context| {
            Ok(CublasLtContext::create(context)?)
        })
    }

    /// Returns the mutex-protected cuDNN handle associated with this runtime's CUDA context, creating it on first use.
    ///
    /// # Errors
    ///
    /// Returns an error if binding the context or creating the handle fails.
    pub fn cudnn(&self) -> Result<&Mutex<CudnnContext>> {
        self.handle(&self.cudnn, "singe cudnn create", |context| {
            Ok(CudnnContext::create(context)?)
        })
    }

    /// Returns the mutex-protected cuTENSOR handle associated with this runtime's CUDA context, creating it on first use.
    ///
    /// # Errors
    ///
    /// Returns an error if binding the context or creating the handle fails.
    pub fn cutensor(&self) -> Result<&Mutex<CutensorContext>> {
        self.handle(&self.cutensor, "singe cutensor create", |context| {
            Ok(CutensorContext::create(context)?)
        })
    }

    // The handle `slot` holds, created by `create` on its first use.
    // Two threads may both create one, and the second's is dropped.
    fn handle<'a, T>(
        &'a self,
        slot: &'a OnceLock<Mutex<T>>,
        range: &'static str,
        create: impl FnOnce(&Arc<CudaContext>) -> Result<T>,
    ) -> Result<&'a Mutex<T>> {
        if let Some(handle) = slot.get() {
            return Ok(handle);
        }

        let _range = CudaNvtxRange::new(|| range);
        self.context.bind()?;
        // A first use may fall inside a stream capture, where the strict modes refuse the allocations creating a handle makes.
        let previous = exchange_capture_mode(StreamCaptureMode::Relaxed)?;
        let created = create(&self.context);
        exchange_capture_mode(previous)?;
        let created = created?;
        Ok(slot.get_or_init(|| Mutex::new(created)))
    }
}

/// Returns the process-wide shared runtime for a CUDA device, creating it on first use.
/// The registry retains the runtime and its caches for the process lifetime.
///
/// # Errors
///
/// Returns an error if `device` is not CUDA, its ordinal cannot be represented, the runtime registry mutex is poisoned, or CUDA context binding or library handle creation fails.
pub fn runtime_for_device(device: Device) -> Result<Arc<CudaRuntime>> {
    static RUNTIMES: OnceLock<Box<[OnceLock<Arc<CudaRuntime>>]>> = OnceLock::new();
    static CREATING: Mutex<()> = Mutex::new(());

    let ordinal = |device_id: i32| usize::try_from(device_id).ok();
    if let Some(runtime) = device
        .cuda_id()
        .and_then(ordinal)
        .and_then(|index| RUNTIMES.get()?.get(index)?.get())
    {
        return Ok(Arc::clone(runtime));
    }

    let device_id = cuda_device_id(device)?;
    let count = device_count()?;
    let runtimes = RUNTIMES.get_or_init(|| (0..count).map(|_| OnceLock::new()).collect());
    let slot = ordinal(device_id)
        .and_then(|index| runtimes.get(index))
        .ok_or_else(|| Error::cuda_device_out_of_range(device_id, count))?;
    let _creating = CREATING.locked("cuda runtime creation")?;
    if let Some(runtime) = slot.get() {
        return Ok(Arc::clone(runtime));
    }

    let runtime = Arc::new(CudaRuntime::create(device_id)?);
    Ok(Arc::clone(slot.get_or_init(|| runtime)))
}

#[cfg(all(test, feature = "testing"))]
mod tests;
