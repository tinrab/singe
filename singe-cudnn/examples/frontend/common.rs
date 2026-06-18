use std::sync::PoisonError;
use std::sync::{LazyLock, Mutex, MutexGuard};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, nvrtc, stream::Stream};
use singe_cudnn::{context::Context, error::Result};

#[allow(dead_code)]
pub fn init_f32_image(size: usize, seed: &mut u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        values.push((*seed as f32) * 2.328_306_4e-10_f32);
    }
    values
}

#[allow(dead_code)]
pub fn count_nonfinite_f32(values: &[f32]) -> usize {
    values.iter().filter(|value| !value.is_finite()).count()
}

pub struct ExampleContext {
    #[allow(dead_code)]
    pub cudnn: Context,
    #[allow(dead_code)]
    pub stream: Stream,
    #[allow(dead_code)]
    _lock: MutexGuard<'static, ()>,
}

static CUDA_CONTEXT_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

impl ExampleContext {
    pub fn create() -> Result<Self> {
        let lock = CUDA_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // Keep the nvrtc dependency linked for frontend example binaries.
        let _ = nvrtc::version()?;
        let cuda = CudaContext::create()?;
        let cudnn = Context::create(&cuda)?;
        let stream = cuda.create_stream()?;
        cudnn.set_stream(Some(&stream))?;
        Ok(Self {
            cudnn,
            stream,
            _lock: lock,
        })
    }

    #[allow(dead_code)]
    pub fn get_compute_capability(&self) -> i32 {
        let properties = self.cudnn.cuda_context().device().properties().unwrap();
        properties.major * 10 + properties.minor
    }

    #[allow(dead_code)]
    pub fn is_blackwell_device(&self) -> bool {
        (100..=120).contains(&self.get_compute_capability())
    }

    #[allow(dead_code)]
    pub fn is_hopper_device(&self) -> bool {
        (90..=100).contains(&self.get_compute_capability())
    }
}

#[allow(dead_code)]
pub fn workspace(size: usize) -> Result<DeviceMemory<u8>> {
    Ok(DeviceMemory::create(size)?)
}

#[allow(dead_code)]
pub fn finish(name: &str, result: Result<()>) -> Result<()> {
    match result {
        Ok(()) => {
            println!("{name}: ok");
            Ok(())
        }
        Err(error) => Err(error),
    }
}
