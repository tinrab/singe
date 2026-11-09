#[cfg(feature = "cuda_nvtx")]
use std::{env, sync::OnceLock};

#[cfg(feature = "cuda_nvtx")]
use singe_cuda::nvtx::LocalRange;

#[derive(Debug)]
pub struct CudaNvtxRange {
    #[cfg(feature = "cuda_nvtx")]
    _range: Option<LocalRange>,
}

impl CudaNvtxRange {
    pub fn create(message: &'static str) -> Self {
        #[cfg(feature = "cuda_nvtx")]
        {
            let range = nvtx_enabled()
                .then(|| LocalRange::create(message).ok())
                .flatten();
            Self { _range: range }
        }

        #[cfg(not(feature = "cuda_nvtx"))]
        {
            let _ = message;
            Self {}
        }
    }
}

#[cfg(feature = "cuda_nvtx")]
fn nvtx_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var("SINGE_CUDA_NVTX").is_ok_and(|value| {
            let value = value.trim();
            !value.is_empty()
                && !matches!(
                    value.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
        })
    })
}
