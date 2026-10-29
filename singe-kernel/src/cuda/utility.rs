#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_scalar",
    feature = "kernel_unary"
))]
use singe_cuda::stream::Stream;

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_scalar",
    feature = "kernel_unary"
))]
use crate::cuda::autotune::{
    self, DeviceFingerprint, KernelCompileConfig, TuneOptions, TuneProblemKey, TuneReport,
    TuningChoice,
};
use crate::{error::Result, utility::ensure_len};

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_scalar",
    feature = "kernel_unary"
))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorTuning {
    pub tile_width: usize,
    pub compile: KernelCompileConfig,
}

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_scalar",
    feature = "kernel_unary"
))]
impl VectorTuning {
    #[must_use]
    pub const fn tile(tile_width: usize) -> Self {
        Self {
            tile_width,
            compile: KernelCompileConfig {
                occupancy: None,
                num_cta_in_cga: None,
                max_divisibility: None,
            },
        }
    }

    #[must_use]
    pub const fn with_compile(mut self, compile: KernelCompileConfig) -> Self {
        self.compile = compile;
        self
    }
}

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_scalar",
    feature = "kernel_unary"
))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorConfig {
    pub length: usize,
    pub tuning: TuningChoice<VectorTuning>,
}

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_scalar",
    feature = "kernel_unary"
))]
impl VectorConfig {
    #[must_use]
    pub const fn create(length: usize) -> Self {
        Self {
            length,
            tuning: TuningChoice::Default,
        }
    }

    #[must_use]
    pub const fn with_tuning(mut self, tuning: TuningChoice<VectorTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    #[must_use]
    pub const fn explicit_tuning(self, tuning: VectorTuning) -> Self {
        self.with_tuning(TuningChoice::Explicit(tuning))
    }

    #[must_use]
    pub const fn cached_tuning(self) -> Self {
        self.with_tuning(TuningChoice::Cached)
    }

    #[must_use]
    pub const fn tune(self, options: TuneOptions) -> Self {
        self.with_tuning(TuningChoice::Tune(options))
    }

    #[must_use]
    pub const fn retune(self, options: TuneOptions) -> Self {
        self.with_tuning(TuningChoice::Retune(options))
    }
}

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_scalar",
    feature = "kernel_unary"
))]
pub fn validate_vector_lengths(
    out_len: usize,
    input_len: usize,
    config: VectorConfig,
) -> Result<()> {
    ensure_len(out_len, config.length)?;
    ensure_len(input_len, config.length)?;
    Ok(())
}

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_scalar",
    feature = "kernel_unary"
))]
pub fn vector_tune_key(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: VectorConfig,
) -> Result<autotune::TuneKey> {
    let device = DeviceFingerprint::create(stream.context().device())?;
    Ok(autotune::TuneKey::create(
        op,
        "cutile",
        device,
        TuneProblemKey::Vector1d {
            dtype: dtype.to_owned(),
            op_kind: op.to_owned(),
            length: config.length,
        },
    ))
}

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_scalar",
    feature = "kernel_unary"
))]
pub fn vector_tile_candidates() -> Vec<VectorTuning> {
    [32, 64, 128, 256]
        .into_iter()
        .map(VectorTuning::tile)
        .collect()
}

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_cast",
    feature = "kernel_elementwise",
    feature = "kernel_quantization",
    feature = "kernel_scalar",
    feature = "kernel_unary"
))]
pub fn resolve_vector_tuning<C, F>(
    stream: &Stream,
    op: &'static str,
    key: autotune::TuneKey,
    choice: TuningChoice<VectorTuning>,
    candidates: C,
    tune: F,
) -> Result<Option<VectorTuning>>
where
    C: FnOnce() -> Vec<VectorTuning>,
    F: FnOnce(TuneOptions, Vec<VectorTuning>) -> Result<TuneReport<VectorTuning>>,
{
    match choice {
        TuningChoice::Default => Ok(None),
        TuningChoice::Explicit(tuning) => Ok(Some(tuning)),
        TuningChoice::Cached => {
            if let Some(tuning) = autotune::cache().get(&key)? {
                Ok(Some(tuning))
            } else {
                Err(autotune::cache_miss_error(op, stream.capture_status()?))
            }
        }
        TuningChoice::Tune(options) => {
            if let Some(tuning) = autotune::cache().get(&key)? {
                return Ok(Some(tuning));
            }
            let report = tune(options, candidates())?;
            autotune::cache().insert(key, report.best);
            Ok(Some(report.best))
        }
        TuningChoice::Retune(options) => {
            let report = tune(options, candidates())?;
            autotune::cache().insert(key, report.best);
            Ok(Some(report.best))
        }
    }
}

#[cfg(any(feature = "kernel_elementwise", feature = "kernel_fused",))]
#[allow(dead_code, reason = "only used by some kernel feature combinations")]
pub fn ensure_binary_lengths(out_len: usize, lhs_len: usize, rhs_len: usize) -> Result<usize> {
    ensure_len(lhs_len, out_len)?;
    ensure_len(rhs_len, out_len)?;
    Ok(out_len)
}

#[cfg(any(
    feature = "kernel_fused",
    feature = "kernel_scalar",
    feature = "kernel_unary",
))]
#[allow(dead_code, reason = "only used by some kernel feature combinations")]
pub fn ensure_unary_lengths(out_len: usize, input_len: usize) -> Result<usize> {
    ensure_len(input_len, out_len)?;
    Ok(out_len)
}
