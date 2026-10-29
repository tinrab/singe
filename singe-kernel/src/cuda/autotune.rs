//! Shared CUDA kernel autotuning support.
//!
//! Kernels that expose tunable launch parameters use three related APIs:
//!
//! * `*_config.tune(options)` runs autotuning on first use and stores the best
//!   tuning in the process-local [`MemoryTuneCache`].
//! * `*_config.cached_tuning()` requires a cache hit for the generated
//!   [`TuneKey`]. This is useful after a warmup pass and is safe for CUDA graph
//!   capture when the entry is already present.
//! * `*_config.explicit_tuning(tuning)` bypasses the cache and launches with a
//!   caller-provided tuning value.
//!
//! The cache is intentionally in-memory only. With the `serde` feature enabled,
//! users can serialize [`TuneReport`], [`TuneKey`], and the operation-specific
//! tuning values returned in `TuneReport::best`. A typical persistence flow is:
//! tune once, write `(report.key, report.best)` to an application cache, then
//! later validate that the loaded key matches the current problem/device and use
//! `explicit_tuning(report.best)` for the actual launch.

use std::{
    any::Any,
    collections::HashMap,
    hash::Hash,
    sync::{Mutex, OnceLock},
};

use singe_cuda::{
    device::Device,
    event::EventRecordFlags,
    stream::{Stream, StreamCaptureStatus},
};

use crate::error::{Error, Result};

const TUNE_KEY_SCHEMA: u32 = 2;

/// Controls how much work an autotune request performs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TuneOptions {
    /// Candidate warmup launches before timing starts.
    pub warmup: usize,
    /// Timed launches per candidate. This is normalized to at least one.
    pub repeats: usize,
    /// Maximum number of candidates to evaluate.
    pub max_candidates: usize,
}

impl TuneOptions {
    /// Sets warmup, repeat, and candidate limits; tuning normalizes zero repeats to one.
    #[must_use]
    pub const fn create(warmup: usize, repeats: usize, max_candidates: usize) -> Self {
        Self {
            warmup,
            repeats,
            max_candidates,
        }
    }

    fn normalized(self) -> Self {
        Self {
            warmup: self.warmup,
            repeats: self.repeats.max(1),
            max_candidates: self.max_candidates,
        }
    }
}

impl Default for TuneOptions {
    fn default() -> Self {
        Self {
            warmup: 1,
            repeats: 5,
            max_candidates: usize::MAX,
        }
    }
}

/// Selects how a kernel obtains its tuning parameters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TuningChoice<T> {
    /// Use the kernel's non-tuned default path.
    #[default]
    Default,
    /// Launch with a caller-provided tuning value.
    Explicit(T),
    /// Require an in-memory cache hit for the generated [`TuneKey`].
    Cached,
    /// Tune on cache miss, then store and use the winner.
    Tune(TuneOptions),
    /// Always tune again, replacing any cached entry.
    Retune(TuneOptions),
}

/// Optional compiler hints passed through to cutile-generated kernels.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KernelCompileConfig {
    /// Optional native compiler occupancy hint; `None` leaves the compiler default.
    pub occupancy: Option<i32>,
    /// Optional compiler hint for the number of CTAs in a CGA.
    pub num_cta_in_cga: Option<i32>,
    /// Optional native compiler divisibility hint; `None` leaves the compiler default.
    pub max_divisibility: Option<i32>,
}

impl KernelCompileConfig {
    /// Replaces the compiler occupancy hint without validating its value.
    #[must_use]
    pub const fn occupancy(mut self, value: i32) -> Self {
        self.occupancy = Some(value);
        self
    }

    /// Replaces the compiler CTA-count hint without validating its value.
    #[must_use]
    pub const fn num_cta_in_cga(mut self, value: i32) -> Self {
        self.num_cta_in_cga = Some(value);
        self
    }

    /// Replaces the compiler divisibility hint without validating its value.
    #[must_use]
    pub const fn max_divisibility(mut self, value: i32) -> Self {
        self.max_divisibility = Some(value);
        self
    }

    /// Returns whether every optional compiler hint is unset.
    #[must_use]
    pub const fn is_default(self) -> bool {
        self.occupancy.is_none() && self.num_cta_in_cga.is_none() && self.max_divisibility.is_none()
    }
}

/// Two-dimensional tile shape used by tunings that expose row and column tiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tile2d {
    /// Number of rows in the tile.
    pub rows: usize,
    /// Number of columns in the tile.
    pub columns: usize,
}

impl Tile2d {
    /// Records a row/column tile shape without validating operation-specific limits.
    #[must_use]
    pub const fn create(rows: usize, columns: usize) -> Self {
        Self { rows, columns }
    }
}

/// Result of an autotune run.
///
/// `best` is the tuning value to persist or pass back through
/// `explicit_tuning`. `key` identifies the exact device/problem shape the
/// tuning was measured for, and `candidates` records timing or failure details
/// for diagnostics.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TuneReport<C> {
    /// Device, operation, backend, and problem identity used for these measurements.
    pub key: TuneKey,
    /// Tuning with the lowest measured average latency.
    pub best: C,
    /// Candidate timings and failures in evaluation order.
    pub candidates: Vec<CandidateMeasurement<C>>,
}

/// Measurement result for one candidate tuning value.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CandidateMeasurement<C> {
    /// Tuning parameters evaluated for this candidate.
    pub config: C,
    /// Completed timing samples in milliseconds, including any collected before a later failure.
    pub samples_ms: Vec<f32>,
    /// Mean measured latency in milliseconds; autotuning reports infinity for failed candidates.
    pub average_ms: f32,
    /// Minimum measured latency in milliseconds; autotuning reports infinity for failed candidates.
    pub min_ms: f32,
    /// Whether the candidate was measured, rejected, or failed.
    pub status: CandidateStatus,
}

/// Outcome for one autotune candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CandidateStatus {
    /// All requested warmup and timing iterations succeeded.
    Measured,
    /// The candidate was excluded, with an explanation.
    Rejected {
        /// Explanation of why the candidate was excluded.
        reason: String,
    },
    /// Preparing, launching, or timing the candidate failed.
    Failed {
        /// Display text of the failure observed while evaluating the candidate.
        error: String,
    },
}

/// Stable-enough device identity included in tune cache keys.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceFingerprint {
    /// CUDA device ordinal in the current process.
    pub ordinal: i32,
    /// Native 16-byte device UUID.
    pub uuid: [u8; 16],
    /// Major CUDA compute-capability version.
    pub sm_major: i32,
    /// Minor CUDA compute-capability version.
    pub sm_minor: i32,
    /// Number of streaming multiprocessors reported by the device.
    pub multiprocessors: i32,
}

impl DeviceFingerprint {
    /// Queries the device identity and architecture used in tuning keys.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA device properties cannot be queried.
    pub fn create(device: Device) -> Result<Self> {
        let properties = device.properties()?;
        Ok(Self {
            ordinal: device.id(),
            uuid: properties.uuid.bytes,
            sm_major: properties.major,
            sm_minor: properties.minor,
            multiprocessors: properties.multi_processor_count,
        })
    }
}

/// Complete key for a measured tuning value.
///
/// Persisted tuning values should be reused only when this key matches the
/// current launch. The schema field exists so future incompatible key layouts
/// can invalidate old persisted records.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TuneKey {
    /// Version identifying the tuning-key representation.
    pub schema: u32,
    /// Kernel operation name.
    pub op: String,
    /// Execution backend name.
    pub backend: String,
    /// Device identity and architecture against which tuning was measured.
    pub device: DeviceFingerprint,
    /// Operation-specific dimensions, layout, and feature settings.
    pub problem: TuneProblemKey,
}

impl TuneKey {
    /// Combines the current schema version with operation, backend, device, and problem identity.
    #[must_use]
    pub fn create(
        op: &'static str,
        backend: &'static str,
        device: DeviceFingerprint,
        problem: TuneProblemKey,
    ) -> Self {
        Self {
            schema: TUNE_KEY_SCHEMA,
            op: op.to_owned(),
            backend: backend.to_owned(),
            device,
            problem,
        }
    }
}

/// Operation-specific dimensions and layout values that affect tuning.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TuneProblemKey {
    /// Single matrix-product geometry and layouts.
    Matmul {
        /// Element-encoding identifier used by the kernel.
        dtype: String,
        /// Logical row count.
        rows: usize,
        /// Logical column count.
        columns: usize,
        /// Matrix contraction dimension K.
        reduction: usize,
        /// Distance in elements between consecutive row positions of the left matrix.
        lhs_row_stride: usize,
        /// Distance in elements between consecutive row positions of the right matrix.
        rhs_row_stride: usize,
        /// Distance in elements between consecutive row positions of the output tensor.
        output_row_stride: usize,
        /// Whether the left matrix is logically transposed.
        transpose_lhs: bool,
        /// Whether the right matrix is logically transposed.
        transpose_rhs: bool,
    },
    /// Batched matrix-product geometry and layouts.
    Bmm {
        /// Element-encoding identifier used by the kernel.
        dtype: String,
        /// Number of independent batches.
        batch: usize,
        /// Logical row count.
        rows: usize,
        /// Logical column count.
        columns: usize,
        /// Matrix contraction dimension K.
        reduction: usize,
        /// Distance in elements between consecutive batch positions of the left matrix.
        lhs_batch_stride: usize,
        /// Distance in elements between consecutive row positions of the left matrix.
        lhs_row_stride: usize,
        /// Distance in elements between consecutive batch positions of the right matrix.
        rhs_batch_stride: usize,
        /// Distance in elements between consecutive row positions of the right matrix.
        rhs_row_stride: usize,
        /// Distance in elements between consecutive batch positions of the output tensor.
        output_batch_stride: usize,
        /// Distance in elements between consecutive row positions of the output tensor.
        output_row_stride: usize,
        /// Whether the left matrix is logically transposed.
        transpose_lhs: bool,
        /// Whether the right matrix is logically transposed.
        transpose_rhs: bool,
    },
    /// Row-wise softmax operation and geometry.
    Softmax {
        /// Element-encoding identifier used by the kernel.
        dtype: String,
        /// Operation variant distinguished by this tuning key.
        op_kind: String,
        /// Logical row count.
        rows: usize,
        /// Logical column count.
        columns: usize,
    },
    /// Reduction operation and row/column geometry.
    Reduction {
        /// Element-encoding identifier used by the kernel.
        dtype: String,
        /// Operation variant distinguished by this tuning key.
        op_kind: String,
        /// Logical row count.
        rows: usize,
        /// Logical column count.
        columns: usize,
    },
    /// Masked reduction operation and geometry.
    MaskReduction {
        /// Element-encoding identifier used by the kernel.
        dtype: String,
        /// Operation variant distinguished by this tuning key.
        op_kind: String,
        /// Logical row count.
        rows: usize,
        /// Logical column count.
        columns: usize,
    },
    /// Normalization geometry and optional affine operands.
    Normalization {
        /// Element-encoding identifier used by the kernel.
        dtype: String,
        /// Operation variant distinguished by this tuning key.
        op_kind: String,
        /// Logical row count.
        rows: usize,
        /// Logical column count.
        columns: usize,
        /// Whether an affine weight operand is present.
        has_weight: bool,
        /// Whether an affine bias operand is present.
        has_bias: bool,
    },
    /// Row-wise sparsemax geometry.
    Sparsemax {
        /// Element-encoding identifier used by the kernel.
        dtype: String,
        /// Logical row count.
        rows: usize,
        /// Logical column count.
        columns: usize,
    },
    /// One-dimensional fused operation and logical length.
    Fused1d {
        /// Element-encoding identifier used by the kernel.
        dtype: String,
        /// Operation variant distinguished by this tuning key.
        op_kind: String,
        /// Logical number of vector elements processed.
        length: usize,
    },
    /// Two-dimensional fused operation and geometry.
    Fused2d {
        /// Element-encoding identifier used by the kernel.
        dtype: String,
        /// Operation variant distinguished by this tuning key.
        op_kind: String,
        /// Logical row count.
        rows: usize,
        /// Logical column count.
        columns: usize,
    },
    /// Vector operation and logical length.
    Vector1d {
        /// Element-encoding identifier used by the kernel.
        dtype: String,
        /// Operation variant distinguished by this tuning key.
        op_kind: String,
        /// Logical number of vector elements processed.
        length: usize,
    },
    /// Attention decode geometry, tensor strides, and split-count limit.
    AttentionSplitK {
        /// Element-encoding identifier used by the kernel.
        dtype: String,
        /// Operation variant distinguished by this tuning key.
        op_kind: String,
        /// Number of independent batches.
        batch: usize,
        /// Number of key/value sequence positions.
        key_len: usize,
        /// Number of query heads.
        heads: usize,
        /// Number of key/value heads.
        kv_heads: usize,
        /// Number of features per attention head.
        head_dim: usize,
        /// Maximum split count included in the attention tuning problem.
        max_splits: usize,
        /// Distance in elements between consecutive batch positions of the query tensor.
        query_batch_stride: usize,
        /// Distance in elements between consecutive batch positions of the key tensor.
        key_batch_stride: usize,
        /// Distance in elements between consecutive batch positions of the value tensor.
        value_batch_stride: usize,
        /// Distance in elements between consecutive batch positions of the output tensor.
        output_batch_stride: usize,
        /// Distance in elements between consecutive head positions of the query tensor.
        query_head_stride: usize,
        /// Distance in elements between consecutive sequence positions of the key tensor.
        key_sequence_stride: usize,
        /// Distance in elements between consecutive sequence positions of the value tensor.
        value_sequence_stride: usize,
        /// Distance in elements between consecutive head positions of the output tensor.
        output_head_stride: usize,
        /// Distance in elements between consecutive head positions of the key tensor.
        key_head_stride: usize,
        /// Distance in elements between consecutive head positions of the value tensor.
        value_head_stride: usize,
    },
}

/// Thread-safe process-local storage of typed tuning values keyed by device and problem identity.
#[derive(Debug)]
pub struct MemoryTuneCache {
    entries: Mutex<HashMap<TuneKey, Box<dyn Any + Send + Sync>>>,
}

impl MemoryTuneCache {
    fn create() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Clones the cached tuning of type `T`, or returns `None` on a key miss.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored tuning has a different Rust type than `T`.
    ///
    /// # Panics
    ///
    /// Panics if the tuning-cache mutex is poisoned.
    pub fn get<T>(&self, key: &TuneKey) -> Result<Option<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let entries = self.entries.lock().expect("tune cache poisoned");
        let Some(value) = entries.get(key) else {
            return Ok(None);
        };
        let cached = value.downcast_ref::<T>().cloned();
        drop(entries);
        cached
            .ok_or_else(|| Error::TuneCacheTypeMismatch { op: key.op.clone() })
            .map(Some)
    }

    /// Stores a typed tuning value, replacing any previous entry for the key.
    ///
    /// # Panics
    ///
    /// Panics if the tuning-cache mutex is poisoned.
    pub fn insert<T>(&self, key: TuneKey, value: T)
    where
        T: Clone + Send + Sync + 'static,
    {
        let mut entries = self.entries.lock().expect("tune cache poisoned");
        entries.insert(key, Box::new(value));
    }

    /// Removes a key and returns whether an entry was present.
    ///
    /// # Panics
    ///
    /// Panics if the tuning-cache mutex is poisoned.
    pub fn remove(&self, key: &TuneKey) -> bool {
        let mut entries = self.entries.lock().expect("tune cache poisoned");
        entries.remove(key).is_some()
    }

    /// Removes every cached tuning value.
    ///
    /// # Panics
    ///
    /// Panics if the tuning-cache mutex is poisoned.
    pub fn clear(&self) {
        let mut entries = self.entries.lock().expect("tune cache poisoned");
        entries.clear();
    }

    /// Returns the number of cached device/problem keys.
    ///
    /// # Panics
    ///
    /// Panics if the tuning-cache mutex is poisoned.
    pub fn len(&self) -> usize {
        let entries = self.entries.lock().expect("tune cache poisoned");
        entries.len()
    }

    /// Returns whether no tuning values are cached.
    ///
    /// # Panics
    ///
    /// Panics if the tuning-cache mutex is poisoned.
    pub fn is_empty(&self) -> bool {
        let entries = self.entries.lock().expect("tune cache poisoned");
        entries.is_empty()
    }
}

static TUNE_CACHE: OnceLock<MemoryTuneCache> = OnceLock::new();

/// Returns the process-local tune cache.
///
/// This cache is not persisted by the crate. Applications that want reuse
/// across process runs should serialize a [`TuneReport`] or a `(TuneKey, best)`
/// pair with the `serde` feature enabled, then launch later with
/// `explicit_tuning(best)` after checking the stored key.
pub fn cache() -> &'static MemoryTuneCache {
    TUNE_CACHE.get_or_init(MemoryTuneCache::create)
}

/// Clears the process-local tune cache.
///
/// # Panics
///
/// Panics if the tuning-cache mutex is poisoned.
pub fn clear_cache() {
    cache().clear();
}

/// Removes one process-local cache entry.
///
/// # Panics
///
/// Panics if the tuning-cache mutex is poisoned.
#[must_use]
pub fn remove_cache_entry(key: &TuneKey) -> bool {
    cache().remove(key)
}

/// Returns the number of entries in the process-local tune cache.
///
/// # Panics
///
/// Panics if the tuning-cache mutex is poisoned.
#[must_use]
pub fn cache_len() -> usize {
    cache().len()
}

/// Returns an error if the stream is currently being captured.
///
/// Tuning launches multiple candidate kernels and records events, so it cannot
/// run inside CUDA graph capture. Use [`TuningChoice::Explicit`] or a populated
/// [`TuningChoice::Cached`] entry when building captured graphs.
///
/// # Errors
///
/// Returns an error if querying stream capture status fails or the stream is capturing or has invalidated capture state.
pub fn ensure_not_capturing(stream: &Stream, op: &'static str) -> Result<()> {
    if stream.capture_status()? == StreamCaptureStatus::None {
        Ok(())
    } else {
        Err(Error::AutotuneDuringCapture { op: op.to_owned() })
    }
}

/// Builds the appropriate error for a cache miss in normal execution or graph capture.
#[must_use]
pub fn cache_miss_error(op: &'static str, capture_status: StreamCaptureStatus) -> Error {
    if capture_status == StreamCaptureStatus::None {
        Error::TuneCacheMiss { op: op.to_owned() }
    } else {
        Error::AutotuneDuringCapture { op: op.to_owned() }
    }
}

/// Measures candidate tunings and returns a report containing the best value.
///
/// The `launch` callback is called for warmup and timed iterations. Failed
/// candidates are recorded in the report and ignored for winner selection. This
/// function does not insert into [`cache`]; kernel-specific wrappers insert and
/// typically relaunch the winning candidate so output buffers contain the
/// selected result.
///
/// # Errors
///
/// Returns an error if tuning is attempted during capture, stream capture status cannot be queried, no candidates remain after applying the limit, timing events cannot be created, or no candidate completes all measurements successfully. Individual candidate launch and timing failures are retained in the report when another candidate succeeds.
#[expect(
    clippy::cast_precision_loss,
    reason = "the f32 algorithm converts integer dimensions and accumulators to its floating-point working precision"
)]
pub fn tune_candidates<C, F>(
    stream: &Stream,
    op: &'static str,
    key: TuneKey,
    options: TuneOptions,
    candidates: impl IntoIterator<Item = C>,
    mut launch: F,
) -> Result<TuneReport<C>>
where
    C: Clone + Send + Sync + 'static,
    F: FnMut(&C) -> Result<()>,
{
    ensure_not_capturing(stream, op)?;
    let options = options.normalized();
    let candidates = candidates
        .into_iter()
        .take(options.max_candidates)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(Error::NoTuneCandidates { op: op.to_owned() });
    }

    let context = stream.context();
    let start = context.create_event()?;
    let stop = context.create_event()?;

    let mut measurements = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let mut failed = None;
        for _ in 0..options.warmup {
            if let Err(error) = launch(&candidate) {
                failed = Some(error.to_string());
                break;
            }
        }
        if let Some(error) = failed {
            measurements.push(CandidateMeasurement {
                config: candidate,
                samples_ms: Vec::new(),
                average_ms: f32::INFINITY,
                min_ms: f32::INFINITY,
                status: CandidateStatus::Failed { error },
            });
            continue;
        }

        let mut samples_ms = Vec::with_capacity(options.repeats);
        let mut failed = None;
        for _ in 0..options.repeats {
            if let Err(error) = start.record(stream, EventRecordFlags::DEFAULT) {
                failed = Some(error.to_string());
                break;
            }
            if let Err(error) = launch(&candidate) {
                failed = Some(error.to_string());
                break;
            }
            if let Err(error) = stop.record(stream, EventRecordFlags::DEFAULT) {
                failed = Some(error.to_string());
                break;
            }
            if let Err(error) = stop.synchronize() {
                failed = Some(error.to_string());
                break;
            }
            match stop.elapsed_time_since(&start) {
                Ok(sample) => samples_ms.push(sample),
                Err(error) => {
                    failed = Some(error.to_string());
                    break;
                }
            }
        }

        if let Some(error) = failed {
            measurements.push(CandidateMeasurement {
                config: candidate,
                samples_ms,
                average_ms: f32::INFINITY,
                min_ms: f32::INFINITY,
                status: CandidateStatus::Failed { error },
            });
            continue;
        }

        let total = samples_ms.iter().copied().sum::<f32>();
        let average_ms = total / samples_ms.len() as f32;
        let min_ms = samples_ms.iter().copied().fold(f32::INFINITY, f32::min);
        measurements.push(CandidateMeasurement {
            config: candidate,
            samples_ms,
            average_ms,
            min_ms,
            status: CandidateStatus::Measured,
        });
    }

    let best = best_measured_candidate(op, &measurements)?;

    Ok(TuneReport {
        key,
        best,
        candidates: measurements,
    })
}

fn best_measured_candidate<C>(
    op: &'static str,
    measurements: &[CandidateMeasurement<C>],
) -> Result<C>
where
    C: Clone,
{
    measurements
        .iter()
        .filter(|measurement| measurement.status == CandidateStatus::Measured)
        .min_by(|lhs, rhs| lhs.average_ms.total_cmp(&rhs.average_ms))
        .map(|measurement| measurement.config.clone())
        .ok_or_else(|| Error::NoSuccessfulTuneCandidates { op: op.to_owned() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key(op: &'static str, columns: usize) -> TuneKey {
        TuneKey::create(
            op,
            "test",
            DeviceFingerprint {
                ordinal: 0,
                uuid: [0; 16],
                sm_major: 9,
                sm_minor: 0,
                multiprocessors: 1,
            },
            TuneProblemKey::Reduction {
                dtype: "f32".to_owned(),
                op_kind: op.to_owned(),
                rows: 2,
                columns,
            },
        )
    }

    #[test]
    fn test_memory_cache_reports_hit_miss_and_len() {
        let cache = MemoryTuneCache::create();
        let key = test_key("cache_hit", 16);

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.get::<u32>(&key).unwrap(), None);

        cache.insert(key.clone(), 42u32);

        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get::<u32>(&key).unwrap(), Some(42));
    }

    #[test]
    fn test_memory_cache_replaces_existing_entries() {
        let cache = MemoryTuneCache::create();
        let key = test_key("cache_replace", 32);

        cache.insert(key.clone(), 1u32);
        cache.insert(key.clone(), 7u32);

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get::<u32>(&key).unwrap(), Some(7));
    }

    #[test]
    fn test_memory_cache_removes_and_clears_entries() {
        let cache = MemoryTuneCache::create();
        let first = test_key("cache_remove_first", 64);
        let second = test_key("cache_remove_second", 128);

        cache.insert(first.clone(), 1u32);
        cache.insert(second.clone(), 2u32);

        assert!(cache.remove(&first));
        assert!(!cache.remove(&first));
        assert_eq!(cache.get::<u32>(&first).unwrap(), None);
        assert_eq!(cache.get::<u32>(&second).unwrap(), Some(2));

        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.get::<u32>(&second).unwrap(), None);
    }

    #[test]
    fn test_memory_cache_rejects_type_mismatch() {
        let cache = MemoryTuneCache::create();
        let key = test_key("cache_type_mismatch", 256);

        cache.insert(key.clone(), 5u32);

        let error = cache.get::<u64>(&key).unwrap_err();
        assert!(matches!(
            error,
            Error::TuneCacheTypeMismatch { op } if op == "cache_type_mismatch"
        ));
    }

    #[test]
    fn test_best_measured_candidate_ignores_failed_and_rejected_entries() {
        let measurements = [
            CandidateMeasurement {
                config: 1u32,
                samples_ms: Vec::new(),
                average_ms: f32::INFINITY,
                min_ms: f32::INFINITY,
                status: CandidateStatus::Failed {
                    error: "launch failed".into(),
                },
            },
            CandidateMeasurement {
                config: 2u32,
                samples_ms: Vec::new(),
                average_ms: f32::INFINITY,
                min_ms: f32::INFINITY,
                status: CandidateStatus::Rejected {
                    reason: "unsupported tile".into(),
                },
            },
            CandidateMeasurement {
                config: 3u32,
                samples_ms: vec![0.4, 0.5],
                average_ms: 0.45,
                min_ms: 0.4,
                status: CandidateStatus::Measured,
            },
            CandidateMeasurement {
                config: 4u32,
                samples_ms: vec![0.2, 0.3],
                average_ms: 0.25,
                min_ms: 0.2,
                status: CandidateStatus::Measured,
            },
        ];

        assert_eq!(
            best_measured_candidate("best_candidate", &measurements).unwrap(),
            4
        );
    }

    #[test]
    fn test_best_measured_candidate_requires_one_measured_entry() {
        let measurements = [
            CandidateMeasurement {
                config: 1u32,
                samples_ms: Vec::new(),
                average_ms: f32::INFINITY,
                min_ms: f32::INFINITY,
                status: CandidateStatus::Failed {
                    error: "launch failed".into(),
                },
            },
            CandidateMeasurement {
                config: 2u32,
                samples_ms: Vec::new(),
                average_ms: f32::INFINITY,
                min_ms: f32::INFINITY,
                status: CandidateStatus::Rejected {
                    reason: "unsupported tile".into(),
                },
            },
        ];

        let error = best_measured_candidate("no_measured_candidate", &measurements).unwrap_err();
        assert!(matches!(
            error,
            Error::NoSuccessfulTuneCandidates { op } if op == "no_measured_candidate"
        ));
    }

    #[test]
    fn test_cache_miss_error_uses_cache_miss_outside_capture() {
        let error = cache_miss_error("cache_miss_policy", StreamCaptureStatus::None);
        assert!(matches!(
            error,
            Error::TuneCacheMiss { op } if op == "cache_miss_policy"
        ));
    }

    #[test]
    fn test_cache_miss_error_uses_capture_error_during_capture() {
        for status in [
            StreamCaptureStatus::Active,
            StreamCaptureStatus::Invalidated,
        ] {
            let error = cache_miss_error("capture_policy", status);
            assert!(matches!(
                error,
                Error::AutotuneDuringCapture { op } if op == "capture_policy"
            ));
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_tune_report_round_trips_through_json() {
        let key = test_key("serde_report", 64);
        let report = TuneReport {
            key: key.clone(),
            best: KernelCompileConfig::default().occupancy(2),
            candidates: vec![CandidateMeasurement {
                config: KernelCompileConfig::default().occupancy(2),
                samples_ms: vec![0.25, 0.5],
                average_ms: 0.375,
                min_ms: 0.25,
                status: CandidateStatus::Measured,
            }],
        };

        let json = serde_json::to_string(&report).unwrap();
        let decoded: TuneReport<KernelCompileConfig> = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.key, key);
        assert_eq!(decoded.best, KernelCompileConfig::default().occupancy(2));
        assert_eq!(decoded.candidates.len(), 1);
        assert_eq!(decoded.candidates[0].status, CandidateStatus::Measured);
    }
}
