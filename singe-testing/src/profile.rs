// What the GPU did while some work ran, from CUPTI's activity records: each kernel's time and launches, by name, and the copies' time.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use singe_cupti::{
    activity::{ActivityBufferOptions, ActivityCollector},
    error::Result,
    types::{ActivityFlushFlag, ActivityKind},
};

const KINDS: &[ActivityKind] = &[ActivityKind::ConcurrentKernel, ActivityKind::MemoryCopy];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KernelWork {
    pub time: Duration,
    pub launches: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuWork {
    pub kernels: HashMap<String, KernelWork>,
    pub copies: Duration,
}

impl GpuWork {
    #[must_use]
    pub fn kernel_time(&self) -> Duration {
        self.kernels.values().map(|kernel| kernel.time).sum()
    }

    #[must_use]
    pub fn launches(&self) -> u64 {
        self.kernels.values().map(|kernel| kernel.launches).sum()
    }

    #[must_use]
    pub fn busy(&self) -> Duration {
        self.kernel_time() + self.copies
    }
}

// Runs `work`, returning what it gave, its wall time, and the GPU's work during it.
pub fn profiled<T>(work: impl FnOnce() -> T) -> Result<(T, Duration, GpuWork)> {
    let collector = ActivityCollector::create(&ActivityBufferOptions::default())?;
    for &kind in KINDS {
        collector.enable(kind)?;
    }

    let start = Instant::now();
    let value = work();
    let wall = start.elapsed();
    collector.flush_all(ActivityFlushFlag::Forced)?;
    for &kind in KINDS {
        collector.disable(kind)?;
    }

    let mut gpu = GpuWork::default();
    for buffer in collector.take_buffers()? {
        for record in buffer.records() {
            let record = record?;
            if let Some(kernel) = record.kernel()? {
                let name = kernel.name.map_or_else(
                    || "?".to_owned(),
                    |name| name.to_string_lossy().into_owned(),
                );
                let entry = gpu.kernels.entry(name).or_default();
                entry.time += Duration::from_nanos(kernel.end.saturating_sub(kernel.start));
                entry.launches += 1;
            } else if let Some(copy) = record.memory_copy()? {
                gpu.copies += Duration::from_nanos(copy.end.saturating_sub(copy.start));
            }
        }
    }
    Ok((value, wall, gpu))
}
