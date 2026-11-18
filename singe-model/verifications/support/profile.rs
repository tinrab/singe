// What the GPU did while some work ran, from CUPTI's activity records: its kernels' time and count, and its copies' time.

use std::time::{Duration, Instant};

use anyhow::Result;
use singe_cupti::{
    activity::{ActivityBufferOptions, ActivityCollector},
    types::{ActivityFlushFlag, ActivityKind},
};

const KINDS: [ActivityKind; 2] = [ActivityKind::ConcurrentKernel, ActivityKind::MemoryCopy];

#[derive(Debug, Clone, Copy, Default)]
pub struct GpuWork {
    pub kernels: Duration,
    pub launches: u64,
    pub copies: Duration,
}

impl GpuWork {
    pub fn busy(&self) -> Duration {
        self.kernels + self.copies
    }
}

// Runs `work`, returning what it gave, its wall time, and the GPU's work during it.
pub fn profiled<T>(work: impl FnOnce() -> Result<T>) -> Result<(T, Duration, GpuWork)> {
    let collector = ActivityCollector::create(&ActivityBufferOptions::default())?;
    for kind in KINDS {
        collector.enable(kind)?;
    }
    let start = Instant::now();
    let value = work()?;
    let wall = start.elapsed();
    collector.flush_all(ActivityFlushFlag::Forced)?;
    for kind in KINDS {
        collector.disable(kind)?;
    }

    let mut gpu = GpuWork::default();
    for buffer in collector.take_buffers()? {
        for record in buffer.records() {
            let record = record?;
            if let Some(kernel) = record.kernel()? {
                gpu.kernels += Duration::from_nanos(kernel.end.saturating_sub(kernel.start));
                gpu.launches += 1;
            } else if let Some(copy) = record.memory_copy()? {
                gpu.copies += Duration::from_nanos(copy.end.saturating_sub(copy.start));
            }
        }
    }
    Ok((value, wall, gpu))
}
