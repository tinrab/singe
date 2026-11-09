use std::cell::Cell;

use crate::storage::cpu::CpuStorage;

/// Current-thread CPU storage boundary copies, available with `testing`.
/// Counts byte imports, host exports, and explicit storage clones.
/// Typed borrows and moves of an owned vector do not copy and are not counted.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuStorageCopyStats {
    /// Number of host byte imports on the current thread.
    pub imports: usize,
    /// Total bytes imported from host byte slices on the current thread.
    pub imported_bytes: usize,
    /// Number of host byte exports on the current thread.
    pub exports: usize,
    /// Total bytes exported to host memory on the current thread.
    pub exported_bytes: usize,
    /// Number of explicit CPU storage copies on the current thread.
    pub clones: usize,
    /// Total bytes copied by explicit CPU storage clones on the current thread.
    pub cloned_bytes: usize,
}

thread_local! {
    static COPIES: Cell<CpuStorageCopyStats> = Cell::default();
}

impl CpuStorage {
    /// Returns CPU storage copy counters for the current thread since its last reset.
    #[must_use]
    pub fn copy_stats() -> CpuStorageCopyStats {
        COPIES.get()
    }

    /// Resets CPU storage copy counters on the current thread to zero.
    pub fn reset_copy_stats() {
        COPIES.set(CpuStorageCopyStats::default());
    }
}

pub fn record_import(bytes: usize) {
    COPIES.with(|copies| {
        copies.update(|mut stats| {
            stats.imports += 1;
            stats.imported_bytes += bytes;
            stats
        });
    });
}

pub fn record_export(bytes: usize) {
    COPIES.with(|copies| {
        copies.update(|mut stats| {
            stats.exports += 1;
            stats.exported_bytes += bytes;
            stats
        });
    });
}

pub fn record_clone(bytes: usize) {
    COPIES.with(|copies| {
        copies.update(|mut stats| {
            stats.clones += 1;
            stats.cloned_bytes += bytes;
            stats
        });
    });
}
