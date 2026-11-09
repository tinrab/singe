//! The CPU intra-op thread count, the persistent worker pool, and chunked parallel loops.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread::available_parallelism,
};

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::error::Result;

/// Smallest number of elements worth handing to another thread.
pub const GRAIN_SIZE: usize = 32_768;

static THREADS: AtomicUsize = AtomicUsize::new(0);

/// The persistent worker pool and the thread count it was built for.
static POOL: Mutex<Option<(usize, Arc<ThreadPool>)>> = Mutex::new(None);

/// Returns the number of threads CPU kernels use, which defaults to the available parallelism.
#[must_use]
pub fn num_threads() -> usize {
    match THREADS.load(Ordering::Relaxed) {
        0 => available_parallelism().map_or(1, usize::from),
        threads => threads,
    }
}

/// Sets the number of threads CPU kernels use; zero restores the default.
pub fn set_num_threads(threads: usize) {
    THREADS.store(threads, Ordering::Relaxed);
}

/// Returns the persistent worker pool for the current thread count, building it on first use and again after the count changes.
///
/// Returns `None` if the operating system cannot start the workers, and callers then run serially.
fn pool() -> Option<Arc<ThreadPool>> {
    let threads = num_threads();
    let mut cached = POOL.lock().ok()?;
    if let Some((count, pool)) = cached.as_ref()
        && *count == threads
    {
        return Some(Arc::clone(pool));
    }

    let built = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("singe-cpu-{index}"))
            .build()
            .ok()?,
    );
    *cached = Some((threads, Arc::clone(&built)));
    drop(cached);
    Some(built)
}

/// Returns `produce(index)` for every index in `0..count`, splitting the range across threads in contiguous chunks.
///
/// Each element is computed independently, so results do not depend on the thread count.
pub(crate) fn collect<T: Send>(count: usize, produce: impl Fn(usize) -> T + Sync) -> Vec<T> {
    collect_with_grain(count, GRAIN_SIZE, produce)
}

/// Calls `work(index, chunk)` on consecutive `chunk_len`-element chunks of `data`, in parallel when `parallel` is true and there are several chunks.
///
/// # Errors
///
/// Returns the first error any chunk returns.
pub(crate) fn for_each_chunk_mut<T: Send>(
    data: &mut [T],
    chunk_len: usize,
    parallel: bool,
    work: impl Fn(usize, &mut [T]) -> Result<()> + Sync,
) -> Result<()> {
    let pool = (parallel && data.len() > chunk_len).then(pool).flatten();
    let Some(pool) = pool else {
        return data
            .chunks_mut(chunk_len.max(1))
            .enumerate()
            .try_for_each(|(index, chunk)| work(index, chunk));
    };
    let chunks = data.chunks_mut(chunk_len.max(1)).collect::<Vec<_>>();
    let mut results = chunks.iter().map(|_| Ok(())).collect::<Vec<Result<()>>>();
    let work = &work;
    pool.scope(|scope| {
        for ((index, chunk), result) in chunks.into_iter().enumerate().zip(&mut results) {
            scope.spawn(move |_| *result = work(index, chunk));
        }
    });
    results.into_iter().collect()
}

/// Returns `produce(index)` for every index in `0..count`, giving each thread at least `grain` indices.
pub(crate) fn collect_with_grain<T: Send>(
    count: usize,
    grain: usize,
    produce: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let threads = num_threads().min(count / grain.max(1)).max(1);
    let pool = (threads > 1).then(pool).flatten();
    let Some(pool) = pool else {
        return (0..count).map(produce).collect();
    };
    let chunk = count.div_ceil(threads);
    let produce = &produce;
    let mut values = Vec::with_capacity(count);
    pool.scope(|scope| {
        for (thread, slots) in values.spare_capacity_mut()[..count]
            .chunks_mut(chunk)
            .enumerate()
        {
            scope.spawn(move |_| {
                for (offset, slot) in slots.iter_mut().enumerate() {
                    slot.write(produce(thread * chunk + offset));
                }
            });
        }
    });
    // SAFETY: the scope returned only after every task wrote each slot of its chunk, and the chunks cover the first `count` slots of the reserved capacity; a panicking task propagates out of the scope before the length is set.
    unsafe { values.set_len(count) };
    values
}

/// Calls `work(row, values)` on every consecutive `row_len`-element row of `data`, giving each thread a contiguous group of rows that covers at least `grain` elements.
///
/// Each row is computed by one thread, so results do not depend on the thread count.
///
/// # Errors
///
/// Returns the first error any row returns.
pub(crate) fn for_each_row_mut<T: Send>(
    data: &mut [T],
    row_len: usize,
    grain: usize,
    work: impl Fn(usize, &mut [T]) -> Result<()> + Sync,
) -> Result<()> {
    let row_len = row_len.max(1);
    let rows = data.len().div_ceil(row_len);
    let threads = num_threads()
        .min(data.len() / grain.max(1))
        .min(rows)
        .max(1);
    let rows_per_thread = rows.div_ceil(threads).max(1);
    for_each_chunk_mut(
        data,
        rows_per_thread * row_len,
        threads > 1,
        |group, chunk| {
            chunk
                .chunks_mut(row_len)
                .enumerate()
                .try_for_each(|(offset, values)| work(group * rows_per_thread + offset, values))
        },
    )
}
