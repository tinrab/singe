//! Byte budgets that cap the emulator's host memory, and the per-entry charges used by
//! its queues and allocations.

use std::{
    collections::{VecDeque, vec_deque},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::emulator::ExecutionError;

/// Charged on top of each heap allocation (a `Box`, an `Arc`, a `Vec` buffer) for the
/// allocator header and rounding.
pub const ALLOCATION_OVERHEAD: usize = 64;

/// Charged on top of each entry of a [`ChargedQueue`] or ordered map. It covers a map
/// node's links and allocation, and a queue slot plus the spare capacity the queue
/// keeps before shrinking.
pub const NODE_OVERHEAD: usize = 128;

/// Returns the charge for one heap-allocated `T`.
pub const fn allocation<T>() -> usize {
    size_of::<T>() + ALLOCATION_OVERHEAD
}

/// Returns the charge for one `T` stored as a queue or map entry.
pub const fn node<T>() -> usize {
    size_of::<T>() + NODE_OVERHEAD
}

#[derive(Debug)]
pub struct Budget {
    limit: usize,
    used: AtomicUsize,
}

impl Budget {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            used: AtomicUsize::new(0),
        }
    }

    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Reservation, ExecutionError> {
        self.used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|total| *total <= self.limit)
            })
            .map_err(|_| ExecutionError::ExecutionStateLimit)?;
        Ok(Reservation {
            budget: Arc::clone(self),
            bytes,
        })
    }

    pub fn release(&self, bytes: usize) {
        let previous = self.used.fetch_sub(bytes, Ordering::AcqRel);
        debug_assert!(previous >= bytes);
    }

    #[cfg(test)]
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
pub struct Reservation {
    budget: Arc<Budget>,
    bytes: usize,
}

impl Reservation {
    #[cfg(test)]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Divides an existing charge without changing the aggregate budget.
    pub fn split(&mut self, bytes: usize) -> Self {
        assert!(bytes <= self.bytes);
        self.bytes -= bytes;
        Self {
            budget: Arc::clone(&self.budget),
            bytes,
        }
    }

    /// Takes responsibility for bytes already charged by a grid.
    pub fn from_charged(budget: Arc<Budget>, bytes: usize) -> Self {
        Self { budget, bytes }
    }

    /// Transfers release responsibility to the worker's grid accounting.
    pub fn commit(mut self) -> usize {
        let bytes = self.bytes;
        self.bytes = 0;
        bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.release(self.bytes);
    }
}

/// A FIFO queue whose entries are charged to a budget one by one.
///
/// A plain `VecDeque` keeps its buffer after entries leave, so memory would stay allocated
/// after their charges were released. This queue gives its buffer back once it is at most
/// a quarter full, which keeps the memory it holds proportional to its charged entries.
#[derive(Debug)]
pub struct ChargedQueue<T>(VecDeque<T>);

impl<T> Default for ChargedQueue<T> {
    fn default() -> Self {
        Self(VecDeque::new())
    }
}

impl<T> ChargedQueue<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn front(&self) -> Option<&T> {
        self.0.front()
    }

    pub fn iter(&self) -> vec_deque::Iter<'_, T> {
        self.0.iter()
    }

    pub fn push_back(&mut self, entry: T) {
        self.0.push_back(entry);
    }

    pub fn append(&mut self, other: &mut Self) {
        self.0.append(&mut other.0);
        other.shrink();
    }

    pub fn pop_front(&mut self) -> Option<T> {
        let entry = self.0.pop_front();
        self.shrink();
        entry
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        let entry = self.0.remove(index);
        self.shrink();
        entry
    }

    /// Moves every entry for which `remove` returns true into a new queue, keeping the
    /// order of both. Rotates entries within the existing buffer instead of allocating.
    pub fn extract(&mut self, mut remove: impl FnMut(&T) -> bool) -> Self {
        let mut removed = Self::new();
        for _ in 0..self.0.len() {
            let Some(entry) = self.0.pop_front() else {
                break;
            };
            if remove(&entry) {
                removed.0.push_back(entry);
            } else {
                self.0.push_back(entry);
            }
        }
        self.shrink();
        removed
    }

    fn shrink(&mut self) {
        if self.0.len() <= self.0.capacity() / 4 {
            self.0.shrink_to(self.0.len() * 2);
        }
    }
}

impl<T> FromIterator<T> for ChargedQueue<T> {
    fn from_iter<I: IntoIterator<Item = T>>(entries: I) -> Self {
        Self(entries.into_iter().collect())
    }
}

impl<'a, T> IntoIterator for &'a ChargedQueue<T> {
    type Item = &'a T;
    type IntoIter = vec_deque::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T> IntoIterator for ChargedQueue<T> {
    type Item = T;
    type IntoIter = vec_deque::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Barrier, thread};

    use super::*;

    #[test]
    fn test_charged_queue_extracts_in_order_and_releases_its_buffer() {
        let mut queue: ChargedQueue<_> = (0..64).collect();
        assert!(queue.extract(|_| false).is_empty());
        assert_eq!(queue.len(), 64);
        let even = queue.extract(|entry| entry % 2 == 0);
        assert_eq!(even.iter().copied().take(3).collect::<Vec<_>>(), [0, 2, 4]);
        assert_eq!(queue.iter().copied().take(3).collect::<Vec<_>>(), [1, 3, 5]);
        assert_eq!(queue.remove(1), Some(3));
        while queue.pop_front().is_some() {}
        assert_eq!(queue.0.capacity(), 0);
    }

    #[test]
    fn test_concurrent_reservations_share_one_limit_and_release_on_drop() {
        let budget = Arc::new(Budget::new(128));
        let barrier = Arc::new(Barrier::new(3));
        // Both threads must be running before the main thread joins the barrier.
        let mut threads = Vec::new();
        for _ in 0..2 {
            let budget = Arc::clone(&budget);
            let barrier = Arc::clone(&barrier);
            threads.push(thread::spawn(move || {
                let reservation = budget.reserve(128);
                barrier.wait();
                barrier.wait();
                reservation.is_ok()
            }));
        }
        barrier.wait();
        assert_eq!(budget.used(), 128);
        barrier.wait();
        assert_eq!(
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .filter(|success| *success)
                .count(),
            1
        );
        assert_eq!(budget.used(), 0);
    }
}
