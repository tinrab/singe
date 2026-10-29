use std::{
    borrow::Borrow,
    collections::HashMap,
    hash::Hash,
    mem,
    num::NonZeroUsize,
    sync::{Mutex, MutexGuard},
};

use crate::{
    backend::execution::ExecutionPlan,
    error::{Error, Result},
};

/// Result of observing an autotuning winner for a native plan cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanCacheAdmission {
    /// This winner needs more observations before it is cached.
    Pending {
        /// Cumulative observations of this native engine/knob tag for the key.
        observations: usize,
        /// Observation count required for admission at the time of this result.
        required: usize,
    },
    /// This observation admitted the plan to the cache.
    Inserted,
    /// An earlier plan remains cached for this key.
    AlreadyCached,
}

#[derive(Debug)]
enum CacheEntry {
    Observing(HashMap<String, usize>),
    Cached(ExecutionPlan),
}

#[derive(Debug)]
struct CacheState<K> {
    entries: HashMap<K, CacheEntry>,
    saturation_count: NonZeroUsize,
}

impl<K: Eq + Hash> CacheState<K> {
    /// Returns rejected keys and displaced entries for destruction after unlocking.
    fn cache_plan(&mut self, key: K, plan: ExecutionPlan) -> (bool, Option<(K, CacheEntry)>) {
        match self.entries.get_mut(&key) {
            Some(CacheEntry::Cached(_)) => (false, Some((key, CacheEntry::Cached(plan)))),
            Some(entry) => {
                let previous = mem::replace(entry, CacheEntry::Cached(plan));
                (true, Some((key, previous)))
            }
            None => {
                self.entries.insert(key, CacheEntry::Cached(plan));
                (true, None)
            }
        }
    }
}

/// Thread-safe cache of built native execution plans, including C++ winner saturation.
///
/// Keys are caller-selected feature vectors or other graph identities. Scope each key to the
/// compatible device/context, operation graph, tensor UIDs, concrete layouts and compile policy.
/// A shape-independent dynamic graph key alone does not identify interchangeable native plans.
///
/// Unlike the frontend engine-choice cache, hits clone the retained native descriptor without
/// rebuilding it. Returned plans remain alive after cache removal or destruction. Bindings,
/// workspace and any other caller-owned execution resources still follow their ordinary contracts.
/// Rejected, replaced and removed keys and entries are released after unlocking the cache.
#[derive(Debug)]
pub struct ExecutionPlanCache<K = Vec<i64>> {
    name: String,
    state: Mutex<CacheState<K>>,
}

impl<K: Eq + Hash> ExecutionPlanCache<K> {
    /// Creates an empty cache that admits the first winner for each key.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            state: Mutex::new(CacheState {
                entries: HashMap::new(),
                saturation_count: NonZeroUsize::MIN,
            }),
        }
    }

    /// Returns the diagnostic name assigned at construction.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sets how often the same winner must be observed before admission.
    /// Observations need not be consecutive. Existing counts and cached plans are preserved.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache mutex is poisoned.
    pub fn set_saturation_count(&self, count: NonZeroUsize) -> Result<()> {
        self.lock()?.saturation_count = count;
        Ok(())
    }

    /// Returns the current observation threshold for admitting a winner.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache mutex is poisoned.
    pub fn saturation_count(&self) -> Result<NonZeroUsize> {
        Ok(self.lock()?.saturation_count)
    }

    /// Returns a shared native plan clone, or `None` while this key has no cached plan.
    /// Accepts borrowed key forms, such as a slice for vector keys or `str` for string keys.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache mutex is poisoned.
    pub fn get<Q: Eq + Hash + ?Sized>(&self, key: &Q) -> Result<Option<ExecutionPlan>>
    where
        K: Borrow<Q>,
    {
        Ok(match self.lock()?.entries.get(key) {
            Some(CacheEntry::Cached(plan)) => Some(plan.clone()),
            _ => None,
        })
    }

    /// Inserts a plan immediately, preserving an earlier cached plan for the same key.
    /// Returns whether this call inserted the plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache mutex is poisoned.
    pub fn insert(&self, key: K, plan: ExecutionPlan) -> Result<bool> {
        let (inserted, retired) = self.lock()?.cache_plan(key, plan);
        drop(retired);
        Ok(inserted)
    }

    /// Records one autotuning winner and caches it when its cumulative count reaches saturation.
    /// Counts are separate for each key and native engine/knob tag. Admission and insertion use
    /// one lock, so concurrent observations cannot replace the first admitted plan.
    /// An already cached key or a saturation count of one needs no native tag query.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache mutex is poisoned or the required native plan-tag query fails.
    pub fn observe_fastest(&self, key: K, plan: &ExecutionPlan) -> Result<PlanCacheAdmission> {
        {
            let mut state = self.lock()?;
            if matches!(state.entries.get(&key), Some(CacheEntry::Cached(_))) {
                return Ok(PlanCacheAdmission::AlreadyCached);
            }
            if state.saturation_count == NonZeroUsize::MIN {
                let (_, retired) = state.cache_plan(key, plan.clone());
                drop(state);
                drop(retired);
                return Ok(PlanCacheAdmission::Inserted);
            }
        }
        // Native metadata can call CUDA; leave other cache operations unblocked during the query.
        let tag = plan.tag()?;
        let mut state = self.lock()?;
        let required = state.saturation_count.get();
        let observations = match state.entries.get_mut(&key) {
            Some(CacheEntry::Cached(_)) => return Ok(PlanCacheAdmission::AlreadyCached),
            Some(CacheEntry::Observing(winners)) => {
                let count = winners.entry(tag.to_owned()).or_default();
                *count = count.saturating_add(1);
                *count
            }
            None if required > 1 => {
                state.entries.insert(
                    key,
                    CacheEntry::Observing(HashMap::from([(tag.to_owned(), 1)])),
                );
                return Ok(PlanCacheAdmission::Pending {
                    observations: 1,
                    required,
                });
            }
            None => 1,
        };
        if observations < required {
            return Ok(PlanCacheAdmission::Pending {
                observations,
                required,
            });
        }
        let (_, retired) = state.cache_plan(key, plan.clone());
        drop(state);
        drop(retired);
        Ok(PlanCacheAdmission::Inserted)
    }

    /// Removes a cached plan and any pending observations for this key.
    /// Accepts the same borrowed key forms as [`Self::get`].
    ///
    /// # Errors
    ///
    /// Returns an error if the cache mutex is poisoned.
    pub fn remove<Q: Eq + Hash + ?Sized>(&self, key: &Q) -> Result<Option<ExecutionPlan>>
    where
        K: Borrow<Q>,
    {
        let entry = self.lock()?.entries.remove_entry(key);
        Ok(match entry {
            Some((_, CacheEntry::Cached(plan))) => Some(plan),
            _ => None,
        })
    }

    /// Removes all plans and pending observations, preserving the saturation setting.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache mutex is poisoned.
    pub fn clear(&self) -> Result<()> {
        let entries = mem::take(&mut self.lock()?.entries);
        // Native plan destruction can call CUDA; release the cache lock first.
        drop(entries);
        Ok(())
    }

    /// Returns the number of admitted plans, excluding pending observations.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache mutex is poisoned.
    pub fn len(&self) -> Result<usize> {
        Ok(self
            .lock()?
            .entries
            .values()
            .filter(|entry| matches!(entry, CacheEntry::Cached(_)))
            .count())
    }

    /// Returns whether the cache contains no admitted plans, regardless of pending observations.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache mutex is poisoned.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(!self
            .lock()?
            .entries
            .values()
            .any(|entry| matches!(entry, CacheEntry::Cached(_))))
    }

    fn lock(&self) -> Result<MutexGuard<'_, CacheState<K>>> {
        self.state
            .lock()
            .map_err(|_| Error::ExecutionPlanCacheLockPoisoned)
    }
}

impl<K: Eq + Hash> Default for ExecutionPlanCache<K> {
    fn default() -> Self {
        Self::new("execution-plan-cache")
    }
}
