use std::{
    borrow::Borrow,
    cell::Cell,
    collections::{HashMap, VecDeque},
    hash::Hash,
};

/// Independent bounds for one process-wide provider cache.
/// An entry limit of zero disables retention.
/// Accounted bytes cover keys and cache-entry records, not container overhead or opaque native-library allocations.
/// Workspace bytes bound the plans' workspace requirements; the cache does not allocate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderCacheLimits {
    /// Maximum retained entry count; zero disables retention.
    pub entries: usize,
    /// Maximum sum of retained key and entry weights in bytes, excluding opaque native allocations.
    pub accounted_bytes: usize,
    /// Maximum sum of retained plans' workspace requirements in bytes; workspace itself is not allocated by the cache.
    pub workspace_bytes: usize,
}

impl Default for ProviderCacheLimits {
    fn default() -> Self {
        Self {
            entries: 64,
            accounted_bytes: 16 << 20,
            workspace_bytes: 1 << 30,
        }
    }
}

/// Snapshot of retained cache ownership and cumulative direct lookups/admissions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderCacheStats {
    /// Number of currently retained entries.
    pub entries: usize,
    /// Sum of retained entries' accounted byte weights.
    pub accounted_bytes: usize,
    /// Sum of retained entries' workspace byte requirements.
    pub workspace_bytes: usize,
    /// Saturated cumulative number of successful direct lookups, including presence checks.
    pub hits: u64,
    /// Saturated cumulative number of unsuccessful direct lookups, including presence checks.
    pub misses: u64,
    /// Saturated cumulative number of entries evicted by limits.
    pub evictions: u64,
    /// Saturated cumulative number of insertions rejected by retention limits.
    pub rejected_admissions: u64,
}

/// Accounted host-byte and workspace-requirement weights assigned to one cache entry.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheWeight {
    /// Key and entry ownership accounted in host bytes.
    pub accounted_bytes: usize,
    /// Workspace requirement in bytes, without a workspace allocation.
    pub workspace_bytes: usize,
}

#[derive(Debug)]
struct Entry<V> {
    value: V,
    weight: CacheWeight,
}

/// FIFO cache with bounded admission.
/// Consumers clone owning handles before releasing the surrounding runtime mutex, so eviction cannot invalidate them.
/// Raw mutable map access is intentionally absent: every insertion enforces limits.
#[derive(Debug)]
pub struct ProviderCache<K, V> {
    entries: HashMap<K, Entry<V>>,
    order: VecDeque<K>,
    limits: ProviderCacheLimits,
    weigh: fn(&K, &V) -> CacheWeight,
    stats: Cell<ProviderCacheStats>,
}

impl<K: Clone + Eq + Hash, V> ProviderCache<K, V> {
    pub(crate) fn new(limits: ProviderCacheLimits, weigh: fn(&K, &V) -> CacheWeight) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            limits,
            weigh,
            stats: Cell::default(),
        }
    }

    /// Returns current retention bounds.
    pub fn limits(&self) -> ProviderCacheLimits {
        self.limits
    }

    /// Applies new limits immediately, dropping only cache-owned handles.
    pub fn set_limits(&mut self, limits: ProviderCacheLimits) {
        self.limits = limits;
        while self.len() > limits.entries
            || self.stats.get().accounted_bytes > limits.accounted_bytes
            || self.stats.get().workspace_bytes > limits.workspace_bytes
        {
            self.evict_oldest();
        }

        self.entries.shrink_to_fit();
        self.order.shrink_to_fit();
    }

    /// Returns current retained weights and cumulative activity counters.
    pub fn stats(&self) -> ProviderCacheStats {
        self.stats.get()
    }

    /// Returns the current retained entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether no entries are retained.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Looks up a retained value and increments the saturated hit or miss counter.
    ///
    /// Lookup does not refresh FIFO eviction order.
    pub fn get<Q: Eq + Hash + ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        let value = self.entries.get(key).map(|entry| &entry.value);
        let mut stats = self.stats.get();
        if value.is_some() {
            stats.hits = stats.hits.saturating_add(1);
        } else {
            stats.misses = stats.misses.saturating_add(1);
        }

        self.stats.set(stats);
        value
    }

    /// Checks presence using a lookup, incrementing the saturated hit or miss counter.
    pub fn contains_key<Q: Eq + Hash + ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.get(key).is_some()
    }

    /// Iterates retained key and value pairs in unspecified map order without updating lookup counters.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(key, entry)| (key, &entry.value))
    }

    /// Iterates retained keys in unspecified map order without updating lookup counters.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.entries.keys()
    }

    /// Iterates retained values in unspecified map order without updating lookup counters.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.values().map(|entry| &entry.value)
    }

    /// Oversized values are not retained.
    /// Consumers keep their own owning clone.
    /// The return value follows map replacement semantics for the previous value.
    ///
    /// Replacement removes the previous entry before checking the new weight.
    /// If admission is rejected, the old value is returned and the key is no longer retained.
    /// Accepted insertions evict oldest entries until all independent bounds fit.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let weight = (self.weigh)(&key, &value);
        let previous = self.entries.remove(&key).map(|entry| {
            self.subtract(entry.weight);
            self.order.retain(|existing| existing != &key);
            entry.value
        });
        if self.limits.entries == 0
            || weight.accounted_bytes > self.limits.accounted_bytes
            || weight.workspace_bytes > self.limits.workspace_bytes
        {
            let mut stats = self.stats.get();
            stats.rejected_admissions = stats.rejected_admissions.saturating_add(1);
            self.stats.set(stats);
            return previous;
        }

        while self.len() >= self.limits.entries
            || self.stats.get().accounted_bytes
                > self.limits.accounted_bytes - weight.accounted_bytes
            || self.stats.get().workspace_bytes
                > self.limits.workspace_bytes - weight.workspace_bytes
        {
            self.evict_oldest();
        }

        self.order.push_back(key.clone());
        self.entries.insert(key, Entry { value, weight });
        let mut stats = self.stats.get();
        stats.entries += 1;
        stats.accounted_bytes += weight.accounted_bytes;
        stats.workspace_bytes += weight.workspace_bytes;
        self.stats.set(stats);
        previous
    }

    /// Releases cache ownership and container allocations; cumulative counters remain.
    pub fn clear(&mut self) {
        self.entries = HashMap::new();
        self.order = VecDeque::new();
        let mut stats = self.stats.get();
        stats.entries = 0;
        stats.accounted_bytes = 0;
        stats.workspace_bytes = 0;
        self.stats.set(stats);
    }

    fn subtract(&self, weight: CacheWeight) {
        let mut stats = self.stats.get();
        stats.entries -= 1;
        stats.accounted_bytes -= weight.accounted_bytes;
        stats.workspace_bytes -= weight.workspace_bytes;
        self.stats.set(stats);
    }

    fn evict_oldest(&mut self) {
        if let Some(key) = self.order.pop_front()
            && let Some(entry) = self.entries.remove(&key)
        {
            self.subtract(entry.weight);
            let mut stats = self.stats.get();
            stats.evictions = stats.evictions.saturating_add(1);
            self.stats.set(stats);
        }
    }
}

#[cfg(test)]
mod tests;
