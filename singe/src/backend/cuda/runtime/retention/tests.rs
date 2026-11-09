use std::sync::Arc;

use super::*;

#[test]
fn test_entry_byte_and_workspace_limits_apply_to_replacements_and_active_owners() {
    let limits = ProviderCacheLimits {
        entries: 2,
        accounted_bytes: 12,
        workspace_bytes: 8,
    };
    let mut cache = ProviderCache::new(limits, |_: &usize, value: &Arc<usize>| CacheWeight {
        accounted_bytes: **value,
        workspace_bytes: **value,
    });
    let retained = Arc::new(4);
    let weak = Arc::downgrade(&retained);
    cache.insert(0, Arc::clone(&retained));
    cache.insert(1, Arc::new(4));
    assert_eq!(cache.get(&0).map(|value| **value), Some(4));
    assert!(cache.get(&9).is_none());
    cache.insert(2, Arc::new(5));
    assert_eq!(cache.len(), 1);
    assert_eq!(*retained, 4);
    assert_eq!(weak.strong_count(), 1);
    cache.insert(2, Arc::new(3));
    assert_eq!(cache.stats().workspace_bytes, 3);
    cache.insert(3, Arc::new(20));
    assert_eq!(cache.stats().rejected_admissions, 1);
    cache.set_limits(ProviderCacheLimits {
        entries: 0,
        ..limits
    });
    assert!(cache.is_empty());
    assert_eq!(cache.stats().accounted_bytes, 0);
    assert_eq!(cache.stats().workspace_bytes, 0);
    assert_eq!(cache.stats().evictions, 3);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.stats().misses, 1);
    drop(retained);
    assert!(weak.upgrade().is_none());
}
