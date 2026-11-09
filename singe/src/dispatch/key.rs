use std::{cell::Cell, fmt};

/// Layer of operation dispatch that can own a kernel.
///
/// Frontend keys run first, in declaration order, and each one either handles the call or passes it on.
/// Exactly one backend key, chosen from the input device, runs last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchKey {
    /// Casts operands to a reduced-precision type inside an [`autocast`](crate::dispatch::autocast) scope.
    Autocast,
    /// Records operations into the active graph capture.
    Capture,
    /// Records backward nodes for operations on tensors that require gradients.
    Autograd,
    /// Runs kernels on the CPU.
    Cpu,
    /// Runs kernels on a CUDA device.
    Cuda,
}

impl DispatchKey {
    /// Every key, from the highest priority to the lowest.
    pub const ALL: [Self; 5] = [
        Self::Autocast,
        Self::Capture,
        Self::Autograd,
        Self::Cpu,
        Self::Cuda,
    ];

    /// Frontend keys in the order they run.
    pub const FRONTEND: [Self; 3] = [Self::Autocast, Self::Capture, Self::Autograd];

    pub(crate) const COUNT: usize = Self::ALL.len();

    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    const fn bit(self) -> u8 {
        1 << self.index()
    }

    /// Returns this key and every frontend key that runs before it.
    #[must_use]
    pub fn and_higher(self) -> DispatchKeySet {
        Self::ALL[..=self.index()]
            .iter()
            .fold(DispatchKeySet::EMPTY, |set, key| set.with(*key))
    }
}

impl fmt::Display for DispatchKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Autocast => "autocast",
            Self::Capture => "capture",
            Self::Autograd => "autograd",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        })
    }
}

/// Set of dispatch keys.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DispatchKeySet(u8);

impl DispatchKeySet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// Returns whether the set contains `key`.
    #[must_use]
    pub const fn contains(self, key: DispatchKey) -> bool {
        self.0 & key.bit() != 0
    }

    /// Returns the set with `key` added.
    #[must_use]
    pub const fn with(self, key: DispatchKey) -> Self {
        Self(self.0 | key.bit())
    }

    /// Returns the union of both sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the frontend keys.
    #[must_use]
    pub const fn frontend() -> Self {
        Self(DispatchKey::Autocast.bit() | DispatchKey::Capture.bit() | DispatchKey::Autograd.bit())
    }
}

thread_local! {
    static EXCLUDED: Cell<DispatchKeySet> = const { Cell::new(DispatchKeySet::EMPTY) };
}

/// Returns the keys excluded on this thread.
pub fn excluded() -> DispatchKeySet {
    EXCLUDED.with(Cell::get)
}

/// Excludes keys on this thread until the guard drops.
#[derive(Debug)]
#[must_use = "keys are excluded only while the guard is alive"]
pub struct ExcludeGuard {
    previous: DispatchKeySet,
}

impl ExcludeGuard {
    pub(crate) fn new(keys: DispatchKeySet) -> Self {
        let previous = EXCLUDED.with(|excluded| excluded.replace(excluded.get().union(keys)));
        Self { previous }
    }
}

impl Drop for ExcludeGuard {
    fn drop(&mut self) {
        EXCLUDED.with(|excluded| excluded.set(self.previous));
    }
}
