//! Shared machinery for the process-wide native loggers of the wrapper crates.
//!
//! A native logger calls a plain C function with no user data, so each crate keeps a [`CallbackSlot`] in a `static` and passes an `extern "C"` trampoline that calls [`CallbackSlot::dispatch`].

use std::{
    borrow::Cow,
    ffi::{CStr, c_char},
    fmt,
    marker::PhantomData,
    mem,
    panic::{self, AssertUnwindSafe},
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
};

/// A message from a native logger, borrowed for the duration of the callback.
///
/// `L` is the crate's typed log level, and `D` holds any further fields the library passes.
pub struct LogRecord<'a, L, D = ()> {
    raw_level: i32,
    function: Cow<'a, str>,
    message: Cow<'a, str>,
    details: D,
    level: PhantomData<fn() -> L>,
}

impl<'a, L, D> LogRecord<'a, L, D> {
    /// Creates a record from its parts.
    #[must_use]
    pub const fn new(
        raw_level: i32,
        function: Cow<'a, str>,
        message: Cow<'a, str>,
        details: D,
    ) -> Self {
        Self {
            raw_level,
            function,
            message,
            details,
            level: PhantomData,
        }
    }

    /// Returns the level as the library passed it.
    #[must_use]
    pub const fn raw_level(&self) -> i32 {
        self.raw_level
    }

    /// Returns the name of the library function that logged the message, or an empty string when the library passes none.
    #[must_use]
    pub fn function(&self) -> &str {
        &self.function
    }

    /// Returns the message text.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the further fields the library passed with the message.
    #[must_use]
    pub const fn details(&self) -> &D {
        &self.details
    }

    /// Copies the record into one that owns its strings.
    #[must_use]
    pub fn into_owned(self) -> OwnedLogRecord<L, D> {
        OwnedLogRecord {
            raw_level: self.raw_level,
            function: self.function.into_owned(),
            message: self.message.into_owned(),
            details: self.details,
            level: PhantomData,
        }
    }
}

impl<L: TryFrom<i32>, D> LogRecord<'_, L, D> {
    /// Returns the typed level, or `None` for a level this crate does not name.
    #[must_use]
    pub fn level(&self) -> Option<L> {
        L::try_from(self.raw_level).ok()
    }
}

impl<L, D: fmt::Debug> fmt::Debug for LogRecord<'_, L, D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LogRecord")
            .field("raw_level", &self.raw_level)
            .field("function", &self.function)
            .field("message", &self.message)
            .field("details", &self.details)
            .finish()
    }
}

/// A [`LogRecord`] that owns its strings, so it can outlive the callback.
pub struct OwnedLogRecord<L, D = ()> {
    raw_level: i32,
    function: String,
    message: String,
    details: D,
    level: PhantomData<fn() -> L>,
}

impl<L, D> OwnedLogRecord<L, D> {
    /// Returns the level as the library passed it.
    #[must_use]
    pub const fn raw_level(&self) -> i32 {
        self.raw_level
    }

    /// Returns the name of the library function that logged the message, or an empty string when the library passes none.
    #[must_use]
    pub fn function(&self) -> &str {
        &self.function
    }

    /// Returns the message text.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the further fields the library passed with the message.
    #[must_use]
    pub const fn details(&self) -> &D {
        &self.details
    }
}

impl<L: TryFrom<i32>, D> OwnedLogRecord<L, D> {
    /// Returns the typed level, or `None` for a level this crate does not name.
    #[must_use]
    pub fn level(&self) -> Option<L> {
        L::try_from(self.raw_level).ok()
    }
}

impl<L, D: Clone> Clone for OwnedLogRecord<L, D> {
    fn clone(&self) -> Self {
        Self {
            raw_level: self.raw_level,
            function: self.function.clone(),
            message: self.message.clone(),
            details: self.details.clone(),
            level: PhantomData,
        }
    }
}

impl<L, D: fmt::Debug> fmt::Debug for OwnedLogRecord<L, D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedLogRecord")
            .field("raw_level", &self.raw_level)
            .field("function", &self.function)
            .field("message", &self.message)
            .field("details", &self.details)
            .finish()
    }
}

type Callback<L, D> = Arc<dyn for<'a> Fn(LogRecord<'a, L, D>) + Send + Sync>;

/// The callback a crate's native logger trampoline calls, kept in a `static`.
///
/// Each installed callback has a generation, so releasing an old one leaves a newer one in place.
pub struct CallbackSlot<L: 'static, D: 'static = ()> {
    library: &'static str,
    install: Mutex<()>,
    current: Mutex<Option<(u64, Callback<L, D>)>>,
    next: AtomicU64,
}

impl<L: 'static, D: 'static> CallbackSlot<L, D> {
    /// Creates an empty slot for `library`, the name a panicking callback is reported under.
    #[must_use]
    pub const fn new(library: &'static str) -> Self {
        Self {
            library,
            install: Mutex::new(()),
            current: Mutex::new(None),
            next: AtomicU64::new(1),
        }
    }

    /// Stores `callback`, then runs `native_set`, which points the library at the crate's trampoline, and returns the callback's generation.
    ///
    /// Installs and releases are serialized with their native calls.
    ///
    /// # Errors
    ///
    /// Returns `native_set`'s error, after restoring the previous callback.
    pub fn install<E>(
        &self,
        callback: impl for<'a> Fn(LogRecord<'a, L, D>) + Send + Sync + 'static,
        native_set: impl FnOnce() -> Result<(), E>,
    ) -> Result<u64, E> {
        let _install = lock(&self.install);
        let generation = self.next.fetch_add(1, Ordering::Relaxed);
        let previous = lock(&self.current).replace((generation, Arc::new(callback)));
        if let Err(error) = native_set() {
            let installed = mem::replace(&mut *lock(&self.current), previous);
            drop(installed);
            return Err(error);
        }

        drop(previous);
        Ok(generation)
    }

    /// Removes the callback of `generation` and runs `native_clear`, which detaches the library's callback, unless a newer callback replaced it.
    ///
    /// # Errors
    ///
    /// Returns `native_clear`'s error; the callback is removed either way.
    pub fn release<E>(
        &self,
        generation: u64,
        native_clear: impl FnOnce() -> Result<(), E>,
    ) -> Result<(), E> {
        let _install = lock(&self.install);
        let removed = {
            let mut current = lock(&self.current);
            if current
                .as_ref()
                .is_none_or(|(installed, _)| *installed != generation)
            {
                return Ok(());
            }

            current.take()
        };
        let result = native_clear();
        drop(removed);
        result
    }

    /// Calls the installed callback with a record of the native arguments and `details`.
    ///
    /// A panicking callback loses this record and is reported in debug builds, since unwinding must not reach the native caller.
    ///
    /// # Safety
    ///
    /// `function` and `message` must each be null or a NUL-terminated string valid for this call.
    pub unsafe fn dispatch_with(
        &self,
        raw_level: i32,
        function: *const c_char,
        message: *const c_char,
        details: D,
    ) {
        let callback = {
            let current = lock(&self.current);
            current.as_ref().map(|(_, callback)| Arc::clone(callback))
        };
        let Some(callback) = callback else {
            return;
        };

        // SAFETY: the caller passes null or NUL-terminated strings valid for this call.
        let record = unsafe { LogRecord::new(raw_level, lossy(function), lossy(message), details) };
        if panic::catch_unwind(AssertUnwindSafe(|| callback(record))).is_err()
            && cfg!(debug_assertions)
        {
            eprintln!("{} logger callback panicked", self.library);
        }
    }
}

impl<L: 'static> CallbackSlot<L> {
    /// Calls the installed callback with a record of the native arguments.
    ///
    /// # Safety
    ///
    /// `function` and `message` must each be null or a NUL-terminated string valid for this call.
    pub unsafe fn dispatch(&self, raw_level: i32, function: *const c_char, message: *const c_char) {
        // SAFETY: the caller guarantees `function` and `message` are null or NUL-terminated strings valid for this call, which is all `dispatch_with` requires.
        unsafe { self.dispatch_with(raw_level, function, message, ()) };
    }
}

impl<L, D> fmt::Debug for CallbackSlot<L, D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CallbackSlot")
            .field("library", &self.library)
            .finish_non_exhaustive()
    }
}

// The slot's state stays consistent across a panic, since every update is a single assignment.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// The caller passes null or a NUL-terminated string valid for `'a`.
unsafe fn lossy<'a>(ptr: *const c_char) -> Cow<'a, str> {
    if ptr.is_null() {
        Cow::Borrowed("")
    } else {
        // SAFETY: the caller passes a NUL-terminated string valid for `'a`.
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy()
    }
}
