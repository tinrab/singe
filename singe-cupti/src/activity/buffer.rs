use std::{
    alloc::{Layout, LayoutError, alloc_zeroed, dealloc, handle_alloc_error},
    ffi::c_void,
    mem::{self, offset_of, size_of},
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    ptr::{self, NonNull},
    result::Result as StdResult,
    slice,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use super::{
    collection::{disable_global, enable_global, flush_all},
    record::ActivityRecords,
};
use crate::{
    callback::Subscriber,
    error::{Error, Result},
    sys, try_ffi,
    types::{ActivityFlushFlag, ActivityKind},
};

type TimestampCallback = dyn Fn() -> u64 + Send + Sync + 'static;

static TIMESTAMP_CALLBACK: Mutex<Option<TimestampCallbackState>> = Mutex::new(None);

type ActivityBufferCompleteCallback = dyn for<'a> Fn(ActivityBuffer<'a>) + Send + Sync + 'static;

// CUPTI's buffer callbacks carry no user data or subscriber, so every registration shares this slot.
static ACTIVITY_BUFFER_CALLBACKS: Mutex<Option<ActivityBufferCallbackState>> = Mutex::new(None);
static NEXT_ACTIVITY_CALLBACK_OWNER_ID: AtomicU64 = AtomicU64::new(1);

// Set before CUPTI is asked for user-defined records and never cleared, since buffers completed later may still hold them.
static USER_DEFINED_RECORDS: AtomicBool = AtomicBool::new(false);

const ACTIVITY_BUFFER_ALIGNMENT: usize = 8;

struct ActivityBufferCallbackState {
    owner_id: u64,
    // Handle address of a subscriber-scoped registration, or `None` for the process-wide one.
    subscriber: Option<usize>,
    buffer_size: usize,
    max_num_records: usize,
    completed: Arc<ActivityBufferCompleteCallback>,
}

struct TimestampCallbackState {
    owner_id: u64,
    callback: Arc<TimestampCallback>,
}

// Records that later buffers may hold user-defined records, whose layouts the typed decoders cannot read.
pub(crate) fn note_user_defined_records() {
    USER_DEFINED_RECORDS.store(true, Ordering::SeqCst);
}

// The C allocator's `free`, which CUPTI documents for releasing the allocations it leaves to the client.
unsafe extern "C" {
    fn free(pointer: *mut c_void);
}

// Allocations CUPTI hands to the client inside records, such as device names, freed when the last copy of their buffer drops.
#[derive(Debug, Default)]
struct ClientAllocations(Vec<NonNull<c_void>>);

// SAFETY: the pointers are `malloc` allocations that only this value frees, and C `free` may run on any thread.
unsafe impl Send for ClientAllocations {}

// SAFETY: shared references expose no pointer, so they cannot race.
unsafe impl Sync for ClientAllocations {}

impl ClientAllocations {
    fn collect(records: ActivityRecords<'_>) -> Self {
        let mut pointers = Vec::new();
        for record in records {
            let Ok(record) = record else {
                break;
            };
            record.client_allocations(&mut pointers);
        }

        // A pointer listed twice would be freed twice.
        pointers.sort_unstable();
        pointers.dedup();
        Self(pointers.into_iter().filter_map(NonNull::new).collect())
    }
}

impl Drop for ClientAllocations {
    fn drop(&mut self) {
        for pointer in self.0.drain(..) {
            // SAFETY: CUPTI allocated each pointer with `malloc` for the client to free, and `collect` took each from one record field of one completed buffer.
            unsafe { free(pointer.as_ptr()) };
        }
    }
}

/// Borrowed bytes from a completed CUPTI activity buffer.
///
/// Record views and borrowed payload fields cannot outlive the source buffer.
/// Allocations that CUPTI leaves to the client, such as device names, `NVLink` port tables, and overhead data, are freed after the completion callback returns, unless an [`ActivityBufferOwned`] copy keeps them.
/// Once user-defined records have been enabled, records decode to no payload and these allocations are not freed.
#[derive(Debug, Clone, Copy)]
pub struct ActivityBuffer<'a> {
    bytes: &'a [u8],
    allocations: &'a Arc<ClientAllocations>,
    full_layout: bool,
}

impl<'a> ActivityBuffer<'a> {
    /// Returns the valid bytes in the completed activity buffer.
    #[must_use]
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Returns the number of valid activity-buffer bytes.
    #[must_use]
    pub const fn len(self) -> usize {
        self.bytes.len()
    }

    /// Returns whether the activity buffer contains no valid bytes.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bytes.is_empty()
    }

    /// Returns an iterator over records in this completed activity buffer.
    ///
    /// Records and their borrowed payload fields retain the buffer lifetime.
    /// Native parsing errors are yielded by the iterator.
    #[must_use]
    pub fn records(self) -> ActivityRecords<'a> {
        ActivityRecords::new(self.bytes, self.full_layout)
    }

    fn to_owned_buffer(self) -> ActivityBufferOwned {
        ActivityBufferOwned {
            bytes: self.bytes.to_vec(),
            _allocations: Arc::clone(self.allocations),
            full_layout: self.full_layout,
        }
    }
}

/// Owns a byte copy of a completed CUPTI activity buffer.
///
/// Decoded records borrow this owner.
/// The owner keeps alive the allocations CUPTI leaves to the client, such as device names, and frees them when its last clone drops.
/// Other native pointer-backed strings and binary images are not deep-copied with the record bytes.
#[derive(Debug, Clone)]
pub struct ActivityBufferOwned {
    bytes: Vec<u8>,
    _allocations: Arc<ClientAllocations>,
    full_layout: bool,
}

impl ActivityBufferOwned {
    /// Returns the raw CUPTI activity bytes captured from a completed buffer.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Transfers ownership of the copied activity-record bytes.
    ///
    /// Native pointer-backed data referenced by records is not deep-copied, and the allocations this owner kept alive are freed.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Returns the number of valid activity-buffer bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns whether the activity buffer contains no valid bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Returns an iterator over records in this completed activity buffer.
    ///
    /// Records and their borrowed payload fields retain the buffer lifetime.
    /// Native parsing errors are yielded by the iterator.
    #[must_use]
    pub fn records(&self) -> ActivityRecords<'_> {
        ActivityRecords::new(&self.bytes, self.full_layout)
    }
}

/// Guard for the current Rust activity-buffer callbacks.
///
/// Dropping the guard clears its callbacks if this registration still owns them.
/// Dropping a replaced registration leaves the newer callbacks installed.
#[derive(Debug)]
pub struct ActivityCallbackRegistration {
    owner_id: u64,
}

impl Drop for ActivityCallbackRegistration {
    fn drop(&mut self) {
        clear_activity_callbacks_if_owner(self.owner_id);
    }
}

/// Guard for the current Rust timestamp callback.
///
/// Dropping the guard removes its callback only if this registration still owns it.
#[derive(Debug)]
pub struct TimestampCallbackRegistration {
    owner_id: u64,
}

impl Drop for TimestampCallbackRegistration {
    fn drop(&mut self) {
        let Ok(mut callback) = TIMESTAMP_CALLBACK.lock() else {
            return;
        };
        if callback
            .as_ref()
            .is_some_and(|callback| callback.owner_id == self.owner_id)
        {
            *callback = None;
        }
    }
}

/// Activity-buffer capacity and record limit for completion callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivityBufferOptions {
    /// Size of each buffer handed to CUPTI, in bytes.
    ///
    /// When CUPTI requests a buffer, a zero size or an invalid allocation layout supplies no buffer.
    pub buffer_size_bytes: usize,
    /// Maximum records CUPTI may write per buffer; zero means no record-count limit.
    pub max_num_records: usize,
}

impl Default for ActivityBufferOptions {
    /// Returns 8 MiB buffers with no record-count limit.
    fn default() -> Self {
        Self {
            buffer_size_bytes: 8 << 20,
            max_num_records: 0,
        }
    }
}

/// Collects byte copies of completed activity buffers behind a shared mutex.
///
/// Buffers accumulate until [`Self::take_buffers`] removes them.
/// Dropping the collector clears its callbacks if it still owns the current registration.
#[derive(Debug)]
pub struct ActivityCollector {
    owner_id: u64,
    buffers: Arc<Mutex<Vec<ActivityBufferOwned>>>,
}

impl ActivityCollector {
    /// Registers CUPTI activity callbacks and stores completed buffers in memory.
    ///
    /// CUPTI supports one process-wide activity buffer callback.
    /// Creating a new collector replaces the previous process-wide callback owner.
    ///
    /// # Errors
    ///
    /// Returns an error if a subscriber-scoped registration is live, the activity callback-state mutex is poisoned, or native callback registration fails.
    pub fn create(options: &ActivityBufferOptions) -> Result<Self> {
        let buffers = Arc::new(Mutex::new(Vec::new()));
        let callback_buffers = Arc::clone(&buffers);
        let owner_id = register_global_callbacks(options, move |buffer| {
            if let Ok(mut callback_buffers) = callback_buffers.lock() {
                callback_buffers.push(buffer.to_owned_buffer());
            }
        })?;

        Ok(Self { owner_id, buffers })
    }

    /// Globally enables collection of `kind`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI is uninitialized, the activity kind is invalid or unsupported, or collection conflicts with the current profiling configuration.
    pub fn enable(&self, kind: ActivityKind) -> Result<()> {
        enable_global(kind)
    }

    /// Globally disables collection of `kind`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI is uninitialized or rejects the activity kind.
    pub fn disable(&self, kind: ActivityKind) -> Result<()> {
        disable_global(kind)
    }

    /// Flushes activity buffers using the requested completion policy.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot flush the process-wide activity buffers.
    pub fn flush_all(&self, flag: ActivityFlushFlag) -> Result<()> {
        flush_all(flag)
    }

    /// Returns the number of completed buffers currently stored by the collector.
    ///
    /// # Errors
    ///
    /// Returns an error if the collector-buffer mutex is poisoned.
    pub fn len(&self) -> Result<usize> {
        Ok(self
            .buffers
            .lock()
            .map_err(|_| Error::LockPoisoned {
                name: "activity collector buffers".into(),
            })?
            .len())
    }

    /// Returns whether the collector currently stores no completed buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if the collector-buffer mutex is poisoned.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Removes all completed buffers from this collector and returns them.
    ///
    /// # Errors
    ///
    /// Returns an error if the collector-buffer mutex is poisoned.
    pub fn take_buffers(&self) -> Result<Vec<ActivityBufferOwned>> {
        Ok(mem::take(&mut *self.buffers.lock().map_err(|_| {
            Error::LockPoisoned {
                name: "activity collector buffers".into(),
            }
        })?))
    }
}

impl Drop for ActivityCollector {
    fn drop(&mut self) {
        clear_activity_callbacks_if_owner(self.owner_id);
    }
}

/// Registers a process-wide Rust callback receiving completed activity buffers.
///
/// Replaces the current process-wide Rust callback owner.
/// Keep the returned registration alive while collecting; dropping it clears its callbacks if it still owns the registration.
/// Buffer bytes are borrowed only during `completed`, which CUPTI may call from any thread, concurrently with other completions.
/// Callback panics are caught by the native trampoline.
///
/// # Errors
///
/// Returns an error if a subscriber-scoped registration is live, the activity callback-state mutex is poisoned, or CUPTI cannot register the native callbacks.
pub fn global_register_callbacks<F>(
    completed: F,
    options: &ActivityBufferOptions,
) -> Result<ActivityCallbackRegistration>
where
    F: for<'a> Fn(ActivityBuffer<'a>) + Send + Sync + 'static,
{
    let owner_id = register_global_callbacks(options, completed)?;
    Ok(ActivityCallbackRegistration { owner_id })
}

fn register_global_callbacks<F>(options: &ActivityBufferOptions, completed: F) -> Result<u64>
where
    F: for<'a> Fn(ActivityBuffer<'a>) + Send + Sync + 'static,
{
    let owner_id = install_callbacks(None, options, Arc::new(completed))?;
    // SAFETY: both trampolines are `extern "C"` functions with the signatures CUPTI expects, and they catch panics before returning.
    let result = unsafe {
        try_ffi!(sys::cuptiActivityRegisterCallbacks(
            Some(activity_buffer_requested_trampoline),
            Some(activity_buffer_completed_trampoline),
        ))
    };
    if let Err(error) = result {
        clear_activity_callbacks_if_owner(owner_id);
        return Err(error);
    }

    Ok(owner_id)
}

/// Registers subscriber-scoped activity buffer callbacks.
///
/// Register callbacks before enabling activity kinds if you want CUPTI to deliver activity records.
/// Callback-based collection cannot be mixed with CUPTI's blocking enqueue/dequeue activity APIs.
/// CUPTI's buffer callbacks do not identify their subscriber, so one registration serves the process at a time: registering again for the same subscriber replaces it, and registering for another subscriber fails while it is live.
/// `completed` may run on any thread, concurrently with other completions.
///
/// # Errors
///
/// - Returns [`Error::ActivityCallbacksInUse`] if a registration for the process or another subscriber is live.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects one of the activity buffer callbacks.
pub fn register_callbacks<F>(
    subscriber: &Subscriber,
    completed: F,
    options: &ActivityBufferOptions,
) -> Result<ActivityCallbackRegistration>
where
    F: for<'a> Fn(ActivityBuffer<'a>) + Send + Sync + 'static,
{
    let subscriber = subscriber.as_raw()?;
    let owner_id = install_callbacks(Some(subscriber.addr()), options, Arc::new(completed))?;
    // SAFETY: the borrowed subscriber keeps `subscriber` live, and both trampolines have the signatures CUPTI expects and catch panics.
    let result = unsafe {
        try_ffi!(sys::cuptiActivityRegisterCallbacks_v2(
            subscriber,
            Some(activity_buffer_requested_trampoline_v2),
            Some(activity_buffer_completed_trampoline_v2),
        ))
    };
    if let Err(error) = result {
        clear_activity_callbacks_if_owner(owner_id);
        return Err(error);
    }

    Ok(ActivityCallbackRegistration { owner_id })
}

// Installs the shared callback state, replacing a registration of the same scope.
fn install_callbacks(
    subscriber: Option<usize>,
    options: &ActivityBufferOptions,
    completed: Arc<ActivityBufferCompleteCallback>,
) -> Result<u64> {
    let mut callbacks = ACTIVITY_BUFFER_CALLBACKS
        .lock()
        .map_err(|_| Error::LockPoisoned {
            name: "activity buffer callbacks".into(),
        })?;
    if callbacks
        .as_ref()
        .is_some_and(|current| current.subscriber != subscriber)
    {
        return Err(Error::ActivityCallbacksInUse);
    }

    let owner_id = NEXT_ACTIVITY_CALLBACK_OWNER_ID.fetch_add(1, Ordering::Relaxed);
    *callbacks = Some(ActivityBufferCallbackState {
        owner_id,
        subscriber,
        buffer_size: options.buffer_size_bytes,
        max_num_records: options.max_num_records,
        completed,
    });
    drop(callbacks);
    Ok(owner_id)
}

fn clear_activity_callbacks_if_owner(owner_id: u64) {
    let Ok(mut callbacks) = ACTIVITY_BUFFER_CALLBACKS.lock() else {
        return;
    };
    if callbacks
        .as_ref()
        .is_some_and(|callbacks| callbacks.owner_id == owner_id)
    {
        *callbacks = None;
    }
}

unsafe extern "C" fn activity_buffer_requested_trampoline(
    buffer: *mut *mut u8,
    size: *mut u64,
    max_num_records: *mut u64,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        request_activity_buffer(buffer, size, max_num_records);
    }));
}

unsafe extern "C" fn activity_buffer_completed_trampoline(
    _context: sys::CUcontext,
    _stream_id: u32,
    buffer: *mut u8,
    size: u64,
    valid_size: u64,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        complete_activity_buffer(buffer, size, valid_size, false);
    }));
}

unsafe extern "C" fn activity_buffer_requested_trampoline_v2(
    buffer: *mut *mut u8,
    size: *mut u64,
    max_num_records: *mut u64,
    _request_info: *mut sys::CUpti_BufferCallbackRequestInfo,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        request_activity_buffer(buffer, size, max_num_records);
    }));
}

unsafe extern "C" fn activity_buffer_completed_trampoline_v2(
    buffer: *mut u8,
    size: u64,
    valid_size: u64,
    complete_info: *mut sys::CUpti_BufferCallbackCompleteInfo,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        complete_activity_buffer(buffer, size, valid_size, has_record_layouts(complete_info));
    }));
}

// Whether CUPTI reports user-defined record layouts in the completed buffer.
fn has_record_layouts(info: *const sys::CUpti_BufferCallbackCompleteInfo) -> bool {
    if info.is_null() {
        return false;
    }

    let required = offset_of!(sys::CUpti_BufferCallbackCompleteInfo, numRecordLayouts)
        + size_of::<sys::size_t>();
    // SAFETY: CUPTI passes a readable completion record whose `structSize` states which fields it holds, and the field is read only when that size covers it.
    unsafe { (*info).structSize >= required as u64 && (*info).numRecordLayouts != 0 }
}

fn request_activity_buffer(buffer: *mut *mut u8, size: *mut u64, max_num_records: *mut u64) {
    if buffer.is_null() || size.is_null() || max_num_records.is_null() {
        return;
    }

    let Some((buffer_size, max_records)) = ({
        let Ok(callbacks) = ACTIVITY_BUFFER_CALLBACKS.lock() else {
            return;
        };
        callbacks
            .as_ref()
            .map(|callbacks| (callbacks.buffer_size, callbacks.max_num_records))
    }) else {
        // SAFETY: the three output pointers were checked non-null on entry, and CUPTI passes them writable to the buffer-request callback.
        unsafe {
            *buffer = ptr::null_mut();
            *size = 0;
            *max_num_records = 0;
        }
        return;
    };

    if buffer_size == 0 {
        // SAFETY: the three output pointers were checked non-null on entry, and CUPTI passes them writable to the buffer-request callback.
        unsafe {
            *buffer = ptr::null_mut();
            *size = 0;
            *max_num_records = 0;
        }
        return;
    }

    let Ok(layout) = activity_buffer_layout(buffer_size) else {
        // SAFETY: the three output pointers were checked non-null on entry, and CUPTI passes them writable to the buffer-request callback.
        unsafe {
            *buffer = ptr::null_mut();
            *size = 0;
            *max_num_records = 0;
        }
        return;
    };
    // SAFETY: `activity_buffer_layout` rejects zero sizes, so `layout` is non-zero-sized.
    let allocation_ptr = unsafe { alloc_zeroed(layout) };
    if allocation_ptr.is_null() {
        handle_alloc_error(layout);
    }

    // SAFETY: the three output pointers were checked non-null on entry, and CUPTI passes them writable to the buffer-request callback.
    unsafe {
        *buffer = allocation_ptr;
        *size = buffer_size as u64;
        *max_num_records = max_records as u64;
    }
}

fn complete_activity_buffer(
    buffer: *mut u8,
    size: u64,
    valid_size: u64,
    user_defined_layouts: bool,
) {
    if buffer.is_null() {
        return;
    }

    // Reclaim the buffer allocation after the callback has copied or consumed
    // the valid bytes reported by CUPTI.
    let allocation = ActivityBufferAllocation { buffer, size };
    let Ok(size) = usize::try_from(size) else {
        return;
    };
    let Ok(valid_size) = usize::try_from(valid_size) else {
        return;
    };
    if valid_size > size {
        return;
    }

    // SAFETY: CUPTI returns the `size`-byte buffer allocated by the request callback, and `valid_size <= size`.
    let bytes = unsafe { slice::from_raw_parts(buffer.cast_const(), valid_size) };
    // The flag is set before CUPTI is asked for user-defined records, so a buffer that may hold them reads it as set.
    let full_layout = !user_defined_layouts && !USER_DEFINED_RECORDS.load(Ordering::SeqCst);
    // CUPTI hands these allocations to the client once per buffer, so they are freed even without a registered callback.
    let allocations = Arc::new(if full_layout {
        ClientAllocations::collect(ActivityRecords::new(bytes, true))
    } else {
        ClientAllocations::default()
    });

    // The callback runs without the lock, so concurrent completions and registration changes do not wait for it.
    let completed = ACTIVITY_BUFFER_CALLBACKS.lock().ok().and_then(|callbacks| {
        callbacks
            .as_ref()
            .map(|callbacks| Arc::clone(&callbacks.completed))
    });
    let result = completed.map(|completed| {
        catch_unwind(AssertUnwindSafe(|| {
            completed(ActivityBuffer {
                bytes,
                allocations: &allocations,
                full_layout,
            });
        }))
    });

    drop(allocations);
    drop(allocation);

    if let Some(Err(payload)) = result {
        resume_unwind(payload);
    }
}

struct ActivityBufferAllocation {
    buffer: *mut u8,
    size: u64,
}

impl Drop for ActivityBufferAllocation {
    fn drop(&mut self) {
        if self.buffer.is_null() {
            return;
        }

        let Ok(size) = usize::try_from(self.size) else {
            return;
        };
        if size == 0 {
            return;
        }

        let Ok(layout) = activity_buffer_layout(size) else {
            return;
        };
        // SAFETY: `self.buffer` came from `alloc_zeroed` with the layout rebuilt here from the same size.
        unsafe {
            dealloc(self.buffer, layout);
        }
    }
}

fn activity_buffer_layout(size: usize) -> StdResult<Layout, LayoutError> {
    Layout::from_size_align(size, ACTIVITY_BUFFER_ALIGNMENT)
}

/// Registers a process-wide timestamp callback for activity records.
///
/// Register the callback before enabling activity kinds.
/// Changing it during a profiling session can make earlier and later records use different timestamp sources.
/// CUPTI may call `callback` from any thread, concurrently.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the timestamp callback.
pub fn register_timestamp_callback<F>(callback: F) -> Result<TimestampCallbackRegistration>
where
    F: Fn() -> u64 + Send + Sync + 'static,
{
    let owner_id = NEXT_ACTIVITY_CALLBACK_OWNER_ID.fetch_add(1, Ordering::Relaxed);
    *TIMESTAMP_CALLBACK.lock().map_err(|_| Error::LockPoisoned {
        name: "timestamp callback".into(),
    })? = Some(TimestampCallbackState {
        owner_id,
        callback: Arc::new(callback),
    });
    // SAFETY: the trampoline is an `extern "C"` function with the signature CUPTI expects, and it catches panics before returning.
    let result = unsafe {
        try_ffi!(sys::cuptiActivityRegisterTimestampCallback(Some(
            timestamp_callback_trampoline
        )))
    };
    if let Err(error) = result {
        let mut callback = TIMESTAMP_CALLBACK.lock().map_err(|_| Error::LockPoisoned {
            name: "timestamp callback".into(),
        })?;
        if callback
            .as_ref()
            .is_some_and(|callback| callback.owner_id == owner_id)
        {
            *callback = None;
        }

        drop(callback);
        return Err(error);
    }
    Ok(TimestampCallbackRegistration { owner_id })
}

unsafe extern "C" fn timestamp_callback_trampoline() -> u64 {
    catch_unwind(AssertUnwindSafe(|| {
        // The callback runs without the lock, so concurrent timestamp requests do not serialize on it.
        let callback = TIMESTAMP_CALLBACK
            .lock()
            .ok()
            .and_then(|callback| callback.as_ref().map(|state| Arc::clone(&state.callback)));
        callback.map_or(0, |callback| callback())
    }))
    .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;
    #[cfg(feature = "testing")]
    use crate::testing::setup_context;

    static ACTIVITY_BUFFER_CALLBACK_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct ActivityBufferCallbackReset;

    impl Drop for ActivityBufferCallbackReset {
        fn drop(&mut self) {
            *ACTIVITY_BUFFER_CALLBACKS
                .lock()
                .expect("activity buffer callbacks poisoned") = None;
        }
    }

    #[test]
    fn test_activity_buffer_request_returns_eight_byte_aligned_buffer() {
        let _lock = ACTIVITY_BUFFER_CALLBACK_TEST_LOCK
            .lock()
            .expect("activity buffer callback test lock poisoned");
        let _reset = ActivityBufferCallbackReset;
        *ACTIVITY_BUFFER_CALLBACKS
            .lock()
            .expect("activity buffer callbacks poisoned") = Some(ActivityBufferCallbackState {
            owner_id: 0,
            subscriber: None,
            buffer_size: 17,
            max_num_records: 3,
            completed: Arc::new(|_| {}),
        });

        let mut buffer = ptr::null_mut();
        let mut size = 0;
        let mut max_num_records = 0;

        request_activity_buffer(&raw mut buffer, &raw mut size, &raw mut max_num_records);

        assert!(!buffer.is_null());
        assert_eq!(buffer.addr() % ACTIVITY_BUFFER_ALIGNMENT, 0);
        assert_eq!(size, 17);
        assert_eq!(max_num_records, 3);

        drop(ActivityBufferAllocation { buffer, size });
    }

    #[test]
    fn test_activity_buffer_rejects_valid_length_larger_than_allocation() {
        let _lock = ACTIVITY_BUFFER_CALLBACK_TEST_LOCK
            .lock()
            .expect("activity buffer callback test lock poisoned");
        let _reset = ActivityBufferCallbackReset;
        *ACTIVITY_BUFFER_CALLBACKS
            .lock()
            .expect("activity buffer callbacks poisoned") = Some(ActivityBufferCallbackState {
            owner_id: 0,
            subscriber: None,
            buffer_size: 8,
            max_num_records: 0,
            completed: Arc::new(|_| panic!("invalid activity buffer reached user callback")),
        });
        let mut buffer = ptr::null_mut();
        let mut size = 0;
        let mut max_num_records = 0;
        request_activity_buffer(&raw mut buffer, &raw mut size, &raw mut max_num_records);
        assert!(!buffer.is_null());
        complete_activity_buffer(buffer, size, size + 1, false);
        assert!(ACTIVITY_BUFFER_CALLBACKS.lock().unwrap().is_some());
    }

    #[test]
    fn test_activity_buffer_complete_keeps_state_and_frees_after_callback_panic() {
        static CALLED: AtomicBool = AtomicBool::new(false);
        let _lock = ACTIVITY_BUFFER_CALLBACK_TEST_LOCK
            .lock()
            .expect("activity buffer callback test lock poisoned");
        let _reset = ActivityBufferCallbackReset;
        CALLED.store(false, Ordering::SeqCst);
        *ACTIVITY_BUFFER_CALLBACKS
            .lock()
            .expect("activity buffer callbacks poisoned") = Some(ActivityBufferCallbackState {
            owner_id: 0,
            subscriber: None,
            buffer_size: 32,
            max_num_records: 0,
            completed: Arc::new(|buffer| {
                assert_eq!(buffer.len(), 5);
                CALLED.store(true, Ordering::SeqCst);
                panic!("activity callback panic");
            }),
        });

        let mut buffer = ptr::null_mut();
        let mut size = 0;
        let mut max_num_records = 0;
        request_activity_buffer(&raw mut buffer, &raw mut size, &raw mut max_num_records);

        let result = catch_unwind(AssertUnwindSafe(|| {
            complete_activity_buffer(buffer, size, 5, false);
        }));

        assert!(result.is_err());
        assert!(CALLED.load(Ordering::SeqCst));
        assert!(
            ACTIVITY_BUFFER_CALLBACKS
                .lock()
                .expect("activity buffer callbacks poisoned")
                .is_some()
        );
    }

    #[cfg(feature = "testing")]
    #[test]
    fn test_activity_collector_copies_completed_buffers() -> Result<()> {
        let _test = setup_context()?;
        let _lock = ACTIVITY_BUFFER_CALLBACK_TEST_LOCK
            .lock()
            .expect("activity buffer callback test lock poisoned");
        let _reset = ActivityBufferCallbackReset;
        let collector = ActivityCollector::create(&ActivityBufferOptions {
            buffer_size_bytes: 32,
            max_num_records: 0,
        })?;

        let mut buffer = ptr::null_mut();
        let mut size = 0;
        let mut max_num_records = 0;
        request_activity_buffer(&raw mut buffer, &raw mut size, &raw mut max_num_records);
        assert!(!buffer.is_null());
        assert_eq!(size, 32);

        // SAFETY: the request callback allocated `size` writable bytes at `buffer`, checked non-null above.
        unsafe {
            let bytes = slice::from_raw_parts_mut(buffer, usize::try_from(size).unwrap());
            bytes[..4].copy_from_slice(&[1, 2, 3, 4]);
        }

        complete_activity_buffer(buffer, size, 4, false);

        assert_eq!(collector.len()?, 1);
        let buffers = collector.take_buffers()?;
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].bytes(), &[1, 2, 3, 4]);
        assert!(collector.is_empty()?);

        Ok(())
    }

    #[cfg(feature = "testing")]
    #[test]
    fn test_activity_collector_drop_only_clears_owned_registration() -> Result<()> {
        let _test = setup_context()?;
        let _lock = ACTIVITY_BUFFER_CALLBACK_TEST_LOCK
            .lock()
            .expect("activity buffer callback test lock poisoned");
        let _reset = ActivityBufferCallbackReset;
        let collector = ActivityCollector::create(&ActivityBufferOptions {
            buffer_size_bytes: 32,
            max_num_records: 0,
        })?;
        let _registration = global_register_callbacks(
            |_| {},
            &ActivityBufferOptions {
                buffer_size_bytes: 64,
                max_num_records: 0,
            },
        )?;

        drop(collector);

        let callbacks = ACTIVITY_BUFFER_CALLBACKS
            .lock()
            .expect("activity buffer callbacks poisoned");
        assert!(callbacks.is_some());
        assert_eq!(callbacks.as_ref().unwrap().buffer_size, 64);

        Ok(())
    }
}
