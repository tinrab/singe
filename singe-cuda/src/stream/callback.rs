use std::{ffi, panic};

use singe_cuda_sys::runtime;

use super::Stream;
use crate::{error::Result, try_ffi};

// Type alias for the trait object Box itself (inner box).
type RustStreamCallbackDyn = Box<dyn FnOnce(Result<()>) + Send + 'static>;

// Type alias for the pointer type stored in the outer box.
type BoxedCallbackPtr = *mut RustStreamCallbackDyn;

type RustHostFunctionDyn = Box<dyn FnOnce() + Send + 'static>;

type BoxedHostFunctionPtr = *mut RustHostFunctionDyn;

impl Stream {
    /// This callback API is slated for eventual deprecation and removal.
    /// If you do not require the callback to execute after a device error, consider using [`sys::cudaLaunchHostFunc`](singe_cuda_sys::runtime::cudaLaunchHostFunc).
    /// Additionally, this callback mechanism is not supported with [`Stream::begin_capture`] and [`Stream::end_capture`], unlike [`sys::cudaLaunchHostFunc`](singe_cuda_sys::runtime::cudaLaunchHostFunc).
    ///
    /// Adds a callback to be called on the host after all currently enqueued items in the stream have completed.
    /// For each [`Stream::add_callback`] call, a callback is executed exactly once.
    /// A panic in the callback is caught and reported on standard error.
    /// The callback blocks later work in the stream until it is finished.
    ///
    /// The callback may be passed a successful status or an error code.
    /// In the event of a device error, all subsequently executed callbacks receive an appropriate [`Status`](crate::error::Status).
    ///
    /// Callbacks must not call CUDA functions.
    /// Attempting to do so may result in [`crate::error::Status::NotPermitted`].
    /// Callbacks must not perform any synchronization that may depend on outstanding device work or other callbacks that are not mandated to run earlier.
    /// Callbacks without a mandated order (in independent streams) execute in undefined order and may be serialized.
    ///
    /// For the purposes of Unified Memory, callback execution makes a number of guarantees:
    ///
    /// * The callback stream is considered idle for the duration of the callback.
    ///   Thus, for example, a callback may always use memory attached to the callback stream.
    /// * The start of execution of a callback has the same effect as synchronizing an event recorded in the same stream immediately before the callback.
    ///   It thus synchronizes streams which have been "joined" before the callback.
    /// * Adding device work to any stream does not have the effect of making the stream active until all preceding callbacks have executed.
    ///   Thus, for example, a callback might use global attached memory even if work has been added to another stream, if it has been properly ordered with an event.
    /// * Completion of a callback does not cause a stream to become active except as described above.
    ///   The callback stream will remain idle if no device work follows the callback, and will remain idle across consecutive callbacks without device work in between.
    ///   Thus, for example, stream synchronization can be done by signaling from a callback at the end of the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA rejects the callback registration, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`].
    pub fn add_callback<F>(&self, callback: F) -> Result<()>
    where
        F: FnOnce(Result<()>) + Send + 'static,
    {
        self.inner.context.bind()?;

        let boxed_dyn_callback: RustStreamCallbackDyn = Box::new(callback);
        let boxed_wrapper: Box<RustStreamCallbackDyn> = Box::new(boxed_dyn_callback);
        let user_data_ptr: BoxedCallbackPtr = Box::into_raw(boxed_wrapper);
        let final_user_data = user_data_ptr.cast();

        let flags = 0u32;

        // SAFETY: the boxed callback is leaked into CUDA and reclaimed exactly once by the trampoline.
        unsafe {
            let status = runtime::cudaStreamAddCallback(
                self.as_raw(),
                Some(stream_callback_trampoline),
                final_user_data, // Pass the thin pointer
                flags,
            );

            // If adding the callback fails, manually reconstruct and drop the *outer* Box to prevent leaking both boxes.
            if status != runtime::cudaError_t::CUDA_SUCCESS {
                // Reconstruct the outer box (Box<Box<dyn Trait>>)
                let _leaked_box = Box::from_raw(user_data_ptr);
                // Drop the reconstructed outer box.
                try_ffi!(status)?;
            }
        }

        Ok(())
    }

    /// Enqueues a host function to run after all currently enqueued work in this stream completes.
    ///
    /// Unlike [`Stream::add_callback`], CUDA does not call this function if the CUDA context is already in an error state, and then drops neither it nor its captures.
    /// The host function must not call CUDA APIs or perform synchronization that depends on outstanding device work.
    /// A panic in the host function is caught and reported on standard error.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is capturing, since a graph would run the `FnOnce` function on every replay.
    /// Also returns an error if the context cannot be bound, CUDA rejects the host function registration, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn launch_host_func<F>(&self, function: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        self.ensure_not_capturing()?;

        let boxed_dyn_function: RustHostFunctionDyn = Box::new(function);
        let boxed_wrapper: Box<RustHostFunctionDyn> = Box::new(boxed_dyn_function);
        let user_data_ptr: BoxedHostFunctionPtr = Box::into_raw(boxed_wrapper);
        let final_user_data = user_data_ptr.cast();

        // SAFETY: the boxed callback is leaked into CUDA and reclaimed exactly once by the trampoline.
        unsafe {
            let status = runtime::cudaLaunchHostFunc(
                self.as_raw(),
                Some(stream_host_function_trampoline),
                final_user_data,
            );

            if status != runtime::cudaError_t::CUDA_SUCCESS {
                let _leaked_box = Box::from_raw(user_data_ptr);
                try_ffi!(status)?;
            }
        }

        Ok(())
    }
}

// Trampoline function to bridge C FFI callback to Rust closure
extern "C" fn stream_callback_trampoline(
    _stream: runtime::cudaStream_t,
    status: runtime::cudaError_t,
    user_data: *mut ffi::c_void,
) {
    if user_data.is_null() {
        return;
    }

    let user_data_ptr = user_data as BoxedCallbackPtr;

    // SAFETY: CUDA calls this trampoline once with the pointer leaked when the callback was enqueued.
    let boxed_callback: Box<RustStreamCallbackDyn> = unsafe { Box::from_raw(user_data_ptr) };
    let callback: RustStreamCallbackDyn = *boxed_callback;

    let result = if status == runtime::cudaError_t::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(status.into())
    };

    // Unwinding must not cross into the CUDA driver.
    if panic::catch_unwind(panic::AssertUnwindSafe(|| callback(result))).is_err() {
        eprintln!("cuda stream callback panicked");
    }
}

extern "C" fn stream_host_function_trampoline(user_data: *mut ffi::c_void) {
    if user_data.is_null() {
        return;
    }

    let user_data_ptr = user_data as BoxedHostFunctionPtr;
    // SAFETY: CUDA calls this trampoline once with the pointer leaked when the callback was enqueued.
    let boxed_function: Box<RustHostFunctionDyn> = unsafe { Box::from_raw(user_data_ptr) };
    let function: RustHostFunctionDyn = *boxed_function;
    // Unwinding must not cross into the CUDA driver.
    if panic::catch_unwind(panic::AssertUnwindSafe(function)).is_err() {
        eprintln!("cuda host function panicked");
    }
}
