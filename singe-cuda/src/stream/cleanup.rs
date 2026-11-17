use singe_core::abort_after_cleanup_failure;

use crate::context::Context;

impl Context {
    /// Waits for all queued work in this context, or prints the error and aborts the process if the wait fails.
    ///
    /// Cleanup code calls this before releasing borrowed storage that queued work may still access.
    pub fn synchronize_or_abort(&self) {
        if let Err(error) = self.synchronize() {
            abort_after_cleanup_failure("synchronize", "cuda context", &error);
        }
    }
}
