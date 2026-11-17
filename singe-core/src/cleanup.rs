use std::{error::Error, process};

use crate::report::ErrorChain;

/// Reports a cleanup step that failed in `Drop`, where the error cannot be returned.
///
/// Debug builds print `failed to <action> <object>: <error chain>` to standard error.
/// Release builds print nothing.
#[cold]
pub fn report_cleanup_failure(action: &str, object: &str, error: &(dyn Error + 'static)) {
    if cfg!(debug_assertions) {
        eprintln!("failed to {action} {object}: {}", ErrorChain(error));
    }
}

/// Prints `failed to <action> <object>: <error chain>` to standard error, then aborts the process.
///
/// Cleanup calls this when returning would release storage that queued work may still access.
/// The message prints in every build, since an abort leaves no other trace.
#[cold]
pub fn abort_after_cleanup_failure(action: &str, object: &str, error: &(dyn Error + 'static)) -> ! {
    eprintln!("failed to {action} {object}: {}", ErrorChain(error));
    process::abort();
}
