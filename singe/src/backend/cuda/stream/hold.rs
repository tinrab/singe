use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use super::CudaStream;
use crate::error::Result;

// How long a hold waits to be released, in case what was enqueued behind it waits for the stream itself.
const LONGEST_HOLD: Duration = Duration::from_secs(5);

// Keeps work enqueued on a stream from starting until this is released or dropped.
// What runs then has no host in between: timing it shows the work, however late the host was to enqueue each part.
#[derive(Debug)]
pub struct StreamHold {
    released: Arc<AtomicBool>,
    overran: Arc<AtomicBool>,
}

impl StreamHold {
    pub fn release(&self) {
        self.released.store(true, Ordering::Release);
    }

    // Whether the hold gave up before its release, which means later work ran before everything was enqueued.
    // It says so only once the stream has passed the hold.
    #[must_use]
    pub fn overran(&self) -> bool {
        self.overran.load(Ordering::Acquire)
    }
}

impl Drop for StreamHold {
    fn drop(&mut self) {
        self.release();
    }
}

impl CudaStream {
    // Holds what is enqueued on this stream from now on.
    //
    // # Errors
    //
    // Returns an error if CUDA cannot enqueue the hold.
    pub fn hold(&self) -> Result<StreamHold> {
        let released = Arc::new(AtomicBool::new(false));
        let overran = Arc::new(AtomicBool::new(false));
        let hold = StreamHold {
            released: Arc::clone(&released),
            overran: Arc::clone(&overran),
        };
        self.stream.launch_host_func(move || {
            let started = Instant::now();
            while !released.load(Ordering::Acquire) {
                if started.elapsed() > LONGEST_HOLD {
                    overran.store(true, Ordering::Release);
                    return;
                }

                thread::yield_now();
            }
        })?;
        Ok(hold)
    }
}
