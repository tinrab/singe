use std::{
    future::Future,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    thread,
};

use singe_ptx::emulator::{ExecutionError, Shutdown, Submission};

pub trait Observe {
    type Value;

    fn observe(self, asynchronous: bool) -> Result<Self::Value, ExecutionError>;
}

impl<T> Observe for Submission<T> {
    type Value = T;

    fn observe(self, asynchronous: bool) -> Result<T, ExecutionError> {
        if asynchronous {
            block_on(self)
        } else {
            self.wait()
        }
    }
}

impl Observe for Shutdown {
    type Value = ();

    fn observe(self, asynchronous: bool) -> Result<(), ExecutionError> {
        if asynchronous {
            block_on(self)
        } else {
            self.wait()
        }
    }
}

struct Notify(thread::Thread);

impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

pub fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(Notify(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}
