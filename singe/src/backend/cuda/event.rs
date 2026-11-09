use std::{sync::Arc, time::Duration};

use singe_cuda::event::{Event, EventFlags, EventRecordFlags};

use crate::{
    backend::cuda::{runtime::runtime_for_device, stream::CudaStream},
    device::Device,
    error::{Error, Result},
};

// Whether an event records when it completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventTiming {
    Recorded,
    Disabled,
}

/// A point in a CUDA stream's work that other streams and the host can wait for.
#[derive(Debug, Clone)]
pub struct CudaEvent {
    event: Arc<Event>,
    device: Device,
}

impl CudaEvent {
    /// Creates an event on `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not a valid CUDA device or CUDA cannot create the event.
    pub fn create(device: Device, timing: EventTiming) -> Result<Self> {
        let runtime = runtime_for_device(device)?;
        let flags = match timing {
            EventTiming::Recorded => EventFlags::DEFAULT,
            EventTiming::Disabled => EventFlags::DISABLE_TIMING,
        };
        Ok(Self {
            event: Arc::new(runtime.context().create_event_with_flags(flags)?),
            device,
        })
    }

    /// Returns the device whose context owns this event.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Marks the current end of `stream`'s submitted work.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another device or CUDA cannot record the event.
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        self.check_stream(stream)?;
        Ok(self.event.record(stream.raw(), EventRecordFlags::DEFAULT)?)
    }

    /// Makes `stream` wait for the work this event marks before running later submissions.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another device or CUDA cannot enqueue the wait.
    pub fn wait(&self, stream: &CudaStream) -> Result<()> {
        self.check_stream(stream)?;
        Ok(stream.raw().wait_event(&self.event)?)
    }

    /// Returns whether the marked work has completed.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA reports a failure other than incomplete work.
    pub fn query(&self) -> Result<bool> {
        Ok(self.event.query()?)
    }

    /// Blocks the host until the marked work completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the marked work failed or CUDA cannot synchronize.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.event.synchronize()?)
    }

    /// Returns the GPU time between `start` and this event; both must have timing enabled and be complete.
    ///
    /// # Errors
    ///
    /// Returns an error if either event lacks timing, has not completed, or belongs to another device.
    pub fn elapsed_since(&self, start: &Self) -> Result<Duration> {
        if start.device != self.device {
            return Err(Error::device_mismatch(
                "cuda event timing",
                self.device,
                start.device,
            ));
        }

        let milliseconds = self.event.elapsed_time_since(&start.event)?;
        Ok(Duration::from_secs_f64(f64::from(milliseconds) / 1_000.0))
    }

    fn check_stream(&self, stream: &CudaStream) -> Result<()> {
        if stream.device() == self.device {
            Ok(())
        } else {
            Err(Error::device_mismatch(
                "cuda event stream",
                self.device,
                stream.device(),
            ))
        }
    }
}
