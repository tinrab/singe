use std::{ptr, slice, task::Poll};

use singe_cuda_sys::driver::{CUresult, CUstream};

use crate::machine::{Machine, Status, size, submit_status};

/// Where the bytes of a copy live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Side {
    Host,
    Device,
}

impl Machine {
    /// Allocates `bytes` of zeroed memory on `device` and returns its address, or zero for an empty allocation.
    pub(crate) fn allocate(&self, device: usize, bytes: u64) -> Result<u64, CUresult> {
        if bytes == 0 {
            return Ok(0);
        }

        let len = size(bytes)?;
        let address = self
            .host(move |memory| memory.allocate(device, len, 256))
            .map_err(|_| CUresult::CUDA_ERROR_OUT_OF_MEMORY)?;
        self.state().allocations.insert(address, (device, bytes));
        Ok(address)
    }

    /// Frees an allocation once the commands of its device that may still use it finish.
    pub(crate) fn free(&self, address: u64) -> Status {
        if address == 0 {
            return Ok(());
        }

        let (device, _) = self
            .state()
            .allocations
            .get(&address)
            .copied()
            .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
        self.synchronize_device(device)?;
        self.host(move |memory| memory.free(address))?;
        self.state().allocations.remove(&address);
        Ok(())
    }

    /// Returns which side of a unified-address copy `address` is on.
    pub(crate) fn side(&self, address: u64) -> Side {
        if self.state().is_device_address(address) {
            Side::Device
        } else {
            Side::Host
        }
    }

    /// Copies `count` bytes in `stream`'s order; host memory is read now and written when the copy runs.
    ///
    /// # Safety
    ///
    /// A host `source` must be valid for reads of `count` bytes now, and a host `destination` for writes of `count` bytes until the copy completes.
    pub(crate) unsafe fn copy(
        &self,
        destination: (u64, Side),
        source: (u64, Side),
        count: u64,
        stream: CUstream,
        device: usize,
    ) -> Status {
        if count == 0 {
            return Ok(());
        }

        let len = size(count)?;
        let stream = self.stream(stream, device)?;
        let (to, from) = (destination.0, source.0);
        let submission = match (destination.1, source.1) {
            (Side::Host, Side::Host) => {
                self.wait(stream.synchronize())?;
                // SAFETY: the caller guarantees both host ranges are valid for `count` bytes.
                unsafe { ptr::copy(from as *const u8, to as *mut u8, len) };
                return Ok(());
            }
            (Side::Device, Side::Host) => {
                // SAFETY: the caller guarantees the host source is valid for reads of `count` bytes now.
                let bytes = unsafe { slice::from_raw_parts(from as *const u8, len) }.to_vec();
                stream.host_task(move |memory| Poll::Ready(memory.write(to, &bytes)))
            }
            (Side::Host, Side::Device) => stream.host_task(move |memory| {
                let bytes = memory.read(from, len)?;
                // SAFETY: the caller guarantees the host destination stays valid for writes of `count` bytes until the copy completes.
                unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), to as *mut u8, len) };
                Poll::Ready(Ok(()))
            }),
            (Side::Device, Side::Device) => {
                stream.host_task(move |memory| Poll::Ready(memory.copy(from, to, len)))
            }
        };
        drop(submission.map_err(|error| submit_status(&error))?);
        Ok(())
    }

    /// Sets `count` bytes at `address` to `value` in `stream`'s order.
    pub(crate) fn fill(
        &self,
        address: u64,
        value: u8,
        count: u64,
        stream: CUstream,
        device: usize,
    ) -> Status {
        if count == 0 {
            return Ok(());
        }

        let bytes = vec![value; size(count)?];
        let stream = self.stream(stream, device)?;
        drop(
            stream
                .host_task(move |memory| Poll::Ready(memory.write(address, &bytes)))
                .map_err(|error| submit_status(&error))?,
        );
        Ok(())
    }
}
