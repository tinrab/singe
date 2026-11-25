use std::{collections::VecDeque, mem, sync::mpsc};

use singe_cuda::memory::{DeviceMemory, MemoryCopyKind, PinnedVec};

use crate::{
    backend::cuda::{self, CudaEvent, EventTiming},
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    parallel,
    shape::Shape,
    tensor::Tensor,
    utility::checked_mul,
};

// How many bytes each staging buffer holds, a multiple of every element size.
// Small buffers filled by many threads read faster than large ones by few: on a 20-core laptop, a warm load of Qwen3-0.6B's 1.2 GB took 72 ms through twenty 512 KiB buffers, 135 ms through eight of 16 MiB, and 160 ms through two of 8 MiB filled by all cores in turn.
const CHUNK_BYTES: usize = 512 << 10;

pub struct CudaUpload {
    device: Device,
    staging: Vec<PinnedVec<u8>>,
    // When each buffer's last copy has read it.
    copied: Vec<CudaEvent>,
}

impl CudaUpload {
    pub fn create(device: Device) -> Result<Self> {
        let context = cuda::execution_context(device)?;
        let pinned = context.runtime().context().pinned_host();
        // One buffer per pool thread, so each thread reads one chunk while the stream copies others.
        let buffers = parallel::num_threads().clamp(4, 64);
        let staging = (0..buffers)
            .map(|_| Ok(pinned.zeroed::<u8>(CHUNK_BYTES)?))
            .collect::<Result<Vec<_>>>()?;
        let copied = (0..buffers)
            .map(|_| CudaEvent::create(device, EventTiming::Disabled))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            device,
            staging,
            copied,
        })
    }

    // Returns a tensor of `shape` whose bytes `fill` writes, given each chunk's offset in them and the chunk to write.
    // `fill` runs on pool threads, several chunks at once, so it must not depend on the order chunks arrive in.
    pub fn tensor(
        &mut self,
        data_type: DataType,
        shape: &Shape,
        fill: impl Fn(usize, &mut [u8]) -> Result<()> + Sync,
    ) -> Result<Tensor> {
        let context = cuda::execution_context(self.device)?;
        context.bind()?;
        let output = context.allocate_output(data_type, shape)?;
        let total = checked_mul(
            shape.element_count(),
            data_type.size_bytes(),
            "cuda upload bytes",
        )?;
        let target = output.region().region().as_ptr().cast::<u8>();
        let chunks = total.div_ceil(CHUNK_BYTES);

        let mut buffers = mem::take(&mut self.staging)
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
        // Buffers at home, the longest idle first, whose last copy is the likeliest done.
        let mut free = (0..buffers.len()).collect::<VecDeque<_>>();
        let (sender, receiver) = mpsc::channel();
        let mut failure = None;
        let mut issued = false;
        parallel::in_place_scope(|scope| {
            let mut next = 0;
            let mut reading = 0;
            loop {
                while failure.is_none() && next < chunks {
                    let Some(slot) = free.pop_front() else {
                        break;
                    };
                    if let Err(error) = self.copied[slot].synchronize() {
                        failure = Some(error);
                        free.push_front(slot);
                        break;
                    }

                    let Some(mut buffer) = buffers[slot].take() else {
                        failure = Some(Error::invalid_argument("staging buffer in use"));
                        break;
                    };
                    let offset = next * CHUNK_BYTES;
                    let length = CHUNK_BYTES.min(total - offset);
                    let (sender, fill) = (sender.clone(), &fill);
                    scope.spawn(move |_| {
                        let result = fill(offset, &mut buffer[..length]);
                        // The receiver outlives the scope, so the send only fails if the loop below has already returned, which it does only once every read is back.
                        let _ = sender.send((slot, offset, length, buffer, result));
                    });
                    next += 1;
                    reading += 1;
                }

                if reading == 0 {
                    break;
                }

                let Ok((slot, offset, length, buffer, result)) = receiver.recv() else {
                    break;
                };
                reading -= 1;
                if failure.is_none() {
                    match result {
                        Ok(()) => {
                            issued = true;
                            if let Err(error) = self.copy(target, (offset, length), &buffer, slot) {
                                failure = Some(error);
                            }
                        }
                        Err(error) => failure = Some(error),
                    }
                }

                buffers[slot] = Some(buffer);
                free.push_back(slot);
            }
        });

        self.staging = buffers.into_iter().flatten().collect();
        if let Some(error) = failure {
            // A buffer may still be read by a copy whose event was never recorded, so it stays out of the pool unless the stream drains.
            if issued && context.stream().synchronize().is_err() {
                for buffer in mem::take(&mut self.staging) {
                    mem::forget(buffer);
                }
            }
            return Err(error);
        }

        // SAFETY: the copies write every byte of the output with the bytes `fill` gave.
        Ok(unsafe { output.assume_init() })
    }

    fn copy(
        &self,
        target: *mut u8,
        (offset, length): (usize, usize),
        buffer: &PinnedVec<u8>,
        slot: usize,
    ) -> Result<()> {
        let context = cuda::execution_context(self.device)?;
        // SAFETY: the output spans `total` bytes past `target`, of which this chunk is one, the page-locked buffer stays untouched until the event recorded after the copy completes, and the output's storage returns to the cache only for reuse ordered after the copy.
        unsafe {
            DeviceMemory::<u8>::copy_async(
                target.add(offset),
                buffer.as_ptr(),
                length,
                MemoryCopyKind::HostToDevice,
                context.stream().raw(),
            )?;
        }
        self.copied[slot].record(context.stream())
    }
}

impl Drop for CudaUpload {
    fn drop(&mut self) {
        // A copy still reading a buffer outlives nothing it reads.
        for event in &self.copied {
            let _ = event.synchronize();
        }
    }
}
