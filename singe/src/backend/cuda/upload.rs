use singe_cuda::memory::{DeviceMemory, HostAllocationFlags, MemoryCopyKind, PinnedHostMemory};

use crate::{
    backend::cuda::{self, CudaEvent, EventTiming},
    data_type::DataType,
    device::Device,
    error::Result,
    shape::Shape,
    tensor::Tensor,
    utility::checked_mul,
};

// How many bytes each staging buffer holds, a multiple of every element size.
const CHUNK_BYTES: usize = 8 << 20;

// Fills device tensors from the host a chunk at a time, through two page-locked buffers: one fills while the other's copy runs on the eager stream.
// A tensor then takes no more host memory than the two buffers, however large it is.
pub struct CudaUpload {
    device: Device,
    staging: [PinnedHostMemory<u8>; 2],
    // When each buffer's last copy has read it.
    copied: [CudaEvent; 2],
    next: usize,
}

impl CudaUpload {
    pub fn create(device: Device) -> Result<Self> {
        let context = cuda::execution_context(device)?;
        let buffer = || {
            PinnedHostMemory::<u8>::zeroed(
                context.runtime().context(),
                CHUNK_BYTES,
                HostAllocationFlags::DEFAULT,
            )
        };
        Ok(Self {
            device,
            staging: [buffer()?, buffer()?],
            copied: [
                CudaEvent::create(device, EventTiming::Disabled)?,
                CudaEvent::create(device, EventTiming::Disabled)?,
            ],
            next: 0,
        })
    }

    // Returns a tensor of `shape` whose bytes `fill` writes, given each chunk's offset in them and the chunk to write.
    pub fn tensor(
        &mut self,
        data_type: DataType,
        shape: &Shape,
        mut fill: impl FnMut(usize, &mut [u8]) -> Result<()>,
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
        let mut offset = 0;
        while offset < total {
            let length = CHUNK_BYTES.min(total - offset);
            let index = self.next;
            self.next = 1 - index;
            self.copied[index].synchronize()?;
            let buffer = &mut self.staging[index].as_slice_mut()[..length];
            fill(offset, buffer)?;
            // SAFETY: the output spans `total` bytes past `target`, the page-locked buffer stays untouched until the event recorded after the copy completes, and the output's storage returns to the cache only for reuse ordered after the copy.
            unsafe {
                DeviceMemory::<u8>::copy_async(
                    target.add(offset),
                    buffer.as_ptr(),
                    length,
                    MemoryCopyKind::HostToDevice,
                    context.stream().raw(),
                )?;
            }

            self.copied[index].record(context.stream())?;
            offset += length;
        }

        // SAFETY: the copies write every byte of the output with the bytes `fill` gave.
        Ok(unsafe { output.assume_init() })
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
