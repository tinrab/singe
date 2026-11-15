use crate::{
    memory::{CudaHostSlice, PinnedHostSlice, PinnedHostSliceMut, PinnedVec},
    view::{ByteBuffer, ByteBufferMut, DeviceRepr, HostRegion, HostSlice, HostSliceMut},
};

unsafe impl<T: DeviceRepr> HostSlice<T> for PinnedVec<T> {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

unsafe impl<T: DeviceRepr> HostSliceMut<T> for PinnedVec<T> {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

// SAFETY: the region names the pinned block's chunk, so every alias shares its identity and offsets.
unsafe impl<T: DeviceRepr> CudaHostSlice<T> for PinnedVec<T> {
    fn host_region(&self) -> HostRegion<'_> {
        self.allocator()
            .host_region(self.as_ptr().cast(), size_of_val(self.as_slice()))
    }
}

// SAFETY: every block of the pool is page-locked for as long as an owner holds the allocator.
unsafe impl<T: DeviceRepr> PinnedHostSlice<T> for PinnedVec<T> {}

unsafe impl<T: DeviceRepr> PinnedHostSliceMut<T> for PinnedVec<T> {}

unsafe impl ByteBuffer for PinnedVec<u8> {
    fn as_byte_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn byte_len(&self) -> usize {
        self.len()
    }
}

unsafe impl ByteBufferMut for PinnedVec<u8> {
    fn as_byte_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }
}
