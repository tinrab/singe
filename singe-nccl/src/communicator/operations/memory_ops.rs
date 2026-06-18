use singe_cuda::{memory::DeviceMemory, stream::Stream};

use crate::{
    buffer::{BufferMut, BufferRef},
    communicator::Communicator,
    error::Result,
    types::DataTypeLike,
};

impl Communicator {
    pub fn all_to_all_memory<T: DataTypeLike>(
        &self,
        send: &DeviceMemory<T>,
        recv: &mut DeviceMemory<T>,
        stream: &Stream,
    ) -> Result<()> {
        self.all_to_all(
            BufferRef::from_device_memory(send)?,
            BufferMut::from_device_memory(recv)?,
            stream,
        )
    }

    pub fn gather_memory<T: DataTypeLike>(
        &self,
        send: &DeviceMemory<T>,
        recv: Option<&mut DeviceMemory<T>>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.gather(
            BufferRef::from_device_memory(send)?,
            recv.map(BufferMut::from_device_memory).transpose()?,
            root,
            stream,
        )
    }

    pub fn scatter_memory<T: DataTypeLike>(
        &self,
        send: Option<&DeviceMemory<T>>,
        recv: &mut DeviceMemory<T>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.scatter(
            send.map(BufferRef::from_device_memory).transpose()?,
            BufferMut::from_device_memory(recv)?,
            root,
            stream,
        )
    }

    pub fn broadcast_in_place_memory<T: DataTypeLike>(
        &self,
        buffer: &mut DeviceMemory<T>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.broadcast_in_place(BufferMut::from_device_memory(buffer)?, root, stream)
    }
}
