use std::{
    os::fd::{AsFd, AsRawFd, OwnedFd},
    ptr,
};

use singe_cuda::stream::StreamBinding;
use singe_cufile_sys as sys;

use crate::{
    buffer::{IoBuffer, IoBufferMut},
    error::{Error, Result, Status},
    stream::AsyncIo,
    try_ffi,
    utility::{null_fs_ops, to_i64, to_u64, to_usize},
};

#[derive(Debug)]
pub struct FileHandle {
    handle: sys::CUfileHandle_t,
    _file: Option<OwnedFd>,
}

// cuFile registrations are owning file handles. Ownership may move between
// threads, while I/O methods keep buffer lifetimes tied to Rust references and
// do not expose shared mutation of wrapper-owned state.
unsafe impl Send for FileHandle {}

impl FileHandle {
    pub fn create(file: &impl AsFd) -> Result<Self> {
        let file = file.as_fd().try_clone_to_owned()?;
        let mut descriptor = sys::CUfileDescr_t {
            type_: sys::CUfileFileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD,
            handle: sys::CUfileDescr_t__bindgen_ty_1 {
                fd: file.as_raw_fd(),
            },
            fs_ops: null_fs_ops(),
        };
        let mut handle = unsafe { Self::register_descriptor(&mut descriptor) }?;
        handle._file = Some(file);
        Ok(handle)
    }

    /// Registers a raw cuFile descriptor and takes ownership of the returned handle.
    ///
    /// # Safety
    ///
    /// `descriptor` must describe a file and filesystem operations accepted by `cuFileHandleRegister`.
    /// Resources referenced by the descriptor must stay valid for the duration of the call.
    /// The returned wrapper deregisters the cuFile handle on drop but only owns the source file when `_file` is set.
    pub(crate) unsafe fn register_descriptor(descriptor: &mut sys::CUfileDescr_t) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cuFileHandleRegister(
                &raw mut handle,
                descriptor as *mut _,
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            _file: None,
        })
    }

    pub(crate) fn as_raw(&self) -> sys::CUfileHandle_t {
        self.handle
    }

    pub fn read<B>(
        &self,
        buffer: &mut B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
    ) -> Result<usize>
    where
        B: IoBufferMut + ?Sized,
    {
        let size = checked_io_size(buffer.byte_len()?, size, buffer_offset)?;
        let result = unsafe {
            sys::cuFileRead(
                self.handle,
                buffer.ptr_mut().as_mut() as _,
                to_u64(size, "size")?,
                to_i64(file_offset, "file_offset")?,
                to_i64(buffer_offset, "buffer_offset")?,
            )
        };
        operation_result(result)
    }

    pub fn write<B>(
        &self,
        buffer: &B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
    ) -> Result<usize>
    where
        B: IoBuffer + ?Sized,
    {
        let size = checked_io_size(buffer.byte_len()?, size, buffer_offset)?;
        let result = unsafe {
            sys::cuFileWrite(
                self.handle,
                buffer.ptr().as_const() as _,
                to_u64(size, "size")?,
                to_i64(file_offset, "file_offset")?,
                to_i64(buffer_offset, "buffer_offset")?,
            )
        };
        operation_result(result)
    }

    pub fn read_async<'a, B>(
        &'a self,
        buffer: &'a mut B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        stream: &'a StreamBinding,
    ) -> Result<AsyncIo<'a>>
    where
        B: IoBufferMut + ?Sized,
    {
        AsyncIo::read(self, buffer, size, file_offset, buffer_offset, stream)
    }

    pub fn write_async<'a, B>(
        &'a self,
        buffer: &'a B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        stream: &'a StreamBinding,
    ) -> Result<AsyncIo<'a>>
    where
        B: IoBuffer + ?Sized,
    {
        AsyncIo::write(self, buffer, size, file_offset, buffer_offset, stream)
    }
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                sys::cuFileHandleDeregister(self.handle);
            }
        }
    }
}

pub(crate) fn operation_result(result: i64) -> Result<usize> {
    if result < 0 {
        let code = result.unsigned_abs();
        if code > sys::CUFILEOP_BASE_ERR as u64
            && let Ok(status) = Status::try_from(code as u32)
        {
            return Err(status.into());
        }

        let code = i32::try_from(code).map_err(|_| Error::OperationFailed(result as isize))?;
        return Err(std::io::Error::from_raw_os_error(code).into());
    }
    to_usize(result, "result")
}

pub(crate) fn checked_io_size(
    buffer_size: usize,
    size: usize,
    buffer_offset: u64,
) -> Result<usize> {
    let offset = to_usize(buffer_offset, "buffer_offset")?;
    let available = buffer_size.checked_sub(offset).ok_or(Error::OutOfRange {
        name: "buffer_offset".into(),
    })?;
    if size > available {
        return Err(Error::OutOfRange {
            name: "size".into(),
        });
    }
    Ok(size)
}
