//! Gzip compression and decompression between Rust readers and writers, on the GPU.

use std::{
    any::Any,
    ffi::c_void,
    io,
    io::{ErrorKind, Read, Write},
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    process, slice,
    sync::Arc,
};

use singe_cuda::{context::Context as CudaContext, stream::Stream};

use crate::{
    error::{Error, Result},
    gzip::CompressOptions,
    manager::bridge_result,
    sys::manager as ffi,
    workspace::TemporaryBuffer,
};

enum CallbackFailure {
    Io(io::Error),
    Panic(Box<dyn Any + Send>),
}

struct Input<'a, R: ?Sized> {
    reader: &'a mut R,
    context: &'a Arc<CudaContext>,
    failure: Option<CallbackFailure>,
}

struct Output<'a, W: ?Sized> {
    writer: &'a mut W,
    context: &'a Arc<CudaContext>,
    failure: Option<CallbackFailure>,
}

unsafe extern "C" fn read<R: Read + Send + ?Sized>(
    context: *mut c_void,
    data: *mut u8,
    capacity: usize,
    count: *mut usize,
) -> i32 {
    // SAFETY: nvCOMP passes back the `Input` context registered for this stream call, which outlives the callback.
    let input = unsafe { &mut *context.cast::<Input<'_, R>>() };
    if input.failure.is_some() {
        return -1;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: nvCOMP provides a writable buffer of `capacity` bytes for this callback.
        let buffer = unsafe { slice::from_raw_parts_mut(data, capacity) };
        loop {
            match input.reader.read(buffer) {
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                result => return result,
            }
        }
    }));
    // User I/O may bind another CUDA context. Native code must resume in
    // the stream's actual context on success, error, and caught unwinding.
    if input.context.bind().is_err() {
        process::abort();
    }
    match result {
        Ok(Ok(bytes)) if bytes <= capacity => {
            // SAFETY: `count` is the output location nvCOMP supplied for this callback.
            unsafe {
                *count = bytes;
            }
            0
        }
        Ok(Ok(_)) => {
            input.failure = Some(CallbackFailure::Io(io::Error::new(
                ErrorKind::InvalidData,
                "reader returned an invalid byte count",
            )));
            -1
        }
        Ok(Err(error)) => {
            input.failure = Some(CallbackFailure::Io(error));
            -1
        }
        Err(panic) => {
            input.failure = Some(CallbackFailure::Panic(panic));
            -1
        }
    }
}

unsafe extern "C" fn write<W: Write + Send + ?Sized>(
    context: *mut c_void,
    data: *const u8,
    count: usize,
) -> i32 {
    // SAFETY: nvCOMP passes back the `Output` context registered for this stream call, which outlives the callback.
    let output = unsafe { &mut *context.cast::<Output<'_, W>>() };
    if output.failure.is_some() {
        return -1;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        output
            .writer
            // SAFETY: nvCOMP provides `count` readable bytes at `data` for this callback.
            .write_all(unsafe { slice::from_raw_parts(data, count) })
    }));
    // Restore native context even when the writer failed or panicked.
    if output.context.bind().is_err() {
        process::abort();
    }
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            output.failure = Some(CallbackFailure::Io(error));
            -1
        }
        Err(panic) => {
            output.failure = Some(CallbackFailure::Panic(panic));
            -1
        }
    }
}

fn callback_result(failure: Option<CallbackFailure>) -> Result<()> {
    match failure {
        None => Ok(()),
        Some(CallbackFailure::Io(error)) => Err(Error::Io(error)),
        Some(CallbackFailure::Panic(panic)) => resume_unwind(panic),
    }
}

/// Gzip compressor and decompressor that reads from a [`Read`] and writes to a [`Write`], using the GPU.
///
/// It owns its temporary GPU memory and runs on one stream.
/// It does not support gzip data with more than one member.
#[derive(Debug)]
pub struct GzipStream {
    workspace: TemporaryBuffer,
    stream: Arc<Stream>,
    options: CompressOptions,
}

impl GzipStream {
    /// Returns the size, in bytes, of the temporary GPU memory that compression with `options` needs.
    ///
    /// # Errors
    ///
    /// Returns an error if nvCOMP fails the query.
    pub fn compression_workspace_size(options: CompressOptions) -> Result<usize> {
        let mut bytes = 0;
        // SAFETY: `singe_nvcomp_gzip_temp_size` receives live handles owned by the wrappers and valid output locations.
        bridge_result(unsafe {
            ffi::singe_nvcomp_gzip_temp_size(1, options.algorithm.into(), &raw mut bytes)
        })?;
        Ok(bytes)
    }

    /// Returns the size, in bytes, of the temporary GPU memory that decompression needs.
    ///
    /// # Errors
    ///
    /// Returns an error if nvCOMP fails the query.
    pub fn decompression_workspace_size() -> Result<usize> {
        let mut bytes = 0;
        // SAFETY: `singe_nvcomp_gzip_temp_size` receives live handles owned by the wrappers and valid output locations.
        bridge_result(unsafe { ffi::singe_nvcomp_gzip_temp_size(0, 0, &raw mut bytes) })?;
        Ok(bytes)
    }

    /// Creates a stream compressor on `stream`, allocating temporary GPU memory for compression with `options` and for decompression.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or cannot be queried, the CUDA context cannot be bound, nvCOMP fails a workspace query, or the allocation fails.
    pub fn create(options: CompressOptions, stream: &Arc<Stream>) -> Result<Self> {
        stream.ensure_not_capturing()?;
        stream.context().bind()?;
        let size =
            Self::compression_workspace_size(options)?.max(Self::decompression_workspace_size()?);
        Ok(Self {
            stream: Arc::clone(stream),
            workspace: TemporaryBuffer::create(stream.context(), size)?,
            options,
        })
    }

    /// Reads all of `input`, compresses it, and writes gzip data to `output`.
    ///
    /// nvCOMP may call `input` and `output` from its own worker threads, which is why they must be `Send`.
    /// Those calls finish before this returns, and interrupted reads are retried.
    /// On error, `output` may already hold part of the result.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or cannot be queried, the CUDA context cannot be bound, reading or writing fails, or nvCOMP fails.
    ///
    /// # Panics
    ///
    /// If `input` or `output` panics, the panic resumes after nvCOMP returns and its GPU work has finished.
    pub fn compress(
        &mut self,
        input: &mut (impl Read + Send + ?Sized),
        output: &mut (impl Write + Send + ?Sized),
    ) -> Result<()> {
        self.run(1, input, output).map(|_| ())
    }

    /// Reads gzip data from `input`, decompresses it, and writes the result to `output`.
    ///
    /// nvCOMP may call `input` and `output` from its own worker threads, which is why they must be `Send`.
    /// Those calls finish before this returns, and interrupted reads are retried.
    /// On error, `output` may already hold part of the result.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or cannot be queried, the CUDA context cannot be bound, reading or writing fails, or nvCOMP fails.
    ///
    /// # Panics
    ///
    /// If `input` or `output` panics, the panic resumes after nvCOMP returns and its GPU work has finished.
    pub fn decompress(
        &mut self,
        input: &mut (impl Read + Send + ?Sized),
        output: &mut (impl Write + Send + ?Sized),
    ) -> Result<()> {
        self.run(0, input, output).map(|_| ())
    }

    /// Returns the uncompressed size, in bytes, of the gzip data in `input`.
    ///
    /// This decompresses everything and discards the output, so it costs as much as [`decompress`](Self::decompress).
    /// nvCOMP may call `input` from its own worker threads, which is why it must be `Send`.
    /// Those calls finish before this returns, and interrupted reads are retried.
    /// On error, `input` may already have been partly read.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or cannot be queried, the CUDA context cannot be bound, reading fails, or nvCOMP fails.
    ///
    /// # Panics
    ///
    /// If `input` panics, the panic resumes after nvCOMP returns and its GPU work has finished.
    pub fn decompressed_size(&mut self, input: &mut (impl Read + Send + ?Sized)) -> Result<usize> {
        self.run(2, input, &mut io::sink())
    }

    fn run<R: Read + Send + ?Sized, W: Write + Send + ?Sized>(
        &mut self,
        operation: i32,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<usize> {
        self.stream.ensure_not_capturing()?;
        self.stream.context().bind()?;
        let mut input = Input {
            reader,
            context: self.stream.context(),
            failure: None,
        };
        let mut output = Output {
            writer,
            context: self.stream.context(),
            failure: None,
        };
        let mut size = 0;
        // SAFETY: `singe_nvcomp_gzip_stream` receives live handles owned by the wrappers and valid output locations.
        let code = unsafe {
            ffi::singe_nvcomp_gzip_stream(
                operation,
                self.options.algorithm.into(),
                Some(read::<R>),
                (&raw mut input).cast(),
                Some(write::<W>),
                (&raw mut output).cast(),
                self.workspace.as_device_ptr().as_ptr().cast(),
                self.workspace.len(),
                self.stream.as_raw().cast(),
                &raw mut size,
            )
        };
        // Callbacks cannot unwind through C++; resume a Rust panic only after the bridge returns.
        callback_result(input.failure)?;
        callback_result(output.failure)?;
        bridge_result(code)?;
        Ok(size)
    }
}

impl Drop for GzipStream {
    fn drop(&mut self) {
        if let Some(allocation) = self.workspace.allocation()
            && self
                .stream
                .defer_drop_until_capture_end(Arc::clone(allocation))
                .is_err()
        {
            process::abort();
        }
    }
}
