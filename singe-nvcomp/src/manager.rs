//! nvCOMP managers, which compress and decompress whole buffers in nvCOMP's own format.
//!
//! A manager splits the input into chunks, compresses them in one batch, and writes a header that records the codec and sizes.
//! [`Compressor::from_compressed`] reads that header back to pick the right codec.
//! Every call waits for its GPU work before returning.
use std::{
    ffi::CStr,
    mem::ManuallyDrop,
    process,
    ptr::{self, NonNull},
    result::Result as StdResult,
    sync::Arc,
};

use singe_cuda::{
    context::Context as CudaContext, error::Error as CudaError, memory::DeviceMemory,
    stream::Stream, view::DeviceSlice,
};
use sys::manager as ffi;

use crate::{
    ans, bitcomp, cascaded, deflate,
    error::{Error, Result},
    gdeflate, gzip,
    lz4::{CompressOptions, DecompressOptions},
    snappy, sys,
    types::{AlignmentRequirements, DecompressBackend, GzipDecompressAlgorithm},
    utility::validate_alignment,
    zstd,
};

/// Layout of the data that a manager writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitstreamKind {
    /// Chunks the input by the manager's chunk size and compresses the chunks in parallel, after a header with nvCOMP metadata.
    ///
    /// Only this layout can hold checksums.
    Native,
    /// Compresses the input with the codec alone, without a header.
    Raw,
    /// Like [`Raw`](Self::Raw), after a header that holds only the uncompressed size.
    WithUncompressedSize,
}

impl BitstreamKind {
    fn into_raw(self) -> ffi::SingeNvcompBitstreamKind {
        match self {
            Self::Native => ffi::SingeNvcompBitstreamKind::SingeNvcompNative,
            Self::Raw => ffi::SingeNvcompBitstreamKind::SingeNvcompRaw,
            Self::WithUncompressedSize => {
                ffi::SingeNvcompBitstreamKind::SingeNvcompWithUncompressedSize
            }
        }
    }
}

/// Whether a manager favors running alongside other GPU work or finishing quickly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPolicy {
    /// Favors running alongside other work on the GPU.
    Concurrency,
    /// Favors finishing each call quickly.
    Latency,
}

impl ExecutionPolicy {
    fn into_raw(self) -> ffi::SingeNvcompExecutionPolicy {
        match self {
            Self::Concurrency => ffi::SingeNvcompExecutionPolicy::SingeNvcompConcurrency,
            Self::Latency => ffi::SingeNvcompExecutionPolicy::SingeNvcompLatency,
        }
    }
}

/// Whether a manager computes checksums when compressing and verifies them when decompressing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumPolicy {
    /// Neither computes nor verifies checksums.
    NoComputeNoVerify,
    /// Computes checksums but does not verify them.
    ComputeAndNoVerify,
    /// Does not compute checksums, and verifies them when the compressed data has them.
    NoComputeAndVerifyIfPresent,
    /// Computes checksums, and verifies them when the compressed data has them.
    ComputeAndVerifyIfPresent,
    /// Computes and verifies checksums.
    ///
    /// Decompression fails if the compressed data has no checksums.
    ComputeAndVerify,
}

impl ChecksumPolicy {
    fn into_raw(self) -> ffi::SingeNvcompChecksumPolicy {
        match self {
            Self::NoComputeNoVerify => ffi::SingeNvcompChecksumPolicy::SingeNvcompNoComputeNoVerify,
            Self::ComputeAndNoVerify => {
                ffi::SingeNvcompChecksumPolicy::SingeNvcompComputeAndNoVerify
            }
            Self::NoComputeAndVerifyIfPresent => {
                ffi::SingeNvcompChecksumPolicy::SingeNvcompNoComputeAndVerifyIfPresent
            }
            Self::ComputeAndVerifyIfPresent => {
                ffi::SingeNvcompChecksumPolicy::SingeNvcompComputeAndVerifyIfPresent
            }
            Self::ComputeAndVerify => ffi::SingeNvcompChecksumPolicy::SingeNvcompComputeAndVerify,
        }
    }
}

/// Codec that a manager uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// LZ4 compression.
    Lz4,
    /// Snappy compression.
    Snappy,
    /// ANS compression.
    Ans,
    /// `GDeflate` compression.
    Gdeflate,
    /// Cascaded compression.
    Cascaded,
    /// Bitcomp compression.
    Bitcomp,
    /// Zstd compression.
    Zstd,
    /// Deflate compression.
    Deflate,
    /// Gzip compression.
    Gzip,
}

impl Format {
    fn into_raw(self) -> ffi::SingeNvcompFormat {
        match self {
            Self::Lz4 => ffi::SingeNvcompFormat::SingeNvcompLz4,
            Self::Snappy => ffi::SingeNvcompFormat::SingeNvcompSnappy,
            Self::Ans => ffi::SingeNvcompFormat::SingeNvcompAns,
            Self::Gdeflate => ffi::SingeNvcompFormat::SingeNvcompGdeflate,
            Self::Cascaded => ffi::SingeNvcompFormat::SingeNvcompCascaded,
            Self::Bitcomp => ffi::SingeNvcompFormat::SingeNvcompBitcomp,
            Self::Zstd => ffi::SingeNvcompFormat::SingeNvcompZstd,
            Self::Deflate => ffi::SingeNvcompFormat::SingeNvcompDeflate,
            Self::Gzip => ffi::SingeNvcompFormat::SingeNvcompGzip,
        }
    }
}

/// Codec and its options for a manager.
#[derive(Debug, Clone, Copy)]
pub enum CodecOptions {
    /// Compression and decompression settings for LZ4.
    Lz4 {
        /// Compression options.
        compress: CompressOptions,
        /// Decompression options.
        decompress: DecompressOptions,
    },
    /// Compression and decompression settings for Snappy.
    Snappy {
        /// Compression options.
        compress: snappy::CompressOptions,
        /// Decompression options.
        decompress: snappy::DecompressOptions,
    },
    /// Compression and decompression settings for ANS.
    Ans {
        /// Compression options.
        compress: ans::CompressOptions,
        /// Decompression options.
        decompress: ans::DecompressOptions,
    },
    /// Compression and decompression settings for `GDeflate`.
    Gdeflate {
        /// Compression options.
        compress: gdeflate::CompressOptions,
        /// Decompression options.
        decompress: gdeflate::DecompressOptions,
    },
    /// Compression and decompression settings for Cascaded.
    Cascaded {
        /// Compression options.
        compress: cascaded::CompressOptions,
        /// Decompression options.
        decompress: cascaded::DecompressOptions,
    },
    /// Compression and decompression settings for Bitcomp.
    Bitcomp {
        /// Compression options.
        compress: bitcomp::CompressOptions,
        /// Decompression options.
        decompress: bitcomp::DecompressOptions,
    },
    /// Compression and decompression settings for Zstd.
    Zstd {
        /// Compression options.
        compress: zstd::CompressOptions,
        /// Decompression options.
        decompress: zstd::DecompressOptions,
    },
    /// Compression and decompression settings for Deflate.
    Deflate {
        /// Compression options.
        compress: deflate::CompressOptions,
        /// Decompression options.
        decompress: deflate::DecompressOptions,
    },
    /// Compression and decompression settings for Gzip.
    Gzip {
        /// Compression options.
        compress: gzip::CompressOptions,
        /// Decompression options.
        decompress: gzip::DecompressOptions,
    },
}

/// Settings for creating a GPU manager.
#[derive(Debug, Clone, Copy)]
pub struct ManagerOptions {
    /// Size, in bytes, of the chunks the input is split into; 65,536 by default.
    ///
    /// It must be nonzero, a multiple of the codec's element size, and within the codec's limits.
    pub chunk_size: usize,
    /// Checksum policy; [`ChecksumPolicy::NoComputeNoVerify`] by default.
    pub checksum: ChecksumPolicy,
    /// Execution policy; [`ExecutionPolicy::Concurrency`] by default.
    pub execution: ExecutionPolicy,
    /// Layout of the compressed data; [`BitstreamKind::Native`] by default.
    pub bitstream: BitstreamKind,
}

impl Default for ManagerOptions {
    fn default() -> Self {
        Self {
            chunk_size: 65536,
            checksum: ChecksumPolicy::NoComputeNoVerify,
            execution: ExecutionPolicy::Concurrency,
            bitstream: BitstreamKind::Native,
        }
    }
}

/// Settings for a manager created from compressed data with [`Compressor::from_compressed`].
#[derive(Debug, Clone, Copy)]
pub struct DetectionOptions {
    /// Checksum policy; [`ChecksumPolicy::NoComputeNoVerify`] by default.
    pub checksum: ChecksumPolicy,
    /// Execution policy; [`ExecutionPolicy::Concurrency`] by default.
    pub execution: ExecutionPolicy,
    /// Engine that runs decompression; [`DecompressBackend::Default`] by default.
    pub backend: DecompressBackend,
    /// Whether to sort chunks before hardware decompression to balance the load; `false` by default.
    ///
    /// Used only by LZ4, Snappy, Deflate, and Gzip.
    pub sort_before_hardware_decompression: bool,
    /// Gzip decompression algorithm; [`GzipDecompressAlgorithm::Naive`] by default.
    ///
    /// Other codecs ignore it.
    pub algorithm: GzipDecompressAlgorithm,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            checksum: ChecksumPolicy::NoComputeNoVerify,
            execution: ExecutionPolicy::Concurrency,
            backend: DecompressBackend::Default,
            sort_before_hardware_decompression: false,
            algorithm: GzipDecompressAlgorithm::Naive,
        }
    }
}

#[derive(Debug)]
struct Handle(NonNull<ffi::SingeNvcompManager>);

impl Handle {
    fn create(raw: *mut ffi::SingeNvcompManager) -> Result<Self> {
        NonNull::new(raw)
            .map(Self)
            .ok_or_else(|| Error::Compressor {
                message: "null manager returned by nvcomp".into(),
            })
    }

    fn as_raw(&self) -> *mut ffi::SingeNvcompManager {
        self.0.as_ptr()
    }

    fn compression_bound(&self, bytes: usize) -> Result<usize> {
        let mut output = 0;
        // SAFETY: `singe_nvcomp_compress_bound` receives live handles owned by the wrappers and valid output locations.
        bridge_result(unsafe {
            ffi::singe_nvcomp_compress_bound(self.as_raw(), bytes, &raw mut output)
        })?;
        Ok(output)
    }

    fn alignments(&self, decompress: bool) -> Result<AlignmentRequirements> {
        let mut value = AlignmentRequirements {
            input: 0,
            output: 0,
            temp: 0,
        };
        // SAFETY: `singe_nvcomp_alignments` receives live handles owned by the wrappers and valid output locations.
        bridge_result(unsafe {
            ffi::singe_nvcomp_alignments(
                self.as_raw(),
                decompress.into(),
                &raw mut value.input,
                &raw mut value.output,
                &raw mut value.temp,
            )
        })?;
        Ok(value)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: `singe_nvcomp_manager_destroy` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            ffi::singe_nvcomp_manager_destroy(self.as_raw());
        }
    }
}

/// Owns the native manager and destroys it when the last owner drops.
/// It keeps the CUDA context alive but not the stream, which avoids a reference cycle during capture cleanup.
#[derive(Debug)]
struct GpuManagerOwner {
    handle: Handle,
    context: Arc<CudaContext>,
}

impl GpuManagerOwner {
    fn create(raw: *mut ffi::SingeNvcompManager, context: &Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            handle: Handle::create(raw)?,
            context: Arc::clone(context),
        })
    }

    fn into_raw(self) -> *mut ffi::SingeNvcompManager {
        let this = ManuallyDrop::new(self);
        let raw = this.handle.as_raw();
        // SAFETY: transfer the native handle without running Handle's destructor;
        // RawCompressor's stream retains the same context after this reference drops.
        unsafe {
            drop(ptr::read(&raw const this.context));
        }
        raw
    }
}

// SAFETY: only private ownership is shared. Native operations require an exclusive
// Compressor borrow; final cleanup binds its context and runs once on any thread.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
unsafe impl Send for GpuManagerOwner {}

unsafe impl Sync for GpuManagerOwner {}

impl Drop for GpuManagerOwner {
    fn drop(&mut self) {
        if self.context.bind().is_err() {
            process::abort();
        }
    }
}

pub(crate) fn bridge_result(code: i32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }

    // SAFETY: `singe_nvcomp_last_error` receives live handles owned by the wrappers and valid output locations.
    let message = unsafe { CStr::from_ptr(ffi::singe_nvcomp_last_error()) }
        .to_string_lossy()
        .into_owned();
    Err(Error::Compressor { message })
}

/// Codec and settings of a manager created with [`Compressor::create`].
#[derive(Debug, Clone, Copy)]
pub struct Configured {
    codec: CodecOptions,
    options: ManagerOptions,
}

impl Configured {
    /// Checks that the chunk size is nonzero and a multiple of the codec's element size.
    ///
    /// nvCOMP checks the other settings when [`Compressor::create`] creates the manager.
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk size is zero or not a multiple of the codec's element size.
    pub fn create(codec: CodecOptions, options: ManagerOptions) -> Result<Self> {
        if options.chunk_size == 0 || !options.chunk_size.is_multiple_of(codec.element_size()) {
            return Err(Error::InvalidValue {
                name: "chunk_size".into(),
            });
        }
        Ok(Self { codec, options })
    }

    /// Returns the codec and its options.
    #[must_use]
    pub const fn codec(&self) -> CodecOptions {
        self.codec
    }

    /// Returns the manager settings.
    #[must_use]
    pub const fn options(&self) -> ManagerOptions {
        self.options
    }
}

/// Marks a manager whose codec was read from compressed data.
///
/// This crate does not know the detected codec's element size, so [`Compressor::<Detected>::compress`] is `unsafe`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Detected;

impl CodecOptions {
    /// Returns the size, in bytes, of one input element for compression.
    ///
    /// Input sizes must be a multiple of it.
    #[must_use]
    pub const fn element_size(self) -> usize {
        match self {
            Self::Lz4 { compress, .. } => compress.data_type.size_bytes(),
            Self::Ans { compress, .. } => compress.data_type.size_bytes(),
            Self::Bitcomp { compress, .. } => compress.data_type.size_bytes(),
            Self::Cascaded { compress, .. } => compress.common.data_type.size_bytes(),
            Self::Snappy { .. }
            | Self::Gdeflate { .. }
            | Self::Zstd { .. }
            | Self::Deflate { .. }
            | Self::Gzip { .. } => 1,
        }
    }
}

/// GPU manager that compresses and decompresses whole buffers on one stream.
///
/// It keeps its stream and CUDA context alive.
/// Every call waits for its GPU work before returning, so input buffers are free to reuse afterwards.
#[derive(Debug)]
pub struct Compressor<C = Configured> {
    owner: Arc<GpuManagerOwner>,
    stream: Arc<Stream>,
    configuration: C,
}

/// Native manager handle with its stream and configuration, from [`Compressor::into_raw`].
///
/// Dropping it leaks the native manager.
/// Pass it back to [`Compressor::from_raw`] to manage it again.
#[derive(Debug)]
pub struct RawCompressor<C = Configured> {
    /// Native manager handle, owned exclusively.
    pub handle: *mut ffi::SingeNvcompManager,
    /// Stream that the native manager uses.
    pub stream: Arc<Stream>,
    /// Codec configuration, which must match the native manager.
    pub configuration: C,
}

/// Compressed data from a GPU manager.
///
/// The allocation may be larger than the compressed data; only the compressed bytes are exposed.
#[derive(Debug)]
pub struct CompressedBuffer {
    memory: DeviceMemory<u8>,
}

impl CompressedBuffer {
    /// Returns the compressed bytes.
    #[must_use]
    pub fn memory(&self) -> &DeviceMemory<u8> {
        &self.memory
    }

    /// Returns the CUDA context of the allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.memory.context()
    }

    /// Returns the size of the compressed data in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// Returns whether the compressed data is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Returns the size of the whole allocation in bytes, including unused space after the compressed bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.memory.allocation().byte_len()
    }

    /// Returns the compressed bytes as device memory.
    #[must_use]
    pub fn into_memory(self) -> DeviceMemory<u8> {
        self.memory
    }
}

impl Compressor<Configured> {
    /// Creates a GPU manager on `stream` that uses `codec` with `options`.
    ///
    /// The manager allocates and frees its own temporary GPU memory.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the chunk size is zero or not a multiple of the codec's element size, the CUDA context cannot be bound, or nvCOMP rejects the settings or fails to create the manager.
    #[expect(
        clippy::too_many_lines,
        reason = "native operation construction keeps its resource ownership and descriptor ordering together"
    )]
    pub fn create(
        codec: CodecOptions,
        options: ManagerOptions,
        stream: &Arc<Stream>,
    ) -> Result<Self> {
        stream.ensure_not_capturing()?;
        let configuration = Configured::create(codec, options)?;
        stream.context().bind()?;
        let mut raw = ptr::null_mut();
        macro_rules! create {
            ($format:ident,$comp:expr,$decomp:expr,$comp_ty:ident,$decomp_ty:ident) => {{
                let compress: sys::$comp_ty = $comp.into();
                let decompress: sys::$decomp_ty = $decomp.into();
                // SAFETY: `singe_nvcomp_manager_create` receives live handles owned by the wrappers and valid output locations.
                bridge_result(unsafe {
                    ffi::singe_nvcomp_manager_create(
                        Format::$format.into_raw(),
                        options.chunk_size,
                        (&raw const compress).cast(),
                        (&raw const decompress).cast(),
                        stream.as_raw().cast(),
                        options.checksum.into_raw(),
                        options.execution.into_raw(),
                        options.bitstream.into_raw(),
                        &raw mut raw,
                    )
                })?;
            }};
        }
        match codec {
            CodecOptions::Lz4 {
                compress,
                decompress,
            } => create!(
                Lz4,
                compress,
                decompress,
                nvcompBatchedLZ4CompressOpts_t,
                nvcompBatchedLZ4DecompressOpts_t
            ),
            CodecOptions::Snappy {
                compress,
                decompress,
            } => create!(
                Snappy,
                compress,
                decompress,
                nvcompBatchedSnappyCompressOpts_t,
                nvcompBatchedSnappyDecompressOpts_t
            ),
            CodecOptions::Ans {
                compress,
                decompress,
            } => create!(
                Ans,
                compress,
                decompress,
                nvcompBatchedANSCompressOpts_t,
                nvcompBatchedANSDecompressOpts_t
            ),
            CodecOptions::Gdeflate {
                compress,
                decompress,
            } => create!(
                Gdeflate,
                compress,
                decompress,
                nvcompBatchedGdeflateCompressOpts_t,
                nvcompBatchedGdeflateDecompressOpts_t
            ),
            CodecOptions::Cascaded {
                compress,
                decompress,
            } => create!(
                Cascaded,
                compress,
                decompress,
                nvcompBatchedCascadedCompressOpts_t,
                nvcompBatchedCascadedDecompressOpts_t
            ),
            CodecOptions::Bitcomp {
                compress,
                decompress,
            } => create!(
                Bitcomp,
                compress,
                decompress,
                nvcompBatchedBitcompCompressOpts_t,
                nvcompBatchedBitcompDecompressOpts_t
            ),
            CodecOptions::Zstd {
                compress,
                decompress,
            } => create!(
                Zstd,
                compress,
                decompress,
                nvcompBatchedZstdCompressOpts_t,
                nvcompBatchedZstdDecompressOpts_t
            ),
            CodecOptions::Deflate {
                compress,
                decompress,
            } => create!(
                Deflate,
                compress,
                decompress,
                nvcompBatchedDeflateCompressOpts_t,
                nvcompBatchedDeflateDecompressOpts_t
            ),
            CodecOptions::Gzip {
                compress,
                decompress,
            } => create!(
                Gzip,
                compress,
                decompress,
                nvcompBatchedGzipCompressOpts_t,
                nvcompBatchedGzipDecompressOpts_t
            ),
        }
        Ok(Self {
            owner: Arc::new(GpuManagerOwner::create(raw, stream.context())?),
            stream: Arc::clone(stream),
            configuration,
        })
    }

    /// Compresses `input` and waits for the result.
    ///
    /// The work finishes before this returns, even when it fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the input size is not a multiple of the codec's element size, stream capture is active or invalidated, `input` belongs to another CUDA context or is misaligned, a CUDA call or allocation fails, or nvCOMP fails or reports an error status.
    pub fn compress<B: DeviceSlice<u8> + ?Sized>(&mut self, input: &B) -> Result<CompressedBuffer> {
        if !input
            .len()
            .is_multiple_of(self.configuration.codec.element_size())
        {
            return Err(Error::InvalidValue {
                name: "compression input element count".into(),
            });
        }
        // SAFETY: creation checks internal chunk sizes and the input has complete scalars.
        unsafe { self.compress_with_layout(input) }
    }
}

impl Compressor<Detected> {
    /// Creates a GPU manager for the codec recorded in the header of `buffer`.
    ///
    /// This synchronizes `stream`.
    /// The manager does not borrow `buffer` afterwards.
    ///
    /// # Safety
    ///
    /// `buffer` must hold complete, valid data written by an nvCOMP manager with [`BitstreamKind::Native`].
    ///
    /// # Errors
    ///
    /// Returns an error if `buffer` is empty or in another CUDA context; stream capture is active or invalidated; a CUDA call fails; or nvCOMP cannot read the header, rejects `options`, or fails to create the manager.
    pub unsafe fn from_compressed<B: DeviceSlice<u8> + ?Sized>(
        buffer: &B,
        options: DetectionOptions,
        stream: &Arc<Stream>,
    ) -> Result<Self> {
        stream.ensure_not_capturing()?;
        require_context(buffer, stream)?;
        validate_input_size(buffer.len(), buffer.len())?;
        let mut raw = ptr::null_mut();
        stream.sync_scope(|_| {
            Ok((|| -> Result<()> {
                stream.join_regions(&[buffer.region()])?;
                // SAFETY: the caller upholds this function's safety contract.
                bridge_result(unsafe {
                    ffi::singe_nvcomp_manager_from_buffer(
                        buffer.as_device_ptr(),
                        stream.as_raw().cast(),
                        0,
                        0,
                        options.checksum.into_raw(),
                        options.execution.into_raw(),
                        options.backend as i32,
                        options.sort_before_hardware_decompression.into(),
                        options.algorithm as i32,
                        &raw mut raw,
                    )
                })?;
                Ok(())
            })())
        })??;
        Ok(Self {
            owner: Arc::new(GpuManagerOwner::create(raw, stream.context())?),
            stream: Arc::clone(stream),
            configuration: Detected,
        })
    }

    /// Compresses `input` with the codec read from compressed data, and waits for the result.
    ///
    /// The work finishes before this returns, even when it fails.
    ///
    /// # Safety
    ///
    /// `input` must meet the detected codec's requirements, such as a size that is a multiple of its element size.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, `input` belongs to another CUDA context or is misaligned, a CUDA call or allocation fails, or nvCOMP fails or reports an error status.
    pub unsafe fn compress<B: DeviceSlice<u8> + ?Sized>(
        &mut self,
        input: &B,
    ) -> Result<CompressedBuffer> {
        // SAFETY: caller establishes the inferred layout's scalar requirements.
        unsafe { self.compress_with_layout(input) }
    }
}

impl<C> Compressor<C> {
    /// Returns the CUDA context of the manager's stream.
    pub fn context(&self) -> &Arc<CudaContext> {
        self.stream.context()
    }

    /// Returns the stream the manager runs on.
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Returns the codec configuration, or [`Detected`] for a manager created from compressed data.
    pub fn configuration(&self) -> &C {
        &self.configuration
    }

    /// Returns the native manager handle, which this wrapper still owns.
    ///
    /// Calls through the handle must use the manager's stream and must not run concurrently with calls through this wrapper.
    pub fn as_raw(&self) -> *mut ffi::SingeNvcompManager {
        self.owner.handle.as_raw()
    }

    /// Takes ownership of a native manager handle from [`into_raw`](Self::into_raw).
    ///
    /// # Safety
    ///
    /// The handle must be live, owned by nobody else, and use `parts.stream`.
    /// `parts.configuration` must match the manager's codec and chunk size.
    /// Each handle may be passed to this function only once.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null.
    pub unsafe fn from_raw(parts: RawCompressor<C>) -> Result<Self> {
        let owner = Arc::new(GpuManagerOwner::create(
            parts.handle,
            parts.stream.context(),
        )?);
        Ok(Self {
            owner,
            stream: parts.stream,
            configuration: parts.configuration,
        })
    }

    /// Releases the native manager handle with its stream and configuration.
    ///
    /// Pass the parts back to [`from_raw`](Self::from_raw) or destroy the handle yourself; dropping them leaks it.
    ///
    /// # Errors
    ///
    /// Returns the manager unchanged if something else still shares the native handle, such as cleanup for a captured graph.
    pub fn into_raw(self) -> StdResult<RawCompressor<C>, Self> {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: move the owner once while suppressing Compressor's destruction.
        let owner = unsafe { ptr::read(&raw const this.owner) };
        match Arc::try_unwrap(owner) {
            Ok(owner) => {
                // SAFETY: move remaining owning fields once; raw ownership retains the stream.
                let stream = unsafe { ptr::read(&raw const this.stream) };
                let configuration = unsafe { ptr::read(&raw const this.configuration) };
                Ok(RawCompressor {
                    handle: owner.into_raw(),
                    stream,
                    configuration,
                })
            }
            Err(owner) => {
                // SAFETY: restore the moved field without dropping its stale bits.
                unsafe {
                    ptr::write(&raw mut this.owner, owner);
                }
                Err(ManuallyDrop::into_inner(this))
            }
        }
    }

    /// Returns the minimum buffer alignments for compression.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, or nvCOMP fails the query.
    pub fn compression_alignments(&mut self) -> Result<AlignmentRequirements> {
        self.stream.ensure_not_capturing()?;
        self.stream.context().bind()?;
        self.owner.handle.alignments(false)
    }

    /// Returns the minimum buffer alignments for decompression.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, or nvCOMP fails the query.
    pub fn decompression_alignments(&mut self) -> Result<AlignmentRequirements> {
        self.stream.ensure_not_capturing()?;
        self.stream.context().bind()?;
        self.owner.handle.alignments(true)
    }

    /// Returns the largest size, in bytes, that `input_bytes` bytes can compress to.
    ///
    /// Use it to size the output buffer; the actual compressed size is usually smaller.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, or nvCOMP fails to configure compression.
    pub fn compression_bound(&mut self, input_bytes: usize) -> Result<usize> {
        self.stream.ensure_not_capturing()?;
        self.stream.context().bind()?;
        self.owner.handle.compression_bound(input_bytes)
    }

    // Native scalar-layout requirements are checked by Configured or the raw
    // Detected caller. Owner, context, alignment, and completion handling is shared.
    unsafe fn compress_with_layout<B: DeviceSlice<u8> + ?Sized>(
        &mut self,
        input: &B,
    ) -> Result<CompressedBuffer> {
        self.stream.ensure_not_capturing()?;
        require_context(input, &self.stream)?;
        let alignment = self.owner.handle.alignments(false)?;
        validate_alignment(
            "manager input",
            alignment.input <= 1 || input.as_device_ptr().addr().is_multiple_of(alignment.input),
            alignment.input,
        )?;
        let capacity = self.owner.handle.compression_bound(input.len())?;
        let memory = DeviceMemory::<u8>::create_uninit(self.context(), capacity)?;
        validate_alignment(
            "manager output",
            alignment.output <= 1 || memory.as_ptr().addr().is_multiple_of(alignment.output),
            alignment.output,
        )?;
        let mut len = 0;
        self.stream.sync_scope(|_| {
            Ok((|| -> Result<()> {
                self.stream.join_regions(&[input.region()])?;
                // SAFETY: the caller upholds this function's safety contract.
                bridge_result(unsafe {
                    ffi::singe_nvcomp_compress(
                        self.owner.handle.as_raw(),
                        input.as_device_ptr(),
                        input.len(),
                        memory.as_mut_ptr().cast(),
                        capacity,
                        &raw mut len,
                    )
                })
            })())
        })??;
        if len > capacity {
            return Err(Error::LengthMismatch {
                name: "compressed manager bytes".into(),
                expected: capacity,
                actual: len,
            });
        }

        // SAFETY: checked native success and completion establish exactly this prefix.
        let memory = unsafe { memory.assume_init_prefix(len)? };
        Ok(CompressedBuffer { memory })
    }

    /// Returns the uncompressed size, in bytes, recorded in `input`.
    ///
    /// This reads the header without decompressing, so it does not check that the data is valid.
    /// It waits for the stream before returning.
    ///
    /// # Safety
    ///
    /// `input` must hold exactly one complete, valid compressed buffer for this manager's codec and options.
    /// To query part of a larger buffer, pass a view of just that part.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is empty, misaligned, or in another CUDA context; stream capture is active or invalidated; a CUDA call fails; or nvCOMP fails to configure decompression.
    pub unsafe fn decompressed_size<B: DeviceSlice<u8> + ?Sized>(
        &mut self,
        input: &B,
    ) -> Result<usize> {
        self.stream.ensure_not_capturing()?;
        require_context(input, &self.stream)?;
        validate_input_size(input.len(), input.len())?;
        let alignment = self.owner.handle.alignments(true)?;
        validate_alignment(
            "manager input",
            alignment.input <= 1 || input.as_device_ptr().addr().is_multiple_of(alignment.input),
            alignment.input,
        )?;
        let mut size = 0;
        self.stream.sync_scope(|_| {
            Ok((|| -> Result<()> {
                self.stream.join_regions(&[input.region()])?;
                // SAFETY: the caller upholds this function's safety contract.
                bridge_result(unsafe {
                    ffi::singe_nvcomp_decompress_size(
                        self.owner.handle.as_raw(),
                        input.as_device_ptr(),
                        input.len(),
                        &raw mut size,
                    )
                })
            })())
        })??;
        Ok(size)
    }

    /// Decompresses `input` into a new allocation and waits for the result.
    ///
    /// The allocation is sized from the uncompressed size in the header.
    ///
    /// # Safety
    ///
    /// `input` must hold exactly one complete, valid compressed buffer for this manager's codec and options.
    ///
    /// # Errors
    ///
    /// Returns an error if [`decompressed_size`](Self::decompressed_size) fails, a CUDA call or allocation fails, the output is misaligned, nvCOMP fails or reports an error status, or the decompressed size differs from the header.
    pub unsafe fn decompress<B: DeviceSlice<u8> + ?Sized>(
        &mut self,
        input: &B,
    ) -> Result<DeviceMemory<u8>> {
        // SAFETY: caller establishes bitstream validity for this manager.
        let capacity = unsafe { self.decompressed_size(input)? };
        let output = DeviceMemory::<u8>::create_uninit(self.context(), capacity)?;
        let alignment = self.owner.handle.alignments(true)?;
        validate_alignment(
            "manager output",
            alignment.output <= 1 || output.as_ptr().addr().is_multiple_of(alignment.output),
            alignment.output,
        )?;
        let mut written = 0;
        self.stream.sync_scope(|_| {
            Ok((|| -> Result<()> {
                self.stream.join_regions(&[input.region()])?;
                // SAFETY: the caller upholds this function's safety contract.
                bridge_result(unsafe {
                    ffi::singe_nvcomp_decompress(
                        self.owner.handle.as_raw(),
                        input.as_device_ptr(),
                        input.len(),
                        output.as_mut_ptr().cast(),
                        capacity,
                        &raw mut written,
                    )
                })
            })())
        })??;
        if written > capacity {
            return Err(Error::LengthMismatch {
                name: "decompressed manager bytes".into(),
                expected: capacity,
                actual: written,
            });
        }
        // SAFETY: checked native success and completion establish exactly this prefix.
        Ok(unsafe { output.assume_init_prefix(written)? })
    }

    /// Frees the manager's temporary GPU memory after waiting for its stream.
    ///
    /// Buffers returned from compression or decompression are not affected.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, a CUDA call fails, or nvCOMP fails to free the memory.
    pub fn deallocate_gpu_memory(&mut self) -> Result<()> {
        self.stream.ensure_not_capturing()?;
        self.stream.context().bind()?;
        // SAFETY: `singe_nvcomp_deallocate` receives live handles owned by the wrappers and valid output locations.
        bridge_result(unsafe { ffi::singe_nvcomp_deallocate(self.owner.handle.as_raw()) })
    }
}

impl<C> Drop for Compressor<C> {
    fn drop(&mut self) {
        // All typed operations are eager. An unrelated active capture must not
        // force a capture-incompatible native destructor or invalidate capture.
        // The deferred owner retains only the context; this stream owns cleanup.
        if self
            .stream
            .defer_drop_until_capture_end(Arc::clone(&self.owner))
            .is_err()
        {
            process::abort();
        }
    }
}

fn require_context<B: DeviceSlice<u8> + ?Sized>(input: &B, stream: &Stream) -> Result<()> {
    if input.context() != stream.context() {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    stream.context().bind()?;
    Ok(())
}

fn validate_input_size(capacity: usize, bytes: usize) -> Result<()> {
    if bytes == 0 || bytes > capacity {
        return Err(Error::InvalidValue {
            name: "compressed input size".into(),
        });
    }
    Ok(())
}

/// CPU manager that compresses and decompresses whole buffers with LZ4 in nvCOMP's format.
///
/// It reads and writes the same format as a GPU manager with [`BitstreamKind::Native`].
/// It cannot compute or verify checksums, and skips any checksums in the data it decompresses.
#[derive(Debug)]
pub struct CpuCompressor {
    handle: Handle,
}

impl CpuCompressor {
    /// Creates a CPU LZ4 manager that compresses in chunks of `chunk_size` bytes.
    ///
    /// nvCOMP's defaults are a chunk size of 65,536 bytes and compression level 9. A `threads` value of 0 uses every available hardware thread.
    ///
    /// # Errors
    ///
    /// Returns an error if `chunk_size` is zero, nvCOMP rejects the settings, or nvCOMP fails to create the manager.
    pub fn create(chunk_size: usize, compression_level: i32, threads: u32) -> Result<Self> {
        if chunk_size == 0 {
            return Err(Error::InvalidValue {
                name: "chunk_size".into(),
            });
        }

        let mut raw = ptr::null_mut();
        // SAFETY: `singe_nvcomp_cpu_create` receives live handles owned by the wrappers and valid output locations.
        bridge_result(unsafe {
            ffi::singe_nvcomp_cpu_create(chunk_size, compression_level, threads, &raw mut raw)
        })?;
        Ok(Self {
            handle: Handle::create(raw)?,
        })
    }

    /// Creates a CPU manager for the codec recorded in the header of `input`.
    ///
    /// A `threads` value of 0 uses every available hardware thread.
    /// The manager does not borrow `input` afterwards.
    ///
    /// # Safety
    ///
    /// `input` must hold complete, valid data in nvCOMP's format that the CPU manager supports.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is empty, or nvCOMP cannot read the header or fails to create the manager.
    pub unsafe fn from_compressed(input: &[u8], threads: u32) -> Result<Self> {
        validate_input_size(input.len(), input.len())?;
        let mut raw = ptr::null_mut();
        // SAFETY: the caller upholds this function's safety contract.
        bridge_result(unsafe {
            ffi::singe_nvcomp_manager_from_buffer(
                input.as_ptr(),
                ptr::null_mut(),
                1,
                threads,
                ChecksumPolicy::NoComputeNoVerify.into_raw(),
                ExecutionPolicy::Concurrency.into_raw(),
                0,
                0,
                0,
                &raw mut raw,
            )
        })?;
        Ok(Self {
            handle: Handle::create(raw)?,
        })
    }

    /// Compresses `input` and returns the compressed bytes.
    ///
    /// The input is first copied into a buffer with the alignment nvCOMP requires.
    ///
    /// # Errors
    ///
    /// Returns an error if nvCOMP fails or reports an error status, an alignment is invalid, or a buffer size overflows.
    pub fn compress(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        let capacity = self.handle.compression_bound(input.len())?;
        let alignments = self.handle.alignments(false)?;
        let input = AlignedBytes::from_slice(input, alignments.input)?;
        let mut output = AlignedBytes::create(capacity, alignments.output)?;
        let mut len = 0;
        // SAFETY: `singe_nvcomp_compress` receives live handles owned by the wrappers and valid output locations.
        bridge_result(unsafe {
            ffi::singe_nvcomp_compress(
                self.handle.as_raw(),
                input.as_slice().as_ptr(),
                input.len,
                output.as_mut_slice().as_mut_ptr(),
                capacity,
                &raw mut len,
            )
        })?;
        Ok(output.as_slice()[..len].to_vec())
    }

    /// Decompresses `input` and returns the decompressed bytes.
    ///
    /// The input is first copied into a buffer with the alignment nvCOMP requires.
    ///
    /// # Safety
    ///
    /// `input` must hold complete, valid data in nvCOMP's format for this manager's codec.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is empty, nvCOMP fails or reports an error status, an alignment is invalid, or a buffer size overflows.
    pub unsafe fn decompress(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        validate_input_size(input.len(), input.len())?;
        let alignments = self.handle.alignments(true)?;
        let input = AlignedBytes::from_slice(input, alignments.input)?;
        let mut capacity = 0;
        // SAFETY: the caller upholds this function's safety contract.
        bridge_result(unsafe {
            ffi::singe_nvcomp_decompress_size(
                self.handle.as_raw(),
                input.as_slice().as_ptr(),
                input.len,
                &raw mut capacity,
            )
        })?;
        let mut output = AlignedBytes::create(capacity, alignments.output)?;
        let mut written = 0;
        // SAFETY: the caller upholds this function's safety contract.
        bridge_result(unsafe {
            ffi::singe_nvcomp_decompress(
                self.handle.as_raw(),
                input.as_slice().as_ptr(),
                input.len,
                output.as_mut_slice().as_mut_ptr(),
                capacity,
                &raw mut written,
            )
        })?;
        Ok(output.as_slice()[..written].to_vec())
    }
}

/// Returns the codec recorded in the header of `input`, which is in host memory.
///
/// # Safety
///
/// `input` must hold complete, valid data written by an nvCOMP manager with [`BitstreamKind::Native`].
///
/// # Errors
///
/// Returns an error if `input` is empty, nvCOMP cannot read the header, or the codec is not a [`Format`] variant.
pub unsafe fn cpu_compression_format(input: &[u8]) -> Result<Format> {
    validate_input_size(input.len(), input.len())?;
    let mut format = 0;
    // SAFETY: the caller upholds this function's safety contract.
    bridge_result(unsafe {
        ffi::singe_nvcomp_format(input.as_ptr(), ptr::null_mut(), 1, &raw mut format)
    })?;
    format_from_raw(format)
}

/// Returns the codec recorded in the header of `input`, which is in device memory.
///
/// This synchronizes the stream.
///
/// # Safety
///
/// `input` must hold complete, valid data written by an nvCOMP manager with [`BitstreamKind::Native`].
///
/// # Errors
///
/// Returns an error if `input` is empty or belongs to another CUDA context, stream capture is active or invalidated, a CUDA call fails, nvCOMP cannot read the header, or the codec is not a [`Format`] variant.
pub unsafe fn compression_format<B: DeviceSlice<u8> + ?Sized>(
    input: &B,
    stream: &Stream,
) -> Result<Format> {
    stream.ensure_not_capturing()?;
    require_context(input, stream)?;
    validate_input_size(input.len(), input.len())?;
    let mut format = 0;
    stream.sync_scope(|_| {
        Ok((|| -> Result<()> {
            stream.join_regions(&[input.region()])?;
            // SAFETY: caller establishes a complete valid HLIF prefix; owners stay borrowed.
            bridge_result(unsafe {
                ffi::singe_nvcomp_format(
                    input.as_device_ptr(),
                    stream.as_raw().cast(),
                    0,
                    &raw mut format,
                )
            })
        })())
    })??;
    format_from_raw(format)
}

fn format_from_raw(raw: i32) -> Result<Format> {
    match raw {
        0 => Ok(Format::Lz4),
        1 => Ok(Format::Snappy),
        2 => Ok(Format::Ans),
        3 => Ok(Format::Gdeflate),
        4 => Ok(Format::Cascaded),
        5 => Ok(Format::Bitcomp),
        6 => Ok(Format::Zstd),
        7 => Ok(Format::Deflate),
        8 => Ok(Format::Gzip),
        _ => Err(Error::InvalidValue {
            name: "compression format".into(),
        }),
    }
}

struct AlignedBytes {
    storage: Vec<u8>,
    offset: usize,
    len: usize,
}

impl AlignedBytes {
    fn create(len: usize, alignment: usize) -> Result<Self> {
        let alignment = alignment.max(1);
        if !alignment.is_power_of_two() {
            return Err(Error::InvalidValue {
                name: "manager alignment".into(),
            });
        }

        let capacity = len
            .checked_add(alignment)
            .ok_or_else(|| Error::OutOfRange {
                name: "aligned buffer size".into(),
            })?;
        let storage = vec![0u8; capacity];
        let offset = storage.as_ptr().align_offset(alignment);
        Ok(Self {
            storage,
            offset,
            len,
        })
    }

    fn from_slice(input: &[u8], alignment: usize) -> Result<Self> {
        let mut value = Self::create(input.len(), alignment)?;
        value.as_mut_slice().copy_from_slice(input);
        Ok(value)
    }

    fn as_slice(&self) -> &[u8] {
        &self.storage[self.offset..self.offset + self.len]
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.storage[self.offset..self.offset + self.len]
    }
}
