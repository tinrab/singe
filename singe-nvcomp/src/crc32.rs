use std::ffi::c_void;

use singe_cuda::{stream::Stream, types::DevicePtr, view::DeviceOutput};
use singe_nvcomp_sys as sys;

use crate::{
    batch::{AsyncBatchResult, DeviceChunkTable, DeviceChunkTableView, DeviceStatuses},
    error::{Error, Result},
    try_ffi,
    types::{Crc32KernelKind, Crc32SegmentKind},
    utility::{codec_size, to_i32, to_u64, validate_same_len},
};

/// Value of `max_input_chunk_bytes` that makes [`heuristic_conf`] read the largest chunk size from the device size table.
///
/// It is 0.
pub const DEDUCED_MAX_INPUT_CHUNK_BYTES: usize =
    codec_size(sys::nvcompCRC32DeducedMaxInputChunkBytes);

/// Chunk sizes that a heuristic kernel-configuration query uses.
#[derive(Debug, Clone, Copy)]
pub enum HeuristicInput<'a> {
    /// Known chunk count and largest chunk size, so nvCOMP reads no device memory.
    MaxChunkBytes {
        /// Number of chunks to compute checksums of.
        num_chunks: usize,
        /// Size of the largest chunk in bytes.
        ///
        /// It must not be [`DEDUCED_MAX_INPUT_CHUNK_BYTES`].
        max_input_chunk_bytes: usize,
    },
    /// Chunk table whose device size array nvCOMP reads.
    ChunkSizes(DeviceChunkTableView<'a>),
}

impl<'a> HeuristicInput<'a> {
    /// Creates a [`MaxChunkBytes`](Self::MaxChunkBytes) input.
    #[must_use]
    pub const fn max_chunk_bytes(num_chunks: usize, max_input_chunk_bytes: usize) -> Self {
        Self::MaxChunkBytes {
            num_chunks,
            max_input_chunk_bytes,
        }
    }

    /// Creates a [`ChunkSizes`](Self::ChunkSizes) input.
    #[must_use]
    pub const fn chunk_sizes(input: DeviceChunkTableView<'a>) -> Self {
        Self::ChunkSizes(input)
    }
}

/// CRC32 parameters: polynomial, initial value, reflections, and final XOR.
///
/// The associated constants cover common CRC32 variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Spec {
    /// Generator polynomial.
    pub poly: u32,
    /// Initial value of the shift register.
    pub init: u32,
    /// Whether to reflect each input byte.
    pub ref_in: bool,
    /// Whether to reflect the final value before the XOR with [`xorout`](Self::xorout).
    pub ref_out: bool,
    /// Value combined with the final result by bitwise XOR.
    pub xorout: u32,
}

impl Spec {
    /// Standard CRC32, also known as CRC-32/PKZIP.
    pub const CRC32: Self = Self::from_raw(sys::nvcompCRC32);

    /// CRC-32/ISCSI.
    pub const CRC32_C: Self = Self::from_raw(sys::nvcompCRC32_C);

    /// CRC-32/BASE91-D.
    pub const CRC32_D: Self = Self::from_raw(sys::nvcompCRC32_D);

    /// CRC-32/AIXM.
    pub const CRC32_Q: Self = Self::from_raw(sys::nvcompCRC32_Q);

    /// CRC-32/MEF.
    pub const CRC32_MEF: Self = Self::from_raw(sys::nvcompCRC32_MEF);

    /// CRC-32/XFER.
    pub const CRC32_XFER: Self = Self::from_raw(sys::nvcompCRC32_XFER);

    /// CRC-32/BZIP2, also known as CRC-32/AAL-5.
    pub const CRC32_BZIP2: Self = Self::from_raw(sys::nvcompCRC32_BZIP2);

    /// CRC-32/POSIX, also known as CRC-32/CKSUM.
    pub const CRC32_POSIX: Self = Self::from_raw(sys::nvcompCRC32_POSIX);

    /// CRC-32/JAMCRC.
    pub const CRC32_JAMCRC: Self = Self::from_raw(sys::nvcompCRC32_JAMCRC);

    /// CRC-32/MPEG-2.
    pub const CRC32_MPEG_2: Self = Self::from_raw(sys::nvcompCRC32_MPEG_2);

    /// CRC-32/AUTOSAR.
    pub const CRC32_AUTOSAR: Self = Self::from_raw(sys::nvcompCRC32_AUTOSAR);

    /// CRC-32/CD-ROM-EDC.
    pub const CRC32_CD_ROM_EDC: Self = Self::from_raw(sys::nvcompCRC32_CD_ROM_EDC);

    /// Creates a value from the native nvCOMP struct.
    #[must_use]
    pub const fn from_raw(raw: sys::nvcompCRC32Spec_t) -> Self {
        Self {
            poly: raw.poly,
            init: raw.init,
            ref_in: raw.ref_in,
            ref_out: raw.ref_out,
            xorout: raw.xorout,
        }
    }

    /// Returns the native nvCOMP struct.
    #[must_use]
    pub const fn as_raw(self) -> sys::nvcompCRC32Spec_t {
        sys::nvcompCRC32Spec_t {
            poly: self.poly,
            init: self.init,
            ref_in: self.ref_in,
            ref_out: self.ref_out,
            xorout: self.xorout,
            reserved: [0; 16],
        }
    }

    /// Converts into the native nvCOMP struct.
    #[must_use]
    pub const fn into_raw(self) -> sys::nvcompCRC32Spec_t {
        self.as_raw()
    }
}

impl Default for Spec {
    fn default() -> Self {
        Self::CRC32
    }
}

impl From<sys::nvcompCRC32Spec_t> for Spec {
    fn from(raw: sys::nvcompCRC32Spec_t) -> Self {
        Self::from_raw(raw)
    }
}

impl From<Spec> for sys::nvcompCRC32Spec_t {
    fn from(value: Spec) -> Self {
        value.into_raw()
    }
}

/// CRC32 kernel launch configuration.
///
/// Get one from [`heuristic_conf_for_chunks`] or [`search_conf`], or build it with [`create`](Self::create).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelConf {
    kernel_kind: Crc32KernelKind,
    bytes_per_read: i32,
    blocks_per_msg: i32,
}

impl KernelConf {
    /// Creates a kernel configuration.
    ///
    /// `bytes_per_read` is the number of bytes each thread reads per step.
    /// `blocks_per_msg` is the number of thread blocks per message, used only with [`Crc32KernelKind::Block`].
    ///
    /// # Errors
    ///
    /// Returns an error if `bytes_per_read` or `blocks_per_msg` does not fit an `i32`.
    pub fn create(
        kernel_kind: Crc32KernelKind,
        bytes_per_read: usize,
        blocks_per_msg: usize,
    ) -> Result<Self> {
        Ok(Self {
            kernel_kind,
            bytes_per_read: to_i32(bytes_per_read, "bytes_per_read")?,
            blocks_per_msg: to_i32(blocks_per_msg, "blocks_per_msg")?,
        })
    }

    /// Creates a value from the native nvCOMP struct.
    #[must_use]
    pub fn from_raw(raw: sys::nvcompCRC32KernelConf_t) -> Self {
        Self {
            kernel_kind: raw.kernel_kind.into(),
            bytes_per_read: raw.bytes_per_read,
            blocks_per_msg: raw.blocks_per_msg,
        }
    }

    /// Returns how the kernel divides work among threads.
    #[must_use]
    pub const fn kernel_kind(self) -> Crc32KernelKind {
        self.kernel_kind
    }

    /// Returns the number of bytes each thread reads per step.
    #[must_use]
    pub const fn bytes_per_read(self) -> i32 {
        self.bytes_per_read
    }

    /// Returns the number of thread blocks per message.
    ///
    /// Only [`Crc32KernelKind::Block`] uses it.
    #[must_use]
    pub const fn blocks_per_msg(self) -> i32 {
        self.blocks_per_msg
    }

    /// Returns the native nvCOMP struct.
    #[must_use]
    pub const fn as_raw(self) -> sys::nvcompCRC32KernelConf_t {
        sys::nvcompCRC32KernelConf_t {
            kernel_kind: match self.kernel_kind {
                Crc32KernelKind::Warp => sys::nvcompCRC32KernelKind_t::nvcompCRC32WarpKernel,
                Crc32KernelKind::Block => sys::nvcompCRC32KernelKind_t::nvcompCRC32BlockKernel,
            },
            bytes_per_read: self.bytes_per_read,
            blocks_per_msg: self.blocks_per_msg,
            reserved: [0; 20],
        }
    }

    /// Converts into the native nvCOMP struct.
    #[must_use]
    pub const fn into_raw(self) -> sys::nvcompCRC32KernelConf_t {
        self.as_raw()
    }
}

impl From<sys::nvcompCRC32KernelConf_t> for KernelConf {
    fn from(raw: sys::nvcompCRC32KernelConf_t) -> Self {
        Self::from_raw(raw)
    }
}

impl From<KernelConf> for sys::nvcompCRC32KernelConf_t {
    fn from(value: KernelConf) -> Self {
        value.into_raw()
    }
}

/// CRC32 computation options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Options {
    /// CRC32 variant to compute.
    pub spec: Spec,
    /// Kernel launch configuration.
    pub kernel_conf: KernelConf,
}

impl Options {
    /// Creates options for `spec` with `kernel_conf`.
    #[must_use]
    pub const fn new(spec: Spec, kernel_conf: KernelConf) -> Self {
        Self { spec, kernel_conf }
    }

    /// Creates a value from the native nvCOMP struct.
    #[must_use]
    pub fn from_raw(raw: sys::nvcompBatchedCRC32Opts_t) -> Self {
        Self {
            spec: raw.spec.into(),
            kernel_conf: raw.kernel_conf.into(),
        }
    }

    /// Returns the native nvCOMP struct.
    #[must_use]
    pub const fn as_raw(self) -> sys::nvcompBatchedCRC32Opts_t {
        sys::nvcompBatchedCRC32Opts_t {
            spec: self.spec.as_raw(),
            kernel_conf: self.kernel_conf.as_raw(),
            reserved: [0; 64],
        }
    }

    /// Converts into the native nvCOMP struct.
    #[must_use]
    pub const fn into_raw(self) -> sys::nvcompBatchedCRC32Opts_t {
        self.as_raw()
    }
}

impl From<sys::nvcompBatchedCRC32Opts_t> for Options {
    fn from(raw: sys::nvcompBatchedCRC32Opts_t) -> Self {
        Self::from_raw(raw)
    }
}

impl From<Options> for sys::nvcompBatchedCRC32Opts_t {
    fn from(value: Options) -> Self {
        value.into_raw()
    }
}

/// Returns a kernel configuration suited to `num_chunks` chunks of at most `max_input_chunk_bytes` each.
///
/// The result depends on the GPU, the chunk count, and the largest chunk size.
/// It suits later batches whose chunks have similar sizes and counts.
/// nvCOMP reads no device memory, so it ignores `stream`.
///
/// # Errors
///
/// Returns an error if `max_input_chunk_bytes` is [`DEDUCED_MAX_INPUT_CHUNK_BYTES`], a size does not fit nvCOMP's integer types, or nvCOMP rejects the query.
pub fn heuristic_conf_for_max_chunk_size(
    num_chunks: usize,
    max_input_chunk_bytes: usize,
    stream: &Stream,
) -> Result<KernelConf> {
    if max_input_chunk_bytes == DEDUCED_MAX_INPUT_CHUNK_BYTES {
        return Err(Error::InvalidValue {
            name: "max_input_chunk_bytes".into(),
        });
    }
    // SAFETY: this branch provides host bounds and no device-side size pointer.
    unsafe {
        heuristic_conf_for_input(
            HeuristicInput::max_chunk_bytes(num_chunks, max_input_chunk_bytes),
            stream,
        )
    }
}

/// Returns a kernel configuration suited to the chunk count and largest chunk of `input`.
///
/// This calls [`heuristic_conf_for_max_chunk_size`] with sizes from the host side of `input`.
///
/// # Errors
///
/// Returns an error if [`heuristic_conf_for_max_chunk_size`] fails.
pub fn heuristic_conf_for_chunks(
    input: &DeviceChunkTable<'_>,
    stream: &Stream,
) -> Result<KernelConf> {
    heuristic_conf_for_max_chunk_size(input.len(), input.max_chunk_bytes(), stream)
}

/// Returns a kernel configuration suited to `input`, with the largest chunk size read from its device size array.
///
/// The query synchronizes `stream`, or the whole device if it does not support stream-ordered memory allocation.
///
/// # Safety
///
/// The device size array of `input` must be initialized, with any writes to it ordered before this call.
///
/// # Errors
///
/// Returns an error if `input` belongs to another CUDA context, a size does not fit nvCOMP's integer types, or nvCOMP rejects the query.
pub unsafe fn heuristic_conf_from_sizes(
    input: DeviceChunkTableView<'_>,
    stream: &Stream,
) -> Result<KernelConf> {
    // SAFETY: caller establishes initialized, ordered device-side sizes.
    unsafe { heuristic_conf_for_input(HeuristicInput::chunk_sizes(input), stream) }
}

/// Returns a kernel configuration suited to `input`.
///
/// See [`heuristic_conf_for_max_chunk_size`] and [`heuristic_conf_from_sizes`] for the two kinds of input.
///
/// # Safety
///
/// For [`ChunkSizes`](HeuristicInput::ChunkSizes), the device size array must be initialized, with any writes to it ordered before this call.
/// For [`MaxChunkBytes`](HeuristicInput::MaxChunkBytes), the largest chunk size must not be [`DEDUCED_MAX_INPUT_CHUNK_BYTES`].
///
/// # Errors
///
/// Returns an error if a [`ChunkSizes`](HeuristicInput::ChunkSizes) table belongs to another CUDA context, a size does not fit nvCOMP's integer types, or nvCOMP rejects the query.
pub unsafe fn heuristic_conf_for_input(
    input: HeuristicInput<'_>,
    stream: &Stream,
) -> Result<KernelConf> {
    match input {
        HeuristicInput::MaxChunkBytes {
            num_chunks,
            max_input_chunk_bytes,
            // SAFETY: the caller upholds this function's safety contract.
        } => unsafe {
            heuristic_conf(DevicePtr::null(), num_chunks, max_input_chunk_bytes, stream)
        },
        HeuristicInput::ChunkSizes(input) => {
            input.validate_context(stream)?;
            // SAFETY: table construction establishes the size array's bounds.
            // The shared draining scope retains its borrow on native errors.
            stream.sync_scope(|_| {
                Ok(unsafe {
                    heuristic_conf(
                        input.device_chunk_bytes(),
                        input.len(),
                        DEDUCED_MAX_INPUT_CHUNK_BYTES,
                        stream,
                    )
                })
            })?
        }
    }
}

/// Returns a kernel configuration suited to chunks of similar size.
///
/// Pass either the largest chunk size in `max_input_chunk_bytes` and a null `device_input_chunk_bytes`, or [`DEDUCED_MAX_INPUT_CHUNK_BYTES`] and a device array of `num_chunks` chunk sizes.
/// The query blocks the host.
/// With a device array, it synchronizes `stream`, or the whole device if it does not support stream-ordered memory allocation; otherwise it ignores `stream`.
///
/// # Safety
///
/// When `max_input_chunk_bytes` is [`DEDUCED_MAX_INPUT_CHUNK_BYTES`], `device_input_chunk_bytes` must point to `num_chunks` initialized `size_t` values in device-accessible memory.
///
/// # Errors
///
/// Returns an error if a size does not fit nvCOMP's integer types or nvCOMP rejects the query.
pub unsafe fn heuristic_conf(
    device_input_chunk_bytes: DevicePtr,
    num_chunks: usize,
    max_input_chunk_bytes: usize,
    stream: &Stream,
) -> Result<KernelConf> {
    let mut raw = sys::nvcompCRC32KernelConf_t::default();
    // SAFETY: `nvcompBatchedCRC32GetHeuristicConf` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nvcompBatchedCRC32GetHeuristicConf(
            device_input_chunk_bytes.cast(),
            to_u64(num_chunks, "num_chunks")?,
            &raw mut raw,
            to_u64(max_input_chunk_bytes, "max_input_chunk_bytes")?,
            stream.as_raw(),
        ))?;
    }
    Ok(raw.into())
}

/// Arguments for [`search_conf`].
#[derive(Debug, Clone, Copy)]
pub struct SearchConf<'a> {
    /// Device array of `num_chunks` pointers to the input chunks, which need no particular alignment.
    pub device_input_chunk_ptrs: DevicePtr,
    /// Device array of `num_chunks` `size_t` sizes, in bytes, of the input chunks.
    pub device_input_chunk_bytes: DevicePtr,
    /// Number of chunks, which is also the length of every device array.
    pub num_chunks: usize,
    /// Device array of `num_chunks` checksums that the benchmark writes to.
    pub device_crc32: DevicePtr,
    /// CRC32 variant to benchmark with.
    pub spec: Spec,
    /// Stream that runs the work.
    pub stream: &'a Stream,
}

/// Benchmarks kernel configurations on the given chunks and returns the fastest.
///
/// [`heuristic_conf`] is usually fast enough and much cheaper.
/// Searching pays off only over many batches whose chunk counts and sizes stay similar, so one configuration fits them all.
/// The search blocks the host and synchronizes `params.stream`, or the whole device if it does not support stream-ordered memory allocation.
///
/// # Safety
///
/// Every pointer must reference initialized device-accessible memory that stays valid until this returns.
/// `device_crc32` must have `num_chunks` entries writable by the device.
///
/// # Errors
///
/// Returns an error if a size does not fit nvCOMP's integer types or nvCOMP rejects the query.
pub unsafe fn search_conf(params: SearchConf<'_>) -> Result<KernelConf> {
    let mut raw = sys::nvcompCRC32KernelConf_t::default();
    // SAFETY: `nvcompBatchedCRC32SearchConf` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::nvcompBatchedCRC32SearchConf(
            params
                .device_input_chunk_ptrs
                .cast::<*const c_void>()
                .cast_const(),
            params.device_input_chunk_bytes.cast(),
            to_u64(params.num_chunks, "num_chunks")?,
            params.device_crc32.cast(),
            params.spec.into(),
            &raw mut raw,
            params.stream.as_raw(),
        ))?;
    }
    Ok(raw.into())
}

/// Arguments for [`compute_raw_async`].
#[derive(Debug, Clone, Copy)]
pub struct ComputeRawAsync<'a> {
    /// Device array of `num_chunks` pointers to the input chunks, which need no particular alignment.
    pub device_input_chunk_ptrs: DevicePtr,
    /// Device array of `num_chunks` `size_t` sizes, in bytes, of the input chunks.
    pub device_input_chunk_bytes: DevicePtr,
    /// Number of chunks, which is also the length of every device array.
    pub num_chunks: usize,
    /// Device array of `num_chunks` checksums.
    ///
    /// For middle and last segments, it must hold the checksums from the previous segment, which nvCOMP updates.
    pub device_crc32: DevicePtr,
    /// CRC32 variant and kernel launch configuration.
    pub options: Options,
    /// Where the chunks fall within their messages.
    pub segment_kind: Crc32SegmentKind,
    /// Device array of `num_chunks` statuses that receive each chunk's result, or null to skip status reporting.
    pub device_statuses: DevicePtr,
    /// Stream that runs the work.
    pub stream: &'a Stream,
}

/// Enqueues CRC32 computation for each chunk on `params.stream`.
///
/// A message can be checksummed in segments over several calls; `segment_kind` says where each segment falls.
/// If a segment turns out to be the last one after it was processed as a first or middle segment, finish the checksum by calling again with a null `device_input_chunk_ptrs` and [`Crc32SegmentKind::Last`].
/// Chunks may have length zero, which helps when messages have different numbers of segments.
/// nvCOMP does not synchronize the stream.
///
/// # Safety
///
/// Every pointer must reference device-accessible memory that stays valid until the work finishes.
/// The input arrays must be initialized, and `device_crc32` and, unless null, `device_statuses` must have `num_chunks` entries writable by the device.
/// For middle and last segments, `device_crc32` must hold the checksums from the previous segment.
///
/// # Errors
///
/// Returns an error if a size does not fit nvCOMP's integer types or nvCOMP rejects the launch.
pub unsafe fn compute_raw_async(params: ComputeRawAsync<'_>) -> Result<()> {
    // SAFETY: `nvcompBatchedCRC32Async` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::nvcompBatchedCRC32Async(
            params
                .device_input_chunk_ptrs
                .cast::<*const c_void>()
                .cast_const(),
            params.device_input_chunk_bytes.cast(),
            to_u64(params.num_chunks, "num_chunks")?,
            params.device_crc32.cast(),
            params.options.into(),
            params.segment_kind.into(),
            params.device_statuses.cast(),
            params.stream.as_raw(),
        ))
    }
}

/// Enqueues CRC32 computation for each chunk of `input`, written to `crc32_values`.
///
/// Before launching, this checks that `crc32_values` and `statuses` have one entry per chunk and share the stream's CUDA context.
/// CRC32 has no alignment requirements.
/// Call [`AsyncBatchResult::check`](crate::batch::AsyncBatchResult::check) on the result to wait for the work and read the statuses.
///
/// # Safety
///
/// The chunks and their tables must be initialized.
/// `input`, `crc32_values`, `statuses`, and the memory they reference must stay alive until the work finishes, including every replay of a captured graph.
/// Nothing else may write to them, or read `crc32_values` or `statuses`, until then.
/// For [`Crc32SegmentKind::Mid`] and [`Crc32SegmentKind::Last`], `crc32_values` must hold the checksums from the previous segment, written before this work runs.
///
/// # Errors
///
/// Returns an error if a check fails, a size does not fit nvCOMP's integer types, or nvCOMP rejects the launch.
pub unsafe fn compute_chunks_async<'a, S>(
    input: DeviceChunkTableView<'a>,
    crc32_values: &'a mut S,
    statuses: &'a mut DeviceStatuses,
    options: Options,
    segment_kind: Crc32SegmentKind,
    stream: &'a Stream,
) -> Result<AsyncBatchResult<'a>>
where
    S: DeviceOutput<u32> + ?Sized,
{
    input.validate_context(stream)?;
    if crc32_values.context() != stream.context() || statuses.context() != stream.context() {
        return Err(Error::StreamContextMismatch);
    }

    validate_same_len("crc32 values", input.len(), crc32_values.len())?;
    statuses.require(input.len())?;

    // SAFETY: the caller upholds this function's safety contract.
    let crc32_values = unsafe { DevicePtr::from_raw(crc32_values.as_mut_ptr().cast()) };

    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        compute_raw_async(ComputeRawAsync {
            device_input_chunk_ptrs: input.device_chunk_ptrs(),
            device_input_chunk_bytes: input.device_chunk_bytes(),
            num_chunks: input.len(),
            device_crc32: crc32_values,
            options,
            segment_kind,
            device_statuses: statuses.as_device_ptr(),
            stream,
        })?;
    }

    // SAFETY: the caller upholds this function's safety contract.
    Ok(unsafe { AsyncBatchResult::from_submission(statuses, stream) })
}
