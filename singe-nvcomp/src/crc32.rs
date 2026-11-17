use std::ffi::c_void;

use singe_cuda::{error::Error as CudaError, stream::Stream, types::DevicePtr, view::DeviceOutput};
use singe_nvcomp_sys as sys;

use crate::{
    batch::{
        AsyncBatchResult, DeviceChunkTable, DeviceChunkTableView, DeviceStatuses, join_operands,
    },
    error::{Error, Result},
    try_ffi,
    types::{Crc32KernelKind, Crc32SegmentKind},
    utility::{codec_size, to_i32, to_u64, validate_same_len},
};

// Passing this as the largest chunk size makes nvCOMP read the sizes from the device size array.
const DEDUCED_MAX_INPUT_CHUNK_BYTES: usize = codec_size(sys::nvcompCRC32DeducedMaxInputChunkBytes);

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
        value.as_raw()
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
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` holds a kernel kind this crate does not name.
    pub fn from_raw(raw: sys::nvcompCRC32KernelConf_t) -> Result<Self> {
        Ok(Self {
            kernel_kind: raw.kernel_kind.try_into()?,
            bytes_per_read: raw.bytes_per_read,
            blocks_per_msg: raw.blocks_per_msg,
        })
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
}

impl TryFrom<sys::nvcompCRC32KernelConf_t> for KernelConf {
    type Error = Error;

    fn try_from(raw: sys::nvcompCRC32KernelConf_t) -> Result<Self> {
        Self::from_raw(raw)
    }
}

impl From<KernelConf> for sys::nvcompCRC32KernelConf_t {
    fn from(value: KernelConf) -> Self {
        value.as_raw()
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
    ///
    /// # Errors
    ///
    /// Returns an error if the library reports a value this crate does not name.
    pub fn from_raw(raw: sys::nvcompBatchedCRC32Opts_t) -> Result<Self> {
        Ok(Self {
            spec: raw.spec.into(),
            kernel_conf: raw.kernel_conf.try_into()?,
        })
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
}

impl TryFrom<sys::nvcompBatchedCRC32Opts_t> for Options {
    type Error = Error;

    fn try_from(raw: sys::nvcompBatchedCRC32Opts_t) -> Result<Self> {
        Self::from_raw(raw)
    }
}

impl From<Options> for sys::nvcompBatchedCRC32Opts_t {
    fn from(value: Options) -> Self {
        value.as_raw()
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
/// Returns an error if `max_input_chunk_bytes` is zero, a size does not fit nvCOMP's integer types, or nvCOMP rejects the query.
pub fn heuristic_conf_for_max_chunk_size(
    num_chunks: usize,
    max_input_chunk_bytes: usize,
    stream: &Stream,
) -> Result<KernelConf> {
    if max_input_chunk_bytes == DEDUCED_MAX_INPUT_CHUNK_BYTES {
        return Err(Error::NotPositive {
            name: "max_input_chunk_bytes".into(),
        });
    }

    // SAFETY: a nonzero largest chunk size makes nvCOMP read no device memory, so the null size array is never dereferenced.
    unsafe { heuristic_conf(DevicePtr::null(), num_chunks, max_input_chunk_bytes, stream) }
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
/// The query waits for `stream`, and nvCOMP may synchronize the whole device if it does not support stream-ordered memory allocation.
///
/// # Safety
///
/// The device size array of `input` must be initialized.
///
/// # Errors
///
/// Returns an error if `input` belongs to another CUDA context, stream capture is active, a size does not fit nvCOMP's integer types, or nvCOMP rejects the query.
pub unsafe fn heuristic_conf_from_sizes(
    input: DeviceChunkTableView<'_>,
    stream: &Stream,
) -> Result<KernelConf> {
    stream.complete(|| {
        input.join_regions(stream)?;
        // SAFETY: the table's size array has `input.len()` entries in the stream's context, the caller guarantees they are initialized, and `complete` waits for the stream before the borrow ends.
        unsafe {
            heuristic_conf(
                input.device_chunk_bytes(),
                input.len(),
                DEDUCED_MAX_INPUT_CHUNK_BYTES,
                stream,
            )
        }
    })
}

// When `max_input_chunk_bytes` is the deduced value, nvCOMP reads `num_chunks` sizes from `device_input_chunk_bytes`.
unsafe fn heuristic_conf(
    device_input_chunk_bytes: DevicePtr,
    num_chunks: usize,
    max_input_chunk_bytes: usize,
    stream: &Stream,
) -> Result<KernelConf> {
    stream.context().bind()?;
    let mut raw = sys::nvcompCRC32KernelConf_t::default();
    // SAFETY: the caller passes a size array that matches `max_input_chunk_bytes`, and the output location is a live local.
    unsafe {
        try_ffi!(sys::nvcompBatchedCRC32GetHeuristicConf(
            device_input_chunk_bytes.cast(),
            to_u64(num_chunks, "num_chunks")?,
            &raw mut raw,
            to_u64(max_input_chunk_bytes, "max_input_chunk_bytes")?,
            stream.as_raw(),
        ))?;
    }
    raw.try_into()
}

/// Benchmarks kernel configurations on the chunks of `input` and returns the fastest.
///
/// [`heuristic_conf_for_chunks`] is usually fast enough and much cheaper.
/// Searching pays off only over many batches whose chunk counts and sizes stay similar, so one configuration fits them all.
/// The benchmark writes scratch checksums to `crc32_values`, and the search waits for `stream`.
///
/// # Safety
///
/// Every chunk of `input` must be initialized.
///
/// # Errors
///
/// Returns an error if `crc32_values` does not have one entry per chunk, an operand belongs to another CUDA context, stream capture is active, a size does not fit nvCOMP's integer types, or nvCOMP rejects the query.
pub unsafe fn search_conf<S>(
    input: DeviceChunkTableView<'_>,
    crc32_values: &mut S,
    spec: Spec,
    stream: &Stream,
) -> Result<KernelConf>
where
    S: DeviceOutput<u32> + ?Sized,
{
    input.validate_context(stream)?;
    if crc32_values.context() != stream.context() {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    validate_same_len(input.len(), crc32_values.len(), "crc32 values")?;
    stream.complete(|| {
        join_operands(
            stream,
            input.regions(),
            &[crc32_values.region()],
            None,
            None,
        )?;
        stream.context().bind()?;
        let mut raw = sys::nvcompCRC32KernelConf_t::default();
        // SAFETY: the table and `crc32_values` have one entry per chunk in the stream's context, the caller guarantees initialized chunks, and `complete` waits for the stream before the borrows end.
        unsafe {
            try_ffi!(sys::nvcompBatchedCRC32SearchConf(
                input
                    .device_chunk_ptrs()
                    .cast::<*const c_void>()
                    .cast_const(),
                input.device_chunk_bytes().cast(),
                to_u64(input.len(), "num_chunks")?,
                crc32_values.as_mut_ptr().cast(),
                spec.into(),
                &raw mut raw,
                stream.as_raw(),
            ))?;
        }
        raw.try_into()
    })
}

/// Enqueues CRC32 computation for each chunk of `input`, written to `crc32_values`.
///
/// A message can be checksummed in segments over several calls; `segment_kind` says where each segment falls.
/// Chunks may have length zero, which helps when messages have different numbers of segments.
/// Before launching, this checks that `crc32_values` and `statuses` have one entry per chunk and share the stream's CUDA context, then orders the work after the operands' producers.
/// CRC32 has no alignment requirements.
/// Call [`AsyncBatchResult::check`] on the result to wait for the work and read the statuses.
///
/// # Safety
///
/// The chunks must be initialized, and the crate's raw execution contract applies to every operand.
/// For [`Crc32SegmentKind::Mid`] and [`Crc32SegmentKind::Last`], `crc32_values` must hold the checksums from the previous segment.
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
    statuses.validate_context(stream)?;
    if crc32_values.context() != stream.context() {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    validate_same_len(input.len(), crc32_values.len(), "crc32 values")?;
    statuses.require(input.len())?;
    join_operands(
        stream,
        input.regions(),
        &[crc32_values.region()],
        None,
        Some(&*statuses),
    )?;
    stream.context().bind()?;

    // SAFETY: the table, `crc32_values`, and `statuses` have one entry per chunk in the stream's context, and the operands are ordered after their producers; the caller keeps them alive and unaliased until the work finishes.
    unsafe {
        try_ffi!(sys::nvcompBatchedCRC32Async(
            input
                .device_chunk_ptrs()
                .cast::<*const c_void>()
                .cast_const(),
            input.device_chunk_bytes().cast(),
            to_u64(input.len(), "num_chunks")?,
            crc32_values.as_mut_ptr().cast(),
            options.into(),
            segment_kind.into(),
            statuses.as_device_ptr().cast(),
            stream.as_raw(),
        ))?;
    }

    // SAFETY: the launch above writes every status slot on `stream`, and the caller keeps the operands alive until it finishes.
    Ok(unsafe { AsyncBatchResult::from_submission(statuses, stream) })
}
