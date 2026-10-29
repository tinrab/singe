use std::{
    collections::BTreeMap,
    fmt::{self, Write},
    mem::MaybeUninit,
    ops::Range,
    ptr::copy_nonoverlapping,
};

use singe_cuda::{
    memory::{DeviceMemory, MemoryCopyKind, MemoryType, PointerAttributes},
    stream::StreamCaptureStatus,
    types::DevicePtr,
};

use crate::{
    backend::{
        execution::VariantPack,
        tensor::{BackendTensorUid, TensorId, TensorSpec},
    },
    context::Context,
    data_type::DataType,
    error::{Error, Result},
    frontend::{graph::TensorRecord, logging::logger},
    utility::to_usize,
};

/// C++ frontend tensor-dump encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDumpFormat {
    /// Formats each element as a hexadecimal integer bit pattern.
    Hexadecimal,
    /// Interprets 8-bit values as unsigned and wider values as signed integers, as C++ does.
    /// Floating-point data is displayed as its integer bit pattern.
    Decimal,
    /// Encodes the contiguous bytes using Base64.
    Base64,
}

/// An explicit contiguous host, device or managed-memory dump request.
/// Element widths of 1, 2, 4, 8 and 16 bytes support hexadecimal and decimal formatting.
/// Base64 accepts any nonzero element width, including caller-managed packed byte storage.
#[derive(Debug)]
pub struct TensorDump<'a> {
    /// Native tensor identity printed in the dump label.
    pub uid: BackendTensorUid,
    /// Diagnostic tensor name printed in the dump label.
    pub name: &'a str,
    /// Address of the initialized contiguous host, device, or managed-memory storage to read.
    pub pointer: DevicePtr,
    /// Number of contiguous elements to read.
    pub element_count: usize,
    /// Storage width of each element in bytes; determines the checked readback size.
    pub element_bytes: usize,
    /// Encoding applied to the copied element bytes.
    pub format: TensorDumpFormat,
}

/// Logs tensor contents using the active frontend logger and its tensor-dump switch.
/// Device reads follow completion of the cuDNN handle's current stream.
/// Active or invalidated capture skips readback and emits a message instead of synchronizing.
///
/// # Safety
/// The pointer must identify initialized storage for `element_count * element_bytes` bytes in the supplied context.
/// Retain the storage until this call returns, and order producers on the handle's stream.
/// No other thread or stream may write that range during readback.
///
/// # Errors
/// Returns invalid width, byte-size overflow, pointer-query, synchronization, transfer or logging errors.
pub unsafe fn dump_tensor(context: &Context, dump: &TensorDump<'_>) -> Result<()> {
    let Some(logger) = logger()?.filter(|logger| logger.tensor_dump_enabled()) else {
        return Ok(());
    };
    if context.stream()?.capture_status()? != StreamCaptureStatus::None {
        logger.message(format_args!(
            "tensor readback skipped during CUDA graph capture"
        ))?;
        return Ok(());
    }
    let bytes = dump
        .element_count
        .checked_mul(dump.element_bytes)
        .ok_or_else(|| Error::out_of_range("tensor dump bytes"))?;
    validate_width(dump.element_bytes, dump.format)?;
    context.bind()?;
    context.stream()?.synchronize()?;
    let mut host = vec![MaybeUninit::uninit(); bytes];
    // SAFETY: the caller establishes initialization, capacity, producer ordering and exclusive readback.
    unsafe {
        read_bytes(dump.pointer, &mut host)?;
    }
    let host = host
        .into_iter()
        // SAFETY: the caller upholds this function's safety contract.
        .map(|byte| unsafe { byte.assume_init() })
        .collect::<Vec<_>>();
    let data = TensorDumpDisplay::create(&host, dump.element_bytes, dump.format)?;
    logger.message(format_args!(
        "tensor dump uid: {} name: {} data: {data}",
        dump.uid, dump.name,
    ))
}

pub unsafe fn log_variant_pack(
    context: &Context,
    pack: &VariantPack,
    tensors: &BTreeMap<TensorId, TensorRecord>,
) -> Result<()> {
    let Some(logger) = logger()?.filter(|logger| logger.tensor_dump_enabled()) else {
        return Ok(());
    };
    if context.cuda_context() != pack.cuda_context() {
        return Err(Error::ContextMismatch {
            name: "variant pack logging".into(),
        });
    }
    if context.stream()?.capture_status()? != StreamCaptureStatus::None {
        logger.diagnostic(format_args!(
            "tensor readback skipped during CUDA graph capture"
        ));
        return Ok(());
    }
    context.bind()?;
    let records = tensors
        .values()
        .filter(|record| record.access.is_some())
        .map(|record| (record.id.into(), record))
        .collect::<BTreeMap<BackendTensorUid, _>>();
    let writes = written_ranges(pack, &records)?;
    let mut synchronized = false;
    for (uid, pointer) in pack.bindings() {
        let attributes = PointerAttributes::query(pointer.as_raw())?;
        logger.diagnostic(format_args!(
            "variant pack uid: {uid} memory type: {} device: {} pointer: {:p} device pointer: {:p} host pointer: {:p}",
            attributes.memory_type, attributes.device, pointer.as_raw(), attributes.device_pointer.as_raw(), attributes.host_pointer,
        ));
        let Some(record) = records.get(&uid) else {
            continue;
        };
        if !record.indexes_memory
            || !record.reads_initial_state
            || record
                .access
                .is_some_and(super::super::operation::TensorAccess::writes)
            || record.tensor.is_workspace()
            || !matches!(record.tensor.data_type, DataType::I32 | DataType::I64)
        {
            continue;
        }
        if record.tensor.ragged_offset.is_some() {
            logger.diagnostic(format_args!(
                "tensor dump uid: {uid} skipped: packed metadata cannot be gathered from nominal dimensions"
            ));
            continue;
        }
        let spec = effective_spec(pack, record);
        let bytes = spec.storage_byte_len(None)?;
        let range = pointer_range(pointer, bytes)?;
        if writes.as_ref().is_none_or(|writes| {
            writes
                .iter()
                .any(|write| range.start < write.end && write.start < range.end)
        }) {
            logger.diagnostic(format_args!(
                "tensor dump uid: {uid} skipped: storage may be written by the graph"
            ));
            continue;
        }
        if !synchronized {
            context.stream()?.synchronize()?;
            synchronized = true;
        }
        let mut storage = vec![MaybeUninit::uninit(); bytes];
        // SAFETY: caller-supplied entry metadata is initialized before execution; graph-written and automatic workspace values are excluded above.
        // Padding is copied as MaybeUninit and never interpreted as a value.
        unsafe {
            read_bytes(pointer, &mut storage)?;
        }
        let width = to_usize(spec.data_type.size(), "metadata element bytes")?;
        let dimensions = spec.logical_dimensions()?;
        let mut values = Vec::new();
        for index in 0..spec.logical_element_count()? {
            let offset = scalar_offset(&spec, &dimensions, index)?
                .checked_mul(width)
                .ok_or(Error::InvalidDataShape)?;
            let end = offset.checked_add(width).ok_or(Error::InvalidDataShape)?;
            let element = storage.get(offset..end).ok_or(Error::InvalidDataShape)?;
            // SAFETY: these bytes describe one initialized metadata element, excluding all stride gaps.
            values.extend(element.iter().map(|byte| unsafe { byte.assume_init() }));
        }
        let data = TensorDumpDisplay::create(&values, width, TensorDumpFormat::Decimal)?;
        let name = spec.name.as_deref().unwrap_or("");
        logger.diagnostic(format_args!(
            "tensor dump uid: {uid} name: {name} data: {data}"
        ));
    }
    Ok(())
}

fn effective_spec(pack: &VariantPack, record: &TensorRecord) -> TensorSpec {
    let mut spec = record.tensor.clone();
    if let Some(shape) = pack.override_for(record.id.into()) {
        spec.shape = shape.clone();
    }
    spec
}

fn written_ranges(
    pack: &VariantPack,
    records: &BTreeMap<BackendTensorUid, &TensorRecord>,
) -> Result<Option<Vec<Range<usize>>>> {
    let mut writes = Vec::new();
    for (uid, pointer) in pack.bindings() {
        let Some(record) = records.get(&uid) else {
            continue;
        };
        if !record
            .access
            .is_some_and(super::super::operation::TensorAccess::writes)
        {
            continue;
        }
        let spec = effective_spec(pack, record);
        // An undeclared ragged write span cannot establish that other metadata is initialized at entry.
        if spec.ragged_offset.is_some() && spec.ragged_byte_capacity.is_none() {
            return Ok(None);
        }
        let bytes = spec.storage_byte_len(None)?;
        if bytes != 0 {
            writes.push(pointer_range(pointer, bytes)?);
        }
    }
    Ok(Some(writes))
}

fn pointer_range(pointer: DevicePtr, bytes: usize) -> Result<Range<usize>> {
    let start = pointer.addr();
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| Error::out_of_range("tensor dump pointer range"))?;
    Ok(start..end)
}

unsafe fn read_bytes(pointer: DevicePtr, destination: &mut [MaybeUninit<u8>]) -> Result<()> {
    if destination.is_empty() {
        return Ok(());
    }
    let attributes = PointerAttributes::query(pointer.as_raw())?;
    match attributes.memory_type {
        // SAFETY: the caller upholds this function's safety contract.
        MemoryType::Device | MemoryType::Managed => unsafe {
            DeviceMemory::<MaybeUninit<u8>>::copy(
                destination.as_mut_ptr(),
                pointer.as_raw().cast(),
                destination.len(),
                MemoryCopyKind::DeviceToHost,
            )?;
        },
        // SAFETY: the caller upholds this function's safety contract.
        MemoryType::Host | MemoryType::Unregistered => unsafe {
            copy_nonoverlapping(
                pointer.as_raw().cast::<MaybeUninit<u8>>(),
                destination.as_mut_ptr(),
                destination.len(),
            );
        },
        _ => {
            return Err(Error::DescriptorMismatch {
                name: "tensor dump memory type".into(),
            });
        }
    }
    Ok(())
}

fn scalar_offset(spec: &TensorSpec, dimensions: &[i64], mut index: usize) -> Result<usize> {
    let vectorization = spec.vectorization()?;
    let lanes = to_usize(
        vectorization.map_or(1, |(count, _)| count),
        "tensor vector count",
    )?;
    let mut vector = 0_usize;
    let mut lane = 0;
    for axis in (0..dimensions.len()).rev() {
        let dimension = to_usize(dimensions[axis], "tensor dimension")?;
        let mut coordinate = index % dimension;
        index /= dimension;
        if vectorization.is_some_and(|(_, dimension)| {
            dimension == i64::try_from(axis).expect("tensor rank fits the native index width")
        }) {
            lane = coordinate % lanes;
            coordinate /= lanes;
        }
        vector = coordinate
            .checked_mul(to_usize(spec.shape.strides()[axis], "tensor stride")?)
            .and_then(|offset| vector.checked_add(offset))
            .ok_or(Error::InvalidDataShape)?;
    }
    vector
        .checked_mul(lanes)
        .and_then(|offset| offset.checked_add(lane))
        .ok_or(Error::InvalidDataShape)
}

fn validate_width(width: usize, format: TensorDumpFormat) -> Result<()> {
    if width == 0 || (format != TensorDumpFormat::Base64 && !matches!(width, 1 | 2 | 4 | 8 | 16)) {
        return Err(Error::out_of_range("tensor dump element width"));
    }
    Ok(())
}

/// Borrowed host bytes formatted with the C++ tensor-dump integer or base64 conventions.
/// Writes directly into the formatting destination without intermediate element strings or collections.
/// Multi-byte values use CUDA's little-endian representation and need no host pointer alignment.
#[derive(Debug, Clone, Copy)]
pub struct TensorDumpDisplay<'a> {
    bytes: &'a [u8],
    width: usize,
    format: TensorDumpFormat,
}

impl<'a> TensorDumpDisplay<'a> {
    /// Validates the element width and complete elements without allocating or reading device memory.
    ///
    /// # Errors
    /// Returns an invalid element width, incomplete final element or base64-length overflow error.
    pub fn create(bytes: &'a [u8], width: usize, format: TensorDumpFormat) -> Result<Self> {
        validate_width(width, format)?;
        if !bytes.len().is_multiple_of(width) {
            return Err(Error::InvalidDataShape);
        }
        if format == TensorDumpFormat::Base64 {
            bytes
                .len()
                .div_ceil(3)
                .checked_mul(4)
                .ok_or_else(|| Error::out_of_range("tensor dump base64 length"))?;
        }
        Ok(Self {
            bytes,
            width,
            format,
        })
    }
}

impl fmt::Display for TensorDumpDisplay<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.format {
            TensorDumpFormat::Base64 => {
                const TABLE: &[u8; 64] =
                    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                for chunk in self.bytes.chunks(3) {
                    let value = (u32::from(chunk[0]) << 16)
                        | (u32::from(chunk.get(1).copied().unwrap_or(0)) << 8)
                        | u32::from(chunk.get(2).copied().unwrap_or(0));
                    formatter.write_char(TABLE[((value >> 18) & 63) as usize] as char)?;
                    formatter.write_char(TABLE[((value >> 12) & 63) as usize] as char)?;
                    formatter.write_char(if chunk.len() > 1 {
                        TABLE[((value >> 6) & 63) as usize] as char
                    } else {
                        '='
                    })?;
                    formatter.write_char(if chunk.len() > 2 {
                        TABLE[(value & 63) as usize] as char
                    } else {
                        '='
                    })?;
                }
                Ok(())
            }
            TensorDumpFormat::Hexadecimal | TensorDumpFormat::Decimal => {
                formatter.write_str("[")?;
                for (index, element) in self.bytes.chunks_exact(self.width).enumerate() {
                    if index != 0 {
                        formatter.write_str(", ")?;
                    }
                    let value = element
                        .iter()
                        .enumerate()
                        .fold(0_u128, |value, (index, &byte)| {
                            value | (u128::from(byte) << (index * 8))
                        });
                    if self.format == TensorDumpFormat::Hexadecimal {
                        write!(formatter, "0x{value:0digits$x}", digits = self.width * 2)?;
                    } else if self.width == 1 {
                        write!(formatter, "{value}")?;
                    } else {
                        let shift = (16 - self.width) * 8;
                        let value = (value << shift).cast_signed() >> shift;
                        write!(formatter, "{value}")?;
                    }
                }
                formatter.write_str("]")
            }
        }
    }
}

/// Formats initialized host bytes into an owned string with C++ tensor-dump conventions.
/// Use [`TensorDumpDisplay`] to write directly into a logger or another formatting destination.
///
/// # Errors
/// Returns an invalid element width, incomplete final element or base64-length overflow error.
pub fn format_tensor_bytes(bytes: &[u8], width: usize, format: TensorDumpFormat) -> Result<String> {
    Ok(TensorDumpDisplay::create(bytes, width, format)?.to_string())
}
