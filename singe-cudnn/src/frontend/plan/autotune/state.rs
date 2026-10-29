use std::mem::MaybeUninit;

use singe_cuda::{
    memory::{DeviceMemory, MemoryCopyKind},
    stream::StreamBinding,
    submission::{Access, AccessMode},
    types::DevicePtr,
};

use crate::{
    backend::{execution::VariantPack, tensor::TensorId},
    context::Context,
    error::{Error, Result},
    frontend::{bindings::Bindings, error::Error as FrontendError, plan::CompiledGraph},
    utility::to_usize,
};

struct BindingRange {
    id: TensorId,
    start: usize,
    end: usize,
    alignment: usize,
    reads: bool,
    writes: bool,
}

struct AliasGroup {
    start: usize,
    end: usize,
    alignment: usize,
    reads: bool,
    writes: bool,
    bindings: Vec<(TensorId, usize)>,
}

#[derive(Debug)]
struct TrialBuffer {
    original: DevicePtr,
    pointer: DevicePtr,
    bytes: usize,
    baseline: Option<DeviceMemory<MaybeUninit<u8>>>,
    _working: DeviceMemory<MaybeUninit<u8>>,
}

/// Private device state for all warmup and timed trials of one tuning request.
/// Keep this alive outside the stream completion scope, including on failed enqueues.
#[derive(Debug)]
pub struct AutotuneState<'a> {
    pub bindings: Bindings<'a>,
    pub accesses: Vec<Access>,
    buffers: Vec<TrialBuffer>,
}

impl<'a> AutotuneState<'a> {
    #[expect(
        clippy::too_many_lines,
        reason = "plan construction and selection keep the complete native execution lifecycle together"
    )]
    pub fn create(
        compiled: &'a CompiledGraph,
        context: &Context,
        bindings: &Bindings<'_>,
        pack: &VariantPack,
    ) -> Result<Self> {
        let mut ranges = Vec::new();
        let mut accesses = Vec::new();
        for binding in compiled.required_tensor_bindings() {
            if let Some(access) = bindings.binding_access(binding.id, AccessMode::Read) {
                accesses.push(access);
            }
            let pointer = bindings
                .ptr(binding.id)
                .ok_or(FrontendError::TensorNotFound(binding.id))?;
            let start = pointer.addr();
            let mut bytes = 0_usize;
            let mut alignment = 16_usize;
            for native in &binding.bindings {
                let record = compiled
                    .core
                    .artifacts
                    .tensors
                    .get(&native.tensor_id)
                    .ok_or(FrontendError::TensorNotFound(native.tensor_id))?;
                let mut spec = record.tensor.clone();
                if let Some(shape) = pack.override_for(record.id.into()) {
                    spec.shape = shape.clone();
                }
                let offset = to_usize(native.byte_offset, "autotune alias byte offset")?;
                let capacity = bindings
                    .byte_capacity(binding.id)
                    .map(|capacity| {
                        capacity
                            .checked_sub(offset)
                            .ok_or(FrontendError::BindingCapacityMismatch {
                                binding_id: binding.id,
                                tensor_id: native.tensor_id,
                                required: offset,
                                actual: capacity,
                            })
                    })
                    .transpose()?;
                if spec.ragged_offset.is_some()
                    && spec.ragged_byte_capacity.is_none()
                    && capacity.is_none()
                {
                    return Err(FrontendError::AutotuneRaggedCapacityRequired {
                        tensor_id: binding.id,
                    }
                    .into());
                }
                let span = spec.storage_byte_len(capacity)?;
                bytes = bytes.max(
                    offset
                        .checked_add(span)
                        .ok_or_else(|| Error::out_of_range("autotune state bytes"))?,
                );
                alignment = alignment.max(to_usize(spec.byte_alignment, "tensor alignment")?);
            }
            if bytes != 0 {
                ranges.push(BindingRange {
                    id: binding.id,
                    start,
                    end: start
                        .checked_add(bytes)
                        .ok_or_else(|| Error::out_of_range("autotune state bytes"))?,
                    alignment,
                    reads: binding.reads_initial_state,
                    writes: binding.access.writes(),
                });
            }
        }
        ranges.sort_unstable_by_key(|range| range.start);
        let mut groups = Vec::<AliasGroup>::new();
        for range in ranges {
            if let Some(group) = groups.last_mut()
                && range.start < group.end
            {
                group.end = group.end.max(range.end);
                group.alignment = group.alignment.max(range.alignment);
                group.reads |= range.reads;
                group.writes |= range.writes;
                group.bindings.push((range.id, range.start));
            } else {
                groups.push(AliasGroup {
                    start: range.start,
                    end: range.end,
                    alignment: range.alignment,
                    reads: range.reads,
                    writes: range.writes,
                    bindings: vec![(range.id, range.start)],
                });
            }
        }

        let mut trial_bindings = bindings.clone().with_tensor_metadata(
            &compiled.core.artifacts.tensors,
            &compiled.core.artifacts.binding_replacements,
        )?;
        let mut buffers = Vec::new();
        for group in groups.into_iter().filter(|group| group.writes) {
            let bytes = group.end - group.start;
            let allocation_bytes = bytes
                .checked_add(group.alignment - 1)
                .ok_or_else(|| Error::out_of_range("autotune state bytes"))?;
            let working =
                DeviceMemory::<u8>::create_uninit(context.cuda_context(), allocation_bytes)?;
            let base = working.as_mut_ptr().cast::<u8>();
            // Preserve the original address modulo every required power-of-two alignment.
            let offset = group.start.wrapping_sub(base.addr()) & (group.alignment - 1);
            let pointer = base.wrapping_add(offset);
            for (id, start) in group.bindings {
                // SAFETY: the entire overlapping group shares one allocation and retains all relative offsets.
                unsafe {
                    trial_bindings.set_raw_buffer(
                        id,
                        context.cuda_context(),
                        DevicePtr::from_raw(pointer.wrapping_add(start - group.start).cast()),
                        group.end - start,
                    )?;
                }
            }
            let baseline = group
                .reads
                .then(|| DeviceMemory::<u8>::create_uninit(context.cuda_context(), bytes))
                .transpose()?;
            buffers.push(TrialBuffer {
                // SAFETY: the source ranges came from the validated caller bindings.
                original: unsafe { DevicePtr::from_raw(group.start as *mut ()) },
                // SAFETY: the allocation covers this aligned group span.
                pointer: unsafe { DevicePtr::from_raw(pointer.cast()) },
                bytes,
                baseline,
                _working: working,
            });
        }
        Ok(Self {
            bindings: trial_bindings,
            accesses,
            buffers,
        })
    }

    /// Saves initial read state after its caller-established producer dependencies on this stream.
    /// Groups consumed only after complete graph production do not read caller output bytes.
    pub unsafe fn snapshot(&self, stream: &StreamBinding) -> Result<()> {
        for buffer in &self.buffers {
            if let Some(baseline) = &buffer.baseline {
                // SAFETY: the caller's input contract covers every readable group; the completion scope retains both sides.
                unsafe {
                    DeviceMemory::<MaybeUninit<u8>>::copy_async_on(
                        baseline.as_mut_ptr(),
                        buffer.original.cast(),
                        buffer.bytes,
                        MemoryCopyKind::DeviceToDevice,
                        stream,
                    )
                }?;
            }
        }
        Ok(())
    }

    /// Restores caller entry state before each invocation.
    /// Outputs and fully produced internal values remain private scratch without a baseline copy.
    pub unsafe fn reset(&self, stream: &StreamBinding) -> Result<()> {
        for buffer in &self.buffers {
            if let Some(baseline) = &buffer.baseline {
                // SAFETY: snapshot precedes each reset and all allocations outlive the stream completion scope.
                unsafe {
                    DeviceMemory::<MaybeUninit<u8>>::copy_async_on(
                        buffer.pointer.cast(),
                        baseline.as_ptr(),
                        buffer.bytes,
                        MemoryCopyKind::DeviceToDevice,
                        stream,
                    )
                }?;
            }
        }
        Ok(())
    }
}
