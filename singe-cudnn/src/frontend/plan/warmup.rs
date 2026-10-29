use std::mem::MaybeUninit;

use singe_cuda::{memory::DeviceMemory, stream::StreamCaptureStatus};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        execution::{WarmupOutcome, capture_plan_warmup},
    },
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::{Bindings, RuntimeOverrides, resolve_binding_source},
        error::Error as FrontendError,
        plan::CompiledGraph,
    },
    scalar::ScalarValue,
    utility::to_usize,
};

impl CompiledGraph {
    /// Warms the selected plan without application buffers by capturing and discarding its operations.
    /// Uses private placeholder device storage and initialized host scalar values, matching C++ frontend warmup.
    /// No kernels or captured workspace initialization operations are launched.
    /// Native execution rejection remains available in [`WarmupOutcome::Captured`].
    /// Active capture is left untouched, and the handle's stream is restored on all exit paths.
    ///
    /// # Errors
    ///
    /// Returns an error if stream or capture-state queries fail, placeholder allocation or scalar preparation fails, binding metadata or alignment is invalid, native variant-pack construction fails, execution-context retention fails, or stream configuration or capture fails. A native execution rejection during completed capture is retained in the returned outcome.
    pub fn warmup(&self, context: &mut Context) -> Result<WarmupOutcome> {
        if context.stream()?.capture_status()? != StreamCaptureStatus::None {
            return Ok(WarmupOutcome::SkippedActiveCapture);
        }
        let pack = CaptureOnlyPack::create(self, context)?;
        // SAFETY: host scalars and placeholder allocations remain live through capture and graph destruction.
        // The private pack is never exposed to normal execution or graph replay.
        unsafe { capture_plan_warmup(self.execution_plan(), context, &pack.descriptor, |_| Ok(())) }
    }

    /// Prepares the selected native plan through a throwaway capture without writing its outputs.
    /// Uses the normal bindings, frontend scratch and optional runtime overrides.
    /// Skips warmup if the handle's current stream is already capturing.
    ///
    /// # Errors
    /// Returns binding, workspace, stream configuration or capture errors.
    ///
    /// # Safety
    /// Keep borrowed device buffers, workspace and by-value host arguments valid
    /// through this call. Host values must be initialized with their declared native
    /// encodings. Device inputs need not be initialized because the captured graph
    /// is discarded without launching its reads or writes.
    pub unsafe fn warmup_with_bindings(
        &self,
        context: &mut Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<MaybeUninit<u8>>>,
        overrides: &RuntimeOverrides,
    ) -> Result<WarmupOutcome> {
        if context.stream()?.capture_status()? != StreamCaptureStatus::None {
            return Ok(WarmupOutcome::SkippedActiveCapture);
        }
        let pack = self.variant_pack_with_overrides(
            context.cuda_context(),
            bindings,
            workspace,
            overrides,
        )?;
        // SAFETY: the caller retains the supplied storage and initialized host arguments;
        // pack metadata and owned scalar storage remain live until capture is discarded.
        unsafe { self.execution_plan().warmup(context, &pack) }
    }
}

/// This descriptor is only valid for a capture that is immediately discarded.
/// It deliberately does not implement or expose the ordinary `VariantPack` execution contract.
struct CaptureOnlyPack<'a> {
    descriptor: BackendDescriptor,
    _device: DeviceMemory<u8>,
    _scalars: Bindings<'a>,
    _pointers: Vec<*mut ()>,
    _ids: Vec<i64>,
}

impl<'a> CaptureOnlyPack<'a> {
    fn create(compiled: &'a CompiledGraph, context: &Context) -> Result<Self> {
        let records = compiled
            .core
            .binding_template
            .ids()
            .iter()
            .map(|id| {
                let record = compiled
                    .core
                    .artifacts
                    .tensors
                    .get(id)
                    .ok_or(FrontendError::TensorNotFound(*id))?;
                let alignment = to_usize(record.tensor.byte_alignment, "warmup tensor alignment")?;
                if !alignment.is_power_of_two() {
                    return Err(Error::out_of_range("warmup tensor alignment"));
                }
                Ok((record, alignment))
            })
            .collect::<Result<Vec<_>>>()?;
        let alignment = records
            .iter()
            .filter(|(record, _)| !record.tensor.is_runtime_scalar())
            .fold(16_usize, |alignment, (_, requested)| {
                alignment.max(*requested)
            });
        // C++ uses one MiB for every device operand and the native workspace during discarded capture.
        // Reserve alignment padding before capture instead of allocating while capture is active.
        let device_bytes = (1024_usize * 1024)
            .checked_add(alignment - 1)
            .ok_or_else(|| Error::out_of_range("warmup storage bytes"))?;
        let device = DeviceMemory::<u8>::zeroed(context.cuda_context(), device_bytes)?;
        let device_pointer = aligned_pointer(device.as_mut_ptr(), alignment)?;
        let mut scalars = compiled.bindings()?;
        let mut pointers = Vec::with_capacity(records.len());
        let mut ids = Vec::with_capacity(records.len());
        for (record, _) in records {
            let pointer = if record.tensor.is_runtime_scalar() {
                if !record.tensor.has_scalar_shape()? {
                    return Err(Error::DescriptorMismatch {
                        name: "warmup scalar shape".into(),
                    });
                }
                let (source, offset) = resolve_binding_source(
                    &compiled.core.artifacts.binding_replacements,
                    record.id,
                )?;
                if offset != 0 {
                    return Err(Error::DescriptorMismatch {
                        name: "runtime scalar alias byte offset".into(),
                    });
                }
                if scalars.scalar_value(source).is_none() {
                    scalars.set_scalar(source, ScalarValue::one(record.tensor.data_type)?)?;
                }
                scalars
                    .ptr(source)
                    .ok_or(FrontendError::TensorNotBound(source))?
                    .as_raw()
                    .cast::<u8>()
            } else {
                device_pointer
            };
            pointers.push(pointer.cast());
            ids.push(record.id.as_i64());
        }
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::VariantPack)?;
        // SAFETY: all arrays and pointees remain live in this private pack until its discarded capture ends.
        // Device placeholder capacities and overlaps are permitted only by capture_plan_warmup's contract.
        unsafe {
            descriptor.set_attribute_void_ptr(
                BackendAttributeName::VariantPackWorkspace,
                device_pointer.cast(),
            )?;
            descriptor.set_attribute_void_ptr_slice(
                BackendAttributeName::VariantPackDataPointers,
                &pointers,
            )?;
        }
        descriptor.set_attribute_i64_slice(BackendAttributeName::VariantPackUniqueIds, &ids)?;
        descriptor.finalize()?;
        Ok(Self {
            descriptor,
            _device: device,
            _scalars: scalars,
            _pointers: pointers,
            _ids: ids,
        })
    }
}

fn aligned_pointer(pointer: *mut u8, alignment: usize) -> Result<*mut u8> {
    let address = pointer.addr();
    let aligned = address
        .checked_next_multiple_of(alignment)
        .ok_or_else(|| Error::out_of_range("warmup storage bytes"))?;
    Ok(pointer.wrapping_add(aligned - address))
}
