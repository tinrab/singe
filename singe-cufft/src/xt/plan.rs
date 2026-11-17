use std::{
    cell::Cell,
    ffi::{CString, c_void},
    marker::PhantomData,
    ptr,
    sync::{Arc, Mutex, MutexGuard},
};

use singe_cuda::{
    context::Context as CudaContext, data_type::DataTypeLike, device::Device,
    error::Error as CudaError, stream::Stream, view::DeviceOutputView,
};
use singe_cufft_sys::xt as sys;

use super::{
    CallbackType, Factors, Layout, SubFormat, WorkAreaPolicy, check_buffer, is_complex,
    memory::DistributedMemory,
};
use crate::{
    error::{Error, Result},
    plan::Plan as BasePlan,
    try_ffi,
    types::Direction,
    utility::{to_i32, to_size_t, to_usize},
};

// The native plan and the callback tables it references, shared with distributed allocations.
#[derive(Debug)]
pub(crate) struct PlanHandle {
    pub(crate) base: BasePlan,
    pub(crate) gate: Mutex<()>,
    callbacks: Mutex<Vec<CallbackMetadata>>,
}

#[derive(Debug)]
struct CallbackMetadata {
    symbol: Option<CString>,
    code: Box<[u8]>,
    callbacks: Vec<*mut c_void>,
    caller_info: Vec<*mut c_void>,
}

// SAFETY: owned tables contain opaque native addresses; Rust never dereferences their pointees.
// The unsafe callback API requires the pointees and device code to remain valid for all uses.
unsafe impl Send for CallbackMetadata {}

impl PlanHandle {
    pub(crate) fn guard(&self) -> Result<MutexGuard<'_, ()>> {
        self.gate.lock().map_err(|_| Error::LockPoisoned {
            name: "cufft plan".into(),
        })
    }

    fn callbacks(&self) -> Result<MutexGuard<'_, Vec<CallbackMetadata>>> {
        self.callbacks.lock().map_err(|_| Error::LockPoisoned {
            name: "cufft callbacks".into(),
        })
    }
}

// SAFETY: the base plan's stream bookkeeping is accessed only under `gate` or while the owning `Plan` is unique.
// Shared allocations retain immutable native ownership and serialize native host operations through `gate`.
unsafe impl Sync for PlanHandle {}

/// A link-time callback that a plan installs before planning.
#[derive(Debug, Clone, Copy)]
pub struct JitCallback<'a> {
    /// Name of the callback function in `code`.
    pub symbol: &'a str,
    /// LTO IR of the callback.
    pub code: &'a [u8],
    /// Element type and access the callback implements.
    pub kind: CallbackType,
    /// Caller-info pointers, one per participating GPU, or one for single-GPU execution.
    pub caller_info: &'a [*mut c_void],
}

/// Extended FFT plan, including the GPU set and the precise execution layout.
#[derive(Debug)]
pub struct Plan {
    base: Arc<PlanHandle>,
    layout: Layout,
    gpus: Vec<i32>,
    workspace_sizes: Vec<usize>,
    stream: Option<Stream>,
    _not_sync: PhantomData<Cell<()>>,
}

impl Plan {
    /// Configures a one-, two- or three-dimensional batched FFT.
    ///
    /// An empty GPU list selects single-GPU execution in `context`.
    /// Distributed storage requires an explicit list of at least two GPUs.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, batch count, embedding extents, strides, arithmetic overflow, a noncomplex execution type, or a GPU list with one or more than 64 entries.
    /// CUDA context binding, native plan creation, GPU selection, and cuFFT planning failures are also returned.
    pub fn create(context: &Arc<CudaContext>, gpus: &[Device], layout: Layout) -> Result<Self> {
        Self::create_inner(context, gpus, layout, None)
    }

    /// Configures an FFT with a link-time callback before planning.
    ///
    /// The plan retains copies of the symbol name, code bytes, and caller-info table.
    ///
    /// # Safety
    ///
    /// Callback code must obey the selected cuFFT callback ABI, access only valid memory, and keep caller-info pointees alive through every queued execution and graph replay, including errors.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`create`](Self::create), and an error if the symbol contains an interior NUL byte, the caller-info count differs from the number of GPUs (one for single-GPU execution), or cuFFT rejects the callback.
    pub unsafe fn create_with_jit_callback(
        context: &Arc<CudaContext>,
        gpus: &[Device],
        layout: Layout,
        callback: JitCallback<'_>,
    ) -> Result<Self> {
        Self::create_inner(context, gpus, layout, Some(callback))
    }

    fn create_inner(
        context: &Arc<CudaContext>,
        gpus: &[Device],
        layout: Layout,
        callback: Option<JitCallback<'_>>,
    ) -> Result<Self> {
        layout.required_elements(true)?;
        layout.required_elements(false)?;
        if !is_complex(layout.execution_type) {
            return Err(Error::out_of_range("execution data type"));
        }

        let mut devices = gpus.iter().map(|device| device.id()).collect::<Vec<_>>();
        if devices.len() == 1 || devices.len() > 64 {
            return Err(Error::out_of_range("GPU count"));
        }

        let base = Arc::new(PlanHandle {
            base: BasePlan::create(context)?,
            gate: Mutex::new(()),
            callbacks: Mutex::new(Vec::new()),
        });
        if !devices.is_empty() {
            // SAFETY: the plan handle is live and `devices` holds the GPU count passed with it.
            unsafe {
                try_ffi!(sys::cufftXtSetGPUs(
                    base.base.as_raw(),
                    to_i32(devices.len(), "GPU count")?,
                    devices.as_mut_ptr()
                ))?;
            }
        }

        if let Some(callback) = callback {
            if callback.caller_info.len() != devices.len().max(1) {
                return Err(Error::LengthMismatch {
                    name: "callback caller-info".into(),
                    expected: devices.len().max(1),
                    actual: callback.caller_info.len(),
                });
            }

            let metadata = CallbackMetadata {
                symbol: Some(CString::new(callback.symbol)?),
                code: callback.code.into(),
                callbacks: Vec::new(),
                caller_info: callback.caller_info.to_vec(),
            };
            base.base.bind()?;
            let mut callbacks = base.callbacks()?;
            callbacks.push(metadata);
            let metadata = callbacks.last_mut().expect("metadata was just pushed");
            let symbol = metadata
                .symbol
                .as_ref()
                .expect("metadata of a JIT callback always carries a symbol")
                .as_ptr();
            let code_size = to_size_t(metadata.code.len(), "callback code size")?;
            // SAFETY: the plan handle is live, and the symbol, code, and caller-info table are retained with the plan.
            unsafe {
                try_ffi!(sys::cufftXtSetJITCallback(
                    base.base.as_raw(),
                    symbol,
                    metadata.code.as_ptr().cast(),
                    code_size,
                    callback.kind.into(),
                    metadata.caller_info.as_mut_ptr(),
                ))?;
            }

            drop(callbacks);
        }

        let mut result = Self {
            base,
            layout,
            gpus: devices,
            workspace_sizes: vec![0; gpus.len().max(1)],
            stream: None,
            _not_sync: PhantomData,
        };
        result.plan_many(false)?;
        Ok(result)
    }

    fn plan_many(&mut self, estimate: bool) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard()?;
        self.base.base.bind()?;
        let mut n = self.layout.dimensions.clone();
        let mut input = self.layout.input_embedding.clone();
        let mut output = self.layout.output_embedding.clone();
        let mut sizes = vec![0; self.workspace_sizes.len()];
        let function = if estimate {
            sys::cufftXtGetSizeMany
        } else {
            sys::cufftXtMakePlanMany
        };
        // SAFETY: `function` is a cuFFT plan-many entry point; it receives the live plan handle and arrays that outlive the call, with one workspace size per GPU.
        unsafe {
            try_ffi!(function(
                self.base.base.as_raw(),
                to_i32(n.len(), "rank")?,
                n.as_mut_ptr(),
                input.as_mut().map_or(ptr::null_mut(), Vec::as_mut_ptr),
                self.layout.input_stride,
                self.layout.input_distance,
                self.layout.input_type.into(),
                output.as_mut().map_or(ptr::null_mut(), Vec::as_mut_ptr),
                self.layout.output_stride,
                self.layout.output_distance,
                self.layout.output_type.into(),
                self.layout.batch,
                sizes.as_mut_ptr(),
                self.layout.execution_type.into()
            ))?;
        }

        self.workspace_sizes = workspace_sizes(&sizes)?;
        Ok(())
    }

    fn ensure_unique(&self) -> Result<()> {
        if Arc::strong_count(&self.base) == 1 {
            Ok(())
        } else {
            Err(Error::OwnershipShared {
                name: "cufft plan".into(),
            })
        }
    }

    /// Returns the CUDA context retained by the native plan.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.base.base.cuda_context()
    }

    /// Associates this plan with a CUDA stream and retains it, or the legacy default stream for `None`.
    ///
    /// Distributed host copies remain synchronous across the participating GPUs.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if context binding or synchronization fails, if `stream` belongs to another context, or if cuFFT rejects the stream.
    pub fn set_stream(&mut self, stream: Option<&Stream>) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard()?;
        self.base.base.bind()?;
        self.base.base.cuda_context().synchronize()?;
        self.base.base.with_stream(stream)?;
        self.stream = stream.cloned();
        Ok(())
    }

    /// Returns workspace bytes for each participating GPU.
    #[must_use]
    pub fn workspace_sizes(&self) -> &[usize] {
        &self.workspace_sizes
    }

    /// Re-estimates workspace bytes for this exact layout and current plan settings.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if context binding fails, or if cuFFT cannot query workspace sizes for this layout.
    pub fn estimate_workspace_sizes(&mut self) -> Result<&[usize]> {
        self.plan_many(true)?;
        Ok(&self.workspace_sizes)
    }

    /// Applies a workspace policy and returns the resulting per-device workspace bytes.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if `limits` does not contain one size per participating GPU, if context binding fails, or if cuFFT rejects the policy or limits.
    pub fn set_work_area_policy(
        &mut self,
        policy: WorkAreaPolicy,
        limits: &[usize],
    ) -> Result<&[usize]> {
        self.ensure_unique()?;
        let _native = self.base.guard()?;
        if limits.len() != self.workspace_sizes.len() {
            return Err(Error::LengthMismatch {
                name: "workspace limits".into(),
                expected: self.workspace_sizes.len(),
                actual: limits.len(),
            });
        }

        self.base.base.bind()?;
        let mut sizes = limits
            .iter()
            .map(|&limit| to_size_t(limit, "workspace limit"))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: the plan handle is live and `sizes` holds one size per participating GPU.
        unsafe {
            try_ffi!(sys::cufftXtSetWorkAreaPolicy(
                self.base.base.as_raw(),
                policy.into(),
                sizes.as_mut_ptr()
            ))?;
        }

        self.workspace_sizes = workspace_sizes(&sizes)?;
        Ok(&self.workspace_sizes)
    }

    /// Assigns external work areas, one per participating GPU, or one for single-GPU execution.
    ///
    /// # Safety
    ///
    /// Keep every work area alive and unused by other work until the plan is destroyed or its work areas are replaced after completion.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if `areas` does not contain one area per participating GPU, an area lives on another GPU or is smaller than its workspace size, context binding fails, or cuFFT rejects the work areas.
    pub unsafe fn set_work_areas(&mut self, areas: &mut [DeviceOutputView<'_, u8>]) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard()?;
        if areas.len() != self.workspace_sizes.len() {
            return Err(Error::LengthMismatch {
                name: "work areas".into(),
                expected: self.workspace_sizes.len(),
                actual: areas.len(),
            });
        }

        let mut pointers = Vec::with_capacity(areas.len());
        for (index, area) in areas.iter_mut().enumerate() {
            let device = self
                .gpus
                .get(index)
                .map_or_else(|| self.cuda_context().device(), |&id| Device::new(id));
            if area.context().device() != device {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            if area.len() < self.workspace_sizes[index] {
                return Err(Error::BufferTooSmall {
                    name: "work area".into(),
                    required: self.workspace_sizes[index],
                    actual: area.len(),
                });
            }

            pointers.push(area.as_mut_ptr().cast::<c_void>());
        }

        self.base.base.bind()?;
        // SAFETY: the plan handle is live, `pointers` holds one checked work area per GPU, and the caller retains them.
        unsafe {
            try_ffi!(sys::cufftXtSetWorkArea(
                self.base.base.as_raw(),
                pointers.as_mut_ptr()
            ))?;
        }
        Ok(())
    }

    /// Allocates pending distributed storage and retains the native plan independently.
    ///
    /// The plan must select at least two GPUs.
    /// No sample or padding bytes are claimed initialized.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than two GPUs were selected, the stream is being captured, CUDA context operations or synchronization fail, cuFFT allocation fails, or the returned descriptor contains null, empty, overflowing, or inconsistent device-allocation metadata.
    pub fn allocate_uninit(&self, format: SubFormat) -> Result<DistributedMemory> {
        if self.gpus.len() < 2 {
            return Err(Error::out_of_range("distributed GPU count"));
        }

        DistributedMemory::allocate(&self.base, &self.layout, self.stream.as_ref(), format)
    }

    /// Enqueues an extended-precision transform with writable input and pending output on the plan's stream.
    ///
    /// Inverse real transforms may overwrite their input spectrum.
    ///
    /// # Safety
    ///
    /// Follow the crate's raw execution contract, holding both operands exclusively.
    ///
    /// # Errors
    ///
    /// Returns an error for a distributed plan, mismatched element types, insufficient buffer capacity, or overflow in the configured storage span.
    /// Context mismatches, dependency ordering, context binding, and cuFFT submission failures are also returned.
    pub unsafe fn execute_async<I: DataTypeLike, O: DataTypeLike>(
        &self,
        input: DeviceOutputView<'_, I>,
        output: DeviceOutputView<'_, O>,
        direction: Direction,
    ) -> Result<()> {
        if self.gpus.len() > 1 {
            return Err(Error::out_of_range("GPU count"));
        }

        check_buffer::<I>(&self.layout, input.len(), true)?;
        check_buffer::<O>(&self.layout, output.len(), false)?;
        let input = input.into_region();
        let output = output.into_region();
        let _native = self.base.guard()?;
        self.base
            .base
            .prepare_execution(self.stream.as_ref(), &[input, output])?;
        // SAFETY: `check_buffer` matched the operands' types and lengths to the layout, `prepare_execution` ordered the plan after their producers, and the caller keeps both operands and the plan alive and exclusive until completion.
        unsafe {
            try_ffi!(sys::cufftXtExec(
                self.base.base.as_raw(),
                input.as_ptr().cast(),
                output.as_ptr().cast(),
                i32::from(direction)
            ))?;
        }
        Ok(())
    }

    /// Enqueues an extended-precision transform over the same writable storage on the plan's stream.
    ///
    /// The allocation must accommodate both representations, including real-transform padding.
    /// The resulting representation may differ from `T`; untouched padding stays uninitialized.
    ///
    /// # Safety
    ///
    /// Follow the crate's raw execution contract, holding `memory` exclusively.
    ///
    /// # Errors
    ///
    /// Returns an error for a distributed plan, mismatched element types, insufficient buffer capacity, or overflow in the configured storage span.
    /// Context mismatches, dependency ordering, context binding, and cuFFT submission failures are also returned.
    pub unsafe fn execute_in_place_async<T: DataTypeLike>(
        &self,
        memory: DeviceOutputView<'_, T>,
        direction: Direction,
    ) -> Result<()> {
        if self.gpus.len() > 1 {
            return Err(Error::out_of_range("GPU count"));
        }

        if T::data_type() != self.layout.input_type {
            return Err(Error::DataTypeMismatch {
                name: "in-place storage".into(),
                expected: self.layout.input_type,
                actual: T::data_type(),
            });
        }

        let input = self
            .layout
            .required_elements(true)?
            .checked_mul(self.layout.input_type.size_of())
            .ok_or_else(|| Error::out_of_range("input bytes"))?;
        let output = self
            .layout
            .required_elements(false)?
            .checked_mul(self.layout.output_type.size_of())
            .ok_or_else(|| Error::out_of_range("output bytes"))?;
        let memory = memory.into_region();
        if memory.byte_len() < input.max(output) {
            return Err(Error::BufferTooSmall {
                name: "in-place storage bytes".into(),
                required: input.max(output),
                actual: memory.byte_len(),
            });
        }

        let _native = self.base.guard()?;
        self.base
            .base
            .prepare_execution(self.stream.as_ref(), &[memory])?;
        // SAFETY: the storage covers both configured representations (checked above), `prepare_execution` ordered the plan after its producers, and the caller keeps `memory` and the plan alive and exclusive until completion.
        unsafe {
            try_ffi!(sys::cufftXtExec(
                self.base.base.as_raw(),
                memory.as_ptr().cast(),
                memory.as_ptr().cast(),
                i32::from(direction)
            ))?;
        }
        Ok(())
    }

    /// Enqueues a distributed transform over pending allocations from this native plan.
    ///
    /// # Safety
    ///
    /// Follow the crate's raw execution contract, holding both allocations exclusively, with input samples initialized in the configured distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if either allocation belongs to another native plan, context binding fails, or cuFFT rejects the distributed submission.
    pub unsafe fn execute_distributed_async(
        &self,
        input: &mut DistributedMemory,
        output: &mut DistributedMemory,
        direction: Direction,
    ) -> Result<()> {
        if !input.belongs_to(&self.base) || !output.belongs_to(&self.base) {
            return Err(Error::PlanMismatch);
        }

        let _native = self.base.guard()?;
        self.base.base.bind()?;
        // SAFETY: both descriptors were allocated by this plan (checked above) and the plan gate is held; the caller keeps them initialized, exclusive, and alive until completion.
        unsafe {
            try_ffi!(sys::cufftXtExecDescriptor(
                self.base.base.as_raw(),
                input.as_raw(),
                output.as_raw(),
                i32::from(direction)
            ))?;
        }
        Ok(())
    }

    /// Enqueues a distributed transform in place without promoting allocation padding.
    ///
    /// # Safety
    ///
    /// Follow the crate's raw execution contract, holding the allocation exclusively, with input samples initialized in the configured distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation belongs to another native plan, its format is not `InPlace` or `InPlaceShuffled`, context binding fails, or cuFFT rejects the distributed submission.
    pub unsafe fn execute_distributed_in_place_async(
        &self,
        memory: &mut DistributedMemory,
        direction: Direction,
    ) -> Result<()> {
        if !memory.belongs_to(&self.base) {
            return Err(Error::PlanMismatch);
        }

        if !matches!(
            memory.format(),
            SubFormat::InPlace | SubFormat::InPlaceShuffled
        ) {
            return Err(Error::out_of_range("in-place format"));
        }

        let _native = self.base.guard()?;
        self.base.base.bind()?;
        // SAFETY: the descriptor was allocated by this plan in an in-place format (checked above) and the plan gate is held; the caller keeps it initialized, exclusive, and alive until completion.
        unsafe {
            try_ffi!(sys::cufftXtExecDescriptor(
                self.base.base.as_raw(),
                memory.as_raw(),
                memory.as_raw(),
                i32::from(direction)
            ))?;
        }
        Ok(())
    }

    /// Returns the factorization metadata for a distributed one-dimensional transform.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails or cuFFT cannot provide one-dimensional factorization metadata for the configured plan.
    pub fn factors(&self) -> Result<Factors> {
        let _native = self.base.guard()?;
        let mut factors = sys::cufftXt1dFactors::default();
        self.base.base.bind()?;
        // SAFETY: the plan handle is live and `factors` is a valid output location of the queried type.
        unsafe {
            try_ffi!(sys::cufftXtQueryPlan(
                self.base.base.as_raw(),
                (&raw mut factors).cast(),
                sys::cufftXtQueryType::CUFFT_QUERY_1D_FACTORS
            ))?;
        }
        Ok(Factors {
            size: factors.size,
            string_count: factors.stringCount,
            string_length: factors.stringLength,
            substring_length: factors.substringLength,
            factor1: factors.factor1,
            factor2: factors.factor2,
            string_mask: factors.stringMask,
            substring_mask: factors.substringMask,
            factor1_mask: factors.factor1Mask,
            factor2_mask: factors.factor2Mask,
            string_shift: factors.stringShift,
            substring_shift: factors.substringShift,
            factor1_shift: factors.factor1Shift,
            factor2_shift: factors.factor2Shift,
        })
    }

    /// Installs legacy callbacks, one device-function pointer per participating GPU.
    ///
    /// The plan retains copies of both pointer tables.
    ///
    /// # Safety
    ///
    /// Functions must obey the specified callback ABI, and callback code plus caller-info memory must remain valid for every queued execution and graph replay, including errors.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if either pointer table lacks one entry per participating GPU, if context binding fails, or if cuFFT rejects the callback installation.
    pub unsafe fn set_callbacks(
        &mut self,
        kind: CallbackType,
        callbacks: &[*mut c_void],
        caller_info: &[*mut c_void],
    ) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard()?;
        if callbacks.len() != self.workspace_sizes.len() {
            return Err(Error::LengthMismatch {
                name: "callbacks".into(),
                expected: self.workspace_sizes.len(),
                actual: callbacks.len(),
            });
        }

        if caller_info.len() != callbacks.len() {
            return Err(Error::LengthMismatch {
                name: "callback caller-info".into(),
                expected: callbacks.len(),
                actual: caller_info.len(),
            });
        }

        self.base.base.bind()?;
        let mut attachments = self.base.callbacks()?;
        attachments.push(CallbackMetadata {
            symbol: None,
            code: Box::new([]),
            callbacks: callbacks.to_vec(),
            caller_info: caller_info.to_vec(),
        });
        let metadata = attachments.last_mut().expect("metadata was just pushed");
        // SAFETY: the plan handle is live, and both tables hold one entry per GPU and are retained with the plan.
        unsafe {
            try_ffi!(sys::cufftXtSetCallback(
                self.base.base.as_raw(),
                metadata.callbacks.as_mut_ptr(),
                kind.into(),
                metadata.caller_info.as_mut_ptr()
            ))?;
        }

        drop(attachments);
        Ok(())
    }

    /// Removes the callback of `kind` from the native plan.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if context binding fails, or if cuFFT cannot clear this callback kind.
    pub fn clear_callback(&mut self, kind: CallbackType) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard()?;
        self.base.base.bind()?;
        // SAFETY: the plan handle is live and its context is bound.
        unsafe {
            try_ffi!(sys::cufftXtClearCallback(
                self.base.base.as_raw(),
                kind.into()
            ))?;
        }
        Ok(())
    }

    /// Sets the shared-memory byte count requested by a callback.
    ///
    /// cuFFT supplies this storage to callbacks of `kind` during transform execution.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if context binding fails, or if cuFFT rejects the callback kind or shared-memory size.
    pub fn set_callback_shared_size(&mut self, kind: CallbackType, bytes: usize) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard()?;
        self.base.base.bind()?;
        let bytes = to_size_t(bytes, "callback shared size")?;
        // SAFETY: the plan handle is live and its context is bound.
        unsafe {
            try_ffi!(sys::cufftXtSetCallbackSharedSize(
                self.base.base.as_raw(),
                kind.into(),
                bytes
            ))?;
        }
        Ok(())
    }

    /// Returns the native FFT plan handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> singe_cufft_sys::cufftHandle {
        self.base.base.as_raw()
    }
}

fn workspace_sizes(sizes: &[sys::size_t]) -> Result<Vec<usize>> {
    sizes
        .iter()
        .map(|&size| to_usize(size, "workspace size"))
        .collect()
}
