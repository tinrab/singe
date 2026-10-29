//! Extended precision, distributed FFT storage and callbacks.

use std::{
    cell::Cell,
    ffi::{CString, c_void},
    marker::PhantomData,
    mem::MaybeUninit,
    process, ptr, slice,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::{
    context::Context as CudaContext,
    data_type::{DataType, DataTypeLike},
    device::Device,
    stream::Stream,
    view::{AllocationId, DeviceOutputView, DeviceRegion, DeviceRegionProvider},
};
use singe_cuda_sys::driver;
use singe_cufft_sys::xt as sys;

use crate::{
    error::{Error, Result},
    plan::Plan as BasePlan,
    try_ffi,
    types::Direction,
    utility::{to_i32, to_usize},
};

/// Layout of distributed FFT storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SubFormat {
    /// by default input is in linear order across GPUs.
    Input = sys::cufftXtSubFormat::CUFFT_XT_FORMAT_INPUT as _,
    /// by default output is in scrambled order depending on transform.
    Output = sys::cufftXtSubFormat::CUFFT_XT_FORMAT_OUTPUT as _,
    /// by default inplace is input order, which is linear across GPUs.
    InPlace = sys::cufftXtSubFormat::CUFFT_XT_FORMAT_INPLACE as _,
    /// shuffled output order after execution of the transform.
    InPlaceShuffled = sys::cufftXtSubFormat::CUFFT_XT_FORMAT_INPLACE_SHUFFLED as _,
    /// shuffled input order prior to execution of 1D transforms.
    Input1dShuffled = sys::cufftXtSubFormat::CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED as _,
    /// used for cuFFTMp only.
    DistributedInput = sys::cufftXtSubFormat::CUFFT_XT_FORMAT_DISTRIBUTED_INPUT as _,
    /// used for cuFFTMp only.
    DistributedOutput = sys::cufftXtSubFormat::CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT as _,
}

impl_enum_conversion!(sys::cufftXtSubFormat, SubFormat);

/// Policy controlling the amount of temporary work area.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum WorkAreaPolicy {
    /// Maximum reduction in workspace.
    Minimal = sys::cufftXtWorkAreaPolicy::CUFFT_WORKAREA_MINIMAL as _,
    /// Use the supplied per-device sizes as workspace limits.
    User = sys::cufftXtWorkAreaPolicy::CUFFT_WORKAREA_USER as _,
    /// Default policy favoring performance.
    Performance = sys::cufftXtWorkAreaPolicy::CUFFT_WORKAREA_PERFORMANCE as _,
}

impl_enum_conversion!(sys::cufftXtWorkAreaPolicy, WorkAreaPolicy);

/// Load or store callback element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CallbackType {
    /// Load single-precision complex values.
    LoadComplex = sys::cufftXtCallbackType::CUFFT_CB_LD_COMPLEX as _,
    /// Load double-precision complex values.
    LoadComplexDouble = sys::cufftXtCallbackType::CUFFT_CB_LD_COMPLEX_DOUBLE as _,
    /// Load single-precision real values.
    LoadReal = sys::cufftXtCallbackType::CUFFT_CB_LD_REAL as _,
    /// Load double-precision real values.
    LoadRealDouble = sys::cufftXtCallbackType::CUFFT_CB_LD_REAL_DOUBLE as _,
    /// Store single-precision complex values.
    StoreComplex = sys::cufftXtCallbackType::CUFFT_CB_ST_COMPLEX as _,
    /// Store double-precision complex values.
    StoreComplexDouble = sys::cufftXtCallbackType::CUFFT_CB_ST_COMPLEX_DOUBLE as _,
    /// Store single-precision real values.
    StoreReal = sys::cufftXtCallbackType::CUFFT_CB_ST_REAL as _,
    /// Store double-precision real values.
    StoreRealDouble = sys::cufftXtCallbackType::CUFFT_CB_ST_REAL_DOUBLE as _,
}

impl_enum_conversion!(sys::cufftXtCallbackType, CallbackType);

/// Factorization and index mapping of a distributed one-dimensional transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Factors {
    /// Transform size.
    pub size: i64,
    /// Number of strings.
    pub string_count: i64,
    /// String length.
    pub string_length: i64,
    /// Substring length.
    pub substring_length: i64,
    /// First transform factor.
    pub factor1: i64,
    /// Second transform factor.
    pub factor2: i64,
    /// String index mask.
    pub string_mask: i64,
    /// Substring index mask.
    pub substring_mask: i64,
    /// First-factor index mask.
    pub factor1_mask: i64,
    /// Second-factor index mask.
    pub factor2_mask: i64,
    /// String index shift.
    pub string_shift: i32,
    /// Substring index shift.
    pub substring_shift: i32,
    /// First-factor index shift.
    pub factor1_shift: i32,
    /// Second-factor index shift.
    pub factor2_shift: i32,
}

/// Batched FFT shape and advanced input/output layouts.
/// All precisions must match; execution type must be complex.
#[derive(Debug, Clone)]
pub struct Layout {
    /// Logical transform extents from outermost to innermost; one to three positive dimensions.
    pub dimensions: Vec<i64>,
    /// Positive number of transforms in the batch.
    pub batch: i64,
    /// Scalar or interleaved-complex representation of the input samples.
    pub input_type: DataType,
    /// Scalar or interleaved-complex representation of the output samples.
    pub output_type: DataType,
    /// Complex representation used for transform computation.
    pub execution_type: DataType,
    /// Physical input extents, one per dimension, or `None` for a contiguous native layout.
    pub input_embedding: Option<Vec<i64>>,
    /// Innermost input sample spacing in input elements; ignored without an input embedding.
    pub input_stride: i64,
    /// Spacing between input batches in input elements; ignored without an input embedding.
    pub input_distance: i64,
    /// Physical output extents, one per dimension, or `None` for a contiguous native layout.
    pub output_embedding: Option<Vec<i64>>,
    /// Innermost output sample spacing in output elements; ignored without an output embedding.
    pub output_stride: i64,
    /// Spacing between output batches in output elements; ignored without an output embedding.
    pub output_distance: i64,
}

impl Layout {
    /// Describes contiguous transforms; cuFFT ignores explicit strides when embeddings are absent.
    #[must_use]
    pub fn contiguous(
        dimensions: Vec<i64>,
        batch: i64,
        input_type: DataType,
        output_type: DataType,
        execution_type: DataType,
    ) -> Self {
        Self {
            dimensions,
            batch,
            input_type,
            output_type,
            execution_type,
            input_embedding: None,
            input_stride: 1,
            input_distance: 0,
            output_embedding: None,
            output_stride: 1,
            output_distance: 0,
        }
    }

    fn required_elements(&self, input: bool) -> Result<usize> {
        if self.dimensions.is_empty()
            || self.dimensions.len() > 3
            || self.batch <= 0
            || self.dimensions.iter().any(|&n| n <= 0)
        {
            return Err(invalid("FFT dimensions"));
        }
        let (ty, embedding, stride, distance) = if input {
            (
                self.input_type,
                &self.input_embedding,
                self.input_stride,
                self.input_distance,
            )
        } else {
            (
                self.output_type,
                &self.output_embedding,
                self.output_stride,
                self.output_distance,
            )
        };
        let mut dimensions = self.dimensions.clone();
        if complex(ty) && (!complex(self.input_type) || !complex(self.output_type)) {
            let last = dimensions
                .last_mut()
                .expect("plans validate at least one dimension");
            *last = *last / 2 + 1;
        }
        let elements = if let Some(embedding) = embedding {
            if embedding.len() != dimensions.len()
                || stride <= 0
                || distance <= 0
                || embedding.iter().zip(&dimensions).any(|(e, n)| e < n)
            {
                return Err(invalid("FFT strides"));
            }
            let mut offset = 0i64;
            for (&extent, &embed) in dimensions.iter().zip(embedding) {
                offset = offset
                    .checked_mul(embed)
                    .and_then(|v| v.checked_add(extent - 1))
                    .ok_or_else(|| invalid("FFT span"))?;
            }
            offset
                .checked_mul(stride)
                .and_then(|v| {
                    (self.batch - 1)
                        .checked_mul(distance)
                        .and_then(|b| v.checked_add(b))
                })
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| invalid("FFT span"))?
        } else {
            dimensions.iter().try_fold(self.batch, |a, &b| {
                a.checked_mul(b).ok_or_else(|| invalid("FFT span"))
            })?
        };
        to_usize(elements, "FFT elements")
    }
}

fn complex(ty: DataType) -> bool {
    matches!(
        ty,
        DataType::ComplexF16 | DataType::ComplexBf16 | DataType::ComplexF32 | DataType::ComplexF64
    )
}

fn invalid(name: &str) -> Error {
    Error::out_of_range(name.to_owned())
}

#[derive(Debug)]
struct PlanHandle {
    base: BasePlan,
    gate: Mutex<()>,
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
    fn guard(&self) -> MutexGuard<'_, ()> {
        self.gate.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

// SAFETY: BasePlan's stream bookkeeping is accessed only under gate or while this owner is unique.
// Shared allocations retain immutable native ownership and serialize native host operations.
unsafe impl Sync for PlanHandle {}

/// Extended FFT plan, including the GPU set and the precise execution layout.
#[derive(Debug)]
pub struct Plan {
    base: Arc<PlanHandle>,
    layout: Layout,
    gpus: Vec<i32>,
    workspace_sizes: Vec<u64>,
    stream: Option<Stream>,
    _not_sync: PhantomData<Cell<()>>,
}

impl Plan {
    /// Configures a one-, two- or three-dimensional batched FFT.
    /// An empty GPU list selects single-GPU execution in `context`.
    /// Distributed storage requires an explicit list of at least two GPUs.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, batch count, embedding extents,
    /// strides, arithmetic overflow, a noncomplex execution type, or a GPU list
    /// with one or more than 64 entries. CUDA context binding, native plan
    /// creation, GPU selection, and cuFFT planning failures are also returned.
    pub fn create(context: &Arc<CudaContext>, gpus: &[i32], layout: Layout) -> Result<Self> {
        Self::create_inner(context, gpus, layout, None)
    }

    fn create_inner(
        context: &Arc<CudaContext>,
        gpus: &[i32],
        layout: Layout,
        callback: Option<(&str, &[u8], CallbackType, &mut [*mut c_void])>,
    ) -> Result<Self> {
        layout.required_elements(true)?;
        layout.required_elements(false)?;
        if !complex(layout.execution_type) {
            return Err(invalid("execution data type"));
        }
        let mut devices = gpus.to_vec();
        if devices.len() == 1 || devices.len() > 64 {
            return Err(invalid("GPU count"));
        }
        let base = Arc::new(PlanHandle {
            base: BasePlan::create(context)?,
            gate: Mutex::new(()),
            callbacks: Mutex::new(Vec::new()),
        });
        if !devices.is_empty() {
            // SAFETY: `cufftXtSetGPUs` receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::cufftXtSetGPUs(
                    base.base.as_raw(),
                    to_i32(devices.len(), "GPU count")?,
                    devices.as_mut_ptr()
                ))?;
            }
        }
        if let Some((symbol, code, kind, caller_info)) = callback {
            if caller_info.len() != devices.len().max(1) {
                return Err(invalid("callback caller-info count"));
            }
            let metadata = CallbackMetadata {
                symbol: Some(CString::new(symbol)?),
                code: code.into(),
                callbacks: Vec::new(),
                caller_info: caller_info.to_vec(),
            };
            base.base.bind()?;
            let mut callbacks = base
                .callbacks
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            callbacks.push(metadata);
            let metadata = callbacks.last_mut().expect("metadata was just pushed");
            // SAFETY: `cufftXtSetJITCallback` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::cufftXtSetJITCallback(
                    base.base.as_raw(),
                    metadata
                        .symbol
                        .as_ref()
                        .expect("JIT callback metadata always carries a symbol")
                        .as_ptr(),
                    metadata.code.as_ptr().cast(),
                    metadata.code.len() as _,
                    kind.into(),
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

    /// Configures an FFT with a link-time callback before planning.
    /// # Safety
    /// Callback code must obey the selected cuFFT callback ABI, access only valid
    /// memory, and keep caller-info pointees alive through every queued execution and graph replay, including errors.
    /// Symbol names, code bytes and pointer tables are retained with the native plan.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, batch count, embedding extents,
    /// strides, arithmetic overflow, a noncomplex execution type, or a GPU list
    /// with one or more than 64 entries. CUDA context binding, native plan
    /// creation, GPU selection, and cuFFT planning failures are also returned.
    /// Also returns an error if `symbol` contains an interior NUL byte, the
    /// caller-info count differs from the number of GPUs (one for single-GPU
    /// execution), or cuFFT rejects the JIT callback.
    pub unsafe fn create_with_jit_callback(
        context: &Arc<CudaContext>,
        gpus: &[i32],
        layout: Layout,
        symbol: &str,
        code: &[u8],
        kind: CallbackType,
        caller_info: &mut [*mut c_void],
    ) -> Result<Self> {
        Self::create_inner(
            context,
            gpus,
            layout,
            Some((symbol, code, kind, caller_info)),
        )
    }

    fn plan_many(&mut self, estimate: bool) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard();
        self.base.base.bind()?;
        let mut n = self.layout.dimensions.clone();
        let mut input = self.layout.input_embedding.clone();
        let mut output = self.layout.output_embedding.clone();
        let function = if estimate {
            sys::cufftXtGetSizeMany
        } else {
            sys::cufftXtMakePlanMany
        };
        // SAFETY: `function` is a cuFFT plan-many entry point; it receives the live plan handle and pointers into vectors that outlive the call.
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
                self.workspace_sizes.as_mut_ptr(),
                self.layout.execution_type.into()
            ))?;
        }
        Ok(())
    }

    fn ensure_unique(&self) -> Result<()> {
        if Arc::strong_count(&self.base) == 1 {
            Ok(())
        } else {
            Err(Error::OwnershipShared)
        }
    }

    /// Borrows the CUDA context retained by the native plan.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.base.base.cuda_context()
    }

    /// Associates this plan with a CUDA stream, retaining it for the plan lifetime.
    /// Distributed host copies remain synchronous across the participating GPUs.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if context
    /// binding or synchronization fails, if `stream` belongs to another context,
    /// or if cuFFT rejects the stream association.
    pub fn set_stream(&mut self, stream: Option<&Stream>) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard();
        self.base.base.bind()?;
        self.base.base.cuda_context().synchronize()?;
        self.base.base.set_stream(stream)?;
        self.stream = stream.cloned();
        Ok(())
    }

    /// Returns workspace bytes for each participating GPU.
    #[must_use]
    pub fn workspace_sizes(&self) -> &[u64] {
        &self.workspace_sizes
    }

    /// Re-estimates workspace for this exact layout and current plan settings.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if context
    /// binding fails, or if cuFFT cannot query workspace sizes for this layout.
    pub fn estimate_workspace_sizes(&mut self) -> Result<&[u64]> {
        self.plan_many(true)?;
        Ok(&self.workspace_sizes)
    }

    /// Applies a workspace policy and returns the resulting per-device workspace sizes.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if `limits`
    /// does not contain one size per participating GPU, if context binding fails,
    /// or if cuFFT rejects the policy or workspace limits.
    pub fn set_work_area_policy(
        &mut self,
        policy: WorkAreaPolicy,
        limits: &[u64],
    ) -> Result<&[u64]> {
        self.ensure_unique()?;
        let _native = self.base.guard();
        if limits.len() != self.workspace_sizes.len() {
            return Err(invalid("workspace limits"));
        }
        self.base.base.bind()?;
        let mut sizes = limits.to_vec();
        // SAFETY: `cufftXtSetWorkAreaPolicy` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftXtSetWorkAreaPolicy(
                self.base.base.as_raw(),
                policy.into(),
                sizes.as_mut_ptr()
            ))?;
        }
        self.workspace_sizes = sizes;
        Ok(&self.workspace_sizes)
    }

    /// Assigns external work areas, one pointer per participating GPU.
    /// # Safety
    /// Each pointer must identify memory on its corresponding GPU with at least
    /// the required size. Keep every work area alive until plan destruction.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if `areas`
    /// does not contain one pointer per participating GPU, if context binding fails,
    /// or if cuFFT rejects the workspace pointer table.
    pub unsafe fn set_work_areas(&mut self, areas: &mut [*mut c_void]) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard();
        if areas.len() != self.workspace_sizes.len() {
            return Err(invalid("workspace pointer count"));
        }
        self.base.base.bind()?;
        // SAFETY: `cufftXtSetWorkArea` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftXtSetWorkArea(
                self.base.base.as_raw(),
                areas.as_mut_ptr()
            ))?;
        }
        Ok(())
    }

    /// Allocates pending distributed storage and retains the native plan independently.
    /// The plan must select at least two GPUs.
    /// No sample or padding bytes are claimed initialized.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than two GPUs were selected, the stream is being
    /// captured, CUDA context operations or synchronization fail, cuFFT allocation
    /// fails, or the returned descriptor contains null, empty, overflowing, or
    /// inconsistent device-allocation metadata.
    pub fn allocate_uninit(&self, format: SubFormat) -> Result<DistributedMemory> {
        if self.gpus.len() < 2 {
            return Err(invalid("distributed GPU count"));
        }
        if let Some(stream) = &self.stream {
            stream.ensure_not_capturing()?;
        }
        let mut raw = ptr::null_mut();
        {
            let _native = self.base.guard();
            self.base.base.bind()?;
            // SAFETY: `cufftXtMalloc` receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::cufftXtMalloc(
                    self.base.base.as_raw(),
                    &raw mut raw,
                    format.into()
                ))?;
            }
        }
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        let mut allocation = DistributedAllocation {
            base: Arc::clone(&self.base),
            raw,
            chunks: Vec::new(),
            byte_len: 0,
            layout: self.layout.clone(),
            stream: self.stream.clone(),
        };
        allocation.describe_chunks()?;
        if allocation.byte_len == 0 {
            return Err(invalid("distributed allocation bytes"));
        }
        // Publish allocation metadata only after SDK allocation work is ready in its actual contexts.
        allocation.base.base.cuda_context().synchronize()?;
        for chunk in &allocation.chunks {
            chunk.context.synchronize()?;
        }
        Ok(DistributedMemory {
            allocation: Arc::new(allocation),
            format,
        })
    }

    /// Queues an extended-precision transform with writable input and pending output.
    /// Inverse real transforms may overwrite their input spectrum.
    ///
    /// # Safety
    ///
    /// Samples read by the configured transform must be initialized when execution reaches them.
    /// Both regions must remain exclusive and live through device completion, including failures and graph replays.
    /// Retain the plan, callback state and workspaces through all queued uses and do not reconfigure them meanwhile.
    ///
    /// # Errors
    ///
    /// Returns an error for a distributed plan, mismatched element types,
    /// insufficient buffer capacity, or overflow in the configured storage span.
    /// Context mismatches, dependency ordering, context binding, and cuFFT
    /// submission failures are also returned. Success reports submission only.
    pub unsafe fn execute_async<I: DataTypeLike, O: DataTypeLike>(
        &self,
        input: DeviceOutputView<'_, I>,
        output: DeviceOutputView<'_, O>,
        direction: Direction,
    ) -> Result<()> {
        if self.gpus.len() > 1 {
            return Err(invalid("single-GPU execution"));
        }
        self.check_buffer::<I>(input.len(), true)?;
        self.check_buffer::<O>(output.len(), false)?;
        let input = input.into_region();
        let output = output.into_region();
        let _native = self.base.guard();
        self.base
            .base
            .prepare_execution(self.stream.as_ref(), &[input, output])?;
        // SAFETY: `cufftXtExec` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cufftXtExec(
                self.base.base.as_raw(),
                input.as_ptr().cast(),
                output.as_ptr().cast(),
                direction as _
            ))?;
        }
        Ok(())
    }

    /// Queues an extended-precision transform over the same writable storage.
    /// The allocation must accommodate both representations, including real-transform padding.
    ///
    /// # Safety
    ///
    /// All logical input samples must be initialized and conform to the configured layout.
    /// The resulting representation may differ from `T`; do not claim untouched padding is initialized.
    /// Retain exclusive storage, the plan, workspaces and callback state through every queued use and graph replay, including failures.
    ///
    /// # Errors
    ///
    /// Returns an error for a distributed plan, mismatched element types,
    /// insufficient buffer capacity, or overflow in the configured storage span.
    /// Context mismatches, dependency ordering, context binding, and cuFFT
    /// submission failures are also returned. Success reports submission only.
    pub unsafe fn execute_in_place_async<T: DataTypeLike>(
        &self,
        memory: DeviceOutputView<'_, T>,
        direction: Direction,
    ) -> Result<()> {
        if self.gpus.len() > 1 || T::data_type() != self.layout.input_type {
            return Err(invalid("in-place input type or GPU count"));
        }
        let input = self
            .layout
            .required_elements(true)?
            .checked_mul(self.layout.input_type.size_of())
            .ok_or_else(|| invalid("input bytes"))?;
        let output = self
            .layout
            .required_elements(false)?
            .checked_mul(self.layout.output_type.size_of())
            .ok_or_else(|| invalid("output bytes"))?;
        let memory = memory.into_region();
        if memory.byte_len() < input.max(output) {
            return Err(invalid("in-place storage"));
        }
        let _native = self.base.guard();
        self.base
            .base
            .prepare_execution(self.stream.as_ref(), &[memory])?;
        // SAFETY: `cufftXtExec` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cufftXtExec(
                self.base.base.as_raw(),
                memory.as_ptr().cast(),
                memory.as_ptr().cast(),
                direction as _
            ))?;
        }
        Ok(())
    }

    fn check_buffer<T: DataTypeLike>(&self, len: usize, input: bool) -> Result<()> {
        check_buffer::<T>(&self.layout, len, input)
    }

    /// Queues a distributed transform over pending allocations from this native plan.
    ///
    /// # Safety
    ///
    /// Input samples must be initialized in the configured distribution when execution reaches them.
    /// Retain exclusive input/output allocations, the plan, workspaces and callback state through completion, errors and every graph replay.
    ///
    /// # Errors
    ///
    /// Returns an error if either allocation belongs to another native plan,
    /// context binding fails, or cuFFT rejects the distributed submission.
    /// Success reports submission only.
    pub unsafe fn execute_distributed_async(
        &self,
        input: &mut DistributedMemory,
        output: &mut DistributedMemory,
        direction: Direction,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.base, &input.allocation.base)
            || !Arc::ptr_eq(&self.base, &output.allocation.base)
        {
            return Err(invalid("distributed allocation plan"));
        }
        let _native = self.base.guard();
        self.base.base.bind()?;
        // SAFETY: `cufftXtExecDescriptor` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cufftXtExecDescriptor(
                self.base.base.as_raw(),
                input.as_raw(),
                output.as_raw(),
                direction as _
            ))?;
        }
        Ok(())
    }

    /// Queues a distributed transform in place without promoting allocation padding.
    ///
    /// # Safety
    ///
    /// Logical input samples must be initialized in the configured distribution.
    /// Retain exclusive allocation access, the plan, workspaces and callbacks through device completion, failures and every graph replay.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation belongs to another native plan, its
    /// format is not `InPlace` or `InPlaceShuffled`, context binding fails, or
    /// cuFFT rejects the distributed submission. Success reports submission only.
    pub unsafe fn execute_distributed_in_place_async(
        &self,
        memory: &mut DistributedMemory,
        direction: Direction,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.base, &memory.allocation.base) {
            return Err(invalid("distributed allocation plan"));
        }
        if !matches!(
            memory.format,
            SubFormat::InPlace | SubFormat::InPlaceShuffled
        ) {
            return Err(invalid("in-place format"));
        }
        let _native = self.base.guard();
        self.base.base.bind()?;
        // SAFETY: `cufftXtExecDescriptor` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cufftXtExecDescriptor(
                self.base.base.as_raw(),
                memory.as_raw(),
                memory.as_raw(),
                direction as _
            ))?;
        }
        Ok(())
    }

    /// Returns the factorization metadata for a distributed one-dimensional transform.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding fails or cuFFT cannot provide
    /// one-dimensional factorization metadata for the configured plan.
    pub fn factors(&self) -> Result<Factors> {
        let _native = self.base.guard();
        let mut factors = sys::cufftXt1dFactors::default();
        self.base.base.bind()?;
        // SAFETY: `cufftXtQueryPlan` receives live handles owned by the wrappers and valid output locations.
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
    /// # Safety
    /// Functions must obey the specified callback ABI, and callback code plus
    /// caller-info memory must remain valid for every queued execution and graph replay, including errors.
    /// Owned copies of the pointer tables remain retained through native plan destruction.
    ///
    /// # Errors
    ///
    /// Returns an error while distributed allocations retain the plan, if either
    /// pointer table lacks one entry per participating GPU, if context binding
    /// fails, or if cuFFT rejects the callback installation.
    ///
    /// # Panics
    ///
    /// Panics if the internal callback attachment list is empty immediately
    /// after appending the new attachment; this indicates an internal invariant violation.
    pub unsafe fn set_callbacks(
        &mut self,
        kind: CallbackType,
        callbacks: &mut [*mut c_void],
        caller_info: &mut [*mut c_void],
    ) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard();
        if callbacks.len() != self.workspace_sizes.len() || caller_info.len() != callbacks.len() {
            return Err(invalid("callback count"));
        }
        self.base.base.bind()?;
        let metadata = CallbackMetadata {
            symbol: None,
            code: Box::new([]),
            callbacks: callbacks.to_vec(),
            caller_info: caller_info.to_vec(),
        };
        let mut attachments = self
            .base
            .callbacks
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        attachments.push(metadata);
        let metadata = attachments.last_mut().expect("metadata was just pushed");
        // SAFETY: `cufftXtSetCallback` receives live handles owned by the wrappers and valid output locations.
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
    /// Returns an error while distributed allocations retain the plan, if context
    /// binding fails, or if cuFFT cannot clear this callback kind.
    pub fn clear_callback(&mut self, kind: CallbackType) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard();
        self.base.base.bind()?;
        // SAFETY: `cufftXtClearCallback` receives live handles owned by the wrappers.
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
    /// Returns an error while distributed allocations retain the plan, if context
    /// binding fails, or if cuFFT rejects the callback kind or shared-memory size.
    pub fn set_callback_shared_size(&mut self, kind: CallbackType, bytes: usize) -> Result<()> {
        self.ensure_unique()?;
        let _native = self.base.guard();
        self.base.base.bind()?;
        // SAFETY: `cufftXtSetCallbackSharedSize` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cufftXtSetCallbackSharedSize(
                self.base.base.as_raw(),
                kind.into(),
                bytes as _
            ))?;
        }
        Ok(())
    }

    /// Borrows the native FFT plan handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> singe_cufft_sys::cufftHandle {
        self.base.base.as_raw()
    }
}

/// Pending distributed storage whose native allocation and plan have shared owners.
/// Owned chunks expose allocation metadata without granting initialized or mutable sample access.
#[derive(Debug)]
pub struct DistributedMemory {
    allocation: Arc<DistributedAllocation>,
    format: SubFormat,
}

#[derive(Debug)]
struct ChunkInfo {
    context: Arc<CudaContext>,
    allocation_id: AllocationId,
    pointer: *mut u8,
    bytes: usize,
}

#[derive(Debug)]
struct DistributedAllocation {
    base: Arc<PlanHandle>,
    raw: *mut sys::cudaLibXtDesc,
    chunks: Vec<ChunkInfo>,
    byte_len: usize,
    layout: Layout,
    stream: Option<Stream>,
}

// SAFETY: host metadata is immutable after construction, and descriptor operations hold the plan gate.
// Queued raw execution requires callers to retain exclusive sample access through completion.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
unsafe impl Send for DistributedAllocation {}

unsafe impl Sync for DistributedAllocation {}

/// Retains one actual device allocation independently of the distributed memory wrapper.
/// Cloning this metadata does not grant sample initialization or exclusive access.
#[derive(Debug, Clone)]
pub struct DistributedChunk {
    allocation: Arc<DistributedAllocation>,
    index: usize,
}

// Native chunk addresses, contexts and ranges are validated once and retained by allocation.
unsafe impl DeviceRegionProvider for DistributedChunk {
    fn region(&self) -> DeviceRegion<'_> {
        let chunk = &self.allocation.chunks[self.index];
        // SAFETY: cuFFT described this chunk for the allocation that `self.allocation` keeps alive.
        unsafe {
            DeviceRegion::from_raw_parts(
                &chunk.context,
                chunk.allocation_id,
                chunk.pointer,
                chunk.bytes,
            )
        }
        .expect("validated distributed allocation region")
    }
}

impl DistributedAllocation {
    fn describe_chunks(&mut self) -> Result<()> {
        self.base.base.bind()?;
        // SAFETY: `self.raw` is null or points at the descriptor cuFFT allocated for this live allocation.
        let descriptor = unsafe { self.raw.as_ref().and_then(|raw| raw.descriptor.as_ref()) }
            .ok_or(Error::NullHandle)?;
        let count = to_usize(descriptor.nGPUs, "GPU count")?;
        if count == 0 || count > descriptor.data.len() {
            return Err(invalid("GPU count"));
        }
        for index in 0..count {
            let bytes = to_usize(descriptor.size[index], "allocation bytes")?;
            self.byte_len = self
                .byte_len
                .checked_add(bytes)
                .ok_or_else(|| invalid("allocation bytes"))?;
            if bytes == 0 {
                continue;
            }
            let pointer = descriptor.data[index].cast::<u8>();
            if pointer.is_null() {
                return Err(Error::NullHandle);
            }
            let mut raw_context: driver::CUcontext = ptr::null_mut();
            // SAFETY: `cuPointerGetAttribute` receives valid output locations.
            unsafe {
                singe_cuda::try_ffi!(driver::cuPointerGetAttribute(
                    (&raw mut raw_context).cast(),
                    driver::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
                    pointer as driver::CUdeviceptr,
                ))?;
            }
            let context = if raw_context == self.base.base.cuda_context().as_raw() {
                Arc::clone(self.base.base.cuda_context())
            } else {
                // Query the pointer's actual context first, then verify the retained primary owner.
                let context =
                    CudaContext::retain_primary_for_device(Device::new(descriptor.GPUs[index]))?;
                if context.as_raw() != raw_context {
                    return Err(invalid("distributed allocation CUDA context"));
                }
                context
            };
            if context.device().id() != descriptor.GPUs[index] {
                return Err(invalid("distributed allocation GPU"));
            }
            let allocation_id = AllocationId::create();
            // SAFETY: cuFFT reported this pointer and size for a live chunk; the region is only validated here.
            unsafe {
                DeviceRegion::from_raw_parts(&context, allocation_id, pointer, bytes)?;
            }
            self.chunks.push(ChunkInfo {
                context,
                allocation_id,
                pointer,
                bytes,
            });
        }
        Ok(())
    }

    fn complete(&self, source: Option<&Self>, submit: impl FnOnce() -> Result<()>) -> Result<()> {
        let _native = self.base.guard();
        if let Some(stream) = &self.stream {
            stream.ensure_not_capturing()?;
        }
        let mut contexts = vec![Arc::clone(self.base.base.cuda_context())];
        for chunk in &self.chunks {
            if !contexts.iter().any(|context| context == &chunk.context) {
                contexts.push(Arc::clone(&chunk.context));
            }
        }
        if let Some(source) = source {
            for chunk in &source.chunks {
                if !contexts.iter().any(|context| context == &chunk.context) {
                    contexts.push(Arc::clone(&chunk.context));
                }
            }
        }
        // Preflight before enqueueing so a rejected synchronous operation has no host borrow to drain.
        for context in &contexts {
            context.synchronize()?;
        }
        let mut drain = ContextDrain {
            contexts,
            drained: false,
        };
        self.base.base.bind()?;
        let result = submit();
        drain.wait()?;
        result
    }
}

struct ContextDrain {
    contexts: Vec<Arc<CudaContext>>,
    drained: bool,
}

impl ContextDrain {
    fn wait(&mut self) -> Result<()> {
        for context in &self.contexts {
            context.synchronize()?;
        }
        self.drained = true;
        Ok(())
    }
}

impl Drop for ContextDrain {
    fn drop(&mut self) {
        if !self.drained && self.wait().is_err() {
            process::abort();
        }
    }
}

impl DistributedMemory {
    /// Native allocation bytes, including physical padding, across all nonempty GPU chunks.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.allocation.byte_len
    }

    /// Iterates over owned metadata for each nonempty GPU allocation.
    ///
    /// Each yielded chunk retains the allocation and plan independently of this
    /// wrapper. It does not grant initialized sample access.
    pub fn chunks(&self) -> impl ExactSizeIterator<Item = DistributedChunk> + '_ {
        (0..self.allocation.chunks.len()).map(|index| DistributedChunk {
            allocation: Arc::clone(&self.allocation),
            index,
        })
    }

    fn check_host_storage<T>(&self, len: usize) -> Result<()> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| invalid("host storage bytes"))?;
        if bytes < self.byte_len() {
            return Err(invalid("host storage bytes"));
        }
        Ok(())
    }

    /// Completes upload of initialized host bytes, including any required native storage padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the host element type differs from the configured
    /// representation, the host storage is too short (including physical padding),
    /// or a size calculation overflows. Stream capture, CUDA context operations
    /// and synchronization, and cuFFT copy failures are also returned.
    pub fn copy_from_host<T: DataTypeLike>(&mut self, values: &[T]) -> Result<()> {
        check_buffer::<T>(&self.allocation.layout, values.len(), true)?;
        self.check_host_storage::<T>(values.len())?;
        // SAFETY: `cufftXtMemcpy` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        self.allocation.complete(None, || unsafe {
            try_ffi!(sys::cufftXtMemcpy(
                self.allocation.base.base.as_raw(),
                self.as_raw().cast(),
                values.as_ptr().cast_mut().cast(),
                sys::cufftXtCopyType::CUFFT_COPY_HOST_TO_DEVICE
            ))
        })
    }

    /// Completes readback into pending host storage without claiming sample or padding validity.
    ///
    /// # Errors
    ///
    /// Returns an error if the host element type differs from the configured
    /// representation, the host storage is too short (including physical padding),
    /// or a size calculation overflows. Stream capture, CUDA context operations
    /// and synchronization, and cuFFT copy failures are also returned.
    pub fn copy_to_host<T: DataTypeLike>(&self, values: &mut [MaybeUninit<T>]) -> Result<()> {
        check_buffer::<T>(&self.allocation.layout, values.len(), false)?;
        self.check_host_storage::<T>(values.len())?;
        // SAFETY: `cufftXtMemcpy` receives live handles owned by the wrappers and valid output locations.
        self.allocation.complete(None, || unsafe {
            try_ffi!(sys::cufftXtMemcpy(
                self.allocation.base.base.as_raw(),
                values.as_mut_ptr().cast(),
                self.as_raw().cast(),
                sys::cufftXtCopyType::CUFFT_COPY_DEVICE_TO_HOST
            ))
        })
    }

    /// Completes readback and exposes only the native copied prefix as initialized values.
    ///
    /// # Safety
    ///
    /// `initialized_elements` must describe a prefix that native readback fully produces as valid `T` values.
    /// This includes any physical padding within that prefix.
    /// Logical FFT sample production alone does not establish initialization for allocation padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the host element type differs from the configured
    /// representation, the host storage is too short (including physical padding),
    /// or a size calculation overflows. Stream capture, CUDA context operations
    /// and synchronization, and cuFFT copy failures are also returned.
    /// Also returns an error if the initialized prefix exceeds either the host
    /// slice or the number of bytes copied, or its byte count overflows.
    pub unsafe fn copy_to_host_unchecked<'host, T: DataTypeLike>(
        &self,
        values: &'host mut [MaybeUninit<T>],
        initialized_elements: usize,
    ) -> Result<&'host mut [T]> {
        let initialized_bytes = initialized_elements
            .checked_mul(size_of::<T>())
            .ok_or_else(|| invalid("initialized prefix bytes"))?;
        if initialized_elements > values.len() || initialized_bytes > self.byte_len() {
            return Err(invalid("initialized prefix"));
        }
        self.copy_to_host(values)?;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { slice::from_raw_parts_mut(values.as_mut_ptr().cast(), initialized_elements) })
    }

    /// Completes a byte copy between allocations from the same native plan.
    /// The destination remains pending; no interpretation of copied padding is introduced.
    ///
    /// # Errors
    ///
    /// Returns an error if the source belongs to another native plan, the
    /// destination has fewer allocation bytes than the source, the stream is
    /// being captured, CUDA context operations or synchronization fail, or cuFFT
    /// cannot perform the copy.
    pub fn copy_from(&mut self, source: &Self) -> Result<()> {
        if !Arc::ptr_eq(&self.allocation.base, &source.allocation.base) {
            return Err(invalid("distributed allocation plan"));
        }
        if self.byte_len() < source.byte_len() {
            return Err(invalid("distributed destination bytes"));
        }
        self.allocation
            // SAFETY: `cufftXtMemcpy` receives live handles owned by the wrappers.
            .complete(Some(&source.allocation), || unsafe {
                try_ffi!(sys::cufftXtMemcpy(
                    self.allocation.base.base.as_raw(),
                    self.as_raw().cast(),
                    source.as_raw().cast(),
                    sys::cufftXtCopyType::CUFFT_COPY_DEVICE_TO_DEVICE
                ))
            })
    }

    /// Borrows the native descriptor; external mutations must preserve its tracked allocation and plan metadata.
    #[must_use]
    pub fn as_raw(&self) -> *mut sys::cudaLibXtDesc {
        self.allocation.raw
    }
}

impl Drop for DistributedAllocation {
    fn drop(&mut self) {
        let _native = self.base.guard();
        if self.base.base.bind().is_ok() {
            // SAFETY: `cufftXtFree` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::cufftXtFree(self.raw)) {
                    #[cfg(debug_assertions)]
                    eprintln!("failed to free distributed cufft allocation: {error}");
                }
            }
        }
    }
}

fn check_buffer<T: DataTypeLike>(layout: &Layout, len: usize, input: bool) -> Result<()> {
    let () = T::LAYOUT;
    let expected_type = if input {
        layout.input_type
    } else {
        layout.output_type
    };
    if T::data_type() != expected_type {
        return Err(invalid("FFT buffer data type"));
    }
    let expected = layout.required_elements(input)?;
    if len < expected {
        return Err(Error::LengthMismatch {
            name: "FFT buffer".into(),
            expected,
            actual: len,
        });
    }
    Ok(())
}

impl_enum_display!(SubFormat, {
    SubFormat::Input => "CUFFT_XT_FORMAT_INPUT",
    SubFormat::Output => "CUFFT_XT_FORMAT_OUTPUT",
    SubFormat::InPlace => "CUFFT_XT_FORMAT_INPLACE",
    SubFormat::InPlaceShuffled => "CUFFT_XT_FORMAT_INPLACE_SHUFFLED",
    SubFormat::Input1dShuffled => "CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED",
    SubFormat::DistributedInput => "CUFFT_XT_FORMAT_DISTRIBUTED_INPUT",
    SubFormat::DistributedOutput => "CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT",
});

impl_enum_display!(WorkAreaPolicy, {
    WorkAreaPolicy::Minimal => "CUFFT_WORKAREA_MINIMAL",
    WorkAreaPolicy::User => "CUFFT_WORKAREA_USER",
    WorkAreaPolicy::Performance => "CUFFT_WORKAREA_PERFORMANCE",
});

impl_enum_display!(CallbackType, {
    CallbackType::LoadComplex => "CUFFT_CB_LD_COMPLEX",
    CallbackType::LoadComplexDouble => "CUFFT_CB_LD_COMPLEX_DOUBLE",
    CallbackType::LoadReal => "CUFFT_CB_LD_REAL",
    CallbackType::LoadRealDouble => "CUFFT_CB_LD_REAL_DOUBLE",
    CallbackType::StoreComplex => "CUFFT_CB_ST_COMPLEX",
    CallbackType::StoreComplexDouble => "CUFFT_CB_ST_COMPLEX_DOUBLE",
    CallbackType::StoreReal => "CUFFT_CB_ST_REAL",
    CallbackType::StoreRealDouble => "CUFFT_CB_ST_REAL_DOUBLE",
});
