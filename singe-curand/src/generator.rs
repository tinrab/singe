//! Completed and raw cuRAND generation with explicit device or host ownership.

use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ptr,
    sync::Arc,
};

use singe_core::{abort_after_cleanup_failure, report_cleanup_failure};
use singe_cuda::{
    context::Context,
    error::Error as CudaError,
    memory::{DeviceMemory, MemoryCopyKind},
    stream::{AsStream, BorrowedStream, Stream},
    view::{AnyBitPatternDeviceRepr, DeviceOutput, DeviceRepr, DeviceViewMut},
};
use singe_curand_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{Ordering, RngType},
    utility::{checked_byte_len, to_size_t},
};

/// A stateful GPU generator retaining its actual context and selected stream.
/// Completed generation accepts pending output storage and writes its full range.
#[derive(Debug)]
pub struct Generator {
    owner: ManuallyDrop<DeviceOwner>,
    stream: Stream,
    rng_type: RngType,
}

/// A CPU generator.
/// Host generation completes before returning borrowed slices.
#[derive(Debug)]
pub struct HostGenerator {
    owner: HostOwner,
    rng_type: RngType,
}

/// Unmanaged device generator ownership, including its actual stream and kind.
/// Dropping these parts does not destroy the native handle.
#[derive(Debug)]
pub struct RawGenerator {
    /// Exclusively owned native device-generator handle.
    pub handle: sys::curandGenerator_t,
    /// Actual stream selected on the native generator, retaining its context.
    pub stream: Stream,
    /// Kind with which the native generator was created.
    pub rng_type: RngType,
}

/// Unmanaged CPU generator ownership.
/// Dropping it does not destroy the handle.
#[derive(Debug)]
pub struct RawHostGenerator {
    /// Exclusively owned native CPU-generator handle.
    pub handle: sys::curandGenerator_t,
    /// Kind with which the native CPU generator was created.
    pub rng_type: RngType,
}

#[derive(Debug)]
struct DeviceOwner {
    raw: sys::curandGenerator_t,
    stream: BorrowedStream,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "CUDA stream handles may be used from any thread, and the owner only destroys its handle"
)]
// SAFETY: cuRAND does not tie a generator to its creating thread, and the owner exposes no methods; it only destroys the generator, after binding its context, from whichever thread drops it.
unsafe impl Send for DeviceOwner {}

// SAFETY: shared references to the owner reach no native state.
unsafe impl Sync for DeviceOwner {}

impl DeviceOwner {
    fn into_handle(self) -> sys::curandGenerator_t {
        let owner = ManuallyDrop::new(self);
        // SAFETY: `owner` is never dropped, so its stream moves out exactly once.
        drop(unsafe { ptr::read(&raw const owner.stream) });
        owner.raw
    }
}

impl Drop for DeviceOwner {
    fn drop(&mut self) {
        // Queued generation may still use the generator state.
        self.stream.synchronize_or_abort();
        // SAFETY: the generator is live and exclusively owned, and synchronizing bound its context.
        if let Err(error) = unsafe { try_ffi!(sys::curandDestroyGenerator(self.raw)) } {
            report_cleanup_failure("destroy", "curand generator", &error);
        }
    }
}

#[derive(Debug)]
struct HostOwner {
    raw: sys::curandGenerator_t,
}

// SAFETY: cuRAND does not tie a CPU generator to its creating thread, and the owner holds the only handle, so moving it moves exclusive use; it is not `Sync`.
unsafe impl Send for HostOwner {}

impl Drop for HostOwner {
    fn drop(&mut self) {
        // SAFETY: the CPU generator is live and exclusively owned.
        if let Err(error) = unsafe { try_ffi!(sys::curandDestroyGenerator(self.raw)) } {
            report_cleanup_failure("destroy", "curand generator", &error);
        }
    }
}

impl Generator {
    /// Creates a GPU generator in the stream's actual context and completes setup.
    ///
    /// [`RngType::PseudoDefault`] selects XORWOW; [`RngType::QuasiDefault`] selects Sobol32.
    /// Pseudorandom generators start with seed and offset zero and the default pseudorandom ordering.
    /// Quasirandom generators start with one dimension, offset zero, and the default quasirandom ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is being captured or its capture state or context cannot be accessed.
    /// Native setup can fail for an unsupported generator kind, allocation or GPU initialization failure, a cuRAND version mismatch, or failure to select the stream.
    /// A null generator handle and errors while completing setup are also reported.
    pub fn create(rng_type: RngType, stream: &Stream) -> Result<Self> {
        stream.complete(|| {
            let mut raw = ptr::null_mut();
            // SAFETY: `raw` is a valid output location, and the capture check bound the stream's context.
            unsafe {
                try_ffi!(sys::curandCreateGenerator(&raw mut raw, rng_type.into()))?;
            }

            // SAFETY: successful creation transfers an exclusive generator in the stream's context.
            // The stream is installed next; if that fails, dropping the generator only waits for the stream and destroys it.
            let generator = unsafe {
                Self::from_raw(RawGenerator {
                    handle: raw,
                    stream: stream.clone(),
                    rng_type,
                })?
            };
            // SAFETY: the generator is live in the bound context, and `stream` belongs to that context and is retained by `generator`.
            unsafe {
                try_ffi!(sys::curandSetStream(generator.as_raw(), stream.as_raw()))?;
            }
            Ok(generator)
        })
    }

    /// Adopts an exclusive device generator and its exact stream and kind.
    ///
    /// # Safety
    ///
    /// The handle must be live, exclusively owned, and created in the stream's actual context with `raw.rng_type`.
    /// It must use `raw.stream`; all prior work must have completed.
    /// Do not change its stream or kind behind this wrapper.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawGenerator) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            owner: ManuallyDrop::new(DeviceOwner {
                raw: raw.handle,
                // SAFETY: the wrapper retains `raw.stream` until this owner drops, directly or through the stream's capture cleanup.
                stream: unsafe { raw.stream.to_unretained() },
            }),
            stream: raw.stream,
            rng_type: raw.rng_type,
        })
    }

    /// Borrows the native handle.
    /// Direct native calls must preserve its actual context and selected stream, respect exclusive access, and keep it alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::curandGenerator_t {
        self.owner.raw
    }

    /// Context that owns the native generator and its selected stream.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.stream.context()
    }

    /// Stream on which this generator produces device output.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Kind with which the generator was created.
    #[must_use]
    pub const fn rng_type(&self) -> RngType {
        self.rng_type
    }

    /// Transfers native ownership with the stream and generator kind.
    #[must_use]
    pub fn into_raw(self) -> RawGenerator {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so the owner and the stream each move out exactly once.
        let (owner, stream) = unsafe {
            (
                ManuallyDrop::take(&mut this.owner),
                ptr::read(&raw const this.stream),
            )
        };
        RawGenerator {
            handle: owner.into_handle(),
            stream,
            rng_type: this.rng_type,
        }
    }

    /// Completes prior work before moving generation to another retained stream.
    ///
    /// # Errors
    ///
    /// Returns [`Error::StreamContextMismatch`] if `stream` belongs to another context.
    /// Returns an error if either stream is being captured, a capture-state or context query fails, or prior work cannot be completed.
    /// Native stream-selection errors are also reported.
    /// The selected stream changes only after those checks succeed.
    pub fn set_stream(&mut self, stream: &Stream) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }

        stream.ensure_not_capturing()?;
        self.sync_scope(|| Ok(()))?;
        // SAFETY: the generator is live, its prior work has completed, `stream` belongs to its context, and the wrapper retains `stream` below.
        unsafe {
            try_ffi!(sys::curandSetStream(self.as_raw(), stream.as_raw()))?;
        }

        // SAFETY: the wrapper retains `stream` until this owner drops, directly or through the stream's capture cleanup.
        self.owner.stream = unsafe { stream.to_unretained() };
        self.stream = stream.clone();
        Ok(())
    }

    fn bind(&self) -> Result<()> {
        self.context().bind()?;
        Ok(())
    }

    fn sync_scope<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        self.stream.complete(f)
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "generator reconfiguration requires exclusive access to native mutable state"
    )]
    fn configure(&mut self, call: impl FnOnce(sys::curandGenerator_t) -> Result<()>) -> Result<()> {
        let raw = self.as_raw();
        self.sync_scope(|| {
            // Prior generation completes before the generator state changes.
            self.stream.synchronize()?;
            call(raw)
        })
    }

    /// Completes initialization of generator state in its selected stream.
    ///
    /// Generation functions initialize state automatically.
    /// Call this method to measure initialization separately from sample generation.
    ///
    /// # Errors
    ///
    /// Returns an error for active or invalid stream capture, context binding or synchronization failure, or a native state-initialization failure.
    /// cuRAND can report allocation failure, an earlier kernel error, or a kernel-launch failure.
    pub fn generate_seeds(&mut self) -> Result<()> {
        // SAFETY: the generator handle is live, and `configure` binds its context.
        self.configure(|raw| unsafe { try_ffi!(sys::curandGenerateSeeds(raw)) })
    }

    /// Enqueues native state initialization without waiting for it.
    ///
    /// # Safety
    ///
    /// Follows the crate's raw execution contract.
    /// Prior use of the generator state must complete before it is reset.
    /// Native setup may need to run once before capture.
    ///
    /// # Errors
    ///
    /// Returns a context-binding error or a cuRAND state-initialization error, including allocation failure, an earlier kernel error, or launch failure.
    pub unsafe fn generate_seeds_async(&mut self) -> Result<()> {
        self.bind()?;
        // SAFETY: `self` owns the live generator and its context is bound; the caller guarantees prior use of the state has completed and keeps the generator and stream alive until the work completes.
        unsafe { try_ffi!(sys::curandGenerateSeeds(self.as_raw())) }
    }

    fn generate<T: DeviceRepr, B: DeviceOutput<T> + ?Sized>(
        &mut self,
        output: &mut B,
        call: impl FnOnce(sys::curandGenerator_t, *mut T, sys::size_t) -> Result<()>,
    ) -> Result<()> {
        if output.context() != self.context() {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        // Capture is rejected before staging memory is allocated.
        self.stream.ensure_not_capturing()?;
        // Native kernels use vector stores that need more than scalar alignment.
        // Preserve CUDA allocation alignment for direct calls, and stage other
        // views without narrowing the public DeviceOutput contract.
        let mut scratch = if !output.is_empty() && !output.as_mut_ptr().addr().is_multiple_of(256) {
            Some(DeviceMemory::<T>::create_uninit(
                self.context(),
                output.len(),
            )?)
        } else {
            None
        };
        self.sync_scope(|| {
            let Some(scratch) = &mut scratch else {
                // SAFETY: the scope waits for the generation before the output and generator borrows end.
                return unsafe { self.enqueue(output, call) };
            };

            // SAFETY: the native producer writes the full aligned allocation, and the copy on the
            // same stream runs after it and after the output's producers. The scope waits for both
            // before the staging allocation and the output borrow are released.
            unsafe {
                self.enqueue(scratch, call)?;
                self.stream.join_regions(&[output.region()])?;
                DeviceMemory::<T>::copy_async(
                    output.as_mut_ptr(),
                    scratch.as_ptr().cast(),
                    output.len(),
                    MemoryCopyKind::DeviceToDevice,
                    &self.stream,
                )?;
            }
            Ok(())
        })
    }

    // The caller upholds the raw execution contract for `output`.
    unsafe fn enqueue<T: DeviceRepr, B: DeviceOutput<T> + ?Sized>(
        &self,
        output: &mut B,
        call: impl FnOnce(sys::curandGenerator_t, *mut T, sys::size_t) -> Result<()>,
    ) -> Result<()> {
        if output.context() != self.context() {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        self.bind()?;
        if output.is_empty() {
            return Ok(());
        }

        self.stream.join_regions(&[output.region()])?;
        call(
            self.as_raw(),
            output.as_mut_ptr(),
            to_size_t(output.len(), "output length")?,
        )
    }
}

impl Drop for Generator {
    fn drop(&mut self) {
        // SAFETY: `drop` runs once, and `self.owner` is not used afterwards.
        let owner = Arc::new(unsafe { ManuallyDrop::take(&mut self.owner) });
        // Destroying the generator is not allowed during capture, so an active capture keeps it until capture ends.
        if let Err(error) = self.stream.defer_drop_until_capture_end(owner) {
            abort_after_cleanup_failure("defer the drop of", "curand generator", &error);
        }
    }
}

impl HostGenerator {
    /// Creates a CPU generator of `rng_type`.
    ///
    /// Default kinds, seeds, offsets, dimensionality, and orderings match [`Generator::create`].
    /// Host generation completes before returning.
    ///
    /// # Errors
    ///
    /// Returns the cuRAND status for an unsupported kind, allocation or initialization failure, or a version mismatch.
    /// A successful creation that returns a null handle is also rejected.
    pub fn create(rng_type: RngType) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a local handle slot that the call writes, and `rng_type` converts to a valid native kind.
        unsafe {
            try_ffi!(sys::curandCreateGeneratorHost(
                &raw mut raw,
                rng_type.into()
            ))?;
        }
        // SAFETY: successful creation returns an exclusive CPU generator.
        unsafe {
            Self::from_raw(RawHostGenerator {
                handle: raw,
                rng_type,
            })
        }
    }

    /// Adopts an exclusively owned CPU generator with matching kind.
    ///
    /// # Safety
    ///
    /// The handle must be a live CPU generator of `raw.rng_type`, exclusively owned and idle.
    /// Transfer ownership once; do not adopt a device generator here.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the raw value transfers ownership of its handle even though its fields are `Copy`"
    )]
    pub unsafe fn from_raw(raw: RawHostGenerator) -> Result<Self> {
        let RawHostGenerator { handle, rng_type } = raw;
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: HostOwner { raw: handle },
            rng_type,
        })
    }

    /// Borrows the CPU handle.
    /// Native calls must remain synchronous and exclusive, and must not destroy the handle or change its target kind.
    #[must_use]
    pub const fn as_raw(&self) -> sys::curandGenerator_t {
        self.owner.raw
    }

    /// Kind with which the CPU generator was created.
    #[must_use]
    pub const fn rng_type(&self) -> RngType {
        self.rng_type
    }

    /// Transfers the CPU handle and kind without destroying native state.
    #[must_use]
    pub fn into_raw(self) -> RawHostGenerator {
        let this = ManuallyDrop::new(self);
        RawHostGenerator {
            handle: this.owner.raw,
            rng_type: this.rng_type,
        }
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "generator reconfiguration requires exclusive access to native mutable state"
    )]
    fn configure(&mut self, call: impl FnOnce(sys::curandGenerator_t) -> Result<()>) -> Result<()> {
        call(self.as_raw())
    }

    /// Initializes the CPU generator's starting states synchronously.
    ///
    /// Generation functions initialize state automatically.
    /// Call this method to measure initialization separately from sample generation.
    ///
    /// # Errors
    ///
    /// Returns the cuRAND status if native state initialization fails, including allocation or generator-initialization errors.
    pub fn generate_seeds(&mut self) -> Result<()> {
        // SAFETY: the CPU generator handle is live and exclusively borrowed.
        self.configure(|raw| unsafe { try_ffi!(sys::curandGenerateSeeds(raw)) })
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "generating samples advances the exclusively borrowed native RNG state"
    )]
    fn generate<T: AnyBitPatternDeviceRepr>(
        &mut self,
        output: &mut [T],
        call: impl FnOnce(sys::curandGenerator_t, *mut T, sys::size_t) -> Result<()>,
    ) -> Result<()> {
        if output.is_empty() {
            return Ok(());
        }

        let length = to_size_t(output.len(), "output length")?;
        if output.as_mut_ptr().addr().is_multiple_of(16) {
            return call(self.as_raw(), output.as_mut_ptr(), length);
        }

        let bytes = checked_byte_len::<T>(output.len(), "output bytes")?;
        let mut scratch = Vec::new();
        scratch.resize_with(bytes.div_ceil(16), || HostBlock {
            _bytes: [MaybeUninit::uninit(); 16],
        });
        let pointer = scratch.as_mut_ptr().cast::<T>();
        call(self.as_raw(), pointer, length)?;
        // SAFETY: successful synchronous native generation writes every requested
        // value. Only u32/u64/f32/f64 instantiate this private helper; the aligned
        // scratch and destination are disjoint and T accepts every initialized bit.
        unsafe {
            ptr::copy_nonoverlapping(pointer, output.as_mut_ptr(), output.len());
        }
        Ok(())
    }
}

#[repr(C, align(16))]
struct HostBlock {
    _bytes: [MaybeUninit<u8>; 16],
}

macro_rules! settings {
    ($owner:ty) => {
        impl $owner {
            /// Sets the pseudorandom seed after completing prior native work.
            ///
            /// Every `u64` seed is valid.
            /// Different seeds select different sequences, but the sequences are not guaranteed to be uncorrelated.
            ///
            /// # Errors
            ///
            /// Returns a cuRAND error if this is not a pseudorandom generator or the native handle is not initialized.
            /// For a device generator, also reports active or invalid capture, context-binding errors, and failures while completing prior work or the update.
            pub fn set_seed(&mut self, seed: u64) -> Result<()> {
                // SAFETY: `configure` passes the live, exclusively borrowed handle, with a device generator's context bound.
                self.configure(|raw| unsafe {
                    try_ffi!(sys::curandSetPseudoRandomGeneratorSeed(raw, seed))
                })
            }
            /// Sets the absolute pseudo- or quasirandom offset after completing prior work.
            ///
            /// `offset` selects an absolute position in the sequence, rather than advancing relative to the current position.
            /// Skip-ahead is unavailable for MTGP32 and MT19937 generators.
            ///
            /// # Errors
            ///
            /// Returns the cuRAND status if the offset update fails.
            /// For a device generator, also reports active or invalid capture, context-binding errors, and synchronization failures.
            pub fn set_offset(&mut self, offset: u64) -> Result<()> {
                // SAFETY: `configure` passes the live, exclusively borrowed handle, with a device generator's context bound.
                self.configure(|raw| unsafe {
                    try_ffi!(sys::curandSetGeneratorOffset(raw, offset))
                })
            }
            /// Sets the native result ordering after completing prior work.
            ///
            /// Choose a pseudorandom ordering for a pseudorandom generator and [`Ordering::QuasiDefault`] for a quasirandom generator.
            /// Available pseudorandom orderings depend on the generator kind and target.
            ///
            /// # Errors
            ///
            /// Returns the cuRAND status if the ordering is unsupported or the generator is not initialized.
            /// For a device generator, also reports capture, context-binding, and synchronization errors.
            pub fn set_ordering(&mut self, ordering: Ordering) -> Result<()> {
                // SAFETY: `configure` passes the live, exclusively borrowed handle, with a device generator's context bound.
                self.configure(|raw| unsafe {
                    try_ffi!(sys::curandSetGeneratorOrdering(raw, ordering.into()))
                })
            }
            /// Sets quasirandom dimensionality after completing prior work.
            ///
            /// `dimensions` must be in `1..=20_000`.
            /// Subsequent output lengths must be multiples of this dimensionality.
            ///
            /// # Errors
            ///
            /// Returns the cuRAND status if the generator is not quasirandom, the dimension count is invalid, or the handle is not initialized.
            /// For a device generator, also reports capture, context-binding, and synchronization errors.
            pub fn set_quasi_random_dimensions(&mut self, dimensions: u32) -> Result<()> {
                // SAFETY: `configure` passes the live, exclusively borrowed handle, with a device generator's context bound.
                self.configure(|raw| unsafe {
                    try_ffi!(sys::curandSetQuasiRandomGeneratorDimensions(
                        raw, dimensions
                    ))
                })
            }
        }
    };
}

settings!(Generator);

settings!(HostGenerator);

macro_rules! generation {
    ($(($name:ident, $enqueue:ident, $ty:ty, $ffi:ident, $doc:literal $(, $arg:ident: $arg_ty:ty)*)),* $(,)?) => {
        impl Generator { $(
            #[doc = $doc]
            /// Completes full-range production and returns its initialized exclusive view.
            /// Errors and unwinding complete before releasing borrowed storage.
            /// Empty outputs skip native generation.
            ///
            /// # Errors
            ///
            /// Returns an error if capture is active or invalid, `output` belongs to another context, or its length cannot fit the native size type.
            /// CUDA errors while binding the context, ordering prior accesses, staging unaligned output, or completing work are propagated.
            /// Native generation errors become [`Error::Curand`]; the distribution's generator and length restrictions still apply.
            /// Errors constructing the initialized output view are also reported.
            pub fn $name<'a, B: DeviceOutput<$ty> + ?Sized>(&mut self, output: &'a mut B $(, $arg: $arg_ty)*) -> Result<DeviceViewMut<'a, $ty>> {
                // SAFETY: `generate` passes the live handle and a pointer to `len` writable elements of `output`.
                self.generate(output, |raw, pointer, len| unsafe { try_ffi!(sys::$ffi(raw, pointer, len $(, $arg)*)) })?;
                // SAFETY: successful completed production initializes the exact
                // output range; this view retains its exclusive source borrow.
                Ok(unsafe { DeviceViewMut::from_raw_parts(output.region(), output.len())? })
            }
            #[doc = $doc]
            /// Enqueues a full-range write on the generator's stream, after the output's producers, without waiting.
            ///
            /// # Safety
            ///
            /// Follows the crate's raw execution contract.
            /// The output must satisfy cuRAND's alignment requirement for vector stores; allocations from CUDA do.
            ///
            /// # Errors
            ///
            /// Returns an error if `output` belongs to another context, context binding or producer ordering fails, or its nonempty length cannot fit the native size type.
            /// Native generation errors become [`Error::Curand`].
            /// Empty outputs skip native generation.
            pub unsafe fn $enqueue<B: DeviceOutput<$ty> + ?Sized>(&mut self, output: &mut B $(, $arg: $arg_ty)*) -> Result<()> {
                // SAFETY: the caller keeps `output`, the generator, and its stream alive and unaccessed until the work completes, and guarantees the alignment cuRAND needs.
                unsafe { self.enqueue(output, |raw, pointer, len| try_ffi!(sys::$ffi(raw, pointer, len $(, $arg)*))) }
            }
        )* }
        impl HostGenerator { $(
            #[doc = $doc]
            /// Generates into a host slice synchronously.
            /// Empty slices skip native generation.
            ///
            /// # Errors
            ///
            /// Returns [`Error::OutOfRange`] if the output length or required staging byte count cannot be represented.
            /// Native generation errors become [`Error::Curand`], including unsupported generator kinds or output lengths for the selected distribution.
            pub fn $name(&mut self, output: &mut [$ty] $(, $arg: $arg_ty)*) -> Result<()> {
                // SAFETY: `generate` passes the live handle and a pointer to `len` writable, aligned elements.
                self.generate(output, |raw, pointer, len| unsafe { try_ffi!(sys::$ffi(raw, pointer, len $(, $arg)*)) })
            }
        )* }
    };
}

generation! {
    (generate_u32, generate_u32_async, u32, curandGenerate, "Generates 32-bit random integers with every bit random.\n\nSupports pseudorandom and 32-bit quasirandom generators. Quasirandom output lengths must be multiples of the configured dimensionality; 64-bit quasirandom generators require `generate_u64` instead."),
    (generate_u64, generate_u64_async, u64, curandGenerateLongLong, "Generates 64-bit quasirandom integers with every bit random.\n\nRequires a 64-bit quasirandom generator and an output length that is a multiple of the configured dimensionality."),
    (generate_uniform_f32, generate_uniform_f32_async, f32, curandGenerateUniform, "Generates uniform `f32` values in `(0, 1]`, excluding zero and including one.\n\nQuasirandom output lengths must be multiples of the configured dimensionality."),
    (generate_uniform_f64, generate_uniform_f64_async, f64, curandGenerateUniformDouble, "Generates uniform `f64` values in `(0, 1]`, excluding zero and including one.\n\nQuasirandom output lengths must be multiples of the configured dimensionality. Device generation requires double-precision support."),
    (generate_normal_f32, generate_normal_f32_async, f32, curandGenerateNormal, "Generates normally distributed `f32` values with `mean` and `standard_deviation`.\n\nPseudorandom generation uses the Box-Muller transform and requires an even output length. Quasirandom generation uses an inverse cumulative distribution and requires a length that is a multiple of the configured dimensionality. CPU, GPU, and different cuRAND versions may produce slightly different values because their transcendental functions can differ.", mean: f32, standard_deviation: f32),
    (generate_normal_f64, generate_normal_f64_async, f64, curandGenerateNormalDouble, "Generates normally distributed `f64` values with `mean` and `standard_deviation`.\n\nPseudorandom generation uses the Box-Muller transform and requires an even output length. Quasirandom generation uses an inverse cumulative distribution and requires a length that is a multiple of the configured dimensionality. Device generation requires double-precision support. CPU, GPU, and different cuRAND versions may produce slightly different values because their transcendental functions can differ.", mean: f64, standard_deviation: f64),
    (generate_log_normal_f32, generate_log_normal_f32_async, f32, curandGenerateLogNormal, "Generates log-normally distributed `f32` values by exponentiating normal variates.\n\n`mean` and `standard_deviation` describe the underlying normal distribution. Pseudorandom generation uses the Box-Muller transform and requires an even output length. Quasirandom generation uses an inverse cumulative distribution and requires a length that is a multiple of the configured dimensionality. CPU, GPU, and different cuRAND versions may produce slightly different values because their transcendental functions can differ.", mean: f32, standard_deviation: f32),
    (generate_log_normal_f64, generate_log_normal_f64_async, f64, curandGenerateLogNormalDouble, "Generates log-normally distributed `f64` values by exponentiating normal variates.\n\n`mean` and `standard_deviation` describe the underlying normal distribution. Pseudorandom generation uses the Box-Muller transform and requires an even output length. Quasirandom generation uses an inverse cumulative distribution and requires a length that is a multiple of the configured dimensionality. Device generation requires double-precision support. CPU, GPU, and different cuRAND versions may produce slightly different values because their transcendental functions can differ.", mean: f64, standard_deviation: f64),
    (generate_poisson_u32, generate_poisson_u32_async, u32, curandGeneratePoisson, "Generates Poisson-distributed `u32` values with mean `lambda`, where `0 < lambda <= 400_000`.\n\nQuasirandom output lengths must be multiples of the configured dimensionality. Device generation requires double-precision support. Native allocation, initialization, range, and kernel errors are propagated.", lambda: f64),
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn test_generates_uniform_values() -> Result<()> {
        let mut generator = HostGenerator::create(RngType::PseudoDefault)?;
        generator.set_seed(1234)?;

        let mut output = [0.0f32; 8];
        generator.generate_uniform_f32(&mut output)?;

        assert!(output.iter().all(|value| *value > 0.0 && *value <= 1.0));
        Ok(())
    }
}
