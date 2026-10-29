//! Completed and raw cuRAND generation with explicit device or host ownership.

use std::{
    mem::{ManuallyDrop, MaybeUninit, size_of},
    process, ptr,
    result::Result as StdResult,
    sync::Arc,
};

use singe_cuda::{
    context::Context,
    error::Error as CudaError,
    memory::{DeviceMemory, MemoryCopyKind},
    stream::Stream,
    view::{AnyBitPatternDeviceRepr, DeviceOutput, DeviceRepr, DeviceViewMut},
};
use singe_cuda_sys::runtime::cudaStream_t;
use singe_curand_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{Method, Ordering, RngType},
    utility::{drain_stream, require_eager, to_size_t},
};

/// A stateful GPU generator retaining its actual context and selected stream.
/// Completed generation accepts pending output storage and writes its full range.
#[derive(Debug)]
pub struct Generator {
    owner: Arc<DeviceOwner>,
    stream: Arc<Stream>,
    rng_type: RngType,
}

/// A CPU generator. Host generation completes before returning borrowed slices.
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
    pub stream: Arc<Stream>,
    /// Kind with which the native generator was created.
    pub rng_type: RngType,
}

/// Unmanaged CPU generator ownership. Dropping it does not destroy the handle.
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
    context: Arc<Context>,
    stream: cudaStream_t,
}

// SAFETY: mutation requires an exclusive Generator borrow; shared ownership exists only
// for deferred capture cleanup, which binds the actual context on final release.
unsafe impl Send for DeviceOwner {}

unsafe impl Sync for DeviceOwner {}

impl Drop for DeviceOwner {
    fn drop(&mut self) {
        // SAFETY: the typed owner or its capture registry keeps this stream live.
        unsafe {
            drain_stream(&self.context, self.stream);
        }
        // SAFETY: `curandDestroyGenerator` receives live handles owned by the wrappers.
        if let Err(error) = unsafe { try_ffi!(sys::curandDestroyGenerator(self.raw)) } {
            #[cfg(debug_assertions)]
            eprintln!("failed to destroy curand generator: {error}");
        }
        // SAFETY: native destruction may enqueue stream-ordered frees; the stream is still live.
        unsafe {
            drain_stream(&self.context, self.stream);
        }
    }
}

#[derive(Debug)]
struct HostOwner {
    raw: sys::curandGenerator_t,
}

// SAFETY: exclusive native state may move to another thread; no shared mutation is exposed.
unsafe impl Send for HostOwner {}

impl Drop for HostOwner {
    fn drop(&mut self) {
        // SAFETY: `curandDestroyGenerator` receives live handles owned by the wrappers.
        if let Err(error) = unsafe { try_ffi!(sys::curandDestroyGenerator(self.raw)) } {
            #[cfg(debug_assertions)]
            eprintln!("failed to destroy curand generator: {error}");
        }
    }
}

impl Generator {
    /// Creates a GPU generator in the stream's actual context and completes setup.
    ///
    /// [`RngType::PseudoDefault`] selects XORWOW; [`RngType::QuasiDefault`]
    /// selects Sobol32. Pseudorandom generators start with seed and offset zero
    /// and the default pseudorandom ordering. Quasirandom generators start with
    /// one dimension, offset zero, and the default quasirandom ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is being captured or its capture state or
    /// context cannot be accessed. Native setup can fail for an unsupported
    /// generator kind, allocation or GPU initialization failure, a cuRAND
    /// version mismatch, or failure to select the stream. A null generator
    /// handle and errors while completing setup are also reported.
    ///
    /// # Panics
    ///
    /// Panics if successful native setup did not install generator ownership,
    /// which indicates an internal invariant failure.
    pub fn create(rng_type: RngType, stream: &Arc<Stream>) -> Result<Self> {
        require_eager(stream)?;
        let mut raw = ptr::null_mut();
        let mut generator = None;
        stream.sync_scope(|_| {
            Ok((|| -> Result<()> {
                // SAFETY: `curandCreateGenerator` receives valid output locations.
                unsafe {
                    try_ffi!(sys::curandCreateGenerator(&raw mut raw, rng_type.into()))?;
                }
                // SAFETY: successful creation produces exclusive device ownership.
                // Install cleanup before a completion failure can return from setup.
                generator = Some(unsafe { Self::from_raw(raw, rng_type, stream)? });
                unsafe { try_ffi!(sys::curandSetStream(raw, stream.as_raw())) }
            })())
        })??;
        Ok(generator.expect("successful native creation installs ownership"))
    }

    /// Adopts an exclusive device generator and its exact stream and kind.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw` is null.
    ///
    /// # Safety
    /// The handle must be live, exclusively owned, and created in this actual
    /// context with `rng_type`. It must use `stream`; all prior work must have
    /// completed. Do not change its stream or kind behind this wrapper.
    pub unsafe fn from_raw(
        raw: sys::curandGenerator_t,
        rng_type: RngType,
        stream: &Arc<Stream>,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: Arc::new(DeviceOwner {
                raw,
                context: Arc::clone(stream.context()),
                stream: stream.as_raw(),
            }),
            stream: Arc::clone(stream),
            rng_type,
        })
    }

    /// Borrows the native handle. Direct native calls must preserve its actual
    /// context and selected stream, respect exclusive access, and keep it alive.
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
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Kind with which the generator was created.
    #[must_use]
    pub const fn rng_type(&self) -> RngType {
        self.rng_type
    }

    /// Binds the retained actual context on the calling host thread.
    ///
    /// # Errors
    ///
    /// Returns the CUDA error if the context cannot be bound to this thread.
    pub fn bind(&self) -> Result<()> {
        self.context().bind()?;
        Ok(())
    }

    /// Transfers native ownership with the stream and generator kind.
    ///
    /// # Errors
    ///
    /// Returns the original generator if its native owner is still shared.
    /// Ownership and the selected stream remain intact in that case.
    pub fn into_raw(self) -> StdResult<RawGenerator, Self> {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: move the owner once while suppressing Generator destruction.
        let owner = unsafe { ptr::read(&raw const this.owner) };
        match Arc::try_unwrap(owner) {
            Ok(owner) => {
                let owner = ManuallyDrop::new(owner);
                // SAFETY: move remaining owning fields and release only the duplicate context.
                let stream = unsafe { ptr::read(&raw const this.stream) };
                // SAFETY: `owner` is never dropped, so its context is read and released exactly once.
                unsafe {
                    drop(ptr::read(&raw const owner.context));
                }
                Ok(RawGenerator {
                    handle: owner.raw,
                    stream,
                    rng_type: this.rng_type,
                })
            }
            Err(owner) => {
                // SAFETY: `this.owner` was moved out above, so writing it back restores the only copy.
                unsafe {
                    ptr::write(&raw mut this.owner, owner);
                }
                Err(ManuallyDrop::into_inner(this))
            }
        }
    }

    /// Completes prior work before moving generation to another retained stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the streams belong to different contexts, either
    /// stream is being captured, a capture-state or context query fails, or
    /// prior work cannot be completed. Native stream-selection errors are also
    /// reported. The selected stream changes only after those checks succeed.
    ///
    /// # Panics
    ///
    /// Panics if the native generator owner is still shared when the stream
    /// field is updated, which violates the wrapper's ownership invariant.
    pub fn set_stream(&mut self, stream: &Arc<Stream>) -> Result<()> {
        if stream.context() != self.context() {
            return Err(CudaError::MemoryContextMismatch.into());
        }
        require_eager(&self.stream)?;
        require_eager(stream)?;
        self.stream.sync_scope(|_| Ok(()))?;
        // curandSetStream only reports an uninitialized handle or success;
        // it cannot fail halfway through a stream switch for a live generator.
        // SAFETY: `curandSetStream` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::curandSetStream(self.as_raw(), stream.as_raw()))?;
        }
        Arc::get_mut(&mut self.owner)
            .expect("live generator owner is exclusive")
            .stream = stream.as_raw();
        self.stream = Arc::clone(stream);
        Ok(())
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "generator reconfiguration requires exclusive access to native mutable state"
    )]
    fn configure(&mut self, call: impl FnOnce(sys::curandGenerator_t) -> Result<()>) -> Result<()> {
        require_eager(&self.stream)?;
        let stream = Arc::clone(&self.stream);
        stream.sync_scope(|_| {
            stream.synchronize()?;
            Ok(call(self.as_raw()))
        })??;
        Ok(())
    }

    /// Completes initialization of generator state in its selected stream.
    ///
    /// Generation functions initialize state automatically. Call this method to
    /// measure initialization separately from sample generation.
    ///
    /// # Errors
    ///
    /// Returns an error for active or invalid stream capture, context binding
    /// or synchronization failure, or a native state-initialization failure.
    /// cuRAND can report allocation failure, an earlier kernel error, or a
    /// kernel-launch failure.
    pub fn generate_seeds(&mut self) -> Result<()> {
        // SAFETY: `curandGenerateSeeds` receives only plain values.
        self.configure(|raw| unsafe { try_ffi!(sys::curandGenerateSeeds(raw)) })
    }

    /// Enqueues native state initialization without retaining asynchronous dependencies.
    ///
    /// # Errors
    ///
    /// Returns a context-binding error or a cuRAND state-initialization error,
    /// including allocation failure, an earlier kernel error, or launch failure.
    ///
    /// # Safety
    /// Retain the generator and stream until completion, including after errors.
    /// Complete prior use before resetting state. During capture, retain them
    /// through every graph replay; native setup may require warming before capture.
    pub unsafe fn generate_seeds_async(&mut self) -> Result<()> {
        self.bind()?;
        // SAFETY: `curandGenerateSeeds` receives live handles owned by the wrappers.
        unsafe { try_ffi!(sys::curandGenerateSeeds(self.as_raw())) }
    }

    /// Enqueues the SDK-internal Poisson method selector.
    ///
    /// # Errors
    ///
    /// Returns an error if `output` belongs to another context, context binding
    /// fails, or its nonempty length cannot fit the native size type. For a
    /// nonempty output, cuRAND errors from the selected method are propagated.
    /// An empty output performs no native generation call.
    ///
    /// # Safety
    /// Establish the method's validity for this SDK and generator kind. It must
    /// write exactly the output range with the native alignment. Retain the
    /// output, generator, and stream until completion on every path; capture
    /// additionally requires retaining all dependencies through graph replay.
    /// This entry point is marked internal by NVIDIA and is not used by completed helpers.
    pub unsafe fn generate_poisson_u32_with_method_async<B: DeviceOutput<u32> + ?Sized>(
        &mut self,
        output: &mut B,
        lambda: f64,
        method: Method,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            self.enqueue(output, |raw, pointer, len| {
                try_ffi!(sys::curandGeneratePoissonMethod(
                    raw,
                    pointer,
                    len,
                    lambda,
                    method.into()
                ))
            })
        }
    }

    fn generate<T: DeviceRepr, B: DeviceOutput<T> + ?Sized>(
        &mut self,
        output: &mut B,
        call: impl FnOnce(sys::curandGenerator_t, *mut T, sys::size_t) -> Result<()>,
    ) -> Result<()> {
        require_eager(&self.stream)?;
        if output.context() != self.context() {
            return Err(CudaError::MemoryContextMismatch.into());
        }
        let stream = Arc::clone(&self.stream);
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
        stream.sync_scope(|_| {
            Ok((|| -> Result<()> {
                stream.join_regions(&[output.region()])?;
                if let Some(scratch) = &mut scratch {
                    // SAFETY: the native producer writes the full aligned allocation;
                    // the same stream orders the copy after that producer. Both owners
                    // and the exclusive output borrow survive scope completion.
                    unsafe {
                        self.enqueue(scratch, call)?;
                        DeviceMemory::<T>::copy_async(
                            output.as_mut_ptr(),
                            scratch.as_ptr().cast(),
                            output.len(),
                            MemoryCopyKind::DeviceToDevice,
                            &stream,
                        )?;
                    }
                    Ok(())
                } else {
                    // SAFETY: output and generator borrows survive scope completion.
                    unsafe { self.enqueue(output, call) }
                }
            })())
        })??;
        Ok(())
    }

    unsafe fn enqueue<T: DeviceRepr, B: DeviceOutput<T> + ?Sized>(
        &mut self,
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
        call(
            self.as_raw(),
            output.as_mut_ptr(),
            to_size_t(output.len(), "output length")?,
        )
    }
}

impl Drop for Generator {
    fn drop(&mut self) {
        if self
            .stream
            .defer_drop_until_capture_end(Arc::clone(&self.owner))
            .is_err()
        {
            process::abort();
        }
    }
}

impl HostGenerator {
    /// Creates a CPU generator of `rng_type`.
    ///
    /// Default kinds, seeds, offsets, dimensionality, and orderings match
    /// [`Generator::create`]. Host generation completes before returning.
    ///
    /// # Errors
    ///
    /// Returns the cuRAND status for an unsupported kind, allocation or
    /// initialization failure, or a version mismatch. A successful creation
    /// that returns a null handle is also rejected.
    pub fn create(rng_type: RngType) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `curandCreateGeneratorHost` receives valid output locations.
        unsafe {
            try_ffi!(sys::curandCreateGeneratorHost(
                &raw mut raw,
                rng_type.into()
            ))?;
        }
        // SAFETY: successful creation returns an exclusive CPU generator.
        unsafe { Self::from_raw(raw, rng_type) }
    }

    /// Adopts an exclusively owned CPU generator with matching kind.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw` is null.
    ///
    /// # Safety
    /// The handle must be a live CPU generator of `rng_type`, exclusively owned
    /// and idle. Transfer ownership once; do not adopt a device generator here.
    pub unsafe fn from_raw(raw: sys::curandGenerator_t, rng_type: RngType) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: HostOwner { raw },
            rng_type,
        })
    }

    /// Borrows the CPU handle. Native calls must remain synchronous and
    /// exclusive, and must not destroy the handle or change its target kind.
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
    /// Generation functions initialize state automatically. Call this method to
    /// measure initialization separately from sample generation.
    ///
    /// # Errors
    ///
    /// Returns the cuRAND status if native state initialization fails, including
    /// allocation or generator-initialization errors.
    pub fn generate_seeds(&mut self) -> Result<()> {
        // SAFETY: `curandGenerateSeeds` receives only plain values.
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
        let bytes = output
            .len()
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::out_of_range("output bytes"))?;
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
            /// Every `u64` seed is valid. Different seeds select different
            /// sequences, but the sequences are not guaranteed to be uncorrelated.
            ///
            /// # Errors
            ///
            /// Returns a cuRAND error if this is not a pseudorandom generator or
            /// the native handle is not initialized. For a device generator,
            /// also reports active or invalid capture, context-binding errors,
            /// and failures while completing prior work or the update.
            pub fn set_seed(&mut self, seed: u64) -> Result<()> {
                // SAFETY: `curandSetPseudoRandomGeneratorSeed` receives only plain values.
                self.configure(|raw| unsafe {
                    try_ffi!(sys::curandSetPseudoRandomGeneratorSeed(raw, seed))
                })
            }
            /// Sets the absolute pseudo- or quasirandom offset after completing prior work.
            ///
            /// `offset` selects an absolute position in the sequence, rather
            /// than advancing relative to the current position. Skip-ahead is
            /// unavailable for MTGP32 and MT19937 generators.
            ///
            /// # Errors
            ///
            /// Returns the cuRAND status if the offset update fails. For a
            /// device generator, also reports active or invalid capture,
            /// context-binding errors, and synchronization failures.
            pub fn set_offset(&mut self, offset: u64) -> Result<()> {
                // SAFETY: `curandSetGeneratorOffset` receives only plain values.
                self.configure(|raw| unsafe {
                    try_ffi!(sys::curandSetGeneratorOffset(raw, offset))
                })
            }
            /// Sets the native result ordering after completing prior work.
            ///
            /// Choose a pseudorandom ordering for a pseudorandom generator and
            /// [`Ordering::QuasiDefault`] for a quasirandom generator. Available
            /// pseudorandom orderings depend on the generator kind and target.
            ///
            /// # Errors
            ///
            /// Returns the cuRAND status if the ordering is unsupported or the
            /// generator is not initialized. For a device generator, also
            /// reports capture, context-binding, and synchronization errors.
            pub fn set_ordering(&mut self, ordering: Ordering) -> Result<()> {
                // SAFETY: `curandSetGeneratorOrdering` receives only plain values.
                self.configure(|raw| unsafe {
                    try_ffi!(sys::curandSetGeneratorOrdering(raw, ordering.into()))
                })
            }
            /// Sets quasirandom dimensionality after completing prior work.
            ///
            /// `dimensions` must be in `1..=20_000`. Subsequent output lengths
            /// must be multiples of this dimensionality.
            ///
            /// # Errors
            ///
            /// Returns the cuRAND status if the generator is not quasirandom,
            /// the dimension count is invalid, or the handle is not initialized.
            /// For a device generator, also reports capture, context-binding,
            /// and synchronization errors.
            pub fn set_quasi_random_dimensions(&mut self, dimensions: u32) -> Result<()> {
                // SAFETY: `curandSetQuasiRandomGeneratorDimensions` receives only plain values.
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
            /// Returns an error if capture is active or invalid, `output`
            /// belongs to another context, or its length cannot fit the native
            /// size type. CUDA errors while binding the context, ordering prior
            /// accesses, staging unaligned output, or completing work are
            /// propagated. Native generation errors become [`Error::Curand`];
            /// the distribution's generator and length restrictions still apply.
            /// Errors constructing the initialized output view are also reported.
            pub fn $name<'a, B: DeviceOutput<$ty> + ?Sized>(&mut self, output: &'a mut B $(, $arg: $arg_ty)*) -> Result<DeviceViewMut<'a, $ty>> {
                // SAFETY: the `ffi` entry point receives only plain values.
                self.generate(output, |raw, pointer, len| unsafe { try_ffi!(sys::$ffi(raw, pointer, len $(, $arg)*)) })?;
                // SAFETY: successful completed production initializes the exact
                // output range; this view retains its exclusive source borrow.
                Ok(unsafe { DeviceViewMut::from_raw_parts(output.region(), output.len())? })
            }
            #[doc = $doc]
            /// Enqueues a full-range write without retaining asynchronous dependencies.
            ///
            /// # Errors
            ///
            /// Returns an error if `output` belongs to another context, context
            /// binding fails, or its nonempty length cannot fit the native size
            /// type. Native generation errors become [`Error::Curand`]. Empty
            /// outputs skip native generation; no completion wait is performed.
            ///
            /// # Safety
            /// Establish output readiness and exclusive access in the selected stream.
            /// Retain output allocations, the generator, and stream through completion,
            /// including after native errors. During capture retain every dependency
            /// unchanged through graph replays. Native alignment requirements apply.
            pub unsafe fn $enqueue<B: DeviceOutput<$ty> + ?Sized>(&mut self, output: &mut B $(, $arg: $arg_ty)*) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
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
            /// Returns [`Error::OutOfRange`] if the output length or required
            /// staging byte count cannot be represented. Native generation
            /// errors become [`Error::Curand`], including unsupported generator
            /// kinds or output lengths for the selected distribution.
            pub fn $name(&mut self, output: &mut [$ty] $(, $arg: $arg_ty)*) -> Result<()> {
                // SAFETY: the `ffi` entry point receives only plain values.
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
