//! cuRAND generator handles and generation routines.

use std::{ptr, sync::Arc};

use singe_cuda::{context::Context as CudaContext, stream::Stream, view::DeviceSliceMut};
use singe_curand_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{Method, Ordering, RngType},
};

/// A cuRAND random number generator.
///
/// Device generators write into CUDA device memory and may be associated with a
/// CUDA stream. Host generators write into host slices and run on the CPU. The
/// handle owns the underlying cuRAND generator and destroys it on drop.
#[derive(Debug)]
pub struct Generator {
    handle: Handle,
}

#[derive(Debug)]
struct Handle {
    raw: sys::curandGenerator_t,
    target: Target,
}

#[derive(Debug)]
enum Target {
    Device { cuda_ctx: Arc<CudaContext> },
    Host,
}

// cuRAND generators own mutable generator state.
// The owner may move between threads, but the wrapper does not expose shared concurrent access.
unsafe impl Send for Handle {}

impl Generator {
    /// Creates a device generator of `rng_type` in `cuda_ctx`.
    ///
    /// The context is bound before calling cuRAND. Device generation methods on
    /// the returned generator require device memory allocated for the same CUDA
    /// context.
    ///
    /// # Errors
    ///
    /// Returns an error if cuRAND cannot allocate or initialize the generator,
    /// if `rng_type` is invalid for the linked cuRAND library, or if the CUDA
    /// context cannot be bound.
    pub fn create(cuda_ctx: &Arc<CudaContext>, rng_type: RngType) -> Result<Self> {
        cuda_ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::curandCreateGenerator(&raw mut handle, rng_type.into(),))?;
        }

        Self::from_raw(
            handle,
            Target::Device {
                cuda_ctx: Arc::clone(cuda_ctx),
            },
        )
    }

    /// Creates a host CPU generator of `rng_type`.
    ///
    /// Host generators write to host slices through the `*_host` generation
    /// methods.
    ///
    /// # Errors
    ///
    /// Returns an error if cuRAND cannot allocate or initialize the generator or
    /// if `rng_type` is invalid for the linked cuRAND library.
    pub fn create_host(rng_type: RngType) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::curandCreateGeneratorHost(
                &raw mut handle,
                rng_type.into(),
            ))?;
        }

        Self::from_raw(handle, Target::Host)
    }

    fn from_raw(raw: sys::curandGenerator_t, target: Target) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle { raw, target },
        })
    }

    /// Sets the CUDA stream used by a device generator.
    ///
    /// Passing `None` resets cuRAND to the null stream. The stream must belong
    /// to the same CUDA context used to create the generator.
    ///
    /// # Errors
    ///
    /// Returns an error for host generators, streams from another CUDA context,
    /// uninitialized cuRAND handles, or CUDA context binding failures.
    pub fn set_stream(&self, stream: Option<&Stream>) -> Result<()> {
        let cuda_ctx = self.cuda_context()?;

        match stream {
            Some(stream) => {
                if cuda_ctx.as_ref() != stream.context() {
                    return Err(Error::StreamContextMismatch);
                }
                cuda_ctx.bind()?;
                unsafe {
                    try_ffi!(sys::curandSetStream(self.as_raw(), stream.as_raw()))?;
                }
            }
            None => {
                cuda_ctx.bind()?;
                unsafe {
                    try_ffi!(sys::curandSetStream(self.as_raw(), ptr::null_mut()))?;
                }
            }
        }

        Ok(())
    }

    /// Sets the seed of a pseudorandom generator.
    ///
    /// # Errors
    ///
    /// Returns an error if the generator is not pseudorandom or if cuRAND cannot
    /// update the seed.
    pub fn set_seed(&self, seed: u64) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::curandSetPseudoRandomGeneratorSeed(self.as_raw(), seed,))?;
        }
        Ok(())
    }

    /// Sets the absolute offset of a pseudo- or quasirandom generator.
    ///
    /// # Errors
    ///
    /// Returns an error if cuRAND cannot update the generator offset.
    pub fn set_offset(&self, offset: u64) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::curandSetGeneratorOffset(self.as_raw(), offset))?;
        }
        Ok(())
    }

    /// Sets the result ordering for a pseudo- or quasirandom generator.
    ///
    /// # Errors
    ///
    /// Returns an error if `ordering` is not valid for the generator or if
    /// cuRAND cannot update the ordering.
    pub fn set_ordering(&self, ordering: Ordering) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::curandSetGeneratorOrdering(
                self.as_raw(),
                ordering.into(),
            ))?;
        }
        Ok(())
    }

    /// Sets the dimensionality of a quasirandom generator.
    ///
    /// cuRAND accepts dimensions from 1 through
    /// [`crate::quasi::MAX_QUASI_RANDOM_DIMENSIONS`].
    ///
    /// # Errors
    ///
    /// Returns an error if the generator is not quasirandom or if `dimensions`
    /// is outside cuRAND's supported range.
    pub fn set_quasi_random_dimensions(&self, dimensions: u32) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::curandSetQuasiRandomGeneratorDimensions(
                self.as_raw(),
                dimensions,
            ))?;
        }
        Ok(())
    }

    /// Initializes starting states for the generator.
    ///
    /// # Errors
    ///
    /// Returns an error if cuRAND cannot allocate state, launch initialization
    /// work, or if a prior CUDA failure is pending.
    pub fn generate_seeds(&self) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::curandGenerateSeeds(self.as_raw()))?;
        }
        Ok(())
    }

    /// Generates 32-bit pseudo- or quasirandom values into device memory.
    ///
    /// For 64-bit quasirandom generators, use
    /// [`Generator::generate_u64_device`] instead.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if the output length is
    /// not valid for the generator dimensions, or if cuRAND generation fails.
    pub fn generate_u32_device(&self, output: &mut impl DeviceSliceMut<u32>) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGenerate(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
            ))?;
        }
        Ok(())
    }

    /// Generates 64-bit quasirandom values into device memory.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if the generator is not a
    /// 64-bit quasirandom generator, or if cuRAND generation fails.
    pub fn generate_u64_device(&self, output: &mut impl DeviceSliceMut<u64>) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGenerateLongLong(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
            ))?;
        }
        Ok(())
    }

    /// Generates `f32` values uniformly distributed in `(0.0, 1.0]` into device memory.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if the output length is
    /// not valid for the generator dimensions, or if cuRAND generation fails.
    pub fn generate_uniform_f32_device(&self, output: &mut impl DeviceSliceMut<f32>) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGenerateUniform(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
            ))?;
        }
        Ok(())
    }

    /// Generates `f64` values uniformly distributed in `(0.0, 1.0]` into device memory.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if double precision is
    /// required but unavailable, or if cuRAND generation fails.
    pub fn generate_uniform_f64_device(&self, output: &mut impl DeviceSliceMut<f64>) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGenerateUniformDouble(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
            ))?;
        }
        Ok(())
    }

    /// Generates normally distributed `f32` values into device memory.
    ///
    /// `mean` and `standard_deviation` parameterize the normal distribution.
    /// For pseudorandom generators, cuRAND requires an even output length.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if the output length is
    /// invalid for the generator, or if cuRAND generation fails.
    pub fn generate_normal_f32_device(
        &self,
        output: &mut impl DeviceSliceMut<f32>,
        mean: f32,
        standard_deviation: f32,
    ) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGenerateNormal(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
                mean,
                standard_deviation,
            ))?;
        }
        Ok(())
    }

    /// Generates normally distributed `f64` values into device memory.
    ///
    /// `mean` and `standard_deviation` parameterize the normal distribution.
    /// For pseudorandom generators, cuRAND requires an even output length.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if double precision is
    /// required but unavailable, or if cuRAND generation fails.
    pub fn generate_normal_f64_device(
        &self,
        output: &mut impl DeviceSliceMut<f64>,
        mean: f64,
        standard_deviation: f64,
    ) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGenerateNormalDouble(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
                mean,
                standard_deviation,
            ))?;
        }
        Ok(())
    }

    /// Generates log-normally distributed `f32` values into device memory.
    ///
    /// `mean` and `standard_deviation` parameterize the associated normal
    /// distribution. For pseudorandom generators, cuRAND requires an even output
    /// length.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if the output length is
    /// invalid for the generator, or if cuRAND generation fails.
    pub fn generate_log_normal_f32_device(
        &self,
        output: &mut impl DeviceSliceMut<f32>,
        mean: f32,
        standard_deviation: f32,
    ) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGenerateLogNormal(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
                mean,
                standard_deviation,
            ))?;
        }
        Ok(())
    }

    /// Generates log-normally distributed `f64` values into device memory.
    ///
    /// `mean` and `standard_deviation` parameterize the associated normal
    /// distribution. For pseudorandom generators, cuRAND requires an even output
    /// length.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if double precision is
    /// required but unavailable, or if cuRAND generation fails.
    pub fn generate_log_normal_f64_device(
        &self,
        output: &mut impl DeviceSliceMut<f64>,
        mean: f64,
        standard_deviation: f64,
    ) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGenerateLogNormalDouble(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
                mean,
                standard_deviation,
            ))?;
        }
        Ok(())
    }

    /// Generates Poisson-distributed `u32` values into device memory.
    ///
    /// `lambda` is the Poisson distribution parameter. cuRAND supports positive
    /// values up to 400,000.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if `lambda` is outside
    /// cuRAND's supported range, or if cuRAND generation fails.
    pub fn generate_poisson_u32_device(
        &self,
        output: &mut impl DeviceSliceMut<u32>,
        lambda: f64,
    ) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGeneratePoisson(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
                lambda,
            ))?;
        }
        Ok(())
    }

    /// Generates Poisson-distributed `u32` values into device memory using `method`.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if `lambda` is outside
    /// cuRAND's supported range, if `method` is unsupported, or if cuRAND
    /// generation fails.
    pub fn generate_poisson_u32_device_with_method(
        &self,
        output: &mut impl DeviceSliceMut<u32>,
        lambda: f64,
        method: Method,
    ) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGeneratePoissonMethod(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
                lambda,
                method.into(),
            ))?;
        }
        Ok(())
    }

    /// Generates binomially distributed `u32` values into device memory.
    ///
    /// `trials` is the number of Bernoulli trials and `probability` is the
    /// success probability.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if the distribution
    /// parameters are outside cuRAND's supported range, or if cuRAND generation
    /// fails.
    pub fn generate_binomial_u32_device(
        &self,
        output: &mut impl DeviceSliceMut<u32>,
        trials: u32,
        probability: f64,
    ) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGenerateBinomial(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
                trials,
                probability,
            ))?;
        }
        Ok(())
    }

    /// Generates binomially distributed `u32` values into device memory using `method`.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a host generator, if the distribution
    /// parameters or method are unsupported, or if cuRAND generation fails.
    pub fn generate_binomial_u32_device_with_method(
        &self,
        output: &mut impl DeviceSliceMut<u32>,
        trials: u32,
        probability: f64,
        method: Method,
    ) -> Result<()> {
        self.bind_device()?;
        unsafe {
            try_ffi!(sys::curandGenerateBinomialMethod(
                self.as_raw(),
                output.as_device_mut_ptr(),
                output.len() as sys::size_t,
                trials,
                probability,
                method.into(),
            ))?;
        }
        Ok(())
    }

    /// Generates 32-bit pseudo- or quasirandom values into a host slice.
    ///
    /// For 64-bit quasirandom generators, use [`Generator::generate_u64_host`].
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if the output length is
    /// not valid for the generator dimensions, or if cuRAND generation fails.
    pub fn generate_u32_host(&self, output: &mut [u32]) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGenerate(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
            ))?;
        }
        Ok(())
    }

    /// Generates 64-bit quasirandom values into a host slice.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if the generator is not
    /// a 64-bit quasirandom generator, or if cuRAND generation fails.
    pub fn generate_u64_host(&self, output: &mut [u64]) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGenerateLongLong(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
            ))?;
        }
        Ok(())
    }

    /// Generates `f32` values uniformly distributed in `(0.0, 1.0]` into a host slice.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if the output length is
    /// not valid for the generator dimensions, or if cuRAND generation fails.
    pub fn generate_uniform_f32_host(&self, output: &mut [f32]) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGenerateUniform(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
            ))?;
        }
        Ok(())
    }

    /// Generates `f64` values uniformly distributed in `(0.0, 1.0]` into a host slice.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if double precision is
    /// required but unavailable, or if cuRAND generation fails.
    pub fn generate_uniform_f64_host(&self, output: &mut [f64]) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGenerateUniformDouble(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
            ))?;
        }
        Ok(())
    }

    /// Generates normally distributed `f32` values into a host slice.
    ///
    /// `mean` and `standard_deviation` parameterize the normal distribution.
    /// For pseudorandom generators, cuRAND requires an even output length.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if the output length is
    /// invalid for the generator, or if cuRAND generation fails.
    pub fn generate_normal_f32_host(
        &self,
        output: &mut [f32],
        mean: f32,
        standard_deviation: f32,
    ) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGenerateNormal(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
                mean,
                standard_deviation,
            ))?;
        }
        Ok(())
    }

    /// Generates normally distributed `f64` values into a host slice.
    ///
    /// `mean` and `standard_deviation` parameterize the normal distribution.
    /// For pseudorandom generators, cuRAND requires an even output length.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if double precision is
    /// required but unavailable, or if cuRAND generation fails.
    pub fn generate_normal_f64_host(
        &self,
        output: &mut [f64],
        mean: f64,
        standard_deviation: f64,
    ) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGenerateNormalDouble(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
                mean,
                standard_deviation,
            ))?;
        }
        Ok(())
    }

    /// Generates log-normally distributed `f32` values into a host slice.
    ///
    /// `mean` and `standard_deviation` parameterize the associated normal
    /// distribution. For pseudorandom generators, cuRAND requires an even output
    /// length.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if the output length is
    /// invalid for the generator, or if cuRAND generation fails.
    pub fn generate_log_normal_f32_host(
        &self,
        output: &mut [f32],
        mean: f32,
        standard_deviation: f32,
    ) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGenerateLogNormal(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
                mean,
                standard_deviation,
            ))?;
        }
        Ok(())
    }

    /// Generates log-normally distributed `f64` values into a host slice.
    ///
    /// `mean` and `standard_deviation` parameterize the associated normal
    /// distribution. For pseudorandom generators, cuRAND requires an even output
    /// length.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if double precision is
    /// required but unavailable, or if cuRAND generation fails.
    pub fn generate_log_normal_f64_host(
        &self,
        output: &mut [f64],
        mean: f64,
        standard_deviation: f64,
    ) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGenerateLogNormalDouble(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
                mean,
                standard_deviation,
            ))?;
        }
        Ok(())
    }

    /// Generates Poisson-distributed `u32` values into a host slice.
    ///
    /// `lambda` is the Poisson distribution parameter. cuRAND supports positive
    /// values up to 400,000.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if `lambda` is outside
    /// cuRAND's supported range, or if cuRAND generation fails.
    pub fn generate_poisson_u32_host(&self, output: &mut [u32], lambda: f64) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGeneratePoisson(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
                lambda,
            ))?;
        }
        Ok(())
    }

    /// Generates Poisson-distributed `u32` values into a host slice using `method`.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if `lambda` is outside
    /// cuRAND's supported range, if `method` is unsupported, or if cuRAND
    /// generation fails.
    pub fn generate_poisson_u32_host_with_method(
        &self,
        output: &mut [u32],
        lambda: f64,
        method: Method,
    ) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGeneratePoissonMethod(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
                lambda,
                method.into(),
            ))?;
        }
        Ok(())
    }

    /// Generates binomially distributed `u32` values into a host slice.
    ///
    /// `trials` is the number of Bernoulli trials and `probability` is the
    /// success probability.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if the distribution
    /// parameters are outside cuRAND's supported range, or if cuRAND generation
    /// fails.
    pub fn generate_binomial_u32_host(
        &self,
        output: &mut [u32],
        trials: u32,
        probability: f64,
    ) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGenerateBinomial(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
                trials,
                probability,
            ))?;
        }
        Ok(())
    }

    /// Generates binomially distributed `u32` values into a host slice using `method`.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` is a device generator, if the distribution
    /// parameters or method are unsupported, or if cuRAND generation fails.
    pub fn generate_binomial_u32_host_with_method(
        &self,
        output: &mut [u32],
        trials: u32,
        probability: f64,
        method: Method,
    ) -> Result<()> {
        self.bind_host()?;
        unsafe {
            try_ffi!(sys::curandGenerateBinomialMethod(
                self.as_raw(),
                output.as_mut_ptr(),
                output.len() as sys::size_t,
                trials,
                probability,
                method.into(),
            ))?;
        }
        Ok(())
    }

    /// Binds the CUDA context associated with this generator, if it is a device generator.
    ///
    /// Host generators do not require a CUDA context bind.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound.
    pub fn bind(&self) -> Result<()> {
        if let Target::Device { cuda_ctx } = &self.handle.target {
            cuda_ctx.bind()?;
        }
        Ok(())
    }

    /// Returns the CUDA context associated with a device generator.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotDeviceGenerator`] for host generators.
    pub fn cuda_context(&self) -> Result<&Arc<CudaContext>> {
        match &self.handle.target {
            Target::Device { cuda_ctx } => Ok(cuda_ctx),
            Target::Host => Err(Error::NotDeviceGenerator),
        }
    }

    /// Returns the raw cuRAND generator handle.
    ///
    /// The returned handle is owned by this wrapper and must not be destroyed by
    /// the caller. It is valid only while `self` is alive.
    pub const fn as_raw(&self) -> sys::curandGenerator_t {
        self.handle.raw
    }

    fn bind_device(&self) -> Result<()> {
        self.cuda_context()?.bind()?;
        Ok(())
    }

    fn bind_host(&self) -> Result<()> {
        match self.handle.target {
            Target::Host => Ok(()),
            Target::Device { .. } => Err(Error::NotHostGenerator),
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = sys::curandDestroyGenerator(self.raw);
            }
        }
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn generates_uniform_values() -> Result<()> {
        let generator = Generator::create_host(RngType::PseudoDefault)?;
        generator.set_seed(1234)?;

        let mut output = [0.0f32; 8];
        generator.generate_uniform_f32_host(&mut output)?;

        assert!(output.iter().all(|value| *value > 0.0 && *value <= 1.0));
        Ok(())
    }
}
