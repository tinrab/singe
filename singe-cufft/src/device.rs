//! Experimental cuFFT device-function discovery and code extraction.

use std::mem::ManuallyDrop;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{LibraryVersion, impl_enum_conversion, impl_enum_display};
use singe_cuda::nvjitlink::{InputType, Linker};
use singe_cufft_sys::device as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    utility::to_usize,
};

/// Direction of a generated device FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Direction {
    /// Forward Fourier transform.
    Forward = sys::cufftDescriptionDirection::CUFFT_DESC_FORWARD as _,
    /// Inverse Fourier transform.
    Inverse = sys::cufftDescriptionDirection::CUFFT_DESC_INVERSE as _,
}

impl_enum_conversion!(sys::cufftDescriptionDirection, Direction);

/// Data transformation performed by a generated FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Transform {
    /// Complex-to-complex Fourier transform.
    ComplexToComplex = sys::cufftDescriptionType::CUFFT_DESC_C2C as _,
    /// Complex-to-real Fourier transform.
    ComplexToReal = sys::cufftDescriptionType::CUFFT_DESC_C2R as _,
    /// Real-to-complex Fourier transform.
    RealToComplex = sys::cufftDescriptionType::CUFFT_DESC_R2C as _,
}

impl_enum_conversion!(sys::cufftDescriptionType, Transform);

/// Precision of generated device FFT elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Precision {
    /// Single precision.
    Single = sys::cufftDescriptionPrecision::CUFFT_DESC_SINGLE as _,
    /// Double precision.
    Double = sys::cufftDescriptionPrecision::CUFFT_DESC_DOUBLE as _,
    /// Half precision.
    Half = sys::cufftDescriptionPrecision::CUFFT_DESC_HALF as _,
}

impl_enum_conversion!(sys::cufftDescriptionPrecision, Precision);

/// Integer trait configuring a device FFT description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DescriptionTrait {
    /// FFT length.
    Size = sys::cufftDescriptionTrait::CUFFT_DESC_TRAIT_SIZE as _,
    /// Transform direction.
    Direction = sys::cufftDescriptionTrait::CUFFT_DESC_TRAIT_DIRECTION as _,
    /// Transform precision.
    Precision = sys::cufftDescriptionTrait::CUFFT_DESC_TRAIT_PRECISION as _,
    /// Transform type.
    Transform = sys::cufftDescriptionTrait::CUFFT_DESC_TRAIT_TYPE as _,
    /// Target streaming-multiprocessor architecture.
    Sm = sys::cufftDescriptionTrait::CUFFT_DESC_TRAIT_SM as _,
    /// Number of elements processed by each thread.
    ElementsPerThread = sys::cufftDescriptionTrait::CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD as _,
}

impl_enum_conversion!(sys::cufftDescriptionTrait, DescriptionTrait);

/// Resource requirements of one device FFT function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FunctionTrait {
    /// Number of elements processed by each thread.
    ElementsPerThread =
        sys::cufftDeviceFunctionTrait::CUFFT_DEVICE_FUNC_TRAIT_ELEMENTS_PER_THREAD as _,
    /// Per-thread storage size required by the function.
    StorageSize = sys::cufftDeviceFunctionTrait::CUFFT_DEVICE_FUNC_TRAIT_STORAGE_SIZE as _,
    /// Shared memory required per FFT.
    SharedMemoryPerFft =
        sys::cufftDeviceFunctionTrait::CUFFT_DEVICE_FUNC_TRAIT_SHARED_MEMORY_PER_FFT as _,
    /// Number of threads per FFT.
    ThreadsPerFft = sys::cufftDeviceFunctionTrait::CUFFT_DEVICE_FUNC_TRAIT_THREADS_PER_FFT as _,
    /// Suggested number of FFTs per thread block.
    SuggestedFftsPerBlock =
        sys::cufftDeviceFunctionTrait::CUFFT_DEVICE_FUNC_TRAIT_SUGGESTED_FFTS_PER_BLOCK as _,
}

impl_enum_conversion!(sys::cufftDeviceFunctionTrait, FunctionTrait);

/// Container format of extracted device code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeContainer {
    /// Link-time optimization intermediate representation.
    LtoIr,
    /// CUDA fatbinary code container.
    Fatbin,
}

/// Owned device code returned by cuFFT, suitable for linking into a CUDA kernel.
#[derive(Debug, Clone)]
pub struct DeviceCode {
    /// Format used to interpret and link `bytes`.
    pub container: CodeContainer,
    /// Owned code container returned by cuFFT, including its native container headers.
    pub bytes: Vec<u8>,
}

/// Owned FFT description. The experimental interface may change in future cuFFT versions.
#[derive(Debug)]
pub struct Description {
    raw: sys::cufftDescriptionHandle,
}

impl Description {
    /// Creates a device FFT description and sets its transform traits.
    ///
    /// # Errors
    ///
    /// Returns a cuFFT error if description creation fails or any supplied transform
    /// trait is rejected, including an unsupported size, precision, or architecture.
    pub fn create(
        size: i64,
        direction: Direction,
        precision: Precision,
        transform: Transform,
        sm: i64,
    ) -> Result<Self> {
        let mut raw = 0;
        // SAFETY: `cufftDescriptionCreate` receives valid output locations.
        unsafe {
            try_ffi!(sys::cufftDescriptionCreate(&raw mut raw))?;
        }
        let mut result = Self { raw };
        for (attribute, value) in [
            (DescriptionTrait::Size, size),
            (DescriptionTrait::Direction, direction as i64),
            (DescriptionTrait::Precision, precision as i64),
            (DescriptionTrait::Transform, transform as i64),
            (DescriptionTrait::Sm, sm),
        ] {
            result.set_trait(attribute, value)?;
        }
        Ok(result)
    }

    /// Sets an integer description trait before creating a device program.
    ///
    /// # Errors
    ///
    /// Returns a cuFFT error if the description trait or its value is unsupported.
    pub fn set_trait(&mut self, attribute: DescriptionTrait, value: i64) -> Result<()> {
        // SAFETY: `cufftDescriptionSetTraitInt64` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cufftDescriptionSetTraitInt64(
                self.raw,
                attribute.into(),
                value
            ))?;
        }
        Ok(())
    }

    /// Returns an integer description trait.
    ///
    /// # Errors
    ///
    /// Returns a cuFFT error if the trait cannot be read from this description.
    pub fn trait_value(&self, attribute: DescriptionTrait) -> Result<i64> {
        let mut value = 0;
        // SAFETY: `cufftDescriptionGetTraitInt64` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftDescriptionGetTraitInt64(
                self.raw,
                attribute.into(),
                &raw mut value
            ))?;
        }
        Ok(value)
    }

    /// Borrows the native description handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cufftDescriptionHandle {
        self.raw
    }

    /// Takes ownership of a cuFFT device description.
    /// # Safety
    /// `raw` must be a live, exclusively owned description handle.
    #[must_use]
    pub unsafe fn from_raw(raw: sys::cufftDescriptionHandle) -> Self {
        Self { raw }
    }

    /// Transfers the description handle to the caller, which must eventually destroy it.
    #[must_use]
    pub fn into_raw(self) -> sys::cufftDescriptionHandle {
        ManuallyDrop::new(self).raw
    }
}

impl Drop for Description {
    fn drop(&mut self) {
        // SAFETY: `cufftDescriptionDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cufftDescriptionDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cufft device description: {error}");
            }
        }
    }
}

/// Device-function database retaining the FFT descriptions used to create it.
#[derive(Debug)]
pub struct DeviceProgram {
    raw: sys::cufftDeviceHandle,
    descriptions: Vec<Description>,
}

impl DeviceProgram {
    /// Creates the function database for these FFT descriptions.
    ///
    /// # Errors
    ///
    /// Returns an error if `descriptions` is empty or cuFFT cannot create the
    /// function database for the supplied descriptions.
    pub fn create(descriptions: Vec<Description>) -> Result<Self> {
        if descriptions.is_empty() {
            return Err(Error::out_of_range("description count"));
        }
        let mut handles = descriptions
            .iter()
            .map(Description::as_raw)
            .collect::<Vec<_>>();
        let mut raw = 0;
        // SAFETY: `cufftDeviceCreate` receives valid output locations.
        unsafe {
            try_ffi!(sys::cufftDeviceCreate(
                &raw mut raw,
                handles.len() as _,
                handles.as_mut_ptr()
            ))?;
        }
        Ok(Self { raw, descriptions })
    }

    /// Borrows the native device-program handle; its descriptions remain owned here.
    #[must_use]
    pub fn as_raw(&self) -> sys::cufftDeviceHandle {
        self.raw
    }

    fn description(&self, index: usize) -> Result<&Description> {
        self.descriptions
            .get(index)
            .ok_or_else(|| Error::out_of_range("description index"))
    }

    /// Reports whether the indexed FFT description is supported.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is outside the retained description list or
    /// cuFFT cannot determine support for that description.
    pub fn is_supported(&self, index: usize) -> Result<bool> {
        let description = self.description(index)?;
        let mut supported = false;
        // SAFETY: `cufftDeviceIsSupported` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftDeviceIsSupported(
                self.raw,
                description.raw,
                &raw mut supported
            ))?;
        }
        Ok(supported)
    }

    /// Lists supported device functions for the indexed description.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is outside the retained description list, cuFFT
    /// cannot enumerate matching functions, or the reported count does not fit `usize`.
    pub fn functions(&self, index: usize) -> Result<Vec<DeviceFunction<'_>>> {
        let description = self.description(index)?;
        let mut count = 0;
        // SAFETY: `cufftDeviceGetNumDeviceFunctions` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftDeviceGetNumDeviceFunctions(
                self.raw,
                description.raw,
                &raw mut count
            ))?;
        }
        let mut handles = vec![0; to_usize(count, "function count")?];
        // SAFETY: `cufftDeviceGetDeviceFunctions` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftDeviceGetDeviceFunctions(
                self.raw,
                description.raw,
                count,
                handles.as_mut_ptr()
            ))?;
        }
        Ok(handles
            .into_iter()
            .map(|raw| DeviceFunction { program: self, raw })
            .collect())
    }

    /// Returns the serialized function database.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFFT cannot query or copy the database text, or the
    /// reported byte count does not fit `usize`. Invalid UTF-8 is replaced lossily.
    pub fn database(&self) -> Result<String> {
        self.database_for(-1)
    }

    fn database_for(&self, function: i32) -> Result<String> {
        let mut size = 0;
        // SAFETY: `cufftDeviceGetDatabaseStrSize` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftDeviceGetDatabaseStrSize(
                self.raw,
                &raw mut size,
                function
            ))?;
        }
        let mut bytes = vec![0; to_usize(size, "database size")?];
        // SAFETY: `cufftDeviceGetDatabaseStr` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftDeviceGetDatabaseStr(
                self.raw,
                size,
                bytes.as_mut_ptr().cast(),
                function
            ))?;
        }
        if bytes.last() == Some(&0) {
            bytes.pop();
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Extracts every LTO IR or fatbinary code container from this database.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFFT cannot enumerate or extract code containers, or a
    /// reported container count or byte size does not fit `usize`.
    pub fn code(&self) -> Result<Vec<DeviceCode>> {
        self.code_for(-1)
    }

    fn code_for(&self, function: i32) -> Result<Vec<DeviceCode>> {
        let mut count = 0;
        // SAFETY: `cufftDeviceGetNumLTOIRs` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftDeviceGetNumLTOIRs(
                self.raw,
                &raw mut count,
                function
            ))?;
        }
        let count_usize = to_usize(count, "code count")?;
        let mut sizes = vec![0; count_usize];
        // SAFETY: `cufftDeviceGetLTOIRSizes` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftDeviceGetLTOIRSizes(
                self.raw,
                count,
                sizes.as_mut_ptr(),
                function
            ))?;
        }
        let mut code = sizes
            .into_iter()
            .map(|size| Ok(vec![0u8; to_usize(size, "code size")?]))
            .collect::<Result<Vec<_>>>()?;
        let mut pointers = code
            .iter_mut()
            .map(|bytes| bytes.as_mut_ptr().cast())
            .collect::<Vec<_>>();
        let mut containers = vec![sys::cufftDeviceCodeContainer::CUFFT_DEVICE_LTOIR; count_usize];
        // SAFETY: `cufftDeviceGetLTOIRs` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftDeviceGetLTOIRs(
                self.raw,
                count,
                pointers.as_mut_ptr(),
                containers.as_mut_ptr(),
                function
            ))?;
        }
        Ok(code
            .into_iter()
            .zip(containers)
            .map(|(bytes, container)| DeviceCode {
                bytes,
                container: match container {
                    sys::cufftDeviceCodeContainer::CUFFT_DEVICE_LTOIR => CodeContainer::LtoIr,
                    sys::cufftDeviceCodeContainer::CUFFT_DEVICE_FATBIN => CodeContainer::Fatbin,
                },
            })
            .collect())
    }
}

impl Drop for DeviceProgram {
    fn drop(&mut self) {
        // SAFETY: `cufftDeviceDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cufftDeviceDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cufft device: {error}");
            }
        }
    }
}

/// A device function borrowing its owning database.
#[derive(Debug)]
pub struct DeviceFunction<'a> {
    program: &'a DeviceProgram,
    raw: sys::cufftDeviceFunctionHandle,
}

impl DeviceFunction<'_> {
    /// Returns one resource requirement of this function.
    ///
    /// # Errors
    ///
    /// Returns a cuFFT error if the resource trait cannot be read for this function.
    pub fn trait_value(&self, attribute: FunctionTrait) -> Result<i64> {
        let mut value = 0;
        // SAFETY: `cufftDeviceGetDeviceFunctionTraitInt64` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cufftDeviceGetDeviceFunctionTraitInt64(
                self.program.raw,
                self.raw,
                attribute.into(),
                &raw mut value
            ))?;
        }
        Ok(value)
    }

    /// Returns database metadata for this function.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFFT cannot query or copy the database text, or the
    /// reported byte count does not fit `usize`. Invalid UTF-8 is replaced lossily.
    pub fn database(&self) -> Result<String> {
        self.program.database_for(self.raw)
    }

    /// Extracts the code containers implementing this function.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFFT cannot enumerate or extract code containers, or a
    /// reported container count or byte size does not fit `usize`.
    pub fn code(&self) -> Result<Vec<DeviceCode>> {
        self.program.code_for(self.raw)
    }
}

/// Returns the semantic version of the experimental device interface.
///
/// # Errors
///
/// Returns a cuFFT error if the device-interface version query fails.
pub fn version() -> Result<LibraryVersion> {
    let (mut major, mut minor, mut patch) = (0, 0, 0);
    // SAFETY: `cufftDeviceGetSemanticVersion` receives valid output locations.
    unsafe {
        try_ffi!(sys::cufftDeviceGetSemanticVersion(
            &raw mut major,
            &raw mut minor,
            &raw mut patch
        ))?;
    }
    Ok(LibraryVersion::new(major as _, minor as _, patch as _))
}

impl DeviceCode {
    /// Adds this code container to an nvJitLink compilation.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL byte or nvJitLink
    /// rejects the input code container.
    pub fn add_to_linker(&self, linker: &mut Linker, name: &str) -> Result<()> {
        let kind = match self.container {
            CodeContainer::LtoIr => InputType::LtoIr,
            CodeContainer::Fatbin => InputType::Fatbin,
        };
        linker.add_data(kind, &self.bytes, name)?;
        Ok(())
    }
}

impl_enum_display!(Direction, {
    Direction::Forward => "CUFFT_DESC_FORWARD",
    Direction::Inverse => "CUFFT_DESC_INVERSE",
});

impl_enum_display!(Transform, {
    Transform::ComplexToComplex => "CUFFT_DESC_C2C",
    Transform::ComplexToReal => "CUFFT_DESC_C2R",
    Transform::RealToComplex => "CUFFT_DESC_R2C",
});

impl_enum_display!(Precision, {
    Precision::Single => "CUFFT_DESC_SINGLE",
    Precision::Double => "CUFFT_DESC_DOUBLE",
    Precision::Half => "CUFFT_DESC_HALF",
});

impl_enum_display!(DescriptionTrait, {
    DescriptionTrait::Size => "CUFFT_DESC_TRAIT_SIZE",
    DescriptionTrait::Direction => "CUFFT_DESC_TRAIT_DIRECTION",
    DescriptionTrait::Precision => "CUFFT_DESC_TRAIT_PRECISION",
    DescriptionTrait::Transform => "CUFFT_DESC_TRAIT_TYPE",
    DescriptionTrait::Sm => "CUFFT_DESC_TRAIT_SM",
    DescriptionTrait::ElementsPerThread => "CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD",
});

impl_enum_display!(FunctionTrait, {
    FunctionTrait::ElementsPerThread => "CUFFT_DEVICE_FUNC_TRAIT_ELEMENTS_PER_THREAD",
    FunctionTrait::StorageSize => "CUFFT_DEVICE_FUNC_TRAIT_STORAGE_SIZE",
    FunctionTrait::SharedMemoryPerFft => "CUFFT_DEVICE_FUNC_TRAIT_SHARED_MEMORY_PER_FFT",
    FunctionTrait::ThreadsPerFft => "CUFFT_DEVICE_FUNC_TRAIT_THREADS_PER_FFT",
    FunctionTrait::SuggestedFftsPerBlock => "CUFFT_DEVICE_FUNC_TRAIT_SUGGESTED_FFTS_PER_BLOCK",
});
