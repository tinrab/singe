use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use singe_cuda::{
    device::{Device, DeviceAttribute},
    error::Error as CudaError,
};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
    },
    context::Context,
    error::{Error, Result},
};

/// cuDNN's native JSON serialization of device properties.
///
/// The format is defined by cuDNN and is opaque to this crate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DevicePropertiesJson(String);

impl DevicePropertiesJson {
    /// Wraps native device-property JSON without parsing or validating it.
    pub fn new(json: impl Into<String>) -> Self {
        Self(json.into())
    }

    /// Borrows the opaque native JSON string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the wrapper and returns the opaque native JSON string.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<[u8]> for DevicePropertiesJson {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

/// Source used to finalize a native device-property snapshot.
/// Serialized properties can be restored independently with [`DeviceProperties::deserialize`].
#[derive(Debug, Clone)]
pub enum DevicePropertiesSource {
    /// Targets this CUDA ordinal without requiring a cuDNN handle or changing the current device.
    Device(Device),
    /// Targets a retained cuDNN handle using `CUDNN_ATTR_DEVICEPROP_HANDLE`.
    Context(Arc<Mutex<Context>>),
}

/// Owns a finalized native device-property snapshot for heuristics, compilation and restoration.
/// Cloning shares the immutable descriptor and preserves its recorded source and architecture.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    descriptor: Arc<BackendDescriptor>,
    source: Option<DevicePropertiesSource>,
    sm_version: Option<i32>,
}

pub(crate) fn sm_version_for_device(device: Device) -> Result<i32> {
    let major = device.attribute(DeviceAttribute::ComputeCapabilityMajor)?;
    let minor = device.attribute(DeviceAttribute::ComputeCapabilityMinor)?;
    major
        .checked_mul(10)
        .and_then(|version| version.checked_add(minor))
        .ok_or_else(|| Error::out_of_range("target sm version"))
}

impl DeviceProperties {
    /// Collects properties from an explicit CUDA device or retained cuDNN handle.
    /// Sets only the chosen native source attribute, avoiding conflicting device and handle targets.
    ///
    /// # Errors
    /// Returns context-lock, context-binding, invalid device or native descriptor errors.
    pub fn create(source: DevicePropertiesSource) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::DeviceProp)?;
        let sm_version = match &source {
            DevicePropertiesSource::Device(device) => {
                if device.id() < 0 {
                    return Err(Error::out_of_range("device properties device id"));
                }
                descriptor
                    .set_attribute_i32(BackendAttributeName::DevicePropDeviceId, device.id())?;
                descriptor.finalize()?;
                sm_version_for_device(*device)?
            }
            DevicePropertiesSource::Context(context) => {
                let context = context.lock().map_err(|_| {
                    Error::Cuda(CudaError::ResourcePoisoned {
                        name: "cudnn context",
                    })
                })?;
                context.bind()?;
                // SAFETY: source retains the handle for this descriptor's complete lifetime.
                unsafe {
                    descriptor.set_attribute_handle(
                        BackendAttributeName::DevicePropHandle,
                        context.as_raw(),
                    )?;
                }
                descriptor.finalize()?;
                sm_version_for_device(context.cuda_context().device())?
            }
        };
        Ok(Self {
            descriptor: Arc::new(descriptor),
            source: Some(source),
            sm_version: Some(sm_version),
        })
    }

    /// Restores native property bytes or a JSON string without creating a live CUDA context.
    /// Native JSON remains distinct from serializing this Rust wrapper with serde.
    ///
    /// # Errors
    ///
    /// Returns an error if native descriptor creation fails, the byte buffer is empty or its length does not fit `i64`, or cuDNN rejects the serialized properties or descriptor finalization.
    pub fn deserialize(data: impl AsRef<[u8]>) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::DeviceProp)?;
        descriptor.set_attribute_char_buffer(
            BackendAttributeName::DevicePropJsonRepresentation,
            data.as_ref(),
        )?;
        descriptor.finalize()?;
        Ok(Self {
            descriptor: Arc::new(descriptor),
            source: None,
            sm_version: None,
        })
    }

    /// Returns the native device-property JSON as an opaque string.
    ///
    /// # Errors
    ///
    /// Returns an error if querying the native character payload fails or reports an invalid or persistently changing count.
    pub fn json_representation(&self) -> Result<DevicePropertiesJson> {
        self.descriptor
            .attribute_char_string(BackendAttributeName::DevicePropJsonRepresentation)
            .map(DevicePropertiesJson)
    }

    /// Returns the exact native device-property character payload, including trailing NUL bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if querying the native payload fails or reports an invalid or persistently changing count.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        self.descriptor
            .attribute_char_buffer(BackendAttributeName::DevicePropJsonRepresentation)
    }

    /// Original construction source, or `None` for restored native snapshots.
    /// Restoration does not invent a current-device association for opaque saved properties.
    #[must_use]
    pub fn source(&self) -> Option<&DevicePropertiesSource> {
        self.source.as_ref()
    }

    /// Target SM version recorded when collecting properties from a known device or handle.
    /// Uses `major * 10 + minor`, such as `90` for Hopper.
    /// Reading it does not query CUDA or lock the retained handle.
    /// Opaque native-byte restoration leaves this metadata unknown.
    #[must_use]
    pub fn sm_version(&self) -> Option<i32> {
        self.sm_version
    }

    /// Borrows the finalized native device-property descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        self.descriptor.as_ref()
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::error::Status as CudaStatus;

    use super::*;
    use crate::{error::Error, testing::setup_context};

    #[test]
    fn test_device_properties_json_roundtrip() -> Result<()> {
        let context_ref = match setup_context() {
            Ok(context) => context,
            Err(Error::Cuda(CudaError::Cuda { code, .. })) if code == CudaStatus::NoDevice => {
                return Ok(());
            }
            Err(error) => return Err(error),
        };

        let properties = DeviceProperties::create(DevicePropertiesSource::Device(
            context_ref.cuda_context().device(),
        ))?;
        let json = properties.json_representation()?;
        let restored = DeviceProperties::deserialize(&json)?;
        let bytes = properties.serialize()?;
        let restored_from_bytes = DeviceProperties::deserialize(&bytes)?;

        assert_eq!(restored.json_representation()?, json);
        assert_eq!(restored_from_bytes.json_representation()?, json);

        Ok(())
    }
}
