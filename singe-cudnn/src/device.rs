use singe_cuda::device::Device;

use crate::{
    attribute::BackendAttributeName,
    context::Context,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::Result,
};

#[derive(Debug)]
pub struct DeviceProperties {
    descriptor: BackendDescriptor,
}

impl DeviceProperties {
    pub fn create(ctx: &Context) -> Result<Self> {
        ctx.bind()?;

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::DeviceProp)?;
        descriptor.set_attribute_i32(
            BackendAttributeName::DevicePropDeviceId,
            Device::current()?.id(),
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    // TODO: Refactor maybe. I don't like how these cuDNN APIs are in conflict with serde.
    pub fn from_json_representation(json: &str) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::DeviceProp)?;
        descriptor
            .set_attribute_char_string(BackendAttributeName::DevicePropJsonRepresentation, json)?;
        descriptor.finalize()?;
        Ok(Self { descriptor })
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::DeviceProp)?;
        descriptor
            .set_attribute_char_buffer(BackendAttributeName::DevicePropJsonRepresentation, data)?;
        descriptor.finalize()?;
        Ok(Self { descriptor })
    }

    pub fn json_representation(&self) -> Result<String> {
        self.descriptor
            .attribute_char_string(BackendAttributeName::DevicePropJsonRepresentation)
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(self.json_representation()?.into_bytes())
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::{error::Error, testing::setup_context};

    #[test]
    fn device_properties_json_roundtrip() -> Result<()> {
        let context = match setup_context() {
            Ok(ctx) => ctx,
            Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
                if code == singe_cuda::error::Status::NoDevice =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };

        let properties = DeviceProperties::create(&context)?;
        let json = properties.json_representation()?;
        let restored = DeviceProperties::from_json_representation(&json)?;
        let bytes = properties.serialize()?;
        let restored_from_bytes = DeviceProperties::deserialize(&bytes)?;

        assert_eq!(restored.json_representation()?, json);
        assert_eq!(restored_from_bytes.json_representation()?, json);

        Ok(())
    }
}
