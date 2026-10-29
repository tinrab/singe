mod common;

use std::sync::{Arc, Mutex};

use singe_cudnn::{
    backend::device::{DeviceProperties, DevicePropertiesSource},
    error::Result,
};

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    let device = context.cudnn.cuda_context().device();
    let by_device = DeviceProperties::create(DevicePropertiesSource::Device(device))?;
    println!(
        "device properties: {} native bytes, SM={:?}",
        by_device.serialize()?.len(),
        by_device.sm_version(),
    );

    // The snapshot retains the cuDNN handle owner while its descriptor is live.
    let by_handle = DeviceProperties::create(DevicePropertiesSource::Context(Arc::new(
        Mutex::new(context.cudnn),
    )))?;
    let native_bytes = by_handle.serialize()?;
    let retained = by_handle.clone();
    drop(by_handle);
    println!(
        "retained handle property snapshot SM={:?}, source={:?}",
        retained.sm_version(),
        retained.source(),
    );
    drop(retained);
    // Restoring these bytes does not create another cuDNN handle or assign a current-device source.
    let restored = DeviceProperties::deserialize(&native_bytes)?;
    println!(
        "restored device properties: {} JSON bytes, source={:?}, SM={:?}",
        restored.json_representation()?.as_str().len(),
        restored.source(),
        restored.sm_version(),
    );
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_device_properties", run())
}
