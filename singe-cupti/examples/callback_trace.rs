//! Port of the CUPTI callback samples: subscribe to the CUDA driver API domain and
//! count the driver functions entered while a small transfer runs.

use std::{
    collections::BTreeMap,
    error::Error,
    sync::{Arc, Mutex},
};

use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};
use singe_cupti::{
    callback::{CallbackPayload, Subscriber},
    types::{ApiCallbackSite, CallbackDomain},
};

fn main() -> Result<(), Box<dyn Error>> {
    let calls = Arc::new(Mutex::new(BTreeMap::<String, usize>::new()));
    let recorded = Arc::clone(&calls);
    let subscriber = Subscriber::create(move |domain, _callback_id, data| {
        data.with_decoded(domain, |data| {
            if let CallbackPayload::Api(api) = data
                && api.site() == ApiCallbackSite::Enter
                && let Some(name) = api.function_name()
            {
                let name = name.to_string_lossy().into_owned();
                *recorded
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .entry(name)
                    .or_default() += 1;
            }
        });
    })?;
    subscriber.enable_domain(true, CallbackDomain::DriverApi)?;

    let context = CudaContext::create_for_device(Device::new(0))?;
    let input = (0..1024).map(|value| value as f32).collect::<Vec<_>>();
    let device = DeviceMemory::from_slice(&context, &input)?;
    let output = device.copy_to_host_vec()?;
    assert_eq!(output, input);

    subscriber.unsubscribe()?;
    let calls = calls.lock().unwrap_or_else(|error| error.into_inner());
    println!(
        "driver API calls observed: {}",
        calls.values().sum::<usize>()
    );
    for (name, count) in calls.iter() {
        println!("{count:>4}  {name}");
    }
    Ok(())
}
