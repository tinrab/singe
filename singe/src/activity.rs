// Framework activity hooks: `testing` builds count each event for tests to observe, and other builds compile them away.

#[cfg(feature = "testing")]
use crate::{storage::cpu::profiling, testing};

pub fn descriptor() {
    #[cfg(feature = "testing")]
    testing::record_descriptor();
}

pub fn invocation() {
    #[cfg(feature = "testing")]
    testing::record_invocation();
}

pub fn storage_request() {
    #[cfg(feature = "testing")]
    testing::record_storage_request();
}

#[cfg(feature = "cuda")]
pub fn cuda_readback() {
    #[cfg(feature = "testing")]
    testing::record_cuda_readback();
}

#[cfg_attr(
    not(feature = "testing"),
    expect(unused_variables, reason = "only `testing` builds count bytes")
)]
pub fn cpu_import(bytes: usize) {
    #[cfg(feature = "testing")]
    profiling::record_import(bytes);
}

#[cfg_attr(
    not(feature = "testing"),
    expect(unused_variables, reason = "only `testing` builds count bytes")
)]
pub fn cpu_export(bytes: usize) {
    #[cfg(feature = "testing")]
    profiling::record_export(bytes);
}

#[cfg_attr(
    not(feature = "testing"),
    expect(unused_variables, reason = "only `testing` builds count bytes")
)]
pub fn cpu_clone(bytes: usize) {
    #[cfg(feature = "testing")]
    profiling::record_clone(bytes);
}
