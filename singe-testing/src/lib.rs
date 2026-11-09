mod bench;
mod compare;
pub mod decoder;
mod inputs;
#[cfg(feature = "cuda_13_4")]
mod paths;
mod reference;
mod snapshot;

#[cfg(feature = "cuda_13_4")]
pub use self::paths::{EVERY_PATH, RunPath, check_paths, target_directory};
#[cfg(feature = "cuda_13_4")]
pub use singe::testing::{maybe_lock_device as lock_device, program_target};

pub use self::{
    bench::{Benches, Summary},
    compare::{HalfGap, Tolerance, assert_matches, assert_within_half_gap, host},
    inputs::{pattern_image, tone_and_chirp},
    reference::{Reference, integers},
    snapshot::{SNAPSHOT_IGNORED, snapshot, snapshot_if_set},
};

pub const CUDA: singe::Device = singe::Device::Cuda(0);

#[macro_export]
macro_rules! reference {
    ($path:expr) => {
        $crate::Reference::load(
            ::std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/references")
                .join(&$path),
        )
    };
}

#[macro_export]
macro_rules! references {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/references")
    };
}
