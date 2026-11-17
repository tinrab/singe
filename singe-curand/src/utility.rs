use singe_core::{checked_arithmetic, checked_conversions};
use singe_curand_sys as sys;

use crate::error::Error;

checked_conversions!(Error {
    to_size_t -> sys::size_t,
});

checked_arithmetic!(Error { checked_byte_len });
