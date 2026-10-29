#[macro_use]
mod macros;

mod basic;

pub use basic::*;

mod linear;

pub use linear::*;

mod cubic;

pub use cubic::*;

mod trilinear;

pub use trilinear::*;

mod palette;

pub use palette::*;
