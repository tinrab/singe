use super::*;

#[macro_use]
#[path = "color_lookup_macros.rs"]
mod macros;

#[path = "color_lookup_basic.rs"]
mod basic;
pub use basic::*;

#[path = "color_lookup_linear.rs"]
mod linear;
pub use linear::*;

#[path = "color_lookup_cubic.rs"]
mod cubic;
pub use cubic::*;

#[path = "color_lookup_trilinear.rs"]
mod trilinear;
pub use trilinear::*;

#[path = "color_lookup_palette.rs"]
mod palette;
pub use palette::*;
