#[macro_use]
mod macros;
mod basic;
mod cubic;
mod linear;
mod palette;
mod trilinear;

pub use self::{basic::*, cubic::*, linear::*, palette::*, trilinear::*};
