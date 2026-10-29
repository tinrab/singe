/// Row-major three-by-four native color-twist coefficients.
pub type ColorTwistMatrix = [[f32; 4]; 3];

/// Row-major four-by-four coefficients for a four-channel color transform.
pub type ColorTwistMatrix4 = [[f32; 4]; 4];

/// Additive output constants for the four-channel color transform.
pub type ColorTwistConstants4 = [f32; 4];

/// Four row-major color-transform rows, each with four coefficients and an additive constant.
pub type ColorTwistBatchConstantsMatrix4 = [[f32; 5]; 4];

mod lookup;

pub use lookup::*;

#[macro_use]
mod twist_macros;

mod twist;

pub use twist::*;

#[macro_use]
mod grayscale_macros;

mod grayscale;

pub use grayscale::*;

#[macro_use]
mod conversion_uyvp_macros;

#[macro_use]
mod conversion_generic_macros;

#[macro_use]
mod conversion_jpeg_macros;

#[macro_use]
mod conversion_twist_macros;

#[macro_use]
mod conversion_nv12_macros;

#[macro_use]
mod conversion_subsampled_macros;

mod uyvp;

pub use uyvp::*;

mod yuv;

pub use yuv::*;

mod ycbcr;

pub use ycbcr::*;

mod subsampled;

pub use subsampled::*;
