use singe_cuda_sys::runtime;

/// Three unsigned 32-bit axis values for CUDA grids, blocks, or clusters.
///
/// The interpretation depends on the receiving API. The default value has all axes set to zero; launch geometry constructors reject zero dimensions.
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Dim3 {
    /// Value along the x axis.
    pub x: u32,
    /// Value along the y axis.
    pub y: u32,
    /// Value along the z axis.
    pub z: u32,
}

impl Dim3 {
    /// Creates axis values without validating launch dimensions or device limits.
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

impl From<runtime::dim3> for Dim3 {
    fn from(dim: runtime::dim3) -> Self {
        Self {
            x: dim.x,
            y: dim.y,
            z: dim.z,
        }
    }
}

impl From<Dim3> for runtime::dim3 {
    fn from(dim: Dim3) -> Self {
        Self {
            x: dim.x,
            y: dim.y,
            z: dim.z,
        }
    }
}
