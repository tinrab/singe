use singe_cuda_sys::runtime;

#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
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
