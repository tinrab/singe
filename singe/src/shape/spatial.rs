use serde::{Deserialize, Serialize};

use crate::graph::DimensionExpr;

pub type Spatial2dI64 = Spatial2d<i64>;

pub type Spatial2dExpr = Spatial2d<DimensionExpr>;

pub type Spatial3dI64 = Spatial3d<i64>;

pub type Padding2d = Spatial2d<Padding>;

// A value per axis of a planar `[.., H, W]` operation, such as its kernel, stride, or size, in NCHW order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Spatial2d<T> {
    pub height: T,
    pub width: T,
}

impl<T> Spatial2d<T> {
    pub const fn new(height: T, width: T) -> Self {
        Self { height, width }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Spatial2d<U> {
        Spatial2d {
            height: f(self.height),
            width: f(self.width),
        }
    }

    pub fn into_array(self) -> [T; 2] {
        [self.height, self.width]
    }
}

impl<T: Copy> Spatial2d<T> {
    pub const fn square(side: T) -> Self {
        Self {
            height: side,
            width: side,
        }
    }
}

impl Spatial2d<Padding> {
    // Pads each side of the height by `height` and each side of the width by `width`.
    #[must_use]
    pub const fn symmetric(height: i64, width: i64) -> Self {
        Self {
            height: Padding::symmetric(height),
            width: Padding::symmetric(width),
        }
    }
}

impl<T> From<(T, T)> for Spatial2d<T> {
    fn from((height, width): (T, T)) -> Self {
        Self { height, width }
    }
}

impl From<(i64, i64)> for Spatial2dExpr {
    fn from((height, width): (i64, i64)) -> Self {
        Self::new(height.into(), width.into())
    }
}

impl From<Spatial2dI64> for Spatial2dExpr {
    fn from(size: Spatial2dI64) -> Self {
        size.map(DimensionExpr::constant)
    }
}

// A value per axis of a volumetric `[.., D, H, W]` operation, in NCDHW order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Spatial3d<T> {
    pub depth: T,
    pub height: T,
    pub width: T,
}

impl<T> Spatial3d<T> {
    pub const fn new(depth: T, height: T, width: T) -> Self {
        Self {
            depth,
            height,
            width,
        }
    }

    // The height and width, without the depth.
    pub fn planar(self) -> Spatial2d<T> {
        Spatial2d::new(self.height, self.width)
    }

    pub fn into_array(self) -> [T; 3] {
        [self.depth, self.height, self.width]
    }
}

impl<T: Copy> Spatial3d<T> {
    pub const fn cube(side: T) -> Self {
        Self {
            depth: side,
            height: side,
            width: side,
        }
    }
}

impl<T> From<(T, T, T)> for Spatial3d<T> {
    fn from((depth, height, width): (T, T, T)) -> Self {
        Self::new(depth, height, width)
    }
}

// The points added before an axis's first element and after its last.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Padding {
    pub before: i64,
    pub after: i64,
}

impl Padding {
    pub const ZERO: Self = Self::new(0, 0);

    #[must_use]
    pub const fn new(before: i64, after: i64) -> Self {
        Self { before, after }
    }

    #[must_use]
    pub const fn symmetric(amount: i64) -> Self {
        Self::new(amount, amount)
    }

    #[must_use]
    pub const fn total(self) -> i64 {
        self.before + self.after
    }
}
