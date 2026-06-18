use super::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Size {
    pub width: i32,
    pub height: i32,
}

impl Size {
    pub const fn new(width: i32, height: i32) -> Self {
        assert!(width > 0, "width must be greater than 0");
        assert!(height > 0, "height must be greater than 0");
        Self { width, height }
    }

    pub const fn from_raw_parts(width: i32, height: i32) -> Self {
        Self { width, height }
    }

    pub fn create(width: i32, height: i32) -> Result<Self> {
        let size = Self::from_raw_parts(width, height);
        size.validate()?;
        Ok(size)
    }

    pub const fn is_positive(self) -> bool {
        self.width > 0 && self.height > 0
    }

    pub fn validate(self) -> Result<()> {
        if self.width <= 0 {
            return Err(Error::ValueNotPositive {
                name: "width".into(),
            });
        }
        if self.height <= 0 {
            return Err(Error::ValueNotPositive {
                name: "height".into(),
            });
        }
        Ok(())
    }

    pub fn rectangle(self) -> Rectangle {
        Rectangle {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        }
    }

    pub fn centered_rectangle(self) -> Rectangle {
        let side = self.width.min(self.height);
        Rectangle {
            x: ((self.width - side) / 2),
            y: ((self.height - side) / 2),
            width: side,
            height: side,
        }
    }
}

impl From<Size> for sys::NppiSize {
    fn from(value: Size) -> Self {
        Self {
            width: value.width,
            height: value.height,
        }
    }
}

impl From<sys::NppiSize> for Size {
    fn from(value: sys::NppiSize) -> Self {
        Self::from_raw_parts(value.width, value.height)
    }
}

impl Display for Size {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_create_rejects_nonpositive_width() {
        assert!(matches!(
            Size::create(0, 1),
            Err(Error::ValueNotPositive { name }) if name == "width"
        ));
    }

    #[test]
    fn size_create_rejects_nonpositive_height() {
        assert!(matches!(
            Size::create(1, -1),
            Err(Error::ValueNotPositive { name }) if name == "height"
        ));
    }

    #[test]
    fn size_raw_parts_preserves_ffi_values() {
        let size = Size::from_raw_parts(0, -1);
        assert_eq!(size.width, 0);
        assert_eq!(size.height, -1);
        assert!(!size.is_positive());
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, Default)]
pub struct Rectangle {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rectangle {
    pub fn size(self) -> Size {
        Size::from_raw_parts(self.width, self.height)
    }
}

impl From<Rectangle> for sys::NppiRect {
    fn from(value: Rectangle) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

impl From<sys::NppiRect> for Rectangle {
    fn from(value: sys::NppiRect) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, Default)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub const ZERO: Point = Point::new(0, 0);

    pub const fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }
}

impl From<Point> for sys::NppiPoint {
    fn from(value: Point) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<sys::NppiPoint> for Point {
    fn from(value: sys::NppiPoint) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Default)]
pub struct Point32f {
    pub x: f32,
    pub y: f32,
}

impl From<Point32f> for sys::NppiPoint32f {
    fn from(value: Point32f) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<sys::NppiPoint32f> for Point32f {
    fn from(value: sys::NppiPoint32f) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Default)]
pub struct Point64f {
    pub x: f64,
    pub y: f64,
}

impl From<Point64f> for sys::NppiPoint64f {
    fn from(value: Point64f) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<sys::NppiPoint64f> for Point64f {
    fn from(value: sys::NppiPoint64f) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<[f64; 2]> for Point64f {
    fn from(value: [f64; 2]) -> Self {
        Self {
            x: value[0],
            y: value[1],
        }
    }
}

impl From<Point64f> for [f64; 2] {
    fn from(value: Point64f) -> Self {
        [value.x, value.y]
    }
}

impl Point64f {
    pub const fn to_array(self) -> [f64; 2] {
        [self.x, self.y]
    }
}

pub type PointF64 = Point64f;

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Default)]
pub struct PointPolar {
    pub rho: f32,
    pub theta: f32,
}

impl From<PointPolar> for sys::NppPointPolar {
    fn from(value: PointPolar) -> Self {
        Self {
            rho: value.rho,
            theta: value.theta,
        }
    }
}

impl From<sys::NppPointPolar> for PointPolar {
    fn from(value: sys::NppPointPolar) -> Self {
        Self {
            rho: value.rho,
            theta: value.theta,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundF64 {
    pub lower: PointF64,
    pub upper: PointF64,
}

impl From<[[f64; 2]; 2]> for BoundF64 {
    fn from(value: [[f64; 2]; 2]) -> Self {
        Self {
            lower: value[0].into(),
            upper: value[1].into(),
        }
    }
}

impl From<BoundF64> for [[f64; 2]; 2] {
    fn from(value: BoundF64) -> Self {
        [value.lower.into(), value.upper.into()]
    }
}

impl BoundF64 {
    pub const fn as_ptr(&self) -> *const [f64; 2] {
        (&self.lower as *const PointF64).cast()
    }

    pub const fn as_mut_ptr(&mut self) -> *mut [f64; 2] {
        (&mut self.lower as *mut PointF64).cast()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadrangleF64 {
    pub points: [PointF64; 4],
}

impl From<[[f64; 2]; 4]> for QuadrangleF64 {
    fn from(value: [[f64; 2]; 4]) -> Self {
        Self {
            points: value.map(PointF64::from),
        }
    }
}

impl From<QuadrangleF64> for [[f64; 2]; 4] {
    fn from(value: QuadrangleF64) -> Self {
        value.points.map(Into::into)
    }
}

impl QuadrangleF64 {
    pub const fn as_ptr(&self) -> *const [f64; 2] {
        self.points.as_ptr().cast()
    }

    pub fn as_mut_ptr(&mut self) -> *mut [f64; 2] {
        self.points.as_mut_ptr().cast()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AffineCoefficients {
    pub values: [[f64; 3]; 2],
}

impl From<[[f64; 3]; 2]> for AffineCoefficients {
    fn from(value: [[f64; 3]; 2]) -> Self {
        Self { values: value }
    }
}

impl From<AffineCoefficients> for [[f64; 3]; 2] {
    fn from(value: AffineCoefficients) -> Self {
        value.values
    }
}

impl AffineCoefficients {
    pub const fn as_ptr(&self) -> *const [f64; 3] {
        self.values.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut [f64; 3] {
        self.values.as_mut_ptr()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PerspectiveCoefficients {
    pub values: [[f64; 3]; 3],
}

impl From<[[f64; 3]; 3]> for PerspectiveCoefficients {
    fn from(value: [[f64; 3]; 3]) -> Self {
        Self { values: value }
    }
}

impl From<PerspectiveCoefficients> for [[f64; 3]; 3] {
    fn from(value: PerspectiveCoefficients) -> Self {
        value.values
    }
}

impl PerspectiveCoefficients {
    pub const fn as_ptr(&self) -> *const [f64; 3] {
        self.values.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut [f64; 3] {
        self.values.as_mut_ptr()
    }
}
