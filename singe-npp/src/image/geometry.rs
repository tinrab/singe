use singe_cuda::types::f16;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::geometry_helpers::*,
    image::view::{
        AC4, C1, C3, C4, ChannelLayout, ImageView, ImageViewMut, PlanarImageView,
        PlanarImageViewMut,
    },
    try_ffi,
    types::{
        AffineCoefficients, Axis, BoundF64, DataTypeLike, InterpolationMode,
        PerspectiveCoefficients, Point, QuadrangleF64, Rectangle,
    },
    utility::to_usize,
    workspace::ScratchBuffer,
};

#[macro_use]
#[path = "geometry_macros.rs"]
mod macros;

#[path = "geometry_mirror.rs"]
mod mirror;
pub use mirror::*;

#[path = "geometry_resize.rs"]
mod resize;
pub use resize::*;

#[path = "geometry_transform.rs"]
mod transform;
pub use transform::*;

#[path = "geometry_affine.rs"]
mod affine;
pub use affine::*;

#[path = "geometry_perspective.rs"]
mod perspective;
pub use perspective::*;

#[path = "geometry_back.rs"]
mod back;
pub use back::*;

#[path = "geometry_resize_advanced.rs"]
mod resize_advanced;
pub use resize_advanced::*;

#[derive(Debug, Clone, Copy)]
pub struct WarpAffine {
    pub source_roi: Rectangle,
    pub destination_roi: Rectangle,
    pub coefficients: AffineCoefficients,
    pub interpolation: InterpolationMode,
}

#[derive(Debug, Clone, Copy)]
pub struct WarpAffineBatch {
    pub source_roi: Rectangle,
    pub destination_roi: Rectangle,
    pub interpolation: InterpolationMode,
}

#[derive(Debug, Clone, Copy)]
pub struct WarpQuad {
    pub source_roi: Rectangle,
    pub source_quadrangle: QuadrangleF64,
    pub destination_roi: Rectangle,
    pub destination_quadrangle: QuadrangleF64,
    pub interpolation: InterpolationMode,
}

#[derive(Debug, Clone, Copy)]
pub struct WarpPerspective {
    pub source_roi: Rectangle,
    pub destination_roi: Rectangle,
    pub coefficients: PerspectiveCoefficients,
    pub interpolation: InterpolationMode,
}

#[derive(Debug, Clone, Copy)]
pub struct WarpPerspectiveBatch {
    pub source_roi: Rectangle,
    pub destination_roi: Rectangle,
    pub interpolation: InterpolationMode,
}

#[derive(Debug, Clone, Copy)]
pub struct Remap {
    pub source_roi: Rectangle,
    pub interpolation: InterpolationMode,
}

#[derive(Debug, Clone, Copy)]
pub struct Rotate {
    pub source_roi: Rectangle,
    pub destination_roi: Rectangle,
    pub angle: f64,
    pub x_shift: f64,
    pub y_shift: f64,
    pub interpolation: InterpolationMode,
}

#[derive(Debug, Clone, Copy)]
pub struct Resize {
    pub source_roi: Rectangle,
    pub destination_roi: Rectangle,
    pub interpolation: InterpolationMode,
}

#[derive(Debug, Clone, Copy)]
pub struct ResizeBatchAdvanced {
    pub source_roi: Rectangle,
    pub destination_roi: Rectangle,
}

#[derive(Debug, Clone, Copy)]
pub struct ResizeSqrPixel {
    pub source_roi: Rectangle,
    pub destination_roi: Rectangle,
    pub x_factor: f64,
    pub y_factor: f64,
    pub x_shift: f64,
    pub y_shift: f64,
    pub interpolation: InterpolationMode,
}

#[derive(Debug, Clone, Copy)]
pub struct ResizeSqrPixelAdvanced {
    pub source_roi: Rectangle,
    pub destination_roi: Rectangle,
    pub x_factor: f64,
    pub y_factor: f64,
    pub interpolation: InterpolationMode,
}

pub fn resize_rect(
    source_roi: Rectangle,
    x_factor: f64,
    y_factor: f64,
    x_shift: f64,
    y_shift: f64,
    interpolation: InterpolationMode,
) -> Result<Rectangle> {
    let mut destination = sys::NppiRect::default();
    unsafe {
        try_ffi!(sys::nppiGetResizeRect(
            source_roi.into(),
            &raw mut destination,
            x_factor,
            y_factor,
            x_shift,
            y_shift,
            i32::from(interpolation),
        ))?;
    }
    Ok(destination.into())
}

pub fn resize_tiled_source_offset(
    source_roi: Rectangle,
    destination_roi: Rectangle,
) -> Result<Point> {
    let mut offset = sys::NppiPoint::default();
    unsafe {
        try_ffi!(sys::nppiGetResizeTiledSourceOffset(
            source_roi.into(),
            destination_roi.into(),
            &raw mut offset,
        ))?;
    }
    Ok(offset.into())
}

pub fn rotate_quad(
    source_roi: Rectangle,
    angle: f64,
    shift_x: f64,
    shift_y: f64,
) -> Result<QuadrangleF64> {
    let mut quadrangle = [[0.0; 2]; 4];
    unsafe {
        try_ffi!(sys::nppiGetRotateQuad(
            source_roi.into(),
            quadrangle.as_mut_ptr().cast(),
            angle,
            shift_x,
            shift_y,
        ))?;
    }
    Ok(quadrangle.into())
}

pub fn rotate_bound(
    source_roi: Rectangle,
    angle: f64,
    shift_x: f64,
    shift_y: f64,
) -> Result<BoundF64> {
    let mut bound = [[0.0; 2]; 2];
    unsafe {
        try_ffi!(sys::nppiGetRotateBound(
            source_roi.into(),
            bound.as_mut_ptr().cast(),
            angle,
            shift_x,
            shift_y,
        ))?;
    }
    Ok(bound.into())
}

pub fn affine_transform(
    source_roi: Rectangle,
    destination_quadrangle: QuadrangleF64,
) -> Result<AffineCoefficients> {
    let mut coefficients = [[0.0; 3]; 2];
    unsafe {
        try_ffi!(sys::nppiGetAffineTransform(
            source_roi.into(),
            destination_quadrangle.as_ptr().cast(),
            coefficients.as_mut_ptr().cast(),
        ))?;
    }
    Ok(coefficients.into())
}

pub fn affine_quad(
    source_roi: Rectangle,
    coefficients: AffineCoefficients,
) -> Result<QuadrangleF64> {
    let mut quadrangle = [[0.0; 2]; 4];
    unsafe {
        try_ffi!(sys::nppiGetAffineQuad(
            source_roi.into(),
            quadrangle.as_mut_ptr().cast(),
            coefficients.as_ptr().cast(),
        ))?;
    }
    Ok(quadrangle.into())
}

pub fn affine_bound(source_roi: Rectangle, coefficients: AffineCoefficients) -> Result<BoundF64> {
    let mut bound = [[0.0; 2]; 2];
    unsafe {
        try_ffi!(sys::nppiGetAffineBound(
            source_roi.into(),
            bound.as_mut_ptr().cast(),
            coefficients.as_ptr().cast(),
        ))?;
    }
    Ok(bound.into())
}

pub fn perspective_transform(
    source_roi: Rectangle,
    destination_quadrangle: QuadrangleF64,
) -> Result<PerspectiveCoefficients> {
    let mut coefficients = [[0.0; 3]; 3];
    unsafe {
        try_ffi!(sys::nppiGetPerspectiveTransform(
            source_roi.into(),
            destination_quadrangle.as_ptr().cast(),
            coefficients.as_mut_ptr().cast(),
        ))?;
    }
    Ok(coefficients.into())
}

pub fn perspective_quad(
    source_roi: Rectangle,
    coefficients: PerspectiveCoefficients,
) -> Result<QuadrangleF64> {
    let mut quadrangle = [[0.0; 2]; 4];
    unsafe {
        try_ffi!(sys::nppiGetPerspectiveQuad(
            source_roi.into(),
            quadrangle.as_mut_ptr().cast(),
            coefficients.as_ptr().cast(),
        ))?;
    }
    Ok(quadrangle.into())
}

pub fn perspective_bound(
    source_roi: Rectangle,
    coefficients: PerspectiveCoefficients,
) -> Result<BoundF64> {
    let mut bound = [[0.0; 2]; 2];
    unsafe {
        try_ffi!(sys::nppiGetPerspectiveBound(
            source_roi.into(),
            bound.as_mut_ptr().cast(),
            coefficients.as_ptr().cast(),
        ))?;
    }
    Ok(bound.into())
}

#[derive(Debug, Clone, Copy)]
pub struct ResizeU8C1 {
    pub source_roi: Rectangle,
    pub destination_roi: Rectangle,
    pub interpolation: InterpolationMode,
}

impl ResizeU8C1 {
    pub fn execute(
        &self,
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageViewMut<'_, u8, C1>,
    ) -> Result<()> {
        unsafe {
            try_ffi!(sys::nppiResize_8u_C1R_Ctx(
                source.as_ptr().cast(),
                source.step(),
                source.size().into(),
                self.source_roi.into(),
                destination.as_mut_ptr().cast(),
                destination.step(),
                destination.size().into(),
                self.destination_roi.into(),
                i32::from(self.interpolation),
                stream_context.as_raw(),
            ))?;
        }
        Ok(())
    }
}
