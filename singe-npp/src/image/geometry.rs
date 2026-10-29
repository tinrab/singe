//! Native image operations on an explicitly selected stream.
//!
//! # Safety
//! Retain each allocation until completion or its last captured replay and establish access ordering.
//! Input and read-write storage must contain valid initialized values for the complete native footprint.
//! Geometric operations may leave pixels outside the transformed source intersection untouched.
//! Native AC4 operations leave destination alpha unchanged.
//! Outputs promise writable capacity only; promotion requires all logical samples to be initialized.
//! Helpers owning temporary scratch or device descriptors complete their native use before releasing it.
//! Such helpers reject capture before allocating those temporary resources.

use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView},
    try_ffi,
    types::{
        AffineCoefficients, BoundF64, InterpolationMode, PerspectiveCoefficients, Point,
        QuadrangleF64, Rectangle,
    },
};

#[macro_use]
mod macros;

mod mirror;

pub use mirror::*;

mod resize;

pub use resize::*;

mod transform;

pub use transform::*;

mod affine;

pub use affine::*;

mod perspective;

pub use perspective::*;

mod back;

pub use back::*;

mod resize_advanced;

pub use resize_advanced::*;

/// ROIs, coefficients, and interpolation for an affine image warp.
#[derive(Debug, Clone, Copy)]
pub struct WarpAffine {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Destination region of interest, in pixels.
    pub destination_roi: Rectangle,
    /// Two-by-three affine coefficient matrix.
    pub coefficients: AffineCoefficients,
    /// Interpolation method used to sample source pixels.
    pub interpolation: InterpolationMode,
}

/// Shared ROIs and interpolation for a batch of affine image warps.
/// Coefficient matrices are supplied separately for each image.
#[derive(Debug, Clone, Copy)]
pub struct WarpAffineBatch {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Destination region of interest, in pixels.
    pub destination_roi: Rectangle,
    /// Interpolation method used to sample source pixels.
    pub interpolation: InterpolationMode,
}

/// Source and destination quadrangles, clipping ROIs, and interpolation for a quadrangle warp.
#[derive(Debug, Clone, Copy)]
pub struct WarpQuad {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Four source vertices defining the quadrangle to transform.
    pub source_quadrangle: QuadrangleF64,
    /// Destination region of interest, in pixels.
    pub destination_roi: Rectangle,
    /// Four destination vertices corresponding to the source quadrangle.
    pub destination_quadrangle: QuadrangleF64,
    /// Interpolation method used to sample source pixels.
    pub interpolation: InterpolationMode,
}

/// ROIs, coefficients, and interpolation for a perspective image warp.
#[derive(Debug, Clone, Copy)]
pub struct WarpPerspective {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Destination region of interest, in pixels.
    pub destination_roi: Rectangle,
    /// Three-by-three perspective coefficient matrix.
    pub coefficients: PerspectiveCoefficients,
    /// Interpolation method used to sample source pixels.
    pub interpolation: InterpolationMode,
}

/// Shared ROIs and interpolation for a batch of perspective image warps.
/// Coefficient matrices are supplied separately for each image.
#[derive(Debug, Clone, Copy)]
pub struct WarpPerspectiveBatch {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Destination region of interest, in pixels.
    pub destination_roi: Rectangle,
    /// Interpolation method used to sample source pixels.
    pub interpolation: InterpolationMode,
}

/// Source ROI and interpolation for sampling an image through coordinate maps.
#[derive(Debug, Clone, Copy)]
pub struct Remap {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Interpolation method used to sample source pixels.
    pub interpolation: InterpolationMode,
}

/// ROIs, rotation angle, translation, and interpolation for rotating an image.
#[derive(Debug, Clone, Copy)]
pub struct Rotate {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Destination region of interest, in pixels.
    pub destination_roi: Rectangle,
    /// Rotation angle, in degrees.
    pub angle: f64,
    /// Horizontal translation applied after the transformation, in pixels.
    pub x_shift: f64,
    /// Vertical translation applied after the transformation, in pixels.
    pub y_shift: f64,
    /// Interpolation method used to sample source pixels.
    pub interpolation: InterpolationMode,
}

/// Source and destination ROIs and interpolation for resizing an image.
#[derive(Debug, Clone, Copy)]
pub struct Resize {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Destination region of interest, in pixels.
    pub destination_roi: Rectangle,
    /// Interpolation method used to sample source pixels.
    pub interpolation: InterpolationMode,
}

/// Source and destination ROIs for one image in an advanced resize batch.
#[derive(Debug, Clone, Copy)]
pub struct ResizeBatchAdvanced {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Destination region of interest, in pixels.
    pub destination_roi: Rectangle,
}

/// ROIs, scale factors, shifts, and interpolation for square-pixel resizing.
#[derive(Debug, Clone, Copy)]
pub struct ResizeSqrPixel {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Destination region of interest, in pixels.
    pub destination_roi: Rectangle,
    /// Horizontal scale factor.
    pub x_factor: f64,
    /// Vertical scale factor.
    pub y_factor: f64,
    /// Horizontal translation applied after the transformation, in pixels.
    pub x_shift: f64,
    /// Vertical translation applied after the transformation, in pixels.
    pub y_shift: f64,
    /// Interpolation method used to sample source pixels.
    pub interpolation: InterpolationMode,
}

/// ROIs, scale factors, and interpolation for advanced square-pixel resizing.
#[derive(Debug, Clone, Copy)]
pub struct ResizeSqrPixelAdvanced {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Destination region of interest, in pixels.
    pub destination_roi: Rectangle,
    /// Horizontal scale factor.
    pub x_factor: f64,
    /// Vertical scale factor.
    pub y_factor: f64,
    /// Interpolation method used to sample source pixels.
    pub interpolation: InterpolationMode,
}

/// Returns the destination rectangle produced by scaling and shifting `source_roi` with the selected interpolation.
///
/// # Errors
///
/// Returns an error if NPP rejects the geometry or transformation parameters.
pub fn resize_rect(
    source_roi: Rectangle,
    x_factor: f64,
    y_factor: f64,
    x_shift: f64,
    y_shift: f64,
    interpolation: InterpolationMode,
) -> Result<Rectangle> {
    let mut destination = sys::NppiRect::default();
    // SAFETY: `nppiGetResizeRect` receives valid output locations.
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

/// Returns the source offset used to resize a destination tile from the given source and destination ROIs.
///
/// # Errors
///
/// Returns an error if NPP rejects the geometry or transformation parameters.
pub fn resize_tiled_source_offset(
    source_roi: Rectangle,
    destination_roi: Rectangle,
) -> Result<Point> {
    let mut offset = sys::NppiPoint::default();
    // SAFETY: `nppiGetResizeTiledSourceOffset` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiGetResizeTiledSourceOffset(
            source_roi.into(),
            destination_roi.into(),
            &raw mut offset,
        ))?;
    }
    Ok(offset.into())
}

/// Returns the quadrangle obtained by rotating `source_roi` by `angle` degrees and then applying the pixel shifts.
///
/// # Errors
///
/// Returns an error if NPP rejects the geometry or transformation parameters.
pub fn rotate_quad(
    source_roi: Rectangle,
    angle: f64,
    shift_x: f64,
    shift_y: f64,
) -> Result<QuadrangleF64> {
    let mut quadrangle = [[0.0; 2]; 4];
    // SAFETY: `nppiGetRotateQuad` receives valid output locations.
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

/// Returns the axis-aligned bounds of `source_roi` after rotation by `angle` degrees and the pixel shifts.
///
/// # Errors
///
/// Returns an error if NPP rejects the geometry or transformation parameters.
pub fn rotate_bound(
    source_roi: Rectangle,
    angle: f64,
    shift_x: f64,
    shift_y: f64,
) -> Result<BoundF64> {
    let mut bound = [[0.0; 2]; 2];
    // SAFETY: `nppiGetRotateBound` receives valid output locations.
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

/// Computes affine coefficients mapping `source_roi` to `destination_quadrangle`.
///
/// # Errors
///
/// Returns an error if NPP rejects the geometry or transformation parameters.
pub fn affine_transform(
    source_roi: Rectangle,
    destination_quadrangle: QuadrangleF64,
) -> Result<AffineCoefficients> {
    let mut coefficients = [[0.0; 3]; 2];
    // SAFETY: `nppiGetAffineTransform` receives valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiGetAffineTransform(
            source_roi.into(),
            destination_quadrangle.as_ptr().cast(),
            coefficients.as_mut_ptr().cast(),
        ))?;
    }
    Ok(coefficients.into())
}

/// Returns the quadrangle obtained by applying `coefficients` to `source_roi`.
///
/// # Errors
///
/// Returns an error if NPP rejects the geometry or transformation parameters.
pub fn affine_quad(
    source_roi: Rectangle,
    coefficients: AffineCoefficients,
) -> Result<QuadrangleF64> {
    let mut quadrangle = [[0.0; 2]; 4];
    // SAFETY: `nppiGetAffineQuad` receives valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiGetAffineQuad(
            source_roi.into(),
            quadrangle.as_mut_ptr().cast(),
            coefficients.as_ptr().cast(),
        ))?;
    }
    Ok(quadrangle.into())
}

/// Returns the axis-aligned bounds of `source_roi` after applying the affine `coefficients`.
///
/// # Errors
///
/// Returns an error if NPP rejects the geometry or transformation parameters.
pub fn affine_bound(source_roi: Rectangle, coefficients: AffineCoefficients) -> Result<BoundF64> {
    let mut bound = [[0.0; 2]; 2];
    // SAFETY: `nppiGetAffineBound` receives valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiGetAffineBound(
            source_roi.into(),
            bound.as_mut_ptr().cast(),
            coefficients.as_ptr().cast(),
        ))?;
    }
    Ok(bound.into())
}

/// Computes perspective coefficients mapping `source_roi` to `destination_quadrangle`.
///
/// # Errors
///
/// Returns an error if NPP rejects the geometry or transformation parameters.
pub fn perspective_transform(
    source_roi: Rectangle,
    destination_quadrangle: QuadrangleF64,
) -> Result<PerspectiveCoefficients> {
    let mut coefficients = [[0.0; 3]; 3];
    // SAFETY: `nppiGetPerspectiveTransform` receives valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiGetPerspectiveTransform(
            source_roi.into(),
            destination_quadrangle.as_ptr().cast(),
            coefficients.as_mut_ptr().cast(),
        ))?;
    }
    Ok(coefficients.into())
}

/// Returns the quadrangle obtained by applying the perspective `coefficients` to `source_roi`.
///
/// # Errors
///
/// Returns an error if NPP rejects the geometry or transformation parameters.
pub fn perspective_quad(
    source_roi: Rectangle,
    coefficients: PerspectiveCoefficients,
) -> Result<QuadrangleF64> {
    let mut quadrangle = [[0.0; 2]; 4];
    // SAFETY: `nppiGetPerspectiveQuad` receives valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiGetPerspectiveQuad(
            source_roi.into(),
            quadrangle.as_mut_ptr().cast(),
            coefficients.as_ptr().cast(),
        ))?;
    }
    Ok(quadrangle.into())
}

/// Returns the axis-aligned bounds of `source_roi` after applying the perspective `coefficients`.
///
/// # Errors
///
/// Returns an error if NPP rejects the geometry or transformation parameters.
pub fn perspective_bound(
    source_roi: Rectangle,
    coefficients: PerspectiveCoefficients,
) -> Result<BoundF64> {
    let mut bound = [[0.0; 2]; 2];
    // SAFETY: `nppiGetPerspectiveBound` receives valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiGetPerspectiveBound(
            source_roi.into(),
            bound.as_mut_ptr().cast(),
            coefficients.as_ptr().cast(),
        ))?;
    }
    Ok(bound.into())
}

/// ROIs and interpolation for resizing a single-channel unsigned 8-bit image.
#[derive(Debug, Clone, Copy)]
pub struct ResizeU8C1 {
    /// Source region of interest, in pixels.
    pub source_roi: Rectangle,
    /// Destination region of interest, in pixels.
    pub destination_roi: Rectangle,
    /// Interpolation method used to sample source pixels.
    pub interpolation: InterpolationMode,
}

impl ResizeU8C1 {
    /// Resizes the source ROI into the destination ROI using this configuration.
    ///
    /// Enqueues work on the selected stream.
    /// Pixels outside the transformed source intersection and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another context, or NPP rejects the resize parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// The ROIs, strides, and allocation bounds must satisfy the native resize routine; this method does not validate ROI containment.
    /// Source pixels must be initialized and ready when read by the selected stream.
    /// Keep the source read-only and the destination exclusively writable and disjoint from the source during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    pub unsafe fn execute(
        &self,
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageOutputView<'_, u8, C1>,
    ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;

        // SAFETY: `nppiResize_8u_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
