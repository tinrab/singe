use crate::{
    error::Result,
    image::geometry,
    pipeline::image::ImagePipeline,
    types::{
        AffineCoefficients, BoundF64, InterpolationMode, PerspectiveCoefficients, Point,
        QuadrangleF64, Rectangle,
    },
};

impl<T, L> ImagePipeline<'_, T, L> {
    /// Returns the destination rectangle produced by scaling and shifting `source_roi` with the selected interpolation.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the geometry or transformation parameters.
    ///
    /// See [`resize_rect`](crate::image::raw::geometry::resize_rect) for the host-side geometry calculation.
    pub fn resize_rect(
        source_roi: Rectangle,
        x_factor: f64,
        y_factor: f64,
        x_shift: f64,
        y_shift: f64,
        interpolation: InterpolationMode,
    ) -> Result<Rectangle> {
        geometry::resize_rect(
            source_roi,
            x_factor,
            y_factor,
            x_shift,
            y_shift,
            interpolation,
        )
    }

    /// Returns the source offset used to resize a destination tile from the given source and destination ROIs.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the geometry or transformation parameters.
    ///
    /// See [`resize_tiled_source_offset`](crate::image::raw::geometry::resize_tiled_source_offset) for the host-side geometry calculation.
    pub fn resize_tiled_source_offset(
        source_roi: Rectangle,
        destination_roi: Rectangle,
    ) -> Result<Point> {
        geometry::resize_tiled_source_offset(source_roi, destination_roi)
    }

    /// Returns the quadrangle obtained by rotating `source_roi` by `angle` degrees and then applying the pixel shifts.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the geometry or transformation parameters.
    ///
    /// See [`rotate_quad`](crate::image::raw::geometry::rotate_quad) for the host-side geometry calculation.
    pub fn rotate_quad(
        source_roi: Rectangle,
        angle: f64,
        shift_x: f64,
        shift_y: f64,
    ) -> Result<QuadrangleF64> {
        geometry::rotate_quad(source_roi, angle, shift_x, shift_y)
    }

    /// Returns the axis-aligned bounds of `source_roi` after rotation by `angle` degrees and the pixel shifts.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the geometry or transformation parameters.
    ///
    /// See [`rotate_bound`](crate::image::raw::geometry::rotate_bound) for the host-side geometry calculation.
    pub fn rotate_bound(
        source_roi: Rectangle,
        angle: f64,
        shift_x: f64,
        shift_y: f64,
    ) -> Result<BoundF64> {
        geometry::rotate_bound(source_roi, angle, shift_x, shift_y)
    }

    /// Computes affine coefficients mapping `source_roi` to `destination_quadrangle`.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the geometry or transformation parameters.
    ///
    /// See [`affine_transform`](crate::image::raw::geometry::affine_transform) for the host-side geometry calculation.
    pub fn affine_transform(
        source_roi: Rectangle,
        destination_quadrangle: QuadrangleF64,
    ) -> Result<AffineCoefficients> {
        geometry::affine_transform(source_roi, destination_quadrangle)
    }

    /// Returns the quadrangle obtained by applying `coefficients` to `source_roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the geometry or transformation parameters.
    ///
    /// See [`affine_quad`](crate::image::raw::geometry::affine_quad) for the host-side geometry calculation.
    pub fn affine_quad(
        source_roi: Rectangle,
        coefficients: AffineCoefficients,
    ) -> Result<QuadrangleF64> {
        geometry::affine_quad(source_roi, coefficients)
    }

    /// Returns the axis-aligned bounds of `source_roi` after applying the affine `coefficients`.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the geometry or transformation parameters.
    ///
    /// See [`affine_bound`](crate::image::raw::geometry::affine_bound) for the host-side geometry calculation.
    pub fn affine_bound(
        source_roi: Rectangle,
        coefficients: AffineCoefficients,
    ) -> Result<BoundF64> {
        geometry::affine_bound(source_roi, coefficients)
    }

    /// Computes perspective coefficients mapping `source_roi` to `destination_quadrangle`.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the geometry or transformation parameters.
    ///
    /// See [`perspective_transform`](crate::image::raw::geometry::perspective_transform) for the host-side geometry calculation.
    pub fn perspective_transform(
        source_roi: Rectangle,
        destination_quadrangle: QuadrangleF64,
    ) -> Result<PerspectiveCoefficients> {
        geometry::perspective_transform(source_roi, destination_quadrangle)
    }

    /// Returns the quadrangle obtained by applying the perspective `coefficients` to `source_roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the geometry or transformation parameters.
    ///
    /// See [`perspective_quad`](crate::image::raw::geometry::perspective_quad) for the host-side geometry calculation.
    pub fn perspective_quad(
        source_roi: Rectangle,
        coefficients: PerspectiveCoefficients,
    ) -> Result<QuadrangleF64> {
        geometry::perspective_quad(source_roi, coefficients)
    }

    /// Returns the axis-aligned bounds of `source_roi` after applying the perspective `coefficients`.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the geometry or transformation parameters.
    ///
    /// See [`perspective_bound`](crate::image::raw::geometry::perspective_bound) for the host-side geometry calculation.
    pub fn perspective_bound(
        source_roi: Rectangle,
        coefficients: PerspectiveCoefficients,
    ) -> Result<BoundF64> {
        geometry::perspective_bound(source_roi, coefficients)
    }
}
