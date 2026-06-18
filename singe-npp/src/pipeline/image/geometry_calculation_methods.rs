use crate::{
    error::Result,
    image::geometry,
    types::{
        AffineCoefficients, BoundF64, InterpolationMode, PerspectiveCoefficients, Point,
        QuadrangleF64, Rectangle,
    },
};

use super::ImagePipeline;

impl<'a, T, L> ImagePipeline<'a, T, L> {
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

    pub fn resize_tiled_source_offset(
        source_roi: Rectangle,
        destination_roi: Rectangle,
    ) -> Result<Point> {
        geometry::resize_tiled_source_offset(source_roi, destination_roi)
    }

    pub fn rotate_quad(
        source_roi: Rectangle,
        angle: f64,
        shift_x: f64,
        shift_y: f64,
    ) -> Result<QuadrangleF64> {
        geometry::rotate_quad(source_roi, angle, shift_x, shift_y)
    }

    pub fn rotate_bound(
        source_roi: Rectangle,
        angle: f64,
        shift_x: f64,
        shift_y: f64,
    ) -> Result<BoundF64> {
        geometry::rotate_bound(source_roi, angle, shift_x, shift_y)
    }

    pub fn affine_transform(
        source_roi: Rectangle,
        destination_quadrangle: QuadrangleF64,
    ) -> Result<AffineCoefficients> {
        geometry::affine_transform(source_roi, destination_quadrangle)
    }

    pub fn affine_quad(
        source_roi: Rectangle,
        coefficients: AffineCoefficients,
    ) -> Result<QuadrangleF64> {
        geometry::affine_quad(source_roi, coefficients)
    }

    pub fn affine_bound(
        source_roi: Rectangle,
        coefficients: AffineCoefficients,
    ) -> Result<BoundF64> {
        geometry::affine_bound(source_roi, coefficients)
    }

    pub fn perspective_transform(
        source_roi: Rectangle,
        destination_quadrangle: QuadrangleF64,
    ) -> Result<PerspectiveCoefficients> {
        geometry::perspective_transform(source_roi, destination_quadrangle)
    }

    pub fn perspective_quad(
        source_roi: Rectangle,
        coefficients: PerspectiveCoefficients,
    ) -> Result<QuadrangleF64> {
        geometry::perspective_quad(source_roi, coefficients)
    }

    pub fn perspective_bound(
        source_roi: Rectangle,
        coefficients: PerspectiveCoefficients,
    ) -> Result<BoundF64> {
        geometry::perspective_bound(source_roi, coefficients)
    }
}
