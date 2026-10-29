use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{
            optional_image_output, validate_border_roi, validate_gradient_vector_outputs,
        },
        view::{C1, C3, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, ImageNormalization, MaskSize, Point},
};

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.",
    gradient_vector_prewitt_border_u8_to_i16_f32_c1,
    u8,
    C1,
    i16,
    nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.",
    gradient_vector_prewitt_border_u8_c3_to_i16_f32_c1,
    u8,
    C3,
    i16,
    nppiGradientVectorPrewittBorder_8u16s_C3C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.",
    gradient_vector_prewitt_border_i16_to_f32_c1,
    i16,
    C1,
    f32,
    nppiGradientVectorPrewittBorder_16s32f_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.",
    gradient_vector_prewitt_border_i16_c3_to_f32_c1,
    i16,
    C3,
    f32,
    nppiGradientVectorPrewittBorder_16s32f_C3C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.",
    gradient_vector_prewitt_border_u16_to_f32_c1,
    u16,
    C1,
    f32,
    nppiGradientVectorPrewittBorder_16u32f_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.",
    gradient_vector_prewitt_border_u16_c3_to_f32_c1,
    u16,
    C3,
    f32,
    nppiGradientVectorPrewittBorder_16u32f_C3C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.",
    gradient_vector_prewitt_border_f32_to_f32_c1,
    f32,
    C1,
    f32,
    nppiGradientVectorPrewittBorder_32f_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.",
    gradient_vector_prewitt_border_f32_c3_to_f32_c1,
    f32,
    C3,
    f32,
    nppiGradientVectorPrewittBorder_32f_C3C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.",
    gradient_vector_scharr_border_u8_to_i16_f32_c1,
    u8,
    C1,
    i16,
    nppiGradientVectorScharrBorder_8u16s_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.",
    gradient_vector_scharr_border_u8_c3_to_i16_f32_c1,
    u8,
    C3,
    i16,
    nppiGradientVectorScharrBorder_8u16s_C3C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.",
    gradient_vector_scharr_border_i16_to_f32_c1,
    i16,
    C1,
    f32,
    nppiGradientVectorScharrBorder_16s32f_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.",
    gradient_vector_scharr_border_i16_c3_to_f32_c1,
    i16,
    C3,
    f32,
    nppiGradientVectorScharrBorder_16s32f_C3C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.",
    gradient_vector_scharr_border_u16_to_f32_c1,
    u16,
    C1,
    f32,
    nppiGradientVectorScharrBorder_16u32f_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.",
    gradient_vector_scharr_border_u16_c3_to_f32_c1,
    u16,
    C3,
    f32,
    nppiGradientVectorScharrBorder_16u32f_C3C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.",
    gradient_vector_scharr_border_f32_to_f32_c1,
    f32,
    C1,
    f32,
    nppiGradientVectorScharrBorder_32f_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.",
    gradient_vector_scharr_border_f32_c3_to_f32_c1,
    f32,
    C3,
    f32,
    nppiGradientVectorScharrBorder_32f_C3C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.",
    gradient_vector_sobel_border_u8_to_i16_f32_c1,
    u8,
    C1,
    i16,
    nppiGradientVectorSobelBorder_8u16s_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.",
    gradient_vector_sobel_border_u8_c3_to_i16_f32_c1,
    u8,
    C3,
    i16,
    nppiGradientVectorSobelBorder_8u16s_C3C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.",
    gradient_vector_sobel_border_i16_to_f32_c1,
    i16,
    C1,
    f32,
    nppiGradientVectorSobelBorder_16s32f_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.",
    gradient_vector_sobel_border_i16_c3_to_f32_c1,
    i16,
    C3,
    f32,
    nppiGradientVectorSobelBorder_16s32f_C3C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.",
    gradient_vector_sobel_border_u16_to_f32_c1,
    u16,
    C1,
    f32,
    nppiGradientVectorSobelBorder_16u32f_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.",
    gradient_vector_sobel_border_u16_c3_to_f32_c1,
    u16,
    C3,
    f32,
    nppiGradientVectorSobelBorder_16u32f_C3C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.",
    gradient_vector_sobel_border_f32_to_f32_c1,
    f32,
    C1,
    f32,
    nppiGradientVectorSobelBorder_32f_C1R_Ctx
);

impl_gradient_vector_border!(
    "Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.",
    gradient_vector_sobel_border_f32_c3_to_f32_c1,
    f32,
    C3,
    f32,
    nppiGradientVectorSobelBorder_32f_C3C1R_Ctx
);

macro_rules! impl_gradient_vector_border_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $gradient_ty:ty, $direct_name:ident) => {
        unsafe impl $trait_name for $source_ty {
            type Gradient = $gradient_ty;

            unsafe fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_offset: Point,
                destination_x: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
                destination_y: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
                destination_magnitude: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
                destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
                mask_size: MaskSize,
                norm: ImageNormalization,
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $direct_name(
                        stream_context,
                        source,
                        source_offset,
                        destination_x,
                        destination_y,
                        destination_magnitude,
                        destination_angle,
                        mask_size,
                        norm,
                        border_type,
                    )
                }
            }
        }
    };
}

macro_rules! impl_gradient_vector_border_c3_to_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $gradient_ty:ty, $direct_name:ident) => {
        unsafe impl $trait_name for $source_ty {
            type Gradient = $gradient_ty;

            unsafe fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                source_offset: Point,
                destination_x: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
                destination_y: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
                destination_magnitude: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
                destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
                mask_size: MaskSize,
                norm: ImageNormalization,
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $direct_name(
                        stream_context,
                        source,
                        source_offset,
                        destination_x,
                        destination_y,
                        destination_magnitude,
                        destination_angle,
                        mask_size,
                        norm,
                        border_type,
                    )
                }
            }
        }
    };
}

/// Dispatches optional gradient-vector outputs with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait GradientVectorPrewittBorderC1: DataTypeLike {
    /// Element type of the X derivative, Y derivative, and gradient-magnitude outputs.
    type Gradient: DataTypeLike;

    /// Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.
    ///
    /// Enqueues work on the selected stream.
    /// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
    /// At least one output must be provided; all supplied outputs must have the same positive dimensions.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn gradient_vector_prewitt_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination_x: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorPrewittBorderC1,
    gradient_vector_prewitt_border,
    u8,
    i16,
    gradient_vector_prewitt_border_u8_to_i16_f32_c1
);

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorPrewittBorderC1,
    gradient_vector_prewitt_border,
    i16,
    f32,
    gradient_vector_prewitt_border_i16_to_f32_c1
);

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorPrewittBorderC1,
    gradient_vector_prewitt_border,
    u16,
    f32,
    gradient_vector_prewitt_border_u16_to_f32_c1
);

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorPrewittBorderC1,
    gradient_vector_prewitt_border,
    f32,
    f32,
    gradient_vector_prewitt_border_f32_to_f32_c1
);

/// Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.
///
/// Enqueues work on the selected stream.
/// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
/// At least one output must be provided; all supplied outputs must have the same positive dimensions.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn gradient_vector_prewitt_border<T: GradientVectorPrewittBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination_x: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::gradient_vector_prewitt_border(
            stream_context,
            source,
            source_offset,
            destination_x,
            destination_y,
            destination_magnitude,
            destination_angle,
            mask_size,
            norm,
            border_type,
        )
    }
}

/// Dispatches optional gradient-vector outputs with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait GradientVectorPrewittBorderC3ToC1: DataTypeLike {
    /// Element type of the X derivative, Y derivative, and gradient-magnitude outputs.
    type Gradient: DataTypeLike;

    /// Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.
    ///
    /// Enqueues work on the selected stream.
    /// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
    /// At least one output must be provided; all supplied outputs must have the same positive dimensions.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn gradient_vector_prewitt_border_color(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination_x: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorPrewittBorderC3ToC1,
    gradient_vector_prewitt_border_color,
    u8,
    i16,
    gradient_vector_prewitt_border_u8_c3_to_i16_f32_c1
);

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorPrewittBorderC3ToC1,
    gradient_vector_prewitt_border_color,
    i16,
    f32,
    gradient_vector_prewitt_border_i16_c3_to_f32_c1
);

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorPrewittBorderC3ToC1,
    gradient_vector_prewitt_border_color,
    u16,
    f32,
    gradient_vector_prewitt_border_u16_c3_to_f32_c1
);

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorPrewittBorderC3ToC1,
    gradient_vector_prewitt_border_color,
    f32,
    f32,
    gradient_vector_prewitt_border_f32_c3_to_f32_c1
);

/// Computes optional X and Y derivatives, gradient magnitude, and angle using a Prewitt filter with border control.
///
/// Enqueues work on the selected stream.
/// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
/// At least one output must be provided; all supplied outputs must have the same positive dimensions.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn gradient_vector_prewitt_border_color<T: GradientVectorPrewittBorderC3ToC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    source_offset: Point,
    destination_x: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::gradient_vector_prewitt_border_color(
            stream_context,
            source,
            source_offset,
            destination_x,
            destination_y,
            destination_magnitude,
            destination_angle,
            mask_size,
            norm,
            border_type,
        )
    }
}

/// Dispatches optional gradient-vector outputs with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait GradientVectorScharrBorderC1: DataTypeLike {
    /// Element type of the X derivative, Y derivative, and gradient-magnitude outputs.
    type Gradient: DataTypeLike;

    /// Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.
    ///
    /// Enqueues work on the selected stream.
    /// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
    /// At least one output must be provided; all supplied outputs must have the same positive dimensions.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn gradient_vector_scharr_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination_x: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorScharrBorderC1,
    gradient_vector_scharr_border,
    u8,
    i16,
    gradient_vector_scharr_border_u8_to_i16_f32_c1
);

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorScharrBorderC1,
    gradient_vector_scharr_border,
    i16,
    f32,
    gradient_vector_scharr_border_i16_to_f32_c1
);

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorScharrBorderC1,
    gradient_vector_scharr_border,
    u16,
    f32,
    gradient_vector_scharr_border_u16_to_f32_c1
);

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorScharrBorderC1,
    gradient_vector_scharr_border,
    f32,
    f32,
    gradient_vector_scharr_border_f32_to_f32_c1
);

/// Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.
///
/// Enqueues work on the selected stream.
/// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
/// At least one output must be provided; all supplied outputs must have the same positive dimensions.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn gradient_vector_scharr_border<T: GradientVectorScharrBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination_x: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::gradient_vector_scharr_border(
            stream_context,
            source,
            source_offset,
            destination_x,
            destination_y,
            destination_magnitude,
            destination_angle,
            mask_size,
            norm,
            border_type,
        )
    }
}

/// Dispatches optional gradient-vector outputs with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait GradientVectorScharrBorderC3ToC1: DataTypeLike {
    /// Element type of the X derivative, Y derivative, and gradient-magnitude outputs.
    type Gradient: DataTypeLike;

    /// Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.
    ///
    /// Enqueues work on the selected stream.
    /// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
    /// At least one output must be provided; all supplied outputs must have the same positive dimensions.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn gradient_vector_scharr_border_color(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination_x: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorScharrBorderC3ToC1,
    gradient_vector_scharr_border_color,
    u8,
    i16,
    gradient_vector_scharr_border_u8_c3_to_i16_f32_c1
);

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorScharrBorderC3ToC1,
    gradient_vector_scharr_border_color,
    i16,
    f32,
    gradient_vector_scharr_border_i16_c3_to_f32_c1
);

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorScharrBorderC3ToC1,
    gradient_vector_scharr_border_color,
    u16,
    f32,
    gradient_vector_scharr_border_u16_c3_to_f32_c1
);

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorScharrBorderC3ToC1,
    gradient_vector_scharr_border_color,
    f32,
    f32,
    gradient_vector_scharr_border_f32_c3_to_f32_c1
);

/// Computes optional X and Y derivatives, gradient magnitude, and angle using a Scharr filter with border control.
///
/// Enqueues work on the selected stream.
/// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
/// At least one output must be provided; all supplied outputs must have the same positive dimensions.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn gradient_vector_scharr_border_color<T: GradientVectorScharrBorderC3ToC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    source_offset: Point,
    destination_x: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::gradient_vector_scharr_border_color(
            stream_context,
            source,
            source_offset,
            destination_x,
            destination_y,
            destination_magnitude,
            destination_angle,
            mask_size,
            norm,
            border_type,
        )
    }
}

/// Dispatches optional gradient-vector outputs with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait GradientVectorSobelBorderC1: DataTypeLike {
    /// Element type of the X derivative, Y derivative, and gradient-magnitude outputs.
    type Gradient: DataTypeLike;

    /// Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.
    ///
    /// Enqueues work on the selected stream.
    /// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
    /// At least one output must be provided; all supplied outputs must have the same positive dimensions.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn gradient_vector_sobel_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination_x: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorSobelBorderC1,
    gradient_vector_sobel_border,
    u8,
    i16,
    gradient_vector_sobel_border_u8_to_i16_f32_c1
);

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorSobelBorderC1,
    gradient_vector_sobel_border,
    i16,
    f32,
    gradient_vector_sobel_border_i16_to_f32_c1
);

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorSobelBorderC1,
    gradient_vector_sobel_border,
    u16,
    f32,
    gradient_vector_sobel_border_u16_to_f32_c1
);

impl_gradient_vector_border_c1_dispatch!(
    GradientVectorSobelBorderC1,
    gradient_vector_sobel_border,
    f32,
    f32,
    gradient_vector_sobel_border_f32_to_f32_c1
);

/// Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.
///
/// Enqueues work on the selected stream.
/// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
/// At least one output must be provided; all supplied outputs must have the same positive dimensions.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn gradient_vector_sobel_border<T: GradientVectorSobelBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination_x: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::gradient_vector_sobel_border(
            stream_context,
            source,
            source_offset,
            destination_x,
            destination_y,
            destination_magnitude,
            destination_angle,
            mask_size,
            norm,
            border_type,
        )
    }
}

/// Dispatches optional gradient-vector outputs with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel representations, derivative orientation, output element types, and kernel semantics.
/// Uphold the documented raw image execution contract, including source halos where required and queued allocation lifetimes.
pub unsafe trait GradientVectorSobelBorderC3ToC1: DataTypeLike {
    /// Element type of the X derivative, Y derivative, and gradient-magnitude outputs.
    type Gradient: DataTypeLike;

    /// Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.
    ///
    /// Enqueues work on the selected stream.
    /// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
    /// At least one output must be provided; all supplied outputs must have the same positive dimensions.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn gradient_vector_sobel_border_color(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination_x: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_y: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_magnitude: Option<&mut ImageOutputView<'_, Self::Gradient, C1>>,
        destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorSobelBorderC3ToC1,
    gradient_vector_sobel_border_color,
    u8,
    i16,
    gradient_vector_sobel_border_u8_c3_to_i16_f32_c1
);

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorSobelBorderC3ToC1,
    gradient_vector_sobel_border_color,
    i16,
    f32,
    gradient_vector_sobel_border_i16_c3_to_f32_c1
);

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorSobelBorderC3ToC1,
    gradient_vector_sobel_border_color,
    u16,
    f32,
    gradient_vector_sobel_border_u16_c3_to_f32_c1
);

impl_gradient_vector_border_c3_to_c1_dispatch!(
    GradientVectorSobelBorderC3ToC1,
    gradient_vector_sobel_border_color,
    f32,
    f32,
    gradient_vector_sobel_border_f32_c3_to_f32_c1
);

/// Computes optional X and Y derivatives, gradient magnitude, and angle using a Sobel filter with border control.
///
/// Enqueues work on the selected stream.
/// `None` disables an output; `mask_size` selects the fixed derivative mask and `norm` selects the magnitude calculation.
/// Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the output-sized ROI must fit within the source image.
/// At least one output must be provided; all supplied outputs must have the same positive dimensions.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep source storage read-only and each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn gradient_vector_sobel_border_color<T: GradientVectorSobelBorderC3ToC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    source_offset: Point,
    destination_x: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_y: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_magnitude: Option<&mut ImageOutputView<'_, T::Gradient, C1>>,
    destination_angle: Option<&mut ImageOutputView<'_, f32, C1>>,
    mask_size: MaskSize,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::gradient_vector_sobel_border_color(
            stream_context,
            source,
            source_offset,
            destination_x,
            destination_y,
            destination_magnitude,
            destination_angle,
            mask_size,
            norm,
            border_type,
        )
    }
}
