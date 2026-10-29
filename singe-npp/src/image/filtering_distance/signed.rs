use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_distance::raw::{
            signed_distance_transform_abs_pba_f32_c1,
            signed_distance_transform_abs_pba_f32_to_f64_c1,
            signed_distance_transform_abs_pba_f32_to_f64_c1_antialiasing,
            signed_distance_transform_abs_pba_f64_c1,
            signed_distance_transform_abs_pba_f64_c1_antialiasing,
            signed_distance_transform_pba_f32_c1, signed_distance_transform_pba_f32_to_f64_c1,
            signed_distance_transform_pba_f32_to_f64_c1_antialiasing,
            signed_distance_transform_pba_f64_c1,
            signed_distance_transform_pba_f64_c1_antialiasing,
        },
        view::{C1, C2, ImageOutputView, ImageView},
    },
    types::DataTypeLike,
};

/// Dispatches signed PBA Euclidean distance transforms with relative Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait SignedDistanceTransformPbaToF32C1: DataTypeLike {
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a signed PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f32` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
    /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
    /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready for native execution.
    /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
    /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
    /// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn signed_distance_transform_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        horizontal_subpixel_offset: f32,
        vertical_subpixel_offset: f32,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
    ) -> Result<()>;
}

unsafe impl SignedDistanceTransformPbaToF32C1 for f32 {
    type Manhattan = i16;

    unsafe fn signed_distance_transform_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        horizontal_subpixel_offset: f32,
        vertical_subpixel_offset: f32,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            signed_distance_transform_pba_f32_c1(
                stream_context,
                source,
                cutoff_value,
                horizontal_subpixel_offset,
                vertical_subpixel_offset,
                destination_voronoi,
                destination_voronoi_indices,
                destination_voronoi_manhattan,
                destination_transform,
            )
        }
    }
}

/// Computes a signed PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f32` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
/// At least one output must have positive dimensions; all supplied outputs must have the source image size.
/// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready for native execution.
/// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
/// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
/// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn signed_distance_transform_pba_to_f32<T: SignedDistanceTransformPbaToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    horizontal_subpixel_offset: f32,
    vertical_subpixel_offset: f32,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::signed_distance_transform_pba_to_f32(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches signed PBA Euclidean distance transforms with absolute Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait SignedDistanceTransformAbsPbaToF32C1: DataTypeLike {
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a signed PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f32` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
    /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
    /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready for native execution.
    /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
    /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
    /// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn signed_distance_transform_abs_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        horizontal_subpixel_offset: f32,
        vertical_subpixel_offset: f32,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
    ) -> Result<()>;
}

unsafe impl SignedDistanceTransformAbsPbaToF32C1 for f32 {
    type Manhattan = u16;

    unsafe fn signed_distance_transform_abs_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        horizontal_subpixel_offset: f32,
        vertical_subpixel_offset: f32,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            signed_distance_transform_abs_pba_f32_c1(
                stream_context,
                source,
                cutoff_value,
                horizontal_subpixel_offset,
                vertical_subpixel_offset,
                destination_voronoi,
                destination_voronoi_indices,
                destination_voronoi_manhattan,
                destination_transform,
            )
        }
    }
}

/// Computes a signed PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f32` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
/// At least one output must have positive dimensions; all supplied outputs must have the source image size.
/// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready for native execution.
/// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
/// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
/// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn signed_distance_transform_abs_pba_to_f32<T: SignedDistanceTransformAbsPbaToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    horizontal_subpixel_offset: f32,
    vertical_subpixel_offset: f32,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::signed_distance_transform_abs_pba_to_f32(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches signed PBA Euclidean distance transforms with relative Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait SignedDistanceTransformPbaToF64C1: DataTypeLike {
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a signed PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f64` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
    /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
    /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready for native execution.
    /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
    /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
    /// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn signed_distance_transform_pba_to_f64(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
    ) -> Result<()>;

    /// Computes a signed PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f64` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
    /// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
    /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
    /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready for native execution.
    /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
    /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
    /// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn signed_distance_transform_pba_to_f64_antialiasing(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_signed_distance_transform_pba_to_f64 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident, $antialiasing_name:ident) => {
        unsafe impl SignedDistanceTransformPbaToF64C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            unsafe fn signed_distance_transform_pba_to_f64(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                horizontal_subpixel_offset: f64,
                vertical_subpixel_offset: f64,
                destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<
                    &mut ImageOutputView<'_, Self::Manhattan, C2>,
                >,
                destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        cutoff_value,
                        horizontal_subpixel_offset,
                        vertical_subpixel_offset,
                        destination_voronoi,
                        destination_voronoi_indices,
                        destination_voronoi_manhattan,
                        destination_transform,
                    )
                }
            }

            unsafe fn signed_distance_transform_pba_to_f64_antialiasing(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                horizontal_subpixel_offset: f64,
                vertical_subpixel_offset: f64,
                destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<
                    &mut ImageOutputView<'_, Self::Manhattan, C2>,
                >,
                destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $antialiasing_name(
                        stream_context,
                        source,
                        cutoff_value,
                        horizontal_subpixel_offset,
                        vertical_subpixel_offset,
                        destination_voronoi,
                        destination_voronoi_indices,
                        destination_voronoi_manhattan,
                        destination_transform,
                    )
                }
            }
        }
    };
}

impl_signed_distance_transform_pba_to_f64!(
    f32,
    i16,
    signed_distance_transform_pba_f32_to_f64_c1,
    signed_distance_transform_pba_f32_to_f64_c1_antialiasing
);

impl_signed_distance_transform_pba_to_f64!(
    f64,
    i16,
    signed_distance_transform_pba_f64_c1,
    signed_distance_transform_pba_f64_c1_antialiasing
);

/// Computes a signed PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
/// At least one output must have positive dimensions; all supplied outputs must have the source image size.
/// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready for native execution.
/// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
/// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
/// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn signed_distance_transform_pba_to_f64<T: SignedDistanceTransformPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    horizontal_subpixel_offset: f64,
    vertical_subpixel_offset: f64,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::signed_distance_transform_pba_to_f64(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Computes a signed PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
/// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
/// At least one output must have positive dimensions; all supplied outputs must have the source image size.
/// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready for native execution.
/// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
/// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
/// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn signed_distance_transform_pba_to_f64_antialiasing<
    T: SignedDistanceTransformPbaToF64C1,
>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    horizontal_subpixel_offset: f64,
    vertical_subpixel_offset: f64,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::signed_distance_transform_pba_to_f64_antialiasing(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches signed PBA Euclidean distance transforms with absolute Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait SignedDistanceTransformAbsPbaToF64C1: DataTypeLike {
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a signed PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f64` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
    /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
    /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready for native execution.
    /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
    /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
    /// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn signed_distance_transform_abs_pba_to_f64(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
    ) -> Result<()>;

    /// Computes a signed PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f64` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
    /// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
    /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
    /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready for native execution.
    /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
    /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
    /// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn signed_distance_transform_abs_pba_to_f64_antialiasing(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_signed_distance_transform_abs_pba_to_f64 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident, $antialiasing_name:ident) => {
        unsafe impl SignedDistanceTransformAbsPbaToF64C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            unsafe fn signed_distance_transform_abs_pba_to_f64(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                horizontal_subpixel_offset: f64,
                vertical_subpixel_offset: f64,
                destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<
                    &mut ImageOutputView<'_, Self::Manhattan, C2>,
                >,
                destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        cutoff_value,
                        horizontal_subpixel_offset,
                        vertical_subpixel_offset,
                        destination_voronoi,
                        destination_voronoi_indices,
                        destination_voronoi_manhattan,
                        destination_transform,
                    )
                }
            }

            unsafe fn signed_distance_transform_abs_pba_to_f64_antialiasing(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                horizontal_subpixel_offset: f64,
                vertical_subpixel_offset: f64,
                destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<
                    &mut ImageOutputView<'_, Self::Manhattan, C2>,
                >,
                destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $antialiasing_name(
                        stream_context,
                        source,
                        cutoff_value,
                        horizontal_subpixel_offset,
                        vertical_subpixel_offset,
                        destination_voronoi,
                        destination_voronoi_indices,
                        destination_voronoi_manhattan,
                        destination_transform,
                    )
                }
            }
        }
    };
}

impl_signed_distance_transform_abs_pba_to_f64!(
    f32,
    u16,
    signed_distance_transform_abs_pba_f32_to_f64_c1,
    signed_distance_transform_abs_pba_f32_to_f64_c1_antialiasing
);

impl_signed_distance_transform_abs_pba_to_f64!(
    f64,
    u16,
    signed_distance_transform_abs_pba_f64_c1,
    signed_distance_transform_abs_pba_f64_c1_antialiasing
);

/// Computes a signed PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
/// At least one output must have positive dimensions; all supplied outputs must have the source image size.
/// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready for native execution.
/// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
/// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
/// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn signed_distance_transform_abs_pba_to_f64<T: SignedDistanceTransformAbsPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    horizontal_subpixel_offset: f64,
    vertical_subpixel_offset: f64,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::signed_distance_transform_abs_pba_to_f64(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Computes a signed PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
/// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
/// At least one output must have positive dimensions; all supplied outputs must have the source image size.
/// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready for native execution.
/// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
/// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
/// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn signed_distance_transform_abs_pba_to_f64_antialiasing<
    T: SignedDistanceTransformAbsPbaToF64C1,
>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    horizontal_subpixel_offset: f64,
    vertical_subpixel_offset: f64,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::signed_distance_transform_abs_pba_to_f64_antialiasing(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches signed PBA Euclidean distance transforms with relative Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait SignedDistanceTransformPbaC1<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Floating-point type for horizontal and vertical subpixel offsets.
    type Shift: DataTypeLike;

    /// Computes a signed PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
    /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
    /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready for native execution.
    /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
    /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
    /// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn signed_distance_transform_pba_c1(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        horizontal_subpixel_offset: Self::Shift,
        vertical_subpixel_offset: Self::Shift,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, Destination, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_signed_distance_transform_pba_c1 {
    ($destination:ty, $shift:ty, $source_trait:ident, $source_method:ident) => {
        unsafe impl<T> SignedDistanceTransformPbaC1<$destination> for T
        where
            T: $source_trait,
        {
            type Manhattan = <T as $source_trait>::Manhattan;
            type Shift = $shift;

            unsafe fn signed_distance_transform_pba_c1(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                horizontal_subpixel_offset: Self::Shift,
                vertical_subpixel_offset: Self::Shift,
                destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<
                    &mut ImageOutputView<'_, Self::Manhattan, C2>,
                >,
                destination_transform: Option<&mut ImageOutputView<'_, $destination, C1>>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    T::$source_method(
                        stream_context,
                        source,
                        cutoff_value,
                        horizontal_subpixel_offset,
                        vertical_subpixel_offset,
                        destination_voronoi,
                        destination_voronoi_indices,
                        destination_voronoi_manhattan,
                        destination_transform,
                    )
                }
            }
        }
    };
}

impl_signed_distance_transform_pba_c1!(
    f32,
    f32,
    SignedDistanceTransformPbaToF32C1,
    signed_distance_transform_pba_to_f32
);

impl_signed_distance_transform_pba_c1!(
    f64,
    f64,
    SignedDistanceTransformPbaToF64C1,
    signed_distance_transform_pba_to_f64
);

/// Computes a signed PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
/// At least one output must have positive dimensions; all supplied outputs must have the source image size.
/// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready for native execution.
/// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
/// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
/// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn signed_distance_transform_pba<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    horizontal_subpixel_offset: T::Shift,
    vertical_subpixel_offset: T::Shift,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, Destination, C1>>,
) -> Result<()>
where
    T: SignedDistanceTransformPbaC1<Destination>,
    Destination: DataTypeLike,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::signed_distance_transform_pba_c1(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Computes a signed PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
/// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
/// At least one output must have positive dimensions; all supplied outputs must have the source image size.
/// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready for native execution.
/// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
/// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
/// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn signed_distance_transform_pba_antialiasing<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    horizontal_subpixel_offset: f64,
    vertical_subpixel_offset: f64,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()>
where
    T: SignedDistanceTransformPbaToF64C1,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        signed_distance_transform_pba_to_f64_antialiasing(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches signed PBA Euclidean distance transforms with absolute Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait SignedDistanceTransformAbsPbaC1<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Floating-point type for horizontal and vertical subpixel offsets.
    type Shift: DataTypeLike;

    /// Computes a signed PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
    /// At least one output must have positive dimensions; all supplied outputs must have the source image size.
    /// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready for native execution.
    /// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
    /// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
    /// Keep image allocations valid until this call completes its native work.
    /// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
    /// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn signed_distance_transform_abs_pba_c1(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        horizontal_subpixel_offset: Self::Shift,
        vertical_subpixel_offset: Self::Shift,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, Destination, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_signed_distance_transform_abs_pba_c1 {
    ($destination:ty, $shift:ty, $source_trait:ident, $source_method:ident) => {
        unsafe impl<T> SignedDistanceTransformAbsPbaC1<$destination> for T
        where
            T: $source_trait,
        {
            type Manhattan = <T as $source_trait>::Manhattan;
            type Shift = $shift;

            unsafe fn signed_distance_transform_abs_pba_c1(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                horizontal_subpixel_offset: Self::Shift,
                vertical_subpixel_offset: Self::Shift,
                destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<
                    &mut ImageOutputView<'_, Self::Manhattan, C2>,
                >,
                destination_transform: Option<&mut ImageOutputView<'_, $destination, C1>>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    T::$source_method(
                        stream_context,
                        source,
                        cutoff_value,
                        horizontal_subpixel_offset,
                        vertical_subpixel_offset,
                        destination_voronoi,
                        destination_voronoi_indices,
                        destination_voronoi_manhattan,
                        destination_transform,
                    )
                }
            }
        }
    };
}

impl_signed_distance_transform_abs_pba_c1!(
    f32,
    f32,
    SignedDistanceTransformAbsPbaToF32C1,
    signed_distance_transform_abs_pba_to_f32
);

impl_signed_distance_transform_abs_pba_c1!(
    f64,
    f64,
    SignedDistanceTransformAbsPbaToF64C1,
    signed_distance_transform_abs_pba_to_f64
);

/// Computes a signed PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
/// At least one output must have positive dimensions; all supplied outputs must have the source image size.
/// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready for native execution.
/// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
/// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
/// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn signed_distance_transform_abs_pba<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    horizontal_subpixel_offset: T::Shift,
    vertical_subpixel_offset: T::Shift,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, Destination, C1>>,
) -> Result<()>
where
    T: SignedDistanceTransformAbsPbaC1<Destination>,
    Destination: DataTypeLike,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::signed_distance_transform_abs_pba_c1(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Computes a signed PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
/// `cutoff_value` classifies source pixels. The horizontal and vertical offsets shift the final transform by the supplied subpixel fractions.
/// Row padding remains unwritten.
///
/// # Errors
///
/// The provided implementations return an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the transform parameters.
/// At least one output must have positive dimensions; all supplied outputs must have the source image size.
/// Workspace-size queries, device allocation, capture checks, and stream synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready for native execution.
/// NPP receives a mutable source pointer; provide exclusive access to source storage during native use.
/// Keep each destination exclusively writable and disjoint from the source and all other destinations during native use.
/// Keep image allocations valid until this call completes its native work.
/// Native PBA requires ROI width and height in `64..=32767`; the provided implementations check positive dimensions but do not enforce that range.
/// Source classification and subpixel parameters must satisfy the native signed-distance input requirements; the provided implementations do not validate their values.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn signed_distance_transform_abs_pba_antialiasing<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    horizontal_subpixel_offset: f64,
    vertical_subpixel_offset: f64,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()>
where
    T: SignedDistanceTransformAbsPbaToF64C1,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        signed_distance_transform_abs_pba_to_f64_antialiasing(
            stream_context,
            source,
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}
