use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_distance::raw::{
            distance_transform_abs_pba_f32_to_f64_c1,
            distance_transform_abs_pba_f32_to_f64_c1_antialiasing,
            distance_transform_abs_pba_f64_c1, distance_transform_abs_pba_f64_c1_antialiasing,
            distance_transform_abs_pba_i8_to_f32_c1, distance_transform_abs_pba_i8_to_f64_c1,
            distance_transform_abs_pba_i8_to_f64_c1_antialiasing,
            distance_transform_abs_pba_i8_to_u16_c1, distance_transform_abs_pba_i16_to_f32_c1,
            distance_transform_abs_pba_i16_to_f64_c1,
            distance_transform_abs_pba_i16_to_f64_c1_antialiasing,
            distance_transform_abs_pba_i16_to_u16_c1, distance_transform_abs_pba_u8_to_f32_c1,
            distance_transform_abs_pba_u8_to_f64_c1,
            distance_transform_abs_pba_u8_to_f64_c1_antialiasing,
            distance_transform_abs_pba_u8_to_u16_c1, distance_transform_abs_pba_u16_to_f32_c1,
            distance_transform_abs_pba_u16_to_f64_c1,
            distance_transform_abs_pba_u16_to_f64_c1_antialiasing,
            distance_transform_abs_pba_u16_to_u16_c1, distance_transform_pba_f32_to_f64_c1,
            distance_transform_pba_f32_to_f64_c1_antialiasing, distance_transform_pba_f64_c1,
            distance_transform_pba_f64_c1_antialiasing, distance_transform_pba_i8_to_f32_c1,
            distance_transform_pba_i8_to_f64_c1, distance_transform_pba_i8_to_f64_c1_antialiasing,
            distance_transform_pba_i8_to_u16_c1, distance_transform_pba_i16_to_f32_c1,
            distance_transform_pba_i16_to_f64_c1,
            distance_transform_pba_i16_to_f64_c1_antialiasing,
            distance_transform_pba_i16_to_u16_c1, distance_transform_pba_u8_to_f32_c1,
            distance_transform_pba_u8_to_f64_c1, distance_transform_pba_u8_to_f64_c1_antialiasing,
            distance_transform_pba_u8_to_u16_c1, distance_transform_pba_u16_to_f32_c1,
            distance_transform_pba_u16_to_f64_c1,
            distance_transform_pba_u16_to_f64_c1_antialiasing,
            distance_transform_pba_u16_to_u16_c1,
        },
        view::{C1, C2, ImageOutputView, ImageView},
    },
    types::DataTypeLike,
};

/// Dispatches PBA Euclidean distance transforms with relative Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait DistanceTransformPbaToU16C1: DataTypeLike {
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses truncated `u16` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
    /// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn distance_transform_pba_to_u16(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, u16, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_pba_to_u16 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident) => {
        unsafe impl DistanceTransformPbaToU16C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            unsafe fn distance_transform_pba_to_u16(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<
                    &mut ImageOutputView<'_, Self::Manhattan, C2>,
                >,
                destination_transform: Option<&mut ImageOutputView<'_, u16, C1>>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        minimum_site_value,
                        maximum_site_value,
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

impl_distance_transform_pba_to_u16!(u8, i16, distance_transform_pba_u8_to_u16_c1);

impl_distance_transform_pba_to_u16!(i8, i16, distance_transform_pba_i8_to_u16_c1);

impl_distance_transform_pba_to_u16!(u16, i16, distance_transform_pba_u16_to_u16_c1);

impl_distance_transform_pba_to_u16!(i16, i16, distance_transform_pba_i16_to_u16_c1);

/// Computes a PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses truncated `u16` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_pba_to_u16<T: DistanceTransformPbaToU16C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, u16, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::distance_transform_pba_to_u16(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches PBA Euclidean distance transforms with absolute Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait DistanceTransformAbsPbaToU16C1: DataTypeLike {
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses truncated `u16` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
    /// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn distance_transform_abs_pba_to_u16(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, u16, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_abs_pba_to_u16 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident) => {
        unsafe impl DistanceTransformAbsPbaToU16C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            unsafe fn distance_transform_abs_pba_to_u16(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<
                    &mut ImageOutputView<'_, Self::Manhattan, C2>,
                >,
                destination_transform: Option<&mut ImageOutputView<'_, u16, C1>>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        minimum_site_value,
                        maximum_site_value,
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

impl_distance_transform_abs_pba_to_u16!(u8, u16, distance_transform_abs_pba_u8_to_u16_c1);

impl_distance_transform_abs_pba_to_u16!(i8, u16, distance_transform_abs_pba_i8_to_u16_c1);

impl_distance_transform_abs_pba_to_u16!(u16, u16, distance_transform_abs_pba_u16_to_u16_c1);

impl_distance_transform_abs_pba_to_u16!(i16, u16, distance_transform_abs_pba_i16_to_u16_c1);

/// Computes a PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses truncated `u16` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_abs_pba_to_u16<T: DistanceTransformAbsPbaToU16C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, u16, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::distance_transform_abs_pba_to_u16(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches PBA Euclidean distance transforms with relative Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait DistanceTransformPbaToF32C1: DataTypeLike {
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f32` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
    /// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn distance_transform_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_pba_to_f32 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident) => {
        unsafe impl DistanceTransformPbaToF32C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            unsafe fn distance_transform_pba_to_f32(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<
                    &mut ImageOutputView<'_, Self::Manhattan, C2>,
                >,
                destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        minimum_site_value,
                        maximum_site_value,
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

impl_distance_transform_pba_to_f32!(u8, i16, distance_transform_pba_u8_to_f32_c1);

impl_distance_transform_pba_to_f32!(i8, i16, distance_transform_pba_i8_to_f32_c1);

impl_distance_transform_pba_to_f32!(u16, i16, distance_transform_pba_u16_to_f32_c1);

impl_distance_transform_pba_to_f32!(i16, i16, distance_transform_pba_i16_to_f32_c1);

/// Computes a PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f32` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_pba_to_f32<T: DistanceTransformPbaToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::distance_transform_pba_to_f32(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches PBA Euclidean distance transforms with absolute Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait DistanceTransformAbsPbaToF32C1: DataTypeLike {
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f32` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
    /// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn distance_transform_abs_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_abs_pba_to_f32 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident) => {
        unsafe impl DistanceTransformAbsPbaToF32C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            unsafe fn distance_transform_abs_pba_to_f32(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<
                    &mut ImageOutputView<'_, Self::Manhattan, C2>,
                >,
                destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        minimum_site_value,
                        maximum_site_value,
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

impl_distance_transform_abs_pba_to_f32!(u8, u16, distance_transform_abs_pba_u8_to_f32_c1);

impl_distance_transform_abs_pba_to_f32!(i8, u16, distance_transform_abs_pba_i8_to_f32_c1);

impl_distance_transform_abs_pba_to_f32!(u16, u16, distance_transform_abs_pba_u16_to_f32_c1);

impl_distance_transform_abs_pba_to_f32!(i16, u16, distance_transform_abs_pba_i16_to_f32_c1);

/// Computes a PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f32` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_abs_pba_to_f32<T: DistanceTransformAbsPbaToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f32, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::distance_transform_abs_pba_to_f32(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches PBA Euclidean distance transforms with relative Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait DistanceTransformPbaToF64C1: DataTypeLike {
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f64` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
    /// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn distance_transform_pba_to_f64(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
    ) -> Result<()>;

    /// Computes a PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f64` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
    /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
    /// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn distance_transform_pba_to_f64_antialiasing(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_pba_to_f64 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident, $antialiasing_name:ident) => {
        unsafe impl DistanceTransformPbaToF64C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            unsafe fn distance_transform_pba_to_f64(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
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
                        minimum_site_value,
                        maximum_site_value,
                        destination_voronoi,
                        destination_voronoi_indices,
                        destination_voronoi_manhattan,
                        destination_transform,
                    )
                }
            }

            unsafe fn distance_transform_pba_to_f64_antialiasing(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
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
                        minimum_site_value,
                        maximum_site_value,
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

impl_distance_transform_pba_to_f64!(
    u8,
    i16,
    distance_transform_pba_u8_to_f64_c1,
    distance_transform_pba_u8_to_f64_c1_antialiasing
);

impl_distance_transform_pba_to_f64!(
    i8,
    i16,
    distance_transform_pba_i8_to_f64_c1,
    distance_transform_pba_i8_to_f64_c1_antialiasing
);

impl_distance_transform_pba_to_f64!(
    u16,
    i16,
    distance_transform_pba_u16_to_f64_c1,
    distance_transform_pba_u16_to_f64_c1_antialiasing
);

impl_distance_transform_pba_to_f64!(
    i16,
    i16,
    distance_transform_pba_i16_to_f64_c1,
    distance_transform_pba_i16_to_f64_c1_antialiasing
);

impl_distance_transform_pba_to_f64!(
    f32,
    i16,
    distance_transform_pba_f32_to_f64_c1,
    distance_transform_pba_f32_to_f64_c1_antialiasing
);

impl_distance_transform_pba_to_f64!(
    f64,
    i16,
    distance_transform_pba_f64_c1,
    distance_transform_pba_f64_c1_antialiasing
);

/// Computes a PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_pba_to_f64<T: DistanceTransformPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::distance_transform_pba_to_f64(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Computes a PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_pba_to_f64_antialiasing<T: DistanceTransformPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::distance_transform_pba_to_f64_antialiasing(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches PBA Euclidean distance transforms with absolute Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait DistanceTransformAbsPbaToF64C1: DataTypeLike {
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f64` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
    /// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn distance_transform_abs_pba_to_f64(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
    ) -> Result<()>;

    /// Computes a PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f64` values.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
    /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
    /// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn distance_transform_abs_pba_to_f64_antialiasing(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_abs_pba_to_f64 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident, $antialiasing_name:ident) => {
        unsafe impl DistanceTransformAbsPbaToF64C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            unsafe fn distance_transform_abs_pba_to_f64(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
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
                        minimum_site_value,
                        maximum_site_value,
                        destination_voronoi,
                        destination_voronoi_indices,
                        destination_voronoi_manhattan,
                        destination_transform,
                    )
                }
            }

            unsafe fn distance_transform_abs_pba_to_f64_antialiasing(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
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
                        minimum_site_value,
                        maximum_site_value,
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

impl_distance_transform_abs_pba_to_f64!(
    u8,
    u16,
    distance_transform_abs_pba_u8_to_f64_c1,
    distance_transform_abs_pba_u8_to_f64_c1_antialiasing
);

impl_distance_transform_abs_pba_to_f64!(
    i8,
    u16,
    distance_transform_abs_pba_i8_to_f64_c1,
    distance_transform_abs_pba_i8_to_f64_c1_antialiasing
);

impl_distance_transform_abs_pba_to_f64!(
    u16,
    u16,
    distance_transform_abs_pba_u16_to_f64_c1,
    distance_transform_abs_pba_u16_to_f64_c1_antialiasing
);

impl_distance_transform_abs_pba_to_f64!(
    i16,
    u16,
    distance_transform_abs_pba_i16_to_f64_c1,
    distance_transform_abs_pba_i16_to_f64_c1_antialiasing
);

impl_distance_transform_abs_pba_to_f64!(
    f32,
    u16,
    distance_transform_abs_pba_f32_to_f64_c1,
    distance_transform_abs_pba_f32_to_f64_c1_antialiasing
);

impl_distance_transform_abs_pba_to_f64!(
    f64,
    u16,
    distance_transform_abs_pba_f64_c1,
    distance_transform_abs_pba_f64_c1_antialiasing
);

/// Computes a PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_abs_pba_to_f64<T: DistanceTransformAbsPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::distance_transform_abs_pba_to_f64(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Computes a PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_abs_pba_to_f64_antialiasing<T: DistanceTransformAbsPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::distance_transform_abs_pba_to_f64_antialiasing(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches PBA Euclidean distance transforms with relative Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait DistanceTransformPbaC1<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
    /// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn distance_transform_pba_c1(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, Destination, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_pba_c1 {
    ($destination:ty, $source_trait:ident, $source_method:ident) => {
        unsafe impl<T> DistanceTransformPbaC1<$destination> for T
        where
            T: $source_trait,
        {
            type Manhattan = <T as $source_trait>::Manhattan;

            unsafe fn distance_transform_pba_c1(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
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
                        minimum_site_value,
                        maximum_site_value,
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

impl_distance_transform_pba_c1!(
    u16,
    DistanceTransformPbaToU16C1,
    distance_transform_pba_to_u16
);

impl_distance_transform_pba_c1!(
    f32,
    DistanceTransformPbaToF32C1,
    distance_transform_pba_to_f32
);

impl_distance_transform_pba_c1!(
    f64,
    DistanceTransformPbaToF64C1,
    distance_transform_pba_to_f64
);

/// Computes a PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_pba<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, Destination, C1>>,
) -> Result<()>
where
    T: DistanceTransformPbaC1<Destination>,
    Destination: DataTypeLike,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::distance_transform_pba_c1(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Computes a PBA Euclidean distance transform with optional Voronoi and relative Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_pba_antialiasing<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()>
where
    T: DistanceTransformPbaToF64C1,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        distance_transform_pba_to_f64_antialiasing(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Dispatches PBA Euclidean distance transforms with absolute Manhattan output.
///
/// # Safety
///
/// Implementations must preserve the native element representations, Manhattan output type, site classification, and transform semantics.
/// Uphold the documented image execution and lifetime contracts, including exclusive native access to source storage and completion of owned temporary resources.
pub unsafe trait DistanceTransformAbsPbaC1<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    /// Component type for the optional two-component Manhattan output.
    type Manhattan: DataTypeLike;

    /// Computes a PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs.
    ///
    /// `None` disables an output; at least one output must be supplied.
    /// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
    /// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
    /// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn distance_transform_abs_pba_c1(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageOutputView<'_, Destination, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_abs_pba_c1 {
    ($destination:ty, $source_trait:ident, $source_method:ident) => {
        unsafe impl<T> DistanceTransformAbsPbaC1<$destination> for T
        where
            T: $source_trait,
        {
            type Manhattan = <T as $source_trait>::Manhattan;

            unsafe fn distance_transform_abs_pba_c1(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
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
                        minimum_site_value,
                        maximum_site_value,
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

impl_distance_transform_abs_pba_c1!(
    u16,
    DistanceTransformAbsPbaToU16C1,
    distance_transform_abs_pba_to_u16
);

impl_distance_transform_abs_pba_c1!(
    f32,
    DistanceTransformAbsPbaToF32C1,
    distance_transform_abs_pba_to_f32
);

impl_distance_transform_abs_pba_c1!(
    f64,
    DistanceTransformAbsPbaToF64C1,
    distance_transform_abs_pba_to_f64
);

/// Computes a PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_abs_pba<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, Destination, C1>>,
) -> Result<()>
where
    T: DistanceTransformAbsPbaC1<Destination>,
    Destination: DataTypeLike,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::distance_transform_abs_pba_c1(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

/// Computes a PBA Euclidean distance transform with optional Voronoi and absolute Manhattan outputs. The distance output uses `f64` values.
///
/// `None` disables an output; at least one output must be supplied.
/// The provided implementations own temporary device scratch, reject stream capture, and return after successful stream completion.
/// Allocates additional antialiasing workspace and enables antialiasing for the double-precision transform.
/// Source values in the inclusive range `minimum_site_value..=maximum_site_value` are sites.
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
/// The source must contain at least one site pixel; the provided implementations do not inspect pixel values or validate the site interval.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn distance_transform_abs_pba_antialiasing<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageOutputView<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageOutputView<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageOutputView<'_, f64, C1>>,
) -> Result<()>
where
    T: DistanceTransformAbsPbaToF64C1,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        distance_transform_abs_pba_to_f64_antialiasing(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}
