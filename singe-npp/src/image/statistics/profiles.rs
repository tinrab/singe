use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        statistics_validation::validate_profile_output,
        view::{C1, ImageView},
    },
    try_ffi,
    types::{DataTypeLike, Point32f, ProfileStatistics},
};

/// Fills a host slice with evenly distributed integer histogram level boundaries.
///
/// `lower_level` supplies the lowest boundary and `upper_level` the greatest boundary.
/// The provided implementation uses `i32`. NPP requires at least two level entries;
/// the wrapper forwards the count and bounds to NPP without further validation.
///
/// # Errors
///
/// Returns an error if the slice length cannot fit `i32` or NPP rejects the level count
/// or boundary parameters.
pub fn even_levels_i32(levels: &mut [i32], lower_level: i32, upper_level: i32) -> Result<()> {
    let level_count = i32::try_from(levels.len()).map_err(|_| Error::out_of_range("levels"))?;
    // SAFETY: `nppiEvenLevelsHost_32s` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiEvenLevelsHost_32s(
            levels.as_mut_ptr().cast(),
            level_count,
            lower_level,
            upper_level,
        ))?;
    }
    Ok(())
}

/// Dispatches generation of evenly distributed host histogram levels.
///
/// # Safety
///
/// Implementations must preserve the native element ABI and operation semantics, and
/// uphold the method's output representation, validation, and memory-access requirements.
pub unsafe trait EvenLevels: DataTypeLike {
    /// Fills a host slice with evenly distributed integer histogram level boundaries.
    ///
    /// `lower_level` supplies the lowest boundary and `upper_level` the greatest boundary.
    /// The provided implementation uses `i32`. NPP requires at least two level entries;
    /// the wrapper forwards the count and bounds to NPP without further validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice length cannot fit `i32` or NPP rejects the level count
    /// or boundary parameters.
    fn even_levels(levels: &mut [Self], lower_level: Self, upper_level: Self) -> Result<()>;
}

unsafe impl EvenLevels for i32 {
    fn even_levels(levels: &mut [Self], lower_level: Self, upper_level: Self) -> Result<()> {
        even_levels_i32(levels, lower_level, upper_level)
    }
}

/// Fills a host slice with evenly distributed integer histogram level boundaries.
///
/// `lower_level` supplies the lowest boundary and `upper_level` the greatest boundary.
/// The provided implementation uses `i32`. NPP requires at least two level entries;
/// the wrapper forwards the count and bounds to NPP without further validation.
///
/// # Errors
///
/// Returns an error if the slice length cannot fit `i32` or NPP rejects the level count
/// or boundary parameters.
pub fn even_levels<T: EvenLevels>(levels: &mut [T], lower_level: T, upper_level: T) -> Result<()> {
    T::even_levels(levels, lower_level, upper_level)
}

/// Computes circular radial pixel-count, mean-intensity, and standard-deviation statistics.
///
/// `profile.len()` selects the sampled radii from zero through that length minus one.
/// `center` is in source pixel coordinates and may lie outside the image.
/// Center coordinates and ellipse parameters, when present, are forwarded without numerical
/// validation. The wrapper queues native work without synchronizing the stream.
///
/// # Errors
///
/// Returns an error if `profile` is empty or its length exceeds `u16`, CUDA context binding
/// fails, the source belongs to another CUDA context, or NPP rejects the image or profile
/// parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). The source pixels must be
/// initialized and ready for the selected stream. The host address of `profile` must also
/// be valid for GPU writes in that CUDA context. Keep its storage exclusively writable and
/// disjoint from the read-only source until native work completes. Retain both allocations
/// and the stream context through queued completion, submission errors, and captured replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native
/// operation. Do not read the output before native completion.
pub unsafe fn circular_radial_profile_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, f32, C1>,
    center: Point32f,
    profile: &mut [ProfileStatistics],
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;

    let sample_count = validate_profile_output(profile)?;

    // SAFETY: `nppiCircularRadialProfile_32f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCircularRadialProfile_32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            center.into(),
            profile.as_mut_ptr().cast(),
            sample_count,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Computes elliptical radial pixel-count, mean-intensity, and standard-deviation statistics.
///
/// `profile.len()` selects the sampled radii from zero through that length minus one.
/// `center` is in source pixel coordinates and may lie outside the image.
/// `astigmatism_ratio` is the orientation-axis to orthogonal-axis ratio; `1.5` means
/// a 1.5-times stretch. `orientation_radians` is clockwise, with zero pointing vertically.
/// Center coordinates and ellipse parameters, when present, are forwarded without numerical
/// validation. The wrapper queues native work without synchronizing the stream.
///
/// # Errors
///
/// Returns an error if `profile` is empty or its length exceeds `u16`, CUDA context binding
/// fails, the source belongs to another CUDA context, or NPP rejects the image or profile
/// parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). The source pixels must be
/// initialized and ready for the selected stream. The host address of `profile` must also
/// be valid for GPU writes in that CUDA context. Keep its storage exclusively writable and
/// disjoint from the read-only source until native work completes. Retain both allocations
/// and the stream context through queued completion, submission errors, and captured replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native
/// operation. Do not read the output before native completion.
pub unsafe fn elliptical_radial_profile_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, f32, C1>,
    center: Point32f,
    astigmatism_ratio: f32,
    orientation_radians: f32,
    profile: &mut [ProfileStatistics],
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;

    let sample_count = validate_profile_output(profile)?;

    // SAFETY: `nppiEllipticalRadialProfile_32f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiEllipticalRadialProfile_32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            center.into(),
            astigmatism_ratio,
            orientation_radians,
            profile.as_mut_ptr().cast(),
            sample_count,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Dispatches circular radial image profiles for a native element type.
///
/// # Safety
///
/// Implementations must preserve the native element ABI and operation semantics, and
/// uphold the method's output representation, validation, and memory-access requirements.
pub unsafe trait CircularRadialProfile: DataTypeLike {
    /// Computes circular radial pixel-count, mean-intensity, and standard-deviation statistics.
    ///
    /// `profile.len()` selects the sampled radii from zero through that length minus one.
    /// `center` is in source pixel coordinates and may lie outside the image.
    /// Center coordinates and ellipse parameters, when present, are forwarded without numerical
    /// validation. The wrapper queues native work without synchronizing the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `profile` is empty or its length exceeds `u16`, CUDA context binding
    /// fails, the source belongs to another CUDA context, or NPP rejects the image or profile
    /// parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw). The source pixels must be
    /// initialized and ready for the selected stream. The host address of `profile` must also
    /// be valid for GPU writes in that CUDA context. Keep its storage exclusively writable and
    /// disjoint from the read-only source until native work completes. Retain both allocations
    /// and the stream context through queued completion, submission errors, and captured replay.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native
    /// operation. Do not read the output before native completion.
    unsafe fn circular_radial_profile(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        center: Point32f,
        profile: &mut [ProfileStatistics],
    ) -> Result<()>;
}

unsafe impl CircularRadialProfile for f32 {
    unsafe fn circular_radial_profile(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        center: Point32f,
        profile: &mut [ProfileStatistics],
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { circular_radial_profile_f32_c1(stream_context, source, center, profile) }
    }
}

/// Computes circular radial pixel-count, mean-intensity, and standard-deviation statistics.
///
/// `profile.len()` selects the sampled radii from zero through that length minus one.
/// `center` is in source pixel coordinates and may lie outside the image.
/// Center coordinates and ellipse parameters, when present, are forwarded without numerical
/// validation. The wrapper queues native work without synchronizing the stream.
///
/// # Errors
///
/// Returns an error if `profile` is empty or its length exceeds `u16`, CUDA context binding
/// fails, the source belongs to another CUDA context, or NPP rejects the image or profile
/// parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). The source pixels must be
/// initialized and ready for the selected stream. The host address of `profile` must also
/// be valid for GPU writes in that CUDA context. Keep its storage exclusively writable and
/// disjoint from the read-only source until native work completes. Retain both allocations
/// and the stream context through queued completion, submission errors, and captured replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native
/// operation. Do not read the output before native completion.
pub unsafe fn circular_radial_profile<T: CircularRadialProfile>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    center: Point32f,
    profile: &mut [ProfileStatistics],
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::circular_radial_profile(stream_context, source, center, profile) }
}

/// Dispatches elliptical radial image profiles for a native element type.
///
/// # Safety
///
/// Implementations must preserve the native element ABI and operation semantics, and
/// uphold the method's output representation, validation, and memory-access requirements.
pub unsafe trait EllipticalRadialProfile: DataTypeLike {
    /// Computes elliptical radial pixel-count, mean-intensity, and standard-deviation statistics.
    ///
    /// `profile.len()` selects the sampled radii from zero through that length minus one.
    /// `center` is in source pixel coordinates and may lie outside the image.
    /// `astigmatism_ratio` is the orientation-axis to orthogonal-axis ratio; `1.5` means
    /// a 1.5-times stretch. `orientation_radians` is clockwise, with zero pointing vertically.
    /// Center coordinates and ellipse parameters, when present, are forwarded without numerical
    /// validation. The wrapper queues native work without synchronizing the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `profile` is empty or its length exceeds `u16`, CUDA context binding
    /// fails, the source belongs to another CUDA context, or NPP rejects the image or profile
    /// parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw). The source pixels must be
    /// initialized and ready for the selected stream. The host address of `profile` must also
    /// be valid for GPU writes in that CUDA context. Keep its storage exclusively writable and
    /// disjoint from the read-only source until native work completes. Retain both allocations
    /// and the stream context through queued completion, submission errors, and captured replay.
    /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native
    /// operation. Do not read the output before native completion.
    unsafe fn elliptical_radial_profile(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        center: Point32f,
        astigmatism_ratio: f32,
        orientation_radians: f32,
        profile: &mut [ProfileStatistics],
    ) -> Result<()>;
}

unsafe impl EllipticalRadialProfile for f32 {
    unsafe fn elliptical_radial_profile(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        center: Point32f,
        astigmatism_ratio: f32,
        orientation_radians: f32,
        profile: &mut [ProfileStatistics],
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            elliptical_radial_profile_f32_c1(
                stream_context,
                source,
                center,
                astigmatism_ratio,
                orientation_radians,
                profile,
            )
        }
    }
}

/// Computes elliptical radial pixel-count, mean-intensity, and standard-deviation statistics.
///
/// `profile.len()` selects the sampled radii from zero through that length minus one.
/// `center` is in source pixel coordinates and may lie outside the image.
/// `astigmatism_ratio` is the orientation-axis to orthogonal-axis ratio; `1.5` means
/// a 1.5-times stretch. `orientation_radians` is clockwise, with zero pointing vertically.
/// Center coordinates and ellipse parameters, when present, are forwarded without numerical
/// validation. The wrapper queues native work without synchronizing the stream.
///
/// # Errors
///
/// Returns an error if `profile` is empty or its length exceeds `u16`, CUDA context binding
/// fails, the source belongs to another CUDA context, or NPP rejects the image or profile
/// parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). The source pixels must be
/// initialized and ready for the selected stream. The host address of `profile` must also
/// be valid for GPU writes in that CUDA context. Keep its storage exclusively writable and
/// disjoint from the read-only source until native work completes. Retain both allocations
/// and the stream context through queued completion, submission errors, and captured replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native
/// operation. Do not read the output before native completion.
pub unsafe fn elliptical_radial_profile<T: EllipticalRadialProfile>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    center: Point32f,
    astigmatism_ratio: f32,
    orientation_radians: f32,
    profile: &mut [ProfileStatistics],
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::elliptical_radial_profile(
            stream_context,
            source,
            center,
            astigmatism_ratio,
            orientation_radians,
            profile,
        )
    }
}
