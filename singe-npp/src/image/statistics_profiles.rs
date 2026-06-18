use super::*;

pub fn even_levels_i32(levels: &mut [i32], lower_level: i32, upper_level: i32) -> Result<()> {
    let level_count = i32::try_from(levels.len()).map_err(|_| Error::OutOfRange {
        name: "levels".into(),
    })?;
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

pub trait EvenLevels: DataTypeLike {
    fn even_levels(levels: &mut [Self], lower_level: Self, upper_level: Self) -> Result<()>;
}

impl EvenLevels for i32 {
    fn even_levels(levels: &mut [Self], lower_level: Self, upper_level: Self) -> Result<()> {
        even_levels_i32(levels, lower_level, upper_level)
    }
}

pub fn even_levels<T: EvenLevels>(levels: &mut [T], lower_level: T, upper_level: T) -> Result<()> {
    T::even_levels(levels, lower_level, upper_level)
}

pub(crate) fn circular_radial_profile_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, f32, C1>,
    center: Point32f,
    profile: &mut [ProfileData],
) -> Result<()> {
    let sample_count = validate_profile_output(profile)?;

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

pub(crate) fn elliptical_radial_profile_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, f32, C1>,
    center: Point32f,
    astigmatism_ratio: f32,
    orientation_radians: f32,
    profile: &mut [ProfileData],
) -> Result<()> {
    let sample_count = validate_profile_output(profile)?;

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

pub trait CircularRadialProfile: DataTypeLike {
    fn circular_radial_profile(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        center: Point32f,
        profile: &mut [ProfileData],
    ) -> Result<()>;
}

impl CircularRadialProfile for f32 {
    fn circular_radial_profile(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        center: Point32f,
        profile: &mut [ProfileData],
    ) -> Result<()> {
        circular_radial_profile_f32_c1(stream_context, source, center, profile)
    }
}

pub fn circular_radial_profile<T: CircularRadialProfile>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    center: Point32f,
    profile: &mut [ProfileData],
) -> Result<()> {
    T::circular_radial_profile(stream_context, source, center, profile)
}

pub trait EllipticalRadialProfile: DataTypeLike {
    fn elliptical_radial_profile(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        center: Point32f,
        astigmatism_ratio: f32,
        orientation_radians: f32,
        profile: &mut [ProfileData],
    ) -> Result<()>;
}

impl EllipticalRadialProfile for f32 {
    fn elliptical_radial_profile(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        center: Point32f,
        astigmatism_ratio: f32,
        orientation_radians: f32,
        profile: &mut [ProfileData],
    ) -> Result<()> {
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

pub fn elliptical_radial_profile<T: EllipticalRadialProfile>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    center: Point32f,
    astigmatism_ratio: f32,
    orientation_radians: f32,
    profile: &mut [ProfileData],
) -> Result<()> {
    T::elliptical_radial_profile(
        stream_context,
        source,
        center,
        astigmatism_ratio,
        orientation_radians,
        profile,
    )
}
