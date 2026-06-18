use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        statistics,
        view::{C1, ImageView},
    },
    types::{Point32f, ProfileData},
};

use super::ImagePipeline;

impl<'a> ImagePipeline<'a, f32, C1> {
    pub fn circular_radial_profile_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, C1>,
        center: Point32f,
        profile: &mut [ProfileData],
    ) -> Result<()> {
        statistics::circular_radial_profile(stream_context, source, center, profile)
    }

    pub fn elliptical_radial_profile_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, C1>,
        center: Point32f,
        astigmatism_ratio: f32,
        orientation_radians: f32,
        profile: &mut [ProfileData],
    ) -> Result<()> {
        statistics::elliptical_radial_profile(
            stream_context,
            source,
            center,
            astigmatism_ratio,
            orientation_radians,
            profile,
        )
    }

    pub fn circular_radial_profile(
        self,
        center: Point32f,
        samples: usize,
    ) -> Result<Vec<ProfileData>> {
        let mut profile = profile_buffer(samples)?;
        {
            let source = self.view()?;
            statistics::circular_radial_profile(
                self.stream_context,
                &source,
                center,
                &mut profile,
            )?;
        }
        Ok(profile)
    }

    pub fn elliptical_radial_profile(
        self,
        center: Point32f,
        astigmatism_ratio: f32,
        orientation_radians: f32,
        samples: usize,
    ) -> Result<Vec<ProfileData>> {
        let mut profile = profile_buffer(samples)?;
        {
            let source = self.view()?;
            statistics::elliptical_radial_profile(
                self.stream_context,
                &source,
                center,
                astigmatism_ratio,
                orientation_radians,
                &mut profile,
            )?;
        }
        Ok(profile)
    }
}

fn profile_buffer(samples: usize) -> Result<Vec<ProfileData>> {
    if samples == 0 {
        return Err(Error::LengthMismatch {
            name: "profile output length".into(),
            expected: 1,
            actual: 0,
        });
    }

    u16::try_from(samples).map_err(|_| Error::OutOfRange {
        name: "profile output length".into(),
    })?;

    Ok(vec![ProfileData::default(); samples])
}
