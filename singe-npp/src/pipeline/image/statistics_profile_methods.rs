use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        statistics,
        view::{C1, ImageView},
    },
    pipeline::image::ImagePipeline,
    types::{Point32f, ProfileStatistics},
};

impl ImagePipeline<'_, f32, C1> {
    /// Computes circular radial pixel-count, mean-intensity, and standard-deviation statistics.
    /// Writes one host record per sampled radius into `profile`.
    /// Radii run from zero through the sample count minus one. `center` uses source pixel
    /// coordinates and may lie outside the image.
    /// Prepares the source, waits for producers on other streams, rejects capture, and completes
    /// native work before returning successfully.
    /// The native call receives the host address of `profile` directly; that address must be
    /// accessible for GPU writes in the selected CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if the sample count is zero or exceeds `u16`, capture checks, CUDA
    /// context binding or region validation, producer waits, native execution, or stream
    /// synchronization fail.
    pub fn circular_radial_profile_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, C1>,
        center: Point32f,
        profile: &mut [ProfileStatistics],
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            statistics::circular_radial_profile(stream_context, source, center, profile)
        })
    }

    /// Computes elliptical radial pixel-count, mean-intensity, and standard-deviation statistics.
    /// Writes one host record per sampled radius into `profile`.
    /// Radii run from zero through the sample count minus one. `center` uses source pixel
    /// coordinates and may lie outside the image.
    /// `astigmatism_ratio` compares the orientation and orthogonal axes; `orientation_radians`
    /// is clockwise from the vertical axis.
    /// Prepares the source, waits for producers on other streams, rejects capture, and completes
    /// native work before returning successfully.
    /// The native call receives the host address of `profile` directly; that address must be
    /// accessible for GPU writes in the selected CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if the sample count is zero or exceeds `u16`, capture checks, CUDA
    /// context binding or region validation, producer waits, native execution, or stream
    /// synchronization fail.
    pub fn elliptical_radial_profile_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, C1>,
        center: Point32f,
        astigmatism_ratio: f32,
        orientation_radians: f32,
        profile: &mut [ProfileStatistics],
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            statistics::elliptical_radial_profile(
                stream_context,
                source,
                center,
                astigmatism_ratio,
                orientation_radians,
                profile,
            )
        })
    }

    /// Computes circular radial pixel-count, mean-intensity, and standard-deviation statistics.
    /// Returns an owned host vector, one record per sampled radius.
    /// Radii run from zero through the sample count minus one. `center` uses source pixel
    /// coordinates and may lie outside the image.
    /// Prepares the source, waits for producers on other streams, rejects capture, and completes
    /// native work before returning successfully.
    /// The native call receives an ordinary host-vector address; GPU access to that host
    /// allocation must be supported by the selected CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if the sample count is zero or exceeds `u16`, capture checks, CUDA
    /// context binding or region validation, producer waits, native execution, or stream
    /// synchronization fail.
    /// Constructing the source view can also fail.
    pub fn circular_radial_profile(
        self,
        center: Point32f,
        samples: usize,
    ) -> Result<Vec<ProfileStatistics>> {
        self.ensure_eager()?;
        let mut profile = profile_buffer(samples)?;
        {
            let source = self.view()?;
            Self::circular_radial_profile_into(self.stream_context, &source, center, &mut profile)?;
        }
        Ok(profile)
    }

    /// Computes elliptical radial pixel-count, mean-intensity, and standard-deviation statistics.
    /// Returns an owned host vector, one record per sampled radius.
    /// Radii run from zero through the sample count minus one. `center` uses source pixel
    /// coordinates and may lie outside the image.
    /// `astigmatism_ratio` compares the orientation and orthogonal axes; `orientation_radians`
    /// is clockwise from the vertical axis.
    /// Prepares the source, waits for producers on other streams, rejects capture, and completes
    /// native work before returning successfully.
    /// The native call receives an ordinary host-vector address; GPU access to that host
    /// allocation must be supported by the selected CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if the sample count is zero or exceeds `u16`, capture checks, CUDA
    /// context binding or region validation, producer waits, native execution, or stream
    /// synchronization fail.
    /// Constructing the source view can also fail.
    pub fn elliptical_radial_profile(
        self,
        center: Point32f,
        astigmatism_ratio: f32,
        orientation_radians: f32,
        samples: usize,
    ) -> Result<Vec<ProfileStatistics>> {
        self.ensure_eager()?;
        let mut profile = profile_buffer(samples)?;
        {
            let source = self.view()?;
            Self::elliptical_radial_profile_into(
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

fn profile_buffer(samples: usize) -> Result<Vec<ProfileStatistics>> {
    if samples == 0 {
        return Err(Error::LengthMismatch {
            name: "profile output length".into(),
            expected: 1,
            actual: 0,
        });
    }

    u16::try_from(samples).map_err(|_| Error::out_of_range("profile output length"))?;

    Ok(vec![ProfileStatistics::default(); samples])
}
