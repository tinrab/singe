use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    error::{Error, Result},
    image::geometry::ResizeBatchAdvanced,
    image::view::{
        ChannelLayout, ImageView, ImageViewMut, image_descriptors, image_descriptors_mut,
    },
    types::{
        AffineCoefficients, MirrorBatchDescriptor, PerspectiveCoefficients, Rectangle,
        ResizeBatchDescriptor, ResizeBatchRoiAdvanced, Size, WarpAffineBatchDescriptor,
        WarpPerspectiveBatchDescriptor,
    },
    utility::batch_size_u32,
};

type ResizeBatchAdvancedDescriptors = (
    u32,
    i32,
    i32,
    DeviceMemory<sys::NppiImageDescriptor>,
    DeviceMemory<sys::NppiImageDescriptor>,
    DeviceMemory<ResizeBatchRoiAdvanced>,
);

pub(super) fn validate_same_size(source: Size, destination: Size) -> Result<()> {
    if source == destination {
        return Ok(());
    }
    Err(Error::SizeMismatch {
        name: "image size".into(),
        expected: source,
        actual: destination,
    })
}

pub(super) fn rect_size(rectangle: Rectangle) -> Size {
    Size {
        width: rectangle.width,
        height: rectangle.height,
    }
}

pub(super) fn resize_batch_descriptors<T, L>(
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
) -> Result<(u32, Size, Size, DeviceMemory<ResizeBatchDescriptor>)>
where
    L: ChannelLayout,
{
    let batch_size = batch_size(sources.len(), destinations.len())?;
    let smallest_source_size = smallest_size(sources.iter().map(ImageView::size))?;
    let smallest_destination_size = smallest_size(destinations.iter().map(ImageViewMut::size))?;

    let descriptors = sources
        .iter()
        .zip(destinations.iter_mut())
        .map(|(source, destination)| ResizeBatchDescriptor {
            source: source.as_ptr().cast(),
            source_step: source.step(),
            destination: destination.as_mut_ptr().cast(),
            destination_step: destination.step(),
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        smallest_source_size,
        smallest_destination_size,
        DeviceMemory::from_slice(&descriptors)?,
    ))
}

pub(super) fn mirror_batch_descriptors<T, L>(
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
) -> Result<(u32, Size, DeviceMemory<MirrorBatchDescriptor>)>
where
    L: ChannelLayout,
{
    let batch_size = batch_size(sources.len(), destinations.len())?;

    for (source, destination) in sources.iter().zip(destinations.iter()) {
        validate_same_size(source.size(), destination.size())?;
    }

    let roi = smallest_size(sources.iter().map(ImageView::size))?;
    let descriptors = sources
        .iter()
        .zip(destinations.iter_mut())
        .map(|(source, destination)| MirrorBatchDescriptor {
            source: source.as_ptr().cast(),
            source_step: source.step(),
            destination: destination.as_mut_ptr().cast(),
            destination_step: destination.step(),
        })
        .collect::<Vec<_>>();

    Ok((batch_size, roi, DeviceMemory::from_slice(&descriptors)?))
}

pub(super) fn mirror_batch_in_place_descriptors<T, L>(
    images: &mut [ImageViewMut<'_, T, L>],
) -> Result<(u32, Size, DeviceMemory<MirrorBatchDescriptor>)>
where
    L: ChannelLayout,
{
    if images.len() <= 1 {
        return Err(Error::OutOfRange {
            name: "batch size".into(),
        });
    }

    let batch_size = u32::try_from(images.len()).map_err(|_| Error::OutOfRange {
        name: "batch size".into(),
    })?;
    let roi = smallest_size(images.iter().map(ImageViewMut::size))?;
    let descriptors = images
        .iter_mut()
        .map(|image| MirrorBatchDescriptor {
            source: image.as_mut_ptr().cast(),
            source_step: image.step(),
            destination: image.as_mut_ptr().cast(),
            destination_step: image.step(),
        })
        .collect::<Vec<_>>();

    Ok((batch_size, roi, DeviceMemory::from_slice(&descriptors)?))
}

pub(super) fn resize_batch_advanced_descriptors<T, L>(
    rois: &[ResizeBatchAdvanced],
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
) -> Result<ResizeBatchAdvancedDescriptors>
where
    L: ChannelLayout,
{
    let batch_size = batch_size(sources.len(), destinations.len())?;
    if sources.len() != rois.len() {
        return Err(Error::LengthMismatch {
            name: "batch roi".into(),
            expected: sources.len(),
            actual: rois.len(),
        });
    }

    let max_destination_width = rois
        .iter()
        .map(|roi| roi.destination_roi.width)
        .max()
        .ok_or_else(|| Error::OutOfRange {
            name: "batch size".into(),
        })?;
    let max_destination_height = rois
        .iter()
        .map(|roi| roi.destination_roi.height)
        .max()
        .ok_or_else(|| Error::OutOfRange {
            name: "batch size".into(),
        })?;

    let source_descriptors = image_descriptors(sources);
    let destination_descriptors = image_descriptors_mut(destinations);
    let roi_descriptors = rois
        .iter()
        .map(|roi| ResizeBatchRoiAdvanced {
            source_roi: roi.source_roi,
            destination_roi: roi.destination_roi,
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        max_destination_width,
        max_destination_height,
        DeviceMemory::from_slice(&source_descriptors)?,
        DeviceMemory::from_slice(&destination_descriptors)?,
        DeviceMemory::from_slice(&roi_descriptors)?,
    ))
}

pub(super) fn warp_affine_batch_descriptors<T, L>(
    coefficients: &[AffineCoefficients],
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
) -> Result<(
    u32,
    Size,
    DeviceMemory<WarpAffineBatchDescriptor>,
    DeviceMemory<AffineCoefficients>,
)>
where
    L: ChannelLayout,
{
    let batch_size = batch_size(sources.len(), destinations.len())?;
    if sources.len() != coefficients.len() {
        return Err(Error::LengthMismatch {
            name: "batch coefficients".into(),
            expected: sources.len(),
            actual: coefficients.len(),
        });
    }
    let smallest_source_size = smallest_size(sources.iter().map(ImageView::size))?;
    let coefficient_memory = DeviceMemory::from_slice(coefficients)?;

    let descriptors = sources
        .iter()
        .zip(destinations.iter_mut())
        .enumerate()
        .map(|(index, (source, destination))| WarpAffineBatchDescriptor {
            source: source.as_ptr().cast(),
            source_step: source.step(),
            destination: destination.as_mut_ptr().cast(),
            destination_step: destination.step(),
            coefficients: unsafe { coefficient_memory.as_mut_ptr().add(index).cast() },
            transformed_coefficients: [[0.0; 3]; 2],
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        smallest_source_size,
        DeviceMemory::from_slice(&descriptors)?,
        coefficient_memory,
    ))
}

pub(super) fn warp_perspective_batch_descriptors<T, L>(
    coefficients: &[PerspectiveCoefficients],
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
) -> Result<(
    u32,
    Size,
    DeviceMemory<WarpPerspectiveBatchDescriptor>,
    DeviceMemory<PerspectiveCoefficients>,
)>
where
    L: ChannelLayout,
{
    let batch_size = batch_size(sources.len(), destinations.len())?;
    if sources.len() != coefficients.len() {
        return Err(Error::LengthMismatch {
            name: "batch coefficients".into(),
            expected: sources.len(),
            actual: coefficients.len(),
        });
    }
    let smallest_source_size = smallest_size(sources.iter().map(ImageView::size))?;
    let coefficient_memory = DeviceMemory::from_slice(coefficients)?;

    let descriptors = sources
        .iter()
        .zip(destinations.iter_mut())
        .enumerate()
        .map(
            |(index, (source, destination))| WarpPerspectiveBatchDescriptor {
                source: source.as_ptr().cast(),
                source_step: source.step(),
                destination: destination.as_mut_ptr().cast(),
                destination_step: destination.step(),
                coefficients: unsafe { coefficient_memory.as_mut_ptr().add(index).cast() },
                transformed_coefficients: [[0.0; 3]; 3],
            },
        )
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        smallest_source_size,
        DeviceMemory::from_slice(&descriptors)?,
        coefficient_memory,
    ))
}

fn batch_size(source_len: usize, destination_len: usize) -> Result<u32> {
    batch_size_u32(source_len, destination_len)
}

fn smallest_size(sizes: impl IntoIterator<Item = Size>) -> Result<Size> {
    sizes
        .into_iter()
        .reduce(|left, right| Size {
            width: left.width.min(right.width),
            height: left.height.min(right.height),
        })
        .ok_or_else(|| Error::OutOfRange {
            name: "batch size".into(),
        })
}
