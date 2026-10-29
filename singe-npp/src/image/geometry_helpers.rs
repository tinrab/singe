use std::sync::Arc;

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
use singe_npp_sys as sys;

use crate::{
    error::{Error, Result},
    image::{
        geometry::ResizeBatchAdvanced,
        view::{
            ChannelLayout, ImageOutputView, ImageView, ImageViewMut, image_descriptors,
            image_descriptors_mut,
        },
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

pub fn validate_same_size(source: Size, destination: Size) -> Result<()> {
    if source == destination {
        return Ok(());
    }
    Err(Error::SizeMismatch {
        name: "image size".into(),
        expected: source,
        actual: destination,
    })
}

pub fn rect_size(rectangle: Rectangle) -> Size {
    Size {
        width: rectangle.width,
        height: rectangle.height,
    }
}

pub fn resize_batch_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageOutputView<'_, T, L>],
) -> Result<(u32, Size, Size, DeviceMemory<ResizeBatchDescriptor>)>
where
    L: ChannelLayout,
{
    let batch_size = batch_size(sources.len(), destinations.len())?;
    let smallest_source_size = smallest_size(sources.iter().map(ImageView::size))?;
    let smallest_destination_size = smallest_size(destinations.iter().map(ImageOutputView::size))?;

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
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &descriptors) }?,
    ))
}

pub fn mirror_batch_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageOutputView<'_, T, L>],
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

    // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
    Ok((batch_size, roi, unsafe {
        DeviceMemory::from_slice_unchecked(allocation_context, &descriptors)
    }?))
}

pub fn mirror_batch_in_place_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    images: &mut [ImageViewMut<'_, T, L>],
) -> Result<(u32, Size, DeviceMemory<MirrorBatchDescriptor>)>
where
    L: ChannelLayout,
{
    if images.len() <= 1 {
        return Err(Error::out_of_range("batch size"));
    }

    let batch_size = u32::try_from(images.len()).map_err(|_| Error::out_of_range("batch size"))?;
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

    // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
    Ok((batch_size, roi, unsafe {
        DeviceMemory::from_slice_unchecked(allocation_context, &descriptors)
    }?))
}

pub fn resize_batch_advanced_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    rois: &[ResizeBatchAdvanced],
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageOutputView<'_, T, L>],
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

    for (roi, destination) in rois.iter().zip(destinations.iter()) {
        roi.source_roi.validate()?;
        roi.destination_roi.validate_within(destination.size())?;
    }

    let max_destination_width = rois
        .iter()
        .map(|roi| roi.destination_roi.width)
        .max()
        .ok_or_else(|| Error::out_of_range("batch size"))?;
    let max_destination_height = rois
        .iter()
        .map(|roi| roi.destination_roi.height)
        .max()
        .ok_or_else(|| Error::out_of_range("batch size"))?;

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
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &source_descriptors) }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(allocation_context, &destination_descriptors)
        }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &roi_descriptors) }?,
    ))
}

pub fn warp_affine_batch_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    coefficients: &[AffineCoefficients],
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageOutputView<'_, T, L>],
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
    let coefficient_memory = DeviceMemory::from_slice(allocation_context, coefficients)?;

    let descriptors = sources
        .iter()
        .zip(destinations.iter_mut())
        .enumerate()
        .map(|(index, (source, destination))| WarpAffineBatchDescriptor {
            source: source.as_ptr().cast(),
            source_step: source.step(),
            destination: destination.as_mut_ptr().cast(),
            destination_step: destination.step(),
            coefficients: coefficient_memory.as_mut_ptr().wrapping_add(index).cast(),
            transformed_coefficients: [[0.0; 3]; 2],
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        smallest_source_size,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &descriptors) }?,
        coefficient_memory,
    ))
}

pub fn warp_perspective_batch_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    coefficients: &[PerspectiveCoefficients],
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageOutputView<'_, T, L>],
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
    let coefficient_memory = DeviceMemory::from_slice(allocation_context, coefficients)?;

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
                coefficients: coefficient_memory.as_mut_ptr().wrapping_add(index).cast(),
                transformed_coefficients: [[0.0; 3]; 3],
            },
        )
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        smallest_source_size,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &descriptors) }?,
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
        .ok_or_else(|| Error::out_of_range("batch size"))
}
