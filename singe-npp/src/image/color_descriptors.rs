use std::sync::Arc;

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
use singe_npp_sys as sys;

use crate::{
    error::{Error, Result},
    image::{
        color::{ColorTwistBatchConstantsMatrix4, ColorTwistMatrix},
        view::{
            C1, C2, C4, ChannelLayout, ImageOutputView, ImageView, ImageViewMut, PlanarImageView,
            image_descriptors, image_descriptors_mut,
        },
    },
    types::{ColorTwistBatchDescriptor, Rectangle, Size},
    utility::batch_size_i32,
};

type SubsampledColorBatchDescriptors = (
    i32,
    Size,
    [*const sys::NppiImageDescriptor; 3],
    DeviceMemory<sys::NppiImageDescriptor>,
    [DeviceMemory<sys::NppiImageDescriptor>; 3],
);

type ColorPlanarBatchDescriptors<const C: usize> = (
    i32,
    Size,
    [*const sys::NppiImageDescriptor; C],
    DeviceMemory<sys::NppiImageDescriptor>,
    [DeviceMemory<sys::NppiImageDescriptor>; C],
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

pub fn validate_cfa_roi(source: Size, source_roi: Rectangle, destination: Size) -> Result<()> {
    if source_roi.x < 0 || source_roi.y < 0 || source_roi.width <= 0 || source_roi.height <= 0 {
        return Err(Error::out_of_range("source roi"));
    }

    let right = source_roi
        .x
        .checked_add(source_roi.width)
        .ok_or_else(|| Error::out_of_range("source roi"))?;
    let bottom = source_roi
        .y
        .checked_add(source_roi.height)
        .ok_or_else(|| Error::out_of_range("source roi"))?;

    if right > source.width || bottom > source.height {
        return Err(Error::SizeMismatch {
            name: "source roi".into(),
            expected: source,
            actual: source_roi.size(),
        });
    }

    validate_same_size(
        Size {
            width: source_roi.width,
            height: source_roi.height,
        },
        destination,
    )
}

pub fn validate_lookup_table(value_count: usize, level_count: usize) -> Result<i32> {
    if value_count != level_count {
        return Err(Error::LengthMismatch {
            name: "lookup table".into(),
            expected: value_count,
            actual: level_count,
        });
    }
    if level_count < 2 {
        return Err(Error::out_of_range("lookup table levels"));
    }
    i32::try_from(level_count).map_err(|_| Error::out_of_range("lookup table levels"))
}

pub fn lookup_table_channel_descriptors<const C: usize>(
    values: &[&DeviceMemory<i32>; C],
    levels: &[&DeviceMemory<i32>; C],
) -> Result<([*const i32; C], [*const i32; C], [i32; C])> {
    let value_pointers = values.map(DeviceMemory::as_ptr);
    let level_pointers = levels.map(DeviceMemory::as_ptr);
    let mut level_counts = [0_i32; C];
    for index in 0..C {
        level_counts[index] = validate_lookup_table(values[index].len(), levels[index].len())?;
    }
    Ok((value_pointers, level_pointers, level_counts))
}

pub fn lookup_table_channel_descriptors_f32<const C: usize>(
    values: &[&DeviceMemory<f32>; C],
    levels: &[&DeviceMemory<f32>; C],
) -> Result<([*const f32; C], [*const f32; C], [i32; C])> {
    let value_pointers = values.map(DeviceMemory::as_ptr);
    let level_pointers = levels.map(DeviceMemory::as_ptr);
    let mut level_counts = [0_i32; C];
    for index in 0..C {
        level_counts[index] = validate_lookup_table(values[index].len(), levels[index].len())?;
    }
    Ok((value_pointers, level_pointers, level_counts))
}

pub fn validate_lookup_table_palette(
    table_length: usize,
    bit_size: i32,
    max_bit_size: i32,
) -> Result<i32> {
    validate_lookup_table_palette_entries(table_length, bit_size, max_bit_size, 1)
}

pub fn validate_lookup_table_palette_entries(
    table_length: usize,
    bit_size: i32,
    max_bit_size: i32,
    values_per_index: usize,
) -> Result<i32> {
    if !(1..=max_bit_size).contains(&bit_size) {
        return Err(Error::out_of_range("lookup table palette bit size"));
    }

    let required_length = 1_usize
        .checked_shl(bit_size as u32)
        .ok_or_else(|| Error::out_of_range("lookup table palette bit size"))?;
    let required_length = required_length
        .checked_mul(values_per_index)
        .ok_or_else(|| Error::out_of_range("lookup table palette"))?;
    if table_length < required_length {
        return Err(Error::LengthMismatch {
            name: "lookup table palette".into(),
            expected: required_length,
            actual: table_length,
        });
    }

    Ok(bit_size)
}

pub fn validate_lookup_table_palette_channels<T, const C: usize>(
    tables: &[&DeviceMemory<T>; C],
    bit_size: i32,
    max_bit_size: i32,
) -> Result<i32> {
    let bit_size = validate_lookup_table_palette(tables[0].len(), bit_size, max_bit_size)?;
    for table in tables.iter().skip(1) {
        validate_lookup_table_palette(table.len(), bit_size, max_bit_size)?;
    }
    Ok(bit_size)
}

pub fn lookup_table_trilinear_descriptors(
    value_count: usize,
    levels: [&[u8]; 3],
) -> Result<([*mut u8; 3], [i32; 3])> {
    let mut level_counts = [0_i32; 3];
    let mut expected_values = 1_usize;
    for (index, level) in levels.iter().enumerate() {
        if level.len() < 2 {
            return Err(Error::out_of_range("lookup table trilinear levels"));
        }
        level_counts[index] = i32::try_from(level.len())
            .map_err(|_| Error::out_of_range("lookup table trilinear levels"))?;
        expected_values = expected_values
            .checked_mul(level.len())
            .ok_or_else(|| Error::out_of_range("lookup table trilinear values"))?;
    }

    if value_count < expected_values {
        return Err(Error::LengthMismatch {
            name: "lookup table trilinear values".into(),
            expected: expected_values,
            actual: value_count,
        });
    }

    let level_pointers = levels.map(|level| level.as_ptr().cast_mut());
    Ok((level_pointers, level_counts))
}

pub fn validate_equal_step(left: i32, right: i32, name: &str) -> Result<()> {
    if left == right {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: name.to_owned().into(),
        expected: left as usize,
        actual: right as usize,
    })
}

pub fn subsampled_size(
    size: Size,
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<Size> {
    if size.width % horizontal_subsampling != 0 {
        return Err(Error::out_of_range("image width"));
    }
    if size.height % vertical_subsampling != 0 {
        return Err(Error::out_of_range("image height"));
    }

    Ok(Size {
        width: size.width / horizontal_subsampling,
        height: size.height / vertical_subsampling,
    })
}

pub fn validate_rgb_source_planes<T>(
    source_0: &ImageView<'_, T, C1>,
    source_1: &ImageView<'_, T, C1>,
    source_2: &ImageView<'_, T, C1>,
) -> Result<Size> {
    validate_same_size(source_0.size(), source_1.size())?;
    validate_same_size(source_0.size(), source_2.size())?;
    validate_equal_step(source_0.step(), source_1.step(), "planar source step")?;
    validate_equal_step(source_0.step(), source_2.step(), "planar source step")?;
    Ok(source_0.size())
}

pub fn validate_rgb_destination_planes<T>(
    roi: Size,
    destination_0: &ImageOutputView<'_, T, C1>,
    destination_1: &ImageOutputView<'_, T, C1>,
    destination_2: &ImageOutputView<'_, T, C1>,
) -> Result<()> {
    validate_same_size(roi, destination_0.size())?;
    validate_same_size(roi, destination_1.size())?;
    validate_same_size(roi, destination_2.size())?;
    validate_equal_step(
        destination_0.step(),
        destination_1.step(),
        "planar destination step",
    )?;
    validate_equal_step(
        destination_0.step(),
        destination_2.step(),
        "planar destination step",
    )?;
    Ok(())
}

pub fn validate_jpeg_subsampled_source_planes<T>(
    source_y: &ImageView<'_, T, C1>,
    blue_chroma: &ImageView<'_, T, C1>,
    red_chroma: &ImageView<'_, T, C1>,
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<Size> {
    validate_same_size(blue_chroma.size(), red_chroma.size())?;

    let expected_chroma = subsampled_size(
        source_y.size(),
        horizontal_subsampling,
        vertical_subsampling,
    )?;
    validate_same_size(expected_chroma, blue_chroma.size())?;
    Ok(source_y.size())
}

pub fn validate_jpeg_subsampled_destination_planes<T>(
    roi: Size,
    destination_y: &ImageOutputView<'_, T, C1>,
    blue_chroma_output: &ImageOutputView<'_, T, C1>,
    red_chroma_output: &ImageOutputView<'_, T, C1>,
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<()> {
    validate_same_size(roi, destination_y.size())?;
    validate_same_size(blue_chroma_output.size(), red_chroma_output.size())?;

    let expected_chroma = subsampled_size(roi, horizontal_subsampling, vertical_subsampling)?;
    validate_same_size(expected_chroma, blue_chroma_output.size())?;
    Ok(())
}

pub fn validate_nv12_source_planes<T>(
    source_y: &ImageView<'_, T, C1>,
    source_uv: &ImageView<'_, T, C2>,
) -> Result<Size> {
    let expected_uv = subsampled_size(source_y.size(), 2, 2)?;
    validate_same_size(expected_uv, source_uv.size())?;
    validate_equal_step(source_y.step(), source_uv.step(), "nv12 source step")?;
    Ok(source_y.size())
}

pub fn validate_subsampled_p2_source_planes<T>(
    source_y: &ImageView<'_, T, C1>,
    source_cbcr: &ImageView<'_, T, C2>,
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<Size> {
    let expected_cbcr = subsampled_size(
        source_y.size(),
        horizontal_subsampling,
        vertical_subsampling,
    )?;
    validate_same_size(expected_cbcr, source_cbcr.size())?;
    Ok(source_y.size())
}

pub fn validate_subsampled_p2_destination_planes<T>(
    roi: Size,
    destination_y: &ImageOutputView<'_, T, C1>,
    blue_chroma_outputcr: &ImageOutputView<'_, T, C2>,
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<()> {
    validate_same_size(roi, destination_y.size())?;
    let expected_cbcr = subsampled_size(roi, horizontal_subsampling, vertical_subsampling)?;
    validate_same_size(expected_cbcr, blue_chroma_outputcr.size())?;
    Ok(())
}

pub fn validate_nv12_destination_planes<T>(
    roi: Size,
    destination_y: &ImageOutputView<'_, T, C1>,
    destination_uv: &ImageOutputView<'_, T, C2>,
) -> Result<()> {
    validate_same_size(roi, destination_y.size())?;
    let expected_uv = subsampled_size(roi, 2, 2)?;
    validate_same_size(expected_uv, destination_uv.size())?;
    Ok(())
}

pub fn color_batch_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageOutputView<'_, T, L>],
) -> Result<(
    i32,
    Size,
    DeviceMemory<sys::NppiImageDescriptor>,
    DeviceMemory<sys::NppiImageDescriptor>,
)>
where
    L: ChannelLayout,
{
    let batch_size = color_batch_size(sources.len(), destinations.len())?;
    for (source, destination) in sources.iter().zip(destinations.iter_mut()) {
        validate_same_size(source.size(), destination.size())?;
    }

    let roi = smallest_color_size(sources.iter().map(ImageView::size))?;
    let source_descriptors = image_descriptors(sources);
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        roi,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &source_descriptors) }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(allocation_context, &destination_descriptors)
        }?,
    ))
}

pub fn color_batch_advanced_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageOutputView<'_, T, L>],
) -> Result<(
    i32,
    Size,
    DeviceMemory<sys::NppiImageDescriptor>,
    DeviceMemory<sys::NppiImageDescriptor>,
)>
where
    L: ChannelLayout,
{
    let batch_size = color_batch_size(sources.len(), destinations.len())?;
    for (source, destination) in sources.iter().zip(destinations.iter_mut()) {
        validate_same_size(source.size(), destination.size())?;
    }

    let max_roi = largest_color_size(destinations.iter().map(ImageOutputView::size))?;
    let source_descriptors = image_descriptors(sources);
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        max_roi,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &source_descriptors) }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(allocation_context, &destination_descriptors)
        }?,
    ))
}

pub fn color_batch_size(source_len: usize, destination_len: usize) -> Result<i32> {
    batch_size_i32(source_len, destination_len)
}

pub fn subsampled_color_batch_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    source_plane_0: &[ImageView<'_, T, C1>],
    source_plane_1: &[ImageView<'_, T, C1>],
    source_plane_2: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageOutputView<'_, T, L>],
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<SubsampledColorBatchDescriptors>
where
    L: ChannelLayout,
{
    let batch_size = color_batch_size(source_plane_0.len(), destinations.len())?;
    if source_plane_1.len() != source_plane_0.len() {
        return Err(Error::LengthMismatch {
            name: "subsampled source plane 1 batch".into(),
            expected: source_plane_0.len(),
            actual: source_plane_1.len(),
        });
    }
    if source_plane_2.len() != source_plane_0.len() {
        return Err(Error::LengthMismatch {
            name: "subsampled source plane 2 batch".into(),
            expected: source_plane_0.len(),
            actual: source_plane_2.len(),
        });
    }

    for (((source_0, source_1), source_2), destination) in source_plane_0
        .iter()
        .zip(source_plane_1.iter())
        .zip(source_plane_2.iter())
        .zip(destinations.iter_mut())
    {
        let roi = validate_jpeg_subsampled_source_planes(
            source_0,
            source_1,
            source_2,
            horizontal_subsampling,
            vertical_subsampling,
        )?;
        validate_same_size(roi, destination.size())?;
    }

    let roi = smallest_color_size(source_plane_0.iter().map(ImageView::size))?;
    let source_descriptor_memory = [
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(
                allocation_context,
                &source_plane_0
                    .iter()
                    .map(ImageView::descriptor)
                    .collect::<Vec<_>>(),
            )
        }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(
                allocation_context,
                &source_plane_1
                    .iter()
                    .map(ImageView::descriptor)
                    .collect::<Vec<_>>(),
            )
        }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(
                allocation_context,
                &source_plane_2
                    .iter()
                    .map(ImageView::descriptor)
                    .collect::<Vec<_>>(),
            )
        }?,
    ];
    let source_batch_list = source_descriptor_memory
        .each_ref()
        .map(DeviceMemory::as_ptr);
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        roi,
        source_batch_list,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(allocation_context, &destination_descriptors)
        }?,
        source_descriptor_memory,
    ))
}

pub fn subsampled_color_batch_advanced_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    source_plane_0: &[ImageView<'_, T, C1>],
    source_plane_1: &[ImageView<'_, T, C1>],
    source_plane_2: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageOutputView<'_, T, L>],
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<SubsampledColorBatchDescriptors>
where
    L: ChannelLayout,
{
    let batch_size = color_batch_size(source_plane_0.len(), destinations.len())?;
    if source_plane_1.len() != source_plane_0.len() {
        return Err(Error::LengthMismatch {
            name: "subsampled source plane 1 batch".into(),
            expected: source_plane_0.len(),
            actual: source_plane_1.len(),
        });
    }
    if source_plane_2.len() != source_plane_0.len() {
        return Err(Error::LengthMismatch {
            name: "subsampled source plane 2 batch".into(),
            expected: source_plane_0.len(),
            actual: source_plane_2.len(),
        });
    }

    for (((source_0, source_1), source_2), destination) in source_plane_0
        .iter()
        .zip(source_plane_1.iter())
        .zip(source_plane_2.iter())
        .zip(destinations.iter_mut())
    {
        let roi = validate_jpeg_subsampled_source_planes(
            source_0,
            source_1,
            source_2,
            horizontal_subsampling,
            vertical_subsampling,
        )?;
        validate_same_size(roi, destination.size())?;
    }

    let max_roi = largest_color_size(destinations.iter().map(ImageOutputView::size))?;
    let source_descriptor_memory = [
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(
                allocation_context,
                &source_plane_0
                    .iter()
                    .map(ImageView::descriptor)
                    .collect::<Vec<_>>(),
            )
        }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(
                allocation_context,
                &source_plane_1
                    .iter()
                    .map(ImageView::descriptor)
                    .collect::<Vec<_>>(),
            )
        }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(
                allocation_context,
                &source_plane_2
                    .iter()
                    .map(ImageView::descriptor)
                    .collect::<Vec<_>>(),
            )
        }?,
    ];
    let source_batch_list = source_descriptor_memory
        .each_ref()
        .map(DeviceMemory::as_ptr);
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        max_roi,
        source_batch_list,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(allocation_context, &destination_descriptors)
        }?,
        source_descriptor_memory,
    ))
}

pub fn color_twist_batch_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    sources: &[ImageView<'_, T, L>],
    twists: &[ColorTwistMatrix],
    destinations: &mut [ImageOutputView<'_, T, L>],
) -> Result<(
    i32,
    Size,
    DeviceMemory<ColorTwistBatchDescriptor>,
    DeviceMemory<f32>,
)>
where
    L: ChannelLayout,
{
    let batch_size = color_batch_size(sources.len(), destinations.len())?;
    if twists.len() != sources.len() {
        return Err(Error::LengthMismatch {
            name: "twist batch".into(),
            expected: sources.len(),
            actual: twists.len(),
        });
    }

    for (source, destination) in sources.iter().zip(destinations.iter_mut()) {
        validate_same_size(source.size(), destination.size())?;
    }

    let roi = smallest_color_size(sources.iter().map(ImageView::size))?;
    let twist_values = twists
        .iter()
        .flat_map(|twist| twist.iter().flat_map(|row| row.iter().copied()))
        .collect::<Vec<_>>();
    let twist_memory = DeviceMemory::from_slice(allocation_context, &twist_values)?;
    let twist_base = twist_memory.as_ptr().cast_mut();

    let batch_descriptors = sources
        .iter()
        .zip(destinations.iter_mut())
        .enumerate()
        .map(|(index, (source, destination))| ColorTwistBatchDescriptor {
            source: source.as_ptr().cast(),
            source_step: source.step(),
            destination: destination.as_mut_ptr().cast(),
            destination_step: destination.step(),
            twist: twist_base.wrapping_add(index * 12),
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        roi,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &batch_descriptors) }?,
        twist_memory,
    ))
}

pub fn color_twist_batch_in_place_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    images: &mut [ImageViewMut<'_, T, L>],
    twists: &[ColorTwistMatrix],
) -> Result<(
    i32,
    Size,
    DeviceMemory<ColorTwistBatchDescriptor>,
    DeviceMemory<f32>,
)>
where
    L: ChannelLayout,
{
    let batch_size = color_batch_size(images.len(), images.len())?;
    if twists.len() != images.len() {
        return Err(Error::LengthMismatch {
            name: "twist batch".into(),
            expected: images.len(),
            actual: twists.len(),
        });
    }

    let roi = smallest_color_size(images.iter().map(ImageViewMut::size))?;
    let twist_values = twists
        .iter()
        .flat_map(|twist| twist.iter().flat_map(|row| row.iter().copied()))
        .collect::<Vec<_>>();
    let twist_memory = DeviceMemory::from_slice(allocation_context, &twist_values)?;
    let twist_base = twist_memory.as_ptr().cast_mut();

    let batch_descriptors = images
        .iter_mut()
        .enumerate()
        .map(|(index, image)| {
            let ptr = image.as_mut_ptr().cast();
            ColorTwistBatchDescriptor {
                source: ptr,
                source_step: image.step(),
                destination: ptr,
                destination_step: image.step(),
                twist: twist_base.wrapping_add(index * 12),
            }
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        roi,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &batch_descriptors) }?,
        twist_memory,
    ))
}

pub fn color_twist_batch_with_constants_descriptors<T>(
    allocation_context: &Arc<CudaContext>,
    sources: &[ImageView<'_, T, C4>],
    twists: &[ColorTwistBatchConstantsMatrix4],
    destinations: &mut [ImageOutputView<'_, T, C4>],
) -> Result<(
    i32,
    Size,
    DeviceMemory<ColorTwistBatchDescriptor>,
    DeviceMemory<f32>,
)> {
    let batch_size = color_batch_size(sources.len(), destinations.len())?;
    if twists.len() != sources.len() {
        return Err(Error::LengthMismatch {
            name: "twist batch".into(),
            expected: sources.len(),
            actual: twists.len(),
        });
    }

    for (source, destination) in sources.iter().zip(destinations.iter_mut()) {
        validate_same_size(source.size(), destination.size())?;
    }

    let roi = smallest_color_size(sources.iter().map(ImageView::size))?;
    let twist_values = twists
        .iter()
        .flat_map(|twist| twist.iter().flat_map(|row| row.iter().copied()))
        .collect::<Vec<_>>();
    let twist_memory = DeviceMemory::from_slice(allocation_context, &twist_values)?;
    let twist_base = twist_memory.as_ptr().cast_mut();

    let batch_descriptors = sources
        .iter()
        .zip(destinations.iter_mut())
        .enumerate()
        .map(|(index, (source, destination))| ColorTwistBatchDescriptor {
            source: source.as_ptr().cast(),
            source_step: source.step(),
            destination: destination.as_mut_ptr().cast(),
            destination_step: destination.step(),
            twist: twist_base.wrapping_add(index * 20),
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        roi,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &batch_descriptors) }?,
        twist_memory,
    ))
}

pub fn color_twist_batch_with_constants_in_place_descriptors<T>(
    allocation_context: &Arc<CudaContext>,
    images: &mut [ImageViewMut<'_, T, C4>],
    twists: &[ColorTwistBatchConstantsMatrix4],
) -> Result<(
    i32,
    Size,
    DeviceMemory<ColorTwistBatchDescriptor>,
    DeviceMemory<f32>,
)> {
    let batch_size = color_batch_size(images.len(), images.len())?;
    if twists.len() != images.len() {
        return Err(Error::LengthMismatch {
            name: "twist batch".into(),
            expected: images.len(),
            actual: twists.len(),
        });
    }

    let roi = smallest_color_size(images.iter().map(ImageViewMut::size))?;
    let twist_values = twists
        .iter()
        .flat_map(|twist| twist.iter().flat_map(|row| row.iter().copied()))
        .collect::<Vec<_>>();
    let twist_memory = DeviceMemory::from_slice(allocation_context, &twist_values)?;
    let twist_base = twist_memory.as_ptr().cast_mut();

    let batch_descriptors = images
        .iter_mut()
        .enumerate()
        .map(|(index, image)| {
            let ptr = image.as_mut_ptr().cast();
            ColorTwistBatchDescriptor {
                source: ptr,
                source_step: image.step(),
                destination: ptr,
                destination_step: image.step(),
                twist: twist_base.wrapping_add(index * 20),
            }
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        roi,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &batch_descriptors) }?,
        twist_memory,
    ))
}

pub fn smallest_color_size(sizes: impl IntoIterator<Item = Size>) -> Result<Size> {
    sizes
        .into_iter()
        .reduce(|left, right| Size {
            width: left.width.min(right.width),
            height: left.height.min(right.height),
        })
        .ok_or_else(|| Error::out_of_range("batch size"))
}

pub fn largest_color_size(sizes: impl IntoIterator<Item = Size>) -> Result<Size> {
    sizes
        .into_iter()
        .reduce(|left, right| Size {
            width: left.width.max(right.width),
            height: left.height.max(right.height),
        })
        .ok_or_else(|| Error::out_of_range("batch size"))
}

pub fn color_planar_batch_descriptors<T, L, const C: usize>(
    allocation_context: &Arc<CudaContext>,
    sources: &[PlanarImageView<'_, T, C>],
    destinations: &mut [ImageOutputView<'_, T, L>],
) -> Result<ColorPlanarBatchDescriptors<C>>
where
    L: ChannelLayout,
{
    let batch_size = color_batch_size(sources.len(), destinations.len())?;
    for (source, destination) in sources.iter().zip(destinations.iter_mut()) {
        validate_same_size(source.planes()[0].size(), destination.size())?;
    }

    let roi = smallest_color_size(sources.iter().map(|source| source.planes()[0].size()))?;
    let mut source_descriptor_memory = Vec::with_capacity(C);
    for plane_index in 0..C {
        let descriptors = sources
            .iter()
            .map(|source| source.planes()[plane_index].descriptor())
            .collect::<Vec<_>>();
        source_descriptor_memory
            // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
            .push(unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &descriptors) }?);
    }
    let source_descriptor_memory: [DeviceMemory<sys::NppiImageDescriptor>; C] =
        source_descriptor_memory
            .try_into()
            .map_err(|_| Error::out_of_range("plane count"))?;
    let source_batch_list = source_descriptor_memory
        .each_ref()
        .map(DeviceMemory::as_ptr);
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        roi,
        source_batch_list,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(allocation_context, &destination_descriptors)
        }?,
        source_descriptor_memory,
    ))
}

pub fn color_planar_batch_advanced_descriptors<T, L, const C: usize>(
    allocation_context: &Arc<CudaContext>,
    sources: &[PlanarImageView<'_, T, C>],
    destinations: &mut [ImageOutputView<'_, T, L>],
) -> Result<ColorPlanarBatchDescriptors<C>>
where
    L: ChannelLayout,
{
    let batch_size = color_batch_size(sources.len(), destinations.len())?;
    for (source, destination) in sources.iter().zip(destinations.iter_mut()) {
        validate_same_size(source.planes()[0].size(), destination.size())?;
    }

    let max_roi = largest_color_size(destinations.iter().map(ImageOutputView::size))?;
    let mut source_descriptor_memory = Vec::with_capacity(C);
    for plane_index in 0..C {
        let descriptors = sources
            .iter()
            .map(|source| source.planes()[plane_index].descriptor())
            .collect::<Vec<_>>();
        source_descriptor_memory
            // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
            .push(unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &descriptors) }?);
    }
    let source_descriptor_memory: [DeviceMemory<sys::NppiImageDescriptor>; C] =
        source_descriptor_memory
            .try_into()
            .map_err(|_| Error::out_of_range("plane count"))?;
    let source_batch_list = source_descriptor_memory
        .each_ref()
        .map(DeviceMemory::as_ptr);
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        max_roi,
        source_batch_list,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(allocation_context, &destination_descriptors)
        }?,
        source_descriptor_memory,
    ))
}
