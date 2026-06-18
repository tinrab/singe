use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    error::{Error, Result},
    image::{
        color::{ColorTwistBatchConstantsMatrix4, ColorTwistMatrix},
        view::{
            C1, C2, C4, ChannelLayout, ImageView, ImageViewMut, PlanarImageView, image_descriptors,
            image_descriptors_mut,
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

pub(super) fn validate_cfa_roi(
    source: Size,
    source_roi: Rectangle,
    destination: Size,
) -> Result<()> {
    if source_roi.x < 0 || source_roi.y < 0 || source_roi.width <= 0 || source_roi.height <= 0 {
        return Err(Error::OutOfRange {
            name: "source roi".into(),
        });
    }

    let right = source_roi
        .x
        .checked_add(source_roi.width)
        .ok_or_else(|| Error::OutOfRange {
            name: "source roi".into(),
        })?;
    let bottom = source_roi
        .y
        .checked_add(source_roi.height)
        .ok_or_else(|| Error::OutOfRange {
            name: "source roi".into(),
        })?;

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

pub(super) fn validate_lookup_table(value_count: usize, level_count: usize) -> Result<i32> {
    if value_count != level_count {
        return Err(Error::LengthMismatch {
            name: "lookup table".into(),
            expected: value_count,
            actual: level_count,
        });
    }
    if level_count < 2 {
        return Err(Error::OutOfRange {
            name: "lookup table levels".into(),
        });
    }
    i32::try_from(level_count).map_err(|_| Error::OutOfRange {
        name: "lookup table levels".into(),
    })
}

pub(super) fn lookup_table_channel_descriptors<const C: usize>(
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

pub(super) fn lookup_table_channel_descriptors_f32<const C: usize>(
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

pub(super) fn validate_lookup_table_palette(
    table_length: usize,
    bit_size: i32,
    max_bit_size: i32,
) -> Result<i32> {
    validate_lookup_table_palette_entries(table_length, bit_size, max_bit_size, 1)
}

pub(super) fn validate_lookup_table_palette_entries(
    table_length: usize,
    bit_size: i32,
    max_bit_size: i32,
    values_per_index: usize,
) -> Result<i32> {
    if !(1..=max_bit_size).contains(&bit_size) {
        return Err(Error::OutOfRange {
            name: "lookup table palette bit size".into(),
        });
    }

    let required_length =
        1_usize
            .checked_shl(bit_size as u32)
            .ok_or_else(|| Error::OutOfRange {
                name: "lookup table palette bit size".into(),
            })?;
    let required_length = required_length
        .checked_mul(values_per_index)
        .ok_or_else(|| Error::OutOfRange {
            name: "lookup table palette".into(),
        })?;
    if table_length < required_length {
        return Err(Error::LengthMismatch {
            name: "lookup table palette".into(),
            expected: required_length,
            actual: table_length,
        });
    }

    Ok(bit_size)
}

pub(super) fn validate_lookup_table_palette_channels<T, const C: usize>(
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

pub(super) fn lookup_table_trilinear_descriptors(
    value_count: usize,
    levels: [&[u8]; 3],
) -> Result<([*mut u8; 3], [i32; 3])> {
    let mut level_counts = [0_i32; 3];
    let mut expected_values = 1_usize;
    for (index, level) in levels.iter().enumerate() {
        if level.len() < 2 {
            return Err(Error::OutOfRange {
                name: "lookup table trilinear levels".into(),
            });
        }
        level_counts[index] = i32::try_from(level.len()).map_err(|_| Error::OutOfRange {
            name: "lookup table trilinear levels".into(),
        })?;
        expected_values =
            expected_values
                .checked_mul(level.len())
                .ok_or_else(|| Error::OutOfRange {
                    name: "lookup table trilinear values".into(),
                })?;
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

pub(super) fn validate_equal_step(left: i32, right: i32, name: &str) -> Result<()> {
    if left == right {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: name.into(),
        expected: left as usize,
        actual: right as usize,
    })
}

pub(super) fn subsampled_size(
    size: Size,
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<Size> {
    if size.width % horizontal_subsampling != 0 {
        return Err(Error::OutOfRange {
            name: "image width".into(),
        });
    }
    if size.height % vertical_subsampling != 0 {
        return Err(Error::OutOfRange {
            name: "image height".into(),
        });
    }

    Ok(Size {
        width: size.width / horizontal_subsampling,
        height: size.height / vertical_subsampling,
    })
}

pub(super) fn validate_rgb_source_planes<T>(
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

pub(super) fn validate_rgb_destination_planes<T>(
    roi: Size,
    destination_0: &ImageViewMut<'_, T, C1>,
    destination_1: &ImageViewMut<'_, T, C1>,
    destination_2: &ImageViewMut<'_, T, C1>,
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

pub(super) fn validate_jpeg_subsampled_source_planes<T>(
    source_y: &ImageView<'_, T, C1>,
    source_cb: &ImageView<'_, T, C1>,
    source_cr: &ImageView<'_, T, C1>,
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<Size> {
    validate_same_size(source_cb.size(), source_cr.size())?;

    let expected_chroma = subsampled_size(
        source_y.size(),
        horizontal_subsampling,
        vertical_subsampling,
    )?;
    validate_same_size(expected_chroma, source_cb.size())?;
    Ok(source_y.size())
}

pub(super) fn validate_jpeg_subsampled_destination_planes<T>(
    roi: Size,
    destination_y: &ImageViewMut<'_, T, C1>,
    destination_cb: &ImageViewMut<'_, T, C1>,
    destination_cr: &ImageViewMut<'_, T, C1>,
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<()> {
    validate_same_size(roi, destination_y.size())?;
    validate_same_size(destination_cb.size(), destination_cr.size())?;

    let expected_chroma = subsampled_size(roi, horizontal_subsampling, vertical_subsampling)?;
    validate_same_size(expected_chroma, destination_cb.size())?;
    Ok(())
}

pub(super) fn validate_nv12_source_planes<T>(
    source_y: &ImageView<'_, T, C1>,
    source_uv: &ImageView<'_, T, C2>,
) -> Result<Size> {
    let expected_uv = subsampled_size(source_y.size(), 2, 2)?;
    validate_same_size(expected_uv, source_uv.size())?;
    validate_equal_step(source_y.step(), source_uv.step(), "nv12 source step")?;
    Ok(source_y.size())
}

pub(super) fn validate_subsampled_p2_source_planes<T>(
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

pub(super) fn validate_subsampled_p2_destination_planes<T>(
    roi: Size,
    destination_y: &ImageViewMut<'_, T, C1>,
    destination_cbcr: &ImageViewMut<'_, T, C2>,
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<()> {
    validate_same_size(roi, destination_y.size())?;
    let expected_cbcr = subsampled_size(roi, horizontal_subsampling, vertical_subsampling)?;
    validate_same_size(expected_cbcr, destination_cbcr.size())?;
    Ok(())
}

pub(super) fn validate_nv12_destination_planes<T>(
    roi: Size,
    destination_y: &ImageViewMut<'_, T, C1>,
    destination_uv: &ImageViewMut<'_, T, C2>,
) -> Result<()> {
    validate_same_size(roi, destination_y.size())?;
    let expected_uv = subsampled_size(roi, 2, 2)?;
    validate_same_size(expected_uv, destination_uv.size())?;
    Ok(())
}

pub(super) fn color_batch_descriptors<T, L>(
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
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
        DeviceMemory::from_slice(&source_descriptors)?,
        DeviceMemory::from_slice(&destination_descriptors)?,
    ))
}

pub(super) fn color_batch_advanced_descriptors<T, L>(
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
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

    let max_roi = largest_color_size(destinations.iter().map(ImageViewMut::size))?;
    let source_descriptors = image_descriptors(sources);
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        max_roi,
        DeviceMemory::from_slice(&source_descriptors)?,
        DeviceMemory::from_slice(&destination_descriptors)?,
    ))
}

pub(super) fn color_batch_size(source_len: usize, destination_len: usize) -> Result<i32> {
    batch_size_i32(source_len, destination_len)
}

pub(super) fn subsampled_color_batch_descriptors<T, L>(
    source_plane_0: &[ImageView<'_, T, C1>],
    source_plane_1: &[ImageView<'_, T, C1>],
    source_plane_2: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageViewMut<'_, T, L>],
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
        DeviceMemory::from_slice(
            &source_plane_0
                .iter()
                .map(ImageView::descriptor)
                .collect::<Vec<_>>(),
        )?,
        DeviceMemory::from_slice(
            &source_plane_1
                .iter()
                .map(ImageView::descriptor)
                .collect::<Vec<_>>(),
        )?,
        DeviceMemory::from_slice(
            &source_plane_2
                .iter()
                .map(ImageView::descriptor)
                .collect::<Vec<_>>(),
        )?,
    ];
    let source_batch_list = source_descriptor_memory
        .each_ref()
        .map(|plane| plane.as_ptr());
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        roi,
        source_batch_list,
        DeviceMemory::from_slice(&destination_descriptors)?,
        source_descriptor_memory,
    ))
}

pub(super) fn subsampled_color_batch_advanced_descriptors<T, L>(
    source_plane_0: &[ImageView<'_, T, C1>],
    source_plane_1: &[ImageView<'_, T, C1>],
    source_plane_2: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageViewMut<'_, T, L>],
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

    let max_roi = largest_color_size(destinations.iter().map(ImageViewMut::size))?;
    let source_descriptor_memory = [
        DeviceMemory::from_slice(
            &source_plane_0
                .iter()
                .map(ImageView::descriptor)
                .collect::<Vec<_>>(),
        )?,
        DeviceMemory::from_slice(
            &source_plane_1
                .iter()
                .map(ImageView::descriptor)
                .collect::<Vec<_>>(),
        )?,
        DeviceMemory::from_slice(
            &source_plane_2
                .iter()
                .map(ImageView::descriptor)
                .collect::<Vec<_>>(),
        )?,
    ];
    let source_batch_list = source_descriptor_memory
        .each_ref()
        .map(|plane| plane.as_ptr());
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        max_roi,
        source_batch_list,
        DeviceMemory::from_slice(&destination_descriptors)?,
        source_descriptor_memory,
    ))
}

pub(super) fn color_twist_batch_descriptors<T, L>(
    sources: &[ImageView<'_, T, L>],
    twists: &[ColorTwistMatrix],
    destinations: &mut [ImageViewMut<'_, T, L>],
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
    let twist_memory = DeviceMemory::from_slice(&twist_values)?;
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
            twist: unsafe { twist_base.add(index * 12) },
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        roi,
        DeviceMemory::from_slice(&batch_descriptors)?,
        twist_memory,
    ))
}

pub(super) fn color_twist_batch_in_place_descriptors<T, L>(
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
    let twist_memory = DeviceMemory::from_slice(&twist_values)?;
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
                twist: unsafe { twist_base.add(index * 12) },
            }
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        roi,
        DeviceMemory::from_slice(&batch_descriptors)?,
        twist_memory,
    ))
}

pub(super) fn color_twist_batch_with_constants_descriptors<T>(
    sources: &[ImageView<'_, T, C4>],
    twists: &[ColorTwistBatchConstantsMatrix4],
    destinations: &mut [ImageViewMut<'_, T, C4>],
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
    let twist_memory = DeviceMemory::from_slice(&twist_values)?;
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
            twist: unsafe { twist_base.add(index * 20) },
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        roi,
        DeviceMemory::from_slice(&batch_descriptors)?,
        twist_memory,
    ))
}

pub(super) fn color_twist_batch_with_constants_in_place_descriptors<T>(
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
    let twist_memory = DeviceMemory::from_slice(&twist_values)?;
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
                twist: unsafe { twist_base.add(index * 20) },
            }
        })
        .collect::<Vec<_>>();

    Ok((
        batch_size,
        roi,
        DeviceMemory::from_slice(&batch_descriptors)?,
        twist_memory,
    ))
}

pub(super) fn smallest_color_size(sizes: impl IntoIterator<Item = Size>) -> Result<Size> {
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

pub(super) fn largest_color_size(sizes: impl IntoIterator<Item = Size>) -> Result<Size> {
    sizes
        .into_iter()
        .reduce(|left, right| Size {
            width: left.width.max(right.width),
            height: left.height.max(right.height),
        })
        .ok_or_else(|| Error::OutOfRange {
            name: "batch size".into(),
        })
}

pub(super) fn color_planar_batch_descriptors<T, L, const C: usize>(
    sources: &[PlanarImageView<'_, T, C>],
    destinations: &mut [ImageViewMut<'_, T, L>],
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
        source_descriptor_memory.push(DeviceMemory::from_slice(&descriptors)?);
    }
    let source_descriptor_memory: [DeviceMemory<sys::NppiImageDescriptor>; C] =
        source_descriptor_memory
            .try_into()
            .map_err(|_| Error::OutOfRange {
                name: "plane count".into(),
            })?;
    let source_batch_list = source_descriptor_memory
        .each_ref()
        .map(|plane| plane.as_ptr());
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        roi,
        source_batch_list,
        DeviceMemory::from_slice(&destination_descriptors)?,
        source_descriptor_memory,
    ))
}

pub(super) fn color_planar_batch_advanced_descriptors<T, L, const C: usize>(
    sources: &[PlanarImageView<'_, T, C>],
    destinations: &mut [ImageViewMut<'_, T, L>],
) -> Result<ColorPlanarBatchDescriptors<C>>
where
    L: ChannelLayout,
{
    let batch_size = color_batch_size(sources.len(), destinations.len())?;
    for (source, destination) in sources.iter().zip(destinations.iter_mut()) {
        validate_same_size(source.planes()[0].size(), destination.size())?;
    }

    let max_roi = largest_color_size(destinations.iter().map(ImageViewMut::size))?;
    let mut source_descriptor_memory = Vec::with_capacity(C);
    for plane_index in 0..C {
        let descriptors = sources
            .iter()
            .map(|source| source.planes()[plane_index].descriptor())
            .collect::<Vec<_>>();
        source_descriptor_memory.push(DeviceMemory::from_slice(&descriptors)?);
    }
    let source_descriptor_memory: [DeviceMemory<sys::NppiImageDescriptor>; C] =
        source_descriptor_memory
            .try_into()
            .map_err(|_| Error::OutOfRange {
                name: "plane count".into(),
            })?;
    let source_batch_list = source_descriptor_memory
        .each_ref()
        .map(|plane| plane.as_ptr());
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        max_roi,
        source_batch_list,
        DeviceMemory::from_slice(&destination_descriptors)?,
        source_descriptor_memory,
    ))
}
