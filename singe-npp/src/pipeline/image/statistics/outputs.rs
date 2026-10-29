use std::{marker::PhantomData, mem::MaybeUninit, sync::Arc};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, view::DeviceRepr};

use crate::{
    error::{Error, Result},
    image::view::{C1, ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    types::Size,
    utility::checked_len,
};

/// Owned device elements containing an image-statistic result.
#[derive(Debug)]
pub struct ImageStatistic<T> {
    values: DeviceMemory<T>,
}

impl<T> ImageStatistic<T> {
    pub(crate) const fn from_values(values: DeviceMemory<T>) -> Self {
        Self { values }
    }

    /// Returns the number of stored statistic elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the statistic allocation contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the device statistic elements.
    #[must_use]
    pub fn values(&self) -> &DeviceMemory<T> {
        &self.values
    }

    /// Borrows the device statistic elements for mutation.
    pub fn values_mut(&mut self) -> &mut DeviceMemory<T> {
        &mut self.values
    }

    /// Returns the owned device statistic elements.
    #[must_use]
    pub fn into_values(self) -> DeviceMemory<T> {
        self.values
    }
}

/// Histogram outputs grouped by included image channel.
#[derive(Debug)]
pub struct ImageHistograms {
    /// Per-channel histogram bin-count arrays.
    pub channels: Vec<ImageStatistic<i32>>,
}

impl ImageHistograms {
    /// Returns the number of included channel histograms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Returns whether there are no included channel histograms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }
}

/// Matching mean and standard-deviation outputs.
#[derive(Debug)]
pub struct ImageMeanStandardDeviation<T> {
    /// Mean values.
    pub mean: ImageStatistic<T>,
    /// Standard-deviation values.
    pub standard_deviation: ImageStatistic<T>,
}

/// Minimum and maximum outputs for an image statistic.
#[derive(Debug)]
pub struct ImageMinMax<T> {
    /// Minimum values.
    pub min: ImageStatistic<T>,
    /// Maximum values.
    pub max: ImageStatistic<T>,
}

/// Statistic values with their corresponding image coordinates.
#[derive(Debug)]
pub struct ImageIndexedStatistic<T> {
    /// Statistic values at the reported coordinates.
    pub value: ImageStatistic<T>,
    /// Horizontal image coordinates of the statistic values.
    pub index_x: ImageStatistic<i32>,
    /// Vertical image coordinates of the statistic values.
    pub index_y: ImageStatistic<i32>,
}

/// Minimum and maximum values with their image coordinates.
#[derive(Debug)]
pub struct ImageIndexedMinMax<T> {
    /// Minimum values and their coordinates.
    pub min: ImageIndexedStatistic<T>,
    /// Maximum values and their coordinates.
    pub max: ImageIndexedStatistic<T>,
}

/// Owned image samples in a dense device allocation with no row padding.
#[derive(Debug)]
pub struct ContiguousImage<T, L = C1> {
    values: DeviceMemory<T>,
    size: Size,
    _marker: PhantomData<L>,
}

impl<T, L> ContiguousImage<T, L>
where
    L: ChannelLayout,
{
    /// Adopts an initialized allocation containing exactly the logical image samples.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the element count overflows, or the allocation length differs from `width * height * L::CHANNELS`.
    pub fn from_buffer(values: DeviceMemory<T>, size: Size) -> Result<Self> {
        let expected = checked_len(size, L::CHANNELS)?;
        if values.len() != expected {
            return Err(Error::LengthMismatch {
                name: "contiguous image elements".into(),
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            values,
            size,
            _marker: PhantomData,
        })
    }

    /// Returns the allocation's actual CUDA context.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.values.context()
    }

    /// Allocates uninitialized logical image samples in a dense device allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the element count overflows, or the device allocation fails.
    pub fn create_uninit(
        context: &Arc<CudaContext>,
        size: Size,
    ) -> Result<ContiguousImage<MaybeUninit<T>, L>>
    where
        T: DeviceRepr,
    {
        let len = checked_len(size, L::CHANNELS)?;
        Ok(ContiguousImage {
            values: DeviceMemory::<T>::create_uninit(context, len)?,
            size,
            _marker: PhantomData,
        })
    }

    /// Returns the image width and height in pixels.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Returns the dense image-sample allocation.
    #[must_use]
    pub const fn values(&self) -> &DeviceMemory<T> {
        &self.values
    }

    /// Returns the owned dense image-sample allocation.
    #[must_use]
    pub fn into_values(self) -> DeviceMemory<T> {
        self.values
    }

    /// Borrows the dense logical pixels as an initialized image view.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical shape, byte stride, or allocation span cannot form a valid image view.
    pub fn view(&self) -> Result<ImageView<'_, T, L>>
    where
        T: DeviceRepr,
    {
        ImageView::from_buffer(&self.values, self.size)
    }

    /// Exclusively borrows the initialized logical pixels as an image view.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical shape, byte stride, or allocation span cannot form a valid image view.
    pub fn view_mut(&mut self) -> Result<ImageViewMut<'_, T, L>>
    where
        T: DeviceRepr,
    {
        ImageViewMut::from_buffer(&mut self.values, self.size)
    }
}

impl<T, L: ChannelLayout> ContiguousImage<MaybeUninit<T>, L> {
    /// Borrows uninitialized logical pixels as a write-only image view without exposing readable `T` values.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical shape, byte stride, or allocation span cannot form a valid image view.
    pub fn output_view(&mut self) -> Result<ImageOutputView<'_, T, L>>
    where
        T: DeviceRepr,
    {
        ImageOutputView::from_buffer(&mut self.values, self.size)
    }

    /// Marks every image element as initialized.
    ///
    /// # Safety
    ///
    /// Every element must contain a valid initialized `T`, with all producing
    /// work ordered before subsequent consumers.
    #[must_use]
    pub unsafe fn assume_init(self) -> ContiguousImage<T, L> {
        ContiguousImage {
            // SAFETY: the caller upholds this function's safety contract.
            values: unsafe { self.values.assume_init() },
            size: self.size,
            _marker: PhantomData,
        }
    }
}

/// Integral images of source samples and squared source samples.
#[derive(Debug)]
pub struct ImageSquaredIntegral<T, S> {
    /// Integral image of the source samples.
    pub integral: ContiguousImage<T, C1>,
    /// Integral image of the squared source samples.
    pub squared: ContiguousImage<S, C1>,
}
