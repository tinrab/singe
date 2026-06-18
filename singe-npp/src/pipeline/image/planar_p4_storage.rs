use crate::{
    error::Result,
    image::view::{C1, PlanarImageView, PlanarImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::{CreatePlanarImage, PlanarImage};

impl<T> PlanarImage<T, 4>
where
    T: Copy,
{
    pub fn create(size: Size) -> Result<Self>
    where
        Workspace: ImageAllocator<T, C1>,
    {
        Ok(Self {
            planes: [
                Workspace::create_image(size)?,
                Workspace::create_image(size)?,
                Workspace::create_image(size)?,
                Workspace::create_image(size)?,
            ],
        })
    }

    pub fn view(&self) -> Result<PlanarImageView<'_, T, 4>> {
        PlanarImageView::create([
            self.planes[0].view()?,
            self.planes[1].view()?,
            self.planes[2].view()?,
            self.planes[3].view()?,
        ])
    }

    pub fn view_mut(&mut self) -> Result<PlanarImageViewMut<'_, T, 4>> {
        let [a, b, c, d] = &mut self.planes;
        PlanarImageViewMut::create([a.view_mut()?, b.view_mut()?, c.view_mut()?, d.view_mut()?])
    }
}

impl<T> CreatePlanarImage<T, 4> for PlanarImage<T, 4>
where
    T: Copy,
    Workspace: ImageAllocator<T, C1>,
{
    fn create(size: Size) -> Result<PlanarImage<T, 4>> {
        PlanarImage::<T, 4>::create(size)
    }

    fn view(&self) -> Result<PlanarImageView<'_, T, 4>> {
        PlanarImage::<T, 4>::view(self)
    }

    fn view_mut(&mut self) -> Result<PlanarImageViewMut<'_, T, 4>> {
        PlanarImage::<T, 4>::view_mut(self)
    }
}
