use crate::{
    error::{Error, Result},
    image::{
        memory::Image,
        view::{C1, C2, PlanarImageView, PlanarImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

#[derive(Debug)]
pub struct PlanarImage<T, const C: usize> {
    pub(super) planes: [Image<T, C1>; C],
}

#[derive(Debug)]
pub struct SubsampledPlanarImage<T> {
    pub(super) y: Image<T, C1>,
    pub(super) cb: Image<T, C1>,
    pub(super) cr: Image<T, C1>,
}

#[derive(Debug)]
pub struct SemiplanarImage<T> {
    pub(super) y: Image<T, C1>,
    pub(super) uv: Image<T, C2>,
}

#[path = "planar_p4_color.rs"]
mod p4_color;
#[path = "planar_color_methods.rs"]
mod planar_color_methods;
#[path = "planar_color_twist_methods.rs"]
mod planar_color_twist_methods;
#[path = "planar_geometry_helpers.rs"]
mod planar_geometry_helpers;
#[path = "planar_geometry_types.rs"]
mod planar_geometry_types;
#[path = "planar_nv12_color_methods.rs"]
mod planar_nv12_color_methods;
#[path = "planar_p4_storage.rs"]
mod planar_p4_storage;
#[path = "planar_storage.rs"]
mod storage;

pub(super) use planar_geometry_types::{
    PlanarRemap, PlanarResize, PlanarResizeSqrPixel, PlanarWarpAffine, PlanarWarpPerspective,
    PlanarWarpQuad,
};

impl<T, const C: usize> PlanarImage<T, C> {
    pub fn planes(&self) -> &[Image<T, C1>; C] {
        &self.planes
    }

    pub fn planes_mut(&mut self) -> &mut [Image<T, C1>; C] {
        &mut self.planes
    }

    pub fn into_planes(self) -> [Image<T, C1>; C] {
        self.planes
    }
}

impl<T> PlanarImage<T, 3>
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
            ],
        })
    }

    pub fn view(&self) -> Result<PlanarImageView<'_, T, 3>> {
        PlanarImageView::create([
            self.planes[0].view()?,
            self.planes[1].view()?,
            self.planes[2].view()?,
        ])
    }

    pub fn view_mut(&mut self) -> Result<PlanarImageViewMut<'_, T, 3>> {
        let [a, b, c] = &mut self.planes;
        PlanarImageViewMut::create([a.view_mut()?, b.view_mut()?, c.view_mut()?])
    }
}

pub trait CreatePlanarImage<T, const C: usize> {
    fn create(size: Size) -> Result<PlanarImage<T, C>>;
    fn view(&self) -> Result<PlanarImageView<'_, T, C>>;
    fn view_mut(&mut self) -> Result<PlanarImageViewMut<'_, T, C>>;
}

impl<T> CreatePlanarImage<T, 3> for PlanarImage<T, 3>
where
    T: Copy,
    Workspace: ImageAllocator<T, C1>,
{
    fn create(size: Size) -> Result<PlanarImage<T, 3>> {
        PlanarImage::<T, 3>::create(size)
    }

    fn view(&self) -> Result<PlanarImageView<'_, T, 3>> {
        PlanarImage::<T, 3>::view(self)
    }

    fn view_mut(&mut self) -> Result<PlanarImageViewMut<'_, T, 3>> {
        PlanarImage::<T, 3>::view_mut(self)
    }
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
