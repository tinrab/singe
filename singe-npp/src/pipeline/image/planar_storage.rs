use crate::{
    image::{
        memory::Image,
        view::{C1, C2},
    },
    types::Size,
};

use super::{SemiplanarImage, SubsampledPlanarImage};

impl<T> SubsampledPlanarImage<T> {
    pub fn y(&self) -> &Image<T, C1> {
        &self.y
    }

    pub fn y_mut(&mut self) -> &mut Image<T, C1> {
        &mut self.y
    }

    pub fn cb(&self) -> &Image<T, C1> {
        &self.cb
    }

    pub fn cb_mut(&mut self) -> &mut Image<T, C1> {
        &mut self.cb
    }

    pub fn cr(&self) -> &Image<T, C1> {
        &self.cr
    }

    pub fn cr_mut(&mut self) -> &mut Image<T, C1> {
        &mut self.cr
    }

    pub fn into_planes(self) -> (Image<T, C1>, Image<T, C1>, Image<T, C1>) {
        (self.y, self.cb, self.cr)
    }
}

impl<T> SemiplanarImage<T> {
    pub fn y(&self) -> &Image<T, C1> {
        &self.y
    }

    pub fn y_mut(&mut self) -> &mut Image<T, C1> {
        &mut self.y
    }

    pub fn uv(&self) -> &Image<T, C2> {
        &self.uv
    }

    pub fn uv_mut(&mut self) -> &mut Image<T, C2> {
        &mut self.uv
    }

    pub fn into_planes(self) -> (Image<T, C1>, Image<T, C2>) {
        (self.y, self.uv)
    }

    pub fn size(&self) -> Size {
        self.y.size()
    }

    pub fn chroma_size(&self) -> Size {
        self.uv.size()
    }
}
