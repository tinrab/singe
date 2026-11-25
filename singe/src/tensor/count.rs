use std::sync::{Arc, RwLock};

use crate::{
    data_type::DataType,
    dispatch::{DispatchKey, DispatchKeySet, ExcludeGuard},
    error::{Error, Result},
    graph::capture,
    scalar::Scalar,
    shape::Shape,
    tensor::{Tensor, TensorInner},
    utility::to_i64,
};

#[derive(Debug, Clone)]
pub struct Count {
    axis: usize,
    rows: Tensor,
}

impl Count {
    pub(crate) fn create(axis: usize, rows: Tensor) -> Result<Self> {
        if rows.data_type() != DataType::I64 || rows.shape().rank() != 0 {
            return Err(Error::invalid_argument("a count is an I64 scalar"));
        }
        Ok(Self { axis, rows })
    }

    #[must_use]
    pub const fn axis(&self) -> usize {
        self.axis
    }

    #[must_use]
    pub const fn rows(&self) -> &Tensor {
        &self.rows
    }

    pub(crate) fn moved_to(&self, axis: usize) -> Self {
        Self {
            axis,
            rows: self.rows.clone(),
        }
    }

    pub(crate) fn is_same(&self, other: &Self) -> bool {
        self.axis == other.axis && self.rows.id() == other.rows.id()
    }
}

impl Tensor {
    #[must_use]
    pub fn count(&self) -> Option<&Count> {
        self.inner.count.as_ref()
    }

    pub(crate) fn with_count(&self, count: Count) -> Result<Self> {
        let rank = self.shape().dimensions().len();
        if count.axis >= rank {
            return Err(Error::dimension_out_of_bounds(format!(
                "count axis {} at rank {rank}",
                count.axis
            )));
        }

        if count.rows.device() != self.device() {
            return Err(Error::device_mismatch(
                "count",
                self.device(),
                count.rows.device(),
            ));
        }

        if self.requires_grad() {
            return Err(Error::invalid_argument(
                "a tensor that requires gradients cannot be counted",
            ));
        }

        Ok(Self {
            inner: Arc::new(TensorInner {
                id: self.inner.id,
                storage: Arc::clone(&self.inner.storage),
                data_type: self.inner.data_type,
                layout: self.inner.layout.clone(),
                autograd: RwLock::default(),
                count: Some(count),
            }),
        })
    }

    pub fn cut_to_count(&self) -> Result<Self> {
        let Some(count) = self.count() else {
            return Ok(self.clone());
        };
        let rows = {
            let _guard = ExcludeGuard::new(DispatchKeySet::frontend());
            count
                .rows
                .to_vec::<i64>()?
                .first()
                .copied()
                .ok_or_else(|| Error::missing_output("count"))?
        };
        let layout = self
            .layout()
            .slice(to_i64(count.axis, "count axis")?, 0, rows, 1)?;
        self.strided_view_unrecorded(layout)
    }

    pub(crate) fn real_positions(&self) -> Result<Option<Self>> {
        let Some(count) = self.count() else {
            return Ok(None);
        };
        let mut real =
            Self::range(self.device(), 0, self.dimension(count.axis)?)?.less(&count.rows)?;
        for _ in count.axis + 1..self.shape().dimensions().len() {
            real = real.unsqueeze(-1)?;
        }
        Ok(Some(real))
    }

    pub(crate) fn fill_padding(&self, value: impl Into<Scalar>) -> Result<Self> {
        let Some(real) = self.real_positions()? else {
            return Ok(self.clone());
        };
        let _guard = ExcludeGuard::new(DispatchKeySet::EMPTY.with(DispatchKey::Counted));
        let data_type = self.data_type();
        let value = value.into();
        let fill = capture::literal(self.device(), |device| {
            value.full(device, data_type, &Shape::scalar())
        })?;
        Self::where_select(&real, self, &fill)
    }
}
