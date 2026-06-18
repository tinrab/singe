use crate::{
    error::{Error, Result},
    mg::context::ContextRef,
    utility::to_i64,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    device_sizes: Vec<i64>,
    host_size: i64,
}

impl Workspace {
    pub fn create(device_sizes: Vec<u64>, host_size: u64) -> Result<Self> {
        Ok(Self {
            device_sizes: device_sizes
                .into_iter()
                .map(|value| to_i64(value, "device workspace size"))
                .collect::<Result<_>>()?,
            host_size: to_i64(host_size, "host workspace size")?,
        })
    }

    pub(crate) fn from_raw(device_sizes: Vec<i64>, host_size: i64) -> Result<Self> {
        if device_sizes.iter().any(|&size| size < 0) || host_size < 0 {
            return Err(Error::OutOfRange {
                name: "workspace size".into(),
            });
        }

        Ok(Self {
            device_sizes,
            host_size,
        })
    }

    pub fn device_sizes(&self) -> &[i64] {
        &self.device_sizes
    }

    pub fn host_size(&self) -> i64 {
        self.host_size
    }

    pub fn device_sizes_bytes(&self) -> Result<Vec<usize>> {
        self.device_sizes
            .iter()
            .copied()
            .map(|size| {
                usize::try_from(size).map_err(|_| Error::OutOfRange {
                    name: "device workspace size".into(),
                })
            })
            .collect()
    }

    pub fn host_size_bytes(&self) -> Result<usize> {
        usize::try_from(self.host_size).map_err(|_| Error::OutOfRange {
            name: "host workspace size".into(),
        })
    }

    pub(crate) fn validate_for_context(&self, context: &ContextRef) -> Result<()> {
        if self.device_sizes.len() != context.device_count() {
            return Err(Error::LengthMismatch {
                name: "device_workspace_size".into(),
                expected: context.device_count(),
                actual: self.device_sizes.len(),
            });
        }
        Ok(())
    }

    pub(crate) fn device_sizes_ptr(&self) -> *const i64 {
        self.device_sizes.as_ptr()
    }
}
