use std::{ffi, ptr};

use singe_cuda_sys::driver;

use crate::{
    dim::Dim3,
    error::{Error, Result},
    try_ffi,
    types::{ClusterSchedulingPolicy, SharedMemoryMode},
    utility::{to_u32, to_u64, to_usize},
};

/// Grid, block and dynamic shared-memory dimensions, independent of launch attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchGeometry {
    grid_dim: Dim3,
    block_dim: Dim3,
    shared_memory_bytes: usize,
}

/// Launch geometry and attributes used when submitting or constructing a kernel node.
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    geometry: LaunchGeometry,
    attributes: Vec<LaunchAttribute>,
}

impl LaunchGeometry {
    /// Creates nonzero grid and block dimensions with a representable shared-memory byte count.
    ///
    /// Device and kernel limits are checked during preparation or by CUDA during submission.
    ///
    /// # Errors
    ///
    /// Returns an error for zero dimensions or a shared-memory byte count exceeding `u32::MAX`.
    pub fn create(grid_dim: Dim3, block_dim: Dim3, shared_memory_bytes: usize) -> Result<Self> {
        validate_dim3(grid_dim, "grid_dim")?;
        validate_dim3(block_dim, "block_dim")?;
        validate_shared_memory_bytes(shared_memory_bytes)?;
        Ok(Self::from_validated(
            grid_dim,
            block_dim,
            shared_memory_bytes,
        ))
    }

    const fn from_validated(grid_dim: Dim3, block_dim: Dim3, shared_memory_bytes: usize) -> Self {
        Self {
            grid_dim,
            block_dim,
            shared_memory_bytes,
        }
    }

    /// Returns the number of blocks along each grid axis.
    #[must_use]
    pub const fn grid_dim(&self) -> Dim3 {
        self.grid_dim
    }

    /// Returns the number of threads along each block axis.
    #[must_use]
    pub const fn block_dim(&self) -> Dim3 {
        self.block_dim
    }

    /// Returns the dynamic shared-memory bytes per block.
    #[must_use]
    pub const fn shared_memory_bytes(&self) -> usize {
        self.shared_memory_bytes
    }

    pub(crate) fn shared_memory_bytes_u32(&self) -> u32 {
        u32::try_from(self.shared_memory_bytes)
            .expect("shared memory byte count is validated on construction")
    }

    /// Replaces the dynamic shared-memory byte count.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count exceeds `u32::MAX`.
    pub fn with_shared_memory_bytes(mut self, shared_memory_bytes: usize) -> Result<Self> {
        validate_shared_memory_bytes(shared_memory_bytes)?;
        self.shared_memory_bytes = shared_memory_bytes;
        Ok(self)
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    ///
    /// Each grid axis rounds up to a whole block.
    /// Dynamic shared memory starts at zero.
    /// Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotPositive` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_1d_grid(element_count: usize, block_size: usize) -> Result<Self> {
        validate_block_dimension(block_size, "block_size")?;
        let grid_size = element_count.div_ceil(block_size);

        validate_grid_dimension(grid_size, "grid_size")?;

        Ok(Self::from_validated(
            Dim3::new(to_u32(grid_size, "grid_size")?, 1, 1),
            Dim3::new(to_u32(block_size, "block_size")?, 1, 1),
            0,
        ))
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_1d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    /// Use [`Self::try_for_1d_grid`] to handle these errors.
    #[must_use]
    pub fn for_1d_grid(element_count: usize, block_size: usize) -> Self {
        Self::try_for_1d_grid(element_count, block_size)
            .expect("invalid 1d cuda launch configuration")
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    /// Equivalent to [`Self::try_for_1d_grid`].
    ///
    /// Each grid axis rounds up to a whole block.
    /// Dynamic shared memory starts at zero.
    /// Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotPositive` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_num_elems(element_count: usize, block_size: usize) -> Result<Self> {
        Self::try_for_1d_grid(element_count, block_size)
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    /// Equivalent to [`Self::try_for_1d_grid`].
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_num_elems`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    /// Use [`Self::try_for_num_elems`] to handle these errors.
    #[must_use]
    pub fn for_num_elems(element_count: usize, block_size: usize) -> Self {
        Self::try_for_num_elems(element_count, block_size)
            .expect("invalid cuda launch configuration")
    }

    /// Creates a two-dimensional launch covering `width` by `height` elements with the supplied block width and height.
    ///
    /// Each grid axis rounds up to a whole block.
    /// Dynamic shared memory starts at zero.
    /// Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotPositive` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_2d_grid(
        width: usize,
        height: usize,
        block_width: usize,
        block_height: usize,
    ) -> Result<Self> {
        validate_block_dimension(block_width, "block_width")?;
        validate_block_dimension(block_height, "block_height")?;
        let grid_x = width.div_ceil(block_width);
        let grid_y = height.div_ceil(block_height);
        validate_grid_dimension(grid_x, "grid_x")?;
        validate_grid_dimension(grid_y, "grid_y")?;

        Ok(Self::from_validated(
            Dim3::new(to_u32(grid_x, "grid_x")?, to_u32(grid_y, "grid_y")?, 1),
            Dim3::new(
                to_u32(block_width, "block_width")?,
                to_u32(block_height, "block_height")?,
                1,
            ),
            0,
        ))
    }

    /// Creates a two-dimensional launch covering `width` by `height` elements with the supplied block width and height.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_2d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    /// Use [`Self::try_for_2d_grid`] to handle these errors.
    #[must_use]
    pub fn for_2d_grid(
        width: usize,
        height: usize,
        block_width: usize,
        block_height: usize,
    ) -> Self {
        Self::try_for_2d_grid(width, height, block_width, block_height)
            .expect("invalid 2d cuda launch configuration")
    }

    /// Creates a three-dimensional launch covering `width` by `height` by `depth` elements with the supplied block dimensions.
    ///
    /// Each grid axis rounds up to a whole block.
    /// Dynamic shared memory starts at zero.
    /// Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotPositive` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_3d_grid(
        width: usize,
        height: usize,
        depth: usize,
        block_width: usize,
        block_height: usize,
        block_depth: usize,
    ) -> Result<Self> {
        validate_block_dimension(block_width, "block_width")?;
        validate_block_dimension(block_height, "block_height")?;
        validate_block_dimension(block_depth, "block_depth")?;
        let grid_x = width.div_ceil(block_width);
        let grid_y = height.div_ceil(block_height);
        let grid_z = depth.div_ceil(block_depth);
        validate_grid_dimension(grid_x, "grid_x")?;
        validate_grid_dimension(grid_y, "grid_y")?;
        validate_grid_dimension(grid_z, "grid_z")?;

        Ok(Self::from_validated(
            Dim3::new(
                to_u32(grid_x, "grid_x")?,
                to_u32(grid_y, "grid_y")?,
                to_u32(grid_z, "grid_z")?,
            ),
            Dim3::new(
                to_u32(block_width, "block_width")?,
                to_u32(block_height, "block_height")?,
                to_u32(block_depth, "block_depth")?,
            ),
            0,
        ))
    }

    /// Creates a three-dimensional launch covering `width` by `height` by `depth` elements with the supplied block dimensions.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_3d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    /// Use [`Self::try_for_3d_grid`] to handle these errors.
    #[must_use]
    pub fn for_3d_grid(
        width: usize,
        height: usize,
        depth: usize,
        block_width: usize,
        block_height: usize,
        block_depth: usize,
    ) -> Self {
        Self::try_for_3d_grid(width, height, depth, block_width, block_height, block_depth)
            .expect("invalid 3d cuda launch configuration")
    }
}

impl LaunchConfig {
    /// Creates a launch configuration with the supplied grid dimensions, block dimensions, and dynamic shared-memory bytes per block.
    ///
    /// No launch attributes are set.
    /// Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotPositive` if any grid or block axis is zero, or `Error::OutOfRange` if `shared_memory_bytes` exceeds `u32::MAX`.
    pub fn create(grid_dim: Dim3, block_dim: Dim3, shared_memory_bytes: usize) -> Result<Self> {
        Ok(LaunchGeometry::create(grid_dim, block_dim, shared_memory_bytes)?.into())
    }

    /// Returns the geometry used by this launch, without its attributes.
    #[must_use]
    pub const fn geometry(&self) -> &LaunchGeometry {
        &self.geometry
    }

    /// Returns the number of blocks along each grid axis.
    #[must_use]
    pub const fn grid_dim(&self) -> Dim3 {
        self.geometry.grid_dim()
    }

    /// Returns the number of threads along each block axis.
    #[must_use]
    pub const fn block_dim(&self) -> Dim3 {
        self.geometry.block_dim()
    }

    /// Returns the dynamic shared-memory bytes per block.
    #[must_use]
    pub const fn shared_memory_bytes(&self) -> usize {
        self.geometry.shared_memory_bytes()
    }

    pub(crate) fn shared_memory_bytes_u32(&self) -> u32 {
        self.geometry.shared_memory_bytes_u32()
    }

    /// Replaces the dynamic shared-memory byte count per block.
    ///
    /// # Errors
    ///
    /// Returns `Error::OutOfRange` if `bytes` exceeds `u32::MAX`.
    pub fn with_shared_memory_bytes(mut self, bytes: usize) -> Result<Self> {
        self.geometry = self.geometry.with_shared_memory_bytes(bytes)?;
        Ok(self)
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    ///
    /// Each grid axis rounds up to a whole block.
    /// Dynamic shared memory starts at zero.
    /// The configuration starts with no launch attributes.
    /// Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotPositive` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_1d_grid(element_count: usize, block_size: usize) -> Result<Self> {
        Ok(LaunchGeometry::try_for_1d_grid(element_count, block_size)?.into())
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_1d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    /// Use [`Self::try_for_1d_grid`] to handle these errors.
    #[must_use]
    pub fn for_1d_grid(element_count: usize, block_size: usize) -> Self {
        LaunchGeometry::for_1d_grid(element_count, block_size).into()
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    /// Equivalent to [`Self::try_for_1d_grid`].
    ///
    /// Each grid axis rounds up to a whole block.
    /// Dynamic shared memory starts at zero.
    /// The configuration starts with no launch attributes.
    /// Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotPositive` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_num_elems(element_count: usize, block_size: usize) -> Result<Self> {
        Ok(LaunchGeometry::try_for_num_elems(element_count, block_size)?.into())
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    /// Equivalent to [`Self::try_for_1d_grid`].
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_num_elems`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    /// Use [`Self::try_for_num_elems`] to handle these errors.
    #[must_use]
    pub fn for_num_elems(element_count: usize, block_size: usize) -> Self {
        LaunchGeometry::for_num_elems(element_count, block_size).into()
    }

    /// Creates a two-dimensional launch covering `width` by `height` elements with the supplied block width and height.
    ///
    /// Each grid axis rounds up to a whole block.
    /// Dynamic shared memory starts at zero.
    /// The configuration starts with no launch attributes.
    /// Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotPositive` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_2d_grid(
        width: usize,
        height: usize,
        block_width: usize,
        block_height: usize,
    ) -> Result<Self> {
        Ok(LaunchGeometry::try_for_2d_grid(width, height, block_width, block_height)?.into())
    }

    /// Creates a two-dimensional launch covering `width` by `height` elements with the supplied block width and height.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_2d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    /// Use [`Self::try_for_2d_grid`] to handle these errors.
    #[must_use]
    pub fn for_2d_grid(
        width: usize,
        height: usize,
        block_width: usize,
        block_height: usize,
    ) -> Self {
        LaunchGeometry::for_2d_grid(width, height, block_width, block_height).into()
    }

    /// Creates a three-dimensional launch covering `width` by `height` by `depth` elements with the supplied block dimensions.
    ///
    /// Each grid axis rounds up to a whole block.
    /// Dynamic shared memory starts at zero.
    /// The configuration starts with no launch attributes.
    /// Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotPositive` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_3d_grid(
        width: usize,
        height: usize,
        depth: usize,
        block_width: usize,
        block_height: usize,
        block_depth: usize,
    ) -> Result<Self> {
        Ok(LaunchGeometry::try_for_3d_grid(
            width,
            height,
            depth,
            block_width,
            block_height,
            block_depth,
        )?
        .into())
    }

    /// Creates a three-dimensional launch covering `width` by `height` by `depth` elements with the supplied block dimensions.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_3d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    /// Use [`Self::try_for_3d_grid`] to handle these errors.
    #[must_use]
    pub fn for_3d_grid(
        width: usize,
        height: usize,
        depth: usize,
        block_width: usize,
        block_height: usize,
        block_depth: usize,
    ) -> Self {
        LaunchGeometry::for_3d_grid(width, height, depth, block_width, block_height, block_depth)
            .into()
    }
}

impl From<LaunchGeometry> for LaunchConfig {
    fn from(geometry: LaunchGeometry) -> Self {
        Self {
            geometry,
            attributes: Vec::new(),
        }
    }
}

fn validate_dim3(value: Dim3, name: &str) -> Result<()> {
    validate_grid_dimension(to_usize(value.x, "grid dimension")?, &format!("{name}.x"))?;
    validate_grid_dimension(to_usize(value.y, "grid dimension")?, &format!("{name}.y"))?;
    validate_grid_dimension(to_usize(value.z, "grid dimension")?, &format!("{name}.z"))?;
    Ok(())
}

fn validate_grid_dimension(value: usize, name: &str) -> Result<()> {
    if value == 0 {
        return Err(Error::not_positive(name.to_owned()));
    }
    Ok(())
}

fn validate_block_dimension(value: usize, name: &str) -> Result<()> {
    if value == 0 {
        return Err(Error::not_positive(name.to_owned()));
    }
    Ok(())
}

fn validate_shared_memory_bytes(value: usize) -> Result<u32> {
    to_u32(value, "shared_memory_bytes")
}

pub(crate) fn validate_dynamic_shared_memory_bytes(value: usize) -> Result<u64> {
    to_u64(value, "dynamic_shared_memory_bytes")
}

/// Per-launch execution configuration.
#[derive(Debug, Clone, Copy)]
pub enum LaunchAttribute {
    /// Whether the kernel is launched as a cooperative kernel.
    Cooperative(bool),
    /// Dimensions of the cluster in blocks.
    ClusterDimension(Dim3),
    /// Cluster scheduling policy preference.
    ClusterSchedulingPolicy(ClusterSchedulingPolicy),
    /// Shared memory configuration for the kernel launch.
    SharedMemoryMode(SharedMemoryMode),
    /// Priority of the kernel launch.
    Priority(i32),
    /// Preferred shared memory carveout percentage.
    PreferredSharedMemoryCarveout(u32),
}

impl LaunchAttribute {
    pub(crate) fn raw(self) -> driver::CUlaunchAttribute {
        use driver::CUlaunchAttributeID as Id;
        let mut raw = driver::CUlaunchAttribute::default();
        match self {
            Self::Cooperative(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
                raw.value.cooperative = i32::from(value);
            }
            Self::ClusterDimension(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
                raw.value.clusterDim = driver::CUlaunchAttributeValue_union__bindgen_ty_1 {
                    x: value.x,
                    y: value.y,
                    z: value.z,
                };
            }
            Self::ClusterSchedulingPolicy(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
                raw.value.clusterSchedulingPolicyPreference = value.into();
            }
            Self::SharedMemoryMode(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE;
                raw.value.sharedMemoryMode = value.into();
            }
            Self::Priority(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_PRIORITY;
                raw.value.priority = value;
            }
            Self::PreferredSharedMemoryCarveout(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
                raw.value.sharedMemCarveout = value;
            }
        }
        raw
    }
}

impl LaunchConfig {
    /// Configures attributes for an extended kernel launch.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotPositive` if a cluster dimension is zero, or `Error::OutOfRange` if the preferred shared-memory carveout exceeds 100 percent.
    /// Device and kernel compatibility are checked later during preparation or submission.
    pub fn with_attribute(mut self, attribute: LaunchAttribute) -> Result<Self> {
        match attribute {
            LaunchAttribute::ClusterDimension(value) => validate_dim3(value, "cluster_dim")?,
            LaunchAttribute::PreferredSharedMemoryCarveout(value) if value > 100 => {
                return Err(Error::out_of_range("preferred shared memory carveout"));
            }
            _ => {}
        }

        let id = attribute.raw().id;
        self.attributes.retain(|existing| existing.raw().id != id);
        self.attributes.push(attribute);
        Ok(self)
    }

    /// Returns the configured launch attributes.
    #[must_use]
    pub fn attributes(&self) -> &[LaunchAttribute] {
        &self.attributes
    }

    pub(crate) fn with_raw_config<R>(
        &self,
        stream: driver::CUstream,
        f: impl FnOnce(&driver::CUlaunchConfig) -> Result<R>,
    ) -> Result<R> {
        let mut attributes = self.attributes.iter().map(|a| a.raw()).collect::<Vec<_>>();
        let config = driver::CUlaunchConfig {
            gridDimX: self.geometry.grid_dim.x,
            gridDimY: self.geometry.grid_dim.y,
            gridDimZ: self.geometry.grid_dim.z,
            blockDimX: self.geometry.block_dim.x,
            blockDimY: self.geometry.block_dim.y,
            blockDimZ: self.geometry.block_dim.z,
            sharedMemBytes: self.shared_memory_bytes_u32(),
            hStream: stream,
            attrs: attributes.as_mut_ptr(),
            numAttrs: to_u32(attributes.len(), "launch attributes")?,
        };
        f(&config)
    }

    pub(crate) unsafe fn launch_raw(
        &self,
        function: driver::CUfunction,
        stream: driver::CUstream,
        arguments: *mut *mut ffi::c_void,
    ) -> Result<()> {
        // SAFETY: `config` and its attribute array live through the closure, and the caller guarantees a live `function` and `stream` in the bound context and `arguments` matching the kernel's parameters.
        self.with_raw_config(stream, |config| unsafe {
            try_ffi!(driver::cuLaunchKernelEx(
                config,
                function,
                arguments,
                ptr::null_mut()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launch_config_rejects_zero_grid_dimensions() {
        let error = LaunchConfig::try_for_1d_grid(0, 128).unwrap_err();
        assert!(matches!(error, Error::NotPositive { name } if name == "grid_size"));

        let error = LaunchConfig::create(Dim3::new(0, 1, 1), Dim3::new(128, 1, 1), 0).unwrap_err();
        assert!(matches!(error, Error::NotPositive { name } if name == "grid_dim.x"));
    }

    #[test]
    fn test_launch_config_rejects_invalid_shared_memory_size() {
        let error = LaunchConfig::try_for_1d_grid(1, 128)
            .unwrap()
            .with_shared_memory_bytes(u32::MAX as usize + 1)
            .unwrap_err();
        assert!(matches!(error, Error::OutOfRange { name } if name == "shared_memory_bytes"));
    }

    #[test]
    fn test_launch_config_exposes_checked_shared_memory_u32() {
        let config = LaunchConfig::try_for_1d_grid(1, 128)
            .unwrap()
            .with_shared_memory_bytes(u32::MAX as usize)
            .unwrap();

        assert_eq!(config.shared_memory_bytes(), u32::MAX as usize);
        assert_eq!(config.shared_memory_bytes_u32(), u32::MAX);
    }

    #[test]
    fn test_occupancy_dynamic_shared_memory_uses_checked_driver_width() {
        assert_eq!(validate_dynamic_shared_memory_bytes(0).unwrap(), 0);
        assert_eq!(
            validate_dynamic_shared_memory_bytes(usize::MAX).unwrap(),
            usize::MAX as u64
        );
    }

    #[test]
    fn test_launch_config_uses_checked_construction() {
        let config = LaunchConfig::create(Dim3::new(1, 1, 1), Dim3::new(32, 1, 1), 0).unwrap();

        assert_eq!(config.grid_dim(), Dim3::new(1, 1, 1));
        assert_eq!(config.block_dim(), Dim3::new(32, 1, 1));
        assert_eq!(config.shared_memory_bytes(), 0);
    }
}
