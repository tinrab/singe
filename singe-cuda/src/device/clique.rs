use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda_sys::driver;

use super::{Device, Uuid};
use crate::{
    driver::entry_points::{
        cuDeviceGetCliqueCount, cuDeviceGetCliqueInfo, cuDeviceGetFabricClusterUuid,
    },
    error::Result,
    utility::to_usize,
};

singe_core::native_enum! {
    /// Fabric clique types.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum CliqueType: driver::CUcliqueType_enum {
        /// Unicast pointer clique.
        UnicastPointer = CU_CLIQUE_TYPE_UNICAST_POINTER,
        /// Multicast pointer clique.
        MulticastPointer = CU_CLIQUE_TYPE_MULTICAST_POINTER,
        /// Unicast logical endpoint clique.
        UnicastLogicalEndpoint = CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT,
        /// Multicast logical endpoint clique.
        MulticastLogicalEndpoint = CU_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT,
    }
}

/// Fabric clique information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CliqueInfo {
    /// Type of the fabric clique.
    pub kind: CliqueType,
    /// ID of the fabric clique.
    pub id: u32,
}

impl Device {
    /// Retrieves the fabric cluster UUID for the given device.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver entry point is unavailable or the native query fails, including unsupported fabric features or an invalid device.
    pub fn fabric_cluster_uuid(self) -> Result<Uuid> {
        let mut value = driver::CUuuid::default();
        // SAFETY: `value` is a local that the driver writes through an exclusive pointer.
        unsafe {
            cuDeviceGetFabricClusterUuid(&raw mut value, self.id())?;
        }
        Ok(value.into())
    }

    /// Returns the fabric clique information for the cliques that the device is part of.
    ///
    /// # Errors
    ///
    /// Returns an error if either required driver entry point is unavailable, the native count or clique-information query fails, or the reported count does not fit `usize`.
    /// No partial list is returned.
    pub fn cliques(self) -> Result<Vec<CliqueInfo>> {
        let mut count = 0;
        // SAFETY: `count` is a local that the driver writes through an exclusive pointer.
        unsafe {
            cuDeviceGetCliqueCount(&raw mut count, self.id())?;
        }

        let mut values = vec![driver::CUcliqueInfo::default(); to_usize(count, "clique count")?];
        // SAFETY: `values` holds `count` elements, the capacity the driver reads from `count` before writing at most that many.
        unsafe {
            cuDeviceGetCliqueInfo(values.as_mut_ptr(), &raw mut count, self.id())?;
        }

        values.truncate(to_usize(count, "clique count")?);
        values
            .into_iter()
            .map(|v| {
                Ok(CliqueInfo {
                    kind: v.type_.try_into()?,
                    id: v.id,
                })
            })
            .collect()
    }
}

impl_enum_display!(CliqueType, {
    CliqueType::UnicastPointer => "CU_CLIQUE_TYPE_UNICAST_POINTER",
    CliqueType::MulticastPointer => "CU_CLIQUE_TYPE_MULTICAST_POINTER",
    CliqueType::UnicastLogicalEndpoint => "CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT",
    CliqueType::MulticastLogicalEndpoint => "CU_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT",
});
