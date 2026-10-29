//! Host creation of device communicators, teams, and kernel resources.
//! Kernels include `nccl_device.h` and receive the communicator by value.

use std::{
    env,
    ffi::c_void,
    path::{Path, PathBuf},
    process::Command,
    ptr,
};

use singe_cuda::{architecture::GpuArchitecture, error::Error as CudaError};

use crate::{
    communicator::{Communicator, Window},
    error::{Error, Result},
    sys, try_ffi,
};

/// GIN connection topology for a device communicator.
#[derive(Debug, Clone, Copy, Default)]
pub enum GinConnection {
    /// Corresponds to [`sys::ncclGinConnectionType_t::NCCL_GIN_CONNECTION_NONE`].
    #[default]
    None,
    /// Corresponds to [`sys::ncclGinConnectionType_t::NCCL_GIN_CONNECTION_FULL`].
    Full,
    /// Corresponds to [`sys::ncclGinConnectionType_t::NCCL_GIN_CONNECTION_RAIL`].
    Rail,
    /// Stride of ranks to connect for GIN.
    CustomStride(i32),
}

/// Device communication backend.
#[derive(Debug, Clone, Copy, Default)]
pub enum GinType {
    /// Sentinel: accept any available backend.
    #[default]
    Automatic,
    /// Intentionally not 1. Must match `NCCL_NET_DEVICE_GIN_PROXY` for backward compatibility.
    Proxy,
    /// Intentionally not 2. Must match `NCCL_NET_DEVICE_GIN_GDAKI` for backward compatibility.
    Gdaki,
    /// Must match `NCCL_NET_DEVICE_GIN_GPI`.
    Gpi,
    /// Must match `NCCL_NET_DEVICE_GIN_EFA_GDA` for backward compatibility.
    EfaGda,
}

impl GinType {
    fn into_raw(self) -> sys::ncclGinType_t {
        use sys::ncclGinType_t::{
            NCCL_GIN_TYPE_EFA_GDA, NCCL_GIN_TYPE_GDAKI, NCCL_GIN_TYPE_GPI, NCCL_GIN_TYPE_NONE,
            NCCL_GIN_TYPE_PROXY,
        };
        match self {
            Self::Automatic => NCCL_GIN_TYPE_NONE,
            Self::Proxy => NCCL_GIN_TYPE_PROXY,
            Self::Gdaki => NCCL_GIN_TYPE_GDAKI,
            Self::Gpi => NCCL_GIN_TYPE_GPI,
            Self::EfaGda => NCCL_GIN_TYPE_EFA_GDA,
        }
    }
}

bitflags::bitflags! {
    /// Bitmask of device CFT capabilities.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct CftCapabilities: i32 {
        /// Corresponds to [`sys::ncclCftCap_t::NCCL_CFT`].
        const UNICAST = sys::ncclCftCap_t::NCCL_CFT as i32;
        /// Corresponds to [`sys::ncclCftCap_t::NCCL_CFT_MULTIMEM`].
        const MULTIMEM = sys::ncclCftCap_t::NCCL_CFT_MULTIMEM as i32;
    }
}

/// Attributes that users are able to customize for a device communicator.
#[derive(Debug, Clone)]
pub struct DeviceRequirements {
    /// Enable multimem on LSA team.
    pub lsa_multimem: bool,
    /// Number of hybrid network/rail barriers to allocate.
    pub barriers: i32,
    /// Number of memory barriers to allocate for the load/store-accessible team.
    pub lsa_barriers: i32,
    /// Number of railed network barriers to allocate.
    pub rail_gin_barriers: i32,
    /// Number of world-team network barriers to allocate.
    pub world_gin_barriers: i32,
    /// Number of blocks requiring local low-latency all-to-all resources.
    pub lsa_lla2a_blocks: i32,
    /// Number of low-latency all-to-all slots requested per block.
    pub lsa_lla2a_slots: i32,
    /// This is a hint, the actual context count in the device communicator may not match.
    pub gin_contexts: i32,
    /// Guaranteed to start at id=0.
    pub gin_signals: i32,
    /// Guaranteed to start at id=0.
    pub gin_counters: i32,
    /// GIN connection topology; requesting GIN resources requires a compatible connection type.
    pub gin_connection: GinConnection,
    /// Whether to request GIN contexts for exclusive use.
    pub gin_exclusive_contexts: bool,
    /// Requested depth of the GIN operation queue.
    pub gin_queue_depth: i32,
    /// Optional network traffic class for GIN operations.
    pub gin_traffic_class: Option<i32>,
    /// Requested device communication backend.
    pub gin_type: GinType,
    /// Compute Fabric Transport capabilities required by device kernels.
    pub cft_capabilities: CftCapabilities,
    /// Number of Compute Fabric Transport barriers to allocate.
    pub cft_barriers: i32,
}

impl Default for DeviceRequirements {
    fn default() -> Self {
        Self {
            lsa_multimem: false,
            barriers: 0,
            lsa_barriers: 0,
            rail_gin_barriers: 0,
            world_gin_barriers: 0,
            lsa_lla2a_blocks: 0,
            lsa_lla2a_slots: 0,
            gin_contexts: 4,
            gin_signals: 0,
            gin_counters: 0,
            gin_connection: GinConnection::None,
            gin_exclusive_contexts: false,
            gin_queue_depth: 0,
            gin_traffic_class: None,
            gin_type: GinType::Automatic,
            cft_capabilities: CftCapabilities::empty(),
            cft_barriers: 0,
        }
    }
}

/// A team of ranks within a communicator.
#[derive(Debug, Clone, Copy)]
pub struct Team<'a> {
    communicator: &'a Communicator,
    raw: sys::ncclTeam_t,
}

/// CFT team hierarchy.
#[derive(Debug, Clone, Copy)]
pub enum CftTeamMode {
    /// Corresponds to [`sys::ncclCftTeamMode_t::NCCL_CFT_TEAM_FLAT`].
    Flat,
    /// Corresponds to [`sys::ncclCftTeamMode_t::NCCL_CFT_TEAM_HIER_MULTIMEM`].
    HierarchicalMultimem,
    /// Corresponds to [`sys::ncclCftTeamMode_t::NCCL_CFT_TEAM_HIER_LSA`].
    HierarchicalLsa,
}

impl Team<'_> {
    /// Returns the number of ranks in this team.
    #[must_use]
    pub fn rank_count(&self) -> i32 {
        self.raw.nRanks
    }

    /// Returns the calling rank's index within the team.
    #[must_use]
    pub fn rank(&self) -> i32 {
        self.raw.rank
    }

    /// Returns the team's rank stride.
    #[must_use]
    pub fn stride(&self) -> i32 {
        self.raw.stride
    }

    /// Returns the native team description.
    #[must_use]
    pub fn as_raw(&self) -> sys::ncclTeam_t {
        self.raw
    }

    /// Maps a team rank to its world-communicator rank.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Returns an error if `rank` is outside this team.
    pub fn rank_to_world(&self, rank: i32) -> Result<i32> {
        self.validate_rank(rank)?;
        // SAFETY: `ncclTeamRankToWorld` receives live handles owned by the wrappers.
        Ok(unsafe { sys::ncclTeamRankToWorld(self.communicator.as_raw(), self.raw, rank) })
    }

    /// Maps a team rank to its load/store-accessible team rank.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Returns an error if `rank` is outside this team.
    pub fn rank_to_lsa(&self, rank: i32) -> Result<i32> {
        self.validate_rank(rank)?;
        // SAFETY: `ncclTeamRankToLsa` receives live handles owned by the wrappers.
        Ok(unsafe { sys::ncclTeamRankToLsa(self.communicator.as_raw(), self.raw, rank) })
    }

    fn validate_rank(&self, rank: i32) -> Result<()> {
        self.communicator.bind()?;
        if !(0..self.rank_count()).contains(&rank) {
            return Err(Error::out_of_range("team rank"));
        }
        Ok(())
    }
}

/// Device communicator API properties reported by NCCL.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    raw: sys::ncclCommProperties_t,
}

impl DeviceProperties {
    /// Returns the native device-property structure.
    #[must_use]
    pub fn as_raw(&self) -> &sys::ncclCommProperties_t {
        &self.raw
    }

    /// Returns whether NCCL device APIs are supported.
    #[must_use]
    pub fn device_api_supported(&self) -> bool {
        self.raw.deviceApiSupport
    }

    /// Returns whether multimem operations are supported.
    #[must_use]
    pub fn multimem_supported(&self) -> bool {
        self.raw.multimemSupport
    }

    /// Returns whether host-initiated remote-memory operations are supported.
    #[must_use]
    pub fn host_rma_supported(&self) -> bool {
        self.raw.hostRmaSupport
    }

    /// Returns whether Compute Fabric Transport is supported.
    #[must_use]
    pub fn cft_supported(&self) -> bool {
        self.raw.cftSupport
    }

    /// Returns whether Compute Fabric Transport multicast is supported.
    #[must_use]
    pub fn cft_multicast_supported(&self) -> bool {
        self.raw.cftMulticastSupport
    }

    /// Returns whether counted Compute Fabric Transport operations are supported.
    #[must_use]
    pub fn cft_counted_supported(&self) -> bool {
        self.raw.cftCountedSupport
    }

    /// True if the GIN type is supported.
    #[must_use]
    pub fn gin_supported(&self, kind: GinType) -> bool {
        match kind {
            GinType::Automatic => self.raw.ginSupport.iter().any(|supported| *supported),
            _ => self.raw.ginSupport[kind.into_raw() as usize],
        }
    }

    /// Returns the number of load/store-accessible teams.
    #[must_use]
    pub fn lsa_team_count(&self) -> i32 {
        self.raw.nLsaTeams
    }

    /// Returns the communicator hash reported by NCCL.
    #[must_use]
    pub fn communicator_hash(&self) -> u64 {
        self.raw.commHash
    }

    /// Returns the minimum supported GIN rank stride.
    #[must_use]
    pub fn minimum_gin_stride(&self) -> i32 {
        self.raw.ginMinStride
    }
}

impl Communicator {
    /// Device communicator API properties.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// NCCL property-query errors are propagated.
    pub fn device_properties(&self) -> Result<DeviceProperties> {
        self.bind()?;
        let mut raw = sys::ncclCommProperties_t {
            size: size_of::<sys::ncclCommProperties_t>() as _,
            magic: sys::NCCL_API_MAGIC,
            version: sys::NCCL_VERSION_CODE,
            ..Default::default()
        };
        // SAFETY: `ncclCommQueryProperties` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclCommQueryProperties(self.as_raw(), &raw mut raw))?;
        }
        Ok(DeviceProperties { raw })
    }

    /// Returns the team containing all communicator ranks.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    pub fn world_team(&self) -> Result<Team<'_>> {
        self.bind()?;
        Ok(Team {
            communicator: self,
            // SAFETY: `ncclTeamWorld` receives live handles owned by the wrappers.
            raw: unsafe { sys::ncclTeamWorld(self.as_raw()) },
        })
    }

    /// Returns the calling rank's load/store-accessible team.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    pub fn lsa_team(&self) -> Result<Team<'_>> {
        self.bind()?;
        Ok(Team {
            communicator: self,
            // SAFETY: `ncclTeamLsa` receives live handles owned by the wrappers.
            raw: unsafe { sys::ncclTeamLsa(self.as_raw()) },
        })
    }

    /// Returns the calling rank's network rail team.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    pub fn rail_team(&self) -> Result<Team<'_>> {
        self.bind()?;
        Ok(Team {
            communicator: self,
            // SAFETY: `ncclTeamRail` receives live handles owned by the wrappers.
            raw: unsafe { sys::ncclTeamRail(self.as_raw()) },
        })
    }

    /// Returns the Compute Fabric Transport multimem team.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    pub fn cft_multimem_team(&self) -> Result<Team<'_>> {
        self.bind()?;
        Ok(Team {
            communicator: self,
            // SAFETY: `ncclTeamCftMultimem` receives live handles owned by the wrappers.
            raw: unsafe { sys::ncclTeamCftMultimem(self.as_raw()) },
        })
    }

    /// Returns a Compute Fabric Transport team for the selected hierarchy mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    pub fn cft_team(&self, mode: CftTeamMode) -> Result<Team<'_>> {
        self.bind()?;
        let mode = match mode {
            CftTeamMode::Flat => sys::ncclCftTeamMode_t::NCCL_CFT_TEAM_FLAT,
            CftTeamMode::HierarchicalMultimem => {
                sys::ncclCftTeamMode_t::NCCL_CFT_TEAM_HIER_MULTIMEM
            }
            CftTeamMode::HierarchicalLsa => sys::ncclCftTeamMode_t::NCCL_CFT_TEAM_HIER_LSA,
        };
        Ok(Team {
            communicator: self,
            // SAFETY: `ncclTeamCft` receives live handles owned by the wrappers.
            raw: unsafe { sys::ncclTeamCft(self.as_raw(), mode) },
        })
    }
}

/// An owned NCCL device communicator. Its host communicator outlives it.
/// Kernel launches must finish before destruction; destruction synchronizes the context.
#[derive(Debug)]
pub struct DeviceCommunicator<'a> {
    communicator: &'a Communicator,
    raw: sys::ncclDevComm_t,
    resources: Vec<Resource>,
    teams: Vec<TeamResource>,
}

/// Requirements for resources allocated during device communicator creation.
#[derive(Debug)]
pub struct DeviceCommunicatorBuilder<'a> {
    communicator: &'a Communicator,
    requirements: DeviceRequirements,
    resources: Vec<Resource>,
    teams: Vec<TeamResource>,
}

#[derive(Debug)]
struct Resource {
    request: Box<sys::ncclDevResourceRequirements_t>,
    handle: ResourceHandle,
}

#[derive(Debug)]
enum ResourceHandle {
    Lsa(Box<sys::ncclLsaBarrierHandle_t>),
    Gin(Box<sys::ncclGinBarrierHandle_t>),
    Cft(Box<sys::ncclCftBarrierHandle_t>),
    Lla2a(Box<sys::ncclLLA2AHandle_t>),
    Buffer(Box<DeviceResource>),
}

/// Handles assigned during device communicator creation.
#[derive(Debug, Default)]
pub struct DeviceResource {
    buffer: sys::ncclDevResourceHandle_t,
    signal_start: sys::ncclGinSignal_t,
    counter_start: sys::ncclGinCounter_t,
}

impl DeviceResource {
    /// Returns the native device buffer handle.
    #[must_use]
    pub fn buffer_handle(&self) -> sys::ncclDevResourceHandle_t {
        self.buffer
    }

    /// Returns the first GIN signal index assigned to this resource.
    #[must_use]
    pub fn signal_start(&self) -> u32 {
        self.signal_start
    }

    /// Returns the first GIN counter index assigned to this resource.
    #[must_use]
    pub fn counter_start(&self) -> u32 {
        self.counter_start
    }
}

#[derive(Debug)]
struct TeamResource {
    request: Box<sys::ncclTeamRequirements_t>,
    handle: Box<sys::ncclMultimemHandle_t>,
}

impl<'a> DeviceCommunicatorBuilder<'a> {
    /// Creates a device communicator builder with the requested capabilities.
    ///
    /// This does not allocate native device resources.
    pub fn new(communicator: &'a Communicator, requirements: DeviceRequirements) -> Self {
        Self {
            communicator,
            requirements,
            resources: Vec::new(),
            teams: Vec::new(),
        }
    }

    /// Requests device buffer storage and optional GIN signals and counters.
    /// Output handles are assigned during device communicator creation.
    ///
    /// # Errors
    ///
    /// Returns an error if `alignment` is not a nonzero power of two or signal or counter counts are negative.
    pub fn resource(
        &mut self,
        bytes: u64,
        alignment: u64,
        signals: i32,
        counters: i32,
    ) -> Result<usize> {
        if !alignment.is_power_of_two() || signals < 0 || counters < 0 {
            return Err(Error::out_of_range("device resource requirements"));
        }
        let mut handle = Box::new(DeviceResource::default());
        let request = Box::new(sys::ncclDevResourceRequirements_t {
            next: ptr::null_mut(),
            bufferSize: bytes,
            bufferAlign: alignment,
            outBufferHandle: &raw mut handle.buffer,
            ginSignalCount: signals,
            ginCounterCount: counters,
            outGinSignalStart: &raw mut handle.signal_start,
            outGinCounterStart: &raw mut handle.counter_start,
        });
        let index = self.resources.len();
        self.resources.push(Resource {
            request,
            handle: ResourceHandle::Buffer(handle),
        });
        Ok(index)
    }

    fn check_team(&self, team: Team<'_>, count: i32) -> Result<()> {
        if !ptr::eq(team.communicator, self.communicator) || count < 0 {
            return Err(Error::out_of_range("device resource team or count"));
        }
        self.communicator.bind()
    }

    /// Device resource requirement for LSA barriers. Returns its resource index.
    ///
    /// # Errors
    ///
    /// Returns an error if the team belongs to another communicator or `count` is negative.
    /// Communicator-state checks, context binding, and native resource-requirement creation errors are propagated.
    pub fn lsa_barriers(&mut self, team: Team<'a>, count: i32) -> Result<usize> {
        self.check_team(team, count)?;
        let mut handle = Box::new(sys::ncclLsaBarrierHandle_t::default());
        let mut request = Box::new(sys::ncclDevResourceRequirements_t::default());
        // SAFETY: `ncclLsaBarrierCreateRequirement` receives only plain values.
        unsafe {
            try_ffi!(sys::ncclLsaBarrierCreateRequirement(
                team.raw,
                count,
                &raw mut *handle,
                &raw mut *request
            ))?;
        }
        let index = self.resources.len();
        self.resources.push(Resource {
            request,
            handle: ResourceHandle::Lsa(handle),
        });
        Ok(index)
    }

    /// Device resource requirement for GIN barriers. Returns its resource index.
    ///
    /// # Errors
    ///
    /// Returns an error if the team belongs to another communicator or `count` is negative.
    /// Communicator-state checks, context binding, and native resource-requirement creation errors are propagated.
    pub fn gin_barriers(&mut self, team: Team<'a>, count: i32) -> Result<usize> {
        self.check_team(team, count)?;
        let mut handle = Box::new(sys::ncclGinBarrierHandle_t::default());
        let mut request = Box::new(sys::ncclDevResourceRequirements_t::default());
        // SAFETY: `ncclGinBarrierCreateRequirement` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::ncclGinBarrierCreateRequirement(
                self.communicator.as_raw(),
                team.raw,
                count,
                &raw mut *handle,
                &raw mut *request
            ))?;
        }
        let index = self.resources.len();
        self.resources.push(Resource {
            request,
            handle: ResourceHandle::Gin(handle),
        });
        Ok(index)
    }

    /// Device resource requirement for CFT barriers. Returns its resource index.
    ///
    /// # Errors
    ///
    /// Returns an error if the team belongs to another communicator or `count` is negative.
    /// Communicator-state checks, context binding, and native resource-requirement creation errors are propagated.
    pub fn cft_barriers(&mut self, team: Team<'a>, count: i32) -> Result<usize> {
        self.check_team(team, count)?;
        let mut handle = Box::new(sys::ncclCftBarrierHandle_t::default());
        let mut request = Box::new(sys::ncclDevResourceRequirements_t::default());
        // SAFETY: `ncclCftBarrierCreateRequirement` receives only plain values.
        unsafe {
            try_ffi!(sys::ncclCftBarrierCreateRequirement(
                team.raw,
                count,
                &raw mut *handle,
                &raw mut *request
            ))?;
        }
        let index = self.resources.len();
        self.resources.push(Resource {
            request,
            handle: ResourceHandle::Cft(handle),
        });
        Ok(index)
    }

    /// Device resource requirement for low-latency all-to-all. Returns its resource index.
    ///
    /// # Errors
    ///
    /// Returns an error if `blocks` or `slots` is negative or NCCL rejects the resource requirement.
    pub fn low_latency_all_to_all(&mut self, blocks: i32, slots: i32) -> Result<usize> {
        if blocks < 0 || slots < 0 {
            return Err(Error::out_of_range("low-latency all-to-all resources"));
        }
        let mut handle = Box::new(sys::ncclLLA2AHandle_t::default());
        let mut request = Box::new(sys::ncclDevResourceRequirements_t::default());
        // SAFETY: `ncclLLA2ACreateRequirement` receives only plain values.
        unsafe {
            try_ffi!(sys::ncclLLA2ACreateRequirement(
                blocks,
                slots,
                &raw mut *handle,
                &raw mut *request
            ))?;
        }
        let index = self.resources.len();
        self.resources.push(Resource {
            request,
            handle: ResourceHandle::Lla2a(handle),
        });
        Ok(index)
    }

    /// Enable multimem on a team. Returns its team-resource index.
    ///
    /// # Errors
    ///
    /// Returns an error if the team belongs to another communicator, the communicator state disallows control operations, or CUDA context binding fails.
    pub fn multimem(&mut self, team: Team<'a>) -> Result<usize> {
        self.check_team(team, 0)?;
        let mut handle = Box::new(sys::ncclMultimemHandle_t::default());
        let request = Box::new(sys::ncclTeamRequirements_t {
            next: ptr::null_mut(),
            team: team.raw,
            multimem: true,
            outMultimemHandle: &raw mut *handle,
        });
        let index = self.teams.len();
        self.teams.push(TeamResource { request, handle });
        Ok(index)
    }

    /// Creates the device communicator and allocates its requested resources.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, a resource count or queue depth is negative, a custom GIN stride is not positive, or CUDA context binding fails.
    /// Also returns an error if NCCL cannot satisfy the requested device resources or capabilities.
    pub fn create(mut self) -> Result<DeviceCommunicator<'a>> {
        self.communicator.require_state_for_operations()?;
        self.communicator.bind()?;
        let r = &self.requirements;
        if [
            r.barriers,
            r.lsa_barriers,
            r.rail_gin_barriers,
            r.world_gin_barriers,
            r.lsa_lla2a_blocks,
            r.lsa_lla2a_slots,
            r.gin_contexts,
            r.gin_signals,
            r.gin_counters,
            r.gin_queue_depth,
            r.cft_barriers,
        ]
        .into_iter()
        .any(|n| n < 0)
        {
            return Err(Error::out_of_range("device requirements"));
        }
        for i in 1..self.resources.len() {
            self.resources[i - 1].request.next = &raw mut *self.resources[i].request;
        }
        for i in 1..self.teams.len() {
            self.teams[i - 1].request.next = &raw mut *self.teams[i].request;
        }
        let (connection, stride) = match r.gin_connection {
            GinConnection::None => (sys::ncclGinConnectionType_t::NCCL_GIN_CONNECTION_NONE, 1),
            GinConnection::Full => (sys::ncclGinConnectionType_t::NCCL_GIN_CONNECTION_FULL, 1),
            GinConnection::Rail => (sys::ncclGinConnectionType_t::NCCL_GIN_CONNECTION_RAIL, 1),
            GinConnection::CustomStride(stride) if stride > 0 => (
                sys::ncclGinConnectionType_t::NCCL_GIN_CONNECTION_CUSTOM_STRIDE,
                stride,
            ),
            GinConnection::CustomStride(_) => {
                return Err(Error::out_of_range("gin custom stride"));
            }
        };
        let requirements = sys::ncclDevCommRequirements_t {
            size: size_of::<sys::ncclDevCommRequirements_t>() as _,
            magic: sys::NCCL_API_MAGIC,
            version: sys::NCCL_VERSION_CODE,
            resourceRequirementsList: self
                .resources
                .first_mut()
                .map_or(ptr::null_mut(), |r| &raw mut *r.request),
            teamRequirementsList: self
                .teams
                .first_mut()
                .map_or(ptr::null_mut(), |r| &raw mut *r.request),
            lsaMultimem: r.lsa_multimem,
            barrierCount: r.barriers,
            lsaBarrierCount: r.lsa_barriers,
            railGinBarrierCount: r.rail_gin_barriers,
            lsaLLA2ABlockCount: r.lsa_lla2a_blocks,
            lsaLLA2ASlotCount: r.lsa_lla2a_slots,
            ginContextCount: r.gin_contexts,
            ginSignalCount: r.gin_signals,
            ginCounterCount: r.gin_counters,
            ginConnectionType: connection,
            ginExclusiveContexts: r.gin_exclusive_contexts,
            ginQueueDepth: r.gin_queue_depth,
            ginTrafficClass: r.gin_traffic_class.unwrap_or(i32::MIN),
            worldGinBarrierCount: r.world_gin_barriers,
            ginStrongSignalsRequired: true,
            ginVaSignalsRequired: true,
            ginCustomStride: stride,
            ginType: r.gin_type.into_raw(),
            useRuntimeVersion: false,
            cftCaps: r.cft_capabilities.bits(),
            cftBarrierCount: r.cft_barriers,
            ..Default::default()
        };
        let mut raw = sys::ncclDevComm_t::default();
        // SAFETY: `ncclDevCommCreate` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclDevCommCreate(
                self.communicator.as_raw(),
                &raw const requirements,
                &raw mut raw
            ))?;
        }
        Ok(DeviceCommunicator {
            communicator: self.communicator,
            raw,
            resources: self.resources,
            teams: self.teams,
        })
    }
}

impl DeviceCommunicator<'_> {
    /// Borrowed kernel argument. Kernels must use the same NCCL headers as these bindings.
    #[must_use]
    pub fn as_raw(&self) -> &sys::ncclDevComm_t {
        &self.raw
    }

    /// Returns the LSA barrier resource at `index`.
    ///
    /// Returns `None` if the index is out of range or identifies another resource type.
    #[must_use]
    pub fn lsa_barriers(&self, index: usize) -> Option<&sys::ncclLsaBarrierHandle_t> {
        match &self.resources.get(index)?.handle {
            ResourceHandle::Lsa(h) => Some(h),
            _ => None,
        }
    }

    /// Returns the GIN barrier resource at `index`.
    ///
    /// Returns `None` if the index is out of range or identifies another resource type.
    #[must_use]
    pub fn gin_barriers(&self, index: usize) -> Option<&sys::ncclGinBarrierHandle_t> {
        match &self.resources.get(index)?.handle {
            ResourceHandle::Gin(h) => Some(h),
            _ => None,
        }
    }

    /// Returns the CFT barrier resource at `index`.
    ///
    /// Returns `None` if the index is out of range or identifies another resource type.
    #[must_use]
    pub fn cft_barriers(&self, index: usize) -> Option<&sys::ncclCftBarrierHandle_t> {
        match &self.resources.get(index)?.handle {
            ResourceHandle::Cft(h) => Some(h),
            _ => None,
        }
    }

    /// Returns the low-latency all-to-all resource at `index`.
    ///
    /// Returns `None` if the index is out of range or identifies another resource type.
    #[must_use]
    pub fn low_latency_all_to_all(&self, index: usize) -> Option<&sys::ncclLLA2AHandle_t> {
        match &self.resources.get(index)?.handle {
            ResourceHandle::Lla2a(h) => Some(h),
            _ => None,
        }
    }

    /// Returns the device buffer resource at `index`.
    ///
    /// Returns `None` if the index is out of range or identifies another resource type.
    #[must_use]
    pub fn resource(&self, index: usize) -> Option<&DeviceResource> {
        match &self.resources.get(index)?.handle {
            ResourceHandle::Buffer(handle) => Some(handle),
            _ => None,
        }
    }

    /// Returns the multimem handle for a requested team, or `None` if the index is out of range.
    #[must_use]
    pub fn multimem(&self, index: usize) -> Option<Multimem<'_>> {
        Some(Multimem {
            communicator: self.communicator,
            handle: &self.teams.get(index)?.handle,
        })
    }
}

impl Drop for DeviceCommunicator<'_> {
    fn drop(&mut self) {
        if self.communicator.bind().is_ok()
            && self.communicator.cuda_context().synchronize().is_ok()
        {
            // SAFETY: `ncclDevCommDestroy` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
            unsafe {
                if let Err(error) = try_ffi!(sys::ncclDevCommDestroy(
                    self.communicator.as_raw(),
                    &raw const self.raw
                )) {
                    #[cfg(debug_assertions)]
                    eprintln!("failed to destroy nccl device communicator: {error}");
                }
            }
        }
    }
}

/// Calculates the slots required by low-latency all-to-all.
///
/// # Errors
///
/// Returns an error if `max_elements` is negative or `max_element_size` is not positive.
pub fn low_latency_all_to_all_slots(max_elements: i32, max_element_size: i32) -> Result<i32> {
    if max_elements < 0 || max_element_size <= 0 {
        return Err(Error::out_of_range("all-to-all element size or count"));
    }
    // SAFETY: `ncclLLA2ACalcSlots` receives only plain values.
    Ok(unsafe { sys::ncclLLA2ACalcSlots(max_elements, max_element_size) })
}

/// Multimem resource borrowed from an allocated device communicator.
#[derive(Debug)]
pub struct Multimem<'a> {
    communicator: &'a Communicator,
    handle: &'a sys::ncclMultimemHandle_t,
}

impl Multimem<'_> {
    /// Returns the native multimem handle.
    #[must_use]
    pub fn as_raw(&self) -> &sys::ncclMultimemHandle_t {
        self.handle
    }
}

/// CFT handle based query result for host code.
#[derive(Debug, Clone, Copy)]
pub struct LogicalEndpoint {
    /// The logical endpoint identifier returned by NCCL.
    pub id: u32,
    /// The endpoint-relative offset returned by NCCL.
    pub offset: u64,
}

impl Window<'_> {
    fn check_offset(&self, offset: u64) -> Result<()> {
        self.communicator.bind()?;
        if offset >= self.byte_len {
            return Err(Error::out_of_range("window offset"));
        }
        Ok(())
    }

    /// VA pointer based query for host code.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Returns an error if `offset` is outside this registered window.
    /// NCCL pointer or endpoint-query errors are propagated.
    pub fn lsa_multimem_pointer(&self, offset: u64) -> Result<*mut c_void> {
        self.check_offset(offset)?;
        let mut value = ptr::null_mut();
        // SAFETY: `ncclGetLsaMultimemDevicePointer` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclGetLsaMultimemDevicePointer(
                self.raw,
                offset,
                &raw mut value
            ))?;
        }
        Ok(value)
    }

    /// Returns a device pointer for the selected multimem mapping at a byte offset.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Returns an error if `offset` is outside this registered window.
    /// The multimem handle must belong to this communicator.
    /// NCCL pointer or endpoint-query errors are propagated.
    pub fn multimem_pointer(&self, offset: u64, multimem: &Multimem<'_>) -> Result<*mut c_void> {
        self.check_offset(offset)?;
        if !ptr::eq(self.communicator, multimem.communicator) {
            return Err(Error::out_of_range("multimem communicator"));
        }
        let mut value = ptr::null_mut();
        // SAFETY: `ncclGetMultimemDevicePointer` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclGetMultimemDevicePointer(
                self.raw,
                offset,
                *multimem.handle,
                &raw mut value
            ))?;
        }
        Ok(value)
    }

    /// Returns the load/store-accessible device pointer for a peer at a byte offset.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Returns an error if `offset` is outside this registered window.
    /// The peer rank must be within the queried team.
    /// NCCL pointer or endpoint-query errors are propagated.
    pub fn lsa_pointer(&self, offset: u64, rank: i32) -> Result<*mut c_void> {
        self.check_offset(offset)?;
        self.communicator.lsa_team()?.validate_rank(rank)?;
        let mut value = ptr::null_mut();
        // SAFETY: `ncclGetLsaDevicePointer` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclGetLsaDevicePointer(
                self.raw,
                offset,
                rank,
                &raw mut value
            ))?;
        }
        Ok(value)
    }

    /// Returns a peer's device pointer at a byte offset.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Returns an error if `offset` is outside this registered window.
    /// The peer rank must be within the queried team.
    /// NCCL pointer or endpoint-query errors are propagated.
    pub fn peer_pointer(&self, offset: u64, rank: i32) -> Result<*mut c_void> {
        self.check_offset(offset)?;
        self.communicator.world_team()?.validate_rank(rank)?;
        let mut value = ptr::null_mut();
        // SAFETY: `ncclGetPeerDevicePointer` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclGetPeerDevicePointer(
                self.raw,
                offset,
                rank,
                &raw mut value
            ))?;
        }
        Ok(value)
    }

    /// CFT handle based query for host code.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Returns an error if `offset` is outside this registered window.
    /// NCCL pointer or endpoint-query errors are propagated.
    pub fn multimem_endpoint(&self, offset: u64) -> Result<LogicalEndpoint> {
        self.check_offset(offset)?;
        let mut value = LogicalEndpoint { id: 0, offset: 0 };
        // SAFETY: `ncclGetMultimemDeviceLeInfo` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclGetMultimemDeviceLeInfo(
                self.raw,
                offset,
                &raw mut value.id,
                &raw mut value.offset
            ))?;
        }
        Ok(value)
    }

    /// Returns the logical endpoint for a rank in the selected Compute Fabric Transport team.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Returns an error if `offset` is outside this registered window.
    /// The peer rank must be within the queried team.
    /// The team must belong to this communicator.
    /// NCCL pointer or endpoint-query errors are propagated.
    pub fn cft_endpoint(&self, offset: u64, rank: i32, team: Team<'_>) -> Result<LogicalEndpoint> {
        self.check_offset(offset)?;
        team.validate_rank(rank)?;
        if !ptr::eq(team.communicator, self.communicator) {
            return Err(Error::out_of_range("window team"));
        }
        let mut value = LogicalEndpoint { id: 0, offset: 0 };
        // SAFETY: `ncclGetCftDeviceLeInfo` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclGetCftDeviceLeInfo(
                self.raw,
                offset,
                rank,
                team.raw,
                &raw mut value.id,
                &raw mut value.offset
            ))?;
        }
        Ok(value)
    }

    /// Returns the peer logical endpoint for a world rank.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Returns an error if `offset` is outside this registered window.
    /// The peer rank must be within the queried team.
    /// NCCL pointer or endpoint-query errors are propagated.
    pub fn peer_endpoint(&self, offset: u64, rank: i32) -> Result<LogicalEndpoint> {
        self.check_offset(offset)?;
        self.communicator.world_team()?.validate_rank(rank)?;
        let mut value = LogicalEndpoint { id: 0, offset: 0 };
        // SAFETY: `ncclGetPeerDeviceLeInfo` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclGetPeerDeviceLeInfo(
                self.raw,
                offset,
                rank,
                &raw mut value.id,
                &raw mut value.offset
            ))?;
        }
        Ok(value)
    }
}

/// Compiles a CUDA source file using the matching NCCL device headers.
/// The source can include `nccl_device.h` and accept `ncclDevComm` kernel arguments.
/// Writes a cubin for the specified physical GPU architecture to `output`.
///
/// NCCL's device headers require the native CUDA compiler and its host C++ headers.
/// Uses `NVCC`, then `CUDA_PATH/bin/nvcc`, then `nvcc` on `PATH`. `CUDA_CCBIN`, when
/// set, selects the host compiler. Compilation uses C++17.
///
/// # Errors
///
/// Returns an error if `architecture` is virtual, the NCCL device-header location is unavailable, the compiler process cannot start, or compilation fails.
pub fn compile_device_kernel(
    source: &Path,
    output: &Path,
    architecture: GpuArchitecture,
) -> Result<()> {
    if architecture.is_virtual() {
        return Err(Error::out_of_range("physical GPU architecture"));
    }
    let include = sys::DEVICE_INCLUDE_PATH
        .ok_or_else(|| Error::out_of_range("nccl device header location"))?;
    let compiler = env::var_os("NVCC")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("CUDA_PATH").map(|root| {
                PathBuf::from(root).join("bin").join(if cfg!(windows) {
                    "nvcc.exe"
                } else {
                    "nvcc"
                })
            })
        })
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    let mut command = Command::new(compiler);
    command
        .arg("--std=c++17")
        .arg("--cubin")
        .arg(format!("--gpu-architecture={architecture}"))
        .arg("--include-path")
        .arg(include)
        .arg(source)
        .arg("--output-file")
        .arg(output);
    for include in [sys::CUDA_INCLUDE_PATH, sys::CCCL_INCLUDE_PATH]
        .into_iter()
        .flatten()
    {
        command.arg("--include-path").arg(include);
    }
    if let Some(compiler) = env::var_os("CUDA_CCBIN") {
        command.arg("--compiler-bindir").arg(compiler);
    }
    let result = command.output().map_err(CudaError::from)?;
    if !result.status.success() {
        return Err(CudaError::Compiler {
            message: String::from_utf8_lossy(&result.stderr).into_owned(),
        }
        .into());
    }
    Ok(())
}
