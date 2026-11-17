use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda_sys::driver;

singe_core::native_enum! {
    /// Attribute of a compiled kernel function, queried or set per function.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum FunctionAttribute: driver::CUfunction_attribute {
        /// The shared memory mode of a function, as a [`SharedMemoryMode`].
        SharedMemoryMode = CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE,
        /// The maximum number of threads per block, beyond which a launch of the function would fail.
        ///
        /// This number depends on both the function and the device on which the function is currently loaded.
        MaxThreadsPerBlock = CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        /// The size in bytes of statically-allocated shared memory required by this function.
        ///
        /// This does not include dynamically-allocated shared memory requested by the user at runtime.
        SharedSizeBytes = CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        /// The size in bytes of user-allocated constant memory required by this function.
        ConstSizeBytes = CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
        /// The size in bytes of local memory used by each thread of this function.
        LocalSizeBytes = CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        /// The number of registers used by each thread of this function.
        NumRegs = CU_FUNC_ATTRIBUTE_NUM_REGS,
        /// The PTX virtual architecture version for which the function was compiled.
        ///
        /// This value is the major PTX version times 10 plus the minor PTX version, so a PTX version 1.3 function returns 13.
        /// Cubins compiled before CUDA 3.0 may report the undefined value 0.
        PtxVersion = CU_FUNC_ATTRIBUTE_PTX_VERSION,
        /// The binary architecture version for which the function was compiled.
        ///
        /// This value is the major binary version times 10 plus the minor binary version, so a binary version 1.3 function returns 13.
        /// Legacy cubins without a properly encoded binary architecture version report 10.
        BinaryVersion = CU_FUNC_ATTRIBUTE_BINARY_VERSION,
        /// Whether the function was compiled with the `-Xptxas --dlcm=ca` option.
        CacheModeCa = CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
        /// The maximum size in bytes of dynamically-allocated shared memory that can be used by this function.
        ///
        /// If the user-specified dynamic shared memory size is larger than this value, the launch will fail.
        ///
        /// The default is [`DeviceAttribute::MaxSharedMemoryPerBlock`](crate::device::DeviceAttribute::MaxSharedMemoryPerBlock) minus [`Self::SharedSizeBytes`], or 0 when the static size exceeds that maximum.
        /// It can be raised to [`DeviceAttribute::MaxSharedMemoryPerBlockOptin`](crate::device::DeviceAttribute::MaxSharedMemoryPerBlockOptin) minus [`Self::SharedSizeBytes`].
        ///
        /// This attribute is ignored if the function or launch sets a shared memory mode, such as [`Self::SharedMemoryMode`].
        ///
        /// This attribute cannot be used to access oversized shared memory, which only a shared memory mode enables.
        MaxDynamicSharedSizeBytes = CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        /// On devices where the L1 cache and shared memory use the same hardware resources, this sets the shared memory carveout preference, in percent of the total shared memory.
        ///
        /// The total is [`DeviceAttribute::MaxSharedMemoryPerMultiprocessor`](crate::device::DeviceAttribute::MaxSharedMemoryPerMultiprocessor).
        /// This is only a hint, and the driver can choose a different ratio if required to execute the function.
        PreferredSharedMemoryCarveout = CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        /// If this attribute is set, the kernel must launch with a valid cluster size specified.
        ClusterSizeMustBeSet = CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET,
        /// The required cluster width in blocks.
        ///
        /// The values must either all be 0 or all be positive.
        /// The validity of the cluster dimensions is otherwise checked at launch time.
        ///
        /// If the value is set during compile time, it cannot be set at runtime.
        /// Setting it at runtime returns [`Status::NotPermitted`](crate::error::Status::NotPermitted).
        RequiredClusterWidth = CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,
        /// The required cluster height in blocks.
        ///
        /// The values must either all be 0 or all be positive.
        /// The validity of the cluster dimensions is otherwise checked at launch time.
        ///
        /// If the value is set during compile time, it cannot be set at runtime.
        /// Setting it at runtime returns [`Status::NotPermitted`](crate::error::Status::NotPermitted).
        RequiredClusterHeight = CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT,
        /// The required cluster depth in blocks.
        ///
        /// The values must either all be 0 or all be positive.
        /// The validity of the cluster dimensions is otherwise checked at launch time.
        ///
        /// If the value is set during compile time, it cannot be set at runtime.
        /// Setting it at runtime returns [`Status::NotPermitted`](crate::error::Status::NotPermitted).
        RequiredClusterDepth = CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH,
        /// Whether the function can be launched with non-portable cluster size.
        ///
        /// 1 is allowed, 0 is disallowed.
        /// A non-portable cluster size may only function on the specific SKUs the program is tested on.
        /// The launch might fail if the program is run on a different hardware platform.
        ///
        /// An occupancy query for the maximum active clusters checks whether a size can launch on the current device.
        ///
        /// A portable cluster size is guaranteed to be functional on all compute capabilities higher than the target compute capability.
        /// The portable cluster size for `sm_90` is 8 blocks per cluster.
        /// This value may increase for future compute capabilities.
        ///
        /// The specific hardware unit may support higher cluster sizes that are not guaranteed to be portable.
        NonPortableClusterSizeAllowed = CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
        /// The block scheduling policy of a function, as a [`ClusterSchedulingPolicy`].
        ClusterSchedulingPolicyPreference = CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
    }
}

impl_enum_display!(FunctionAttribute, {
    Self::SharedMemoryMode => "CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE",
    Self::MaxThreadsPerBlock => "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
    Self::SharedSizeBytes => "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
    Self::ConstSizeBytes => "CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES",
    Self::LocalSizeBytes => "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
    Self::NumRegs => "CU_FUNC_ATTRIBUTE_NUM_REGS",
    Self::PtxVersion => "CU_FUNC_ATTRIBUTE_PTX_VERSION",
    Self::BinaryVersion => "CU_FUNC_ATTRIBUTE_BINARY_VERSION",
    Self::CacheModeCa => "CU_FUNC_ATTRIBUTE_CACHE_MODE_CA",
    Self::MaxDynamicSharedSizeBytes => "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
    Self::PreferredSharedMemoryCarveout => "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT",
    Self::ClusterSizeMustBeSet => "CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET",
    Self::RequiredClusterWidth => "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH",
    Self::RequiredClusterHeight => "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT",
    Self::RequiredClusterDepth => "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH",
    Self::NonPortableClusterSizeAllowed => "CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED",
    Self::ClusterSchedulingPolicyPreference => "CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE",
});

singe_core::native_enum! {
    /// Function cache configurations
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum FunctionCache: driver::CUfunc_cache {
        /// no preference for shared memory or L1 (default).
        PreferNone = CU_FUNC_CACHE_PREFER_NONE,
        /// prefer larger shared memory and smaller L1 cache.
        PreferShared = CU_FUNC_CACHE_PREFER_SHARED,
        /// prefer larger L1 cache and smaller shared memory.
        PreferL1 = CU_FUNC_CACHE_PREFER_L1,
        /// prefer equal sized L1 cache and shared memory.
        PreferEqual = CU_FUNC_CACHE_PREFER_EQUAL,
    }
}

impl_enum_display!(FunctionCache, {
    Self::PreferNone => "CU_FUNC_CACHE_PREFER_NONE",
    Self::PreferShared => "CU_FUNC_CACHE_PREFER_SHARED",
    Self::PreferL1 => "CU_FUNC_CACHE_PREFER_L1",
    Self::PreferEqual => "CU_FUNC_CACHE_PREFER_EQUAL",
});

singe_core::native_enum! {
    /// Deprecated:
    ///
    /// Shared memory configurations
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[deprecated]
    #[non_exhaustive]
    pub enum SharedMemoryConfig: driver::CUsharedconfig {
        /// set default shared memory bank size.
        DefaultBankSize = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE,
        /// set shared memory bank width to four bytes.
        FourByteBankSize = CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE,
        /// set shared memory bank width to eight bytes.
        EightByteBankSize = CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE,
    }
}

impl_enum_display!(SharedMemoryConfig, {
    Self::DefaultBankSize => "CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE",
    Self::FourByteBankSize => "CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE",
    Self::EightByteBankSize => "CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE",
});

singe_core::native_enum! {
    /// Shared memory carveout configurations for a function's [`FunctionAttribute::PreferredSharedMemoryCarveout`] attribute.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(i32)]
    #[non_exhaustive]
    pub enum SharedMemoryCarveout: driver::CUshared_carveout {
        /// No preference for shared memory or L1 (default).
        Default = CU_SHAREDMEM_CARVEOUT_DEFAULT,
        /// Prefer maximum available shared memory, minimum L1 cache.
        MaxShared = CU_SHAREDMEM_CARVEOUT_MAX_SHARED,
        /// Prefer maximum available L1 cache, minimum shared memory.
        MaxL1 = CU_SHAREDMEM_CARVEOUT_MAX_L1,
    }
}

impl_enum_display!(SharedMemoryCarveout, {
    Self::Default => "CU_SHAREDMEM_CARVEOUT_DEFAULT",
    Self::MaxShared => "CU_SHAREDMEM_CARVEOUT_MAX_SHARED",
    Self::MaxL1 => "CU_SHAREDMEM_CARVEOUT_MAX_L1",
});

singe_core::native_enum! {
    /// Shared memory configuration for a kernel.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SharedMemoryMode: driver::CUsharedMemoryMode_enum {
        /// Uses the function's [`FunctionAttribute::MaxDynamicSharedSizeBytes`] for shared memory on launch.
        Default = CU_SHARED_MEMORY_MODE_DEFAULT,
        /// Requires the requested dynamic shared memory size to be portable, within [`DeviceAttribute::MaxSharedMemoryPerBlock`](crate::device::DeviceAttribute::MaxSharedMemoryPerBlock).
        RequirePortable = CU_SHARED_MEMORY_MODE_REQUIRE_PORTABLE,
        /// Allows a non-portable dynamic shared memory size, within [`DeviceAttribute::MaxSharedMemoryPerBlockOptin`](crate::device::DeviceAttribute::MaxSharedMemoryPerBlockOptin).
        AllowNonPortable = CU_SHARED_MEMORY_MODE_ALLOW_NON_PORTABLE,
        /// Specifies that oversized shared memory configurations may be used (with the limitation of only 8kB L1 cache).
        AllowOversizedSharedMemory = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY,
        /// Specifies that oversized shared memory configurations may be used (with the limitation of only 8kB L1 cache), and prefer an oversized shared memory configuration.
        PreferOversizedSharedMemory = CU_SHARED_MEMORY_MODE_PREFER_OVERSIZED_SHARED_MEMORY,
    }
}

singe_core::native_enum! {
    /// Cluster scheduling policy.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ClusterSchedulingPolicy: driver::CUclusterSchedulingPolicy_enum {
        /// the default policy.
        Default = CU_CLUSTER_SCHEDULING_POLICY_DEFAULT,
        /// spread the blocks within a cluster to the SMs.
        Spread = CU_CLUSTER_SCHEDULING_POLICY_SPREAD,
        /// allow the hardware to load-balance the blocks in a cluster to the SMs.
        LoadBalancing = CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING,
        /// Spread the blocks within a cluster to the SMs, and prefer placement that prioritizes Distributed Shared Memory locality for Rubin-class GPUs.
        ///
        /// This is a preference only.
        /// The driver may ignore this request when the launch is not eligible, when the device does not support the requested placement behavior, or when other launch constraints take precedence.
        ///
        /// This policy is intended for clustered kernels whose performance depends on communication through Distributed Shared Memory among blocks in the same cluster.
        RubinDsmemLocality = CU_CLUSTER_SCHEDULING_POLICY_RUBIN_DSMEM_LOCALITY,
    }
}

impl_enum_display!(SharedMemoryMode, {
    SharedMemoryMode::Default => "CU_SHARED_MEMORY_MODE_DEFAULT",
    SharedMemoryMode::RequirePortable => "CU_SHARED_MEMORY_MODE_REQUIRE_PORTABLE",
    SharedMemoryMode::AllowNonPortable => "CU_SHARED_MEMORY_MODE_ALLOW_NON_PORTABLE",
    SharedMemoryMode::AllowOversizedSharedMemory => "CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY",
    SharedMemoryMode::PreferOversizedSharedMemory => "CU_SHARED_MEMORY_MODE_PREFER_OVERSIZED_SHARED_MEMORY",
});

impl_enum_display!(ClusterSchedulingPolicy, {
    ClusterSchedulingPolicy::Default => "CU_CLUSTER_SCHEDULING_POLICY_DEFAULT",
    ClusterSchedulingPolicy::Spread => "CU_CLUSTER_SCHEDULING_POLICY_SPREAD",
    ClusterSchedulingPolicy::LoadBalancing => "CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING",
    ClusterSchedulingPolicy::RubinDsmemLocality => "CU_CLUSTER_SCHEDULING_POLICY_RUBIN_DSMEM_LOCALITY",
});
