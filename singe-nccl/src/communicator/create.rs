use crate::{
    communicator::{
        Arc, AtomicU8, Communicator, CommunicatorBuilder, CommunicatorBuilderMode,
        CommunicatorState, Config, CudaContext, Device, Error, Handle, RefCell, Result, UniqueId,
        ensure_primary_context, ptr, sys, to_i32, try_ffi,
    },
    group::{is_active, with_group},
};

impl<'a> CommunicatorBuilder<'a> {
    /// Selects devices for local clique creation.
    #[must_use]
    pub fn for_devices(devices: &[Device]) -> Self {
        Self {
            mode: CommunicatorBuilderMode::Devices(
                devices.iter().map(|device| device.id()).collect(),
            ),
            config: None,
        }
    }

    /// Selects CUDA device IDs for local clique creation.
    #[must_use]
    pub fn for_device_ids(device_ids: &[i32]) -> Self {
        Self {
            mode: CommunicatorBuilderMode::Devices(device_ids.to_vec()),
            config: None,
        }
    }

    /// Configures one rank using its CUDA context and a shared unique identifier.
    #[must_use]
    pub fn rank(
        cuda_context: &'a Arc<CudaContext>,
        rank: i32,
        rank_count: i32,
        unique_id: UniqueId,
    ) -> Self {
        Self {
            mode: CommunicatorBuilderMode::Rank {
                cuda_context,
                rank,
                rank_count,
                unique_id,
            },
            config: None,
        }
    }

    /// Configures one rank using multiple shared unique identifiers.
    #[must_use]
    pub fn scalable_rank(
        cuda_context: &'a Arc<CudaContext>,
        rank: i32,
        rank_count: i32,
        unique_ids: &[UniqueId],
    ) -> Self {
        Self {
            mode: CommunicatorBuilderMode::ScalableRank {
                cuda_context,
                rank,
                rank_count,
                unique_ids: unique_ids.to_vec(),
            },
            config: None,
        }
    }

    /// Sets optional communicator initialization overrides.
    #[must_use]
    pub fn with_config(mut self, config: Config<'a>) -> Self {
        self.config = Some(config);
        self
    }

    /// Creates the rank selected by a single-rank builder.
    ///
    /// # Errors
    ///
    /// Returns an error if rank arguments or identifier counts are invalid, the CUDA context is not the device primary context, CUDA setup fails, or NCCL cannot initialize the communicator.
    /// Returns [`Error::InvalidBuilderMode`] for a device-list builder.
    pub fn create(self) -> Result<Communicator> {
        match self.mode {
            CommunicatorBuilderMode::Rank {
                cuda_context,
                rank,
                rank_count,
                unique_id,
            } => create_rank_impl(cuda_context, rank_count, unique_id, rank, self.config),
            CommunicatorBuilderMode::ScalableRank {
                cuda_context,
                rank,
                rank_count,
                unique_ids,
            } => {
                create_rank_scalable_impl(cuda_context, rank_count, rank, &unique_ids, self.config)
            }
            CommunicatorBuilderMode::Devices(_) => Err(Error::InvalidBuilderMode {
                operation: "single communicator creation".into(),
            }),
        }
    }

    /// Creates all communicators selected by a device-list builder.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBuilderMode`] if the builder selects a single rank or contains a communicator configuration.
    /// Also returns an error if a group is already open, the device count exceeds the native integer range, CUDA primary-context setup fails, or NCCL initialization fails.
    pub fn create_all(self) -> Result<Vec<Communicator>> {
        if self.config.is_some() {
            return Err(Error::InvalidBuilderMode {
                operation: "configured communicator clique creation".into(),
            });
        }

        match self.mode {
            CommunicatorBuilderMode::Devices(device_ids) => create_all_impl(&device_ids),
            CommunicatorBuilderMode::Rank { .. } | CommunicatorBuilderMode::ScalableRank { .. } => {
                Err(Error::InvalidBuilderMode {
                    operation: "communicator clique creation".into(),
                })
            }
        }
    }
}

fn create_all_impl(device_ids: &[i32]) -> Result<Vec<Communicator>> {
    if is_active() {
        return Err(Error::GroupActive);
    }
    let contexts = device_ids
        .iter()
        .map(|&device_id| {
            CudaContext::retain_primary_for_device(Device::new(device_id)).map_err(Error::from)
        })
        .collect::<Result<Vec<_>>>()?;
    let rank_count = to_i32(contexts.len(), "device_ids")?;
    let unique_id = crate::unique_id()?.into_raw();
    // Stable output slots stay live through the outermost group end.
    let mut handles = vec![ptr::null_mut(); contexts.len()];
    let result = with_group(|| {
        for (rank, (context, handle)) in contexts.iter().zip(&mut handles).enumerate() {
            context.bind()?;
            // SAFETY: every rank is initialized in its actual retained context.
            unsafe {
                try_ffi!(sys::ncclCommInitRank(
                    handle,
                    rank_count,
                    unique_id,
                    to_i32(rank, "rank")?
                ))?;
            }
        }
        Ok(())
    });
    if let Err(error) = result {
        for (context, handle) in contexts.iter().zip(&handles) {
            if !handle.is_null() && context.bind().is_ok() {
                // SAFETY: the failed initialization still transfers native ownership.
                unsafe {
                    sys::ncclCommAbort(*handle);
                }
            }
        }
        return Err(error);
    }
    handles
        .into_iter()
        .zip(contexts)
        .map(|(handle, context)| communicator_from_raw(handle, context))
        .collect()
}

fn create_rank_impl(
    cuda_context: &Arc<CudaContext>,
    rank_count: i32,
    unique_id: UniqueId,
    rank: i32,
    config: Option<Config<'_>>,
) -> Result<Communicator> {
    ensure_primary_context(cuda_context)?;
    cuda_context.bind()?;

    let mut handle = ptr::null_mut();
    let status = match config {
        Some(config) => {
            let mut config = config.into_raw();
            // SAFETY: `ncclCommInitRankConfig` receives valid output locations.
            unsafe {
                sys::ncclCommInitRankConfig(
                    &raw mut handle,
                    rank_count,
                    unique_id.into_raw(),
                    rank,
                    &raw mut config,
                )
            }
        }
        // SAFETY: `ncclCommInitRank` receives valid output locations.
        None => unsafe {
            sys::ncclCommInitRank(&raw mut handle, rank_count, unique_id.into_raw(), rank)
        },
    };
    communicator_from_initialization(handle, Arc::clone(cuda_context), status)
}

fn create_rank_scalable_impl(
    cuda_context: &Arc<CudaContext>,
    rank_count: i32,
    rank: i32,
    unique_ids: &[UniqueId],
    config: Option<Config<'_>>,
) -> Result<Communicator> {
    ensure_primary_context(cuda_context)?;
    cuda_context.bind()?;

    let mut raw_ids: Vec<_> = unique_ids.iter().copied().map(UniqueId::into_raw).collect();
    let raw_id_count = to_i32(raw_ids.len(), "unique_ids")?;

    let mut handle = ptr::null_mut();
    let mut config = config.map(Config::into_raw);
    // SAFETY: `ncclCommInitRankScalable` receives valid output locations.
    let status = unsafe {
        sys::ncclCommInitRankScalable(
            &raw mut handle,
            rank_count,
            rank,
            raw_id_count,
            raw_ids.as_mut_ptr(),
            config.as_mut().map_or(ptr::null_mut(), ptr::from_mut),
        )
    };

    communicator_from_initialization(handle, Arc::clone(cuda_context), status)
}

pub fn communicator_from_raw(
    raw: sys::ncclComm_t,
    cuda_context: Arc<CudaContext>,
) -> Result<Communicator> {
    if raw.is_null() {
        return Err(Error::NullHandle);
    }

    Ok(Communicator {
        handle: Handle {
            raw,
            cuda_context,
            state: AtomicU8::new(CommunicatorState::Active.into_raw()),
            initialization_output: None,
            launch_events: RefCell::new(Vec::new()),
        },
    })
}

// Nonblocking initialization transfers ownership even while work is in progress.
pub fn communicator_from_initialization(
    raw: sys::ncclComm_t,
    cuda_context: Arc<CudaContext>,
    status: sys::ncclResult_t,
) -> Result<Communicator> {
    if !matches!(
        status,
        sys::ncclResult_t::ncclSuccess | sys::ncclResult_t::ncclInProgress
    ) {
        if !raw.is_null() {
            // NCCL can return an initialization failure with an owned handle.
            // SAFETY: `ncclCommAbort` receives only plain values.
            unsafe { sys::ncclCommAbort(raw) };
        }
        return Err(status.into());
    }
    let communicator = communicator_from_raw(raw, cuda_context)?;
    if status == sys::ncclResult_t::ncclInProgress {
        communicator.set_state(CommunicatorState::Initializing);
    }
    Ok(communicator)
}
