use super::*;

impl<'a> CommunicatorBuilder<'a> {
    pub fn for_devices(devices: &[Device]) -> Self {
        Self {
            mode: CommunicatorBuilderMode::Devices(
                devices.iter().map(|device| device.id()).collect(),
            ),
            config: None,
        }
    }

    pub fn for_device_ids(device_ids: &[i32]) -> Self {
        Self {
            mode: CommunicatorBuilderMode::Devices(device_ids.to_vec()),
            config: None,
        }
    }

    pub fn rank(
        cuda_ctx: &'a Arc<CudaContext>,
        rank: i32,
        rank_count: i32,
        unique_id: UniqueId,
    ) -> Self {
        Self {
            mode: CommunicatorBuilderMode::Rank {
                cuda_ctx,
                rank,
                rank_count,
                unique_id,
            },
            config: None,
        }
    }

    pub fn scalable_rank(
        cuda_ctx: &'a Arc<CudaContext>,
        rank: i32,
        rank_count: i32,
        unique_ids: &[UniqueId],
    ) -> Self {
        Self {
            mode: CommunicatorBuilderMode::ScalableRank {
                cuda_ctx,
                rank,
                rank_count,
                unique_ids: unique_ids.to_vec(),
            },
            config: None,
        }
    }

    pub fn with_config(mut self, config: Config<'a>) -> Self {
        self.config = Some(config);
        self
    }

    pub fn create(self) -> Result<Communicator> {
        match self.mode {
            CommunicatorBuilderMode::Rank {
                cuda_ctx,
                rank,
                rank_count,
                unique_id,
            } => create_rank_impl(cuda_ctx, rank_count, unique_id, rank, self.config),
            CommunicatorBuilderMode::ScalableRank {
                cuda_ctx,
                rank,
                rank_count,
                unique_ids,
            } => create_rank_scalable_impl(cuda_ctx, rank_count, rank, &unique_ids, self.config),
            CommunicatorBuilderMode::Devices(_) => Err(Error::InvalidBuilderMode {
                operation: "single communicator creation".into(),
            }),
        }
    }

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
    let current_device = Device::current().ok();

    let contexts: Result<Vec<_>> = device_ids
        .iter()
        .copied()
        .map(|device_id| {
            Device::new(device_id).set_current()?;
            CudaContext::create().map_err(Error::from)
        })
        .collect();

    if let Some(device) = current_device {
        device.set_current()?;
    }

    let contexts = contexts?;
    let device_count = to_i32(device_ids.len(), "device_ids")?;
    let mut handles = vec![ptr::null_mut(); device_ids.len()];

    unsafe {
        try_ffi!(sys::ncclCommInitAll(
            handles.as_mut_ptr(),
            device_count,
            device_ids.as_ptr(),
        ))?;
    }

    handles
        .into_iter()
        .zip(contexts)
        .map(|(handle, cuda_ctx)| communicator_from_raw(handle, cuda_ctx))
        .collect()
}

fn create_rank_impl(
    cuda_ctx: &Arc<CudaContext>,
    rank_count: i32,
    unique_id: UniqueId,
    rank: i32,
    config: Option<Config<'_>>,
) -> Result<Communicator> {
    cuda_ctx.bind()?;

    let mut handle = ptr::null_mut();
    match config {
        Some(config) => {
            let mut config = config.into_raw();
            unsafe {
                try_ffi!(sys::ncclCommInitRankConfig(
                    &raw mut handle,
                    rank_count,
                    unique_id.into_raw(),
                    rank,
                    &raw mut config,
                ))?;
            }
        }
        None => unsafe {
            try_ffi!(sys::ncclCommInitRank(
                &raw mut handle,
                rank_count,
                unique_id.into_raw(),
                rank,
            ))?;
        },
    }

    communicator_from_raw(handle, Arc::clone(cuda_ctx))
}

fn create_rank_scalable_impl(
    cuda_ctx: &Arc<CudaContext>,
    rank_count: i32,
    rank: i32,
    unique_ids: &[UniqueId],
    config: Option<Config<'_>>,
) -> Result<Communicator> {
    cuda_ctx.bind()?;

    let mut raw_ids: Vec<_> = unique_ids.iter().copied().map(UniqueId::into_raw).collect();
    let raw_id_count = to_i32(raw_ids.len(), "unique_ids")?;

    let mut handle = ptr::null_mut();
    let mut config = config.map(Config::into_raw);
    unsafe {
        try_ffi!(sys::ncclCommInitRankScalable(
            &raw mut handle,
            rank_count,
            rank,
            raw_id_count,
            raw_ids.as_mut_ptr(),
            config
                .as_mut()
                .map_or(ptr::null_mut(), |config| config as *mut _),
        ))?;
    }

    communicator_from_raw(handle, Arc::clone(cuda_ctx))
}

pub(super) fn communicator_from_raw(
    raw: sys::ncclComm_t,
    cuda_ctx: Arc<CudaContext>,
) -> Result<Communicator> {
    if raw.is_null() {
        return Err(Error::NullHandle);
    }

    Ok(Communicator {
        handle: Handle {
            raw,
            cuda_ctx,
            state: AtomicU8::new(CommunicatorState::Active.into_raw()),
        },
    })
}
