//! How a kernel's work is divided among GPU threads, and the launch shape that follows.
//!
//! A schedule says nothing about what the kernel computes; the autotuner searches schedules for one kernel, and each emitter rejects the schedules it cannot express.

pub mod contraction;
pub mod parallel_tile;
pub mod space;

use thiserror::Error;

use crate::{
    ir::{AxisId, AxisKind, Expr, Kernel, Size},
    schedule::contraction::{Contraction, FusedContraction},
};

/// How threads cover a kernel's iteration domain.
///
/// A row schedule's loop over reduction axes covers `vector` points at once only where every access it makes walks the innermost reduction axis contiguously or not at all, and one point at a time elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(tag = "kind", rename_all = "snake_case")
)]
pub enum Schedule {
    /// One thread per `vector` consecutive points of the innermost parallel axis and per point of the other parallel axes, flattened with the last axis varying fastest, in blocks of `threads`.
    /// A thread walks the reduction axes itself, so its reductions combine values in order; kernels with reduction axes take a `vector` of one.
    /// A thread reads and writes each operand that the innermost axis walks with a stride of one through one wide memory access when its lanes are in range and the address is aligned.
    Elements {
        threads: u32,
        vector: u32,
    },
    /// One block of `threads` threads per point of the parallel axes.
    /// The threads split the reduction axes, flattened, `vector` consecutive points of the innermost one at a time, and combine their partial results through warp shuffles and shared memory.
    /// With `tiles` other than `[1, 1]`, each block covers `tiles[0]` points of one parallel axis and `tiles[1]` of another, as [`parallel_tile`] describes.
    Rows {
        threads: u32,
        vector: u32,
        tiles: [u32; 2],
    },
    /// `lanes` lanes of a warp per point of the parallel axes, in blocks of `threads` threads that hold `threads / lanes` points.
    /// The lanes split the reduction axes, flattened, `vector` consecutive points of the innermost one at a time, and combine their partial results through warp shuffles alone, so rows need no block synchronization; fewer than 32 lanes put several short rows in one warp.
    /// With `tiles` other than `[1, 1]`, each group of lanes covers `tiles[0]` points of one parallel axis and `tiles[1]` of another, as [`parallel_tile`] describes.
    Warps {
        threads: u32,
        vector: u32,
        lanes: u32,
        tiles: [u32; 2],
    },
    // `threads / splits` consecutive points of the parallel axes per block of `threads` threads, consecutive threads on consecutive points, so loads of a reduction over outer axes coalesce.
    // The `splits` groups of threads split the reduction axes, flattened, and combine their partial results through shared memory, every thread folding them in the same order.
    Columns {
        threads: u32,
        splits: u32,
    },
    /// One tile block per `size` consecutive points of the parallel axes, flattened with the last axis varying fastest, for kernels without reduction axes.
    /// A tile target decides how its threads cover the tile.
    Tiles {
        size: u32,
    },
    /// One tile block per point of the parallel axes, holding the reduction axes, flattened, in one tile of `width` points, which must cover them.
    TileRows {
        width: u32,
    },
    /// One tile block per `m` by `n` tile of a [`Contraction`]'s rows and columns at each point of its other parallel axes, which steps through the contraction `k` points at a time with tensor-core matrix products.
    /// `m`, `n`, and `k` are powers of two from 16 to 256.
    TileMatmul {
        m: u32,
        n: u32,
        k: u32,
    },
    // One block of four warps per `m` by `n` tile of a fused contraction's rows and columns at each point of its other parallel axes, which computes `k` points of each factor at a time into shared memory and multiplies them with `mma.sync`.
    // `m` and `n` are 32, 64, or 128, and `k` is 32 or 64.
    Mma {
        m: u32,
        n: u32,
        k: u32,
    },
}

/// A schedule that cannot run a kernel.
// The schedules the CUDA C++ emitter runs, in which threads cover points, rows, or warps; `Schedule` names the same variants for the tuner and its records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThreadSchedule {
    Elements {
        threads: u32,
        vector: u32,
    },
    Rows {
        threads: u32,
        vector: u32,
        tiles: [u32; 2],
    },
    Warps {
        threads: u32,
        vector: u32,
        lanes: u32,
        tiles: [u32; 2],
    },
    Columns {
        threads: u32,
        splits: u32,
    },
    Mma {
        m: u32,
        n: u32,
        k: u32,
    },
}

// The schedules the Tile IR emitter runs, in which a tile block covers points, a row, or a contraction tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileSchedule {
    Tiles { size: u32 },
    TileRows { width: u32 },
    TileMatmul { m: u32, n: u32, k: u32 },
}

// A schedule as the emitter that runs it takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleKind {
    Thread(ThreadSchedule),
    Tile(TileSchedule),
}

impl From<ThreadSchedule> for Schedule {
    fn from(schedule: ThreadSchedule) -> Self {
        match schedule {
            ThreadSchedule::Elements { threads, vector } => Self::Elements { threads, vector },
            ThreadSchedule::Rows {
                threads,
                vector,
                tiles,
            } => Self::Rows {
                threads,
                vector,
                tiles,
            },
            ThreadSchedule::Warps {
                threads,
                vector,
                lanes,
                tiles,
            } => Self::Warps {
                threads,
                vector,
                lanes,
                tiles,
            },
            ThreadSchedule::Columns { threads, splits } => Self::Columns { threads, splits },
            ThreadSchedule::Mma { m, n, k } => Self::Mma { m, n, k },
        }
    }
}

impl From<TileSchedule> for Schedule {
    fn from(schedule: TileSchedule) -> Self {
        match schedule {
            TileSchedule::Tiles { size } => Self::Tiles { size },
            TileSchedule::TileRows { width } => Self::TileRows { width },
            TileSchedule::TileMatmul { m, n, k } => Self::TileMatmul { m, n, k },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ScheduleError {
    #[error("{schedule} schedules need {requirement}, got {threads} threads")]
    Threads {
        schedule: &'static str,
        requirement: &'static str,
        threads: u32,
    },
    #[error("schedules take a vector of 1, 2, 4, 8, or 16 points, got {vector}")]
    Vector { vector: u32 },
    #[error("warp schedules give each point 1, 2, 4, 8, 16, or 32 lanes, got {lanes}")]
    Lanes { lanes: u32 },
    #[error(
        "column schedules split the reduction 2, 4, 8, 16, or 32 ways with fewer splits than threads, got {splits} of {threads}"
    )]
    Splits { splits: u32, threads: u32 },
    #[error("tile schedules take a power of two from 1 to {MAX_TILE} points, got {points}")]
    TileSize { points: u32 },
    #[error("a tile of {width} points cannot hold rows of {points}")]
    TileTooNarrow { width: u32, points: u64 },
    #[error("`{what}` count mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("the kernel's work does not fit a CUDA grid")]
    GridTooLarge,
    #[error("parallel tiles cover 1, 2, 4, 8, or 16 points, got {tile}")]
    Tile { tile: u32 },
    #[error("a tile of {factor} points does not divide an axis of {extent}")]
    TileFactor { factor: u32, extent: u64 },
    #[error("the kernel cannot be tiled: {reason}")]
    Untileable { reason: &'static str },
    #[error(
        "tensor-core tiles have sides that are powers of two from 16 to 256, got {m} by {n} by {k}"
    )]
    MatmulTile { m: u32, n: u32, k: u32 },
    #[error("tensor-core schedules run plain contractions only")]
    NotContraction,
    #[error(
        "mma.sync tiles are 32, 64, or 128 by 32, 64, or 128 by 32 or 64, got {m} by {n} by {k}"
    )]
    MmaTile { m: u32, n: u32, k: u32 },
    #[error("mma.sync schedules run contractions of half-precision factors only")]
    NotFusedContraction,
}

/// The grid and block a schedule launches, both one-dimensional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Launch {
    pub blocks: u32,
    pub threads: u32,
}

impl Schedule {
    #[must_use]
    pub const fn kind(self) -> ScheduleKind {
        match self {
            Self::Elements { threads, vector } => {
                ScheduleKind::Thread(ThreadSchedule::Elements { threads, vector })
            }
            Self::Rows {
                threads,
                vector,
                tiles,
            } => ScheduleKind::Thread(ThreadSchedule::Rows {
                threads,
                vector,
                tiles,
            }),
            Self::Warps {
                threads,
                vector,
                lanes,
                tiles,
            } => ScheduleKind::Thread(ThreadSchedule::Warps {
                threads,
                vector,
                lanes,
                tiles,
            }),
            Self::Columns { threads, splits } => {
                ScheduleKind::Thread(ThreadSchedule::Columns { threads, splits })
            }
            Self::Mma { m, n, k } => ScheduleKind::Thread(ThreadSchedule::Mma { m, n, k }),
            Self::Tiles { size } => ScheduleKind::Tile(TileSchedule::Tiles { size }),
            Self::TileRows { width } => ScheduleKind::Tile(TileSchedule::TileRows { width }),
            Self::TileMatmul { m, n, k } => {
                ScheduleKind::Tile(TileSchedule::TileMatmul { m, n, k })
            }
        }
    }

    // The schedule as the CUDA C++ emitter takes it, or `None` for a tile schedule.
    #[must_use]
    pub const fn thread(self) -> Option<ThreadSchedule> {
        match self.kind() {
            ScheduleKind::Thread(schedule) => Some(schedule),
            ScheduleKind::Tile(_) => None,
        }
    }

    // The schedule as the Tile IR emitter takes it, or `None` for a thread schedule.
    #[must_use]
    pub const fn tile(self) -> Option<TileSchedule> {
        match self.kind() {
            ScheduleKind::Tile(schedule) => Some(schedule),
            ScheduleKind::Thread(_) => None,
        }
    }

    /// Returns how many points of two parallel axes each point of this schedule covers: its `tiles`, or `[1, 1]` for schedules without them.
    #[must_use]
    pub const fn tiles(self) -> [u32; 2] {
        match self {
            Self::Rows { tiles, .. } | Self::Warps { tiles, .. } => tiles,
            Self::Elements { .. }
            | Self::Columns { .. }
            | Self::Tiles { .. }
            | Self::TileRows { .. }
            | Self::TileMatmul { .. }
            | Self::Mma { .. } => [1, 1],
        }
    }

    /// Returns this schedule with tiles of one, which targets run a tiled kernel under.
    #[must_use]
    pub const fn untiled(self) -> Self {
        match self {
            Self::Rows {
                threads, vector, ..
            } => Self::Rows {
                threads,
                vector,
                tiles: [1, 1],
            },
            Self::Warps {
                threads,
                vector,
                lanes,
                ..
            } => Self::Warps {
                threads,
                vector,
                lanes,
                tiles: [1, 1],
            },
            other => other,
        }
    }

    /// Checks that the schedule's thread count suits it.
    ///
    /// # Errors
    ///
    /// Returns an error if an `Elements` schedule has no threads or more than 1024 or a vector other than 1, 2, 4, 8, or 16, or a `Rows` or `Warps` schedule's threads are not a multiple of 32 between 32 and 1024.
    pub fn validate(self) -> Result<(), ScheduleError> {
        match self {
            Self::Elements { threads, .. } if threads == 0 || threads > 1024 => {
                Err(ScheduleError::Threads {
                    schedule: "element",
                    requirement: "1 to 1024 threads",
                    threads,
                })
            }
            Self::Elements { vector, .. }
            | Self::Rows { vector, .. }
            | Self::Warps { vector, .. }
                if !matches!(vector, 1 | 2 | 4 | 8 | 16) =>
            {
                Err(ScheduleError::Vector { vector })
            }
            Self::Warps { lanes, .. } if !matches!(lanes, 1 | 2 | 4 | 8 | 16 | 32) => {
                Err(ScheduleError::Lanes { lanes })
            }
            Self::Columns { threads, splits }
                if !matches!(splits, 2 | 4 | 8 | 16 | 32) || splits >= threads =>
            {
                Err(ScheduleError::Splits { splits, threads })
            }
            Self::Rows { tiles, .. } | Self::Warps { tiles, .. }
                if tiles.iter().any(|tile| !matches!(tile, 1 | 2 | 4 | 8 | 16)) =>
            {
                let tile = if matches!(tiles[0], 1 | 2 | 4 | 8 | 16) {
                    tiles[1]
                } else {
                    tiles[0]
                };
                Err(ScheduleError::Tile { tile })
            }
            Self::Rows { threads, .. }
            | Self::Warps { threads, .. }
            | Self::Columns { threads, .. }
                if !(32..=1024).contains(&threads) || threads % 32 != 0 =>
            {
                Err(ScheduleError::Threads {
                    schedule: match self {
                        Self::Rows { .. } => "row",
                        Self::Warps { .. } => "warp",
                        _ => "column",
                    },
                    requirement: "a multiple of 32 threads from 32 to 1024",
                    threads,
                })
            }
            Self::Tiles { size: points } | Self::TileRows { width: points }
                if !points.is_power_of_two() || points > MAX_TILE =>
            {
                Err(ScheduleError::TileSize { points })
            }
            Self::TileMatmul { m, n, k }
                if [m, n, k]
                    .iter()
                    .any(|side| !side.is_power_of_two() || !(16..=256).contains(side)) =>
            {
                Err(ScheduleError::MatmulTile { m, n, k })
            }
            Self::Mma { m, n, k }
                if !matches!(m, 32 | 64 | 128)
                    || !matches!(n, 32 | 64 | 128)
                    || !matches!(k, 32 | 64)
                    // Two stages of both tiles, the weight's in either order, two bytes a value, fit 48 KiB of static shared memory.
                    || 4 * (m * (k + 8) + (k * (n + 8)).max(n * (k + 8))) > 48 * 1024 =>
            {
                Err(ScheduleError::MmaTile { m, n, k })
            }
            _ => Ok(()),
        }
    }

    /// Returns the launch that covers `kernel` with this schedule when its argument sizes take the values in `arguments`, in [`Kernel::argument_sizes`] order.
    ///
    /// A kernel with no work launches no blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if the schedule is invalid, `arguments` has the wrong length, a tile row is narrower than the reduction, or the work needs more blocks than a grid holds.
    pub fn launch(self, kernel: &Kernel, arguments: &[u64]) -> Result<Launch, ScheduleError> {
        self.validate()?;
        if let Some((tiled, arguments)) = parallel_tile::tiled(kernel, self, arguments)? {
            return self.untiled().launch(&tiled, &arguments);
        }

        let sizes = size_values(kernel, arguments)?;
        if let Self::TileMatmul { m, n, .. } = self {
            let contraction = Contraction::of(kernel).ok_or(ScheduleError::NotContraction)?;
            let extent = |axis: AxisId| sizes[kernel.axes[axis.0].extent.0];
            let blocks = contraction
                .batch
                .iter()
                .try_fold(1_u64, |blocks, &axis| blocks.checked_mul(extent(axis)))
                .and_then(|blocks| {
                    blocks.checked_mul(extent(contraction.rows).div_ceil(u64::from(m)))
                })
                .and_then(|blocks| {
                    blocks.checked_mul(extent(contraction.columns).div_ceil(u64::from(n)))
                })
                .ok_or(ScheduleError::GridTooLarge)?;
            return Ok(Launch {
                blocks: u32::try_from(blocks)
                    .ok()
                    .filter(|&blocks| blocks <= i32::MAX.cast_unsigned())
                    .ok_or(ScheduleError::GridTooLarge)?,
                threads: 1,
            });
        }

        if let Self::Mma { m, n, .. } = self {
            let contraction =
                FusedContraction::of(kernel).ok_or(ScheduleError::NotFusedContraction)?;
            let extent = |axis: AxisId| sizes[kernel.axes[axis.0].extent.0];
            let blocks = contraction
                .batch
                .iter()
                .try_fold(1_u64, |blocks, &axis| blocks.checked_mul(extent(axis)))
                .and_then(|blocks| {
                    blocks.checked_mul(extent(contraction.rows).div_ceil(u64::from(m)))
                })
                .and_then(|blocks| {
                    blocks.checked_mul(extent(contraction.columns).div_ceil(u64::from(n)))
                })
                .ok_or(ScheduleError::GridTooLarge)?;
            return Ok(Launch {
                blocks: u32::try_from(blocks)
                    .ok()
                    .filter(|&blocks| blocks <= i32::MAX.cast_unsigned())
                    .ok_or(ScheduleError::GridTooLarge)?,
                threads: MMA_THREADS,
            });
        }

        if let Self::TileRows { width } = self {
            let points = reduction_points(kernel, &sizes);
            if points > u64::from(width) {
                return Err(ScheduleError::TileTooNarrow { width, points });
            }
        }

        let innermost = innermost_parallel_axis(kernel);
        let vector = match self {
            Self::Elements { vector, .. } => u64::from(vector),
            Self::Rows { .. }
            | Self::Warps { .. }
            | Self::Columns { .. }
            | Self::TileRows { .. }
            | Self::TileMatmul { .. }
            | Self::Mma { .. } => 1,
            Self::Tiles { size } => u64::from(size),
        };
        let points = kernel
            .axes
            .iter()
            .enumerate()
            .filter(|(_, axis)| axis.kind == AxisKind::Parallel)
            .try_fold(1_u64, |points, (index, axis)| {
                let extent = sizes[axis.extent.0];
                if Some(AxisId(index)) == innermost {
                    points.checked_mul(extent.div_ceil(vector))
                } else {
                    points.checked_mul(extent)
                }
            })
            .ok_or(ScheduleError::GridTooLarge)?;
        let (blocks, threads) = match self {
            Self::Elements { threads, .. } => (points.div_ceil(u64::from(threads)), threads),
            Self::Rows { threads, .. } => (points, threads),
            Self::Warps { threads, lanes, .. } => {
                (points.div_ceil(u64::from(threads / lanes)), threads)
            }
            Self::Columns { threads, splits } => {
                (points.div_ceil(u64::from(threads / splits)), threads)
            }
            // A tile block is launched as one thread; the tile target sizes the block itself.
            Self::Tiles { .. } | Self::TileRows { .. } | Self::TileMatmul { .. } => (points, 1),
            Self::Mma { .. } => (points, MMA_THREADS),
        };
        Ok(Launch {
            blocks: u32::try_from(blocks)
                .ok()
                .filter(|&blocks| blocks <= i32::MAX.cast_unsigned())
                .ok_or(ScheduleError::GridTooLarge)?,
            threads,
        })
    }
}

/// The most points a tile schedule's tile holds.
pub const MAX_TILE: u32 = 4096;

// The threads of an `Mma` block: four warps, two along its rows by two along its columns.
pub const MMA_THREADS: u32 = 128;

/// Returns the tile schedule for `kernel` when its argument sizes take the values in `arguments`: tiles of up to 1024 points for kernels without reduction axes, and otherwise one tile row as wide as the reduction, rounded up to a power of two.
///
/// # Errors
///
/// Returns an error if `arguments` has the wrong length or the reduction has more than [`MAX_TILE`] points.
pub fn tile_schedule(kernel: &Kernel, arguments: &[u64]) -> Result<Schedule, ScheduleError> {
    let sizes = size_values(kernel, arguments)?;
    if !kernel
        .axes
        .iter()
        .any(|axis| axis.kind == AxisKind::Reduction)
    {
        return Ok(Schedule::Tiles { size: 1024 });
    }

    let points = reduction_points(kernel, &sizes);
    let width = points.max(1).next_power_of_two();
    u32::try_from(width)
        .ok()
        .filter(|&width| width <= MAX_TILE)
        .map(|width| Schedule::TileRows { width })
        .ok_or(ScheduleError::TileTooNarrow {
            width: MAX_TILE,
            points,
        })
}

/// Returns the number of points of `kernel`'s reduction axes, flattened, whose sizes are `sizes`.
fn reduction_points(kernel: &Kernel, sizes: &[u64]) -> u64 {
    kernel
        .axes
        .iter()
        .filter(|axis| axis.kind == AxisKind::Reduction)
        .map(|axis| sizes[axis.extent.0])
        .fold(1, u64::saturating_mul)
}

/// Returns the element schedule with `threads` threads and the widest vector that helps `kernel`.
///
/// The widest operand that the innermost parallel axis walks with a stride of one bounds the vector at 16 bytes.
/// Kernels with reduction axes or combining stores, kernels whose innermost axis no access walks with a stride of one, and kernels with an access that walks it with a stride other than zero or one or through an indexed term get a vector of one.
#[must_use]
pub fn element_schedule(kernel: &Kernel, threads: u32) -> Schedule {
    Schedule::Elements {
        threads,
        vector: vector_width(kernel),
    }
}

pub(crate) fn vector_width(kernel: &Kernel) -> u32 {
    if kernel
        .axes
        .iter()
        .any(|axis| axis.kind == AxisKind::Reduction)
    {
        return 1;
    }

    let Some(innermost) = innermost_parallel_axis(kernel) else {
        return 1;
    };
    let loads = kernel.nodes.iter().filter_map(|node| match &node.expr {
        Expr::Load(access) => Some(access),
        _ => None,
    });
    if kernel.stores.iter().any(|store| store.combiner.is_some()) {
        return 1;
    }

    let mut widest = 0;
    for access in loads.chain(kernel.stores.iter().map(|store| &store.access)) {
        if access
            .indirect
            .iter()
            .any(|&(node, _)| kernel.domain(node).contains(&innermost))
        {
            return 1;
        }
        for &(axis, stride) in &access.strides {
            match kernel.sizes[stride.0] {
                _ if axis != innermost => {}
                Size::Static(0) => {}
                Size::Static(1) => {
                    widest = widest.max(kernel.operands[access.operand.0].element.size_bytes());
                }
                _ => return 1,
            }
        }
    }
    match widest {
        0 => 1,
        bytes => u32::try_from(16 / bytes).expect("an element is at most 16 bytes"),
    }
}

// Whether some load walks `kernel`'s innermost reduction axis contiguously; a kernel whose loads all stride along it reduces outer axes, whose consecutive points are its parallel axes'.
#[must_use]
pub fn reduces_contiguously(kernel: &Kernel) -> bool {
    let Some(innermost) = kernel
        .axes
        .iter()
        .rposition(|axis| axis.kind == AxisKind::Reduction)
        .map(AxisId)
    else {
        return true;
    };
    kernel.nodes.iter().any(|node| match &node.expr {
        Expr::Load(access) => access
            .strides
            .iter()
            .any(|&(axis, stride)| axis == innermost && kernel.sizes[stride.0] == Size::Static(1)),
        _ => false,
    })
}

/// Returns the widest vector, at most `limit`, that the loops over `kernel`'s reduction axes can cover at once: 16 bytes of the widest operand that walks the innermost reduction axis, or one when no access walks it contiguously.
#[must_use]
pub fn reduction_vector(kernel: &Kernel, limit: u32) -> u32 {
    let Some(innermost) = kernel
        .axes
        .iter()
        .rposition(|axis| axis.kind == AxisKind::Reduction)
        .map(AxisId)
    else {
        return 1;
    };
    let loads = kernel.nodes.iter().filter_map(|node| match &node.expr {
        Expr::Load(access) => Some(access),
        _ => None,
    });
    let mut widest = 0;
    for access in loads.chain(kernel.stores.iter().map(|store| &store.access)) {
        for &(axis, stride) in &access.strides {
            if axis == innermost && kernel.sizes[stride.0] == Size::Static(1) {
                widest = widest.max(kernel.operands[access.operand.0].element.size_bytes());
            }
        }
    }

    let vector = match widest {
        0 => return 1,
        bytes => u32::try_from(16 / bytes)
            .expect("an element is at most 16 bytes")
            .min(limit),
    };
    // A vector wider than a static innermost axis covers it whole at several points of the reduction axis before it, when its extent divides the vector, and would otherwise leave lanes idle.
    let extent = match kernel.sizes[kernel.axes[innermost.0].extent.0] {
        Size::Static(extent) => u32::try_from(extent).unwrap_or(u32::MAX),
        Size::Argument => u32::MAX,
    };
    let spans_outer = kernel.axes[..innermost.0]
        .iter()
        .any(|axis| axis.kind == AxisKind::Reduction)
        && extent > 0
        && vector.is_multiple_of(extent);
    if extent >= vector || spans_outer {
        vector
    } else {
        vector.min(if extent == 0 { 1 } else { 1 << extent.ilog2() })
    }
}

/// Returns the last parallel axis, which varies fastest.
pub(crate) fn innermost_parallel_axis(kernel: &Kernel) -> Option<AxisId> {
    kernel
        .axes
        .iter()
        .rposition(|axis| axis.kind == AxisKind::Parallel)
        .map(AxisId)
}

/// Returns the value of every size of `kernel`, taking argument sizes from `arguments` in order.
pub fn size_values(kernel: &Kernel, arguments: &[u64]) -> Result<Vec<u64>, ScheduleError> {
    let expected = kernel.argument_sizes().count();
    if arguments.len() != expected {
        return Err(ScheduleError::LengthMismatch {
            what: "argument size",
            expected,
            actual: arguments.len(),
        });
    }

    let mut arguments = arguments.iter();
    Ok(kernel
        .sizes
        .iter()
        .map(|size| match size {
            Size::Static(value) => *value,
            Size::Argument => *arguments.next().expect("argument count was checked"),
        })
        .collect())
}
