use singe_codegen::emit::KernelSource;
use singe_cuda::{
    architecture::GpuArchitecture,
    dim::Dim3,
    module::{KernelParameters, LaunchConfig},
};

use crate::{
    backend::cuda::codegen::description::KernelDescription,
    error::{Error, Result},
    utility::to_usize,
};

// A generated kernel's code: source text in a language a compiler reads, or a binary intermediate representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KernelCode {
    Source(KernelSource),
    // CUDA Tile IR bytecode, which `tileiras` compiles.
    TileIr(Vec<u8>),
}

impl KernelCode {
    #[must_use]
    pub const fn source(&self) -> Option<&KernelSource> {
        match self {
            Self::Source(source) => Some(source),
            Self::TileIr(_) => None,
        }
    }
}

/// One argument of a generated kernel: a buffer the caller supplies at launch, or a value fixed when the kernel was planned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelArgument {
    /// The address of the caller's input buffer with this index.
    Input(usize),
    /// The address of the caller's output buffer with this index.
    Output(usize),
    /// The address of the status word a checking kernel reports data errors to.
    Status,
    // The address of the kernel's scratch buffer with this index, of the size `KernelAbi::scratch` gives.
    Scratch(usize),
    Bool(bool),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
    // An extent or offset the kernel takes at launch, passed as an unsigned 64-bit integer.
    Size(u64),
}

/// The arguments a generated kernel takes, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelAbi {
    pub arguments: Vec<KernelArgument>,
    // The bytes of each scratch buffer, which launchers allocate zeroed and keep for the kernel alone; a kernel that relies on a buffer starting zeroed leaves it zeroed, so a graph can replay it.
    pub scratch: Vec<u64>,
}

impl KernelAbi {
    /// Returns whether the kernel validates data and reports failures through its status word.
    #[must_use]
    pub fn checks(&self) -> bool {
        self.arguments.contains(&KernelArgument::Status)
    }

    /// Returns the launch arguments for buffers `inputs` and `outputs`, with `status` for a checking kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel names an input or output buffer the caller did not supply.
    pub fn arguments(
        &self,
        inputs: &[*mut u8],
        outputs: &[*mut u8],
        scratch: &[*mut u8],
        status: *mut u8,
    ) -> Result<KernelParameters<'static>> {
        let buffer = |buffers: &[*mut u8], index: usize| {
            buffers
                .get(index)
                .copied()
                .ok_or_else(|| Error::invalid_argument("cuda generated kernel operand count"))
        };
        let mut parameters = KernelParameters::new();
        for &argument in &self.arguments {
            match argument {
                KernelArgument::Input(index) => parameters.owned_arg(buffer(inputs, index)?),
                KernelArgument::Output(index) => parameters.owned_arg(buffer(outputs, index)?),
                KernelArgument::Status => parameters.owned_arg(status),
                KernelArgument::Scratch(index) => parameters.owned_arg(buffer(scratch, index)?),
                KernelArgument::Bool(value) => parameters.owned_arg(value),
                KernelArgument::I32(value) => parameters.owned_arg(value),
                KernelArgument::U32(value) => parameters.owned_arg(value),
                KernelArgument::I64(value) => parameters.owned_arg(value),
                KernelArgument::U64(value) | KernelArgument::Size(value) => {
                    parameters.owned_arg(value)
                }
                KernelArgument::F32(value) => parameters.owned_arg(value),
                KernelArgument::F64(value) => parameters.owned_arg(value),
            };
        }
        Ok(parameters)
    }
}

/// How many blocks of how many threads a kernel launches, in up to three dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelLaunch {
    pub blocks: u64,
    pub threads: u32,
    // The dynamic shared memory each block takes, in bytes; past 48 KiB the runtime raises the kernel's limit when it loads it.
    pub shared_bytes: u32,
    // The grid's extents along its second and third dimensions, `[1, 1]` for a one-dimensional grid of `blocks` blocks.
    pub grid: [u32; 2],
}

/// A kernel ready to compile and launch: its name, its code, its arguments, and how its work is divided.
///
/// The runtime compiles it by its code, so whatever produced the code, a hand-written generator or an emitter, launches the same way.
#[derive(Debug, Clone)]
pub struct KernelArtifact {
    /// The kernel's function name, `singe_` followed by what it computes, as profilers report it.
    pub name: String,
    pub code: KernelCode,
    pub abi: KernelAbi,
    pub launch: KernelLaunch,
    // What the kernel computes and what its arguments hold, as far as its producer knew; it never changes the code.
    pub description: KernelDescription,
    // The architecture the kernel compiles for, or `None` for the device that runs it.
    pub target: Option<GpuArchitecture>,
    pub requirements: CompileRequirements,
}

// What a kernel's code needs of the device it compiles for beyond what every target has.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CompileRequirements {
    // The lowest compute capability whose instructions the code uses, such as 8.0 for half-precision `mma.sync`.
    pub minimum_capability: Option<(u32, u32)>,
    // Whether the code uses instructions of one architecture only, which compile for the `a` variant of the device's target, such as `sm_120a`.
    pub architecture_specific: bool,
}

impl CompileRequirements {
    // The requirements of code that runs `mma.sync` m16n8k16 on `F16` or `BF16` operands.
    pub const TENSOR_CORES: Self = Self {
        minimum_capability: Some((8, 0)),
        architecture_specific: false,
    };

    // The requirements of code that runs Blackwell's `tcgen05`, which only architecture-specific targets assemble.
    pub const TENSOR_ENGINE: Self = Self {
        minimum_capability: Some((10, 0)),
        architecture_specific: true,
    };
}

impl KernelArtifact {
    /// Returns whether the kernel launches no blocks, so it need not run.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.launch.blocks == 0 || self.launch.grid[0] == 0 || self.launch.grid[1] == 0
    }

    /// Returns the launch configuration covering the kernel's work.
    ///
    /// # Errors
    ///
    /// Returns an error if the work does not fit a CUDA grid.
    pub fn launch_config(&self) -> Result<LaunchConfig> {
        let blocks = u32::try_from(self.launch.blocks)
            .ok()
            .filter(|&blocks| blocks <= i32::MAX.cast_unsigned())
            .ok_or_else(|| Error::out_of_range("cuda kernel blocks"))?;
        Ok(LaunchConfig::create(
            Dim3::new(blocks, self.launch.grid[0], self.launch.grid[1]),
            Dim3::new(self.launch.threads, 1, 1),
            to_usize(self.launch.shared_bytes, "cuda kernel shared memory")?,
        )?)
    }
}
