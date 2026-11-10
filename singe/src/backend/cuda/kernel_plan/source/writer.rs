use singe_codegen::emit::cuda_c::SourceWriter;

use crate::{
    backend::{
        cuda::{
            codegen::{
                CompileRequirements, KernelAbi, KernelArgument, KernelArtifact, KernelCode,
                KernelLaunch,
            },
            kernel_plan::source::{KernelInput, KernelPlan, LoadedKernels, describe, kernel_name},
        },
        kernel_lowering::ArgumentLabels,
        registry::Operation,
    },
    data_type::DataType,
    error::{Error, Result},
    utility::to_u32,
    value::ValueSpec,
};

// The threads per block of hand-written element kernels.
const ELEMENT_THREADS: u32 = 128;

// Writes a hand-written kernel: each parameter it declares adds the launch argument that fills it, so the source and the ABI agree by construction.
#[derive(Debug, Default)]
pub struct KernelWriter {
    source: SourceWriter,
    arguments: Vec<KernelArgument>,
    inputs: usize,
    outputs: usize,
    scratch: Vec<u64>,
    labels: ArgumentLabels,
    requirements: CompileRequirements,
    min_blocks: Option<u32>,
}

impl KernelWriter {
    pub fn new() -> Self {
        Self::default()
    }

    // Declares the blocks per SM the kernel's launch bound asks for, which trades registers for blocks that fit together.
    pub const fn min_blocks(&mut self, min_blocks: Option<u32>) {
        self.min_blocks = min_blocks;
    }

    // Declares what the kernel's code needs of the device it compiles for.
    pub fn require(&mut self, requirements: CompileRequirements) {
        self.requirements = requirements;
    }

    // Declares the next input buffer, read as elements of `data_type`, and returns its name.
    pub fn input(&mut self, data_type: DataType, name: &str) -> Result<String> {
        let ty = value_type(data_type)?;
        self.arguments.push(KernelArgument::Input(self.inputs));
        self.inputs += 1;
        Ok(self
            .source
            .parameter(format!("const {ty}* __restrict__"), name))
    }

    // Declares a scratch buffer of `elements` values of `data_type`, which launchers allocate zeroed; a kernel that relies on it starting zeroed must leave it zeroed.
    pub fn scratch(&mut self, data_type: DataType, elements: usize, name: &str) -> Result<String> {
        let ty = value_type(data_type)?;
        let bytes = elements
            .checked_mul(data_type.size_bytes())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| Error::out_of_range("cuda kernel scratch bytes"))?;
        self.arguments
            .push(KernelArgument::Scratch(self.scratch.len()));
        self.scratch.push(bytes);
        Ok(self.source.parameter(format!("{ty}* __restrict__"), name))
    }

    pub fn output(&mut self, data_type: DataType, name: &str) -> Result<String> {
        let ty = value_type(data_type)?;
        self.arguments.push(KernelArgument::Output(self.outputs));
        self.outputs += 1;
        Ok(self.source.parameter(format!("{ty}* __restrict__"), name))
    }

    // Declares the status word the kernel reports data errors to, which is null under stream capture, and returns its name.
    pub fn status(&mut self) -> String {
        self.arguments.push(KernelArgument::Status);
        self.source.parameter("unsigned int*", "status")
    }

    // Declares an unsigned 64-bit value fixed when the kernel was planned, such as a block offset.
    pub fn scalar(&mut self, name: &str, value: u64, label: &str) -> String {
        self.arguments.push(KernelArgument::U64(value));
        self.labels.scalars.push(label.to_owned());
        self.source.parameter("const unsigned long long", name)
    }

    // Declares an extent the kernel takes at launch.
    pub fn size(&mut self, name: &str, value: u64, label: &str) -> String {
        self.arguments.push(KernelArgument::Size(value));
        self.labels.sizes.push(label.to_owned());
        self.source.parameter("const unsigned long long", name)
    }

    pub fn declare(&mut self, code: &str) {
        self.source.declare(code);
    }

    pub fn lines(&mut self, code: &str) {
        self.source.lines(code);
    }

    // Starts a kernel with one thread per element of `count` elements, which the body reads as `index`, and returns its launch.
    pub fn elements(&mut self, count: usize) -> KernelLaunch {
        let count = count as u64;
        self.size("count", count, "element count");
        self.lines(
            "const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
if (index >= count) return;",
        );
        KernelLaunch {
            blocks: count.div_ceil(u64::from(ELEMENT_THREADS)),
            threads: ELEMENT_THREADS,
            shared_bytes: 0,
            grid: [1, 1],
        }
    }

    // Starts a kernel with one block of `threads` threads per row, which the body reads as `row` and `lane` and which may synchronize with `__syncthreads`, and returns its launch.
    pub fn rows(&mut self, rows: usize, threads: u32) -> KernelLaunch {
        self.lines(
            "const unsigned long long row = blockIdx.x;
const unsigned int lane = threadIdx.x;",
        );
        KernelLaunch {
            blocks: rows as u64,
            threads,
            shared_bytes: 0,
            grid: [1, 1],
        }
    }

    // Starts a kernel of `blocks` blocks of `threads` threads, whose body divides the work itself and reads its thread as `lane`, and returns its launch.
    pub fn blocks(&mut self, blocks: usize, threads: u32) -> KernelLaunch {
        self.lines("const unsigned int lane = threadIdx.x;");
        KernelLaunch {
            blocks: blocks as u64,
            threads,
            shared_bytes: 0,
            grid: [1, 1],
        }
    }

    // Starts a kernel of a three-dimensional grid of `blocks` blocks of `threads` threads, whose body reads its block's coordinates from `blockIdx` and its thread as `lane`, and returns its launch.
    pub fn grid(&mut self, blocks: [usize; 3], threads: u32) -> Result<KernelLaunch> {
        // CUDA bounds a grid's second and third extents at 65535.
        let extent = |blocks: usize, limit: u64| {
            u64::try_from(blocks)
                .ok()
                .filter(|&blocks| blocks <= limit)
                .ok_or_else(|| Error::out_of_range("cuda kernel grid"))
        };
        let [x, y, z] = blocks;
        let (y, z) = (extent(y, 65535)?, extent(z, 65535)?);
        self.lines("const unsigned int lane = threadIdx.x;");
        Ok(KernelLaunch {
            blocks: extent(x, u64::from(i32::MAX.cast_unsigned()))?,
            threads,
            shared_bytes: 0,
            grid: [
                to_u32(y, "cuda kernel grid")?,
                to_u32(z, "cuda kernel grid")?,
            ],
        })
    }

    // Finishes the kernel named after `label`, which reads `inputs`, computes `operations`, and writes `outputs`.
    pub fn finish(
        self,
        label: &str,
        launch: KernelLaunch,
        outputs: Vec<ValueSpec>,
        inputs: &[KernelInput],
        operations: &[Operation],
    ) -> KernelPlan {
        let name = kernel_name(label);
        let description = describe(inputs, &outputs, operations, &[], self.labels);
        KernelPlan {
            before: Vec::new(),
            loaded: LoadedKernels::default(),
            outputs,
            kernel: KernelArtifact {
                code: KernelCode::Source(
                    self.source
                        .finish(name.clone())
                        .with_launch_bounds(launch.threads, self.min_blocks),
                ),
                name,
                abi: KernelAbi {
                    arguments: self.arguments,
                    scratch: self.scratch,
                },
                launch,
                description,
                target: None,
                requirements: self.requirements,
            },
        }
    }
}

// The C++ type a buffer of `data_type` holds, by its bits for the floating types the prelude converts.
pub fn value_type(data_type: DataType) -> Result<&'static str> {
    Ok(match data_type {
        DataType::F32 => "float",
        DataType::F64 => "double",
        DataType::Bool | DataType::U8 | DataType::F8E4M3 | DataType::F8E5M2 => "unsigned char",
        DataType::I8 => "signed char",
        DataType::U16 | DataType::F16 | DataType::Bf16 => "unsigned short",
        DataType::I16 => "short",
        DataType::U32 => "unsigned int",
        DataType::I32 => "int",
        DataType::U64 => "unsigned long long",
        DataType::I64 => "long long",
        DataType::F4E2M1 => return Err(Error::unsupported_data_type("cuda kernel buffer")),
    })
}
