//! Generated kernels from the kernel IR: which language each is emitted in, and the arguments and launch it takes.

use singe_codegen::{
    abi::{Bindings, Parameter, parameters},
    emit::EmitError,
    ir::{ElementType, Kernel, Literal, OperandRole},
    schedule::{Schedule, ScheduleKind, parallel_tile, tile_schedule},
};

use crate::{
    backend::cuda::{
        codegen::{
            cuda_c,
            description::KernelDescription,
            kernel::{
                CompileRequirements, KernelAbi, KernelArgument, KernelArtifact, KernelCode,
                KernelLaunch,
            },
            planning::{KernelDestination, PlanningContext},
            tile,
        },
        tuning::{TunedChoice, schedule_key},
    },
    error::{CompileError, Error, Result},
    utility::{to_i32, to_u32, to_u64},
};

/// The code generated kernels are emitted as when their lowering chose a thread schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelLanguage {
    /// CUDA C++, compiled by NVRTC.
    #[default]
    CudaC,
    /// CUDA Tile IR, compiled by `tileiras`, for the kernels its emitter supports.
    TileIr,
}

/// Returns `kernel` planned under the schedule its lowering chose, `schedule`, with its scalars and argument sizes bound to `bindings`.
///
/// A schedule recorded by kernel tuning replaces `schedule`; when `context` asks for tuning, a kernel with no record is tuned first, and `context`'s recorder collects the kernel.
/// Otherwise, when `context` asks for [`KernelLanguage::TileIr`], a thread schedule is replaced by the tile schedule [`tile_schedule`] picks, and kernels the tile emitter cannot write keep CUDA C++ under `schedule`.
///
/// # Errors
///
/// Returns an error if the kernel, schedule, or bindings are invalid, or the emitter does not support the kernel.
pub fn plan_kernel(
    kernel: &Kernel,
    schedule: Schedule,
    bindings: &Bindings,
    description: &KernelDescription,
    context: &PlanningContext,
) -> Result<KernelArtifact> {
    if let Some(recorder) = &context.recorder {
        recorder.note(kernel, schedule, bindings, description, context.target)?;
    }

    let targeted = |artifact: KernelArtifact| KernelArtifact {
        target: context.target,
        ..artifact
    };
    let program = context.destination == KernelDestination::Program;
    let key = schedule_key(kernel, bindings);
    if let Some(records) = context.records()
        && let Some(TunedChoice::Schedule(recorded)) = records.get(&key)
    {
        // A schedule a record names that no longer emits is an error, since emitting another would quietly run what tuning did not choose.
        if !(program && recorded.thread().is_none())
            && let Ok(generated) = emit_kernel(kernel, recorded, bindings, description)
        {
            records.note_used(&key);
            return Ok(targeted(generated));
        }

        return Err(Error::StaleRecord { key });
    }

    // `tileiras` compiles for `sm_80` and later.
    let tile = context.language == KernelLanguage::TileIr
        && context
            .target
            .is_none_or(|target| target.compute_capability() >= (8, 0));
    if matches!(
        schedule,
        Schedule::Elements { .. }
            | Schedule::Rows { .. }
            | Schedule::Warps { .. }
            | Schedule::Columns { .. }
    ) && tile
    {
        let tiled = tile_schedule(kernel, &bindings.sizes)
            .map_err(Error::from)
            .and_then(|tiled| emit_kernel(kernel, tiled, bindings, description));
        match tiled {
            Ok(tiled) => return Ok(targeted(tiled)),
            // Kernels the tile emitter cannot write, or rows too wide for one tile, stay CUDA C++.
            Err(Error::Compile(
                CompileError::Emit(EmitError::Unsupported { .. }) | CompileError::Schedule(_),
            )) => {}
            Err(error) => return Err(error),
        }
    }
    emit_kernel(kernel, schedule, bindings, description).map(targeted)
}

/// Returns `kernel` emitted under exactly `schedule`, as Tile IR for a tile schedule and CUDA C++ otherwise, with its scalars and argument sizes bound to `bindings`.
///
/// Inputs and outputs are numbered separately, each in operand order, so the caller passes input and output buffers as the kernel's operands list them.
/// A schedule with a parallel tile emits the kernel [`parallel_tile::tiled`] returns, which reads and writes the same operands.
///
/// # Errors
///
/// Returns an error if the kernel, schedule, or bindings are invalid, or the emitter does not support the kernel.
pub fn emit_kernel(
    kernel: &Kernel,
    schedule: Schedule,
    bindings: &Bindings,
    description: &KernelDescription,
) -> Result<KernelArtifact> {
    // A parallel tile runs as the tiled kernel, under the schedule without its tile.
    if let Some((tiled, sizes)) = parallel_tile::tiled(kernel, schedule, &bindings.sizes)? {
        let bindings = Bindings {
            sizes,
            scalars: bindings.scalars.clone(),
        };
        // The tiled kernel takes other sizes, which the description's labels do not name.
        let description = KernelDescription {
            sizes: Vec::new(),
            ..description.clone()
        };
        return emit_kernel(&tiled, schedule.untiled(), &bindings, &description);
    }

    let code = match schedule.kind() {
        ScheduleKind::Tile(schedule) => tile::emit(kernel, schedule)?,
        ScheduleKind::Thread(schedule) => cuda_c::emit(kernel, schedule, &bindings.sizes)?,
    };
    let launch = schedule.launch(kernel, &bindings.sizes)?;
    let code = match code {
        KernelCode::Source(source) => {
            KernelCode::Source(source.with_launch_bounds(launch.threads, None))
        }
        code @ KernelCode::TileIr(_) => code,
    };
    let mut inputs = 0;
    let mut outputs = 0;
    let mut sizes = bindings.sizes.iter();
    let arguments = parameters(kernel)
        .into_iter()
        .map(|parameter| {
            Ok(match parameter {
                Parameter::Operand(operand) => match kernel.operands[operand.0].role {
                    OperandRole::Input => {
                        inputs += 1;
                        KernelArgument::Input(inputs - 1)
                    }
                    OperandRole::Output => {
                        outputs += 1;
                        KernelArgument::Output(outputs - 1)
                    }
                },
                Parameter::Status => KernelArgument::Status,
                Parameter::Scalar(scalar, element) => {
                    let literal = *bindings
                        .scalars
                        .get(scalar.0)
                        .ok_or_else(|| Error::invalid_argument("kernel scalar binding"))?;
                    scalar_argument(element, literal)?
                }
                Parameter::Size(_) => KernelArgument::Size(
                    *sizes
                        .next()
                        .ok_or_else(|| Error::invalid_argument("kernel size binding"))?,
                ),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(KernelArtifact {
        name: kernel.name.clone(),
        code,
        abi: KernelAbi {
            arguments,
            scratch: Vec::new(),
        },
        launch: KernelLaunch {
            blocks: u64::from(launch.blocks),
            threads: launch.threads,
            shared_bytes: 0,
            grid: [1, 1],
        },
        description: description.clone(),
        target: None,
        requirements: if matches!(schedule, Schedule::Mma { .. }) {
            CompileRequirements::TENSOR_CORES
        } else {
            CompileRequirements::default()
        },
    })
}

/// Returns `literal` as the argument a scalar of `element` passes.
#[expect(
    clippy::cast_possible_truncation,
    reason = "an `F32` scalar is its value rounded to `F32`"
)]
fn scalar_argument(element: ElementType, literal: Literal) -> Result<KernelArgument> {
    Ok(match (element, literal) {
        (ElementType::Bool, Literal::Bool(value)) => KernelArgument::Bool(value),
        (ElementType::F32, Literal::Float(value)) => KernelArgument::F32(value as f32),
        (ElementType::F64, Literal::Float(value)) => KernelArgument::F64(value),
        (ElementType::I32, Literal::Int(value)) => {
            KernelArgument::I32(to_i32(value, "kernel scalar")?)
        }
        (ElementType::U32, Literal::Int(value)) => {
            KernelArgument::U32(to_u32(value, "kernel scalar")?)
        }
        (ElementType::I64, Literal::Int(value)) => KernelArgument::I64(value),
        (ElementType::U64, Literal::Int(value)) => {
            KernelArgument::U64(to_u64(value, "kernel scalar")?)
        }
        _ => return Err(Error::unsupported_data_type("kernel scalar")),
    })
}
