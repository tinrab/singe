use std::iter;

use singe_codegen::emit::{EntryLinkage, KernelSource, SourceLanguage, SourceParameter};

use crate::{
    backend::cuda::codegen::{
        description::{BufferDescription, KernelDescription},
        kernel::{KernelArgument, KernelArtifact, KernelCode, KernelLaunch},
    },
    error::{Error, Result},
};

// Who reads a kernel's text.
// The generated computation is the same for each; they differ in what the text brings with it and how the kernel is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelOutput {
    // What NVRTC compiles for Singe to launch: the helpers and the kernel, found by its C name.
    Nvrtc,
    // The kernel alone, for reading.
    Listing { documentation: bool },
    // A `.cu` file of its own for `nvcc`: the runtime include, the helpers, the kernel, and an inline `launch_<name>` that launches it with its planned grid and values.
    File { documentation: bool },
    // One kernel of a program's translation unit, which holds the includes and helpers and puts the kernel and its launcher in a namespace, so kernels may share names.
    ProgramUnit(ProgramLaunch),
}

// How a program launches a kernel on its stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramLaunch {
    // One kernel after another, each starting once the one before has finished.
    Serial,
    // The kernel waits for what ran before it, and lets what follows start, but starts in turn itself.
    Dependent,
    // As `Dependent`, and the kernel may start before the kernel it follows has finished.
    DependentOnKernel,
}

impl KernelArtifact {
    pub fn render(&self, output: KernelOutput) -> Result<String> {
        let source = self.cuda_source()?;
        let (prelude, documentation, launches, includes, linkage) = match output {
            KernelOutput::Nvrtc => (true, false, false, false, EntryLinkage::C),
            KernelOutput::Listing { documentation } => {
                (false, documentation, false, false, EntryLinkage::C)
            }
            KernelOutput::File { documentation } => {
                (true, documentation, true, true, EntryLinkage::C)
            }
            KernelOutput::ProgramUnit(_) => (false, false, true, false, EntryLinkage::Cpp),
        };
        let parameters = source.entry().parameters();
        if (documentation || launches) && parameters.len() != self.abi.arguments.len() {
            return Err(Error::length_mismatch(
                "cuda kernel parameters",
                self.abi.arguments.len(),
                parameters.len(),
            ));
        }

        let arguments = describe_arguments(self, parameters);
        let mut out = String::new();
        if includes {
            out.push_str("#include <cuda_runtime.h>\n");
            if arguments.iter().any(|argument| argument.non_finite) {
                out.push_str("#include <limits>\n");
            }

            out.push('\n');
        }

        if prelude {
            push_piece(&mut out, source.prelude());
        }

        push_piece(&mut out, source.declarations());
        if documentation {
            push_piece(&mut out, &kernel_documentation(self, &arguments));
        }

        let launch = match output {
            KernelOutput::ProgramUnit(launch) => launch,
            _ => ProgramLaunch::Serial,
        };
        let definition = if launch == ProgramLaunch::Serial {
            source.render_definition(linkage)
        } else {
            source.render_dependent_definition(linkage)
        };
        push_piece(&mut out, &definition);
        if launches {
            if includes {
                out.push('\n');
            }

            if documentation {
                out.push_str(&launcher_documentation(self, &arguments));
            }

            out.push_str(&launcher(self, source, &arguments, launch));
        }
        Ok(out)
    }

    // Whether the kernel's launcher passes a value only `<limits>` can spell.
    pub fn launcher_needs_limits(&self) -> Result<bool> {
        let parameters = self.cuda_source()?.entry().parameters();
        Ok(describe_arguments(self, parameters)
            .iter()
            .any(|argument| argument.non_finite))
    }

    fn cuda_source(&self) -> Result<&KernelSource> {
        match &self.code {
            KernelCode::Source(source) if source.language() == SourceLanguage::CudaC => Ok(source),
            _ => Err(Error::unsupported_operation_with_details(
                "cuda c++ rendering",
                format!("kernel `{}` has no cuda c++ source", self.name),
            )),
        }
    }
}

// One kernel argument as the renderer writes it: its parameter, and for a value fixed at planning its C++ literal.
struct Argument<'a> {
    parameter: Option<&'a SourceParameter>,
    literal: Option<String>,
    documentation: String,
    non_finite: bool,
}

fn describe_arguments<'a>(
    artifact: &KernelArtifact,
    parameters: &'a [SourceParameter],
) -> Vec<Argument<'a>> {
    let description = &artifact.description;
    let (mut scalars, mut sizes) = (0, 0);
    artifact
        .abi
        .arguments
        .iter()
        .enumerate()
        .map(|(index, &argument)| {
            let documentation = match argument {
                KernelArgument::Input(input) => {
                    buffer_documentation("input", input, description.inputs.get(input))
                }
                KernelArgument::Output(output) => {
                    buffer_documentation("output", output, description.outputs.get(output))
                }
                KernelArgument::Status => "status word: a zeroed `unsigned int`, or null, into which the kernel writes the first status code it reports".to_owned(),
                KernelArgument::Scratch(index) => format!(
                    "scratch buffer {index}: {} zeroed bytes the kernel alone uses, which it leaves zeroed",
                    artifact.abi.scratch.get(index).copied().unwrap_or(0)
                ),
                KernelArgument::Size(_) => {
                    sizes += 1;
                    labeled("size", sizes - 1, &description.sizes)
                }
                _ => {
                    scalars += 1;
                    labeled("scalar", scalars - 1, &description.scalars)
                }
            };
            let literal = literal(argument);
            let documentation = match &literal {
                Some(literal) => format!("{documentation}, planned as `{literal}`"),
                None => documentation,
            };
            Argument {
                parameter: parameters.get(index),
                literal,
                documentation,
                non_finite: match argument {
                    KernelArgument::F32(value) => !value.is_finite(),
                    KernelArgument::F64(value) => !value.is_finite(),
                    _ => false,
                },
            }
        })
        .collect()
}

fn buffer_documentation(role: &str, index: usize, buffer: Option<&BufferDescription>) -> String {
    let Some(buffer) = buffer else {
        return format!("{role} {index}");
    };
    let name = buffer
        .name
        .as_ref()
        .map_or_else(String::new, |name| format!(" `{name}`"));
    let producer = buffer
        .producer
        .as_ref()
        .map_or_else(String::new, |producer| format!(", the {producer} result"));
    format!(
        "{role} {index}{name}{producer}: {} {:?}, strides {:?}",
        buffer.data_type,
        buffer.layout.dimensions(),
        buffer.layout.strides()
    )
}

fn labeled(kind: &str, index: usize, labels: &[String]) -> String {
    labels.get(index).map_or_else(
        || format!("{kind} {index}"),
        |label| format!("{kind} {index}, {label}"),
    )
}

fn kernel_documentation(artifact: &KernelArtifact, arguments: &[Argument<'_>]) -> String {
    let target = artifact.target.map_or_else(String::new, |target| {
        format!(" for `{target}` (`nvcc -arch={target}`)")
    });
    let launch = format!(
        "/// Planned{target} to launch as {} blocks of {} threads.\n///\n",
        blocks_text(artifact.launch),
        artifact.launch.threads
    );
    [summary(&artifact.name, &artifact.description), launch]
        .into_iter()
        .chain(arguments.iter().filter_map(parameter_documentation))
        .collect()
}

fn launcher_documentation(artifact: &KernelArtifact, arguments: &[Argument<'_>]) -> String {
    let summary = format!(
        "/// Launches `{}` on `stream` as {} blocks of {} threads with the values it was planned with, and returns the launch error.\n///\n",
        artifact.name,
        blocks_text(artifact.launch),
        artifact.launch.threads
    );
    iter::once(summary)
        .chain(
            arguments
                .iter()
                .filter(|argument| argument.literal.is_none())
                .filter_map(parameter_documentation),
        )
        .chain(["/// @param stream the stream the kernel runs on\n".to_owned()])
        .collect()
}

fn parameter_documentation(argument: &Argument<'_>) -> Option<String> {
    argument.parameter.map(|parameter| {
        format!(
            "/// @param {} {}\n",
            parameter.name(),
            argument.documentation
        )
    })
}

fn summary(name: &str, description: &KernelDescription) -> String {
    if description.operations.is_empty() {
        format!("/// `{name}`.\n")
    } else {
        format!(
            "/// `{name}` computes {}.\n",
            description.operations.join(", ")
        )
    }
}

fn launcher(
    artifact: &KernelArtifact,
    source: &KernelSource,
    arguments: &[Argument<'_>],
    launch: ProgramLaunch,
) -> String {
    let mut parameters = arguments
        .iter()
        .filter(|argument| argument.literal.is_none())
        .filter_map(|argument| argument.parameter.map(ToString::to_string))
        .collect::<Vec<_>>();
    parameters.push("cudaStream_t stream = 0".to_owned());
    let body = if artifact.is_empty() {
        "    return cudaSuccess;\n".to_owned()
    } else {
        let values = arguments
            .iter()
            .filter_map(|argument| {
                argument.literal.clone().or_else(|| {
                    argument
                        .parameter
                        .map(|parameter| parameter.name().to_owned())
                })
            })
            .collect::<Vec<_>>();
        let shared = artifact.launch.shared_bytes;
        // More than 48 KiB of dynamic shared memory needs the kernel's limit raised first.
        let limit = if shared > 48 * 1024 {
            format!(
                "    const cudaError_t limit = cudaFuncSetAttribute({}, cudaFuncAttributeMaxDynamicSharedMemorySize, {shared});\n    if (limit != cudaSuccess) return limit;\n",
                source.entry().name()
            )
        } else {
            String::new()
        };
        if launch == ProgramLaunch::DependentOnKernel {
            // The runtime leaves ordering this kernel against the one before to its `griddepcontrol.wait`.
            format!(
                "{limit}    cudaLaunchAttribute attribute;\n    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;\n    attribute.val.programmaticStreamSerializationAllowed = 1;\n    cudaLaunchConfig_t config;\n    config.gridDim = dim3({grid});\n    config.blockDim = dim3({threads});\n    config.dynamicSmemBytes = {shared};\n    config.stream = stream;\n    config.numAttrs = 1;\n    config.attrs = &attribute;\n    return cudaLaunchKernelEx(&config, {name}, {values});\n",
                grid = grid_expression(artifact.launch),
                threads = artifact.launch.threads,
                name = source.entry().name(),
                values = values.join(", ")
            )
        } else {
            format!(
                "{limit}    {}<<<{}, {}, {shared}, stream>>>({});\n    return cudaGetLastError();\n",
                source.entry().name(),
                grid_expression(artifact.launch),
                artifact.launch.threads,
                values.join(", ")
            )
        }
    };
    format!(
        "inline cudaError_t launch_{}({}) {{\n{body}}}\n",
        artifact.name,
        parameters.join(", ")
    )
}

// Each nonempty piece starts on a line of its own.
fn push_piece(out: &mut String, piece: &str) {
    if piece.is_empty() {
        return;
    }

    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }

    out.push_str(piece);
}

// Buffers have no literal: the caller passes them.
fn literal(argument: KernelArgument) -> Option<String> {
    Some(match argument {
        KernelArgument::Input(_)
        | KernelArgument::Output(_)
        | KernelArgument::Status
        | KernelArgument::Scratch(_) => {
            return None;
        }
        KernelArgument::Bool(value) => value.to_string(),
        // C++ has no negative literals, and negating the literal of the minimum's magnitude overflows its type.
        KernelArgument::I32(i32::MIN) => "(-2147483647 - 1)".to_owned(),
        KernelArgument::I32(value) => value.to_string(),
        KernelArgument::U32(value) => format!("{value}U"),
        KernelArgument::I64(i64::MIN) => "(-9223372036854775807LL - 1)".to_owned(),
        KernelArgument::I64(value) => format!("{value}LL"),
        KernelArgument::U64(value) | KernelArgument::Size(value) => format!("{value}ULL"),
        KernelArgument::F32(value) if value.is_finite() => format!("{value:?}f"),
        KernelArgument::F32(value) => non_finite_literal("float", value.is_nan(), value > 0.0),
        KernelArgument::F64(value) if value.is_finite() => format!("{value:?}"),
        KernelArgument::F64(value) => non_finite_literal("double", value.is_nan(), value > 0.0),
    })
}

fn non_finite_literal(ty: &str, nan: bool, positive: bool) -> String {
    let limits = format!("std::numeric_limits<{ty}>");
    if nan {
        format!("{limits}::quiet_NaN()")
    } else if positive {
        format!("{limits}::infinity()")
    } else {
        format!("-{limits}::infinity()")
    }
}

// The launch's blocks as the documentation counts them: a number, or the three extents of a larger grid.
fn blocks_text(launch: KernelLaunch) -> String {
    match launch.grid {
        [1, 1] => launch.blocks.to_string(),
        [y, z] => format!("{} by {y} by {z}", launch.blocks),
    }
}

// The launch's grid as a launch expression takes it.
fn grid_expression(launch: KernelLaunch) -> String {
    match launch.grid {
        [1, 1] => launch.blocks.to_string(),
        [y, z] => format!("dim3({}, {y}, {z})", launch.blocks),
    }
}

#[cfg(test)]
mod tests;
