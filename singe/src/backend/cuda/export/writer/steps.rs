use std::collections::HashMap;

use crate::{
    backend::cuda::{
        codegen::{
            CompileRequirements, KernelArgument, KernelArtifact, KernelCode, KernelOutput,
            ProgramLaunch,
        },
        export::{
            program::{BufferView, Library, ProgramStep},
            writer::{LibraryCode, Renderer, string_literal},
        },
    },
    error::{Error, Result},
    utility::emit,
};

// What rendering steps collects besides the statements that run them.
#[derive(Default)]
pub struct StepOutput {
    pub units: HashMap<String, String>,
    pub kernels: String,
    pub prelude: Option<&'static str>,
    pub non_finite: bool,
    pub needs: CompileRequirements,
    pub code: Vec<(Library, LibraryCode)>,
    pub calls: usize,
}

impl Renderer<'_> {
    // One kernel's launch: the buffers its launcher takes, in its ABI's order.
    fn launch(
        &self,
        namespace: &str,
        artifact: &KernelArtifact,
        buffers: (&[BufferView], &[BufferView], &[usize], Option<usize>),
    ) -> Result<String> {
        let (inputs, outputs, scratch, status) = buffers;
        let KernelCode::Source(source) = &artifact.code else {
            return Err(Error::unsupported_operation(
                "program kernel without cuda c++ source",
            ));
        };
        let parameters = source.entry().parameters();
        let mut arguments = Vec::new();
        for (position, argument) in artifact.abi.arguments.iter().enumerate() {
            let address = match *argument {
                KernelArgument::Input(index) => self.address(&inputs[index])?,
                KernelArgument::Output(index) => self.address(&outputs[index])?,
                KernelArgument::Scratch(index) => {
                    format!("singe_at(s->buffers[{}], 0ULL)", scratch[index])
                }
                KernelArgument::Status => format!(
                    "singe_at(s->buffers[{}], 0ULL)",
                    status.ok_or_else(|| Error::missing_input("program status word"))?
                ),
                _ => continue,
            };
            let ty = parameters
                .get(position)
                .ok_or_else(|| Error::missing_input("program kernel parameter"))?
                .ty()
                .replace(" __restrict__", "");
            arguments.push(format!("reinterpret_cast<{ty}>({address})"));
        }

        arguments.push("stream".to_owned());
        Ok(format!(
            "    SINGE_CHECK(singe_kernels_{prefix}::{namespace}::launch_{}({}));\n",
            artifact.name,
            arguments.join(", "),
            prefix = self.prefix,
        ))
    }

    pub fn render_steps(&self, steps: &[ProgramStep], out: &mut StepOutput) -> Result<String> {
        let mut body = String::new();
        let mut follows_kernel = false;
        for step in steps {
            let index = out.calls;
            out.calls += 1;
            let launch = match (self.program.dependent_launch, follows_kernel) {
                (false, _) => ProgramLaunch::Serial,
                (true, false) => ProgramLaunch::Dependent,
                (true, true) => ProgramLaunch::DependentOnKernel,
            };
            follows_kernel = matches!(step, ProgramStep::Kernel { .. });
            let label = string_literal(&step_label(index, step));
            emit!(body, "    singe_step = {label};");
            match step {
                ProgramStep::Kernel {
                    artifact,
                    inputs,
                    outputs,
                    scratch,
                    status,
                } => {
                    let namespace = kernel_namespace(artifact, launch, out)?;
                    let scratch = scratch.iter().map(|id| id.0).collect::<Vec<_>>();
                    body.push_str(&self.launch(
                        &namespace,
                        artifact,
                        (inputs, outputs, &scratch, status.map(|id| id.0)),
                    )?);
                }
                ProgramStep::Copy {
                    source,
                    destination,
                } => {
                    let bytes = source.layout.element_count() * source.data_type.size_bytes();
                    emit!(
                        body,
                        "    SINGE_CHECK(cudaMemcpyAsync({}, {}, {bytes}ULL, cudaMemcpyDeviceToDevice, stream));",
                        self.address(destination)?,
                        self.address(source)?
                    );
                }
                ProgramStep::Fill { destination } => {
                    let bytes =
                        destination.layout.element_count() * destination.data_type.size_bytes();
                    emit!(
                        body,
                        "    SINGE_CHECK(cudaMemsetAsync({}, 0, {bytes}ULL, stream));",
                        self.address(destination)?
                    );
                }
                ProgramStep::Library { call, .. } => {
                    let mut library = self.library(index, call)?;
                    body.push_str(&library.run);
                    if !library.create.is_empty() {
                        library
                            .create
                            .insert_str(0, &format!("    singe_step = {label};\n"));
                    }

                    out.code.push((call.library(), library));
                }
            }
        }
        Ok(body)
    }
}

// What a failure in `step` names it by.
fn step_label(index: usize, step: &ProgramStep) -> String {
    match step {
        ProgramStep::Kernel { artifact, .. } => format!("step {index}, kernel {}", artifact.name),
        ProgramStep::Copy { .. } => format!("step {index}, a copy"),
        ProgramStep::Fill { .. } => format!("step {index}, a fill"),
        ProgramStep::Library { call, operations } => {
            format!("step {index}, {} for {operations}", call.library())
        }
    }
}

// The namespace of `artifact`'s definition, writing it the first time.
// Identical kernels share one definition, and each definition has a namespace of its own, so kernels may share a name.
fn kernel_namespace(
    artifact: &KernelArtifact,
    launch: ProgramLaunch,
    out: &mut StepOutput,
) -> Result<String> {
    out.needs.architecture_specific |= artifact.requirements.architecture_specific;
    if let KernelCode::Source(source) = &artifact.code {
        out.prelude.get_or_insert_with(|| source.prelude());
    }

    let unit = artifact.render(KernelOutput::ProgramUnit(launch))?;
    if let Some(namespace) = out.units.get(&unit) {
        return Ok(namespace.clone());
    }

    let namespace = format!("{}_{}", artifact.name, out.units.len());
    out.non_finite |= artifact.launcher_needs_limits()?;
    emit!(
        out.kernels,
        "namespace {namespace} {{\n{unit}}}  // namespace {namespace}\n"
    );
    out.units.insert(unit, namespace.clone());
    Ok(namespace)
}
