// Whole graphs as CUDA C++ programs, whose steps are generated kernels and library calls, as far as the allowed providers permit.
//
// What Singe links and what a program calls are separate decisions: planning reads only the graph, so this module needs no device and no library, and the user's own compile links what the program calls.
// Each step has one implementation, decided here; a library that refuses it makes the program's `create` function fail, naming the library, the call, and the step.

mod builder;
pub mod compiled;
mod program;
mod writer;

use std::sync::Arc;

use singe_cuda::architecture::GpuArchitecture;

use self::builder::build_program;
pub use self::{
    program::{
        Activation, BufferId, BufferKind, BufferView, CudaProgram, CudaProgramSet, GemmOperand,
        Library, LibraryCall, OperationProvider, ProgramBuffer, ProgramEntry, ProgramState,
        ProgramStep, ProgramValue, StateKind, StepProvider,
    },
    writer::{ProgramFiles, set::ProgramSetFiles},
};
use crate::{
    backend::{
        cuda::{
            codegen::{KernelDestination, KernelLanguage, KernelRecorder, PlanningContext},
            compiled::{
                build::prepare::channels_last_convolution_chains,
                plan::lowering::{CudaGraphProvider, LibraryFeature, LoweringPolicy, ModuleKernel},
            },
            kernel_plan::KernelPlanner,
            tuning::{TunedChoice, TuningRecords, dependent_launch_key},
        },
        registry::Operation,
    },
    device::Device,
    error::{Error, Result},
    graph::{Graph, GraphCompileOptions},
    nn::PageLayout,
};

// Exports graphs as CUDA C++ programs for one architecture.
#[derive(Debug, Clone)]
pub struct CudaExport {
    target: GpuArchitecture,
    libraries: Vec<LibraryFeature>,
    modules: Vec<ModuleKernel>,
    fusion: bool,
    // What a tuning candidate's program may run, in place of the lists above.
    policy: Option<LoweringPolicy>,
    state_functions: bool,
    records: Option<Arc<TuningRecords>>,
    library_table: bool,
    decoding: Option<PageLayout>,
}

impl CudaExport {
    // Exports for `target`, which need not be on this machine, with every library allowed and fusion on.
    #[must_use]
    pub fn new(target: GpuArchitecture) -> Self {
        Self {
            target,
            libraries: LibraryFeature::ALL.to_vec(),
            modules: vec![ModuleKernel::All],
            fusion: true,
            policy: None,
            state_functions: false,
            records: None,
            library_table: false,
            decoding: None,
        }
    }

    // Writes, beside the program, a loop that generates through it one token at a time, for a decoder step graph whose paged cache `layout` describes, as `DecoderStepGraph::layout` gives it.
    // The loop is the way to run an exported model with nothing of Singe: `<prefix>_generate.cu`, which the manifest says how to build.
    #[must_use]
    pub const fn with_decoding(mut self, layout: PageLayout) -> Self {
        self.decoding = Some(layout);
        self
    }

    // Writes what records force on libraries as places in a table given at creation, so programs that differ only in those share one build.
    // Only tuning reads programs this way, and what it keeps ships as literals.
    #[must_use]
    pub const fn with_library_table(mut self, library_table: bool) -> Self {
        self.library_table = library_table;
        self
    }

    // Allows the program to call only `libraries`, each for what it names, whatever is on this machine.
    // An allowed library takes every operation it can run, and everything else becomes generated kernels and their decompositions.
    // A `Library` or a feature enum's `All` is the whole library, repeating a library adds to what it may run, and the default is every library.
    #[must_use]
    pub fn with_libraries<L: Into<LibraryFeature>>(
        mut self,
        libraries: impl IntoIterator<Item = L>,
    ) -> Self {
        self.libraries = libraries.into_iter().map(Into::into).collect();
        self
    }

    // Allows the program no library, so it is pure CUDA.
    #[must_use]
    pub fn without_libraries(mut self) -> Self {
        self.libraries.clear();
        self
    }

    // Lets the recognizer choose only `modules`, and with none every node takes its generic route.
    #[must_use]
    pub fn with_modules(mut self, modules: impl IntoIterator<Item = ModuleKernel>) -> Self {
        self.modules = modules.into_iter().collect();
        self
    }

    // Allows the program what `policy` allows, so it runs as a graph under that policy would, as the tuner's candidates must.
    #[must_use]
    pub fn with_lowering_policy(mut self, policy: LoweringPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    // Chooses whether generated kernels may merge neighbouring operations, which they do by default.
    #[must_use]
    pub const fn with_fusion(mut self, fusion: bool) -> Self {
        self.fusion = fusion;
        self
    }

    // Chooses whether the program has a function pair that reads and replaces each tensor it holds, which it does not by default.
    // A model has hundreds of such tensors, whose values `load_weights` sets together; a generator's functions are always there, since they set its seed.
    #[must_use]
    pub const fn with_state_functions(mut self, state_functions: bool) -> Self {
        self.state_functions = state_functions;
        self
    }

    // The providers lowering's rules read for what this export allows.
    fn policy(&self) -> LoweringPolicy {
        if let Some(policy) = &self.policy {
            return policy.clone();
        }

        let mut providers = vec![CudaGraphProvider::Alias, CudaGraphProvider::Kernel];
        if self.fusion {
            providers.push(CudaGraphProvider::Fusion);
        }
        LoweringPolicy::new(providers)
            .with_libraries(self.libraries.iter().copied())
            .with_modules(self.modules.iter().copied())
    }

    // Reads what `tune_graph` chose, which holds only for the architecture it measured; a program contains only the winners.
    #[must_use]
    pub fn with_tuning_records(mut self, records: TuningRecords) -> Self {
        self.records = Some(Arc::new(records));
        self
    }

    // Writes several graphs of one model as one program, each under a name C++ accepts, whose tensors in common exist once.
    // Graphs that hold different values under one state name are not one model, and generators advance per step, so neither can be shared.
    // With a page layout, the set's generation loop decodes through the step named `decode`, after running the step named `encode` once if there is one, and steps' inputs of one name are one buffer, so an encode step can write the caches the decoder reads.
    //
    // # Errors
    //
    // Returns an error for such graphs, a name C++ does not accept, or what `export` returns.
    pub fn export_steps(&self, graphs: &[(&str, &Graph)]) -> Result<CudaProgramSet> {
        if self.state_functions {
            return Err(Error::invalid_argument(
                "a program of several steps has no functions for its state",
            ));
        }

        let mut entries: Vec<ProgramEntry> = Vec::with_capacity(graphs.len());
        for (name, graph) in graphs {
            let mut program = self.export(graph)?;
            program.decoding = None;
            let mut shared = Vec::with_capacity(program.state.len());
            for state in &program.state {
                if matches!(state.kind, StateKind::Generator { .. }) {
                    return Err(Error::invalid_argument(
                        "a program of several steps cannot share a generator",
                    ));
                }

                // State is one tensor under one name in every step, and a step that differs is not of this model.
                // A constant's name may be only its place in one step's graph, so what it holds says whether it is shared.
                let same_holdings = |other: &ProgramState| {
                    other.values == state.values
                        && other.data_type == state.data_type
                        && other.layout == state.layout
                };
                let mut owner = None;
                for (index, entry) in entries.iter().enumerate() {
                    let position = if matches!(state.kind, StateKind::Constant) {
                        entry.program.state.iter().position(|other| {
                            matches!(other.kind, StateKind::Constant) && same_holdings(other)
                        })
                    } else {
                        let position = entry
                            .program
                            .state
                            .iter()
                            .position(|other| other.graph_name == state.graph_name);
                        if position.is_some_and(|at| !same_holdings(&entry.program.state[at])) {
                            return Err(Error::invalid_argument(
                                "steps of one program hold different values under one state name",
                            ));
                        }

                        position
                    };
                    if let Some(position) = position {
                        owner = Some(entry.shared[position].unwrap_or((index, position)));
                        break;
                    }
                }

                shared.push(owner);
            }

            entries.push(ProgramEntry {
                name: (*name).to_owned(),
                program,
                shared,
            });
        }

        if self.decoding.is_some() && !entries.iter().any(|entry| entry.name == "decode") {
            return Err(Error::invalid_argument(
                "a program of several steps generates through a step named `decode`",
            ));
        }
        Ok(CudaProgramSet {
            entries,
            decoding: self.decoding,
        })
    }

    // Writes `graph` as a program.
    //
    // Whether a library is on this machine has no bearing on this, since the program only names its calls.
    // An operation only a library outside the allowed providers runs whole runs as its decomposition, and the program records that it does.
    //
    // # Errors
    //
    // Returns `Error::Export` naming every operation no allowed provider and no decomposition can run, or another error if planning fails.
    pub fn export(&self, graph: &Graph) -> Result<CudaProgram> {
        // Planning never touches a device, so constants stay on the host.
        let graph = graph.placed_on(Device::Cpu)?;
        let policy = self.policy();
        let merged = self
            .records
            .as_ref()
            .is_some_and(|records| records.merges_projections());
        let compiled = graph.compile_with_options(
            GraphCompileOptions::new()
                .with_cuda_providers(policy.graph_providers())
                .with_merged_projections(merged),
        )?;
        let ir = if policy.allows(CudaGraphProvider::CudnnBackend, Operation::Conv2d) {
            channels_last_convolution_chains(compiled.shared_ir())?
        } else {
            Arc::clone(&compiled.shared_ir())
        };
        let recorder = KernelRecorder::new();
        let context = PlanningContext {
            language: KernelLanguage::CudaC,
            records: self.records.clone(),
            recorder: Some(recorder.clone()),
            generated_convolutions: true,
            destination: KernelDestination::Program,
            target: Some(self.target),
        };
        context.check_records()?;
        let planner = KernelPlanner::new(context);
        let mut program = build_program(
            &ir,
            compiled.value_order(),
            &policy,
            &planner,
            &recorder,
            self.target,
        )?;
        program.decomposed.clone_from(&compiled.analysis.decomposed);
        program.state_functions = self.state_functions;
        program.library_table = self.library_table;
        program.decoding = self.decoding;
        // A record decides, for targets that have the instructions.
        if self.target.compute_capability() >= (9, 0)
            && let Some(records) = planner.context().records()
            && records.get(&dependent_launch_key()) == Some(TunedChoice::DependentLaunch)
        {
            records.note_used(&dependent_launch_key());
            program.dependent_launch = true;
        }

        Ok(program)
    }
}

#[cfg(test)]
mod tests;
