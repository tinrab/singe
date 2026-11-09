use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde_json::json;

use crate::{
    backend::{
        cuda::{
            codegen::{CompileRequirements, kernel_compile_options},
            export::{
                program::{BufferKind, Library, ProgramValue, StateKind},
                writer::{
                    CONFIGURATION_REFUSED, CUBLAS_BASE, CUDNN_BASE, CUFFT_BASE, CUTENSOR_BASE,
                    EMBEDDED_VALUE_BYTES, LIBRARY_VERSION, LIBRARY_VERSIONS, ProgramFiles,
                    Renderer, cudnn::TABLE_KNOBS_HELPER, fill_template, steps::StepOutput,
                },
            },
            kernel_plan::value_type,
        },
        registry::Operation,
    },
    error::{Error, Result},
    graph::{ComputePrecision, Determinism, NumericalPolicy},
    utility::emit,
};

// C++ for the program's own buffers.
#[derive(Default)]
struct BufferCode<'a> {
    create: String,
    destroy: String,
    load: String,
    embedded: String,
    weights: Vec<u8>,
    weights_offsets: HashMap<&'a str, usize>,
}

// C++ a program holds once for a library it calls.
struct LibrarySetup {
    include: &'static str,
    members: &'static str,
    create: String,
    destroy: &'static str,
    helpers: &'static str,
}

fn library_setup(library: Library, numerics: NumericalPolicy) -> LibrarySetup {
    let deterministic = numerics.determinism() == Determinism::Require;
    match library {
        Library::Cublas => {
            let math = if numerics.precision() == ComputePrecision::Full {
                "CUBLAS_PEDANTIC_MATH"
            } else {
                "CUBLAS_TF32_TENSOR_OP_MATH"
            };
            let atomics = if deterministic {
                format!(
                    "\n    SINGE_LIBRARY({CUBLAS_BASE}, cublasSetAtomicsMode(s->cublas, CUBLAS_ATOMICS_NOT_ALLOWED));"
                )
            } else {
                String::new()
            };
            LibrarySetup {
                include: "#include <cublas_v2.h>\n",
                members: "    cublasHandle_t cublas;\n",
                create: format!(
                    "    SINGE_LIBRARY({CUBLAS_BASE}, cublasCreate(&s->cublas));\n    {{\n        int version = 0;\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasGetVersion(s->cublas, &version));\n        if (version / 10000 != CUBLAS_VER_MAJOR) return singe_failed({LIBRARY_VERSION}, \"cublasGetVersion\");\n    }}\n    SINGE_LIBRARY({CUBLAS_BASE}, cublasSetMathMode(s->cublas, static_cast<cublasMath_t>({math} | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));{atomics}\n"
                ),
                destroy: "    if (s->cublas != nullptr) cublasDestroy(s->cublas);\n",
                helpers: "",
            }
        }
        Library::CublasLt => LibrarySetup {
            include: "#include <cublasLt.h>\n",
            members: "    cublasLtHandle_t cublaslt;\n",
            create: format!(
                "    if (static_cast<int>(cublasLtGetVersion() / 10000) != CUBLAS_VER_MAJOR) return singe_failed({LIBRARY_VERSION}, \"cublasLtGetVersion\");\n    SINGE_LIBRARY({CUBLAS_BASE}, cublasLtCreate(&s->cublaslt));\n"
            ),
            destroy: "    if (s->cublaslt != nullptr) cublasLtDestroy(s->cublaslt);\n",
            helpers: "",
        },
        Library::Cudnn => LibrarySetup {
            include: "#include <cudnn.h>\n#include <limits>\n#include <vector>\n",
            members: "    cudnnHandle_t cudnn;\n    std::vector<cudnnBackendDescriptor_t> cudnn_retained;\n",
            create: format!(
                "    if (static_cast<int>(cudnnGetVersion() / 10000) != CUDNN_MAJOR) return singe_failed({LIBRARY_VERSION}, \"cudnnGetVersion\");\n    SINGE_LIBRARY({CUDNN_BASE}, cudnnCreate(&s->cudnn));\n"
            ),
            destroy: "    for (cudnnBackendDescriptor_t retained : s->cudnn_retained) cudnnBackendDestroyDescriptor(retained);\n    if (s->cudnn != nullptr) cudnnDestroy(s->cudnn);\n",
            helpers: include_str!("cudnn_helpers.cu"),
        },
        // cuFFT counts its version in thousands, unlike the other libraries.
        Library::Cufft => LibrarySetup {
            include: "#include <cufft.h>\n",
            members: "",
            create: format!(
                "    {{\n        int version = 0;\n        SINGE_LIBRARY({CUFFT_BASE}, cufftGetVersion(&version));\n        if (version / 1000 != CUFFT_VER_MAJOR) return singe_failed({LIBRARY_VERSION}, \"cufftGetVersion\");\n    }}\n"
            ),
            destroy: "",
            helpers: "",
        },
        Library::Cutensor => LibrarySetup {
            include: "#include <cutensor.h>\n",
            members: "    cutensorHandle_t cutensor;\n",
            create: format!(
                "    if (static_cast<int>(cutensorGetVersion() / 10000) != CUTENSOR_MAJOR) return singe_failed({LIBRARY_VERSION}, \"cutensorGetVersion\");\n    SINGE_LIBRARY({CUTENSOR_BASE}, cutensorCreate(&s->cutensor));\n"
            ),
            destroy: "    if (s->cutensor != nullptr) cutensorDestroy(s->cutensor);\n",
            helpers: "",
        },
    }
}

const fn link_flag(library: Library) -> &'static str {
    match library {
        Library::Cublas => "-lcublas",
        Library::CublasLt => "-lcublasLt",
        Library::Cudnn => "-lcudnn",
        Library::Cufft => "-lcufft",
        Library::Cutensor => "-lcutensor",
    }
}

// The `nvcc` spelling of a compile argument: `--gpu-architecture` drops a target's suffix there, compiling `sm_100a` for plain `sm_100`, so a suffixed target goes through `-gencode`.
fn nvcc_argument(argument: &str) -> String {
    argument
        .strip_prefix("--gpu-architecture=sm_")
        .filter(|target| target.ends_with(['a', 'f']))
        .map_or_else(
            || argument.to_owned(),
            |target| format!("-gencode=arch=compute_{target},code=sm_{target}"),
        )
}

impl Renderer<'_> {
    fn buffer_code(&self) -> BufferCode<'_> {
        let mut code = BufferCode::default();
        for (index, buffer) in self.program.buffers.iter().enumerate() {
            if matches!(buffer.kind, BufferKind::Input(_) | BufferKind::Output(_))
                || buffer.bytes == 0
            {
                continue;
            }

            if let BufferKind::State(state) = buffer.kind
                && self.shared.get(state).copied().unwrap_or(false)
            {
                emit!(code.create, "    s->buffers[{index}] = shared[{state}];");
                continue;
            }

            emit!(code.destroy, "    cudaFree(s->buffers[{index}]);");
            let bytes = buffer.bytes;
            emit!(
                code.create,
                "    SINGE_CHECK(cudaMalloc(&s->buffers[{index}], {bytes}ULL));"
            );
            let zero = format!("    SINGE_CHECK(cudaMemset(s->buffers[{index}], 0, {bytes}ULL));");
            match &buffer.kind {
                BufferKind::Scratch | BufferKind::Status => emit!(code.create, "{zero}"),
                BufferKind::State(state) => {
                    let state = &self.program.state[*state];
                    if matches!(state.kind, StateKind::Generator { .. }) {
                        // A run writes it before anything reads it.
                    } else if state.values.len() <= EMBEDDED_VALUE_BYTES {
                        let values = state
                            .values
                            .iter()
                            .map(u8::to_string)
                            .collect::<Vec<_>>()
                            .join(",");
                        emit!(
                            code.embedded,
                            "static const unsigned char singe_value_{index}[] = {{{values}}};"
                        );
                        emit!(
                            code.create,
                            "    SINGE_CHECK(cudaMemcpy(s->buffers[{index}], singe_value_{index}, {bytes}ULL, cudaMemcpyHostToDevice));"
                        );
                    } else {
                        emit!(code.create, "{zero}");
                        emit!(
                            code.load,
                            "    SINGE_CHECK(cudaMemcpy(s->buffers[{index}], static_cast<const char*>(weights) + {}ULL, {bytes}ULL, cudaMemcpyHostToDevice));",
                            code.weights.len(),
                        );
                        code.weights_offsets
                            .insert(state.name.as_str(), code.weights.len());
                        code.weights.extend_from_slice(&state.values);
                    }
                }
                _ => {}
            }
        }
        code
    }

    // The statements that hand each generator its state at the start of a run, adding what the state holds and sets up for them.
    fn generators(&self, members: &mut String, create: &mut String) -> Result<String> {
        let mut generators = String::new();
        for (index, state) in self.program.state.iter().enumerate() {
            let StateKind::Generator { blocks } = state.kind else {
                continue;
            };
            let name = &state.name;
            let ty = value_type(state.data_type)?;
            let buffer = self.state_buffer(index)?;
            let [seed, offset] = [&state.values[..8], &state.values[8..16]].map(|word| {
                u64::from_ne_bytes(word.try_into().expect("a generator state holds two words"))
            });
            emit!(members, "    {ty} generator_{name}[2];");
            emit!(
                create,
                "    s->generator_{name}[0] = static_cast<{ty}>({seed}ULL);\n    s->generator_{name}[1] = static_cast<{ty}>({offset}ULL);"
            );
            // A copy from pageable memory has reached staging when this returns, so the state can advance at once.
            emit!(
                generators,
                "    SINGE_CHECK(cudaMemcpyAsync(s->buffers[{buffer}], s->generator_{name}, sizeof(s->generator_{name}), cudaMemcpyHostToDevice, stream));\n    s->generator_{name}[1] += static_cast<{ty}>({blocks}ULL);"
            );
        }
        Ok(generators)
    }

    // The statements that report a run's status word, or that it has none.
    fn finish_status(&self) -> String {
        let status = self
            .program
            .buffers
            .iter()
            .position(|buffer| buffer.kind == BufferKind::Status);
        status.map_or_else(
            || "    if (status != nullptr) *status = 0U;\n".to_owned(),
            |index| {
                format!(
                    "    unsigned int word = 0U;\n    SINGE_CHECK(cudaMemcpy(&word, s->buffers[{index}], sizeof(word), cudaMemcpyDeviceToHost));\n    SINGE_CHECK(cudaMemset(s->buffers[{index}], 0, sizeof(word)));\n    if (status != nullptr) *status = word;\n"
                )
            },
        )
    }

    fn manifest(
        &self,
        libraries: &BTreeSet<Library>,
        needs: CompileRequirements,
        buffers: &BufferCode<'_>,
    ) -> Result<String> {
        let prefix = self.prefix;
        let program = self.program;
        let compile = kernel_compile_options(program.target, needs, program.numerics.precision())?
            .as_arguments()
            .iter()
            .map(|argument| nvcc_argument(argument))
            .collect::<Vec<_>>()
            .join(" ");
        let link = libraries
            .iter()
            .map(|library| link_flag(*library))
            .collect::<Vec<_>>()
            .join(" ");
        let value = |value: &ProgramValue| {
            json!({
                "name": value.name,
                "data_type": value.data_type.to_string(),
                "dimensions": value.layout.dimensions(),
                "strides": value.layout.strides(),
                "offset": value.layout.storage_offset(),
                "written": value.written,
                "count": value.count.map(|count| program.outputs[count].name.as_str()),
            })
        };
        let decoding = program.decoding.map(|layout| {
            json!({
                "sequences": layout.sequences,
                "capacity": layout.capacity,
                "page_size": layout.page_size,
                "scratch_page": layout.scratch_page,
                "table_length": layout.table_length,
                "generate": format!(
                    "nvcc -std=c++17 {compile} {prefix}.cu {prefix}_generate.cu {link} -o {prefix}_generate"
                ),
            })
        });
        serde_json::to_string_pretty(&json!({
            "target": program.target.to_string(),
            "decoding": decoding,
            "tuning_records": {
                "used": program.record_coverage.used,
            },
            "compile": format!("nvcc -std=c++17 {compile} -c {prefix}.cu"),
            "numerics": {
                "precision": format!("{:?}", program.numerics.precision()),
                "determinism": format!("{:?}", program.numerics.determinism()),
            },
            "link": link,
            "libraries": LIBRARY_VERSIONS
                .iter()
                .filter(|(library, _)| libraries.contains(library))
                .map(|&(library, version)| (library.name(), version))
                .collect::<BTreeMap<_, _>>(),
            "inputs": program.inputs.iter().map(value).collect::<Vec<_>>(),
            "outputs": program.outputs.iter().map(value).collect::<Vec<_>>(),
            "state": program.state.iter().map(|state| json!({
                "name": state.name,
                "graph_name": state.graph_name,
                "kind": format!("{:?}", state.kind),
                "data_type": state.data_type.to_string(),
                "dimensions": state.layout.dimensions(),
                "strides": state.layout.strides(),
                "offset": state.layout.storage_offset(),
                "weights_offset": buffers.weights_offsets.get(state.name.as_str()),
            })).collect::<Vec<_>>(),
            "providers": program.providers.iter().map(|provider| json!({
                "operation": provider.operation.map(Operation::name),
                "value": provider.value,
                "provider": provider.provider.name(),
            })).collect::<Vec<_>>(),
            "decomposed": program.decomposed.iter().map(|operation| operation.name()).collect::<Vec<_>>(),
            "tuned_kernels": program.tuned_kernels,
            "weights_bytes": buffers.weights.len(),
        }))
        .map_err(|error| Error::Compile(error.into()))
    }

    pub fn render(&self) -> Result<ProgramFiles> {
        let prefix = self.prefix;
        let program = self.program;
        let mut out = StepOutput::default();
        let body = self.render_steps(&program.steps, &mut out)?;
        let libraries = out
            .code
            .iter()
            .map(|(library, _)| *library)
            .collect::<BTreeSet<_>>();
        let buffers = self.buffer_code();
        let mut create = buffers.create.clone();
        let mut destroy = buffers.destroy.clone();
        let mut members = format!("    void* buffers[{}];\n", program.buffers.len().max(1));

        let generators = self.generators(&mut members, &mut create)?;
        let mut includes = String::new();
        let mut handles_create = String::new();
        let mut handles_destroy = String::new();
        let mut helpers = String::new();
        for &library in &libraries {
            let setup = library_setup(library, program.numerics);
            includes.push_str(setup.include);
            members.push_str(setup.members);
            emit!(handles_create, "    singe_step = \"setting up {library}\";");
            handles_create.push_str(&setup.create);
            handles_destroy.push_str(setup.destroy);
            helpers.push_str(setup.helpers);
        }

        if program.library_table {
            emit!(
                helpers,
                "static const int64_t* singe_library_table = nullptr;\nextern \"C\" void {prefix}_set_library_table(const int64_t* table) {{\n    singe_library_table = table;\n}}"
            );
            if libraries.contains(&Library::Cudnn) {
                helpers.push_str(TABLE_KNOBS_HELPER);
            }
        }

        for (_, library) in &out.code {
            members.push_str(&library.members);
            create.push_str(&library.create);
        }

        for (_, library) in out.code.iter().rev() {
            destroy.insert_str(0, &library.destroy);
        }

        let source = fill_template(
            include_str!("program.cu.in"),
            &[
                ("CUDNN_BASE", &CUDNN_BASE.to_string()),
                ("CONFIGURATION_REFUSED", &CONFIGURATION_REFUSED.to_string()),
                (
                    "limits",
                    if out.non_finite {
                        "#include <limits>\n"
                    } else {
                        ""
                    },
                ),
                ("includes", &includes),
                ("prefix", prefix),
                ("prelude", out.prelude.unwrap_or("")),
                ("kernels", &out.kernels),
                ("embedded", &buffers.embedded),
                ("helpers", &helpers),
                ("members", &members),
                ("handles_create", &handles_create),
                ("state_buffer_cases", &self.state_buffer_cases()),
                ("create", &create),
                ("load", &buffers.load),
                ("state_definitions", &self.state_definitions()?),
                (
                    "inputs_used",
                    if program.inputs.is_empty() {
                        "    (void)inputs;\n"
                    } else {
                        ""
                    },
                ),
                ("generators", &generators),
                ("body", &body),
                ("finish_status", &self.finish_status()),
                ("destroy", &destroy),
                ("handles_destroy", &handles_destroy),
            ],
        );
        let manifest = self.manifest(&libraries, out.needs, &buffers)?;
        let generate = self.generation_loop(buffers.weights.len())?;
        let forced = self.forced.take();
        Ok(ProgramFiles {
            header: self.header()?,
            generate,
            source,
            weights: buffers.weights,
            manifest,
            forced_configurations: forced.count,
            library_table: forced.table,
        })
    }
}
