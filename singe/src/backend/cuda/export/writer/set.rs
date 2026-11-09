// Renders several steps of one model as one program: each step is a whole program in its own file, and one more file owns what they share.

use serde_json::{Value, json};

use crate::{
    backend::cuda::export::{
        program::{BufferKind, CudaProgramSet, ProgramEntry},
        writer::{
            CONFIGURATION_REFUSED, CUBLAS_BASE, CUDNN_BASE, CUFFT_BASE, CUTENSOR_BASE,
            LIBRARY_VERSION, ProgramFiles, Renderer,
            generate::{LoopStep, generation_loop_source},
        },
    },
    error::{Error, Result},
    utility::emit,
};

// The files of a program of several steps, by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramSetFiles {
    pub files: Vec<(String, String)>,
    pub weights: Vec<u8>,
}

fn entry_prefix(prefix: &str, entry: &ProgramEntry) -> String {
    format!("{prefix}_{}", entry.name)
}

impl CudaProgramSet {
    // Renders the program's files, with `prefix` starting every name they declare.
    //
    // # Errors
    //
    // Returns an error if a step has no C++ form or `prefix` is not a C++ identifier.
    pub fn render(&self, prefix: &str) -> Result<ProgramSetFiles> {
        if self.entries.is_empty() {
            return Err(Error::invalid_argument(
                "a program of several steps has at least one",
            ));
        }

        let mut files = Vec::new();
        let mut weights = Vec::new();
        let mut steps = Vec::new();
        let mut includes = String::new();
        let mut members = String::new();
        let mut create = String::new();
        let mut load = String::new();
        let mut entries_code = String::new();
        let mut declarations = String::new();
        let mut destroy = String::new();
        let mut flags = String::new();
        let mut sources = Vec::new();
        let mut links = Vec::new();
        for (index, entry) in self.entries.iter().enumerate() {
            let name = &entry.name;
            let step = entry_prefix(prefix, entry);
            let shared = entry.shared.iter().map(Option::is_some).collect::<Vec<_>>();
            let ProgramFiles {
                header,
                source,
                weights: own_weights,
                manifest,
                ..
            } = entry.program.render_as(&step, shared)?;
            let parsed: Value =
                serde_json::from_str(&manifest).map_err(|error| Error::Compile(error.into()))?;
            steps.push(json!({
                "name": name,
                "header": format!("{step}.h"),
                "source": format!("{step}.cu"),
                "manifest": format!("{step}.json"),
                "compile": parsed["compile"],
                "link": parsed["link"],
                "weights_offset": weights.len(),
                "weights_bytes": own_weights.len(),
            }));
            // A step compiles as `nvcc -std=c++17 FLAGS -c STEP.cu`, and the loop builds every step with the same flags.
            if flags.is_empty() {
                parsed["compile"]
                    .as_str()
                    .unwrap_or_default()
                    .trim_start_matches("nvcc -std=c++17")
                    .trim_end_matches(&format!("-c {step}.cu"))
                    .trim()
                    .clone_into(&mut flags);
            }

            sources.push(format!("{step}.cu"));
            links.extend(
                parsed["link"]
                    .as_str()
                    .unwrap_or_default()
                    .split_whitespace()
                    .map(ToOwned::to_owned),
            );
            files.push((format!("{step}.h"), header));
            files.push((format!("{step}.cu"), source));
            files.push((format!("{step}.json"), manifest));

            emit!(includes, "#include \"{step}.h\"");
            emit!(members, "    {step}_state* {name};");
            if entry.shared.iter().any(Option::is_some) {
                let pointers = entry
                    .shared
                    .iter()
                    .map(|shared| {
                        shared.map_or_else(
                            || "nullptr".to_owned(),
                            |(owner, tensor)| {
                                let owner = &self.entries[owner];
                                format!(
                                    "{}_state_buffer(s->{}, {tensor})",
                                    entry_prefix(prefix, owner),
                                    owner.name,
                                )
                            },
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                emit!(
                    create,
                    "    {{\n        void* const shared[] = {{{pointers}}};\n        SINGE_SET_CHECK({step}, {step}_create_shared(&s->{name}, shared));\n    }}"
                );
            } else {
                emit!(
                    create,
                    "    SINGE_SET_CHECK({step}, {step}_create(&s->{name}));"
                );
            }

            emit!(
                load,
                "    SINGE_SET_STEP({step}, {step}_load_weights(s->{name}, static_cast<const char*>(weights) + {}ULL));",
                weights.len(),
            );
            weights.extend_from_slice(&own_weights);
            emit!(
                entries_code,
                "extern \"C\" {prefix}_status {prefix}_run_{name}({prefix}_state* s, const {step}_inputs* inputs, const {step}_outputs* outputs, cudaStream_t stream) {{\n    SINGE_SET_STEP({step}, {step}_run(s->{name}, inputs, outputs, stream));\n    return {prefix}_ok;\n}}\n\nextern \"C\" {prefix}_status {prefix}_finish_{name}({prefix}_state* s, cudaStream_t stream, unsigned int* status) {{\n    SINGE_SET_STEP({step}, {step}_finish(s->{name}, stream, status));\n    return {prefix}_ok;\n}}\n"
            );
            emit!(
                declarations,
                "// Enqueues the step `{name}` on `stream`; the caller keeps its buffers alive until the stream finishes.\n{prefix}_status {prefix}_run_{name}({prefix}_state* state, const {step}_inputs* inputs, const {step}_outputs* outputs, cudaStream_t stream);\n// Waits for `stream`, and stores the first code a checking kernel of `{name}` reported, or zero, in `status` when it is not null.\n{prefix}_status {prefix}_finish_{name}({prefix}_state* state, cudaStream_t stream, unsigned int* status);"
            );
            // Later steps read what earlier ones own, so they go first.
            let destroy_step = format!("    {step}_destroy(s->{name});\n");
            destroy.insert_str(0, &destroy_step);
            debug_assert!(
                entry
                    .shared
                    .iter()
                    .flatten()
                    .all(|(owner, _)| *owner < index)
            );
        }

        let guard = prefix.to_uppercase();
        let header = format!(
            "// Generated by Singe.\n#ifndef {guard}_H\n#define {guard}_H\n\n#include <cuda_runtime.h>\n\n{includes}\nstruct {prefix}_state;\n\n// What every function returns, with the values each step's own status type has.\nenum {prefix}_status : int {{\n    {prefix}_ok = 0,\n    {prefix}_cublas_error = {CUBLAS_BASE},\n    {prefix}_cufft_error = {CUFFT_BASE},\n    {prefix}_cudnn_error = {CUDNN_BASE},\n    {prefix}_cutensor_error = {CUTENSOR_BASE},\n    {prefix}_library_version = {LIBRARY_VERSION},\n    {prefix}_configuration_refused = {CONFIGURATION_REFUSED},\n}};\n\nextern \"C\" {{\n// Describes the last failure on the calling thread, in whichever step it was.\nconst char* {prefix}_failure(void);\n// Allocates every step's buffers, holding the tensors steps have in common once, and plans their library calls.\n{prefix}_status {prefix}_create({prefix}_state** state);\n// Copies in every tensor `{prefix}.weights` holds, from the bytes of that file.\n{prefix}_status {prefix}_load_weights({prefix}_state* state, const void* weights);\n{declarations}void {prefix}_destroy({prefix}_state* state);\n}}\n\n#endif\n"
        );
        let source = format!(
            "// Generated by Singe.\n#include <new>\n\n#include \"{prefix}.h\"\n\n// The failure function of the step that failed last on this thread.\nstatic thread_local const char* (*singe_failed_step)(void) = nullptr;\n\n#define SINGE_SET_STEP(step, expression) do {{ const int singe_status = static_cast<int>(expression); if (singe_status != 0) {{ singe_failed_step = step##_failure; return static_cast<{prefix}_status>(singe_status); }} }} while (false)\n#define SINGE_SET_CHECK(step, expression) do {{ const int singe_status = static_cast<int>(expression); if (singe_status != 0) {{ singe_failed_step = step##_failure; {prefix}_destroy(s); return static_cast<{prefix}_status>(singe_status); }} }} while (false)\n\nstruct {prefix}_state {{\n{members}}};\n\nextern \"C\" const char* {prefix}_failure(void) {{\n    return singe_failed_step == nullptr ? \"\" : singe_failed_step();\n}}\n\nextern \"C\" {prefix}_status {prefix}_create({prefix}_state** state) {{\n    {prefix}_state* s = new (std::nothrow) {prefix}_state{{}};\n    if (s == nullptr) return static_cast<{prefix}_status>(cudaErrorMemoryAllocation);\n{create}    *state = s;\n    return {prefix}_ok;\n}}\n\nextern \"C\" {prefix}_status {prefix}_load_weights({prefix}_state* s, const void* weights) {{\n{load}    return {prefix}_ok;\n}}\n\n{entries_code}extern \"C\" void {prefix}_destroy({prefix}_state* s) {{\n    if (s == nullptr) return;\n{destroy}    delete s;\n}}\n"
        );
        let decoding = self
            .decoding
            .map(|layout| {
                let step = |name: &str| {
                    self.entries
                        .iter()
                        .find(|entry| entry.name == name)
                        .map(|entry| {
                            (
                                &entry.program,
                                entry_prefix(prefix, entry),
                                format!("{prefix}_run_{name}"),
                                format!("{prefix}_finish_{name}"),
                            )
                        })
                };
                let (program, types, run, finish) = step("decode").ok_or_else(|| {
                    Error::missing_input("a generation loop's `decode` step")
                })?;
                let encode = step("encode");
                let source = generation_loop_source(
                    prefix,
                    layout,
                    LoopStep {
                        program,
                        types: &types,
                        run: &run,
                        finish: &finish,
                    },
                    encode
                        .as_ref()
                        .map(|(program, types, run, finish)| LoopStep {
                            program,
                            types,
                            run,
                            finish,
                        }),
                    weights.len(),
                )?;
                files.push((format!("{prefix}_generate.cu"), source));
                links.sort_unstable();
                links.dedup();
                Result::Ok(json!({
                    "capacity": layout.capacity,
                    "page_size": layout.page_size,
                    "generate": format!(
                        "nvcc -std=c++17 {flags} {prefix}.cu {} {prefix}_generate.cu {} -o {prefix}_generate",
                        sources.join(" "),
                        links.join(" "),
                    ),
                }))
            })
            .transpose()?;
        let manifest = serde_json::to_string_pretty(&json!({
            "compile": format!("nvcc -std=c++17 -c {prefix}.cu"),
            "steps": steps,
            "weights_bytes": weights.len(),
            "decoding": decoding,
        }))
        .map_err(|error| Error::Compile(error.into()))?;
        files.push((format!("{prefix}.h"), header));
        files.push((format!("{prefix}.cu"), source));
        files.push((format!("{prefix}.json"), manifest));
        Ok(ProgramSetFiles { files, weights })
    }
}

impl Renderer<'_> {
    // The statements that hand out each state tensor's buffer, for another step to share.
    pub fn state_buffer_cases(&self) -> String {
        let mut cases = String::new();
        for (index, buffer) in self.program.buffers.iter().enumerate() {
            if let BufferKind::State(state) = buffer.kind {
                emit!(
                    cases,
                    "    case {state}:\n        return s->buffers[{index}];"
                );
            }
        }
        cases
    }
}
