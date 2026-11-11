// A loop that generates through a decoder's choosing steps with nothing of Singe, writing each step's tensors as `PageTable` would and running its rows as `RunningBatch` does, after an encode step if the model has one.

use crate::{
    backend::cuda::{
        export::{
            builder::span_bytes,
            program::{CudaProgram, Decoding, StateKind},
            writer::{Renderer, fill_template},
        },
        kernel_plan::value_type,
    },
    data_type::DataType,
    error::{Error, Result},
    utility::emit,
};

// What a decoder step reads besides the caches it carries from one step to the next, as `DecodeStep` names them, which the loop's `Step` holds under the same names.
const STEP_INPUTS: [&str; 11] = [
    "tokens",
    "positions",
    "write_pages",
    "write_slots",
    "query_offsets",
    "page_offsets",
    "pages",
    "last_page_lengths",
    "last_rows",
    "real_tokens",
    "sequence_ids",
];

// Where a choosing step reads which rows choose their sequence's first token.
const FIRST_TOKENS: &str = "first_tokens";

// One program a generation loop runs, and the names its C++ declarations take.
#[derive(Debug, Clone, Copy)]
pub struct LoopStep<'a> {
    pub program: &'a CudaProgram,
    // What its `_inputs` and `_outputs` structs start with.
    pub types: &'a str,
    // The functions that enqueue it and wait for it.
    pub run: &'a str,
    pub finish: &'a str,
}

// The steps a generation loop runs: `decode`, of a token per row, and optionally `prompt`, of more rows, and `encode`, once before the prompt.
#[derive(Debug, Clone, Copy)]
pub struct LoopSteps<'a> {
    pub decode: LoopStep<'a>,
    pub prompt: Option<LoopStep<'a>>,
    pub encode: Option<LoopStep<'a>>,
}

// The device buffers of a loop: one per step input of each decoding step, since steps differ in rows, and one per other input name, which steps share, and per output.
#[derive(Default)]
struct Buffers {
    named: Vec<(String, u64)>,
}

impl Buffers {
    fn index(&mut self, name: &str, bytes: u64) -> Result<usize> {
        if let Some(index) = self.named.iter().position(|(other, _)| other == name) {
            if self.named[index].1 != bytes {
                return Err(Error::invalid_argument(format!(
                    "steps of a generation loop disagree on the size of {name}"
                )));
            }
            return Ok(index);
        }

        self.named.push((name.to_owned(), bytes));
        Ok(self.named.len() - 1)
    }
}

// The rows a decoding step reads, from its token input.
fn step_rows(program: &CudaProgram) -> Result<i64> {
    let tokens = program
        .inputs
        .iter()
        .find(|input| input.graph_name == "tokens")
        .ok_or_else(|| Error::missing_input("a generation loop step's tokens"))?;
    i64::try_from(tokens.layout.element_count())
        .map_err(|_| Error::out_of_range("generation loop step rows"))
}

// The C++ that runs a decoding step after writing the step inputs it reads and which rows choose a first token, and points `step_tokens` at the tokens it chose.
fn step_code(
    step: &LoopStep<'_>,
    variable: &str,
    sequences: i64,
    buffers: &mut Buffers,
) -> Result<String> {
    let mut code = String::new();
    for input in &step.program.inputs {
        let name = input.graph_name.as_str();
        let value = if STEP_INPUTS.contains(&name) {
            format!("step.{name}")
        } else if name == FIRST_TOKENS {
            "first".to_owned()
        } else {
            continue;
        };
        if input.data_type != DataType::I64 {
            return Err(Error::invalid_argument(
                "a generation loop writes 64-bit step inputs",
            ));
        }

        let index = buffers.index(
            &format!("{variable}{name}"),
            span_bytes(input.data_type, &input.layout)?,
        )?;
        emit!(code, "            upload(buffer_{index}, {value});");
    }

    let [tokens] = step.program.outputs.as_slice() else {
        return Err(Error::invalid_argument(
            "a generation loop reads a choosing step, whose only output is the chosen tokens",
        ));
    };
    if tokens.data_type != DataType::I64
        || i64::try_from(tokens.layout.element_count()).ok() != Some(sequences)
    {
        return Err(Error::invalid_argument(
            "a generation loop reads a token per row from a choosing step",
        ));
    }

    let tokens = buffers.index(
        &format!("{variable}output {}", tokens.graph_name),
        span_bytes(tokens.data_type, &tokens.layout)?,
    )?;
    emit!(
        code,
        "            check({run}(state, &{variable}inputs, &{variable}outputs, nullptr), \"{run}\");\n            unsigned int word = 0;\n            check({finish}(state, nullptr, &word), \"{finish}\");\n            if (word != 0) {{\n                fprintf(stderr, \"a checking kernel reported %u\\n\", word);\n                return 1;\n            }}\n            step_tokens = buffer_{tokens};",
        run = step.run,
        finish = step.finish,
    );
    Ok(code)
}

// Writes a generation loop over `steps`, choosing steps of the rows and paged cache `decoding` describes, through the program declared by `prefix`, with `seed` the C++ that seeds its draws from `seed`.
// With an encode step, the loop first runs it once on a source the command line names, which writes what the decoding steps read, such as cross-attention keys and values, for one sequence.
pub fn generation_loop_source(
    prefix: &str,
    decoding: &Decoding,
    steps: LoopSteps<'_>,
    seed: &str,
    weights_bytes: usize,
) -> Result<String> {
    let layout = decoding.layout;
    let sequences =
        i64::try_from(layout.sequences).map_err(|_| Error::out_of_range("sequences"))?;
    if steps.encode.is_some() && sequences != 1 {
        return Err(Error::invalid_argument(
            "a generation loop with an encode step runs one sequence",
        ));
    }

    if step_rows(steps.decode.program)? != sequences {
        return Err(Error::invalid_argument(
            "a generation loop's decode step reads a token per row",
        ));
    }

    let mut buffers = Buffers::default();
    let mut step_lines = String::new();
    let mut prompt_rows = sequences;
    if let Some(prompt) = &steps.prompt {
        prompt_rows = step_rows(prompt.program)?;
        if prompt_rows <= sequences {
            return Err(Error::invalid_argument(
                "a generation loop's prompt step reads more rows than its decode step",
            ));
        }

        let code = step_code(prompt, "prompt_", sequences, &mut buffers)?;
        emit!(
            step_lines,
            "        if (!decode) {{\n{code}\n        }} else {{"
        );
    } else {
        emit!(step_lines, "        {{");
    }

    let code = step_code(&steps.decode, "", sequences, &mut buffers)?;
    emit!(step_lines, "{code}\n        }}");

    let all = steps
        .encode
        .iter()
        .map(|step| ("encode_", step))
        .chain([("", &steps.decode)])
        .chain(steps.prompt.iter().map(|step| ("prompt_", step)))
        .collect::<Vec<_>>();
    // Every other input is shared by name: the caches every step reads, and the encode step's source.
    let mut allocations = String::new();
    let mut struct_lines = String::new();
    let mut source = None;
    let (mut length, mut position, mut rows) = (None, None, None);
    for (variable, step) in &all {
        emit!(
            struct_lines,
            "    {types}_inputs {variable}inputs;\n    {types}_outputs {variable}outputs;",
            types = step.types
        );
        let decodes = !variable.starts_with("encode");
        for value in &step.program.inputs {
            let name = value.graph_name.as_str();
            let bytes = span_bytes(value.data_type, &value.layout)?;
            let is_step = decodes && (STEP_INPUTS.contains(&name) || name == FIRST_TOKENS);
            let index = if is_step {
                buffers.index(&format!("{variable}{name}"), bytes)?
            } else {
                buffers.index(name, bytes)?
            };
            if !is_step && !name.starts_with("cache.") {
                if decodes {
                    return Err(Error::invalid_argument(format!(
                        "a generation loop does not know what to write to step input {name}"
                    )));
                }

                match name {
                    "source_length" => length = Some(index),
                    "source_position" => position = Some(index),
                    _ => {
                        let element = value.data_type.size_bytes();
                        if source.replace((index, bytes, element)).is_some() {
                            return Err(Error::invalid_argument(
                                "an encode step reads one source besides its length, position, and the caches",
                            ));
                        }
                    }
                }
            }

            let ty = value_type(value.data_type)?;
            let qualifier = if value.written { "" } else { "const " };
            emit!(
                allocations,
                "    {variable}inputs.{field} = static_cast<{qualifier}{ty}*>(buffer_{index});",
                field = value.name
            );
        }

        for value in &step.program.outputs {
            let bytes = span_bytes(value.data_type, &value.layout)?;
            let index = buffers.index(&format!("{variable}output {}", value.graph_name), bytes)?;
            if !decodes && value.graph_name == "source_rows" {
                rows = Some(index);
            }

            let ty = value_type(value.data_type)?;
            emit!(
                allocations,
                "    {variable}outputs.{field} = static_cast<{ty}*>(buffer_{index});",
                field = value.name
            );
        }
    }

    let mut buffer_lines = String::new();
    for (index, (_, bytes)) in buffers.named.iter().enumerate() {
        emit!(
            buffer_lines,
            "    void* buffer_{index} = device_zeroed({bytes}ULL);"
        );
    }

    let frees = (0..buffers.named.len())
        .map(|index| format!("    cudaFree(buffer_{index});"))
        .collect::<Vec<_>>()
        .join("\n");
    let weights = if weights_bytes == 0 {
        "    const std::vector<char> weights;\n".to_owned()
    } else {
        format!(
            "    const std::vector<char> weights = read_file(\"{prefix}.weights\", {weights_bytes}ULL);\n"
        )
    };
    let (usage, arguments, source_argument, encode_code) = match (steps.encode, source) {
        (None, _) => (
            "[--seed N] NEW_TOKENS (TOKEN... | --prompts FILE)",
            "2",
            String::new(),
            String::new(),
        ),
        (Some(step), Some((index, bytes, element))) => {
            // A source file may be shorter than its buffer, whose rest stays zero, and a length input says how much it held; a position says where the prompt's starred token stands, and rows how many times it stands there.
            let mut code = format!(
                "    {{\n        if (prompts.size() != 1) {{\n            fprintf(stderr, \"a model with an encode step decodes one prompt\\n\");\n            return 2;\n        }}\n        const std::vector<char> source = read_file(source_path, {bytes}ULL, false);\n        check_cuda(cudaMemcpy(buffer_{index}, source.data(), source.size(), cudaMemcpyHostToDevice), \"cudaMemcpy\");\n"
            );
            if let Some(length) = length {
                emit!(
                    code,
                    "        upload(buffer_{length}, std::vector<int64_t>{{static_cast<int64_t>(source.size() / {element})}});"
                );
            }

            if let Some(position) = position {
                emit!(
                    code,
                    "        upload(buffer_{position}, std::vector<int64_t>{{static_cast<int64_t>(prompts[0].placeholder_at)}});"
                );
            }

            emit!(
                code,
                "        check({run}(state, &encode_inputs, &encode_outputs, nullptr), \"{run}\");\n        unsigned int word = 0;\n        check({finish}(state, nullptr, &word), \"{finish}\");\n        if (word != 0) {{\n            fprintf(stderr, \"a checking kernel reported %u\\n\", word);\n            return 1;\n        }}",
                run = step.run,
                finish = step.finish,
            );
            if let Some(rows) = rows {
                emit!(
                    code,
                    "        check_cuda(cudaMemcpy(&placeholder_rows, buffer_{rows}, sizeof(placeholder_rows), cudaMemcpyDeviceToHost), \"cudaMemcpy\");"
                );
            }

            emit!(code, "    }}");
            (
                "[--seed N] NEW_TOKENS SOURCE (TOKEN... | --prompts FILE)",
                "3",
                "    const char* source_path = argv[argument++];\n".to_owned(),
                code,
            )
        }
        (Some(_), None) => {
            return Err(Error::invalid_argument(
                "an encode step reads a source besides the caches",
            ));
        }
    };
    let stop_tokens = decoding
        .stop_tokens
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    Ok(fill_template(
        include_str!("generate.cu.in"),
        &[
            ("prefix", prefix),
            ("usage", usage),
            ("arguments", arguments),
            ("source_argument", &source_argument),
            ("sequences", &sequences.to_string()),
            ("page_size", &layout.page_size.to_string()),
            ("capacity", &layout.capacity.to_string()),
            ("scratch_page", &layout.scratch_page.to_string()),
            ("table_length", &layout.table_length.to_string()),
            ("largest", &(prompt_rows - sequences).to_string()),
            ("prompt_rows", &prompt_rows.to_string()),
            ("stop_tokens", &stop_tokens),
            ("weights", &weights),
            ("seed", seed),
            ("structs", &struct_lines),
            ("buffers", &buffer_lines),
            ("allocations", &allocations),
            ("encode", &encode_code),
            ("steps", &step_lines),
            ("frees", &frees),
        ],
    ))
}

impl Renderer<'_> {
    // The generation loop of a program exported with decoding, a choosing step of a token per row, whose generators it seeds itself.
    pub fn generation_loop(&self, weights_bytes: usize) -> Result<Option<String>> {
        let Some(decoding) = &self.program.decoding else {
            return Ok(None);
        };
        let prefix = self.prefix;
        let mut seed = String::new();
        for state in &self.program.state {
            if matches!(state.kind, StateKind::Generator { .. }) {
                emit!(
                    seed,
                    "    {{\n        const long long generator[2] = {{seed, 0}};\n        check({prefix}_set_{name}(state, generator), \"{prefix}_set_{name}\");\n    }}",
                    name = state.name
                );
            }
        }

        let (run, finish) = (format!("{prefix}_run"), format!("{prefix}_finish"));
        generation_loop_source(
            prefix,
            decoding,
            LoopSteps {
                decode: LoopStep {
                    program: self.program,
                    types: prefix,
                    run: &run,
                    finish: &finish,
                },
                prompt: None,
                encode: None,
            },
            &seed,
            weights_bytes,
        )
        .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{DecodeStep, TensorGroup};

    #[test]
    fn test_the_loop_writes_every_input_a_decode_step_has() {
        let step = DecodeStep::meta(2, 1, 4).unwrap();
        let names = step
            .named_tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert_eq!(names, STEP_INPUTS);
    }
}
