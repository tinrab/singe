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

// What a decoder step reads besides its caches, one packed tensor, which the loop's `pack` writes from its `Step` in `PACKED_ORDER`.
const STEP_INPUT: &str = "step";

// Where a choosing step reads which rows choose their sequence's first token.
const FIRST_TOKENS: &str = "first_tokens";

const PREVIOUS: &str = "previous";

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
// A buffer that `alias` names lies inside another, from a byte offset, and is neither allocated nor freed itself.
struct Buffer {
    name: String,
    bytes: u64,
    alias: Option<(usize, u64)>,
}

#[derive(Default)]
struct Buffers {
    named: Vec<Buffer>,
}

impl Buffers {
    fn index(&mut self, name: &str, bytes: u64) -> Result<usize> {
        if let Some(index) = self.find(name) {
            if self.named[index].bytes != bytes {
                return Err(Error::invalid_argument(format!(
                    "steps of a generation loop disagree on the size of {name}"
                )));
            }
            return Ok(index);
        }

        self.named.push(Buffer {
            name: name.to_owned(),
            bytes,
            alias: None,
        });
        Ok(self.named.len() - 1)
    }

    fn alias(&mut self, name: &str, bytes: u64, base: usize, offset: u64) -> usize {
        self.named.push(Buffer {
            name: name.to_owned(),
            bytes,
            alias: Some((base, offset)),
        });
        self.named.len() - 1
    }

    fn find(&self, name: &str) -> Option<usize> {
        self.named.iter().position(|buffer| buffer.name == name)
    }
}

// The rows a decoding step reads, from the length of its packed step input over `sequences` sequences and a page table of `table` entries.
fn step_rows(program: &CudaProgram, sequences: i64, table: i64) -> Result<i64> {
    let step = program
        .inputs
        .iter()
        .find(|input| input.graph_name == STEP_INPUT)
        .ok_or_else(|| Error::missing_input("a generation loop step's packed step"))?;
    let length = i64::try_from(step.layout.element_count())
        .map_err(|_| Error::out_of_range("generation loop step rows"))?;
    Ok((length - 2 * (sequences + 1) - 3 * sequences - table) / 5)
}

// The C++ that enqueues a decoding step after writing the step inputs it reads, which rows choose a first token, and the tokens the step before chose, and points `step_tokens` at the tokens it chooses; with the bytes it uploads.
// It launches through the loop's graph `slot`, zero for the decode step and one for the prompt step.
fn step_code(
    step: &LoopStep<'_>,
    variable: &str,
    slot: usize,
    sequences: i64,
    buffers: &mut Buffers,
) -> Result<(String, u64)> {
    let input = |name: &str| {
        step.program
            .inputs
            .iter()
            .find(|input| input.graph_name == name)
    };
    let packed = input(STEP_INPUT)
        .ok_or_else(|| Error::missing_input("a generation loop step's packed step"))?;
    if packed.data_type != DataType::I64 {
        return Err(Error::invalid_argument(
            "a generation loop writes 64-bit step inputs",
        ));
    }

    // The rows choosing their first token follow the packed step in its buffer, so one copy writes both.
    let step_bytes = span_bytes(packed.data_type, &packed.layout)?;
    let first_bytes = input(FIRST_TOKENS)
        .map(|first| span_bytes(first.data_type, &first.layout))
        .transpose()?
        .unwrap_or(0);
    let index = buffers.index(&format!("{variable}{STEP_INPUT}"), step_bytes + first_bytes)?;
    let mut code = String::new();
    if first_bytes > 0 {
        buffers.alias(
            &format!("{variable}{FIRST_TOKENS}"),
            first_bytes,
            index,
            step_bytes,
        );
        emit!(
            code,
            "            std::vector<int64_t> packed = pack(step);\n            packed.insert(packed.end(), first_tokens.begin(), first_tokens.end());\n            upload_staged(buffer_{index}, packed, staging[parity]);"
        );
    } else {
        emit!(
            code,
            "            upload_staged(buffer_{index}, pack(step), staging[parity]);"
        );
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

    let token_bytes = span_bytes(tokens.data_type, &tokens.layout)?;
    let tokens = buffers.index(
        &format!("{variable}output {}", tokens.graph_name),
        token_bytes,
    )?;
    let previous = input(PREVIOUS)
        .filter(|previous| previous.data_type == DataType::I64)
        .ok_or_else(|| {
            Error::missing_input("the tokens a generation loop step reads from the step before")
        })?;
    let previous = buffers.index(PREVIOUS, span_bytes(previous.data_type, &previous.layout)?)?;
    emit!(
        code,
        "            if (previous != nullptr) check_cuda(cudaMemcpyAsync(buffer_{previous}, previous->tokens, {token_bytes}ULL, cudaMemcpyDeviceToDevice, loop_stream), \"cudaMemcpyAsync\");\n            check(launch(step_graphs[{slot}], [&] {{ return static_cast<int>({run}(state, &{variable}inputs, &{variable}outputs, loop_stream)); }}), \"{run}\");\n            step_tokens = buffer_{tokens};\n            ran[{slot}] = true;",
        run = step.run,
    );
    Ok((code, step_bytes + first_bytes))
}

fn finish_code(step: &LoopStep<'_>, slot: usize) -> String {
    format!(
        "    if (ran[{slot}]) {{\n        unsigned int word = 0;\n        check({finish}(state, loop_stream, &word), \"{finish}\");\n        if (word != 0) {{\n            fprintf(stderr, \"a checking kernel reported %u\\n\", word);\n            return 1;\n        }}\n    }}",
        finish = step.finish,
    )
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

    if step_rows(steps.decode.program, sequences, layout.table_length)? != sequences {
        return Err(Error::invalid_argument(
            "a generation loop's decode step reads a token per row",
        ));
    }

    let mut buffers = Buffers::default();
    let mut step_lines = String::new();
    let mut finish_lines = String::new();
    let mut staging = 0;
    let mut prompt_rows = sequences;
    if let Some(prompt) = &steps.prompt {
        prompt_rows = step_rows(prompt.program, sequences, layout.table_length)?;
        if prompt_rows <= sequences {
            return Err(Error::invalid_argument(
                "a generation loop's prompt step reads more rows than its decode step",
            ));
        }

        let (code, bytes) = step_code(prompt, "prompt_", 1, sequences, &mut buffers)?;
        staging = bytes;
        emit!(
            step_lines,
            "        if (!decode) {{\n{code}\n        }} else {{"
        );
        emit!(finish_lines, "{}", finish_code(prompt, 1));
    } else {
        emit!(step_lines, "        {{");
    }

    let (code, bytes) = step_code(&steps.decode, "", 0, sequences, &mut buffers)?;
    let staging = staging.max(bytes) / 8;
    emit!(step_lines, "{code}\n        }}");
    emit!(finish_lines, "{}", finish_code(&steps.decode, 0));

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
    let (mut length, mut position, mut rows, mut grid) = (None, None, None, None);
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
            let is_step = decodes && (name == STEP_INPUT || name == FIRST_TOKENS);
            let index = if is_step {
                buffers
                    .find(&format!("{variable}{name}"))
                    .ok_or_else(|| Error::missing_input("a generation loop step's buffer"))?
            } else {
                buffers.index(name, bytes)?
            };
            if !is_step && name != PREVIOUS && !name.starts_with("cache.") {
                if decodes {
                    return Err(Error::invalid_argument(format!(
                        "a generation loop does not know what to write to step input {name}"
                    )));
                }

                match name {
                    "source_length" => length = Some(index),
                    "source_position" => position = Some(index),
                    "source_grid" => grid = Some((index, bytes)),
                    _ => {
                        let element = value.data_type.size_bytes();
                        if source.replace((index, bytes, element)).is_some() {
                            return Err(Error::invalid_argument(
                                "an encode step reads one source besides its length, position, grid, and the caches",
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
    for (index, buffer) in buffers.named.iter().enumerate() {
        if let Some((base, offset)) = buffer.alias {
            emit!(
                buffer_lines,
                "    void* buffer_{index} = static_cast<char*>(buffer_{base}) + {offset};"
            );
        } else {
            emit!(
                buffer_lines,
                "    void* buffer_{index} = device_zeroed({bytes}ULL);",
                bytes = buffer.bytes
            );
        }
    }

    let frees = buffers
        .named
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.alias.is_none())
        .map(|(index, _)| format!("    cudaFree(buffer_{index});"))
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
            "[--seed N] [--ids] [--raw] [--tokens] [--stream] [--time] NEW_TOKENS (PROMPT... | --prompts FILE)",
            "2",
            String::new(),
            String::new(),
        ),
        (Some(step), Some((index, bytes, element))) => {
            // A source file starts with its grid when the step reads one, and may be shorter than its buffer, whose rest stays zero, and a length input says how much it held; a position says where the prompt's starred token stands, and rows how many times it stands there.
            let header = grid.map_or(0, |(_, bytes)| bytes);
            let mut code = format!(
                "    {{\n        if (prompts.size() != 1) {{\n            fprintf(stderr, \"a model with an encode step decodes one prompt\\n\");\n            return 2;\n        }}\n        const std::vector<char> file = read_file(source_path, {header}ULL + {bytes}ULL + 1ULL, false);\n        if (file.size() < {header}ULL || file.size() > {header}ULL + {bytes}ULL) {{\n            fprintf(stderr, \"%s holds %zu bytes, where the encode step reads %llu to %llu\\n\", source_path, file.size(), {header}ULL, {header}ULL + {bytes}ULL);\n            return 2;\n        }}\n        const std::vector<char> source(file.begin() + {header}, file.end());\n        check_cuda(cudaMemcpy(buffer_{index}, source.data(), source.size(), cudaMemcpyHostToDevice), \"cudaMemcpy\");\n"
            );
            if let Some((grid, bytes)) = grid {
                emit!(
                    code,
                    "        check_cuda(cudaMemcpy(buffer_{grid}, file.data(), {bytes}ULL, cudaMemcpyHostToDevice), \"cudaMemcpy\");"
                );
            }

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
                "[--seed N] [--ids] [--raw] [--tokens] [--stream] [--time] NEW_TOKENS SOURCE (PROMPT... | --prompts FILE)",
                "3",
                "    const char* source_path = encode_only ? nullptr : argv[argument++];\n"
                    .to_owned(),
                code,
            )
        }
        (Some(_), None) => {
            return Err(Error::invalid_argument(
                "an encode step reads a source besides the caches",
            ));
        }
    };
    let samples = all.iter().any(|(_, step)| step.program.has_generators());
    let tokenizer = if decoding.tokenizer.is_some() {
        format!("{prefix}.tokenizer")
    } else {
        String::new()
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
            ("tokenizer", &tokenizer),
            ("graph_replay_code", include_str!("graph_replay.cu.in")),
            ("tokenizer_code", include_str!("tokenizer.cu.in")),
            ("prompt_code", include_str!("prompts.cu.in")),
            ("paged_step_code", include_str!("paged_step.cu.in")),
            ("samples", if samples { "true" } else { "false" }),
            ("weights", &weights),
            ("seed", seed),
            ("structs", &struct_lines),
            ("buffers", &buffer_lines),
            ("allocations", &allocations),
            ("encode", &encode_code),
            ("steps", &step_lines),
            ("finish", &finish_lines),
            ("staging_words", &staging.to_string()),
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
    use crate::nn::{DecodeStep, PACKED_ORDER, TensorGroup};

    #[test]
    fn test_the_loop_packs_a_step_as_decode_step_unpacks_it() {
        let step = DecodeStep::meta(2, 1, 4).unwrap();
        let names = step
            .named_tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert_eq!(names, [STEP_INPUT]);
        let template = include_str!("paged_step.cu.in");
        let pack = &template[template.find("pack(const Step& step)").unwrap()..];
        let places = PACKED_ORDER
            .iter()
            .map(|name| pack.find(&format!("&step.{name}")).unwrap())
            .collect::<Vec<_>>();
        assert!(places.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
