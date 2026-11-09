// A loop that generates through a decoder step's program with nothing of Singe, writing each step's paging tensors as `PageTable` would for one sequence, after an encode step if the model has one.

use crate::{
    backend::cuda::{
        export::{
            builder::span_bytes,
            program::{CudaProgram, StateKind},
            writer::{Renderer, fill_template},
        },
        kernel_plan::value_type,
    },
    data_type::DataType,
    error::{Error, Result},
    nn::PageLayout,
    utility::emit,
};

// What a decoder step reads besides the state it carries from one step to the next, in the order `DecodeStep` names them.
const STEP_INPUTS: [&str; 9] = [
    "tokens",
    "positions",
    "write_pages",
    "write_slots",
    "query_offsets",
    "page_offsets",
    "pages",
    "last_page_lengths",
    "last_rows",
];

// The C++ that turns one logit of `data_type` into a `float`.
fn logit_conversion(data_type: DataType) -> Result<&'static str> {
    Ok(match data_type {
        DataType::F32 => "    return value;\n",
        DataType::F64 => "    return static_cast<float>(value);\n",
        DataType::Bf16 => {
            "    const uint32_t bits = static_cast<uint32_t>(value) << 16;\n    float result;\n    memcpy(&result, &bits, sizeof(result));\n    return result;\n"
        }
        DataType::F16 => {
            "    __half half;\n    memcpy(&half, &value, sizeof(half));\n    return __half2float(half);\n"
        }
        _ => return Err(Error::unsupported_data_type("generation loop logits")),
    })
}

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

// Writes a generation loop over `decode`, a decoder step of one sequence whose paged cache `layout` describes, through the program declared by `prefix`.
// With `encode`, the loop first runs that step once on a source the command line names, which writes what the decoder steps read, such as cross-attention keys and values.
// The steps' inputs of one name are one buffer.
pub fn generation_loop_source(
    prefix: &str,
    layout: PageLayout,
    decode: LoopStep<'_>,
    encode: Option<LoopStep<'_>>,
    weights_bytes: usize,
) -> Result<String> {
    if layout.sequences != 1 {
        return Err(Error::invalid_argument(
            "a generation loop runs a step of one sequence",
        ));
    }

    let program = decode.program;
    let [logits] = program.outputs.as_slice() else {
        return Err(Error::invalid_argument(
            "a generation loop reads a step with logits as its only output",
        ));
    };
    let steps = encode.iter().chain([&decode]);
    if steps.clone().any(|step| {
        step.program
            .state
            .iter()
            .any(|state| matches!(state.kind, StateKind::Generator { .. }))
    }) {
        return Err(Error::invalid_argument(
            "a generation loop takes the most likely token, and draws nothing",
        ));
    }

    // One buffer per input name across the steps, and one per output.
    let mut allocations = String::new();
    let mut buffers = Vec::<(String, u64)>::new();
    let mut buffer_of = |name: &str, bytes: u64| -> Result<usize> {
        if let Some(index) = buffers.iter().position(|(other, _)| other == name) {
            if buffers[index].1 != bytes {
                return Err(Error::invalid_argument(
                    "steps of a generation loop disagree on an input's size",
                ));
            }
            return Ok(index);
        }

        buffers.push((name.to_owned(), bytes));
        Ok(buffers.len() - 1)
    };
    let mut buffer_lines = String::new();
    let mut source = None;
    let mut struct_lines = String::new();
    for (step, variable) in encode
        .iter()
        .map(|step| (step, "encode_"))
        .chain([(&decode, "")])
    {
        emit!(
            struct_lines,
            "    {types}_inputs {variable}inputs;\n    {types}_outputs {variable}outputs;",
            types = step.types
        );
        for value in &step.program.inputs {
            let bytes = span_bytes(value.data_type, &value.layout)?;
            let index = buffer_of(&value.graph_name, bytes)?;
            let ty = value_type(value.data_type)?;
            let qualifier = if value.written { "" } else { "const " };
            emit!(
                allocations,
                "    {variable}inputs.{name} = static_cast<{qualifier}{ty}*>(buffer_{index});",
                name = value.name
            );
            let is_step = STEP_INPUTS.contains(&value.graph_name.as_str());
            if !variable.is_empty() && !is_step && !value.graph_name.starts_with("cache.") {
                if source.is_some() {
                    return Err(Error::invalid_argument(
                        "an encode step reads one source besides the caches",
                    ));
                }

                source = Some((index, bytes));
            }
        }
        for value in &step.program.outputs {
            let bytes = span_bytes(value.data_type, &value.layout)?;
            let index = buffer_of(&format!("{variable}output {}", value.graph_name), bytes)?;
            let ty = value_type(value.data_type)?;
            emit!(
                allocations,
                "    {variable}outputs.{name} = static_cast<{ty}*>(buffer_{index});",
                name = value.name
            );
        }
    }

    for (index, (_, bytes)) in buffers.iter().enumerate() {
        emit!(
            buffer_lines,
            "    void* buffer_{index} = device_zeroed({bytes}ULL);"
        );
    }

    let logits_index = buffers
        .iter()
        .position(|(name, _)| *name == format!("output {}", logits.graph_name))
        .ok_or_else(|| Error::missing_output("generation loop logits"))?;

    let step_buffer = |name: &str| -> Result<String> {
        let found = program
            .inputs
            .iter()
            .find(|input| input.graph_name == name)
            .ok_or_else(|| Error::missing_input("generation loop step input"))?;
        if found.data_type != DataType::I64 {
            return Err(Error::invalid_argument(
                "a generation loop writes 64-bit step inputs",
            ));
        }

        let index = buffers
            .iter()
            .position(|(other, _)| other == name)
            .ok_or_else(|| Error::missing_input("generation loop step input"))?;
        Ok(format!("buffer_{index}"))
    };
    let step_buffers = STEP_INPUTS
        .iter()
        .map(|name| step_buffer(name))
        .collect::<Result<Vec<_>>>()?;
    let frees = (0..buffers.len())
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
    let (usage, first_token, encode_code) = match (encode, source) {
        (None, _) => ("NEW_TOKENS TOKEN...", "2", String::new()),
        (Some(step), Some((index, bytes))) => (
            "NEW_TOKENS SOURCE TOKEN...",
            "3",
            format!(
                "    {{\n        // The source's bytes, as the encode step reads them.\n        const std::vector<char> source = read_file(argv[2], {bytes}ULL);\n        check_cuda(cudaMemcpy(buffer_{index}, source.data(), source.size(), cudaMemcpyHostToDevice), \"cudaMemcpy\");\n        check({run}(state, &encode_inputs, &encode_outputs, nullptr), \"{run}\");\n        unsigned int word = 0;\n        check({finish}(state, nullptr, &word), \"{finish}\");\n        if (word != 0) {{\n            fprintf(stderr, \"a checking kernel reported %u\\n\", word);\n            return 1;\n        }}\n    }}\n",
                run = step.run,
                finish = step.finish,
            ),
        ),
        (Some(_), None) => {
            return Err(Error::invalid_argument(
                "an encode step reads a source besides the caches",
            ));
        }
    };
    let logit_type = value_type(logits.data_type)?;
    Ok(fill_template(
        include_str!("generate.cu.in"),
        &[
            ("prefix", prefix),
            ("usage", usage),
            ("first_token", first_token),
            ("page_size", &layout.page_size.to_string()),
            ("capacity", &layout.capacity.to_string()),
            ("table_length", &layout.table_length.to_string()),
            ("logit_type", logit_type),
            ("logit_count", &logits.layout.element_count().to_string()),
            ("logit_conversion", logit_conversion(logits.data_type)?),
            ("weights", &weights),
            ("structs", &struct_lines),
            ("buffers", &buffer_lines),
            ("allocations", &allocations),
            ("encode", &encode_code),
            ("run", decode.run),
            ("finish", decode.finish),
            ("tokens", &step_buffers[0]),
            ("positions", &step_buffers[1]),
            ("write_pages", &step_buffers[2]),
            ("write_slots", &step_buffers[3]),
            ("query_offsets", &step_buffers[4]),
            ("page_offsets", &step_buffers[5]),
            ("pages", &step_buffers[6]),
            ("last_page_lengths", &step_buffers[7]),
            ("last_rows", &step_buffers[8]),
            ("logits", &format!("buffer_{logits_index}")),
            ("frees", &frees),
        ],
    ))
}

impl Renderer<'_> {
    // The generation loop, for a program exported with a page layout, which takes one token for one sequence each step.
    pub fn generation_loop(&self, weights_bytes: usize) -> Result<Option<String>> {
        let Some(layout) = self.program.decoding else {
            return Ok(None);
        };
        let prefix = self.prefix;
        let (run, finish) = (format!("{prefix}_run"), format!("{prefix}_finish"));
        generation_loop_source(
            prefix,
            layout,
            LoopStep {
                program: self.program,
                types: prefix,
                run: &run,
                finish: &finish,
            },
            None,
            weights_bytes,
        )
        .map(Some)
    }
}
