use std::{
    collections::{BTreeMap, HashMap},
    hash::BuildHasher,
    path::{Path, PathBuf},
};
#[cfg(feature = "cuda_13_4")]
use std::{
    fs, iter,
    process::{self, Command},
    slice,
    time::Duration,
};

use singe::{
    DataType, Device, Tensor,
    generation::{
        CachedDecoder, Decoder, DecoderOptions, GenerationOptions, PromptChunks, RunningBatch,
        generate, generate_batch,
    },
    nn::{DecoderModel, Model},
    random::Generator,
    tensor::SamplingOptions,
};
#[cfg(feature = "cuda_13_4")]
use singe::{
    Shape,
    backend::cuda::{
        LoweringPolicy,
        export::{CudaExport, CudaProgram},
        tuning::{
            GraphTuneOptions, Measurement, TunedChoice, TunedRecord, TuningRecords,
            fused_attention_key, merged_projections_key, tune_graph,
        },
    },
    backend::registry::Operation,
    generation::DecoderStepGraph,
    graph::{Graph, GraphCompileOptions, GraphCudaOptions},
    nn::{PageTable, TensorGroup},
    testing::{CompiledProgram, ProgramInstance, build_generation_loop, program_target},
};

use crate::{Tolerance, assert_matches, integers};

const fn type_key(data_type: DataType) -> &'static str {
    match data_type {
        DataType::F32 => "f32",
        DataType::F16 => "f16",
        _ => "bf16",
    }
}

pub fn host_rows(logits: &Tensor) -> Vec<Vec<f32>> {
    let width = usize::try_from(logits.shape().dimensions()[1]).unwrap();
    logits
        .cast(DataType::F32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap()
        .chunks_exact(width)
        .map(<[f32]>::to_vec)
        .collect()
}

#[must_use]
pub fn host_logits(logits: &Tensor) -> Vec<f32> {
    host_rows(logits).remove(0)
}

pub fn length(values: &[impl Sized]) -> i64 {
    i64::try_from(values.len()).unwrap()
}

#[must_use]
pub fn best_token(logits: &[f32]) -> i64 {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(best, _)| i64::try_from(best).unwrap())
        .unwrap()
}

pub fn assert_one_specialization<L: DecoderModel>(decoder: &CachedDecoder<L>) {
    let stats = decoder.executor().unwrap().cache_stats();
    assert_eq!(stats.entries(), 1, "{stats:?}");
    assert_eq!(stats.evictions(), 0, "{stats:?}");
}

pub fn assert_chunks_compiled<L: DecoderModel>(decoder: &CachedDecoder<L>, steps: usize) {
    assert_eq!(decoder.executors().count(), steps);
    for executor in decoder.executors() {
        let stats = executor.cache_stats();
        assert_eq!(stats.entries(), 1, "{stats:?}");
        assert_eq!(stats.evictions(), 0, "{stats:?}");
    }
}

#[must_use]
pub fn decoder<L: DecoderModel>(
    model: &Model<L>,
    options: DecoderOptions,
    compiled: bool,
) -> CachedDecoder<L> {
    let mut decoder = CachedDecoder::create(model, options).unwrap();
    if compiled {
        decoder.compile().unwrap();
    }
    decoder
}

#[derive(Debug, Clone)]
pub struct Reference {
    pub prompt: Vec<i64>,
    pub other_prompt: Vec<i64>,
    pub split: usize,
    pub prompt_logits: Vec<f32>,
    pub split_logits: Vec<f32>,
    pub steps: Vec<(Vec<f32>, i64)>,
}

impl Reference {
    #[must_use]
    pub fn read(directory: &Path, data_type: DataType) -> Self {
        let reference = crate::Reference::load(directory);
        let key = type_key(data_type);
        let logits = |name: &str| reference.values(&format!("{key}.{name}"));
        let step_logits = logits("step_logits");
        let tokens = integers(&reference.metadata()["tokens"][key]);
        let width = step_logits.len() / tokens.len();
        Self {
            prompt: reference.integers("prompt"),
            other_prompt: reference.integers("other_prompt"),
            split: usize::try_from(reference.metadata()["split"].as_i64().unwrap()).unwrap(),
            prompt_logits: logits("prompt_logits"),
            split_logits: logits("split_logits"),
            steps: step_logits
                .chunks(width)
                .map(<[f32]>::to_vec)
                .zip(tokens)
                .collect(),
        }
    }

    #[must_use]
    pub fn tokens(&self) -> Vec<i64> {
        self.steps.iter().map(|(_, token)| *token).collect()
    }
}

pub struct Suite<L> {
    pub references: &'static str,
    pub task: &'static str,
    pub family: &'static str,
    pub models: &'static [&'static str],
    pub load: fn(&Path, Device, DataType) -> Model<L>,
    pub f32_tolerance: Tolerance,
    pub bf16_tolerance: Tolerance,
}

impl<L: DecoderModel> Suite<L> {
    #[must_use]
    pub fn directory(&self, name: &str) -> PathBuf {
        Path::new(self.references)
            .join(self.task)
            .join(self.family)
            .join(name)
    }

    fn load(&self, name: &str, device: Device, data_type: DataType) -> (Model<L>, Reference) {
        let directory = self.directory(name);
        let reference = Reference::read(&directory, data_type);
        ((self.load)(&directory, device, data_type), reference)
    }

    pub fn check_logits(&self, device: Device) {
        for (data_type, tolerance) in [
            (DataType::F32, self.f32_tolerance),
            (DataType::Bf16, self.bf16_tolerance),
        ] {
            for name in self.models {
                let (model, reference) = self.load(name, device, data_type);
                check_prompts(&model, &reference, tolerance);
                for compiled in [false, true] {
                    for options in [
                        DecoderOptions::new(1, 32),
                        // Pages of another size only change where the slots live.
                        DecoderOptions::new(1, 32).with_page_size(5),
                    ] {
                        check_steps(&model, &reference, tolerance, options, compiled);
                    }
                }
            }
        }
    }

    #[cfg(feature = "cuda_13_4")]
    pub fn check_tuned(&self, device: Device) {
        self.check_tuned_model(self.models[0], device);
    }

    #[cfg(feature = "cuda_13_4")]
    pub fn check_tuned_model(&self, name: &str, device: Device) {
        let data_type = DataType::Bf16;
        let (model, reference) = self.load(name, device, data_type);
        let options = DecoderOptions::new(1, 32);
        let mut decoder = CachedDecoder::create(&model, options).unwrap();
        let step = decoder.step_graph().unwrap();
        let records = tune_step(&step.graph, &step.inputs, device, Measurement::GraphReplay);
        assert!(!records.is_empty(), "{}", self.family);
        decoder
            .compile_with(GraphCudaOptions::new().with_tuning_records(records))
            .unwrap();
        let decoder = follow_steps(decoder, &reference, self.bf16_tolerance, true);
        assert_one_specialization(&decoder);

        let Some(architecture) = program_target(device) else {
            return;
        };
        let records = tune_step(&step.graph, &step.inputs, device, Measurement::Program);
        let program = CudaExport::new(architecture)
            .with_tuning_records(records)
            .export(&step.graph)
            .unwrap();
        assert!(
            program.record_coverage.used > 0,
            "{}: {:?}",
            self.family,
            program.record_coverage
        );
        let compiled =
            CompiledProgram::create(&program, &format!("{}_tuned", self.family)).unwrap();
        let generating = ProgramDecoder::create(&compiled, &step, options);
        follow_steps(generating, &reference, self.bf16_tolerance, false);
        self.check_running_program(name, device);
        self.check_tuned_chunks(name, device);
        self.check_tuned_program_steps(name, device);
        self.check_merged_projections(name, device);
        self.check_fused_attention(name, device);
    }

    #[cfg(feature = "cuda_13_4")]
    fn check_fused_attention(&self, name: &str, device: Device) {
        let mut records = TuningRecords::new(None);
        records.insert(
            fused_attention_key(),
            TunedRecord {
                choice: TunedChoice::FusedAttention,
                default_ns: None,
                best_ns: 0,
                bytes: 0,
            },
        );
        let (model, reference) = self.load(name, device, DataType::Bf16);
        let options = DecoderOptions::new(1, 32);
        let mut decoder = CachedDecoder::create(&model, options).unwrap();
        let step = decoder.step_graph().unwrap();
        decoder
            .compile_with(GraphCudaOptions::new().with_tuning_records(records.clone()))
            .unwrap();
        follow_steps(decoder, &reference, self.bf16_tolerance, true);

        let Some(architecture) = program_target(device) else {
            return;
        };
        let program = CudaExport::new(architecture)
            .with_tuning_records(records.clone())
            .export(&step.graph)
            .unwrap();
        let source_of = |program: &CudaProgram| program.render("model").unwrap().source;
        let fused = source_of(&program).contains("rotary_paged_attention");
        eprintln!("{}: fused attention {fused}", self.family);
        // What is left of the cache write module still runs as one kernel beside it.
        let writes_cache = source_of(&program).contains("rotary_cache_write");
        assert!(!fused || writes_cache, "{}", self.family);
        let compiled =
            CompiledProgram::create(&program, &format!("{}_fused", self.family)).unwrap();
        let generating = ProgramDecoder::create(&compiled, &step, options);
        follow_steps(generating, &reference, self.bf16_tolerance, false);

        // The fused kernel takes half precision only, so that is what this generates in.
        let (model, reference) = self.load(name, device, DataType::Bf16);
        let options = running_options(&reference);
        let mut decoder = CachedDecoder::create(&model, options).unwrap();
        decoder
            .compile_with(GraphCudaOptions::new().with_tuning_records(records.clone()))
            .unwrap();
        let expected = alone(&model, &reference);
        check_running(&mut decoder, &reference, &expected);
        if self.family == "qwen3" {
            assert!(fused);
        }
    }

    #[cfg(feature = "cuda_13_4")]
    fn check_merged_projections(&self, name: &str, device: Device) {
        let (model, reference) = self.load(name, device, DataType::Bf16);
        let options = DecoderOptions::new(1, 32);
        let mut decoder = CachedDecoder::create(&model, options).unwrap();
        let step = decoder.step_graph().unwrap();
        let matmuls = |merged: bool| {
            step.graph
                .compile_with_options(GraphCompileOptions::new().with_merged_projections(merged))
                .unwrap()
                .ir()
                .operations()
                .iter()
                .filter(|operation| operation.opcode() == Some(Operation::MatMul))
                .count()
        };
        let (apart, together) = (matmuls(false), matmuls(true));
        eprintln!("{}: {apart} matmuls apart, {together} merged", self.family);
        let mut records = TuningRecords::new(None);
        records.insert(
            merged_projections_key(),
            TunedRecord {
                choice: TunedChoice::MergedProjections,
                default_ns: None,
                best_ns: 0,
                bytes: 0,
            },
        );
        decoder
            .compile_with(GraphCudaOptions::new().with_tuning_records(records.clone()))
            .unwrap();
        follow_steps(decoder, &reference, self.bf16_tolerance, true);

        let Some(architecture) = program_target(device) else {
            return;
        };
        let program = CudaExport::new(architecture)
            .with_tuning_records(records)
            .export(&step.graph)
            .unwrap();
        eprintln!(
            "{}: merged keeps the cache write module: {}",
            self.family,
            program
                .render("model")
                .unwrap()
                .source
                .contains("rotary_cache")
        );
        let compiled =
            CompiledProgram::create(&program, &format!("{}_merged", self.family)).unwrap();
        let generating = ProgramDecoder::create(&compiled, &step, options);
        follow_steps(generating, &reference, self.bf16_tolerance, false);
    }

    #[cfg(feature = "cuda_13_4")]
    fn check_running_program(&self, name: &str, device: Device) {
        let (model, reference) = self.load(name, device, DataType::F32);
        let options = running_options(&reference);
        let step = CachedDecoder::create(&model, options)
            .unwrap()
            .step_graph()
            .unwrap();
        let program = CudaExport::new(program_target(device).unwrap())
            .export(&step.graph)
            .unwrap();
        let compiled =
            CompiledProgram::create(&program, &format!("{}_running", self.family)).unwrap();
        let mut decoder = ProgramDecoder::create(&compiled, &step, options);
        let expected = alone(&model, &reference);
        check_running(&mut decoder, &reference, &expected);

        // With a step per chunk in one program, a joining prompt splits into none of its tokens alone, and weights exist once.
        let source = CachedDecoder::create(&model, options).unwrap();
        let tokens = |chunk: usize| format!("t{}", options.sequences + chunk);
        let graphs = iter::once(0)
            .chain(PROMPT_CHUNKS)
            .map(|chunk| (tokens(chunk), source.chunk_step_graph(chunk).unwrap()))
            .collect::<Vec<_>>();
        let named = graphs
            .iter()
            .map(|(name, step)| (name.as_str(), &step.graph))
            .collect::<Vec<_>>();
        let export = CudaExport::new(program_target(device).unwrap());
        let set = export.export_steps(&named).unwrap();
        let shared = set
            .entries
            .iter()
            .map(|entry| entry.shared.iter().flatten().count())
            .sum::<usize>();
        assert!(shared > 0, "{}", self.family);
        let compiled =
            CompiledProgram::create_set(&set, &format!("{}_steps", self.family)).unwrap();
        assert_eq!(
            compiled.weights_bytes(),
            compiled.program().render("model").unwrap().weights.len()
        );
        let mut decoder = ProgramDecoder::create(&compiled, &step, options);
        check_running(&mut decoder, &reference, &expected);
        assert_eq!(decoder.split_steps, 0);
    }

    #[cfg(feature = "cuda_13_4")]
    fn check_tuned_program_steps(&self, name: &str, device: Device) {
        let (model, reference) = self.load(name, device, DataType::F32);
        let options = running_options(&reference);
        let source = CachedDecoder::create(&model, options).unwrap();
        let step = source.step_graph().unwrap();
        let chunk = source.chunk_step_graph(PROMPT_CHUNKS[0]).unwrap();
        let mut records = tune_step(&step.graph, &step.inputs, device, Measurement::Program);
        records
            .merge(tune_step(
                &chunk.graph,
                &chunk.inputs,
                device,
                Measurement::Program,
            ))
            .unwrap();
        let names = [
            format!("t{}", options.sequences),
            format!("t{}", options.sequences + PROMPT_CHUNKS[0]),
        ];
        let set = CudaExport::new(program_target(device).unwrap())
            .with_tuning_records(records)
            .export_steps(&[
                (names[0].as_str(), &step.graph),
                (names[1].as_str(), &chunk.graph),
            ])
            .unwrap();
        assert!(
            set.entries
                .iter()
                .any(|entry| entry.program.record_coverage.used > 0)
        );
        let compiled =
            CompiledProgram::create_set(&set, &format!("{}_tuned_steps", self.family)).unwrap();
        let mut decoder = ProgramDecoder::create(&compiled, &step, options);
        check_running(&mut decoder, &reference, &alone(&model, &reference));
        assert_eq!(decoder.split_steps, 0);
    }

    #[cfg(feature = "cuda_13_4")]
    fn check_tuned_chunks(&self, name: &str, device: Device) {
        let (model, reference) = self.load(name, device, DataType::F32);
        let options = running_options(&reference).with_prompt_chunks(PromptChunks {
            smallest: PROMPT_CHUNKS[0],
            largest: PROMPT_CHUNKS[0],
        });
        let mut decoder = CachedDecoder::create(&model, options).unwrap();
        let tune = |chunk| {
            let step = decoder.chunk_step_graph(chunk).unwrap();
            tune_step(&step.graph, &step.inputs, device, Measurement::GraphReplay)
        };
        let mut records = tune(0);
        records.merge(tune(PROMPT_CHUNKS[0])).unwrap();
        let cuda = GraphCudaOptions::new().with_tuning_records(records);
        decoder.compile_with(cuda).unwrap();
        check_running(&mut decoder, &reference, &alone(&model, &reference));
        assert_chunks_compiled(&decoder, 2);
    }

    pub fn check_decoding(&self, device: Device) {
        for (data_type, tolerance) in [
            (DataType::F32, self.f32_tolerance),
            (DataType::Bf16, self.bf16_tolerance),
        ] {
            let (model, reference) = self.load(self.models[0], device, data_type);
            for compiled in [false, true] {
                // Half precision can break a tiny model's near ties either way, so exact tokens are checked in F32 alone.
                if data_type == DataType::F32 {
                    check_generation(&model, &reference, compiled);
                }

                check_batch(
                    &model,
                    [&reference.prompt, &reference.other_prompt],
                    tolerance,
                    compiled,
                );
                check_held_sequence(&model, &reference, compiled);
                // Half precision can break a tiny model's near ties either way.
                if data_type == DataType::F32 {
                    let mut running = decoder(&model, running_options(&reference), compiled);
                    check_running(&mut running, &reference, &alone(&model, &reference));
                    if compiled {
                        assert_one_specialization(&running);
                        let options =
                            running_options(&reference).with_prompt_chunks(PromptChunks {
                                smallest: PROMPT_CHUNKS[0],
                                largest: PROMPT_CHUNKS[1],
                            });
                        let mut chunked = CachedDecoder::create(&model, options).unwrap();
                        chunked.compile().unwrap();
                        check_running(&mut chunked, &reference, &alone(&model, &reference));
                        assert_chunks_compiled(&chunked, PROMPT_CHUNKS.len() + 1);
                    }
                }

                check_reuse(&model, &reference.prompt, compiled);
                check_sampling_repeats(&model, &reference.prompt, compiled);
            }

            check_shared_step(&model, &reference.prompt);
            check_failed_step(&model);
        }
        #[cfg(feature = "cuda_13_4")]
        if matches!(device, Device::Cuda(_)) {
            self.check_generation_loop(device);
        }
    }

    #[cfg(feature = "cuda_13_4")]
    fn check_generation_loop(&self, device: Device) {
        let Some(architecture) = program_target(device) else {
            return;
        };
        let (model, reference) = self.load(self.models[0], device, DataType::F32);
        let options = DecoderOptions::new(1, 32);
        let step = CachedDecoder::create(&model, options)
            .unwrap()
            .step_graph()
            .unwrap();
        let program = CudaExport::new(architecture)
            .with_decoding(step.layout)
            .export(&step.graph)
            .unwrap();
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../target/programs/{}-generate-{}",
            self.family,
            process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        let executable = build_generation_loop(&program, &directory).unwrap();
        let tokens = reference.tokens();
        let output = Command::new(&executable)
            .arg(tokens.len().to_string())
            .args(reference.prompt.iter().map(i64::to_string))
            .current_dir(&directory)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}: {}",
            self.family,
            String::from_utf8_lossy(&output.stderr)
        );
        let generated = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| line.parse::<i64>().unwrap())
            .collect::<Vec<_>>();
        let compiled =
            CompiledProgram::create(&program, &format!("{}_generate", self.family)).unwrap();
        let mut decoder = ProgramDecoder::create(&compiled, &step, options);
        let expected = generate(
            &mut decoder,
            &reference.prompt,
            &GenerationOptions {
                max_new_tokens: tokens.len(),
                ..GenerationOptions::default()
            },
            None,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(generated, expected, "{}", self.family);
        assert_eq!(generated, tokens, "{}", self.family);
        let _ = fs::remove_dir_all(&directory);
    }
}

#[cfg(feature = "cuda_13_4")]
pub struct ProgramDecoder {
    instance: ProgramInstance,
    steps: BTreeMap<usize, (String, Vec<String>)>,
    caches: HashMap<String, Tensor>,
    pages: PageTable,
    device: Device,
    latest: Vec<Vec<f32>>,
    split_steps: usize,
}

#[cfg(feature = "cuda_13_4")]
impl ProgramDecoder {
    #[must_use]
    pub fn create(
        compiled: &CompiledProgram,
        step: &DecoderStepGraph,
        options: DecoderOptions,
    ) -> Self {
        let device = step.logits.device();
        let caches = step
            .inputs
            .iter()
            .filter(|(name, _)| name.starts_with("cache."))
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();
        let steps = compiled
            .steps()
            .iter()
            .map(|(name, program)| {
                let tokens = name
                    .strip_prefix('t')
                    .map_or(options.sequences, |tokens| tokens.parse().unwrap());
                let inputs = program
                    .inputs
                    .iter()
                    .map(|input| input.graph_name.clone())
                    .collect();
                (tokens, (name.clone(), inputs))
            })
            .collect();
        Self {
            instance: compiled.instance().unwrap(),
            steps,
            caches,
            pages: options.page_table().unwrap(),
            device,
            latest: vec![Vec::new(); options.sequences],
            split_steps: 0,
        }
    }

    fn run(&mut self, total: usize, row_tokens: &[&[i64]]) -> singe::Result<()> {
        let step = self.pages.step_padded(self.device, row_tokens, total)?;
        let step_inputs = step.named_tensors().into_iter().collect::<HashMap<_, _>>();
        let (name, inputs) = &self.steps[&total];
        let operands = inputs
            .iter()
            .map(|input| {
                step_inputs
                    .get(input)
                    .unwrap_or_else(|| &self.caches[input])
                    .clone()
            })
            .collect::<Vec<_>>();
        let outputs = if name.is_empty() {
            self.instance.run(&operands, 1).unwrap()
        } else {
            self.instance.run_step(name, &operands, 1).unwrap()
        };
        let rows = host_rows(&outputs[0]);
        for (row, tokens) in row_tokens.iter().enumerate() {
            if !tokens.is_empty() {
                self.latest[row].clone_from(&rows[row]);
            }
        }
        Ok(())
    }
}

#[cfg(feature = "cuda_13_4")]
impl Decoder for ProgramDecoder {
    fn sequences(&self) -> usize {
        self.pages.sequences()
    }

    fn step(&mut self, tokens: &[&[i64]]) -> singe::Result<Tensor> {
        // A step pads to the smallest exported one that holds it, and one larger than all of them runs a token per row at a time.
        let total = PageTable::step_tokens(tokens);
        if let Some(&size) = self.steps.range(total..).map(|(size, _)| size).next() {
            self.run(size, tokens)?;
        } else {
            self.split_steps += 1;
            let longest = tokens.iter().map(|tokens| tokens.len()).max().unwrap_or(0);
            for position in 0..longest {
                let row_tokens = tokens
                    .iter()
                    .map(|tokens| tokens.get(position).map_or(&[][..], slice::from_ref))
                    .collect::<Vec<_>>();
                self.run(self.pages.sequences(), &row_tokens)?;
            }
        }

        let width = self.latest.iter().map(Vec::len).max().unwrap_or(0);
        let values = self
            .latest
            .iter()
            .flat_map(|row| {
                row.iter()
                    .copied()
                    .chain(iter::repeat_n(0.0, width - row.len()))
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(
            self.device,
            Shape::create([length(&self.latest), i64::try_from(width).unwrap()])?,
            values,
        )
    }

    fn largest_chunk(&self) -> Option<usize> {
        let sequences = self.pages.sequences();
        self.steps.keys().last().map(|total| total - sequences)
    }

    fn reset(&mut self) {
        self.pages.reset();
    }

    fn release(&mut self, sequence: usize) -> singe::Result<()> {
        self.pages.release(sequence)
    }
}

#[cfg(feature = "cuda_13_4")]
#[must_use]
pub fn tune_step<S: BuildHasher>(
    graph: &Graph,
    inputs: &HashMap<String, Tensor, S>,
    device: Device,
    measurement: Measurement,
) -> TuningRecords {
    let records = tune_graph(
        graph,
        inputs,
        device,
        &GraphTuneOptions {
            policy: LoweringPolicy::default(),
            // A candidate takes longer to create as a program, so a budget that lets a choice point compare candidates is larger.
            choice_budget: match measurement {
                Measurement::GraphReplay => Duration::from_millis(300),
                Measurement::Program => Duration::from_secs(4),
            },
            runs: 2,
            schedules: false,
            measurement,
            build_directory: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../target/programs/tuning"),
            memory_limit: None,
            // A plan builds in seconds as a program, so one budget does not suit both.
            plan_budget: match measurement {
                Measurement::GraphReplay => Duration::from_millis(500),
                Measurement::Program => Duration::from_secs(6),
            },
            plan_seed: 0,
            // Two runs on a tiny model say little about speed, and these tests want records to read.
            plan_speed: false,
            ..GraphTuneOptions::default()
        },
    )
    .unwrap();
    // Nothing a tiny model can choose fails or disagrees, alone or as a whole, so anything that does is a bug.
    let report = records.report().unwrap();
    let problems = report.problems().collect::<Vec<_>>();
    assert!(
        problems.is_empty(),
        "{measurement:?}: {problems:?}\n{report}"
    );
    for check in &report.checks {
        assert!(check.default_failure.is_none(), "{report}");
        assert!(!check.default_disagreed, "{report}");
        assert!(check.dropped.is_empty(), "{report}");
    }
    records
}

pub fn check_prompts<L: DecoderModel>(
    model: &Model<L>,
    reference: &Reference,
    tolerance: Tolerance,
) {
    let options = DecoderOptions::new(1, 32);
    let mut whole = decoder(model, options, false);
    assert_matches(
        &host_logits(&whole.step(&[&reference.prompt]).unwrap()),
        &reference.prompt_logits,
        tolerance,
        "whole prompt",
    );
    let mut parts = decoder(model, options, false);
    let (first, second) = reference.prompt.split_at(reference.split);
    parts.step(&[first]).unwrap();
    assert_matches(
        &host_logits(&parts.step(&[second]).unwrap()),
        &reference.split_logits,
        tolerance,
        "split prompt",
    );
}

pub fn check_steps<L: DecoderModel>(
    model: &Model<L>,
    reference: &Reference,
    tolerance: Tolerance,
    options: DecoderOptions,
    compiled: bool,
) {
    let decoder = decoder(model, options, compiled);
    let decoder = follow_steps(decoder, reference, tolerance, compiled);
    if compiled {
        assert_one_specialization(&decoder);
    }
}

fn follow_steps<D: Decoder>(
    mut decoder: D,
    reference: &Reference,
    tolerance: Tolerance,
    compiled: bool,
) -> D {
    let mut logits = decoder.step(&[&reference.prompt]).unwrap();
    for (number, (expected, token)) in reference.steps.iter().enumerate() {
        assert_matches(
            &host_logits(&logits),
            expected,
            tolerance,
            &format!("step {number}, compiled: {compiled}"),
        );
        logits = decoder.step(&[&[*token]]).unwrap();
    }
    decoder
}

pub fn check_generation<L: DecoderModel>(model: &Model<L>, reference: &Reference, compiled: bool) {
    let mut decoder = decoder(model, DecoderOptions::new(1, 32), compiled);
    let tokens = reference.tokens();
    let options = GenerationOptions {
        max_new_tokens: tokens.len(),
        ..GenerationOptions::default()
    };
    let generated = generate(&mut decoder, &reference.prompt, &options, None, |_| Ok(())).unwrap();
    assert_eq!(generated, tokens, "compiled: {compiled}");
    if compiled {
        assert_one_specialization(&decoder);
    }
}

pub fn check_batch<L: DecoderModel>(
    model: &Model<L>,
    prompts: [&[i64]; 2],
    tolerance: Tolerance,
    compiled: bool,
) {
    let options = GenerationOptions {
        max_new_tokens: 4,
        ..GenerationOptions::default()
    };
    // Compare like with like, since half-precision steps round differently compiled.
    let mut alone = decoder(model, DecoderOptions::new(1, 32), compiled);
    let expected_logits = prompts.map(|prompt| {
        alone.reset();
        host_logits(&alone.step(&[prompt]).unwrap())
    });
    let expected_tokens =
        prompts.map(|prompt| generate(&mut alone, prompt, &options, None, |_| Ok(())).unwrap());

    let batch_options = DecoderOptions::new(2, 32).with_page_size(3);
    let mut batch = decoder(model, batch_options, compiled);
    let rows = host_rows(&batch.step(&prompts).unwrap());
    assert_matches(&rows[0], &expected_logits[0], tolerance, "first sequence");
    assert_matches(&rows[1], &expected_logits[1], tolerance, "second sequence");
    let generated = generate_batch(&mut batch, &prompts, &options, None, |_, _| Ok(())).unwrap();
    assert_eq!(generated[0], expected_tokens[0], "compiled: {compiled}");
    assert_eq!(generated[1], expected_tokens[1], "compiled: {compiled}");
    if compiled {
        assert_one_specialization(&batch);
    }
}

pub fn check_reuse<L: DecoderModel>(model: &Model<L>, prompt: &[i64], compiled: bool) {
    let mut decoder = decoder(model, DecoderOptions::new(1, 32), compiled);
    let options = GenerationOptions {
        max_new_tokens: 4,
        ..GenerationOptions::default()
    };
    let first = generate(&mut decoder, prompt, &options, None, |_| Ok(())).unwrap();
    generate(&mut decoder, &[3, 4, 5], &options, None, |_| Ok(())).unwrap();
    let again = generate(&mut decoder, prompt, &options, None, |_| Ok(())).unwrap();
    assert_eq!(first, again, "compiled: {compiled}");
}

pub fn check_shared_step<L: DecoderModel>(model: &Model<L>, prompt: &[i64]) {
    let options = DecoderOptions::new(1, 32);
    let mut first = decoder(model, options, true);
    let mut second = decoder(model, options, false);
    second.share_compiled(&first).unwrap();
    let other = [3, 17, 8];
    let mut alone = decoder(model, options, true);
    let expected = |decoder: &mut CachedDecoder<L>, prompt: &[i64]| {
        decoder.reset();
        let mut logits = decoder.step(&[prompt]).unwrap();
        let mut rows = Vec::new();
        for _ in 0..3 {
            let token = best_token(&host_logits(&logits));
            rows.push(token);
            logits = decoder.step(&[&[token]]).unwrap();
        }
        rows
    };
    let first_alone = expected(&mut alone, prompt);
    let second_alone = expected(&mut alone, &other);
    let mut first_logits = first.step(&[prompt]).unwrap();
    let mut second_logits = second.step(&[&other]).unwrap();
    let (mut first_tokens, mut second_tokens) = (Vec::new(), Vec::new());
    for _ in 0..3 {
        let token = best_token(&host_logits(&first_logits));
        first_tokens.push(token);
        first_logits = first.step(&[&[token]]).unwrap();
        let token = best_token(&host_logits(&second_logits));
        second_tokens.push(token);
        second_logits = second.step(&[&[token]]).unwrap();
    }

    assert_eq!(first_tokens, first_alone);
    assert_eq!(second_tokens, second_alone);
    let mut wrong_size = decoder(model, DecoderOptions::new(1, 16), false);
    assert!(wrong_size.share_compiled(&first).is_err());
}

pub fn check_sampling_repeats<L: DecoderModel>(model: &Model<L>, prompt: &[i64], compiled: bool) {
    let sample = |seed| {
        let mut decoder = decoder(model, DecoderOptions::new(1, 32), compiled);
        let options = GenerationOptions {
            max_new_tokens: 16,
            stop_tokens: Vec::new(),
            sampling: Some(
                SamplingOptions::new()
                    .with_temperature(0.6)
                    .with_top_k(Some(20))
                    .with_top_p(0.95),
            ),
            ..GenerationOptions::default()
        };
        let generator = Generator::new(seed);
        generate(&mut decoder, prompt, &options, Some(&generator), |_| Ok(())).unwrap()
    };
    let first = sample(0);
    assert_eq!(first.len(), 16);
    assert_eq!(first, sample(0));
    assert_ne!(first, sample(1));
}

pub fn check_failed_step<L: DecoderModel>(model: &Model<L>) {
    let mut decoder = decoder(model, DecoderOptions::new(2, 4), false);
    assert!(decoder.step(&[&[1, 2, 3], &[4]]).is_ok());
    assert!(decoder.step(&[&[5], &[6, 7, 8, 9]]).is_err());
    let mut alone = self::decoder(model, DecoderOptions::new(2, 4), false);
    alone.step(&[&[1, 2, 3], &[4]]).unwrap();
    let rows = host_rows(&decoder.step(&[&[5], &[6]]).unwrap());
    let expected = host_rows(&alone.step(&[&[5], &[6]]).unwrap());
    assert_eq!(rows, expected);
}

const RUNNING_NEW_TOKENS: usize = 4;
const PROMPT_CHUNKS: [usize; 2] = [1, 3];

fn running_options(reference: &Reference) -> DecoderOptions {
    const PAGE_SIZE: i64 = 4;
    let new_tokens = i64::try_from(RUNNING_NEW_TOKENS).unwrap();
    // A sequence's last token is returned and never written.
    let pages = |prompt: &[i64]| (length(prompt) + new_tokens - 2) / PAGE_SIZE + 1;
    let held = pages(&reference.prompt) + pages(&reference.other_prompt);
    DecoderOptions::new(2, 4 * (length(&reference.prompt) + new_tokens))
        .with_page_size(PAGE_SIZE)
        .with_cache_tokens(held * PAGE_SIZE)
}

fn running_generation() -> GenerationOptions {
    GenerationOptions {
        max_new_tokens: RUNNING_NEW_TOKENS,
        ..GenerationOptions::default()
    }
}

fn alone<L: DecoderModel>(model: &Model<L>, reference: &Reference) -> [Vec<i64>; 2] {
    let mut decoder = decoder(model, DecoderOptions::new(1, 32), false);
    let options = running_generation();
    [&reference.prompt, &reference.other_prompt]
        .map(|prompt| generate(&mut decoder, prompt, &options, None, |_| Ok(())).unwrap())
}

fn check_running(decoder: &mut impl Decoder, reference: &Reference, expected: &[Vec<i64>; 2]) {
    let mut batch = RunningBatch::create(decoder, running_generation()).unwrap();
    let mut generated = BTreeMap::<usize, Vec<i64>>::new();
    let mut prompts = HashMap::new();
    let mut waiting = vec![
        &reference.prompt,
        &reference.other_prompt,
        &reference.prompt,
    ];
    waiting.reverse();
    let mut joined_third = false;
    let mut steps = 0;
    while !batch.is_empty() || !waiting.is_empty() {
        // One sequence at first, so the next joins a batch that is already running.
        let wanted = if steps == 0 { 1 } else { waiting.len() };
        for _ in 0..wanted {
            let Some(prompt) = waiting.last() else { break };
            let Some(id) = batch.join(prompt).unwrap() else {
                break;
            };
            prompts.insert(id, usize::from(*prompt != &reference.prompt));
            joined_third |= id == 2 && steps > 0;
            waiting.pop();
        }

        for token in batch.advance(None).unwrap() {
            generated
                .entry(token.sequence)
                .or_default()
                .push(token.token);
        }

        steps += 1;
        assert!(steps < 100, "a running batch made no progress");
    }

    assert!(
        joined_third,
        "the third sequence never joined a running batch"
    );
    assert_eq!(generated.len(), 3);
    for (id, tokens) in &generated {
        assert_eq!(tokens, &expected[prompts[id]], "sequence {id}");
    }
}

pub fn check_held_sequence<L: DecoderModel>(
    model: &Model<L>,
    reference: &Reference,
    compiled: bool,
) {
    let prompts: [&[i64]; 2] = [&reference.prompt, &reference.other_prompt];
    let new_tokens = 5;
    let mut alone = decoder(model, DecoderOptions::new(1, 32), compiled);
    let options = GenerationOptions {
        max_new_tokens: new_tokens,
        ..GenerationOptions::default()
    };
    let first = generate(&mut alone, prompts[0], &options, None, |_| Ok(())).unwrap();
    let second = generate(&mut alone, prompts[1], &options, None, |_| Ok(())).unwrap();
    // Stopping the first sequence at its first token leaves it a slot short of the second's steps, were it to keep advancing.
    let capacity = length(prompts[0]).max(length(prompts[1]) + length(&second));
    let mut batch = decoder(model, DecoderOptions::new(2, capacity), compiled);
    let options = GenerationOptions {
        max_new_tokens: new_tokens,
        stop_tokens: vec![first[0]],
        sampling: None,
        ..GenerationOptions::default()
    };
    if second.contains(&first[0]) {
        return;
    }

    let generated = generate_batch(&mut batch, &prompts, &options, None, |_, _| Ok(())).unwrap();
    assert_eq!(generated[0], [first[0]], "compiled: {compiled}");
    assert_eq!(generated[1], second, "compiled: {compiled}");
    if compiled {
        assert_one_specialization(&batch);
    }
}
