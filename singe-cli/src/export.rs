// `singe export`: writes a model as a CUDA C++ program, autotuned with `--tune`, or a snapshot as an ONNX file.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use bytesize::ByteSize;
use clap::{Args, ValueEnum};
use singe::{
    DataType, Device,
    backend::cuda::export::{
        GpuArchitecture, Library,
        compiled::{build_written_loop, build_written_runner, program_target},
    },
    graph::ProgramInputBinding,
    onnx::WeightPlacement,
    tensor::SamplingOptions,
};
use singe_model::{
    Source,
    export::{
        cuda::{
            self as cuda_export, Exported,
            file::{Build, FileExport, FileExported, export_file, export_snapshot},
        },
        onnx as onnx_export,
        rewrite::{check_rewrite, rewrite},
    },
    function::Shapes,
    task::Task,
    tune::{TuneLevel, Tuning},
};
use singe_server::config::TuneConfig;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Target {
    Cuda,
    Onnx,
}

#[derive(Clone, Copy, ValueEnum)]
enum Weights {
    Inline,
    Beside,
}

#[derive(Clone, Copy, ValueEnum)]
enum ExportType {
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Level {
    Quick,
    Default,
    Thorough,
}

impl From<Level> for TuneLevel {
    fn from(level: Level) -> Self {
        match level {
            Level::Quick => Self::Quick,
            Level::Default => Self::Default,
            Level::Thorough => Self::Thorough,
        }
    }
}

#[derive(Args)]
pub struct ExportArguments {
    /// `cuda` writes a C++ program; `onnx` writes `model.onnx`.
    #[arg(long, value_enum)]
    target: Target,
    /// An ONNX file, with its external data beside it.
    #[arg(long, conflicts_with = "model", required_unless_present = "model")]
    file: Option<PathBuf>,
    /// A model directory, or a Hugging Face repository the local cache holds.
    #[arg(long)]
    model: Option<String>,
    /// The task whose head `--model` exports, or else its default.
    #[arg(long)]
    task: Option<String>,
    /// The GPU architecture, such as `sm_120`, or `native` for the GPU of `--device`.
    #[arg(long, default_value = "native")]
    arch: String,
    /// The GPU that tuning and `--check` run on.
    #[arg(long, default_value_t = 0)]
    device: i32,
    /// The type of `--model`'s weights and floating inputs.
    #[arg(long = "type", value_enum)]
    data_type: Option<ExportType>,
    /// `--target onnx`: weights in `model.onnx` or beside it.
    #[arg(long, value_enum)]
    weights: Option<Weights>,
    /// Fixes a named dimension, as `batch_size=8`; several values, as `sequence_length=128,512`, export a step per combination.
    #[arg(long = "dim", value_parser = parse_dim)]
    dims: Vec<(String, Vec<i64>)>,
    /// Fixes every axis of one input, as `pixel_values=1x3x224x224`.
    #[arg(long = "input", value_parser = parse_input)]
    inputs: Vec<(String, Vec<i64>)>,
    /// Libraries the program may call: `all`, or a list of cublas, cublaslt, cudnn, cufft, and cutensor; without it, the program runs generated kernels alone.
    #[arg(long)]
    libraries: Option<String>,
    /// Keeps generated kernels from merging neighbouring operations.
    #[arg(long)]
    no_fusion: bool,
    /// Adds functions that read and replace each state tensor.
    #[arg(long)]
    state_functions: bool,
    /// Prints each choice point's candidates, times, and winner.
    #[arg(long, requires = "tune")]
    tune_report: bool,
    #[command(flatten)]
    output: OutputArguments,
    #[command(flatten)]
    generation: GenerationArguments,
    #[command(flatten)]
    tune: TuneArguments,
}

#[derive(Args)]
struct OutputArguments {
    /// The directory to write, created if missing.
    #[arg(long = "output")]
    directory: PathBuf,
    /// Writes into an output directory that is not empty.
    #[arg(long)]
    force: bool,
    /// Builds `libmodel.so` and the runner with nvcc.
    #[arg(long)]
    build: bool,
    /// Builds the program and runs it on the tuning inputs, comparing its outputs with Singe's; with `--target onnx --file`, compares the written file with the original.
    #[arg(long)]
    check: bool,
}

#[derive(Args)]
struct GenerationArguments {
    /// A generating model's tokens per sequence, prompt included.
    #[arg(long)]
    capacity: Option<i64>,
    /// A generating model's sequences at once.
    #[arg(long)]
    rows: Option<usize>,
    /// The rows of a generating model's prompt step.
    #[arg(long)]
    prompt_rows: Option<usize>,
    /// The seconds of audio a speech model's encode step reads.
    #[arg(long)]
    audio_seconds: Option<i64>,
    /// The side of an autoencoder's square image.
    #[arg(long)]
    image_size: Option<i64>,
    /// The most patches of an image a vision-language model reads.
    #[arg(long)]
    image_patches: Option<i64>,
    /// Samples a generating model's tokens at this temperature rather than taking the likeliest.
    #[arg(long)]
    temperature: Option<f32>,
    /// Samples among a generating model's likeliest tokens alone.
    #[arg(long)]
    top_k: Option<i64>,
    /// Samples among the likeliest tokens whose probabilities reach this sum.
    #[arg(long)]
    top_p: Option<f32>,
}

impl GenerationArguments {
    const fn is_any_given(&self) -> bool {
        self.capacity.is_some()
            || self.rows.is_some()
            || self.prompt_rows.is_some()
            || self.audio_seconds.is_some()
            || self.image_size.is_some()
            || self.image_patches.is_some()
            || self.temperature.is_some()
            || self.top_k.is_some()
            || self.top_p.is_some()
    }
}

#[derive(Args)]
pub struct TuneArguments {
    /// Times the lowering's choices on the local GPU and keeps the fastest; without it, a program or a served file takes the lowering's own choices.
    #[arg(long)]
    pub tune: bool,
    /// Reuses tuning records instead of measuring.
    #[arg(long, requires = "tune")]
    pub tune_records: Option<PathBuf>,
    /// Where measured records go; for `export`, `OUTPUT/tuning.json` by default.
    #[arg(long, requires = "tune")]
    pub save_tune_records: Option<PathBuf>,
    /// Time per choice point, such as 10s or 500ms.
    #[arg(long, requires = "tune", value_parser = humantime::parse_duration)]
    pub tune_budget: Option<Duration>,
    /// Timed runs per candidate.
    #[arg(long, requires = "tune")]
    pub tune_runs: Option<usize>,
    /// Also searches generated kernels' schedules.
    #[arg(long, requires = "tune", overrides_with = "no_tune_schedules")]
    pub tune_schedules: bool,
    /// Keeps generated kernels' schedules as the lowering chose them.
    #[arg(long, requires = "tune", overrides_with = "tune_schedules")]
    pub no_tune_schedules: bool,
    /// Time to search whole plans after the choice points, such as 2m; 0s skips it.
    #[arg(long, requires = "tune", value_parser = humantime::parse_duration)]
    pub tune_plan_budget: Option<Duration>,
    /// Seeds the plan search.
    #[arg(long, requires = "tune")]
    pub tune_seed: Option<u64>,
    /// The most memory a candidate may use, such as 4GiB.
    #[arg(long, requires = "tune")]
    pub tune_memory_limit: Option<ByteSize>,
    /// Candidates compiled at once.
    #[arg(long, requires = "tune")]
    pub tune_jobs: Option<usize>,
    /// A safetensors file of the inputs tuning, and `--check` with it, run on.
    #[arg(long, requires = "tune")]
    pub tune_inputs: Option<PathBuf>,
    /// `quick` is 1s and 5 runs without schedules; `thorough` 30s, 50 runs, and a 60s plan budget.
    #[arg(long, requires = "tune", value_enum)]
    pub tune_level: Option<Level>,
}

impl TuneArguments {
    const fn schedules(&self) -> Option<bool> {
        match (self.tune_schedules, self.no_tune_schedules) {
            (true, _) => Some(true),
            (_, true) => Some(false),
            _ => None,
        }
    }

    pub fn config(&self) -> TuneConfig {
        TuneConfig {
            enabled: self.tune,
            level: self.tune_level.map(TuneLevel::from),
            budget: self.tune_budget,
            runs: self.tune_runs,
            schedules: self.schedules(),
            plan_budget: self.tune_plan_budget,
            seed: self.tune_seed,
            memory_limit: self.tune_memory_limit,
            jobs: self.tune_jobs,
            records: self.tune_records.clone(),
            save_records: self.save_tune_records.clone(),
        }
    }

    fn get_tuning(&self) -> Result<Option<Tuning>> {
        let tuning = self.config().get_tuning()?;
        Ok(tuning.map(|tuning| Tuning {
            inputs: self.tune_inputs.clone(),
            ..tuning
        }))
    }
}

pub fn named_shapes(
    flag: &str,
    values: &[(String, Vec<i64>)],
) -> Result<BTreeMap<String, Vec<i64>>> {
    let mut named = BTreeMap::new();
    for (name, shape) in values {
        if named.insert(name.clone(), shape.clone()).is_some() {
            bail!("{flag} {name} given twice");
        }
    }
    Ok(named)
}

fn shapes(arguments: &ExportArguments) -> Result<Shapes> {
    Ok(Shapes {
        dimensions: named_shapes("--dim", &arguments.dims)?,
        inputs: named_shapes("--input", &arguments.inputs)?,
    })
}

fn task(arguments: &ExportArguments, source: &Source) -> Result<Task> {
    Ok(match &arguments.task {
        Some(task) => Task::from_id(task)?,
        None => Task::default_for(source)?,
    })
}

pub fn parse_dim(text: &str) -> Result<(String, Vec<i64>)> {
    let (name, values) = text
        .split_once('=')
        .with_context(|| format!("dimension {text}, rather than NAME=N or NAME=N,N"))?;
    let values = values
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<i64>()
                .ok()
                .filter(|&value| value > 0)
                .with_context(|| format!("dimension size {value}, rather than a positive integer"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((name.to_owned(), values))
}

pub fn parse_input(text: &str) -> Result<(String, Vec<i64>)> {
    let (name, shape) = text.split_once('=').with_context(|| {
        format!("input shape {text}, rather than NAME=D0xD1, such as pixel_values=1x3x224x224")
    })?;
    let shape = shape
        .split('x')
        .map(|size| {
            size.trim()
                .parse::<i64>()
                .ok()
                .filter(|&size| size >= 0)
                .with_context(|| format!("axis size {size}, rather than an integer"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((name.to_owned(), shape))
}

const fn data_type(choice: ExportType) -> DataType {
    match choice {
        ExportType::F32 => DataType::F32,
        ExportType::F16 => DataType::F16,
        ExportType::Bf16 => DataType::Bf16,
    }
}

fn libraries(text: Option<&str>) -> Result<Vec<Library>> {
    match text {
        None => Ok(Vec::new()),
        Some("all") => Ok(Library::ALL.to_vec()),
        Some(list) => list
            .split(',')
            .map(|name| Ok(name.trim().parse::<Library>()?))
            .collect(),
    }
}

fn prepare_output(arguments: &ExportArguments) -> Result<()> {
    let output = &arguments.output.directory;
    let occupied = fs::read_dir(output).is_ok_and(|mut entries| entries.next().is_some());
    if occupied && !arguments.output.force {
        bail!(
            "output directory {} is not empty; pass --force to write into it",
            output.display()
        );
    }
    fs::create_dir_all(output).with_context(|| format!("creating {}", output.display()))
}

pub fn export(arguments: &ExportArguments) -> Result<()> {
    if arguments.file.is_some() && arguments.task.is_some() {
        bail!(
            "--task with --file, since a file's task is chosen where it is served, with serve --task"
        );
    }

    if arguments.file.is_some() && arguments.data_type.is_some() {
        bail!("--type with --file, since converting an ONNX file's type is not written yet");
    }

    match arguments.target {
        Target::Onnx => export_onnx(arguments),
        Target::Cuda => export_cuda(arguments),
    }
}

fn export_onnx(arguments: &ExportArguments) -> Result<()> {
    if arguments.tune.tune {
        bail!(
            "tuning chooses CUDA kernels, which an ONNX file cannot name; drop the tuning options with --target onnx"
        );
    }

    if arguments.output.build
        || arguments.arch != "native"
        || arguments.device != 0
        || arguments.no_fusion
        || arguments.state_functions
        || arguments.libraries.is_some()
    {
        bail!(
            "--build, --arch, --device, --no-fusion, --state-functions, and --libraries apply to --target cuda"
        );
    }

    if arguments.generation.is_any_given() {
        bail!("the generation options apply to --target cuda for generating models");
    }

    let weights = match arguments.weights {
        None => WeightPlacement::Automatic,
        Some(Weights::Inline) => WeightPlacement::Inline,
        Some(Weights::Beside) => WeightPlacement::Beside,
    };
    let output = &arguments.output.directory;
    if let Some(file) = &arguments.file {
        let shapes = shapes(arguments)?;
        prepare_output(arguments)?;
        let rewritten = rewrite(file, output, &shapes, weights)?;
        println!(
            "wrote model.onnx to {}: {} nodes, from {}",
            output.display(),
            rewritten.nodes_after,
            rewritten.nodes_before
        );
        if arguments.output.check {
            for check in check_rewrite(file, output, &shapes)? {
                println!("checked in Singe's importer, against the original file: {check}");
            }
        }
        return Ok(());
    }

    if arguments.output.check || !arguments.dims.is_empty() {
        bail!(
            "--check and --dim apply to --target onnx with --file; fix a model's export shape with --input"
        );
    }

    let name = arguments.model.as_deref().context("--model or --file")?;
    let source = Source::find(name)?;
    let task = task(arguments, &source)?;
    let (input, input_shape) = match arguments.inputs.as_slice() {
        [] => (None, None),
        [(input, shape)] => (Some(input), Some(shape.clone())),
        _ => bail!("several --input shapes; an ONNX export fixes its main input's shape alone"),
    };
    let options = onnx_export::ExportOptions {
        data_type: arguments.data_type.map_or(DataType::F32, data_type),
        weights,
        input_shape,
    };
    let (program, state) = onnx_export::program(&source, task, &options)?;
    if let Some(input) = input {
        let main = program
            .signature()
            .inputs()
            .iter()
            .find(|value| matches!(value.binding(), ProgramInputBinding::Runtime))
            .map_or("none", |value| value.name());
        if main != input {
            bail!(
                "--input {input}, while an export fixes its main input's shape, which is {main} for this model"
            );
        }
    }

    prepare_output(arguments)?;
    onnx_export::write(&program, &state, output, weights)?;

    println!("wrote model.onnx for {} to {}", task.id(), output.display());
    Ok(())
}

fn architecture(arguments: &ExportArguments, device: Device) -> Result<GpuArchitecture> {
    if arguments.arch == "native" {
        return program_target(device).context(
            "--arch native without a GPU and nvcc to read it from; name an architecture such as sm_120",
        );
    }
    Ok(arguments.arch.parse()?)
}

enum ExportInput<'a> {
    File(&'a Path),
    Snapshot { name: &'a str, task: Task },
}

fn export_cuda(arguments: &ExportArguments) -> Result<()> {
    if arguments.weights.is_some() {
        bail!("--weights applies to --target onnx");
    }

    let device = Device::Cuda(arguments.device);
    let architecture = architecture(arguments, device)?;
    let options = FileExport {
        architecture,
        device,
        shapes: shapes(arguments)?,
        libraries: libraries(arguments.libraries.as_deref())?,
        fusion: !arguments.no_fusion,
        state_functions: arguments.state_functions,
        tuning: arguments.tune.get_tuning()?,
        build: if arguments.output.check {
            Build::Check
        } else if arguments.output.build {
            Build::Library
        } else {
            Build::Write
        },
    };
    let input = match (&arguments.file, &arguments.model) {
        (Some(file), _) => ExportInput::File(file),
        (None, Some(name)) => {
            let source = Source::find(name)?;
            let task = task(arguments, &source)?;
            if matches!(
                task,
                Task::TextGeneration
                    | Task::AutomaticSpeechRecognition
                    | Task::ImageTextToText
                    | Task::ImageToImage
            ) {
                return export_loop(arguments, &source, task, options);
            }

            ExportInput::Snapshot { name, task }
        }
        (None, None) => bail!("--model or --file"),
    };
    if arguments.generation.is_any_given() {
        bail!("the generation options apply to generating models and autoencoders");
    }

    if program_target(device).is_some_and(|local| local != architecture) {
        if options
            .tuning
            .as_ref()
            .is_some_and(|tuning| tuning.records.is_none())
        {
            bail!(
                "tuning measures on the local GPU, so --tune for {architecture} needs --tune-records"
            );
        }

        if arguments.output.check {
            bail!("--check runs the program on the local GPU, which cannot run {architecture}");
        }
    }

    prepare_output(arguments)?;
    let output = &arguments.output.directory;
    let exported = match input {
        ExportInput::Snapshot { name, task } => {
            let data_type = arguments.data_type.map_or(DataType::F32, data_type);
            export_snapshot(name, task, data_type, output, &options)?
        }
        ExportInput::File(file) => export_file(file, output, &options)?,
    };
    report(arguments, &exported);
    Ok(())
}

fn report(arguments: &ExportArguments, exported: &FileExported) {
    let manifest = &exported.manifest;
    let steps = manifest
        .steps
        .iter()
        .map(|step| step.name.as_str())
        .collect::<Vec<_>>();
    println!(
        "wrote {} program for {} to {}: steps {}",
        if manifest.tuned {
            "a tuned"
        } else {
            "an untuned"
        },
        manifest.architecture,
        arguments.output.directory.display(),
        steps.join(", ")
    );
    if arguments.tune_report
        && let Some(report) = exported
            .records
            .as_ref()
            .and_then(|records| records.report())
    {
        println!("{report}");
    }
    for check in &exported.checks {
        println!("check {check}");
    }
}

fn export_loop(
    arguments: &ExportArguments,
    source: &Source,
    task: Task,
    file: FileExport,
) -> Result<()> {
    if arguments.tune.tune
        || file.build == Build::Check
        || !arguments.dims.is_empty()
        || !arguments.inputs.is_empty()
        || !file.fusion
        || file.state_functions
    {
        bail!(
            "task {} exports a generation loop or an autoencoder, which tuning, --check, --dim, --input, --no-fusion, and --state-functions do not apply to yet",
            task.id()
        );
    }

    let mut options = cuda_export::ExportOptions::new(file.architecture);
    options.device = file.device;
    options.libraries = file.libraries;
    if let Some(choice) = arguments.data_type {
        options.data_type = data_type(choice);
    }

    let generation = &arguments.generation;
    if let Some(capacity) = generation.capacity {
        options.capacity = capacity;
    }

    if let Some(rows) = generation.rows {
        options.rows = rows;
    }

    if let Some(rows) = generation.prompt_rows {
        options.prompt_rows = rows;
    }

    if let Some(seconds) = generation.audio_seconds {
        options.audio_seconds = seconds;
    }

    if let Some(size) = generation.image_size {
        options.image_size = size;
    }

    options.image_patches = generation.image_patches;
    if generation.temperature.is_some() || generation.top_k.is_some() || generation.top_p.is_some()
    {
        options.sampling = Some(
            SamplingOptions::new()
                .with_temperature(generation.temperature.unwrap_or(1.0))
                .with_top_k(generation.top_k)
                .with_top_p(generation.top_p.unwrap_or(1.0)),
        );
    }

    prepare_output(arguments)?;
    let output = &arguments.output.directory;
    let exported = cuda_export::export(source, task, output, &options)?;
    if file.build != Build::Write {
        match exported {
            Exported::Generator => {
                build_written_loop(output)?;
            }
            Exported::Runner => {
                build_written_runner(output, "model")?;
            }
            Exported::Autoencoder => {
                build_written_runner(output, "encoder")?;
                build_written_runner(output, "decoder")?;
            }
        }
    }

    println!(
        "wrote an untuned {exported:?} program for {} to {}",
        task.id(),
        output.display()
    );
    Ok(())
}
