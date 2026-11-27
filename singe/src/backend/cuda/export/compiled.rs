// Builds exported programs as their users would, with `nvcc`, then loads and runs them, which tuning and tests both do.

use std::{
    env,
    ffi::{CStr, c_char, c_void},
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
    ptr,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use libloading::{Error as LibraryError, Library};
use serde_json::{Error as JsonError, Value};
use singe_core::report_cleanup_failure;
use singe_cuda::architecture::GpuArchitecture;
use thiserror::Error as ThisError;

use crate::{
    backend::cuda::{
        codegen::disk_cache::{self, EntryKind},
        export::{
            CudaProgram, CudaProgramSet, ProgramValue, RowCount,
            writer::{
                CONFIGURATION_REFUSED, CUBLAS_BASE, LIBRARY_FILE, LIBRARY_VERSION, ProgramArtifact,
            },
        },
        runtime::runtime_for_device,
        tensor_region::CudaTensorSource,
    },
    data_type::DataType,
    device::Device,
    error::Error,
    graph::{ir::CountedOutput, narrow_counted},
    layout::Layout,
    tensor::Tensor,
};

// Why a program could not be built or run.
#[derive(Debug, ThisError)]
pub enum ProgramError {
    #[error("failed to write the program")]
    Export(#[from] Error),
    #[error("failed to access the program's files")]
    Io(#[from] io::Error),
    #[error("the manifest is not valid json")]
    Manifest(#[from] JsonError),
    #[error("failed to load the program")]
    Load(#[from] LibraryError),
    #[error("{found} inputs, where the program takes {expected}")]
    InputCount { expected: usize, found: usize },
    #[error("input `{name}` as {found}, where the program takes {expected}")]
    Input {
        name: String,
        expected: String,
        found: String,
    },
    #[error("nvcc failed in {directory}:\n{output}")]
    Compiler { directory: PathBuf, output: String },
    #[error(
        "model.weights in {directory} holds {found} bytes, where the manifest records {expected}"
    )]
    Weights {
        directory: PathBuf,
        found: usize,
        expected: usize,
    },
    #[error("{entry} returned {status} with status word {word} in {directory}: {failure}")]
    Status {
        entry: String,
        status: i32,
        word: u32,
        // What the program said about the failure: the step, the call, and what it returned.
        failure: String,
        directory: PathBuf,
    },
}

impl ProgramError {
    // Whether a library refused a configuration the program's tuning records chose.
    #[must_use]
    pub const fn is_refusal(&self) -> bool {
        matches!(
            self,
            Self::Status {
                status: CONFIGURATION_REFUSED,
                ..
            }
        )
    }

    // Whether a library refused what the program had it run: a step, with one of its own statuses, or a configuration its records chose.
    #[must_use]
    pub const fn is_library_refusal(&self) -> bool {
        match self {
            Self::Status { status, .. } => {
                (*status >= CUBLAS_BASE && *status < LIBRARY_VERSION)
                    || *status == CONFIGURATION_REFUSED
            }
            _ => false,
        }
    }
}

type Create = unsafe extern "C" fn(*mut *mut c_void) -> i32;

type LoadWeights = unsafe extern "C" fn(*mut c_void, *const c_void) -> i32;

type Accessor = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;

type Run =
    unsafe extern "C" fn(*mut c_void, *const *mut c_void, *const *mut c_void, *mut c_void) -> i32;

type Finish = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u32) -> i32;

type Destroy = unsafe extern "C" fn(*mut c_void);

type Failure = unsafe extern "C" fn() -> *const c_char;

type SetLibraryTable = unsafe extern "C" fn(*const i64);

// Returns the architecture of `device`, or `None` without a compiler or a usable device.
#[must_use]
pub fn program_target(device: Device) -> Option<GpuArchitecture> {
    Command::new("nvcc").arg("--version").output().ok()?;
    Some(runtime_for_device(device).ok()?.architecture())
}

// A program compiled into a loaded shared library, which its instances keep loaded.
#[derive(Clone)]
pub struct CompiledProgram {
    built: Arc<Built>,
}

const PROGRAM_DEVICE: Device = Device::Cuda(0);

struct Built {
    steps: Vec<ProgramInterface>,
    library: Library,
    weights: Arc<Vec<u8>>,
    directory: PathBuf,
    files: FileRetention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileRetention {
    Removed,
    Kept,
}

impl Drop for Built {
    fn drop(&mut self) {
        // A program that fails to compile never gets here, so its files stay for reading what the compiler rejected.
        if self.files == FileRetention::Removed
            && let Err(error) = fs::remove_dir_all(&self.directory)
        {
            report_cleanup_failure("remove", "program directory", &error);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramInterface {
    pub name: String,
    pub inputs: Vec<ProgramValue>,
    pub outputs: Vec<ProgramValue>,
    pub state: Vec<ProgramValue>,
}

impl ProgramInterface {
    fn of(program: &CudaProgram) -> Self {
        Self {
            name: String::new(),
            inputs: program.inputs.clone(),
            outputs: program.outputs.clone(),
            state: program
                .state
                .iter()
                .map(|state| ProgramValue {
                    name: state.name.clone(),
                    graph_name: state.graph_name.clone(),
                    data_type: state.data_type,
                    layout: state.layout.clone(),
                    written: false,
                    count: None,
                })
                .collect(),
        }
    }

    fn read(name: String, manifest: &Value) -> Result<Self, ProgramError> {
        let outputs = manifest_array(&manifest["outputs"])?;
        let output_names = outputs
            .iter()
            .map(|output| manifest_text(output, "name"))
            .collect::<Result<Vec<_>, _>>()?;
        let values = |key: &str| {
            manifest_array(&manifest[key])?
                .iter()
                .map(|value| manifest_value(value, &output_names))
                .collect::<Result<Vec<_>, ProgramError>>()
        };
        Ok(Self {
            name,
            inputs: values("inputs")?,
            outputs: values("outputs")?,
            state: values("state")?,
        })
    }
}

fn manifest_array(value: &Value) -> Result<&Vec<Value>, Error> {
    value
        .as_array()
        .ok_or_else(|| Error::invalid_argument("manifest values"))
}

fn manifest_text(value: &Value, key: &str) -> Result<String, Error> {
    value[key]
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| Error::invalid_argument(format!("manifest {key}")))
}

fn manifest_numbers(value: &Value, key: &str) -> Result<Vec<i64>, Error> {
    manifest_array(&value[key])?
        .iter()
        .map(|number| {
            number
                .as_i64()
                .ok_or_else(|| Error::invalid_argument(format!("manifest {key}")))
        })
        .collect()
}

fn manifest_value(value: &Value, output_names: &[String]) -> Result<ProgramValue, ProgramError> {
    let offset = value["offset"]
        .as_u64()
        .and_then(|offset| usize::try_from(offset).ok())
        .ok_or_else(|| Error::invalid_argument("manifest offset"))?;
    let layout = Layout::with_strides(
        manifest_numbers(value, "dimensions")?,
        manifest_numbers(value, "strides")?,
    )?
    .with_offset(offset);
    let count = value["count"]
        .as_object()
        .map(|count| {
            let output = count["output"]
                .as_str()
                .and_then(|count| output_names.iter().position(|name| name == count))
                .ok_or_else(|| Error::invalid_argument("manifest count output"))?;
            let axis = count["axis"]
                .as_u64()
                .and_then(|axis| usize::try_from(axis).ok())
                .ok_or_else(|| Error::invalid_argument("manifest count axis"))?;
            Ok::<_, ProgramError>(RowCount { output, axis })
        })
        .transpose()?;
    Ok(ProgramValue {
        name: manifest_text(value, "name")?,
        graph_name: manifest_text(value, "graph_name")?,
        data_type: manifest_text(value, "data_type")?.parse()?,
        layout,
        // State tensors are never written by a run, so their entries leave it out.
        written: value["written"].as_bool().unwrap_or(false),
        count,
    })
}

fn read_weights(directory: &Path, manifest: &Value) -> Result<Vec<u8>, ProgramError> {
    let weights = match fs::read(directory.join("model.weights")) {
        Ok(weights) => weights,
        Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.into()),
    };
    let expected = manifest["weights_bytes"]
        .as_u64()
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| Error::invalid_argument("manifest weights_bytes"))?;
    if weights.len() != expected {
        return Err(ProgramError::Weights {
            directory: directory.to_owned(),
            found: weights.len(),
            expected,
        });
    }
    Ok(weights)
}

fn describe_tensor(data_type: DataType, layout: &Layout, device: Device) -> String {
    format!(
        "{data_type} {:?} with strides {:?} from {} on {device}",
        layout.dimensions(),
        layout.strides(),
        layout.storage_offset()
    )
}

fn read_manifest(directory: &Path, prefix: &str) -> Result<Value, ProgramError> {
    let manifest = fs::read_to_string(directory.join(format!("{prefix}.json")))?;
    Ok(serde_json::from_str(&manifest)?)
}

fn field(manifest: &Value, name: &str) -> String {
    manifest[name].as_str().unwrap_or_default().to_owned()
}

// A manifest's compile command, without the compiler's own name.
fn compile_arguments(manifest: &Value) -> Vec<String> {
    field(manifest, "compile")
        .split_whitespace()
        .skip(1)
        .map(ToOwned::to_owned)
        .collect()
}

fn run_compiler(arguments: &[String], directory: &Path) -> Result<(), ProgramError> {
    let output = Command::new("nvcc")
        .args(arguments)
        .current_dir(directory)
        .output()?;
    compiler_result(&output, directory)
}

// What `nvcc --version` says, read once, which keys built programs so another toolkit builds them again.
fn compiler_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        Command::new("nvcc")
            .arg("--version")
            .output()
            .map_or_default(|output| String::from_utf8_lossy(&output.stdout).into_owned())
    })
}

// Builds `libmodel.so` from the `model.h` and `model.cu` written under `directory`, with `arguments`, or copies the library an earlier build of the same files, arguments, and compiler kept in the cache of programs.
//
// Tuning builds the same candidates in each tune, and test runs build the same programs each time, at seconds a build.
fn build_library(arguments: &[String], directory: &Path) -> Result<(), ProgramError> {
    let header = fs::read(directory.join("model.h"))?;
    let source = fs::read(directory.join("model.cu"))?;
    let Some(entry) = disk_cache::entry(
        EntryKind::Program,
        [
            compiler_version().as_bytes(),
            header.as_slice(),
            source.as_slice(),
        ]
        .into_iter()
        .chain(arguments.iter().map(String::as_bytes)),
    ) else {
        return run_compiler(arguments, directory);
    };
    let library = directory.join(LIBRARY_FILE);
    if entry.copy_to(&library) {
        return Ok(());
    }

    run_compiler(arguments, directory)?;
    entry.store(&library);
    Ok(())
}

fn compiler_result(output: &Output, directory: &Path) -> Result<(), ProgramError> {
    if output.status.success() {
        return Ok(());
    }

    Err(ProgramError::Compiler {
        directory: directory.to_owned(),
        output: format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    })
}

fn library_flags(arguments: &mut Vec<String>) {
    for variable in ["CUDNN_INCLUDE_PATH", "CUTENSOR_INCLUDE_PATH"] {
        if let Ok(path) = env::var(variable) {
            arguments.push(format!("-I{path}"));
        }
    }
    for variable in ["LD_LIBRARY_PATH", "CUDNN_PATH", "CUTENSOR_PATH"] {
        if let Ok(paths) = env::var(variable) {
            let paths = paths.split(':').filter(|path| !path.is_empty());
            arguments.extend(paths.map(|path| format!("-L{path}")));
        }
    }
}

// A program's compile command, which also links it, run under `directory`.
fn program_arguments(directory: &Path) -> Result<Vec<String>, ProgramError> {
    let manifest = read_manifest(directory, "model")?;
    let mut arguments = compile_arguments(&manifest);
    arguments.retain(|argument| argument != "-c");
    library_flags(&mut arguments);
    arguments.extend(
        field(&manifest, "link")
            .split_whitespace()
            .map(ToOwned::to_owned),
    );
    arguments.extend(["-shared", "-Xcompiler", "-fPIC", "-o", LIBRARY_FILE].map(ToOwned::to_owned));
    Ok(arguments)
}

// Writes `program` into `directory` with the generation loop it was exported with, and builds that loop as the manifest says, returning the executable, which reads its weights from the same directory.
//
// # Errors
//
// Returns an error if the program has no loop, or writing or compiling fails.
pub fn build_generation_loop(
    program: &CudaProgram,
    directory: &Path,
) -> Result<PathBuf, ProgramError> {
    program.write_to(directory, "model")?;
    build_written_loop(directory)
}

// Writes a program of several steps into `directory` with its generation loop, and builds that loop as the manifest says, returning the executable.
//
// # Errors
//
// Returns an error if the set has no loop, or writing or compiling fails.
pub fn build_set_generation_loop(
    program: &CudaProgramSet,
    directory: &Path,
) -> Result<PathBuf, ProgramError> {
    program.write_to(directory, "model")?;
    build_written_loop(directory)
}

// Writes `program`, exported without a page layout, into `directory` with its runner, and builds the runner as the manifest says, returning the executable, which reads its weights from the same directory.
//
// # Errors
//
// Returns an error if the program has no runner, or writing or compiling fails.
pub fn build_runner(program: &CudaProgram, directory: &Path) -> Result<PathBuf, ProgramError> {
    program.write_to(directory, "model")?;
    build_written_runner(directory, "model")
}

// Builds the runner of the program `directory` holds, written as `prefix`, as its manifest says, returning the executable, `{prefix}_run`, which reads its weights from the same directory.
//
// # Errors
//
// Returns an error if the program has no runner, or reading its manifest or compiling fails.
pub fn build_written_runner(directory: &Path, prefix: &str) -> Result<PathBuf, ProgramError> {
    let manifest = read_manifest(directory, prefix)?;
    let command = manifest["run"]
        .as_str()
        .ok_or_else(|| Error::invalid_argument("program has no runner"))?;
    let mut arguments = command
        .split_whitespace()
        .skip(1)
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    library_flags(&mut arguments);
    run_compiler(&arguments, directory)?;
    Ok(directory.join(format!("{prefix}_run")))
}

// Builds the generation loop of the program set `directory` holds, written as `model`, as its manifest says, returning the executable, `model_generate`.
//
// # Errors
//
// Returns an error if the set has no generation loop, or reading its manifest or compiling fails.
pub fn build_written_loop(directory: &Path) -> Result<PathBuf, ProgramError> {
    let manifest = read_manifest(directory, "model")?;
    let command = manifest["decoding"]["generate"]
        .as_str()
        .ok_or_else(|| Error::invalid_argument("program has no generation loop"))?;
    let mut arguments = command
        .split_whitespace()
        .skip(1)
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    library_flags(&mut arguments);
    run_compiler(&arguments, directory)?;
    Ok(directory.join("model_generate"))
}

fn build_set(
    directory: &Path,
    manifest: &Value,
    steps: &[(String, Value)],
) -> Result<(), ProgramError> {
    let mut links = Vec::new();
    let mut units = vec![manifest];
    for (_, step) in steps {
        links.extend(
            field(step, "link")
                .split_whitespace()
                .map(ToOwned::to_owned),
        );
        units.push(step);
    }

    let mut objects = Vec::new();
    for unit in units {
        let mut arguments = compile_arguments(unit);
        let source = arguments
            .last()
            .cloned()
            .ok_or_else(|| Error::invalid_argument("program compile command"))?;
        let object = source.replace(".cu", ".o");
        library_flags(&mut arguments);
        arguments.extend(["-Xcompiler", "-fPIC", "-o", &object].map(ToOwned::to_owned));
        run_compiler(&arguments, directory)?;
        objects.push(object);
    }

    let mut arguments = objects;
    links.sort_unstable();
    links.dedup();
    library_flags(&mut arguments);
    arguments.extend(links);
    arguments.extend(["-shared", "-o", LIBRARY_FILE].map(ToOwned::to_owned));
    run_compiler(&arguments, directory)
}

// Writes what compiling `files` needs, which leaves out its weights, since those are read from memory.
fn write_for_compiling(files: &ProgramArtifact, directory: &Path) -> Result<(), ProgramError> {
    fs::create_dir_all(directory)?;
    fs::write(directory.join("model.h"), &files.header)?;
    fs::write(directory.join("model.cu"), &files.source)?;
    fs::write(directory.join("model.json"), &files.manifest)?;
    Ok(())
}

impl CompiledProgram {
    // What each program in `programs` builds into, as its entry says, on at most `jobs` compilers at once.
    // Those built together keep one copy of the weights they share, since a tuning step's candidates differ in little else.
    #[must_use]
    pub fn create_many(
        programs: Vec<(CudaProgram, ProgramArtifact, PathBuf)>,
        jobs: usize,
    ) -> Vec<Result<Self, ProgramError>> {
        let commands = programs
            .iter()
            .map(|(_, files, directory)| {
                write_for_compiling(files, directory)?;
                program_arguments(directory)
            })
            .collect::<Vec<_>>();
        let next = AtomicUsize::new(0);
        let mut outputs = (0..programs.len()).map(|_| None).collect::<Vec<_>>();
        let finished = thread::scope(|scope| {
            let workers = (0..jobs.clamp(1, programs.len().max(1)))
                .map(|_| {
                    scope.spawn(|| {
                        let mut finished = Vec::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some((_, _, directory)) = programs.get(index) else {
                                return finished;
                            };

                            if let Ok(arguments) = &commands[index] {
                                finished.push((index, build_library(arguments, directory)));
                            }
                        }
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .filter_map(|worker| worker.join().ok())
                .flatten()
                .collect::<Vec<_>>()
        });
        for (index, output) in finished {
            outputs[index] = Some(output);
        }

        let mut shared = Vec::<Arc<Vec<u8>>>::new();
        programs
            .into_iter()
            .zip(commands)
            .zip(outputs)
            .map(|(((program, files, directory), command), output)| {
                command?;
                output
                    .ok_or_else(|| Error::invalid_argument("a compiler thread stopped early"))??;
                let weights =
                    if let Some(same) = shared.iter().find(|weights| ***weights == files.weights) {
                        Arc::clone(same)
                    } else {
                        let weights = Arc::new(files.weights);
                        shared.push(Arc::clone(&weights));
                        weights
                    };
                Self::loaded(&program, directory, weights)
            })
            .collect()
    }

    fn loaded(
        program: &CudaProgram,
        directory: PathBuf,
        weights: Arc<Vec<u8>>,
    ) -> Result<Self, ProgramError> {
        // SAFETY: the library is the program just compiled, and its initializers only register kernels.
        let library = unsafe { Library::new(directory.join(LIBRARY_FILE)) }?;
        Ok(Self {
            built: Arc::new(Built {
                steps: vec![ProgramInterface::of(program)],
                library,
                weights,
                directory,
                files: FileRetention::Removed,
            }),
        })
    }

    // Writes `program` under `directory`, and compiles it with its manifest's command.
    //
    // # Errors
    //
    // Returns the compiler's output if `nvcc` rejects the program, or a description of a failure to access its files.
    pub fn create_in(program: &CudaProgram, directory: PathBuf) -> Result<Self, ProgramError> {
        program.write_to(&directory, "model")?;
        Self::create_from_written(program, directory)
    }

    // Compiles what `program` wrote under `directory`, which a caller may have read or changed first.
    //
    // # Errors
    //
    // Returns the compiler's output if `nvcc` rejects the program, or a description of a failure to run it.
    pub fn create_from_written(
        program: &CudaProgram,
        directory: PathBuf,
    ) -> Result<Self, ProgramError> {
        let arguments = program_arguments(&directory)?;
        build_library(&arguments, &directory)?;
        let weights = program.render("model")?.weights;
        Self::loaded(program, directory, Arc::new(weights))
    }

    // Writes a program of several steps under `directory`, and compiles each of its files with its manifest's command, as its user would.
    //
    // # Errors
    //
    // Returns the compiler's output if `nvcc` rejects the program, or a description of a failure to access its files.
    pub fn create_set_in(
        program: &CudaProgramSet,
        directory: PathBuf,
    ) -> Result<Self, ProgramError> {
        program.write_to(&directory, "model")?;
        Self::load(directory, FileRetention::Removed)
    }

    pub fn open(directory: PathBuf) -> Result<Self, ProgramError> {
        Self::load(directory, FileRetention::Kept)
    }

    fn load(directory: PathBuf, files: FileRetention) -> Result<Self, ProgramError> {
        let manifest = read_manifest(&directory, "model")?;
        let set = manifest["steps"].as_array();
        let steps = match set {
            Some(entries) => entries
                .iter()
                .map(|entry| {
                    let name = manifest_text(entry, "name")?;
                    let step = read_manifest(&directory, &format!("model_{name}"))?;
                    Ok((name, step))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?,
            None => vec![(String::new(), manifest.clone())],
        };
        if !directory.join(LIBRARY_FILE).exists() {
            if set.is_some() {
                build_set(&directory, &manifest, &steps)?;
            } else {
                build_library(&program_arguments(&directory)?, &directory)?;
            }
        }

        let weights = read_weights(&directory, &manifest)?;
        let steps = steps
            .into_iter()
            .map(|(name, step)| ProgramInterface::read(name, &step))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        // SAFETY: the library is the program its manifests describe, and its initializers only register kernels.
        let library = unsafe { Library::new(directory.join(LIBRARY_FILE)) }?;
        Ok(Self {
            built: Arc::new(Built {
                steps,
                library,
                weights: Arc::new(weights),
                directory,
                files,
            }),
        })
    }

    // Creates the program's state with its exported values.
    //
    // # Errors
    //
    // Returns the status `create` or `load_weights` reports.
    pub fn instance(&self) -> Result<ProgramInstance, ProgramError> {
        self.instance_with_table(&[])
    }

    // As `instance`, for a program written to read its forced library configurations from a table, which this one holds.
    //
    // # Errors
    //
    // Returns the status `create` or `load_weights` reports.
    pub fn instance_with_table(&self, table: &[i64]) -> Result<ProgramInstance, ProgramError> {
        let built = &self.built;
        let mut state = ptr::null_mut();
        // SAFETY: these functions have these signatures in the generated header, and the weights and table outlive the calls that read them.
        unsafe {
            if !table.is_empty() {
                let set = built
                    .library
                    .get::<SetLibraryTable>(b"model_set_library_table")?;
                set(table.as_ptr());
            }

            let create = built.library.get::<Create>(b"model_create")?;
            let status = create(&raw mut state);
            if status != 0 {
                return Err(built.failed("model_create", status, 0));
            }

            let instance = ProgramInstance {
                program: Arc::clone(built),
                state,
            };
            let load = built.library.get::<LoadWeights>(b"model_load_weights")?;
            let status = load(state, built.weights.as_ptr().cast());
            if status != 0 {
                return Err(built.failed("model_load_weights", status, 0));
            }
            Ok(instance)
        }
    }

    // Runs a new instance `runs` times on `inputs`, in the header's order, and returns what the last run wrote.
    //
    // # Errors
    //
    // Returns the status any entry point or checking kernel reports.
    pub fn run(&self, inputs: &[Tensor], runs: usize) -> Result<Vec<Tensor>, ProgramError> {
        self.instance()?.run(inputs, runs)
    }

    #[must_use]
    pub fn steps(&self) -> &[ProgramInterface] {
        &self.built.steps
    }

    // How many bytes of weights its file holds, which a program of several steps holds once.
    #[must_use]
    pub fn weights_bytes(&self) -> usize {
        self.built.weights.len()
    }
}

impl Built {
    // The name of an entry point for the step at `step`.
    fn symbol(&self, entry: &str, step: usize) -> String {
        let name = &self.steps[step].name;
        if name.is_empty() {
            format!("model_{entry}")
        } else {
            format!("model_{entry}_{name}")
        }
    }

    fn failed(&self, entry: &str, status: i32, word: u32) -> ProgramError {
        // SAFETY: the generated header declares this function with this signature, and it returns a string of the calling thread's that lives until its next failure.
        let failure = unsafe {
            self.library
                .get::<Failure>(b"model_failure")
                .ok()
                .map(|failure| failure())
                .filter(|text| !text.is_null())
                .map(|text| CStr::from_ptr(text).to_string_lossy().into_owned())
        };
        ProgramError::Status {
            entry: entry.to_owned(),
            status,
            word,
            failure: failure.unwrap_or_default(),
            directory: self.directory.clone(),
        }
    }
}

// The address the program counts a tensor's `offset` from.
fn origin(tensor: &Tensor) -> *mut c_void {
    let region = tensor.tensor_region().unwrap();
    let bytes = tensor.storage_offset() * tensor.data_type().size_bytes();
    region
        .region()
        .as_ptr()
        .cast::<u8>()
        .wrapping_sub(bytes)
        .cast()
}

// A program's state, destroyed on drop.
pub struct ProgramInstance {
    program: Arc<Built>,
    state: *mut c_void,
}

impl ProgramInstance {
    fn check_inputs(&self, step: usize, inputs: &[Tensor]) -> Result<(), ProgramError> {
        // The program reads each input at the address and layout its manifest gives, so a tensor of another count, type, layout, or device would be read out of bounds.
        let declared = &self.program.steps[step].inputs;
        if declared.len() != inputs.len() {
            return Err(ProgramError::InputCount {
                expected: declared.len(),
                found: inputs.len(),
            });
        }

        for (value, input) in declared.iter().zip(inputs) {
            let layout = Layout::from_tensor(input);
            if input.data_type() != value.data_type
                || layout != value.layout
                || input.device() != PROGRAM_DEVICE
            {
                return Err(ProgramError::Input {
                    name: value.graph_name.clone(),
                    expected: describe_tensor(value.data_type, &value.layout, PROGRAM_DEVICE),
                    found: describe_tensor(input.data_type(), &layout, input.device()),
                });
            }
        }
        Ok(())
    }

    fn output_buffers(&self, step: usize) -> Result<Vec<Tensor>, ProgramError> {
        self.program.steps[step]
            .outputs
            .iter()
            .map(|output| {
                Ok(Tensor::zeroed_on(
                    PROGRAM_DEVICE,
                    output.data_type,
                    output.layout.shape(),
                )?)
            })
            .collect()
    }

    // Times `runs` runs of the program on `inputs`, in the header's order, each from `model_run` until `model_finish` returns, with its buffers allocated beforehand.
    //
    // # Errors
    //
    // Returns the status any entry point or checking kernel reports.
    pub fn timed_runs(
        &self,
        inputs: &[Tensor],
        runs: usize,
    ) -> Result<Vec<Duration>, ProgramError> {
        self.check_inputs(0, inputs)?;
        let compiled = &self.program;
        let outputs = self.output_buffers(0)?;
        let input_pointers = inputs.iter().map(origin).collect::<Vec<_>>();
        let output_pointers = outputs.iter().map(origin).collect::<Vec<_>>();
        let mut times = Vec::with_capacity(runs);
        // SAFETY: the functions have these signatures in the generated header, a struct of pointers is laid out as these arrays are, and every buffer outlives the runs.
        unsafe {
            let run = compiled.library.get::<Run>(b"model_run")?;
            let finish = compiled.library.get::<Finish>(b"model_finish")?;
            for _ in 0..runs {
                let started = Instant::now();
                let status = run(
                    self.state,
                    input_pointers.as_ptr(),
                    output_pointers.as_ptr(),
                    ptr::null_mut(),
                );
                if status != 0 {
                    return Err(compiled.failed("model_run", status, 0));
                }

                let mut word = 0_u32;
                let status = finish(self.state, ptr::null_mut(), &raw mut word);
                if status != 0 || word != 0 {
                    return Err(compiled.failed("model_finish", status, word));
                }

                times.push(started.elapsed());
            }
        }
        Ok(times)
    }

    // Replaces the state tensor `name` with `values`, which have the tensor's layout.
    //
    // # Errors
    //
    // Returns the status the program's function reports.
    pub fn set(&self, name: &str, values: &Tensor) -> Result<(), ProgramError> {
        self.access(&format!("model_set_{name}"), origin(values))
    }

    // Replaces the state tensor `name` with `values`, which are in host memory and have the tensor's layout.
    //
    // # Errors
    //
    // Returns the status the program's function reports.
    pub fn set_from_host(&self, name: &str, values: &Tensor) -> Result<(), ProgramError> {
        let bytes = values.to_bytes()?;
        self.access(
            &format!("model_set_{name}"),
            bytes.as_ptr().cast_mut().cast(),
        )
    }

    // Returns a generator's seed and next block.
    //
    // # Errors
    //
    // Returns the status the program's function reports.
    pub fn generator(&self, name: &str) -> Result<(i64, i64), ProgramError> {
        let mut values = [0_i64; 2];
        self.access(&format!("model_get_{name}"), values.as_mut_ptr().cast())?;
        Ok(values.into())
    }

    // Copies the state tensor `name` out, in the layout it was exported with.
    //
    // # Errors
    //
    // Returns the status the program's function reports.
    pub fn get(&self, name: &str) -> Result<Tensor, ProgramError> {
        let state = self.program.steps[0]
            .state
            .iter()
            .find(|state| state.name == name)
            .ok_or_else(|| Error::missing_input("program state tensor"))?;
        let values = Tensor::zeroed_on(PROGRAM_DEVICE, state.data_type, state.layout.shape())?;
        self.access(&format!("model_get_{name}"), origin(&values))?;
        Ok(values)
    }

    // Replaces a generator's `[seed, next block]`, which live in host memory.
    //
    // # Errors
    //
    // Returns the status the program's function reports.
    pub fn set_generator(&self, name: &str, seed: i64, offset: i64) -> Result<(), ProgramError> {
        let values = [seed, offset];
        self.access(
            &format!("model_set_{name}"),
            values.as_ptr().cast_mut().cast(),
        )
    }

    fn access(&self, symbol: &str, values: *mut c_void) -> Result<(), ProgramError> {
        // SAFETY: the program declares this function with this signature, up to the type its pointers name, and copies only what the state tensor holds.
        let status = unsafe {
            let function = self.program.library.get::<Accessor>(symbol.as_bytes())?;
            function(self.state, values)
        };
        if status != 0 {
            return Err(self.program.failed(symbol, status, 0));
        }
        Ok(())
    }

    // Runs the program `runs` times on `inputs`, in the header's order, and returns what the last run wrote, with each counted result narrowed to its count.
    //
    // # Errors
    //
    // Returns the status any entry point or checking kernel reports.
    pub fn run(&self, inputs: &[Tensor], runs: usize) -> Result<Vec<Tensor>, ProgramError> {
        self.run_at(0, inputs, runs)
    }

    // Runs the step called `name` as `run` runs a program of one.
    //
    // # Errors
    //
    // Returns an error if there is no such step, or the status any entry point or checking kernel reports.
    pub fn run_step(
        &self,
        name: &str,
        inputs: &[Tensor],
        runs: usize,
    ) -> Result<Vec<Tensor>, ProgramError> {
        let step = self
            .program
            .steps
            .iter()
            .position(|step| step.name == name)
            .ok_or_else(|| Error::invalid_argument("program step"))?;
        self.run_at(step, inputs, runs)
    }

    fn run_at(
        &self,
        step: usize,
        inputs: &[Tensor],
        runs: usize,
    ) -> Result<Vec<Tensor>, ProgramError> {
        self.check_inputs(step, inputs)?;
        let compiled = &self.program;
        let outputs = self.output_buffers(step)?;
        let input_pointers = inputs.iter().map(origin).collect::<Vec<_>>();
        let output_pointers = outputs.iter().map(origin).collect::<Vec<_>>();
        let (run_name, finish_name) = (
            compiled.symbol("run", step),
            compiled.symbol("finish", step),
        );
        // SAFETY: the functions have these signatures in the generated header, a struct of pointers is laid out as an array of them, and every tensor outlives the run.
        unsafe {
            let run = compiled.library.get::<Run>(&run_name)?;
            let finish = compiled.library.get::<Finish>(&finish_name)?;
            for _ in 0..runs {
                let status = run(
                    self.state,
                    input_pointers.as_ptr(),
                    output_pointers.as_ptr(),
                    ptr::null_mut(),
                );
                if status != 0 {
                    return Err(compiled.failed("model_run", status, 0));
                }

                let mut word = 0_u32;
                let status = finish(self.state, ptr::null_mut(), &raw mut word);
                if status != 0 || word != 0 {
                    return Err(compiled.failed("model_finish", status, word));
                }
            }
        }

        // Counts are the trailing outputs.
        let counted = compiled.steps[step]
            .outputs
            .iter()
            .enumerate()
            .filter_map(|(position, output)| {
                output.count.map(|count| CountedOutput {
                    position,
                    axis: count.axis,
                })
            })
            .collect::<Vec<_>>();
        Ok(narrow_counted(&counted, outputs)?)
    }
}

// SAFETY: the state is host memory the program allocated and `Drop` frees, and its CUDA resources belong to the device's primary context, which any thread may use.
unsafe impl Send for ProgramInstance {}

impl Drop for ProgramInstance {
    fn drop(&mut self) {
        // SAFETY: the state came from this library's `create`, and nothing uses it after this.
        unsafe {
            if let Ok(destroy) = self.program.library.get::<Destroy>(b"model_destroy") {
                destroy(self.state);
            }
        }
    }
}
