// Builds exported programs as their users would, with `nvcc`, then loads and runs them, which tuning and tests both do.

use std::{
    env,
    ffi::{CStr, c_char, c_void},
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
    ptr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use libloading::Library;
use singe_cuda::architecture::GpuArchitecture;
use thiserror::Error as ThisError;

use crate::{
    backend::cuda::{
        export::{
            CudaProgram, CudaProgramSet,
            writer::{CONFIGURATION_REFUSED, CUBLAS_BASE, LIBRARY_VERSION, ProgramFiles},
        },
        runtime::runtime_for_device,
        tensor_region::CudaTensorSource,
    },
    device::Device,
    error::Error,
    graph::narrow_counted,
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
    Manifest(#[from] serde_json::Error),
    #[error("failed to load the program")]
    Load(#[from] libloading::Error),
    #[error("nvcc failed in {directory}:\n{output}")]
    Compiler { directory: PathBuf, output: String },
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

struct Built {
    // Its steps by name; the one step of a program of one has none.
    steps: Vec<(String, CudaProgram)>,
    library: Library,
    weights: Arc<Vec<u8>>,
    directory: PathBuf,
}

// Programs that fail to compile keep their files, for reading what the compiler rejected.
impl Drop for Built {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

fn read_manifest(directory: &Path, prefix: &str) -> Result<serde_json::Value, ProgramError> {
    let manifest = fs::read_to_string(directory.join(format!("{prefix}.json")))?;
    Ok(serde_json::from_str(&manifest)?)
}

fn field(manifest: &serde_json::Value, name: &str) -> String {
    manifest[name].as_str().unwrap_or_default().to_owned()
}

// A manifest's compile command, without the compiler's own name.
fn compile_arguments(manifest: &serde_json::Value) -> Vec<String> {
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
    arguments
        .extend(["-shared", "-Xcompiler", "-fPIC", "-o", "libmodel.so"].map(ToOwned::to_owned));
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

fn build_written_loop(directory: &Path) -> Result<PathBuf, ProgramError> {
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

// Writes what compiling `files` needs, which leaves out its weights, since those are read from memory.
fn write_for_compiling(files: &ProgramFiles, directory: &Path) -> Result<(), ProgramError> {
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
        programs: Vec<(CudaProgram, ProgramFiles, PathBuf)>,
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
                                let output = Command::new("nvcc")
                                    .args(arguments)
                                    .current_dir(directory)
                                    .output();
                                finished.push((index, output));
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
                let output = output
                    .ok_or_else(|| Error::invalid_argument("a compiler thread stopped early"))??;
                compiler_result(&output, &directory)?;
                let weights =
                    if let Some(same) = shared.iter().find(|weights| ***weights == files.weights) {
                        Arc::clone(same)
                    } else {
                        let weights = Arc::new(files.weights);
                        shared.push(Arc::clone(&weights));
                        weights
                    };
                Self::loaded(program, directory, weights)
            })
            .collect()
    }

    fn loaded(
        program: CudaProgram,
        directory: PathBuf,
        weights: Arc<Vec<u8>>,
    ) -> Result<Self, ProgramError> {
        // SAFETY: the library is the program just compiled, and its initializers only register kernels.
        let library = unsafe { Library::new(directory.join("libmodel.so")) }?;
        Ok(Self {
            built: Arc::new(Built {
                steps: vec![(String::new(), program)],
                library,
                weights,
                directory,
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
        run_compiler(&arguments, &directory)?;
        // SAFETY: the library is the program just compiled, and its initializers only register kernels.
        let library = unsafe { Library::new(directory.join("libmodel.so")) }?;
        let weights = program.render("model")?.weights;
        Ok(Self {
            built: Arc::new(Built {
                steps: vec![(String::new(), program.clone())],
                library,
                weights: Arc::new(weights),
                directory,
            }),
        })
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
        let manifest = read_manifest(&directory, "model")?;
        let mut objects = Vec::new();
        let mut units = vec![manifest];
        let mut links = Vec::new();
        for entry in &program.entries {
            let step = read_manifest(&directory, &format!("model_{}", entry.name))?;
            links.extend(
                field(&step, "link")
                    .split_whitespace()
                    .map(ToOwned::to_owned),
            );
            units.push(step);
        }

        for unit in &units {
            let mut arguments = compile_arguments(unit);
            let source = arguments
                .last()
                .cloned()
                .ok_or_else(|| Error::invalid_argument("program compile command"))?;
            let object = source.replace(".cu", ".o");
            library_flags(&mut arguments);
            arguments.extend(["-Xcompiler", "-fPIC", "-o", &object].map(ToOwned::to_owned));
            run_compiler(&arguments, &directory)?;
            objects.push(object);
        }

        let mut arguments = objects;
        links.sort_unstable();
        links.dedup();
        library_flags(&mut arguments);
        arguments.extend(links);
        arguments.extend(["-shared", "-o", "libmodel.so"].map(ToOwned::to_owned));
        run_compiler(&arguments, &directory)?;
        // SAFETY: the library is the program just compiled, and its initializers only register kernels.
        let library = unsafe { Library::new(directory.join("libmodel.so")) }?;
        let weights = program.render("model")?.weights;
        Ok(Self {
            built: Arc::new(Built {
                steps: program
                    .entries
                    .iter()
                    .map(|entry| (entry.name.clone(), entry.program.clone()))
                    .collect(),
                library,
                weights: Arc::new(weights),
                directory,
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

    // The program this was built from, the first step of several.
    #[must_use]
    pub fn program(&self) -> &CudaProgram {
        &self.built.steps[0].1
    }

    // Each step's name and program.
    #[must_use]
    pub fn steps(&self) -> &[(String, CudaProgram)] {
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
    fn symbol(&self, entry: &str, step: usize) -> Vec<u8> {
        let name = &self.steps[step].0;
        if name.is_empty() {
            format!("model_{entry}").into_bytes()
        } else {
            format!("model_{entry}_{name}").into_bytes()
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
    fn output_buffers(&self, step: usize) -> Result<Vec<Tensor>, ProgramError> {
        self.program.steps[step]
            .1
            .outputs
            .iter()
            .map(|output| {
                Ok(Tensor::zeroed_on(
                    Device::Cuda(0),
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
            .1
            .state
            .iter()
            .find(|state| state.name == name)
            .ok_or_else(|| Error::missing_input("program state tensor"))?;
        let values = Tensor::zeroed_on(Device::Cuda(0), state.data_type, state.layout.shape())?;
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
            .position(|(step, _)| step == name)
            .ok_or_else(|| Error::invalid_argument("program step"))?;
        self.run_at(step, inputs, runs)
    }

    fn run_at(
        &self,
        step: usize,
        inputs: &[Tensor],
        runs: usize,
    ) -> Result<Vec<Tensor>, ProgramError> {
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
            .1
            .outputs
            .iter()
            .enumerate()
            .filter_map(|(index, output)| output.count.map(|_| index))
            .collect::<Vec<_>>();
        Ok(narrow_counted(&counted, outputs)?)
    }
}

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
