use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    fs, io,
    path::Path,
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};
use singe_codegen::{emit::EMITTER_VERSION, schedule::Schedule};
use singe_cuda::{device::Device as NativeCudaDevice, driver, nvrtc};
use singe_cudnn::backend::execution::EngineChoice;

use crate::{
    backend::{
        cuda::{
            codegen::tile_compiler_version,
            kernel_plan::KernelParameters,
            runtime::{CublasLtMatmulAlgorithmFingerprint, runtime_for_device},
            tuning::{keys::merged_projections_key, report::TuningReport},
        },
        registry::CudaGraphProvider,
    },
    device::Device,
    error::{CompileError, Error, Result},
    utility::Locked,
};

// How a record's time was taken, which is part of what it means: a stream launch pays host costs a graph replay pays once.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Measurement {
    // Replayed in a CUDA graph, which graphs read.
    #[default]
    GraphReplay,
    // Built as the program it is and run through `model_run` and `model_finish`, which programs read.
    Program,
}

// What records were measured under, which they hold for and no other.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningEnvironment {
    pub device: String,
    pub architecture: String,
    pub driver: String,
    pub nvrtc: String,
    pub tileiras: String,
    pub emitter: u32,
    pub measurement: Measurement,
    // An algorithm or engine a record names means something only to these versions.
    pub cublas: String,
    pub cudnn: String,
}

impl TuningEnvironment {
    pub(crate) fn current(device: Device, measurement: Measurement) -> Result<Self> {
        let ordinal = device
            .cuda_id()
            .ok_or_else(|| Error::invalid_argument("tuning device"))?;
        let properties = NativeCudaDevice::new(ordinal).properties()?;
        Ok(Self {
            device: properties.name,
            architecture: runtime_for_device(device)?.architecture().to_string(),
            driver: driver::version()?.to_string(),
            nvrtc: nvrtc::version()?.to_string(),
            // A toolkit without `tileiras` still tunes CUDA C++ schedules.
            tileiras: tile_compiler_version().unwrap_or_default(),
            emitter: EMITTER_VERSION,
            measurement,
            cublas: cublas_version(),
            cudnn: cudnn_version(),
        })
    }
}

fn cublas_version() -> String {
    singe_cublas::version().map_or_else(|_| String::new(), |version| version.to_string())
}

fn cudnn_version() -> String {
    singe_cudnn::version().map_or_else(|_| String::new(), |version| version.to_string())
}

// One way to run a choice point's nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TunedWay {
    Library(CudaGraphProvider),
    Generated,
    // Members of a fusion group each run on their own.
    Separate,
}

// What a record chose, stated outright so it never depends on a list that can change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TunedChoice {
    Way(TunedWay),
    Parameters(KernelParameters),
    Schedule(Schedule),
    // How cuBLASLt runs a matmul that a way record sends to it.
    CublasLt(CublasLtMatmulAlgorithmFingerprint),
    // Which engine and knobs cuDNN runs a node with that a way record sends to it.
    Cudnn(EngineChoice),
    // A program's kernels overlap those they follow, which is one choice for a whole program, since timing one kernel cannot show a dependency.
    DependentLaunch,
    // Sibling matmuls run as one, which is also one choice for a whole graph, since it can break a fusion worth more than it saves.
    MergedProjections,
    // A query's norm and rotation run in its attention's kernel, which takes them from the module that writes the cache.
    FusedAttention,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TunedRecord {
    pub choice: TunedChoice,
    // The times of the first candidate, which has none when it failed, and of the winner, in nanoseconds, and what the winner allocates, which only a reader of the file uses.
    pub default_ns: Option<u64>,
    pub best_ns: u64,
    pub bytes: u64,
}

// How many reads of these records found a usable entry; one that no longer fits is an error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecordCoverage {
    pub used: usize,
}

// What a compile or export has read so far.
#[derive(Debug, Default)]
struct Reads {
    coverage: RecordCoverage,
    used: BTreeSet<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RecordsFile {
    environment: Option<TuningEnvironment>,
    entries: BTreeMap<String, TunedRecord>,
    report: Option<TuningReport>,
}

// What the tuning step measured, which compiling and exporting read and never add to.
#[derive(Debug, Default)]
pub struct TuningRecords {
    environment: Option<TuningEnvironment>,
    entries: BTreeMap<String, TunedRecord>,
    report: Option<TuningReport>,
    reads: Arc<Mutex<Reads>>,
}

// A clone counts its own reads, since each compile or export reports what it read.
impl Clone for TuningRecords {
    fn clone(&self) -> Self {
        Self {
            environment: self.environment.clone(),
            entries: self.entries.clone(),
            report: self.report.clone(),
            reads: Arc::default(),
        }
    }
}

impl PartialEq for TuningRecords {
    fn eq(&self, other: &Self) -> bool {
        self.environment == other.environment && self.entries == other.entries
    }
}

impl Eq for TuningRecords {}

impl TuningRecords {
    #[must_use]
    pub fn new(environment: Option<TuningEnvironment>) -> Self {
        Self {
            environment,
            ..Self::default()
        }
    }

    // Reads records from `path`; a record that no longer fits is found when it is read, and is work to redo.
    //
    // # Errors
    //
    // Returns an error if the file cannot be read or parsed.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(CompileError::TuningIo)?;
        let file: RecordsFile = serde_json::from_slice(&bytes)
            .map_err(|error| CompileError::TuningIo(io::Error::other(error)))?;
        Ok(Self {
            environment: file.environment,
            entries: file.entries,
            report: file.report,
            reads: Arc::default(),
        })
    }

    // Writes records to `path` atomically, so a reader never sees half a file.
    //
    // # Errors
    //
    // Returns an error if the file cannot be written.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = RecordsFile {
            environment: self.environment.clone(),
            entries: self.entries.clone(),
            report: self.report.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&file)
            .map_err(|error| CompileError::TuningIo(io::Error::other(error)))?;
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, bytes)
            .and_then(|()| fs::rename(&temporary, path))
            .map_err(|error| CompileError::TuningIo(error).into())
    }

    #[must_use]
    pub const fn environment(&self) -> Option<&TuningEnvironment> {
        self.environment.as_ref()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> impl Iterator<Item = (&str, &TunedRecord)> {
        self.entries
            .iter()
            .map(|(key, record)| (key.as_str(), record))
    }

    pub fn insert(&mut self, key: String, record: TunedRecord) {
        self.entries.insert(key, record);
    }

    // Adds what `other` found, as for another size of the same step, whose keys differ.
    //
    // # Errors
    //
    // Returns an error if `other` was measured under another environment, for which these records do not hold.
    pub fn merge(&mut self, other: Self) -> Result<()> {
        if self.environment != other.environment {
            return Err(Error::invalid_argument(
                "tuning records of different environments do not merge",
            ));
        }

        self.entries.extend(other.entries);
        match (&mut self.report, other.report) {
            (Some(report), Some(other)) => {
                report.choices.extend(other.choices);
                report.checks.extend(other.checks);
                report.searches.extend(other.searches);
            }
            (report, other) => {
                if report.is_none() {
                    *report = other;
                }
            }
        }
        Ok(())
    }

    // Whether these records hold where `measurement` is how work runs, on `architecture` when that is known, as records with no environment always do.
    #[must_use]
    pub fn holds_for(&self, architecture: Option<&str>, measurement: Measurement) -> bool {
        self.mismatch(architecture, measurement).is_none()
    }

    // What differs between where these records were made and where they are read, if anything does.
    #[must_use]
    pub fn mismatch(&self, architecture: Option<&str>, measurement: Measurement) -> Option<String> {
        let environment = self.environment.as_ref()?;
        let differs = |what: &str, made: &dyn Debug, here: &dyn Debug| {
            Some(format!("{what} {made:?}, where this has {here:?}"))
        };
        if environment.emitter != EMITTER_VERSION {
            return differs(
                "with kernel emitter",
                &environment.emitter,
                &EMITTER_VERSION,
            );
        }

        if environment.measurement != measurement {
            return differs("measured as", &environment.measurement, &measurement);
        }

        if environment.cublas != cublas_version() {
            return differs("with cuBLAS", &environment.cublas, &cublas_version());
        }

        if environment.cudnn != cudnn_version() {
            return differs("with cuDNN", &environment.cudnn, &cudnn_version());
        }

        match architecture {
            Some(architecture) if environment.architecture != architecture => {
                differs("for", &environment.architecture, &architecture)
            }
            _ => None,
        }
    }

    // These records as read where `measurement` is how work runs, which lets the tuner plan what a program's records leave.
    pub(crate) fn measured_as(mut self, measurement: Measurement) -> Self {
        if let Some(environment) = &mut self.environment {
            environment.measurement = measurement;
        }
        self
    }

    pub(crate) fn get(&self, key: &str) -> Option<TunedChoice> {
        self.entries.get(key).map(|record| record.choice.clone())
    }

    // Whether a record runs sibling matmuls as one.
    pub(crate) fn merges_projections(&self) -> bool {
        self.get(&merged_projections_key()) == Some(TunedChoice::MergedProjections)
    }

    // Whether a record chooses `way`, which lets planning make that way possible.
    pub(crate) fn chooses(&self, way: TunedWay) -> bool {
        self.entries
            .values()
            .any(|record| record.choice == TunedChoice::Way(way))
    }

    pub(crate) fn note_used(&self, key: &str) {
        if let Ok(mut reads) = self.reads.locked("tuning records") {
            reads.coverage.used += 1;
            reads.used.insert(key.to_owned());
        }
    }

    // The choices reads of these records have used, which with the same graph decide what a plan runs.
    pub(crate) fn used_choices(&self) -> Vec<(String, TunedChoice)> {
        self.reads.locked("tuning records").map_or_else(
            |_| Vec::new(),
            |reads| {
                reads
                    .used
                    .iter()
                    .filter_map(|key| Some((key.clone(), self.get(key)?)))
                    .collect()
            },
        )
    }

    // What the tuning step tried, when these records came from one.
    #[must_use]
    pub const fn report(&self) -> Option<&TuningReport> {
        self.report.as_ref()
    }

    pub(crate) fn set_report(&mut self, report: TuningReport) {
        self.report = Some(report);
    }

    // How reads of these records have gone so far.
    #[must_use]
    pub fn coverage(&self) -> RecordCoverage {
        self.reads
            .locked("tuning records")
            .map_or_else(|_| RecordCoverage::default(), |reads| reads.coverage)
    }
}
