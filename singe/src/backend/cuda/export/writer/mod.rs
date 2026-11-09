// Renders a program as C++: the kernels, each in its own namespace, then functions that create its state and run its steps in order on a stream.

mod cudnn;
mod generate;
mod libraries;
pub mod set;
mod source;
mod state;
mod steps;

use std::{cell::RefCell, fs, path::Path};

use crate::{
    backend::cuda::{
        cudnn_backend::CudnnBackendTensorMetadata,
        export::program::{BufferKind, BufferView, CudaProgram, CudaProgramSet, Library},
    },
    data_type::DataType,
    error::{Error, Result},
};

// Values larger than this go to the weights file rather than into the source.
pub const EMBEDDED_VALUE_BYTES: usize = 16 << 10;

// A program's status: zero, a `cudaError_t` below every base, a library's base plus that library's own status, or one of the statuses below, which Singe defines.
// The library loaded has another major version than the headers the program compiled against.
pub const LIBRARY_VERSION: i32 = 5_000_000;

// A library refused a configuration the program's tuning records chose.
pub const CONFIGURATION_REFUSED: i32 = 5_000_001;

// The version of each library these calls were written against, which a test holds to the pinned headers.
pub const LIBRARY_VERSIONS: [(Library, &str); 5] = [
    (Library::Cublas, "13.8"),
    (Library::CublasLt, "13.8"),
    (Library::Cudnn, "9.27"),
    (Library::Cufft, "12.4"),
    (Library::Cutensor, "2.8"),
];

pub const CUBLAS_BASE: i32 = 1_000_000;
pub const CUFFT_BASE: i32 = 2_000_000;
pub const CUDNN_BASE: i32 = 3_000_000;
pub const CUTENSOR_BASE: i32 = 4_000_000;

// A program's files, named by its prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramFiles {
    pub header: String,
    pub source: String,
    pub weights: Vec<u8>,
    pub manifest: String,
    // A loop that generates through the program, for one exported with a page layout.
    pub generate: Option<String>,
    // How many configurations its records force on its libraries.
    pub forced_configurations: usize,
    // What those configurations hold, in a program written to read them from a table.
    pub library_table: Vec<i64>,
}

pub fn cuda_type(data_type: DataType) -> Result<&'static str> {
    Ok(match data_type {
        DataType::F16 => "CUDA_R_16F",
        DataType::Bf16 => "CUDA_R_16BF",
        DataType::F32 => "CUDA_R_32F",
        DataType::F64 => "CUDA_R_64F",
        DataType::F8E4M3 => "CUDA_R_8F_E4M3",
        DataType::F8E5M2 => "CUDA_R_8F_E5M2",
        _ => return Err(Error::unsupported_data_type("program library call")),
    })
}

pub fn cudnn_type(data_type: DataType) -> Result<&'static str> {
    Ok(match data_type {
        DataType::F16 => "CUDNN_DATA_HALF",
        DataType::Bf16 => "CUDNN_DATA_BFLOAT16",
        DataType::F32 => "CUDNN_DATA_FLOAT",
        DataType::F64 => "CUDNN_DATA_DOUBLE",
        DataType::I32 => "CUDNN_DATA_INT32",
        _ => return Err(Error::unsupported_data_type("program convolution")),
    })
}

// `text` as a C++ string literal.
pub fn string_literal(text: &str) -> String {
    let mut literal = String::with_capacity(text.len() + 2);
    literal.push('"');
    for character in text.chars() {
        match character {
            '"' | '\\' => {
                literal.push('\\');
                literal.push(character);
            }
            character if character.is_ascii_graphic() || character == ' ' => {
                literal.push(character);
            }
            _ => literal.push('?'),
        }
    }

    literal.push('"');
    literal
}

pub fn integers(values: impl IntoIterator<Item = i64>) -> String {
    values
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn alignment(view: &BufferView) -> i64 {
    CudnnBackendTensorMetadata::byte_alignment_for_layout(view.data_type, &view.layout)
}

// C++ for one library call: what the state holds for it, and the statements that set it up, run it, and release it.
#[derive(Default)]
pub struct LibraryCode {
    pub members: String,
    pub create: String,
    pub run: String,
    pub destroy: String,
}

pub struct Renderer<'a> {
    pub program: &'a CudaProgram,
    pub prefix: &'a str,
    // Which state tensors another program owns, which this one reads and neither loads nor frees; empty when it owns them all.
    pub shared: Vec<bool>,
    // What its records force on its libraries, as the steps write them.
    pub forced: RefCell<ForcedConfigurations>,
}

// How many configurations a program's records force on its libraries, and the table that holds them if it reads one.
#[derive(Debug, Default)]
pub struct ForcedConfigurations {
    pub count: usize,
    pub table: Vec<i64>,
}

impl Renderer<'_> {
    // A value a record forces, as C++: a literal, or its place in the table a tuning candidate reads, whose source then does not depend on it.
    pub fn forced_value(&self, value: i64) -> String {
        if !self.program.library_table {
            return value.to_string();
        }

        let mut forced = self.forced.borrow_mut();
        forced.table.push(value);
        format!("singe_library_table[{}]", forced.table.len() - 1)
    }

    // A byte address in device memory for `view`'s first element.
    pub fn address(&self, view: &BufferView) -> Result<String> {
        let buffer = &self.program.buffers[view.buffer.0];
        let base = match buffer.kind {
            BufferKind::Input(index) => format!("inputs->{}", self.program.inputs[index].name),
            BufferKind::Output(index) => format!("outputs->{}", self.program.outputs[index].name),
            _ => format!("s->buffers[{}]", view.buffer.0),
        };
        let bytes = view
            .layout
            .storage_offset()
            .checked_mul(view.data_type.size_bytes())
            .ok_or_else(|| Error::out_of_range("program view offset"))?;
        Ok(format!("singe_at({base}, {bytes}ULL)"))
    }

    // The buffer holding the state tensor with this index.
    pub fn state_buffer(&self, state: usize) -> Result<usize> {
        self.program
            .buffers
            .iter()
            .position(|buffer| buffer.kind == BufferKind::State(state))
            .ok_or_else(|| Error::missing_input("program state buffer"))
    }
}

// Replaces each `@name@` of `template` with its value in one pass, so inserted text is never rescanned.
pub fn fill_template(template: &str, values: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find('@') {
        let (before, marked) = rest.split_at(start);
        let (name, after) = marked[1..]
            .split_once('@')
            .expect("a template placeholder closes");
        let (_, value) = values
            .iter()
            .find(|(key, _)| *key == name)
            .unwrap_or_else(|| panic!("template placeholder `{name}` has no value"));
        out.push_str(before);
        out.push_str(value);
        rest = after;
    }

    out.push_str(rest);
    out
}

impl CudaProgram {
    // Renders the program's files, with `prefix` starting every name they declare.
    //
    // # Errors
    //
    // Returns an error if a step has no C++ form or `prefix` is not a C++ identifier.
    pub fn render(&self, prefix: &str) -> Result<ProgramFiles> {
        self.render_as(prefix, Vec::new())
    }

    // Renders the program as `render` does, reading the state tensors `shared` marks from another program.
    pub(crate) fn render_as(&self, prefix: &str, shared: Vec<bool>) -> Result<ProgramFiles> {
        if prefix.is_empty()
            || !prefix
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '_')
            || prefix.starts_with(|character: char| character.is_ascii_digit())
        {
            return Err(Error::invalid_argument(
                "program prefix must be a c++ identifier",
            ));
        }
        Renderer {
            program: self,
            prefix,
            shared,
            forced: RefCell::default(),
        }
        .render()
    }

    // Writes `<prefix>.h`, `<prefix>.cu`, `<prefix>.json`, `<prefix>.weights` when the program has large constants, and `<prefix>_generate.cu` when it was exported with a page layout.
    //
    // # Errors
    //
    // Returns an error if rendering fails or a file cannot be written.
    pub fn write_to(&self, directory: impl AsRef<Path>, prefix: &str) -> Result<()> {
        let files = self.render(prefix)?;
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        fs::write(directory.join(format!("{prefix}.h")), files.header)?;
        fs::write(directory.join(format!("{prefix}.cu")), files.source)?;
        fs::write(directory.join(format!("{prefix}.json")), files.manifest)?;
        if let Some(generate) = files.generate {
            fs::write(directory.join(format!("{prefix}_generate.cu")), generate)?;
        }

        if !files.weights.is_empty() {
            fs::write(directory.join(format!("{prefix}.weights")), files.weights)?;
        }
        Ok(())
    }
}

impl CudaProgramSet {
    // Writes every step's files and `<prefix>.h`, `<prefix>.cu`, `<prefix>.json`, and `<prefix>.weights` when the steps have large constants.
    //
    // # Errors
    //
    // Returns an error if rendering fails or a file cannot be written.
    pub fn write_to(&self, directory: impl AsRef<Path>, prefix: &str) -> Result<()> {
        let rendered = self.render(prefix)?;
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        for (name, contents) in rendered.files {
            fs::write(directory.join(name), contents)?;
        }

        if !rendered.weights.is_empty() {
            fs::write(
                directory.join(format!("{prefix}.weights")),
                rendered.weights,
            )?;
        }
        Ok(())
    }
}
