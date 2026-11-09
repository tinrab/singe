// Exports an ONNX model as a CUDA C++ program.
//
// `cargo run -p singe --features onnx,cuda_13_4 --example onnx_program -- <model.onnx> <directory> <architecture> [library ...]`
//
// With no libraries named the program is pure CUDA, and with `cublas`, `cublaslt`, `cudnn`, `cufft`, or `cutensor` it may call them.
// Operations only a library left out runs whole become their decompositions, which `model.json` lists.

use std::{env, error::Error, process};

use singe::{
    OnnxModel, OnnxOptions,
    backend::cuda::export::{CudaExport, Library},
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let [model, directory, architecture, libraries @ ..] = arguments.as_slice() else {
        eprintln!("usage: onnx_program <model.onnx> <directory> <architecture> [library ...]");
        process::exit(2);
    };
    let libraries = libraries
        .iter()
        .map(|library| library.parse())
        .collect::<Result<Vec<Library>, _>>()?;
    let model = OnnxModel::create(model, OnnxOptions::new())?;
    let program = CudaExport::new(architecture.parse()?)
        .with_libraries(libraries)
        .export(&model.graph()?)?;
    program.write_to(directory, "model")?;
    println!("wrote {} steps to {directory}", program.steps.len());
    Ok(())
}
