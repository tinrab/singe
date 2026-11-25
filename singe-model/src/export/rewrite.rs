use std::{collections::BTreeMap, fs, path::Path};

use singe::{
    Device,
    onnx::{Error as OnnxError, OnnxModel, OnnxOptions, WeightPlacement, export_file},
};
use singe_onnx::Model;

use crate::{
    error::{Error, Result},
    export::{compare, matched_lines},
    function::{
        Shapes, example_inputs,
        onnx::{options, spec},
    },
};

const FILE: &str = "model.onnx";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rewritten {
    pub nodes_before: usize,
    pub nodes_after: usize,
}

pub fn rewrite(
    path: &Path,
    directory: &Path,
    shapes: &Shapes,
    weights: WeightPlacement,
) -> Result<Rewritten> {
    let model = OnnxModel::create(path, rewrite_options(shapes)?)?;
    fs::create_dir_all(directory).map_err(Error::io(directory))?;
    let output = directory.join(FILE);
    export_file(model.program(), model.initializers(), &output, weights)?;
    Ok(Rewritten {
        nodes_before: nodes(path)?,
        nodes_after: nodes(&output)?,
    })
}

pub fn check_rewrite(path: &Path, directory: &Path, shapes: &Shapes) -> Result<Vec<String>> {
    let options = rewrite_options(shapes)?;
    let model = OnnxModel::create(path, options.clone())?;
    let rewritten = OnnxModel::create(directory.join(FILE), options)?;
    let specs = model
        .inputs()
        .iter()
        .filter(|info| !info.has_initializer)
        .map(|info| spec(info, Vec::new()))
        .collect::<Vec<_>>();
    let inputs = example_inputs(&specs, Device::Cpu)?;
    let want = model.run(&inputs)?;
    let have = rewritten.run(&inputs)?;
    let comparisons = model
        .outputs()
        .iter()
        .map(|info| {
            let missing =
                || Error::invalid(format!("a rewritten file without the output {}", info.name));
            compare(
                &info.name,
                want.get(&info.name).ok_or_else(missing)?,
                have.get(&info.name).ok_or_else(missing)?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    matched_lines(comparisons)
}

fn rewrite_options(shapes: &Shapes) -> Result<OnnxOptions> {
    let mut dimensions = BTreeMap::new();
    for (name, sizes) in &shapes.dimensions {
        let &[size] = sizes.as_slice() else {
            return Err(Error::invalid(format!(
                "dimension {name} with several sizes, where an ONNX file holds one size per axis"
            )));
        };
        dimensions.insert(name.clone(), size);
    }
    Ok(options(&dimensions, &shapes.inputs, Device::Cpu))
}

fn nodes(path: &Path) -> Result<usize> {
    let model = Model::from_path(path).map_err(OnnxError::from)?;
    Ok(model.graph.nodes.len())
}
