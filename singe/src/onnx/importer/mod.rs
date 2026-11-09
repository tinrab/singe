mod import;
mod io;
mod shape_values;
mod values;

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
};

pub use self::import::import_model;

use crate::{
    data_type::DataType,
    device::Device,
    graph::{Graph, ir::ValueId},
    onnx::{
        model::{OnnxModelMetadata, OnnxValueInfo},
        options::OnnxOptions,
    },
    shape::Shape,
    tensor::Tensor,
};

#[derive(Debug)]
pub struct ImportedModel {
    pub(crate) specialized_initializers: HashSet<String>,
    pub(crate) device: Device,
    pub(crate) graph: Graph,
    pub(crate) inputs: Vec<OnnxValueInfo>,
    pub(crate) outputs: Vec<OnnxValueInfo>,
    pub(crate) metadata: OnnxModelMetadata,
}

#[derive(Debug, Clone)]
pub struct ImportedValue {
    pub(crate) node: ValueId,
    pub(crate) data_type: DataType,
    pub(crate) shape: Shape,
}

pub struct Importer {
    specialized_initializers: RefCell<HashSet<String>>,
    graph: Graph,
    values: HashMap<String, ImportedValue>,
    initializer_tensors: HashMap<String, Tensor>,
    inputs: Vec<OnnxValueInfo>,
    options: OnnxOptions,
    model_dir: Option<PathBuf>,
    opsets: BTreeMap<Option<String>, i64>,
}

impl Importer {
    pub(crate) fn graph(&mut self) -> &mut Graph {
        &mut self.graph
    }

    pub(crate) fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    pub(crate) fn options(&self) -> &OnnxOptions {
        &self.options
    }

    pub(crate) fn opset_version(&self, domain: Option<&str>) -> i64 {
        let domain = domain.and_then(|domain| (!domain.is_empty()).then(|| domain.to_string()));
        self.opsets.get(&domain).copied().unwrap_or(1)
    }
}
