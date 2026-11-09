mod compatibility;

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use singe_onnx::Model;

use crate::{
    device::Device,
    graph::Graph,
    onnx::{
        error::{Error, Result},
        export_contract::apply_shape_contract,
        importer::{ImportedModel, Importer},
        model::OnnxModelMetadata,
        ops::lower_node,
        options::OnnxOptions,
    },
};

use self::compatibility::{validate_inputs, validated_opsets};

#[expect(
    clippy::too_many_lines,
    reason = "the import transaction keeps model validation and graph assembly together"
)]
pub fn import_model(
    model: Model,
    mut options: OnnxOptions,
    model_dir: Option<PathBuf>,
) -> Result<ImportedModel> {
    apply_shape_contract(&model, &mut options)?;
    options.validate()?;
    let opsets = validated_opsets(&model)?;
    let device = options.device;
    options.device = Device::Cpu;
    let mut importer = Importer {
        specialized_initializers: RefCell::new(HashSet::new()),
        graph: Graph::new(),
        values: HashMap::new(),
        initializer_tensors: HashMap::new(),
        inputs: Vec::new(),
        options,
        model_dir,
        opsets: opsets.clone(),
    };
    importer.import_initializers(&model.graph.initializers)?;
    importer.import_inputs(&model.graph.inputs)?;
    importer.import_initializer_constants()?;
    for (index, node) in model.graph.nodes.iter().enumerate() {
        let mut node = node.clone();
        while node.outputs.last().is_some_and(String::is_empty) {
            node.outputs.pop();
        }

        validate_inputs(&importer, &node)?;
        let operands = node
            .inputs
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| {
                let value = importer.value(name)?;
                Ok(format!(
                    "{name}: {:?} {:?}",
                    value.data_type,
                    value.shape.dimensions()
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let source = format!(
            "onnx node {index} `{}` domain `{}` {} opset {} operands [{}]",
            node.name,
            node.domain.as_deref().unwrap_or(""),
            node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            operands.join(", ")
        );
        let start = importer.graph.node_count();
        lower_node(&mut importer, &node)?;
        importer.propagate_shape_value(&node)?;
        importer.graph.set_source_location_from(start, &source);
    }

    let mut metadata_names = HashSet::new();
    for info in &model.graph.value_info {
        if !metadata_names.insert(&info.name) {
            return Err(Error::DuplicateValue(info.name.clone()));
        }

        importer.validate_value_info(info)?;
    }

    let outputs = importer.import_outputs(&model.graph.outputs)?;
    importer.validate_symbolic_metadata(&model.graph)?;
    for info in model
        .graph
        .inputs
        .iter()
        .chain(&model.graph.value_info)
        .chain(&model.graph.outputs)
    {
        importer.record_shape_constraint(info)?;
    }

    let output_values = outputs
        .iter()
        .map(|output| {
            importer
                .values
                .get(&output.name)
                .map(|value| (output.name.clone(), value.node, 0))
                .ok_or_else(|| Error::MissingValue(output.name.clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    importer.graph.set_named_output_projections(output_values)?;
    let specialized_initializers = importer.specialized_initializers.into_inner();
    for name in &specialized_initializers {
        if let Some(tensor) = importer.graph.initializers.get(name).cloned() {
            importer.graph.add_static_initializer(name, tensor)?;
        }
    }

    let metadata = OnnxModelMetadata {
        ir_version: model.ir_version,
        producer_name: model.producer_name,
        producer_version: model.producer_version,
        domain: model.domain,
        model_version: model.model_version,
        operator_sets: opsets,
    };
    Ok(ImportedModel {
        specialized_initializers,
        device,
        graph: importer.graph,
        inputs: importer.inputs,
        outputs,
        metadata,
    })
}
