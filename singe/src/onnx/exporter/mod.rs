mod ops;
mod proto;
mod shapes;

use std::collections::{HashMap, HashSet};

use singe_onnx::{
    encode_proto,
    proto::onnx::{
        AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
        StringStringEntryProto,
    },
};

use crate::{
    data_type::DataType,
    graph::{
        CaptureInfo, GraphProgram, ProgramInputBinding, StateBindingKind, StateSnapshot,
        decompose_all_attention,
        ir::{Graph, ValueId},
    },
    onnx::{
        error::Result,
        export_contract::{SHAPE_CONTRACT, annotate_dimension},
        exporter::proto::{raw_tensor, tensor, unsupported, value_info},
    },
    scalar::Scalar,
};

/// Exports a logical program and an explicit immutable state publication as inline ONNX opset-20 protobuf bytes.
/// No provider is selected and no tensor operation executes.
/// CUDA state readback, when needed, is explicit export work.
///
/// Unsupported operations, value-specialized capture, and unrepresentable contracts return an error.
///
/// # Errors
///
/// Returns an error if state differs from the program schema, capture or state was specialized to observed values, the graph contains unsupported operations, types, shape expressions, or output projections, or initializer readback or graph conversion fails.
#[expect(
    clippy::too_many_lines,
    reason = "the export transaction keeps model sections and their cross-references together"
)]
pub fn export(program: &GraphProgram, state: &StateSnapshot) -> Result<Vec<u8>> {
    program.validate_state_schema(state)?;
    if program
        .capture_info()
        .is_some_and(CaptureInfo::allows_value_specialization)
    {
        return Err(unsupported(
            "value-specialized capture requires host-value guards that ONNX cannot represent",
        ));
    }

    if !program.specialized_state().is_empty() {
        return Err(unsupported(
            "value-specialized state is outside the export contract",
        ));
    }

    // ONNX has no operation for fused attention, so it exports as what it computes.
    let graph = decompose_all_attention(program.logical_graph())?;
    let ir = graph.to_ir()?;
    let source = &ir;
    if !source.counted_outputs().is_empty() {
        return Err(unsupported(
            "outputs whose extent depends on values are outside the export contract",
        ));
    }

    // A graph that writes inputs returns their new values as trailing outputs, named `present.<input>` beside the input they update.
    let visible_outputs = source
        .output_values()
        .len()
        .checked_sub(source.input_writes().len())
        .ok_or_else(|| unsupported("missing written-input outputs"))?;
    let mut exporter = Exporter {
        source,
        graph: GraphProto {
            name: "singe".into(),
            ..Default::default()
        },
        names: HashMap::new(),
        used: program
            .signature()
            .inputs()
            .iter()
            .map(|input| input.name().to_owned())
            .chain(
                program
                    .signature()
                    .outputs()
                    .iter()
                    .filter_map(|output| output.name().map(str::to_owned)),
            )
            .collect(),
        visiting: HashSet::new(),
        serial: 0,
        dynamic: program
            .signature()
            .inputs()
            .iter()
            .any(|input| !input.dimension_guards().is_empty()),
    };
    for input in program.signature().inputs() {
        let mut info = value_info(
            input.name().into(),
            input.spec().data_type(),
            input
                .spec()
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .map(|(axis, &value)| {
                    use singe_onnx::proto::onnx::tensor_shape_proto::dimension::Value;
                    Some(
                        input
                            .dimension_guards()
                            .iter()
                            .find(|guard| guard.axis() == axis)
                            .map_or(Value::DimValue(value), |guard| {
                                Value::DimParam(guard.name().into())
                            }),
                    )
                })
                .collect(),
        )?;
        for guard in input.dimension_guards() {
            annotate_dimension(
                &mut info,
                guard.axis(),
                input.spec().shape().dimensions()[guard.axis()],
                guard.min(),
                guard.max(),
            );
        }
        match input.binding() {
            ProgramInputBinding::Runtime => exporter.graph.input.push(info),
            ProgramInputBinding::State(kind) => {
                let value = state
                    .tensor(input.name())
                    .ok_or_else(|| unsupported(format!("missing state `{}`", input.name())))?;
                exporter
                    .graph
                    .initializer
                    .push(tensor(input.name().into(), value)?);
                if kind == StateBindingKind::Default {
                    exporter.graph.input.push(info);
                }
            }
        }
    }

    // Recursion also follows dependencies carried only by shape expressions.
    for value in source.values() {
        exporter.emit(value.id())?;
    }

    exporter.constraints()?;
    for (index, &output) in source.output_values().iter().enumerate() {
        if source
            .result_for_value(output)
            .is_some_and(|(_, result)| result.index() != 0)
        {
            return Err(unsupported("multi-result output projection"));
        }

        let value = exporter.emit(output)?;
        let name = index.checked_sub(visible_outputs).map_or_else(
            || {
                source
                    .output_name(index)
                    .map_or_else(|| exporter.fresh(), str::to_owned)
            },
            |write| format!("present.{}", source.input_writes()[write]),
        );
        if value != name {
            // A public output cannot redefine an input or initializer in ONNX SSA.
            if exporter.names.values().any(|existing| existing == &name) {
                return Err(unsupported(format!(
                    "output `{name}` would redefine another value"
                )));
            }

            exporter.node("Identity", vec![value], name.clone(), vec![]);
        }

        let spec = program.signature().outputs()[index].spec();
        let dims = if exporter.dynamic {
            vec![None; spec.shape().dimensions().len()]
        } else {
            shapes::fixed_dimensions(spec.shape().dimensions())
        };
        exporter
            .graph
            .output
            .push(value_info(name, spec.data_type(), dims)?);
    }
    Ok(encode_proto(&ModelProto {
        ir_version: 10,
        producer_name: "singe".into(),
        producer_version: env!("CARGO_PKG_VERSION").into(),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 20,
        }],
        metadata_props: vec![StringStringEntryProto {
            key: SHAPE_CONTRACT.into(),
            value: "1".into(),
        }],
        graph: Some(exporter.graph),
        ..Default::default()
    }))
}

struct Exporter<'a> {
    source: &'a Graph,
    graph: GraphProto,
    names: HashMap<ValueId, String>,
    used: HashSet<String>,
    visiting: HashSet<ValueId>,
    serial: usize,
    dynamic: bool,
}

impl Exporter<'_> {
    fn fresh(&mut self) -> String {
        loop {
            let name = format!("singe_{}", self.serial);
            self.serial += 1;
            if self.used.insert(name.clone()) {
                return name;
            }
        }
    }

    fn node(
        &mut self,
        operation: &str,
        inputs: Vec<String>,
        output: String,
        attribute: Vec<AttributeProto>,
    ) {
        let name = self.fresh();
        self.graph.node.push(NodeProto {
            name,
            op_type: operation.into(),
            input: inputs,
            output: vec![output],
            attribute,
            ..Default::default()
        });
    }

    fn temporary(
        &mut self,
        operation: &str,
        inputs: Vec<String>,
        attributes: Vec<AttributeProto>,
    ) -> String {
        let output = self.fresh();
        self.node(operation, inputs, output.clone(), attributes);
        output
    }

    fn literal(&mut self, dtype: DataType, shape: Vec<i64>, bytes: Vec<u8>) -> Result<String> {
        let name = self.fresh();
        self.graph
            .initializer
            .push(raw_tensor(name.clone(), dtype, shape, bytes)?);
        Ok(name)
    }

    fn scalar(&mut self, value: Scalar, dtype: DataType) -> Result<String> {
        self.literal(dtype, vec![], proto::scalar_bytes(value, dtype)?)
    }

    fn integers(&mut self, shape: Vec<i64>, values: &[i64]) -> Result<String> {
        self.literal(
            DataType::I64,
            shape,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
    }

    fn emit(&mut self, value: ValueId) -> Result<String> {
        if let Some(name) = self.names.get(&value) {
            return Ok(name.clone());
        }

        if !self.visiting.insert(value) {
            return Err(unsupported("cycle in logical graph or shape expressions"));
        }

        let source = self.source;
        let output = if let Some(name) = source.required_value(value)?.binding().input_name() {
            name.to_owned()
        } else if let Some(shape) = source.shape_program().values.get(&value) {
            self.shape_value(shape.clone())?
        } else if let Some(tensor) = source.constant_tensor(value) {
            let name = self.fresh();
            self.graph
                .initializer
                .push(proto::tensor(name.clone(), tensor)?);
            name
        } else {
            let operation = source.required_operation_for_value(value)?;
            if operation.results().len() != 1 {
                return Err(unsupported("multi-result operation"));
            }

            let inputs = operation
                .inputs()
                .iter()
                .map(|&input| self.emit(input))
                .collect::<Result<Vec<_>>>()?;
            let output = self.fresh();
            self.operation(value, operation, inputs, output.clone())?;
            output
        };
        self.names.insert(value, output.clone());
        self.visiting.remove(&value);
        Ok(output)
    }
}
