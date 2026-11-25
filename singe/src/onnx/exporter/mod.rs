pub(crate) mod compare;
mod elementwise;
pub(crate) mod external;
mod fused;
mod ops;
mod proto;
mod shapes;

use std::{
    collections::{HashMap, HashSet},
    mem,
};

use singe_onnx::proto::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, StringStringEntryProto,
};

use crate::{
    data_type::DataType,
    graph::{
        CaptureInfo, GraphProgram, InputDimensionGuard, ProgramInputBinding, StateBindingKind,
        StateSnapshot, decompose_attention,
        ir::{Graph, ValueId},
    },
    onnx::{
        error::Result,
        export_contract::{SHAPE_CONTRACT, annotate_dimension},
        exporter::proto::{raw_tensor, tensor, unsupported, value_info},
    },
    scalar::Scalar,
    shape::Shape,
};

/// Exports a logical program and an explicit immutable state publication as inline ONNX protobuf bytes at IR version 14 and opset 28.
/// No provider is selected and no tensor operation executes.
/// CUDA state readback, when needed, is explicit export work.
///
/// Unsupported operations, value-specialized capture, and unrepresentable contracts return an error.
///
/// # Errors
///
/// Returns an error if state differs from the program schema, capture was specialized to observed values, the graph contains unsupported operations, types, shape expressions, or output projections, or initializer readback or graph conversion fails.
pub fn export(program: &GraphProgram, state: &StateSnapshot) -> Result<Vec<u8>> {
    external::encoded(&model(program, state)?)
}

fn model(program: &GraphProgram, state: &StateSnapshot) -> Result<ModelProto> {
    Ok(exported(program, state)?.model)
}

struct Exported {
    model: ModelProto,
    graph: Graph,
    ir: Graph,
    names: HashMap<ValueId, String>,
}

fn exported(program: &GraphProgram, state: &StateSnapshot) -> Result<Exported> {
    program.validate_state_schema(state)?;
    if program
        .capture_info()
        .is_some_and(CaptureInfo::allows_value_specialization)
    {
        return Err(unsupported(
            "value-specialized capture requires host-value guards that ONNX cannot represent",
        ));
    }

    // `Attention` has no sinks and aligns its causal mask to key lengths at the bottom right, so attention with either exports as what it computes.
    let graph = decompose_attention(program.logical_graph(), |attention| {
        attention.sinks || (attention.key_lengths && (attention.causal || attention.masked))
    })?;
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
        dynamic: program.signature().inputs().iter().any(|input| {
            input
                .dimension_guards()
                .iter()
                .any(|guard| !is_fixed(guard))
        }),
        grown: Default::default(),
        traced: None,
        origin: None,
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
                            .find(|guard| guard.axis() == axis && !is_fixed(guard))
                            .map_or(Value::DimValue(value), |guard| {
                                Value::DimParam(guard.name().into())
                            }),
                    )
                })
                .collect(),
        )?;
        for guard in input
            .dimension_guards()
            .iter()
            .filter(|guard| !is_fixed(guard))
        {
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
                // State the importer folded into constants cannot be overridden, since the values computed from it are already constants.
                if kind == StateBindingKind::Default
                    && !program.specialized_state().contains_key(input.name())
                {
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
        if exporter.dynamic {
            exporter.require_growth(output)?;
        }
    }

    prune(&mut exporter.graph, program);

    let names = exporter.names;
    let model = ModelProto {
        ir_version: IR_VERSION,
        producer_name: "singe".into(),
        producer_version: env!("CARGO_PKG_VERSION").into(),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: OPSET_VERSION,
        }],
        metadata_props: vec![StringStringEntryProto {
            key: SHAPE_CONTRACT.into(),
            value: "1".into(),
        }],
        graph: Some(exporter.graph),
        ..Default::default()
    };
    Ok(Exported {
        model,
        graph,
        ir,
        names,
    })
}

fn is_fixed(guard: &InputDimensionGuard) -> bool {
    // A dimension whose range holds one size, as `singe export --dim` fixes one, is written as that size.
    guard.min().is_some() && guard.min() == guard.max()
}

fn prune(graph: &mut GraphProto, program: &GraphProgram) {
    // Every value is emitted, so shape values no output depends on leave nodes behind; the shape constraints `value_info` annotates stay, as checks on the values they name.
    let mut read = graph
        .output
        .iter()
        .chain(&graph.value_info)
        .map(|value| value.name.clone())
        .collect::<HashSet<_>>();
    let mut kept = Vec::with_capacity(graph.node.len());
    for node in mem::take(&mut graph.node).into_iter().rev() {
        if node.output.iter().any(|output| read.contains(output)) {
            read.extend(node.input.iter().cloned());
            kept.push(node);
        }
    }

    kept.reverse();
    graph.node = kept;
    read.extend(graph.input.iter().map(|input| input.name.clone()));
    // State the importer folded into constants, such as a reshape's target, leaves initializers no node reads; other state stays, read or not, since its names are the model's published state.
    graph.initializer.retain(|initializer| {
        read.contains(&initializer.name)
            || !program.specialized_state().contains_key(&initializer.name)
    });
}

const IR_VERSION: i64 = 14;
const OPSET_VERSION: i64 = 28;

struct Exporter<'a> {
    source: &'a Graph,
    graph: GraphProto,
    names: HashMap<ValueId, String>,
    used: HashSet<String>,
    visiting: HashSet<ValueId>,
    serial: usize,
    dynamic: bool,
    grown: [HashMap<ValueId, Option<Shape>>; 2],
    traced: Option<String>,
    origin: Option<String>,
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
        self.outputs_node(operation, inputs, vec![output], attribute);
    }

    fn outputs_node(
        &mut self,
        operation: &str,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attribute: Vec<AttributeProto>,
    ) {
        let name = self.fresh();
        self.graph.node.push(NodeProto {
            name,
            op_type: operation.into(),
            input: inputs,
            output: outputs,
            attribute,
            doc_string: self.origin.clone().unwrap_or_default(),
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
            let inputs = operation
                .inputs()
                .iter()
                .map(|&input| self.emit(input))
                .collect::<Result<Vec<_>>>()?;
            let origin = operation.opcode().map(|opcode| {
                operation.source_location().map_or_else(
                    || format!("{opcode:?}"),
                    |location| format!("{opcode:?} at {location}"),
                )
            });
            let previous = mem::replace(&mut self.origin, origin);
            let output = if operation.results().len() == 1 {
                let output = self.fresh();
                self.operation(value, operation, inputs, output.clone())?;
                output
            } else {
                let outputs = operation
                    .results()
                    .iter()
                    .map(|_| self.fresh())
                    .collect::<Vec<_>>();
                self.results(operation, &inputs, outputs.clone())?;
                for (result, name) in operation.results().iter().zip(&outputs) {
                    if result.value() != value {
                        self.names.insert(result.value(), name.clone());
                    }
                }
                operation
                    .results()
                    .iter()
                    .zip(outputs)
                    .find_map(|(result, name)| (result.value() == value).then_some(name))
                    .ok_or_else(|| unsupported("operation without the requested result"))?
            };
            self.origin = previous;
            output
        };
        self.names.insert(value, output.clone());
        self.visiting.remove(&value);
        Ok(output)
    }
}
