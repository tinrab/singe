use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use crate::{
    error::{Error, Result},
    graph::{
        GraphProgram, ProgramFingerprint,
        fingerprint::{
            operation::hash_ir_operation, shape::hash_shape_with_symbols,
            shapes::hash_shape_program, stable::StableHasher, values::hash_value_ids,
        },
        symbolic::SymbolicDimension,
    },
};

pub fn fingerprint_program(program: &GraphProgram) -> Result<ProgramFingerprint> {
    let logical = program.logical_graph();
    let mut hasher = StableHasher::new();
    if logical.shape_program.is_empty() {
        "singe.logical.program.v3"
    } else {
        "singe.logical.program.v5.shapes"
    }
    .hash(&mut hasher);
    program.specialized_state().hash(&mut hasher);
    let mut state_groups = HashMap::new();
    for binding in program.state_schema() {
        binding.name().hash(&mut hasher);
        // Binding IDs belong to their schema.
        // Hash the named tie relation, independently of parameter registration order and slot numbering.
        let next_group = state_groups.len();
        state_groups
            .entry(binding.id())
            .or_insert(next_group)
            .hash(&mut hasher);
    }

    let signature = program.signature();
    signature.inputs().len().hash(&mut hasher);
    for input in signature.inputs() {
        input.name().hash(&mut hasher);
        input.binding().hash(&mut hasher);
        input.spec().data_type().hash(&mut hasher);
        let mut symbols = vec![None; input.spec().shape().dimensions().len()];
        for guard in input.dimension_guards() {
            symbols[guard.axis()] = Some(SymbolicDimension::new(
                guard.name(),
                guard.min(),
                guard.max(),
            ));
        }

        hash_shape_with_symbols(input.spec().shape(), Some(&symbols), &mut hasher);
    }

    if !logical.shape_program.is_empty() {
        hash_shape_program(logical, &mut hasher)?;
        return Ok(ProgramFingerprint::new(hasher.finish()));
    }

    let compiled = program.logical_analysis()?;
    let ir = compiled.ir();
    let mut values = HashMap::new();
    for &operation in &compiled.analysis.operation_order {
        for value in ir.required_operation(operation)?.result_values() {
            let index = values.len();
            values.insert(value, index);
        }
    }

    for &id in &compiled.analysis.operation_order {
        let operation = ir.required_operation(id)?;
        for value in operation.result_values() {
            let metadata = ir.required_value(value)?;
            metadata.data_type().hash(&mut hasher);
            hash_shape_with_symbols(
                metadata.shape(),
                compiled.analysis.symbolic_shapes.value_symbols(value),
                &mut hasher,
            );
        }
        if operation.opcode().is_some() {
            hash_ir_operation(operation, &values, &mut hasher)?;
        } else {
            let value = operation
                .result_values()
                .next()
                .ok_or_else(|| Error::missing_output("fingerprint program"))?;
            if let Some(name) = ir.required_value(value)?.binding().input_name() {
                "input".hash(&mut hasher);
                name.hash(&mut hasher);
            } else {
                "constant".hash(&mut hasher);
                ir.required_constant_tensor(value)?
                    .to_bytes()?
                    .hash(&mut hasher);
            }
        }
    }

    signature.outputs().len().hash(&mut hasher);
    for output in signature.outputs() {
        output.name().hash(&mut hasher);
    }

    hash_value_ids(ir.output_values(), &values, &mut hasher)?;
    Ok(ProgramFingerprint::new(hasher.finish()))
}
