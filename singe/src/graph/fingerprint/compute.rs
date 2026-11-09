use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use crate::{
    error::{Error, Result},
    graph::{
        CompiledGraph, Graph,
        fingerprint::{
            GraphFingerprint, GraphFingerprintOptions,
            operation::hash_ir_operation,
            shape::{hash_device, hash_layout_with_symbols},
            stable::StableHasher,
            values::{
                hash_constant_tensor, hash_ir_value, hash_static_initializer, hash_value_ids,
            },
        },
        ir::{OperationId, ValueBindingKind, ValueId},
        symbolic::SymbolicDimension,
    },
};

impl CompiledGraph {
    /// Computes a stable graph fingerprint with explicit fingerprint options.
    ///
    /// # Errors
    ///
    /// Returns an error if node metadata is invalid or host constant bytes cannot be copied for hashable constants.
    pub fn fingerprint_with_options(
        &self,
        options: &GraphFingerprintOptions,
    ) -> Result<GraphFingerprint> {
        if options.compile() != self.compile_options() {
            return Err(Error::invalid_argument("fingerprint graph compile policy"));
        }
        fingerprint_graph(self, options)
    }
}

impl Graph {
    /// Compiles and fingerprints this graph with default compile options.
    ///
    /// # Errors
    ///
    /// Returns an error if graph compilation or fingerprinting fails.
    pub fn fingerprint(&self) -> Result<GraphFingerprint> {
        self.fingerprint_with_options(&GraphFingerprintOptions::default())
    }

    /// Compiles and fingerprints this graph with explicit fingerprint options.
    ///
    /// # Errors
    ///
    /// Returns an error if graph compilation or fingerprinting fails.
    pub fn fingerprint_with_options(
        &self,
        options: &GraphFingerprintOptions,
    ) -> Result<GraphFingerprint> {
        let compiled = self.compile_for_fingerprint_options(options)?;
        compiled.fingerprint_with_options(options)
    }

    /// Compiles this graph with the normalization policy included in a fingerprint options.
    ///
    /// # Errors
    ///
    /// Returns an error if graph compilation fails.
    pub fn compile_for_fingerprint_options(
        &self,
        options: &GraphFingerprintOptions,
    ) -> Result<CompiledGraph> {
        self.compile_with_options(options.compile())
    }
}

pub fn fingerprint_graph(
    compiled: &CompiledGraph,
    options: &GraphFingerprintOptions,
) -> Result<GraphFingerprint> {
    let ir = compiled.ir();
    let execution_order = &compiled.analysis.operation_order;
    let mut hasher = StableHasher::new();
    "singe.graph.fingerprint".hash(&mut hasher);
    options.hash(&mut hasher);
    execution_order.len().hash(&mut hasher);
    let symbolic_shapes = &compiled.analysis.symbolic_shapes;
    let local_values = local_result_values(ir, execution_order)?;

    let mut cacheable = hash_signature(compiled, &mut hasher)?;
    for (index, &id) in execution_order.iter().enumerate() {
        index.hash(&mut hasher);
        let operation = ir.required_operation(id)?;
        let projected_values = operation.result_values().collect::<Vec<_>>();
        let value_id = projected_values
            .first()
            .copied()
            .ok_or_else(|| Error::missing_output("fingerprint graph"))?;
        projected_values.len().hash(&mut hasher);
        for projected_value_id in projected_values {
            let value = ir.required_value(projected_value_id)?;
            hash_ir_value(
                value,
                symbolic_shapes.value_symbols(projected_value_id),
                &mut hasher,
            );
        }

        hash_ir_operation(operation, &local_values, &mut hasher)?;
        match ir.required_value_binding_kind(value_id)? {
            ValueBindingKind::StaticInput => {
                let initializer = ir.required_initializer_tensor(value_id)?;
                hash_static_initializer(initializer, &mut hasher);
            }
            ValueBindingKind::InitializerInput => {
                let initializer = ir.required_initializer_tensor(value_id)?;
                let local_node = index;
                cacheable &= hash_constant_tensor(local_node, initializer, &mut hasher)?;
            }
            ValueBindingKind::Constant => {
                let tensor = ir.required_constant_tensor(value_id)?;
                let local_node = index;
                cacheable &= hash_constant_tensor(local_node, tensor, &mut hasher)?;
            }
            ValueBindingKind::RuntimeInput | ValueBindingKind::OperationOutput => {}
        }
    }

    hash_value_ids(ir.output_values(), &local_values, &mut hasher)?;

    Ok(GraphFingerprint {
        value: hasher.finish(),
        cacheable,
    })
}

fn local_result_values(
    ir: &Graph,
    execution_order: &[OperationId],
) -> Result<HashMap<ValueId, usize>> {
    execution_order
        .iter()
        .try_fold(HashMap::new(), |mut local_values, id| {
            for value in ir.required_operation(*id)?.result_values() {
                let index = local_values.len();
                local_values.insert(value, index);
            }
            Ok::<_, Error>(local_values)
        })
}

fn hash_signature(compiled: &CompiledGraph, hasher: &mut StableHasher) -> Result<bool> {
    let inputs = compiled.signature().inputs();
    for output in compiled.signature().outputs() {
        output.name().hash(hasher);
    }

    inputs.len().hash(hasher);
    let mut cacheable = true;
    for (index, input) in inputs.iter().enumerate() {
        input.name().hash(hasher);
        input.data_type().hash(hasher);
        hash_device(input.device(), hasher);
        let mut symbols = vec![None; input.layout().dimensions().len()];
        for guard in input.dimension_guards() {
            symbols[guard.axis()] = Some(SymbolicDimension::new(
                guard.name(),
                guard.min(),
                guard.max(),
            ));
        }

        hash_layout_with_symbols(input.layout(), Some(&symbols), hasher);
        input.is_static_initializer().hash(hasher);
        input.has_initializer().hash(hasher);
        if let Some(initializer) = &input.initializer {
            if input.is_static_initializer() {
                hash_static_initializer(initializer, hasher);
            } else {
                cacheable &= hash_constant_tensor(index, initializer, hasher)?;
            }
        }
    }
    Ok(cacheable)
}
