use crate::{
    backend::cuda::compiled::plan::cache_key::{
        ConcreteShapeBucket, ConcreteShapeBucketDimension, DynamicShapeMode, DynamicShapePolicy,
        SymbolicDimensionKey, dynamic::symbolic_dimension_key_material_from_ir,
    },
    error::Result,
    graph::ir::Graph,
};

pub fn concrete_shape_bucket_from_ir(
    ir: &Graph,
    policy: &DynamicShapePolicy,
) -> Result<Option<ConcreteShapeBucket>> {
    if matches!(policy.mode(), DynamicShapeMode::Reject) || !ir.has_input_dimension_guards() {
        return Ok(None);
    }

    let mut dimensions = Vec::new();
    for guard in symbolic_dimension_key_material_from_ir(ir) {
        let value = concrete_input_dimension_from_ir(ir, &guard)?;
        dimensions.push(ConcreteShapeBucketDimension::new(
            guard.input(),
            guard.axis(),
            guard.symbol(),
            value,
        ));
    }
    Ok(Some(ConcreteShapeBucket::new(dimensions)))
}

pub fn concrete_input_dimension_from_ir(ir: &Graph, guard: &SymbolicDimensionKey) -> Result<i64> {
    ir.required_input_dimension(guard.input(), guard.axis())
}
