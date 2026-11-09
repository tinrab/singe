use crate::{
    backend::cuda::compiled::plan::lowering::{CudaGraphProvider, RealizedLoweringPlan},
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
};

pub fn record_realized_cuda_lowering(
    realized_lowering: &mut RealizedLoweringPlan,
    id: ValueId,
    ir: &Graph,
    provider: CudaGraphProvider,
) -> Result<()> {
    let ir_operation = ir.required_operation_for_value(id)?;
    let operation_id = ir_operation.id();
    let operation = ir_operation
        .opcode()
        .ok_or_else(|| Error::missing_node("record realized cuda lowering"))?;
    realized_lowering.record(id, operation_id, provider, operation);
    Ok(())
}
