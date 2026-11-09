use crate::{error::Result, graph::ir::ValueId};

// What a compiled graph and an exported program both try first for a node, in the order they must agree on: a fusion group the node leads, a kernel with several results, an alias, and a reduction of nothing.
//
// They differ after this: graphs take cuDNN's fusions and zero-work nodes, and programs merge library matmuls before they get here.
pub trait NodePlacement {
    fn fusion_group(&mut self, id: ValueId) -> Result<bool>;

    fn kernel_result(&mut self, id: ValueId) -> Result<bool>;

    fn alias(&mut self, id: ValueId) -> Result<bool>;

    fn empty_reduction(&mut self, id: ValueId) -> Result<bool>;
}

// Places the node producing `id` if one of the shared rules takes it, and returns whether one did.
pub fn place_by_shared_rules(placement: &mut impl NodePlacement, id: ValueId) -> Result<bool> {
    Ok(placement.fusion_group(id)?
        || placement.kernel_result(id)?
        || placement.alias(id)?
        || placement.empty_reduction(id)?)
}
