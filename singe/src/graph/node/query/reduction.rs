use crate::{
    backend::registry::{Operation, OperationAttributeKind},
    graph::{
        ir::{ValueId, operation_attributes},
        node::NodeRef,
    },
};

impl NodeRef<'_> {
    /// Returns reduction input, dimensions, and keepdim flag for shape-changing reductions.
    pub(crate) fn reduction_shape_inputs(&self) -> Option<(ValueId, Vec<i64>, bool)> {
        if !self
            .operation()?
            .has_attribute_kind(OperationAttributeKind::Reduction)
        {
            return None;
        }

        let input = self.inputs().first().copied()?;
        let operation_attributes = operation_attributes(self);
        let attributes = operation_attributes.reduction()?;
        Some((input, attributes.dimensions.clone(), attributes.keepdim))
    }
}

impl NodeRef<'_> {
    /// Returns input, axis, and singleton/empty folded value for softmax-like nodes.
    pub(crate) fn softmax_fold_inputs(&self) -> Option<(ValueId, i64, f32)> {
        match self.operation()? {
            Operation::Softmax => Some((
                self.inputs().first().copied()?,
                operation_attributes(self).dimension()?,
                1.0,
            )),
            Operation::LogSoftmax => Some((
                self.inputs().first().copied()?,
                operation_attributes(self).dimension()?,
                0.0,
            )),
            _ => None,
        }
    }

    /// Returns input and axis for arg-reduction nodes.
    pub(crate) fn arg_reduction_input(&self) -> Option<(ValueId, i64)> {
        match self.operation()? {
            Operation::ArgMax | Operation::ArgMin => Some((
                self.inputs().first().copied()?,
                operation_attributes(self).dimension()?,
            )),
            _ => None,
        }
    }
}

impl NodeRef<'_> {
    /// Returns input, k, axis, selection direction, and sorting flag for top-k nodes.
    pub(crate) fn topk(&self) -> Option<(ValueId, i64, i64, bool, bool)> {
        if !self
            .operation()?
            .has_attribute_kind(OperationAttributeKind::TopK)
        {
            return None;
        }

        let input = self.inputs().first().copied()?;
        let attributes = operation_attributes(self).topk().copied()?;
        Some((
            input,
            attributes.k,
            attributes.dim,
            attributes.largest,
            attributes.sorted,
        ))
    }

    /// Returns input, k, axis, and sorted flag for top-k value nodes.
    pub(crate) fn topk_values_identity(&self) -> Option<(ValueId, i64, i64, bool)> {
        if self.operation() != Some(Operation::TopKValues) {
            return None;
        }

        let input = self.inputs().first().copied()?;
        let attributes = operation_attributes(self).topk().copied()?;
        Some((input, attributes.k, attributes.dim, attributes.sorted))
    }
}
