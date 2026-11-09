use crate::{
    error::{Error, Result},
    graph::{Graph, ir::ValueId},
};

impl Graph {}

impl Graph {}

impl Graph {
    /// Returns padding for an input rank without its trailing zero pairs.
    pub(crate) fn node_normalized_padding(
        &self,
        input: ValueId,
        padding: &[(i64, i64)],
    ) -> Result<Vec<(i64, i64)>> {
        let rank = self.required_value_shape(input)?.rank();
        let rank = usize::try_from(rank)
            .map_err(|_| Error::out_of_range("graph pad input rank".to_string()))?;
        if padding.len() > rank {
            return Err(Error::length_mismatch(
                "graph pad rank",
                rank,
                padding.len(),
            ));
        }

        let mut padding = padding.to_vec();
        while padding.last().copied() == Some((0, 0)) {
            padding.pop();
        }
        Ok(padding)
    }
}

impl Graph {}

impl Graph {}
