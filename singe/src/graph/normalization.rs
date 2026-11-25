use crate::{
    error::{Error, Result},
    graph::{Graph, ir::ValueId},
    shape::spatial::Padding,
};

impl Graph {
    /// Returns padding for an input rank without its trailing zero pairs.
    pub(crate) fn node_normalized_padding(
        &self,
        input: ValueId,
        padding: &[Padding],
    ) -> Result<Vec<Padding>> {
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
        while padding.last() == Some(&Padding::ZERO) {
            padding.pop();
        }
        Ok(padding)
    }
}
