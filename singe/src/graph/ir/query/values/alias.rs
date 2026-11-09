#[cfg(feature = "cuda")]
use std::collections::HashMap;

use crate::graph::ir::Graph;

#[cfg(feature = "cuda")]
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::graph::ir::ValueId;

impl Graph {
    /// Returns alias-group members keyed by alias root value.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if an alias references a value absent from the IR.
    #[cfg(feature = "cuda")]
    pub fn alias_group_members(&self) -> Result<HashMap<ValueId, Vec<ValueId>>> {
        let mut members = HashMap::<ValueId, Vec<ValueId>>::new();
        for value in &self.values {
            let root = self
                .alias_group_root_value(value.id())?
                .unwrap_or_else(|| value.id());
            members.entry(root).or_default().push(value.id());
        }
        Ok(members)
    }

    /// Returns the alias-group root for one value.
    ///
    /// Returns `None` for a value with no alias or a chain that exceeds the number of IR values.
    ///
    /// # Errors
    ///
    /// Returns a missing-node error if `id` or a referenced alias source is absent from the IR.
    #[cfg(feature = "cuda")]
    pub fn alias_group_root_value(&self, id: ValueId) -> Result<Option<ValueId>> {
        let mut current = id;
        let mut traversed = 0;
        loop {
            if traversed > self.values.len() {
                return Ok(None);
            }

            let Some(alias) = self.required_value_alias(current)? else {
                return Ok(None);
            };
            current = alias.source();
            traversed += 1;
            if !self.required_value_is_alias(current)? {
                return Ok(Some(current));
            }
        }
    }

    /// Returns whether one required IR value is an alias.
    #[cfg(feature = "cuda")]
    fn required_value_is_alias(&self, id: ValueId) -> Result<bool> {
        Ok(self.required_value(id)?.alias().is_some())
    }
}
