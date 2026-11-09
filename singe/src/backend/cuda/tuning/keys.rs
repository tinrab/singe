use std::fmt::Write as _;

use sha2::{Digest, Sha256};
use singe_codegen::{abi::Bindings, ir::Kernel};

use crate::{
    error::Result,
    graph::{
        NumericalPolicy,
        ir::{Graph, ValueId},
    },
};

// The record that says how a whole program launches its kernels.
#[must_use]
pub fn dependent_launch_key() -> String {
    "dependent_launch".to_owned()
}

// The record that says whether a whole graph runs sibling matmuls as one.
#[must_use]
pub fn merged_projections_key() -> String {
    "merged_projections".to_owned()
}

// The record that says whether a whole graph's attention runs with its query's rotation.
#[must_use]
pub fn fused_attention_key() -> String {
    "fused_attention".to_owned()
}

// A value's type and the elements its layout addresses, since a kernel's choices depend on both, and a stride that addresses nothing does not.
fn value_key(ir: &Graph, value: ValueId, key: &mut String) -> Result<()> {
    let layout = ir.required_value_layout(value)?;
    let _ = write!(
        key,
        "{:?}{:?}{:?}+{}",
        ir.required_value_data_type(value)?,
        layout.dimensions(),
        layout.addressing_strides(),
        layout.addressing_offset() % 8,
    );
    Ok(())
}

fn operation_key(ir: &Graph, id: ValueId, key: &mut String) -> Result<()> {
    let operation = ir.required_operation_for_value(id)?;
    let _ = write!(
        key,
        "{}({:?})",
        operation.opcode().map_or("none", |opcode| opcode.name()),
        operation.attributes()
    );
    for &input in operation.inputs() {
        key.push(' ');
        value_key(ir, input, key)?;
    }

    key.push_str(" -> ");
    value_key(ir, id, key)
}

// The key of the choice among the ways to run the node producing `id`.
// Attention's two results are one choice, so its result operations share a key.
//
// # Errors
//
// Returns an error if the node or one of its values is missing.
pub fn way_key(ir: &Graph, id: ValueId) -> Result<String> {
    let mut key = format!("way|{:?}|", NumericalPolicy::current());
    let operation = ir.required_operation_for_value(id)?;
    let _ = write!(
        key,
        "{}({:?})",
        operation.opcode().map_or("none", |opcode| {
            opcode
                .name()
                .trim_end_matches("_output")
                .trim_end_matches("_lse")
        }),
        operation.attributes()
    );
    for &input in operation.inputs() {
        key.push(' ');
        value_key(ir, input, &mut key)?;
    }
    Ok(key)
}

// The key of the choice of how a library runs the node producing `id`, which a way record sends to it.
//
// # Errors
//
// Returns an error if the node or one of its values is missing.
pub fn library_key(ir: &Graph, id: ValueId) -> Result<String> {
    Ok(format!("library|{}", way_key(ir, id)?))
}

// The key of the choice whether `members` run as one group.
//
// # Errors
//
// Returns an error if a node or one of its values is missing.
pub fn group_key(ir: &Graph, members: &[ValueId]) -> Result<String> {
    let mut key = format!("group|{:?}|", NumericalPolicy::current());
    for &member in members {
        operation_key(ir, member, &mut key)?;
        key.push(';');
    }
    Ok(key)
}

// The key of a generated kernel's schedule: its IR's hash, since schedules depend on all of it, and its sizes.
#[must_use]
pub fn schedule_key(kernel: &Kernel, bindings: &Bindings) -> String {
    let digest = Sha256::digest(format!("{kernel:?}").as_bytes());
    let mut key = format!("schedule|{}|", kernel.name);
    for byte in digest.iter().take(16) {
        let _ = write!(key, "{byte:02x}");
    }

    let _ = write!(key, "|{:?}", bindings.sizes);
    key
}
