//! The parameters every emitted kernel takes, in order, whatever the target.
//!
//! An emitted kernel's parameters are its operands' addresses in operand order, then the address of its status word when it reports status, then its scalars in declaration order, each in its type, then its argument sizes in [`Kernel::argument_sizes`] order, each an unsigned 64-bit integer.
//! Launching code builds the argument list from [`parameters`], so it never depends on an emitter.

use crate::ir::{ElementType, Kernel, Literal, OperandId, ScalarId, SizeId};

/// One parameter of an emitted kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Parameter {
    /// The address of an operand's first element.
    Operand(OperandId),
    /// The address of the status word, an unsigned 32-bit integer that starts at zero, into which the kernel writes the first status code it reports; it may be null.
    Status,
    /// A scalar, passed as its element type.
    Scalar(ScalarId, ElementType),
    /// An argument size, passed as an unsigned 64-bit integer.
    Size(SizeId),
}

/// Returns `kernel`'s parameters in the order every emitter declares them.
#[must_use]
pub fn parameters(kernel: &Kernel) -> Vec<Parameter> {
    let operands = (0..kernel.operands.len()).map(|index| Parameter::Operand(OperandId(index)));
    let scalars = kernel
        .scalars
        .iter()
        .enumerate()
        .map(|(index, &element)| Parameter::Scalar(ScalarId(index), element));
    let sizes = kernel.argument_sizes().map(Parameter::Size);
    let status = kernel.reports_status().then_some(Parameter::Status);
    operands.chain(status).chain(scalars).chain(sizes).collect()
}

/// The launch values of a kernel's argument sizes and scalars.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bindings {
    /// One value per argument size, in [`Kernel::argument_sizes`] order.
    pub sizes: Vec<u64>,
    /// One value per scalar, in declaration order.
    pub scalars: Vec<Literal>,
}
