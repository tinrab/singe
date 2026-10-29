//! Loads, stores, atomics, and fences.

use crate::{
    ast::{FloatRoundingMode, ScalarType},
    emulator::{
        engine::{Engine, Result},
        execution_error::ExecutionError,
        integer,
        ir::{self, AddressConversion, Atomic, MemoryOperation},
        memory::{Memory, WINDOW_BYTES, Window, canonical_address, generic_address},
        numeric::{self, mask},
    },
};

impl Engine {
    pub fn execute_memory(
        &mut self,
        global: &mut Memory,
        lane: usize,
        operation: &MemoryOperation,
    ) -> Result<()> {
        match operation {
            MemoryOperation::Load {
                ty,
                space,
                destinations,
                address,
                ..
            } => {
                let address = self.address(lane, *address, *space)?;
                let bytes = (ty.bit_width() / 8) as usize;
                let data = self
                    .read_memory(
                        global,
                        lane,
                        address,
                        bytes * destinations.len(),
                        bytes * destinations.len(),
                        *space,
                    )?
                    .to_vec();
                for (destination, chunk) in destinations.iter().zip(data.chunks_exact(bytes)) {
                    let mut value = [0; 16];
                    value[..bytes].copy_from_slice(chunk);
                    self.write(lane, *destination, u128::from_le_bytes(value), *ty);
                }
            }
            MemoryOperation::Store {
                ty,
                space,
                address,
                sources,
                ..
            } => {
                let address = self.address(lane, *address, *space)?;
                let bytes = (ty.bit_width() / 8) as usize;
                let mut data = Vec::with_capacity(bytes * sources.len());
                for source in sources {
                    data.extend_from_slice(&self.source(lane, source, *ty)?.to_le_bytes()[..bytes]);
                }
                self.store_memory(global, lane, address, &data, data.len(), *space)?;
            }
            MemoryOperation::ConvertAddress {
                ty,
                space,
                conversion,
                destination,
                source,
            } => {
                let value = numeric::low_u64(self.source(lane, source, *ty)?);
                let base = Window::of(*space).base();
                let value = if *conversion == AddressConversion::ToGeneric {
                    base.checked_add(value)
                        .filter(|v| *v < base + WINDOW_BYTES)
                        .map(generic_address)
                } else {
                    canonical_address(value)
                        .checked_sub(base)
                        .filter(|v| *v < WINDOW_BYTES)
                }
                .filter(|v| u128::from(*v) <= mask(ty.bit_width()))
                .ok_or_else(|| {
                    ExecutionError::Memory("address outside state-space window".into())
                })?;
                self.write(lane, *destination, u128::from(value), *ty);
            }
            MemoryOperation::Atomic {
                op,
                ty,
                space,
                destination,
                address,
                value,
                ..
            } => {
                let address = self.address(lane, *address, *space)?;
                let size = (ty.bit_width() / 8) as usize;
                let mut bytes = [0; 16];
                bytes[..size]
                    .copy_from_slice(self.read_memory(global, lane, address, size, size, *space)?);
                let old = u128::from_le_bytes(bytes);
                let value = self.source(lane, value, *ty)?;
                let compare = match op {
                    Atomic::CompareSwap { compare } => Some(self.source(lane, compare, *ty)?),
                    _ => None,
                };
                let new = atomic_update(op, *ty, old, value, compare);
                self.store_memory(
                    global,
                    lane,
                    address,
                    &new.to_le_bytes()[..size],
                    size,
                    *space,
                )?;
                self.write(lane, *destination, old, *ty);
            }
            MemoryOperation::Fence => {}
        }
        Ok(())
    }
}

/// Returns the value an atomic operation stores, given the old memory value and its operand;
/// `compare` is the resolved comparand of a compare-and-swap.
fn atomic_update(
    op: &Atomic,
    ty: ScalarType,
    old: u128,
    value: u128,
    compare: Option<u128>,
) -> u128 {
    match op {
        Atomic::Add if ty == ScalarType::F32 => numeric::float_binary(
            ir::FloatBinary::Add,
            ir::FloatFormat::F32,
            old,
            value,
            ir::FloatControl {
                rounding: FloatRoundingMode::Rn,
                ftz: true,
                clamp: ir::Clamp::None,
                nan: false,
            },
        ),
        Atomic::Add => old.wrapping_add(value),
        Atomic::Min => integer::binary(ir::IntegerBinary::Min, ty, old, value)
            .expect("min is defined for every input"),
        Atomic::Max => integer::binary(ir::IntegerBinary::Max, ty, old, value)
            .expect("max is defined for every input"),
        Atomic::And => old & value,
        Atomic::Or => old | value,
        Atomic::Xor => old ^ value,
        Atomic::Exchange => value,
        Atomic::CompareSwap { .. } => {
            if compare == Some(old) {
                value
            } else {
                old
            }
        }
        Atomic::Increment => {
            if old >= value {
                0
            } else {
                old + 1
            }
        }
        Atomic::Decrement => {
            if old == 0 || old > value {
                value
            } else {
                old - 1
            }
        }
    }
}
