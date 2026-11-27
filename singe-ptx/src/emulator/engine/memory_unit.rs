//! Loads, stores, atomics, and fences.

use crate::{
    ast::{FloatRoundingMode, ScalarType},
    emulator::{
        engine::{Engine, Result},
        execution_error::ExecutionError,
        integer,
        ir::{self, AddressConversion, Atomic, MemoryOperation, MulticastAccess},
        memory::{Memory, Target, WINDOW_BYTES, Window, canonical_address, generic_address},
        numeric::{self, mask},
    },
};

impl Engine {
    pub(crate) fn execute_memory(
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
                        .filter(|v| u128::from(*v) <= mask(ty.bit_width()))
                        .ok_or_else(|| {
                            ExecutionError::Memory("address outside state-space window".into())
                        })?
                } else {
                    // PTX leaves converting an address of another space undefined, and compilers convert before testing the space with `isspacep`, so only an access through the result faults.
                    let converted = canonical_address(value).wrapping_sub(base);
                    (u128::from(converted) & mask(ty.bit_width())) as u64
                };
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
            MemoryOperation::Multicast {
                access,
                ty,
                space,
                address,
                destinations,
                sources,
            } => {
                let address = self.address(lane, *address, *space)?;
                let bytes = (ty.bit_width() / 8) as usize;
                let count = destinations.len().max(sources.len());
                let targets = global.multicast_targets(
                    address,
                    bytes * count,
                    bytes * count,
                    Some(self.actor(lane)),
                )?;
                let element = |target: &Target, index: usize| Target {
                    base: target.base,
                    offset: target.offset + index * bytes,
                };
                let read = |global: &Memory, location: &Target| -> Result<u128> {
                    let mut value = [0; 16];
                    value[..bytes].copy_from_slice(global.read_target(location, bytes)?);
                    Ok(u128::from_le_bytes(value))
                };
                if let MulticastAccess::LoadReduce(op) = access {
                    for (index, destination) in destinations.iter().enumerate() {
                        let mut combined = None;
                        for target in &targets {
                            let value = read(global, &element(target, index))?;
                            combined = Some(
                                combined
                                    .map_or(value, |old| atomic_update(op, *ty, old, value, None)),
                            );
                        }

                        self.write(lane, *destination, combined.unwrap_or(0), *ty);
                    }
                } else {
                    let values = sources
                        .iter()
                        .map(|source| self.source(lane, source, *ty))
                        .collect::<Result<Vec<_>>>()?;
                    for target in &targets {
                        for (index, value) in values.iter().enumerate() {
                            let location = element(target, index);
                            let new = if let MulticastAccess::Reduce(op) = access {
                                atomic_update(op, *ty, read(global, &location)?, *value, None)
                            } else {
                                *value
                            };
                            global.write_target(&location, &new.to_le_bytes()[..bytes]);
                        }
                    }
                }
            }
            MemoryOperation::Fence => {}
        }
        Ok(())
    }
}

/// Returns the value an atomic operation stores, given the old memory value and its operand; `compare` is the resolved comparand of a compare-and-swap.
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
