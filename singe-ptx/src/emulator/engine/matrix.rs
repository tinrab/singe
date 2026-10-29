//! Execution of `ldmatrix` and `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
//!
//! Lane fragments follow PTX 9.4 sections 9.7.16.5.8 and 9.7.16.5.15.
//! All 32 lanes rendezvous at the same instruction. Inputs are
//! copied before any destination write, including when D aliases C or A.

use std::sync::atomic::AtomicBool;

use crate::{
    ast::{FloatRoundingMode, ScalarType, StateSpace},
    emulator::{
        engine::{Engine, Result, Waiting, cancel},
        execution_error::ExecutionError,
        ir::{
            Address, Clamp, ConversionControl, ConversionRounding, Destination, FloatControl,
            FloatFormat, FunctionId, MatrixLoad, MatrixMultiply, Source,
        },
        memory::Memory,
        numeric,
        target::WARP_SIZE,
    },
};

#[derive(Clone, Copy)]
pub struct MatrixWaiting {
    function: FunctionId,
    instruction: usize,
    operation: MatrixOperation,
}

#[derive(Clone, Copy)]
enum MatrixOperation {
    Rows4(MatrixRows<4>),
    Columns2(MatrixRows<2>),
    Multiply(MultiplyLane),
}

pub const SCRATCH_BYTES_PER_LANE: usize = size_of::<MatrixOperation>();

#[derive(Clone, Copy)]
struct MatrixRows<const N: usize> {
    destinations: [Destination; N],
    row: [u16; 8],
}

#[derive(Clone, Copy)]
struct MultiplyLane {
    destinations: [Destination; 4],
    a: [u32; 4],
    b: [u32; 2],
    c: [u32; 4],
}

enum MatrixBatch {
    Rows4(Vec<MatrixRows<4>>),
    Columns2(Vec<MatrixRows<2>>),
    Multiply(Vec<MultiplyLane>),
}

const CONVERSION: ConversionControl = ConversionControl {
    rounding: ConversionRounding::None,
    ftz: false,
    clamp: Clamp::None,
    finite: false,
};

const ARITHMETIC: FloatControl = FloatControl {
    rounding: FloatRoundingMode::Rn,
    ftz: false,
    clamp: Clamp::None,
    nan: false,
};

fn half(value: u16) -> u32 {
    numeric::low_u32(numeric::convert(
        u128::from(value),
        ScalarType::F16,
        ScalarType::F32,
        CONVERSION,
    ))
}

impl Engine {
    fn matrix_warp(&self, lane: usize) -> Result<()> {
        let start = lane - self.lanes[lane].thread % WARP_SIZE;
        if self
            .lanes
            .get(start + 31)
            .is_none_or(|last| last.cta != self.lanes[lane].cta)
        {
            return Err(ExecutionError::Invalid(
                "matrix operation requires a full 32-lane warp".into(),
            ));
        }
        Ok(())
    }

    fn park_matrix(&mut self, lane: usize, operation: MatrixOperation) {
        let frame = self.current(lane);
        self.lanes[lane].waiting = Some(Waiting::Matrix(MatrixWaiting {
            function: frame.function,
            instruction: frame.pc,
            operation,
        }));
        self.mark_waiting(lane);
    }

    pub(crate) fn wait_matrix_load(
        &mut self,
        memory: &Memory,
        lane: usize,
        layout: MatrixLoad,
        address: Address,
    ) -> Result<()> {
        self.matrix_warp(lane)?;
        let contributing =
            matches!(layout, MatrixLoad::Rows4(_)) || self.lanes[lane].warp_lane() < 16;
        let mut row = [0u16; 8];
        if contributing {
            let address = self.address(lane, address, Some(StateSpace::Shared))?;
            let bytes =
                self.read_memory(memory, lane, address, 16, 16, Some(StateSpace::Shared))?;
            for (value, bytes) in row.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *value = u16::from_le_bytes(*bytes);
            }
        }
        let operation = match layout {
            MatrixLoad::Rows4(destinations) => {
                MatrixOperation::Rows4(MatrixRows { destinations, row })
            }
            MatrixLoad::Columns2(destinations) => {
                MatrixOperation::Columns2(MatrixRows { destinations, row })
            }
        };
        self.park_matrix(lane, operation);
        Ok(())
    }

    fn matrix_sources<const N: usize>(
        &self,
        lane: usize,
        sources: &[Source; N],
        ty: ScalarType,
    ) -> Result<[u32; N]> {
        let mut values = [0; N];
        for (value, source) in values.iter_mut().zip(sources) {
            *value = numeric::low_u32(self.source(lane, source, ty)?);
        }
        Ok(values)
    }

    pub(crate) fn wait_matrix_multiply(
        &mut self,
        lane: usize,
        multiply: &MatrixMultiply,
    ) -> Result<()> {
        self.matrix_warp(lane)?;
        let operation = MatrixOperation::Multiply(MultiplyLane {
            destinations: multiply.destinations,
            a: self.matrix_sources(lane, &multiply.a, ScalarType::B32)?,
            b: self.matrix_sources(lane, &multiply.b, ScalarType::B32)?,
            c: self.matrix_sources(lane, &multiply.c, ScalarType::F32)?,
        });
        self.park_matrix(lane, operation);
        Ok(())
    }

    pub(crate) fn resolve_matrices(
        &mut self,
        warps: &[usize],
        cancellation: &AtomicBool,
    ) -> Result<()> {
        for &warp in warps {
            cancel(cancellation)?;
            let range = self.warps[warp].lanes.clone();
            let Some((leader, first)) =
                range
                    .clone()
                    .find_map(|lane| match self.lanes[lane].waiting {
                        Some(Waiting::Matrix(waiting)) => Some((lane, waiting)),
                        _ => None,
                    })
            else {
                continue;
            };
            // The grid reserves scratch for the largest matrix snapshot per
            // lane before execution, including this bounded rendezvous copy.
            let mut batch = match first.operation {
                MatrixOperation::Rows4(_) => MatrixBatch::Rows4(Vec::with_capacity(32)),
                MatrixOperation::Columns2(_) => MatrixBatch::Columns2(Vec::with_capacity(32)),
                MatrixOperation::Multiply(_) => MatrixBatch::Multiply(Vec::with_capacity(32)),
            };
            let mut participants = 0;
            for lane in range.clone() {
                cancel(cancellation)?;
                if self.lanes[lane].frames.is_empty() {
                    return Err(self.location(
                        leader,
                        ExecutionError::Invalid("matrix participant has exited".into()),
                    ));
                }
                if let Some(Waiting::Matrix(waiting)) = self.lanes[lane].waiting {
                    if (first.function, first.instruction)
                        != (waiting.function, waiting.instruction)
                    {
                        return Err(self.location(
                            lane,
                            ExecutionError::Invalid(
                                "matrix participants reached different instructions".into(),
                            ),
                        ));
                    }
                    match (&mut batch, waiting.operation) {
                        (MatrixBatch::Rows4(snapshots), MatrixOperation::Rows4(snapshot)) => {
                            snapshots.push(snapshot);
                        }
                        (MatrixBatch::Columns2(snapshots), MatrixOperation::Columns2(snapshot)) => {
                            snapshots.push(snapshot);
                        }
                        (MatrixBatch::Multiply(snapshots), MatrixOperation::Multiply(snapshot)) => {
                            snapshots.push(snapshot);
                        }
                        _ => {
                            return Err(self.location(
                                lane,
                                ExecutionError::Invalid("incompatible matrix operations".into()),
                            ));
                        }
                    }
                    participants += 1;
                }
            }
            if participants != 32 {
                continue;
            }
            match batch {
                MatrixBatch::Rows4(snapshots) => {
                    self.load_matrix_rows(range.start, &snapshots, cancellation)?;
                }
                MatrixBatch::Columns2(snapshots) => {
                    self.load_matrix_columns(range.start, &snapshots, cancellation)?;
                }
                MatrixBatch::Multiply(snapshots) => {
                    self.multiply_matrix_fragments(range.start, &snapshots, cancellation)?;
                }
            }
            for lane in range {
                self.release_waiter(lane);
            }
        }
        Ok(())
    }

    fn load_matrix_rows(
        &mut self,
        start: usize,
        snapshots: &[MatrixRows<4>],
        cancellation: &AtomicBool,
    ) -> Result<()> {
        for (lane, snapshot) in snapshots.iter().enumerate() {
            cancel(cancellation)?;
            let group = lane / 4;
            let pair = lane % 4 * 2;
            for (matrix, destination) in snapshot.destinations.iter().enumerate() {
                let row = snapshots[matrix * 8 + group].row;
                let value = u32::from(row[pair]) | u32::from(row[pair + 1]) << 16;
                self.write(
                    start + lane,
                    *destination,
                    u128::from(value),
                    ScalarType::B32,
                );
            }
        }
        Ok(())
    }

    fn load_matrix_columns(
        &mut self,
        start: usize,
        snapshots: &[MatrixRows<2>],
        cancellation: &AtomicBool,
    ) -> Result<()> {
        for (lane, snapshot) in snapshots.iter().enumerate() {
            cancel(cancellation)?;
            let group = lane / 4;
            let pair = lane % 4 * 2;
            for (matrix, destination) in snapshot.destinations.iter().enumerate() {
                let low = snapshots[matrix * 8 + pair].row;
                let high = snapshots[matrix * 8 + pair + 1].row;
                let value = u32::from(low[group]) | u32::from(high[group]) << 16;
                self.write(
                    start + lane,
                    *destination,
                    u128::from(value),
                    ScalarType::B32,
                );
            }
        }
        Ok(())
    }

    fn multiply_matrix_fragments(
        &mut self,
        start: usize,
        snapshots: &[MultiplyLane],
        cancellation: &AtomicBool,
    ) -> Result<()> {
        let mut a = [[0u32; 16]; 16];
        let mut b = [[0u32; 8]; 16];
        for (lane, snapshot) in snapshots.iter().enumerate() {
            cancel(cancellation)?;
            for element in 0..8 {
                let row = lane / 4 + (element / 2 % 2) * 8;
                let col = lane % 4 * 2 + element % 2 + element / 4 * 8;
                a[row][col] = half(numeric::low_u16(
                    snapshot.a[element / 2] >> (element % 2 * 16),
                ));
            }
            for element in 0..4 {
                let row = lane % 4 * 2 + element % 2 + element / 2 * 8;
                let col = lane / 4;
                b[row][col] = half(numeric::low_u16(
                    snapshot.b[element / 2] >> (element % 2 * 16),
                ));
            }
        }
        for (lane, snapshot) in snapshots.iter().enumerate() {
            cancel(cancellation)?;
            for element in 0..4 {
                let row = lane / 4 + element / 2 * 8;
                let col = lane % 4 * 2 + element % 2;
                let mut value = u128::from(snapshot.c[element]);
                for inner in 0..16 {
                    value = numeric::fma(
                        FloatFormat::F32,
                        u128::from(a[row][inner]),
                        u128::from(b[inner][col]),
                        value,
                        ARITHMETIC,
                    );
                }
                self.write(
                    start + lane,
                    snapshot.destinations[element],
                    value,
                    ScalarType::F32,
                );
            }
        }
        Ok(())
    }
}
