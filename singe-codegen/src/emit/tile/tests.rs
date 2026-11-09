use std::{env, fs, path::PathBuf, process::Command};

use cutile_ir::bytecode::{BytecodeVersion, write_bytecode_version};

use crate::{
    emit::{EmitError, tile::emit},
    ir::{
        Access, AxisKind, BinaryOp, Combiner, ElementType, Kernel, Literal, OperandRole, Size,
        UnaryOp,
    },
    schedule::{Schedule, TileSchedule, tile_schedule},
};

/// Builds `out[i, j] = f32(x[i, j]) + y[j]` for BF16 `x`, F32 `y`, and an argument row count.
fn broadcast_addition() -> Kernel {
    let mut kernel = Kernel::new("singe_tile_test_add");
    let x = kernel.add_operand(ElementType::Bf16, OperandRole::Input);
    let y = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let columns = kernel.add_size(Size::Static(5));
    let one = kernel.add_size(Size::Static(1));
    let i = kernel.add_axis(rows, AxisKind::Parallel);
    let j = kernel.add_axis(columns, AxisKind::Parallel);
    let matrix = |operand| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, columns), (j, one)],
    };
    let lhs = kernel.load(matrix(x));
    let rhs = kernel.load(Access {
        operand: y,
        base: None,
        indirect: Vec::new(),
        strides: vec![(j, one)],
    });
    let sum = kernel.binary(BinaryOp::Add, lhs, rhs);
    let half = kernel.literal(ElementType::F32, Literal::Float(0.5));
    let above = kernel.binary(BinaryOp::Greater, sum, half);
    let scaled = kernel.binary(BinaryOp::Mul, sum, half);
    let result = kernel.select(above, sum, scaled);
    kernel.store(matrix(out), result);
    kernel
}

/// Builds the softmax of BF16 rows of an argument width.
fn softmax() -> Kernel {
    let mut kernel = Kernel::new("singe_tile_test_softmax");
    let x = kernel.add_operand(ElementType::Bf16, OperandRole::Input);
    let out = kernel.add_operand(ElementType::Bf16, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let width = kernel.add_size(Size::Argument);
    let one = kernel.add_size(Size::Static(1));
    let row = kernel.add_axis(rows, AxisKind::Parallel);
    let column = kernel.add_axis(width, AxisKind::Reduction);
    let matrix = |operand| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: vec![(row, width), (column, one)],
    };
    let value = kernel.load(matrix(x));
    let maximum = kernel.reduce(Combiner::Max, vec![column], value);
    let shifted = kernel.binary(BinaryOp::Sub, value, maximum);
    let exponential = kernel.unary(UnaryOp::Exp, shifted);
    let sum = kernel.reduce(Combiner::Sum, vec![column], exponential);
    let ratio = kernel.binary(BinaryOp::Div, exponential, sum);
    kernel.store(matrix(out), ratio);
    kernel
}

/// Compiles `kernel` under `schedule` with `tileiras` for the development GPU, or returns `false` when the toolkit has no `tileiras`.
fn compiles(kernel: &Kernel, schedule: Schedule) -> bool {
    let tileiras = env::var_os("CUDA_PATH").map_or_else(
        || PathBuf::from("tileiras"),
        |root| PathBuf::from(root).join("bin/tileiras"),
    );
    if Command::new(&tileiras).arg("--version").output().is_err() {
        return false;
    }
    let module = emit(kernel, schedule.tile().expect("a tile schedule")).unwrap();
    let bytecode = write_bytecode_version(&module, BytecodeVersion::V13_4).unwrap();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/tile-emitter-tests")
        .join(&kernel.name);
    fs::create_dir_all(&directory).unwrap();
    let input = directory.join("kernel.bc");
    let output = directory.join("kernel.cubin");
    fs::write(&input, bytecode).unwrap();
    let result = Command::new(&tileiras)
        .args(["--gpu-name", "sm_120", "-o"])
        .arg(&output)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "tileiras rejected {}:\n{}\n{}",
        kernel.name,
        String::from_utf8_lossy(&result.stderr),
        module.to_mlir_text()
    );
    true
}

#[test]
fn test_element_and_row_kernels_compile_with_tileiras() {
    let add = broadcast_addition();
    assert_eq!(
        tile_schedule(&add, &[3]).unwrap(),
        Schedule::Tiles { size: 1024 }
    );
    compiles(&add, Schedule::Tiles { size: 64 });
    let rows = softmax();
    let schedule = tile_schedule(&rows, &[4, 100]).unwrap();
    assert_eq!(schedule, Schedule::TileRows { width: 128 });
    compiles(&rows, schedule);
}

#[test]
fn test_tile_emission_rejects_mismatched_reductions_and_narrow_rows() {
    assert!(matches!(
        emit(&softmax(), TileSchedule::Tiles { size: 64 }),
        Err(EmitError::Unsupported { .. })
    ));
    assert!(matches!(
        emit(&softmax(), TileSchedule::TileRows { width: 100 }),
        Err(EmitError::Schedule(_))
    ));
    // A row narrower than its reduction cannot launch.
    assert!(
        Schedule::TileRows { width: 64 }
            .launch(&softmax(), &[4, 100])
            .is_err()
    );
}
