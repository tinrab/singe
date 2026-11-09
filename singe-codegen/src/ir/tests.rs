use crate::ir::{
    Access, AxisId, AxisKind, BinaryOp, Combiner, ElementType, Expr, Kernel, Literal, NodeId,
    OperandId, OperandRole, Size, UnaryOp, VerifyError,
};

/// Builds `out[i, j] = x[i, j] + y[j]` over an argument row count and four columns.
fn broadcast_add() -> Kernel {
    let mut kernel = Kernel::new("singe_add");
    let x = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let y = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let columns = kernel.add_size(Size::Static(4));
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
    kernel.store(matrix(out), sum);
    kernel
}

#[test]
fn test_kernels_print_one_line_per_declaration_node_and_store() {
    let kernel = broadcast_add();
    kernel.verify().unwrap();
    assert_eq!(
        kernel.to_string(),
        "kernel singe_add
size s0 = argument
size s1 = 4
size s2 = 1
operand o0 = input f32
operand o1 = input f32
operand o2 = output f32
axis a0 = parallel s0
axis a1 = parallel s1
%0 = f32 load o0[a0*s1 + a1*s2]
%1 = f32 load o1[a1*s2]
%2 = f32 add %0 %1
store o2[a0*s1 + a1*s2] = %2
"
    );
    assert_eq!(kernel.argument_sizes().count(), 1);
}

#[test]
fn test_domains_follow_accesses_and_drop_reduced_axes() {
    let mut kernel = Kernel::new("singe_row_sum");
    let x = kernel.add_operand(ElementType::Bf16, OperandRole::Input);
    let out = kernel.add_operand(ElementType::Bf16, OperandRole::Output);
    let rows = kernel.add_size(Size::Static(3));
    let width = kernel.add_size(Size::Argument);
    let one = kernel.add_size(Size::Static(1));
    let row = kernel.add_axis(rows, AxisKind::Parallel);
    let column = kernel.add_axis(width, AxisKind::Reduction);
    let value = kernel.load(Access {
        operand: x,
        base: None,
        indirect: Vec::new(),
        strides: vec![(row, width), (column, one)],
    });
    assert_eq!(kernel.nodes[value.0].element, ElementType::F32);
    let sum = kernel.reduce(Combiner::Sum, vec![column], value);
    assert_eq!(kernel.domain(value), [row, column]);
    assert_eq!(kernel.domain(sum), [row]);
    kernel.store(
        Access {
            operand: out,
            base: None,
            indirect: Vec::new(),
            strides: vec![(row, one)],
        },
        sum,
    );
    kernel.verify().unwrap();
}

#[test]
fn test_verification_rejects_kernels_that_break_the_rules() {
    // A value that depends on an axis its store does not address.
    let mut kernel = broadcast_add();
    kernel.stores[0].access.strides.truncate(1);
    assert_eq!(
        kernel.verify(),
        Err(VerifyError::StoreDomain { store: 0, axis: 1 })
    );

    // Reducing a parallel axis.
    let mut kernel = broadcast_add();
    let sum = kernel.reduce(Combiner::Sum, vec![AxisId(1)], NodeId(2));
    kernel.stores[0].value = sum;
    assert_eq!(
        kernel.verify(),
        Err(VerifyError::ReducedParallelAxis { node: 3, axis: 1 })
    );

    // Mixing types, and transcendental functions of integers.
    let mut kernel = broadcast_add();
    let index = kernel.add_node(ElementType::I64, Expr::AxisIndex(AxisId(0)));
    kernel.binary(BinaryOp::Add, NodeId(2), index);
    assert!(matches!(
        kernel.verify(),
        Err(VerifyError::Type { node: 4, .. })
    ));
    let mut kernel = broadcast_add();
    let index = kernel.add_node(ElementType::I64, Expr::AxisIndex(AxisId(0)));
    kernel.unary(UnaryOp::Exp, index);
    assert!(matches!(
        kernel.verify(),
        Err(VerifyError::Operation {
            node: 4,
            op: "exp",
            ..
        })
    ));

    // Literals of the wrong kind, writes to inputs, and unwritten outputs.
    let mut kernel = broadcast_add();
    kernel.literal(ElementType::F32, Literal::Int(1));
    assert_eq!(
        kernel.verify(),
        Err(VerifyError::Literal {
            node: 3,
            element: "f32"
        })
    );
    let mut kernel = broadcast_add();
    kernel.stores[0].access.operand = OperandId(0);
    assert_eq!(
        kernel.verify(),
        Err(VerifyError::StoreToInput {
            store: 0,
            operand: 0
        })
    );
    let mut kernel = broadcast_add();
    kernel.stores.clear();
    assert_eq!(
        kernel.verify(),
        Err(VerifyError::UnwrittenOutput { operand: 2 })
    );
}
