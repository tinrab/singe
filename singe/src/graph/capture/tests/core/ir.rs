use super::*;
use crate::graph::ir::Graph;

macro_rules! find_attribute {
    ($ir:ident, $kind:expr, $accessor:ident) => {
        $ir.operations().iter().find_map(|operation| {
            (operation.opcode() == Some($kind))
                .then(|| operation.attributes().$accessor())
                .flatten()
        })
    };
}

pub fn assert_matmul_capture_ir(ir: &Graph) {
    assert!(
        ir.operations()
            .iter()
            .any(|operation| operation.opcode() == Some(Operation::MatMul))
    );

    assert!(ir.operations().iter().any(|operation| {
        operation.opcode() == Some(Operation::QuantizedMatMul) && operation.inputs().len() == 6
    }));

    let matmul_like_ops = ir
        .operations()
        .iter()
        .filter(|operation| {
            matches!(
                operation.opcode(),
                Some(Operation::MatMul | Operation::QuantizedMatMul)
            )
        })
        .count();
    assert_eq!(matmul_like_ops, 2);
}

pub fn assert_numeric_capture_ir(ir: &Graph) {
    let cast = find_attribute!(ir, Operation::Cast, cast_data_type);
    assert_eq!(cast, Some(DataType::I32));

    let quantization = find_attribute!(ir, Operation::Quantize, quantization);
    assert_eq!(
        quantization.map(|attributes| (attributes.axis, attributes.data_type)),
        Some((None, Some(DataType::I8))),
    );

    let dequantization = find_attribute!(ir, Operation::Dequantize, quantization);
    assert_eq!(
        dequantization.map(|attributes| (attributes.axis, attributes.data_type)),
        Some((None, None)),
    );

    let scale = find_attribute!(ir, Operation::Scale, scalar);
    assert_eq!(scale, Some(2.0.into()));

    let pad = find_attribute!(ir, Operation::Pad, pad);
    assert_eq!(
        pad.map(|attributes| (attributes.padding.clone(), attributes.value)),
        Some((vec![(1, 1)], -1.0)),
    );

    let clamp = find_attribute!(ir, Operation::Clamp, clamp);
    assert_eq!(
        clamp.map(|clamp| (clamp.min, clamp.max)),
        Some((1.25, 1.75))
    );
}
