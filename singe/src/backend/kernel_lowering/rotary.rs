use singe_codegen::{
    ir::{AxisKind, BinaryOp, Size},
    schedule::element_schedule,
};

use crate::{
    backend::{
        kernel_lowering::{
            domain::{DomainKernel, LoweredKernel},
            element_type,
        },
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    utility::to_u64,
    value::ValueSpec,
};

/// Lowers the rotary embedding of `[B, H, S, D]` `input` by `[S, D / 2]` `cos` and `sin`, rotating each adjacent pair of a head, as one element-wise kernel over the pairs.
///
/// Each point reads its pair's two values, the second through an access one stride past the first, and writes both rotated values.
///
/// # Errors
///
/// Returns an error if the shapes do not match, the rotation leaves part of each head unrotated, or a type is not floating-point.
pub fn rotary_embedding(
    data_type: DataType,
    input: &Layout,
    cos: (DataType, &Layout),
    sin: (DataType, &Layout),
) -> Result<LoweredKernel> {
    let problem = spec::rotary_embedding_problem(
        input.shape(),
        cos.1.shape(),
        sin.1.shape(),
        "cuda generated rotary_embedding",
    )?;
    if problem.rotary_dim != problem.head_dim
        || !data_type.is_float()
        || !cos.0.is_float()
        || !sin.0.is_float()
    {
        return Err(Error::unsupported_operation(
            "cuda generated rotary_embedding of part of a head",
        ));
    }

    let dimensions = [
        problem.batch,
        problem.heads,
        problem.sequence,
        problem.rotary_pairs,
    ];
    let strides = input.strides();
    let paired = |strides: &[i64]| vec![strides[0], strides[1], strides[2], strides[3] * 2];
    // The tables vary along the sequence and the pairs, and broadcast over the batch and the heads.
    let table = |layout: &Layout| vec![0, 0, layout.strides()[0], layout.strides()[1]];
    let output = Layout::contiguous(problem.output_shape.dimensions())?;
    let columns = [
        paired(strides),
        table(cos.1),
        table(sin.1),
        paired(output.strides()),
    ];
    let mut builder = DomainKernel::create(
        "singe_rotary_embedding",
        &dimensions,
        &[AxisKind::Parallel; 4],
        &columns,
    )?;
    let element = element_type(data_type)?;
    let compute = element.compute();
    // The pair's second value is one input stride, or one output element, past its first.
    let first = builder.input(element, 0);
    let mut second = first.clone();
    second.base = Some(
        builder
            .kernel
            .add_size(Size::Static(to_u64(strides[3], "rotary input stride")?)),
    );
    let first = builder.kernel.load(first);
    let first = builder.convert(first, compute);
    let second = builder.kernel.load(second);
    let second = builder.convert(second, compute);
    let table_value = |builder: &mut DomainKernel, (data_type, _): (DataType, &Layout), column| {
        let access = builder.input(element_type(data_type)?, column);
        let loaded = builder.kernel.load(access);
        Ok::<_, Error>(builder.convert(loaded, compute))
    };
    let cosine = table_value(&mut builder, cos, 1)?;
    let sine = table_value(&mut builder, sin, 2)?;
    let rotated_first = {
        let a = builder.kernel.binary(BinaryOp::Mul, first, cosine);
        let b = builder.kernel.binary(BinaryOp::Mul, second, sine);
        builder.kernel.binary(BinaryOp::Sub, a, b)
    };
    let rotated_second = {
        let a = builder.kernel.binary(BinaryOp::Mul, first, sine);
        let b = builder.kernel.binary(BinaryOp::Mul, second, cosine);
        builder.kernel.binary(BinaryOp::Add, a, b)
    };
    let stored_first = builder.output(element, 3);
    let mut stored_second = stored_first.clone();
    stored_second.base = Some(builder.kernel.add_size(Size::Static(1)));
    builder.kernel.store(stored_first, rotated_first);
    builder.kernel.store(stored_second, rotated_second);
    let schedule = element_schedule(&builder.kernel, 128);
    Ok(builder.finish(ValueSpec::new(data_type, problem.output_shape), schedule))
}
