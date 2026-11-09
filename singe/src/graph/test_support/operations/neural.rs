use super::*;

pub fn add_rotary_embedding_operation(
    graph: &mut Graph,
    input: ValueId,
    cos: ValueId,
    sin: ValueId,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::rotary_embedding(
        None,
        input,
        cos,
        sin,
        graph_operation_output(output),
    ))
}

pub fn add_scaled_dot_product_attention_operation(
    graph: &mut Graph,
    query: ValueId,
    key: ValueId,
    value: ValueId,
    scale: Option<f32>,
    causal: bool,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::scaled_dot_product_attention(
        None,
        query,
        key,
        value,
        scale,
        causal,
        graph_operation_output(output),
    ))
}

pub fn add_contraction_operation(
    graph: &mut Graph,
    a: ValueId,
    a_mode: Vec<i64>,
    b: ValueId,
    b_mode: Vec<i64>,
    c: ValueId,
    c_mode: Vec<i64>,
    d_mode: Vec<i64>,
    alpha: f32,
    beta: f32,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::contraction(
        None,
        a,
        a_mode,
        b,
        b_mode,
        c,
        c_mode,
        d_mode,
        alpha,
        beta,
        graph_operation_output(output),
    ))
}

pub fn add_layer_normalization_operation(
    graph: &mut Graph,
    operation: Operation,
    input: ValueId,
    normalized_shape: Vec<i64>,
    weight: Option<ValueId>,
    bias: Option<ValueId>,
    eps: f32,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::layer_normalization(
        None,
        operation,
        input,
        normalized_shape,
        weight,
        bias,
        eps,
        graph_operation_output(output),
    ))
}

pub fn add_batch_normalization_operation(
    graph: &mut Graph,
    input: ValueId,
    weight: Option<ValueId>,
    bias: Option<ValueId>,
    running_mean: Option<ValueId>,
    running_var: Option<ValueId>,
    training: bool,
    momentum: f32,
    eps: f32,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::batch_normalization(
        None,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        graph_operation_output(output),
    ))
}

pub fn add_conv2d_operation(
    graph: &mut Graph,
    input: ValueId,
    weight: ValueId,
    bias: Option<ValueId>,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::conv2d(
        None,
        input,
        weight,
        bias,
        (1, 1),
        (0, 0, 0, 0),
        (1, 1),
        1,
        graph_operation_output(output),
    ))
}

pub fn add_pool2d_operation(
    graph: &mut Graph,
    operation: Operation,
    input: ValueId,
    kernel_size: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    ceil_mode: bool,
    count_include_pad: bool,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::pool2d(
        None,
        operation,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        count_include_pad,
        graph_operation_output(output),
    ))
}
