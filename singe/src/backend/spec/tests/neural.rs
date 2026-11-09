use crate::{
    backend::spec::{
        attention_problem, batch_norm_problem, conv2d_problem, normalization_problem,
        pool2d_problem, repeat_kv_problem, rotary_embedding_problem, validate_conv2d_bias,
    },
    error::Error,
    shape::Shape,
};

#[test]
fn test_attention_problem_describes_output_shape() {
    let query = Shape::create([2, 3, 4, 5]).unwrap();
    let key = Shape::create([2, 3, 6, 5]).unwrap();
    let value = Shape::create([2, 3, 6, 7]).unwrap();
    let problem = attention_problem(&query, &key, &value, "test attention").unwrap();

    assert_eq!(problem.batch, 2);
    assert_eq!(problem.heads, 3);
    assert_eq!(problem.query_len, 4);
    assert_eq!(problem.key_len, 6);
    assert_eq!(problem.head_dim, 5);
    assert_eq!(problem.value_dim, 7);
    assert_eq!(problem.output_shape.dimensions(), &[2, 3, 4, 7]);
}

#[test]
fn test_attention_problem_rejects_wrong_rank() {
    let query = Shape::create([2, 3, 4]).unwrap();
    let key = Shape::create([2, 3, 6, 5]).unwrap();
    let value = Shape::create([2, 3, 6, 7]).unwrap();

    assert!(matches!(
        attention_problem(&query, &key, &value, "test attention"),
        Err(Error::InvalidTensorRank {
            operation,
            name,
            expected_rank: 4,
            actual_rank: 3,
            ..
        }) if operation == "test attention" && name == "query"
    ));
}

#[test]
fn test_attention_problem_rejects_mismatched_sequence_length() {
    let query = Shape::create([2, 3, 4, 5]).unwrap();
    let key = Shape::create([2, 3, 6, 5]).unwrap();
    let value = Shape::create([2, 3, 7, 7]).unwrap();

    assert!(matches!(
        attention_problem(&query, &key, &value, "test attention"),
        Err(Error::LengthMismatch { ref name, expected, actual })
            if name == "test attention value length" && expected == 6 && actual == 7
    ));
}

#[test]
fn test_repeat_kv_problem_describes_output_shape() {
    let input = Shape::create([2, 3, 4, 5]).unwrap();
    let problem = repeat_kv_problem(&input, 2, "test repeat_kv").unwrap();

    assert_eq!(problem.batch, 2);
    assert_eq!(problem.heads, 3);
    assert_eq!(problem.output_heads, 6);
    assert_eq!(problem.sequence, 4);
    assert_eq!(problem.head_dim, 5);
    assert_eq!(problem.repeats, 2);
    assert_eq!(problem.output_shape.dimensions(), &[2, 6, 4, 5]);
}

#[test]
fn test_repeat_kv_problem_rejects_non_positive_repeats() {
    let input = Shape::create([2, 3, 4, 5]).unwrap();

    assert!(matches!(
        repeat_kv_problem(&input, 0, "test repeat_kv"),
        Err(Error::InvalidArgument { ref name }) if name == "test repeat_kv repeats"
    ));
}

#[test]
fn test_repeat_kv_problem_rejects_wrong_rank() {
    let input = Shape::create([2, 3, 4]).unwrap();

    assert!(matches!(
        repeat_kv_problem(&input, 2, "test repeat_kv"),
        Err(Error::InvalidTensorRank {
            operation,
            name,
            expected_rank: 4,
            actual_rank: 3,
            ..
        }) if operation == "test repeat_kv" && name == "input"
    ));
}

#[test]
fn test_rotary_embedding_problem_describes_shape_and_pairs() {
    let input = Shape::create([2, 3, 4, 6]).unwrap();
    let cos = Shape::create([4, 2]).unwrap();
    let sin = Shape::create([4, 2]).unwrap();
    let problem = rotary_embedding_problem(&input, &cos, &sin, "test rotary").unwrap();

    assert_eq!(problem.output_shape, input);
    assert_eq!(problem.batch, 2);
    assert_eq!(problem.heads, 3);
    assert_eq!(problem.sequence, 4);
    assert_eq!(problem.head_dim, 6);
    assert_eq!(problem.rotary_pairs, 2);
    assert_eq!(problem.rotary_dim, 4);
}

#[test]
fn test_rotary_embedding_problem_rejects_trig_shape_mismatch() {
    let input = Shape::create([2, 3, 4, 6]).unwrap();
    let cos = Shape::create([4, 2]).unwrap();
    let sin = Shape::create([4, 3]).unwrap();

    assert!(matches!(
        rotary_embedding_problem(&input, &cos, &sin, "test rotary"),
        Err(Error::ShapeMismatch { ref name, ref expected, ref actual })
            if name == "test rotary sin shape"
                && expected == &vec![4, 2]
                && actual == &vec![4, 3]
    ));
}

#[test]
fn test_rotary_embedding_problem_rejects_too_many_rotary_pairs() {
    let input = Shape::create([2, 3, 4, 3]).unwrap();
    let cos = Shape::create([4, 2]).unwrap();
    let sin = Shape::create([4, 2]).unwrap();

    assert!(matches!(
        rotary_embedding_problem(&input, &cos, &sin, "test rotary"),
        Err(Error::LengthMismatch { ref name, expected, actual })
            if name == "test rotary rotary dimension" && expected == 3 && actual == 4
    ));
}

#[test]
fn test_conv2d_problem_describes_output_shape() {
    let input = Shape::create([2, 4, 8, 8]).unwrap();
    let weight = Shape::create([6, 2, 3, 3]).unwrap();
    let problem = conv2d_problem(
        &input,
        &weight,
        (2, 2),
        (1, 1, 1, 1),
        (1, 1),
        2,
        "test conv2d",
    )
    .unwrap();
    assert_eq!(problem.output_shape.dimensions(), &[2, 6, 4, 4]);
    assert_eq!(problem.input_channels, 4);
    assert_eq!(problem.output_channels, 6);
    assert_eq!(problem.effective_kernel_height, 3);
    assert_eq!(problem.effective_kernel_width, 3);
    assert_eq!(problem.output_height, 4);
    assert_eq!(problem.output_width, 4);
    assert_eq!(problem.stride_height, 2);
    assert_eq!(problem.stride_width, 2);
    assert_eq!(problem.pad_height_start, 1);
    assert_eq!(problem.pad_height_end, 1);
    assert_eq!(problem.pad_width_start, 1);
    assert_eq!(problem.pad_width_end, 1);
    assert_eq!(problem.dilation_height, 1);
    assert_eq!(problem.dilation_width, 1);
    assert_eq!(problem.groups, 2);
}

#[test]
fn test_conv2d_bias_validation_accepts_matching_vector() {
    let input = Shape::create([2, 4, 8, 8]).unwrap();
    let weight = Shape::create([6, 2, 3, 3]).unwrap();
    let bias = Shape::create([6]).unwrap();
    let problem = conv2d_problem(
        &input,
        &weight,
        (1, 1),
        (0, 0, 0, 0),
        (1, 1),
        2,
        "test conv2d",
    )
    .unwrap();

    validate_conv2d_bias(&problem, Some(&bias), "test conv2d").unwrap();
}

#[test]
fn test_conv2d_bias_validation_rejects_wrong_channel_count() {
    let input = Shape::create([2, 4, 8, 8]).unwrap();
    let weight = Shape::create([6, 2, 3, 3]).unwrap();
    let bias = Shape::create([5]).unwrap();
    let problem = conv2d_problem(
        &input,
        &weight,
        (1, 1),
        (0, 0, 0, 0),
        (1, 1),
        2,
        "test conv2d",
    )
    .unwrap();

    assert!(matches!(
        validate_conv2d_bias(&problem, Some(&bias), "test conv2d"),
        Err(Error::LengthMismatch { name, expected: 6, actual: 5 })
            if name == "test conv2d bias channels"
    ));
}

#[test]
fn test_conv2d_problem_rejects_wrong_weight_channels() {
    let input = Shape::create([2, 4, 8, 8]).unwrap();
    let weight = Shape::create([6, 3, 3, 3]).unwrap();
    assert!(matches!(
        conv2d_problem(
            &input,
            &weight,
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            2,
            "test conv2d",
        ),
        Err(Error::LengthMismatch { name, expected: 2, actual: 3 })
            if name == "test conv2d weight channels"
    ));
}

#[test]
fn test_conv2d_problem_rejects_grouped_channel_mismatch() {
    let input = Shape::create([2, 5, 8, 8]).unwrap();
    let weight = Shape::create([6, 1, 3, 3]).unwrap();
    assert!(matches!(
        conv2d_problem(
            &input,
            &weight,
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            2,
            "test conv2d",
        ),
        Err(Error::ShapeMismatch { name, expected, actual })
            if name == "test conv2d grouped channels"
                && expected == vec![2]
                && actual == vec![5, 6]
    ));
}

#[test]
fn test_normalization_problem_describes_feature_and_batch_shape() {
    let input = Shape::create([2, 3, 4]).unwrap();
    let problem = normalization_problem(&input, &[3, 4], "test normalization").unwrap();

    assert_eq!(problem.output_shape, input);
    assert_eq!(problem.normalized_shape, vec![3, 4]);
    assert_eq!(problem.split_axis, 1);
    assert_eq!(problem.feature_size, 12);
    assert_eq!(problem.batch_count, 2);
}

#[test]
fn test_normalization_problem_allows_empty_batch_dimension() {
    let input = Shape::create([0, 4]).unwrap();
    let problem = normalization_problem(&input, &[4], "test normalization").unwrap();

    assert_eq!(problem.feature_size, 4);
    assert_eq!(problem.batch_count, 0);
}

#[test]
fn test_normalization_problem_rejects_empty_normalized_shape() {
    let input = Shape::create([2, 3]).unwrap();

    assert!(matches!(
        normalization_problem(&input, &[], "test normalization"),
        Err(Error::LengthMismatch { ref name, expected, actual })
            if name == "test normalization normalized rank" && expected == 2 && actual == 0
    ));
}

#[test]
fn test_normalization_problem_rejects_mismatched_suffix() {
    let input = Shape::create([2, 3, 4]).unwrap();

    assert!(matches!(
        normalization_problem(&input, &[2, 4], "test normalization"),
        Err(Error::ShapeMismatch { ref name, ref expected, ref actual })
            if name == "test normalization normalized shape"
                && expected == &vec![2, 4]
                && actual == &vec![3, 4]
    ));
}

#[test]
fn test_batch_norm_problem_describes_iteration_sizes() {
    let input = Shape::create([2, 3, 4, 5]).unwrap();
    let parameter = Shape::create([3]).unwrap();
    let problem = batch_norm_problem(
        &input,
        Some(&parameter),
        Some(&parameter),
        Some(&parameter),
        Some(&parameter),
        "test batch_norm",
    )
    .unwrap();

    assert_eq!(problem.output_shape, input);
    assert_eq!(problem.batch_size, 2);
    assert_eq!(problem.feature_count, 3);
    assert_eq!(problem.spatial_len, 20);
}

#[test]
fn test_batch_norm_problem_allows_empty_batch_dimension() {
    let input = Shape::create([0, 3, 4]).unwrap();
    let problem = batch_norm_problem(&input, None, None, None, None, "test batch_norm").unwrap();

    assert_eq!(problem.batch_size, 0);
    assert_eq!(problem.feature_count, 3);
    assert_eq!(problem.spatial_len, 4);
}

#[test]
fn test_batch_norm_problem_rejects_bad_parameter_count() {
    let input = Shape::create([2, 3, 4]).unwrap();
    let parameter = Shape::create([4]).unwrap();

    assert!(matches!(
        batch_norm_problem(&input, Some(&parameter), None, None, None, "test batch_norm"),
        Err(Error::LengthMismatch { ref name, expected, actual })
            if name == "test batch_norm weight elements" && expected == 3 && actual == 4
    ));
}

#[test]
fn test_pool2d_problem_describes_batched_output_shape_and_iteration_metadata() {
    let input = Shape::create([2, 3, 5, 5]).unwrap();
    let problem = pool2d_problem(
        &input,
        (2, 2),
        (2, 2),
        (0, 0, 0, 0),
        (1, 1),
        false,
        false,
        "test pool2d",
    )
    .unwrap();

    assert_eq!(problem.rank, 4);
    assert_eq!(problem.batch, 2);
    assert_eq!(problem.channels, 3);
    assert_eq!(problem.input_height, 5);
    assert_eq!(problem.input_width, 5);
    assert_eq!(problem.output_height, 2);
    assert_eq!(problem.output_width, 2);
    assert_eq!(problem.kernel_height, 2);
    assert_eq!(problem.kernel_width, 2);
    assert_eq!(problem.stride_height, 2);
    assert_eq!(problem.stride_width, 2);
    assert_eq!(problem.output_shape.dimensions(), &[2, 3, 2, 2]);
}

#[test]
fn test_pool2d_problem_preserves_unbatched_rank() {
    let input = Shape::create([3, 5, 5]).unwrap();
    let problem = pool2d_problem(
        &input,
        (3, 3),
        (1, 1),
        (1, 1, 1, 1),
        (1, 1),
        false,
        true,
        "test pool2d",
    )
    .unwrap();

    assert_eq!(problem.rank, 3);
    assert_eq!(problem.batch, 1);
    assert_eq!(problem.output_shape.dimensions(), &[3, 5, 5]);
}

#[test]
fn test_pool2d_problem_rejects_avg_pool_dilation() {
    let input = Shape::create([2, 3, 5, 5]).unwrap();

    assert!(matches!(
        pool2d_problem(
            &input,
            (2, 2),
            (1, 1),
            (0, 0, 0, 0),
            (2, 1),
            false,
            true,
            "test pool2d",
        ),
        Err(Error::InvalidArgument { ref name }) if name == "test pool2d avg pool dilation"
    ));
}
