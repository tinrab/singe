//! Copies, transposes, padding, slicing, tiling, and layout transforms.

pub fn transpose_2d(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = input[row * cols + col];
        }
    }
    output
}

pub fn pack_downsample_2d(input: &[f32], rows: usize, cols: usize, factor: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    let output_rows = rows / factor;
    let output_cols = cols * factor;
    for row in 0..output_rows {
        for packed in 0..output_cols {
            let offset = packed / cols;
            let feature = packed % cols;
            output[row * output_cols + packed] = input[(row * factor + offset) * cols + feature];
        }
    }
    output
}
