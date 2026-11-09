use std::collections::HashMap;

use crate::{
    backend::cpu::contraction::{indices::flatten_indices, modes::indices_for_modes},
    error::{Error, Result},
};

// One operand of a contraction: its mode labels, strides, and values.
#[derive(Clone, Copy)]
pub struct ContractionOperand<'a> {
    pub modes: &'a [i64],
    pub strides: &'a [i64],
    pub data: &'a [f32],
}

// `alpha * sum(a * b) + beta * c`, with `d_modes` labelling the output's axes.
pub struct Contraction<'a> {
    pub a: ContractionOperand<'a>,
    pub b: ContractionOperand<'a>,
    pub c: ContractionOperand<'a>,
    pub d_modes: &'a [i64],
    pub reduction_modes: &'a [i64],
    pub output_indices: &'a [Vec<i64>],
    pub reduction_indices: &'a [Vec<i64>],
    pub alpha: f32,
    pub beta: f32,
}

pub fn execute_contraction(contraction: &Contraction<'_>, output: &mut [f32]) -> Result<()> {
    let Contraction {
        a,
        b,
        c,
        d_modes,
        reduction_modes,
        output_indices,
        reduction_indices,
        alpha,
        beta,
    } = *contraction;
    for (output_offset, d_indices) in output_indices.iter().enumerate() {
        let d_index_map = d_modes
            .iter()
            .copied()
            .zip(d_indices.iter().copied())
            .collect::<HashMap<_, _>>();

        let sum = contraction_sum(a, b, reduction_modes, reduction_indices, &d_index_map)?;
        let c_offset = c_offset(c.modes, c.strides, &d_index_map)?;
        output[output_offset] = beta.mul_add(c.data[c_offset], alpha * sum);
    }
    Ok(())
}

fn contraction_sum(
    a: ContractionOperand<'_>,
    b: ContractionOperand<'_>,
    reduction_modes: &[i64],
    reduction_indices: &[Vec<i64>],
    d_index_map: &HashMap<i64, i64>,
) -> Result<f32> {
    let mut sum = 0.0f32;
    for reduction_values in reduction_indices {
        let reduction_map = reduction_modes
            .iter()
            .copied()
            .zip(reduction_values.iter().copied())
            .collect::<HashMap<_, _>>();

        let a_indices = indices_for_modes(a.modes, d_index_map, &reduction_map)?;
        let b_indices = indices_for_modes(b.modes, d_index_map, &reduction_map)?;

        let a_offset = flatten_indices(&a_indices, a.strides)?;
        let b_offset = flatten_indices(&b_indices, b.strides)?;
        sum = a.data[a_offset].mul_add(b.data[b_offset], sum);
    }
    Ok(sum)
}

fn c_offset(c_mode: &[i64], c_strides: &[i64], d_index_map: &HashMap<i64, i64>) -> Result<usize> {
    let c_indices = c_mode
        .iter()
        .map(|mode| {
            d_index_map
                .get(mode)
                .copied()
                .ok_or_else(|| Error::out_of_range(format!("cpu contraction output mode {mode}")))
        })
        .collect::<Result<Vec<_>>>()?;
    flatten_indices(&c_indices, c_strides)
}
