use std::collections::{BTreeSet, HashMap, HashSet};

use crate::{
    error::{Error, Result},
    shape::Shape,
    utility::to_usize,
};

pub fn mode_dims(
    a_shape: &Shape,
    a_mode: &[i64],
    b_shape: &Shape,
    b_mode: &[i64],
    c_shape: &Shape,
    c_mode: &[i64],
) -> Result<HashMap<i64, i64>> {
    let mut dims = HashMap::new();
    insert_mode_dims(&mut dims, a_shape, a_mode)?;
    insert_mode_dims(&mut dims, b_shape, b_mode)?;
    insert_mode_dims(&mut dims, c_shape, c_mode)?;
    Ok(dims)
}

pub fn contraction_modes(a_mode: &[i64], b_mode: &[i64], d_mode: &[i64]) -> Vec<i64> {
    let d_set = d_mode.iter().copied().collect::<HashSet<_>>();
    a_mode
        .iter()
        .chain(b_mode.iter())
        .copied()
        .filter(|mode| !d_set.contains(mode))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub fn indices_for_modes(
    modes: &[i64],
    d_index_map: &HashMap<i64, i64>,
    reduction_map: &HashMap<i64, i64>,
) -> Result<Vec<i64>> {
    modes
        .iter()
        .map(|mode| {
            d_index_map
                .get(mode)
                .or_else(|| reduction_map.get(mode))
                .copied()
                .ok_or_else(|| Error::out_of_range(format!("cpu contraction input mode {mode}")))
        })
        .collect()
}

fn insert_mode_dims(dims: &mut HashMap<i64, i64>, shape: &Shape, modes: &[i64]) -> Result<()> {
    for (&mode, &dim) in modes.iter().zip(shape.dimensions()) {
        match dims.get(&mode) {
            Some(existing) if *existing != dim => {
                return Err(Error::length_mismatch(
                    format!("cpu contraction mode {mode} dimension"),
                    to_usize(
                        *existing,
                        &format!("cpu contraction mode {mode} existing dimension"),
                    )?,
                    to_usize(dim, &format!("cpu contraction mode {mode} input dimension"))?,
                ));
            }
            Some(_) => {}
            None => {
                dims.insert(mode, dim);
            }
        }
    }

    Ok(())
}
