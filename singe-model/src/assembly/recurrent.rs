//! The rules that import `PyTorch`'s recurrent layers, `torch.nn.RNN`, `LSTM`, and `GRU`, into the recurrent stacks [`Schema::recurrent`](crate::schema::Schema::recurrent) registers.
//!
//! `PyTorch` names a layer's tensors `weight_ih_l{k}`, `weight_hh_l{k}`, `bias_ih_l{k}`, `bias_hh_l{k}`, and an LSTM's projection `weight_hr_l{k}`, each with `_reverse` for a bidirectional layer's reverse direction; Singe's are `{k}.input`, `{k}.hidden`, `{k}.projection`, and `{k}.reverse.*`.

use singe::{Error as SingeError, nn::TensorSource};

use crate::format::{Rules, join};

/// Imports the recurrent layer whose tensors `PyTorch` stores under `from`, such as `decoder.lstm`, into `to`.
#[must_use]
pub fn recurrent_rules(from: &str, to: &str) -> Rules {
    let to = to.to_owned();
    Rules::new().custom(&join(from, "*"), move |found| {
        let name = found.captures.last().copied().unwrap_or_default();
        let target = recurrent_target(name).ok_or_else(|| {
            SingeError::invalid_argument(format!("`{}` is no recurrent layer's tensor", found.name))
        })?;
        Ok(vec![(
            join(&to, &target),
            TensorSource::Stored(found.name.to_owned()),
        )])
    })
}

/// Returns Singe's name of a `PyTorch` recurrent tensor's last segment, such as `0.reverse.input.weight` for `weight_ih_l0_reverse`, or `None` for a name that is not one.
fn recurrent_target(name: &str) -> Option<String> {
    let (name, reverse) = name
        .strip_suffix("_reverse")
        .map_or((name, false), |name| (name, true));
    let (kind, layer) = name.rsplit_once("_l")?;
    let layer: u32 = layer.parse().ok()?;
    let (projection, parameter) = match kind {
        "weight_ih" => ("input", "weight"),
        "weight_hh" => ("hidden", "weight"),
        "bias_ih" => ("input", "bias"),
        "bias_hh" => ("hidden", "bias"),
        "weight_hr" => ("projection", "weight"),
        _ => return None,
    };
    let direction = if reverse { ".reverse" } else { "" };
    Some(format!("{layer}{direction}.{projection}.{parameter}"))
}
