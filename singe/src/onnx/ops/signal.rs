// ONNX's signal operations: the windows and the mel weight matrix as constants, since their inputs must be, and `DFT` and `STFT` as matmuls with each frequency's cosines and sines, which run on every backend and join a compiled graph.
// A transform's length must be known when the model loads, as `dft_length` and `frame_length` are when they are constants.

use std::f64::consts::PI;

use singe_onnx::{Node, TensorDataType as OnnxTensorDataType};

use crate::{
    data_type::DataType,
    device::Device,
    error::Result as SingeResult,
    graph::{capture, ir::ValueId},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{constant::register_constant_tensor, expect_input_range, register_node_outputs},
        tensor::data_type as onnx_data_type,
        utility::node_label,
    },
    shape::Shape,
    tensor::Tensor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxWindow {
    Hann,
    Hamming,
    Blackman,
}

fn refuse(importer: &Importer, node: &Node, reason: impl Into<String>) -> Error {
    Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        reason,
    )
}

// The value of the scalar input `index`, which must be a constant.
fn static_scalar(importer: &Importer, node: &Node, index: usize) -> Result<Option<f64>> {
    if node.inputs.get(index).is_none_or(String::is_empty) {
        return Ok(None);
    }

    let Some((_, tensor)) = importer.optional_static_input_tensor(node, index)? else {
        return Err(refuse(
            importer,
            node,
            format!("input {index} other than a constant"),
        ));
    };
    match tensor.cast(DataType::F64)?.to_vec::<f64>()?.as_slice() {
        [value] => Ok(Some(*value)),
        _ => Err(refuse(
            importer,
            node,
            format!("input {index} other than a scalar"),
        )),
    }
}

fn required_scalar(importer: &Importer, node: &Node, index: usize) -> Result<f64> {
    static_scalar(importer, node, index)?
        .ok_or_else(|| refuse(importer, node, format!("no input {index}")))
}

fn count(importer: &Importer, node: &Node, value: f64) -> Result<i64> {
    if value.fract() != 0.0 || value < 0.0 || value > i64::MAX as f64 {
        return Err(refuse(importer, node, "a count other than a whole number"));
    }

    Ok(value as i64)
}

// The type `output_datatype` names, float by default.
fn output_type(node: &Node, attrs: &Attributes<'_>) -> Result<DataType> {
    let invalid = || {
        Error::invalid_attribute(
            node_label(node),
            "output_datatype",
            "unknown tensor data type",
        )
    };
    let code = i32::try_from(attrs.int("output_datatype", 1)?).map_err(|_| invalid())?;
    let data_type = OnnxTensorDataType::try_from(code).map_err(|_| invalid())?;
    onnx_data_type(&node_label(node), data_type)
}

fn constant(
    importer: &mut Importer,
    node: &Node,
    shape: Shape,
    values: Vec<f64>,
    data_type: DataType,
) -> Result<()> {
    let tensor = Tensor::from_vec(importer.options().device, shape, values)?.cast(data_type)?;
    register_constant_tensor(importer, node, &tensor)
}

pub fn lower_window(importer: &mut Importer, node: &Node, window: OnnxWindow) -> Result<()> {
    expect_input_range(importer, node, 1, 1)?;
    let attrs = Attributes::create(node)?;
    let size = count(importer, node, required_scalar(importer, node, 0)?)?;
    let periodic = attrs.int("periodic", 1)? != 0;
    let span = if periodic { size } else { size - 1 } as f64;
    let values = (0..size)
        .map(|index| {
            let angle = 2.0 * PI * index as f64 / span;
            match window {
                OnnxWindow::Hann => (PI * index as f64 / span).sin().powi(2),
                // ONNX's Hamming coefficients are 25/46 and 21/46, where PyTorch's are 0.54 and 0.46.
                OnnxWindow::Hamming => 25.0 / 46.0 - 21.0 / 46.0 * angle.cos(),
                OnnxWindow::Blackman => 0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos(),
            }
        })
        .collect();
    let data_type = output_type(node, &attrs)?;
    constant(importer, node, Shape::create([size])?, values, data_type)
}

// The triangular filters of `num_mel_bins` bands between the two edges, `[dft_length / 2 + 1, num_mel_bins]`, as ONNX's reference computes them: band edges on HTK's mel scale, `(num_mel_bins + 2)` steps apart, each rounded down to a frequency bin.
pub fn lower_mel_weight_matrix(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 5, 5)?;
    let attrs = Attributes::create(node)?;
    let mels = count(importer, node, required_scalar(importer, node, 0)?)?;
    let length = count(importer, node, required_scalar(importer, node, 1)?)?;
    let sample_rate = required_scalar(importer, node, 2)?;
    let lower = required_scalar(importer, node, 3)?;
    let upper = required_scalar(importer, node, 4)?;
    if mels == 0 || sample_rate <= 0.0 {
        return Err(refuse(importer, node, "no mel bins or no sample rate"));
    }

    let bins = length / 2 + 1;
    let mel = |hertz: f64| 2595.0 * (1.0 + hertz / 700.0).log10();
    let (low, high) = (mel(lower), mel(upper));
    let step = (high - low) / (mels + 2) as f64;
    let edges = (0..mels + 2)
        .map(|index| {
            let hertz = 700.0 * (10_f64.powf((index as f64 * step + low) / 2595.0) - 1.0);
            ((length + 1) as f64 * hertz / sample_rate).floor() as i64
        })
        .collect::<Vec<_>>();
    let columns = usize::try_from(mels).map_err(|_| refuse(importer, node, "too many mel bins"))?;
    let mut values = vec![0.0; usize::try_from(bins).unwrap_or(0) * columns];
    // A filter row outside the transform's bins, which the onnx reference fails on, refuses the node after the loop.
    let mut outside = false;
    let mut set = |row: i64, column: usize, value: f64| match usize::try_from(row) {
        Ok(row) if row < values.len() / columns => values[row * columns + column] = value,
        _ => outside = true,
    };
    for (column, edge) in edges.windows(3).enumerate() {
        let &[left, center, right] = edge else {
            continue;
        };
        if center == left {
            set(center, column, 1.0);
        } else {
            for bin in left..=center {
                set(bin, column, (bin - left) as f64 / (center - left) as f64);
            }
        }

        for bin in center..right {
            set(bin, column, (right - bin) as f64 / (right - center) as f64);
        }
    }

    if outside {
        return Err(refuse(
            importer,
            node,
            "mel edges outside the transform's bins",
        ));
    }

    let data_type = output_type(node, &attrs)?;
    constant(
        importer,
        node,
        Shape::create([bins, mels])?,
        values,
        data_type,
    )
}

// The cosines and sines of a transform of `length` over `rows` samples to `columns` frequencies, `[rows, columns]` each, in `data_type`.
fn bases(
    device: Device,
    rows: i64,
    columns: i64,
    length: i64,
    data_type: DataType,
) -> SingeResult<(Tensor, Tensor)> {
    let angles = (0..rows)
        .flat_map(|row| {
            (0..columns)
                .map(move |column| 2.0 * PI * ((row * column) % length) as f64 / length as f64)
        })
        .collect::<Vec<_>>();
    let matrix = |values: Vec<f64>| {
        capture::literal(device, |device| {
            Tensor::from_vec(device, Shape::create([rows, columns])?, values)
        })?
        .cast(data_type)
    };
    Ok((
        matrix(angles.iter().map(|angle| angle.cos()).collect())?,
        matrix(angles.iter().map(|angle| angle.sin()).collect())?,
    ))
}

#[derive(Debug, Clone, Copy)]
struct Transform {
    length: i64,
    inverse: bool,
    onesided: bool,
}

impl Transform {
    const fn frequencies(&self) -> i64 {
        if self.onesided {
            self.length / 2 + 1
        } else {
            self.length
        }
    }

    // Transforms `real` and `imaginary` along their last axis, `[…, L]`, to `[…, K, 2]`.
    fn apply(&self, real: &Tensor, imaginary: Option<&Tensor>) -> SingeResult<Tensor> {
        let dimensions = real.shape().dimensions().to_vec();
        let samples = dimensions.last().copied().unwrap_or(0);
        let rows = samples.min(self.length);
        let flat = |tensor: &Tensor| -> SingeResult<Tensor> {
            tensor
                .narrow(-1, 0, rows)?
                .contiguous()?
                .reshape([-1, rows])
        };
        let (cosines, sines) = bases(
            real.device(),
            rows,
            self.frequencies(),
            self.length,
            real.data_type(),
        )?;
        let real = flat(real)?;
        let imaginary = imaginary.map(flat).transpose()?;
        let real_cosines = real.matmul(&cosines)?;
        let real_sines = real.matmul(&sines)?;
        // `Σ x e^{-iθ}` forward and `Σ x e^{+iθ} / N` inverse, of `x = a + ib`.
        let (out_real, out_imaginary) = match (&imaginary, self.inverse) {
            (None, false) => (real_cosines, real_sines.mul_scalar(-1.0)?),
            (None, true) => (real_cosines, real_sines),
            (Some(imaginary), false) => (
                real_cosines.add(&imaginary.matmul(&sines)?)?,
                imaginary.matmul(&cosines)?.sub(&real_sines)?,
            ),
            (Some(imaginary), true) => (
                real_cosines.sub(&imaginary.matmul(&sines)?)?,
                real_sines.add(&imaginary.matmul(&cosines)?)?,
            ),
        };
        let (out_real, out_imaginary) = if self.inverse {
            let scale = 1.0 / self.length as f64;
            (
                out_real.mul_scalar(scale)?,
                out_imaginary.mul_scalar(scale)?,
            )
        } else {
            (out_real, out_imaginary)
        };

        let mut shape = dimensions;
        if let Some(last) = shape.last_mut() {
            *last = self.frequencies();
        }

        shape.push(1);
        Tensor::concat(
            &[
                out_real.reshape(shape.clone())?,
                out_imaginary.reshape(shape)?,
            ],
            -1,
        )
    }
}

// The real and imaginary parts of `[…, 1]` or `[…, 2]`, without the last axis.
fn components(signal: &Tensor) -> SingeResult<(Tensor, Option<Tensor>)> {
    let parts = signal.shape().dimensions().last().copied().unwrap_or(1);
    let last = signal.shape().rank() - 1;
    let imaginary = (parts == 2).then(|| signal.select(last, 1)).transpose()?;
    Ok((signal.select(last, 0)?, imaginary))
}

pub fn lower_dft(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 1, 3)?;
    let attrs = Attributes::create(node)?;
    let version = importer.opset_version(node.domain.as_deref());
    let input = importer.required_input(node, 0)?;
    let rank = input.shape.rank();
    let axis = if version >= 20 {
        static_scalar(importer, node, 2)?
            .map_or(Ok(-2), |axis| count_signed(importer, node, axis))?
    } else {
        attrs.int("axis", 1)?
    };
    let axis = if axis < 0 { axis + rank } else { axis };
    let parts = input.shape.dimensions().last().copied().unwrap_or(0);
    if !(0..rank - 1).contains(&axis) || !matches!(parts, 1 | 2) {
        return Err(refuse(
            importer,
            node,
            "an axis or a last dimension other than real or complex parts",
        ));
    }

    let inverse = attrs.int("inverse", 0)? != 0;
    let onesided = attrs.int("onesided", 0)? != 0;
    if onesided && (inverse || parts == 2) {
        return Err(refuse(
            importer,
            node,
            "a one-sided inverse or a one-sided transform of complex input",
        ));
    }

    let samples = input.shape.dimensions()[usize::try_from(axis).unwrap_or(0)];
    let length = match static_scalar(importer, node, 1)? {
        Some(length) => count(importer, node, length)?,
        None => samples,
    };
    if length == 0 {
        return Err(refuse(importer, node, "an empty transform"));
    }

    let transform = Transform {
        length,
        inverse,
        onesided,
    };
    if !importer.have_fixed_shapes(&[input.node])? {
        return Err(refuse(importer, node, "an input with run-time axes"));
    }

    let device = importer.options().device;
    let [id] = importer
        .graph()
        .append_traced(&[input.node], device, |inputs| {
            let (real, imaginary) = components(inputs[0])?;
            let last = rank - 2;
            let real = real.transpose(axis, last)?;
            let imaginary = imaginary
                .map(|part| part.transpose(axis, last))
                .transpose()?;
            Ok(vec![
                transform
                    .apply(&real, imaginary.as_ref())?
                    .transpose(axis, last)?,
            ])
        })?[..]
    else {
        return Err(refuse(importer, node, "a transform without one result"));
    };
    register_traced(importer, node, id)
}

fn count_signed(importer: &Importer, node: &Node, value: f64) -> Result<i64> {
    if value.fract() != 0.0 || value.abs() > i64::MAX as f64 {
        return Err(refuse(importer, node, "an axis other than a whole number"));
    }

    Ok(value as i64)
}

fn register_traced(importer: &mut Importer, node: &Node, id: ValueId) -> Result<()> {
    let traced = importer.graph().required_node(id)?;
    let value = ImportedValue::new(id, traced.output_data_type(), traced.output_shape().clone());
    register_node_outputs(importer, node, vec![value])
}

// `[B, N, 1 or 2]` to `[B, frames, K, 2]`: frames of `frame_length` every `frame_step` samples, each weighted by `window` and transformed.
pub fn lower_stft(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 2, 4)?;
    let attrs = Attributes::create(node)?;
    let signal = importer.required_input(node, 0)?;
    let &[_, samples, parts] = signal.shape.dimensions() else {
        return Err(refuse(
            importer,
            node,
            "a signal other than [batch, samples, parts]",
        ));
    };
    if !matches!(parts, 1 | 2) {
        return Err(refuse(
            importer,
            node,
            "a last dimension other than real or complex parts",
        ));
    }

    let step = count(importer, node, required_scalar(importer, node, 1)?)?;
    let window = importer.optional_input(node, 2)?;
    let window_length = window
        .as_ref()
        .and_then(|window| window.shape.dimensions().first().copied());
    let length = match static_scalar(importer, node, 3)? {
        Some(length) => count(importer, node, length)?,
        None => window_length.unwrap_or(samples),
    };
    if step == 0
        || length == 0
        || length > samples
        || window_length.is_some_and(|window| window != length)
    {
        return Err(refuse(
            importer,
            node,
            "a frame step, frame length, or window that frames nothing",
        ));
    }

    let transform = Transform {
        length,
        inverse: false,
        onesided: attrs.int("onesided", 1)? != 0,
    };
    let mut operands = vec![signal.node];
    operands.extend(window.as_ref().map(|window| window.node));
    if !importer.have_fixed_shapes(&operands)? {
        return Err(refuse(importer, node, "inputs with run-time axes"));
    }

    let device = importer.options().device;
    let [id] = importer
        .graph()
        .append_traced(&operands, device, |inputs| {
            let (real, imaginary) = components(inputs[0])?;
            let frame = |part: &Tensor| -> SingeResult<Tensor> {
                let frames = part.unfold(1, length, step)?;
                match inputs.get(1) {
                    Some(window) => frames.mul(&window.cast(frames.data_type())?),
                    None => Ok(frames),
                }
            };
            let real = frame(&real)?;
            let imaginary = imaginary.as_ref().map(frame).transpose()?;
            Ok(vec![transform.apply(&real, imaginary.as_ref())?])
        })?[..]
    else {
        return Err(refuse(importer, node, "a transform without one result"));
    };
    register_traced(importer, node, id)
}
