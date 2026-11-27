// CPU kernels of the operations that sample maps at points, computed in `F64` as the definitions read.

use crate::{
    backend::cpu::utility::f64_values,
    device::Device,
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
    utility::to_usize,
};

fn indices(tensor: &Tensor) -> Result<Vec<usize>> {
    tensor
        .contiguous()?
        .to_vec::<i64>()?
        .into_iter()
        .map(|value| to_usize(value, "deformable levels"))
        .collect()
}

pub fn deformable_attention(inputs: &[Tensor]) -> Result<Tensor> {
    let [value, shapes, starts, locations, weights] = inputs else {
        return Err(Error::missing_input("cpu deformable attention operands"));
    };
    let &[batch_dimension, rows, heads_dimension, channel_dimension] = value.shape().dimensions()
    else {
        return Err(Error::invalid_argument("deformable attention value"));
    };
    let &[_, query_dimension, _, levels, points, _] = locations.shape().dimensions() else {
        return Err(Error::invalid_argument("deformable attention locations"));
    };
    let extent = |dimension: i64| to_usize(dimension, "deformable attention extent");
    let (batch, rows, heads, channels) = (
        extent(batch_dimension)?,
        extent(rows)?,
        extent(heads_dimension)?,
        extent(channel_dimension)?,
    );
    let (queries, levels, points) = (extent(query_dimension)?, extent(levels)?, extent(points)?);
    let (value_data, shapes, starts) = (f64_values(value)?, indices(shapes)?, indices(starts)?);
    let (locations_data, weights_data) = (f64_values(locations)?, f64_values(weights)?);
    let mut output = vec![0.0; batch * queries * heads * channels];
    for b in 0..batch {
        for q in 0..queries {
            for h in 0..heads {
                let out = ((b * queries + q) * heads + h) * channels;
                for l in 0..levels {
                    let (height, width, start) = (shapes[2 * l], shapes[2 * l + 1], starts[l]);
                    for p in 0..points {
                        let sample = (((b * queries + q) * heads + h) * levels + l) * points + p;
                        let x = locations_data[2 * sample] * width as f64 - 0.5;
                        let y = locations_data[2 * sample + 1] * height as f64 - 0.5;
                        let weight = weights_data[sample];
                        let (left, top) = (x.floor(), y.floor());
                        let (dx, dy) = (x - left, y - top);
                        for (row, column, share) in [
                            (top, left, (1.0 - dy) * (1.0 - dx)),
                            (top, left + 1.0, (1.0 - dy) * dx),
                            (top + 1.0, left, dy * (1.0 - dx)),
                            (top + 1.0, left + 1.0, dy * dx),
                        ] {
                            if row < 0.0
                                || column < 0.0
                                || row >= height as f64
                                || column >= width as f64
                            {
                                continue;
                            }

                            let place = start + row as usize * width + column as usize;
                            if place >= rows {
                                return Err(Error::out_of_range("deformable attention level"));
                            }

                            let base = ((b * rows + place) * heads + h) * channels;
                            for c in 0..channels {
                                output[out + c] += weight * share * value_data[base + c];
                            }
                        }
                    }
                }
            }
        }
    }

    let shape = Shape::create([
        batch_dimension,
        query_dimension,
        heads_dimension * channel_dimension,
    ])?;
    Tensor::from_vec(Device::Cpu, shape, output)?.cast(value.data_type())
}
