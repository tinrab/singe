use std::hash::{Hash, Hasher};

use crate::{device::Device, graph::symbolic::SymbolicDimension, layout::Layout, shape::Shape};

pub fn hash_shape(shape: &Shape, hasher: &mut impl Hasher) {
    shape.dimensions().hash(hasher);
}

pub fn hash_shape_with_symbols(
    shape: &Shape,
    symbols: Option<&[Option<SymbolicDimension>]>,
    hasher: &mut impl Hasher,
) {
    shape.dimensions().len().hash(hasher);
    for (axis, dimension) in shape.dimensions().iter().enumerate() {
        if let Some(symbol) = symbols
            .and_then(|symbols| symbols.get(axis))
            .and_then(Option::as_ref)
        {
            "symbolic".hash(hasher);
            symbol.name().hash(hasher);
            symbol.min().hash(hasher);
            symbol.max().hash(hasher);
        } else {
            "static".hash(hasher);
            dimension.hash(hasher);
        }
    }
}

fn hash_strides_with_symbols(
    layout: &Layout,
    symbols: Option<&[Option<SymbolicDimension>]>,
    hasher: &mut impl Hasher,
) {
    let Some(symbols) = symbols else {
        layout.strides().hash(hasher);
        return;
    };
    if !layout.is_contiguous() {
        layout.strides().hash(hasher);
        return;
    }

    layout.strides().len().hash(hasher);
    for axis in 0..layout.strides().len() {
        if symbols.iter().skip(axis + 1).any(Option::is_some) {
            "contiguous_symbolic_stride".hash(hasher);
            for (dimension, symbol) in layout
                .dimensions()
                .iter()
                .skip(axis + 1)
                .zip(symbols.iter().skip(axis + 1))
            {
                if let Some(symbol) = symbol {
                    "symbolic".hash(hasher);
                    symbol.name().hash(hasher);
                    symbol.min().hash(hasher);
                    symbol.max().hash(hasher);
                } else {
                    "static".hash(hasher);
                    dimension.hash(hasher);
                }
            }
        } else {
            "static_stride".hash(hasher);
            layout.strides()[axis].hash(hasher);
        }
    }
}

pub fn hash_layout_with_symbols(
    layout: &Layout,
    symbols: Option<&[Option<SymbolicDimension>]>,
    hasher: &mut impl Hasher,
) {
    hash_shape_with_symbols(layout.shape(), symbols, hasher);
    hash_strides_with_symbols(layout, symbols, hasher);
    layout.storage_offset().hash(hasher);
}

pub fn hash_device(device: Device, hasher: &mut impl Hasher) {
    device.hash(hasher);
}
