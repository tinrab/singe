use std::{f32::consts::LN_2 as F32_LN_2, f64::consts::LN_2};

use num_traits::NumCast;

use crate::{
    backend::{
        cpu::utility::{NativeFloat, cpu_device_mismatch, get_f32_data, logical_values},
        registry::ScaledDotProductAttentionAttributes,
        spec,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    parallel::{self, GRAIN_SIZE},
    shape::Shape,
    tensor::Tensor,
    utility::to_usize,
};

pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    sinks: Option<&Tensor>,
    key_lengths: Option<&Tensor>,
    attention: ScaledDotProductAttentionAttributes,
) -> Result<Tensor> {
    if query.device() != Device::Cpu || key.device() != Device::Cpu || value.device() != Device::Cpu
    {
        let actual = [query, key, value]
            .into_iter()
            .map(Tensor::device)
            .find(|device| *device != Device::Cpu)
            .unwrap_or(Device::Cpu);
        return Err(cpu_device_mismatch(
            "cpu scaled_dot_product_attention input",
            actual,
        ));
    }

    // Grouped key and value heads repeat for the query heads that share them.
    let (query_heads, key_heads) = (
        query.shape().dimensions().get(1),
        key.shape().dimensions().get(1),
    );
    if let (Some(&query_heads), Some(&key_heads)) = (query_heads, key_heads)
        && key_heads > 0
        && query_heads != key_heads
        && query_heads % key_heads == 0
    {
        let groups = query_heads / key_heads;
        return scaled_dot_product_attention(
            query,
            &key.repeat_kv(groups)?,
            &value.repeat_kv(groups)?,
            mask,
            sinks,
            key_lengths,
            attention,
        );
    }
    match query.data_type() {
        DataType::F32 => attend::<f32>(query, key, value, [mask, sinks, key_lengths], attention),
        DataType::F64 => attend::<f64>(query, key, value, [mask, sinks, key_lengths], attention),
        _ => Err(Error::unsupported_data_type(
            "cpu scaled_dot_product_attention",
        )),
    }
}

// `extras` are the mask, the sinks, and the key lengths, each if the attention has it.
fn attend<T: NativeFloat>(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    [mask, sinks, key_lengths]: [Option<&Tensor>; 3],
    attention: ScaledDotProductAttentionAttributes,
) -> Result<Tensor> {
    let (batch, heads, query_len, key_len, head_dim, value_dim) =
        scaled_dot_product_attention_shape(query.shape(), key.shape(), value.shape())?;
    let scale = attention
        .scale
        .map_or_else(
            || <T as NumCast>::from(head_dim).map(|head_dim| head_dim.sqrt().recip()),
            <T as NumCast>::from,
        )
        .filter(|scale| scale.is_finite())
        .ok_or_else(|| {
            Error::unsupported_operation("cpu scaled_dot_product_attention scale must be finite")
        })?;

    let output_shape = Shape::create([batch, heads, query_len, value_dim])?;
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let query_data = logical_values::<T>(&query)?;
    let key_data = logical_values::<T>(&key)?;
    let value_data = logical_values::<T>(&value)?;
    // The mask, broadcast to the scores' `[B, H, L, S]`, adds to each score.
    let mask = mask
        .map(|mask| {
            mask.expand(&Shape::create([batch, heads, query_len, key_len])?)?
                .contiguous()
        })
        .transpose()?;
    let mask_data = mask.as_ref().map(logical_values::<T>).transpose()?;
    let sinks = sinks.map(Tensor::contiguous).transpose()?;
    let sinks_data = sinks.as_ref().map(logical_values::<T>).transpose()?;
    let lengths = key_lengths
        .map(|lengths| lengths.to_device(Device::Cpu)?.to_vec::<i32>())
        .transpose()?;
    let output_data = execute_scaled_dot_product_attention(
        &query_data,
        &key_data,
        &value_data,
        mask_data.as_deref(),
        sinks_data.as_deref(),
        lengths.as_deref(),
        (batch, heads, query_len, key_len, head_dim, value_dim),
        output_shape.element_count(),
        scale,
        attention,
    )?;
    drop(sinks_data);
    drop(mask_data);
    drop(query_data);

    Tensor::from_vec(Device::Cpu, output_shape, output_data)
}

pub fn rotary_embedding(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if input.device() != Device::Cpu || cos.device() != Device::Cpu || sin.device() != Device::Cpu {
        let actual = [input, cos, sin]
            .into_iter()
            .map(Tensor::device)
            .find(|device| *device != Device::Cpu)
            .unwrap_or(Device::Cpu);
        return Err(cpu_device_mismatch("cpu rotary_embedding input", actual));
    }

    for tensor in [input, cos, sin] {
        if tensor.data_type() != DataType::F32 {
            return Err(Error::unsupported_data_type("cpu rotary_embedding"));
        }
    }

    let problem = spec::rotary_embedding_problem(
        input.shape(),
        cos.shape(),
        sin.shape(),
        "cpu rotary_embedding",
    )?;
    let output_shape = problem.output_shape.clone();

    let input = input.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let input_data = get_f32_data(&input)?;
    let cos_data = get_f32_data(&cos)?;
    let sin_data = get_f32_data(&sin)?;
    let mut output_data = input_data.to_vec();
    if input_data.is_empty() || problem.rotary_pairs == 0 {
        return Tensor::from_vec(Device::Cpu, output_shape, output_data);
    }

    let sequence = to_usize(problem.sequence, "cpu rotary_embedding sequence")?;
    let head_dim = to_usize(problem.head_dim, "cpu rotary_embedding head dimension")?;
    let rotary_pairs = to_usize(problem.rotary_pairs, "cpu rotary_embedding rotary pairs")?;
    // Each token row of `head_dim` values rotates its leading pairs by its position's angles.
    parallel::for_each_row_mut(&mut output_data, head_dim, GRAIN_SIZE, |row, output| {
        let token = &input_data[row * head_dim..(row + 1) * head_dim];
        let position = row % sequence * rotary_pairs;
        for pair in 0..rotary_pairs {
            let cos_value = cos_data[position + pair];
            let sin_value = sin_data[position + pair];
            let (even, odd) = (token[2 * pair], token[2 * pair + 1]);
            output[2 * pair] = f32::mul_add(odd, -sin_value, even * cos_value);
            output[2 * pair + 1] = f32::mul_add(even, sin_value, odd * cos_value);
        }
        Ok(())
    })?;
    drop(sin_data);
    drop(cos_data);
    drop(input_data);

    Tensor::from_vec(Device::Cpu, output_shape, output_data)
}

/// Computes attention one query row at a time, splitting rows across threads; each row is computed by one thread in a fixed order.
pub fn execute_scaled_dot_product_attention<T: NativeFloat>(
    query_data: &[T],
    key_data: &[T],
    value_data: &[T],
    mask_data: Option<&[T]>,
    sinks_data: Option<&[T]>,
    key_lengths: Option<&[i32]>,
    dimensions: (i64, i64, i64, i64, i64, i64),
    output_element_count: usize,
    scale: T,
    attention: ScaledDotProductAttentionAttributes,
) -> Result<Vec<T>> {
    let softcap = attention.softcap.and_then(<T as NumCast>::from);
    let window = attention
        .window
        .map(|window| to_usize(window, "cpu scaled_dot_product_attention window"))
        .transpose()?;
    let (_, heads, query_len, key_len, head_dim, value_dim) = dimensions;
    let heads = to_usize(heads, "cpu scaled_dot_product_attention heads")?;
    let query_len = to_usize(query_len, "cpu scaled_dot_product_attention query length")?;
    let key_len = to_usize(key_len, "cpu scaled_dot_product_attention key length")?;
    let head_dim = to_usize(head_dim, "cpu scaled_dot_product_attention head dimension")?;
    let value_dim = to_usize(
        value_dim,
        "cpu scaled_dot_product_attention value dimension",
    )?;
    let mut output_data = vec![T::zero(); output_element_count];
    let work = (key_len * (head_dim + value_dim)).max(1);
    parallel::for_each_row_mut(
        &mut output_data,
        value_dim,
        (GRAIN_SIZE / work).max(1) * value_dim.max(1),
        |row, output| {
            let batch_head = row / query_len.max(1);
            let query_index = row % query_len.max(1);
            let query = &query_data[row * head_dim..(row + 1) * head_dim];
            let keys =
                &key_data[batch_head * key_len * head_dim..(batch_head + 1) * key_len * head_dim];
            let values = &value_data
                [batch_head * key_len * value_dim..(batch_head + 1) * key_len * value_dim];
            // A causal mask hides the keys after the query, and a window those `window` or more positions before it.
            let end = if attention.causal {
                key_len.min(query_index.saturating_add(1))
            } else {
                key_len
            };
            // A batch's count of keys hides those past it, whatever else would show them.
            let end = key_lengths.map_or(end, |lengths| {
                let length = lengths[batch_head / heads.max(1)];
                end.min(usize::try_from(length.max(0)).unwrap_or(0))
            });
            let start = window.map_or(0, |window| (query_index + 1).saturating_sub(window));
            let visible = start..end.max(start);
            let score = |key_index: usize| {
                let score = query
                    .iter()
                    .zip(&keys[key_index * head_dim..(key_index + 1) * head_dim])
                    .fold(T::zero(), |score, (query, key)| query.mul_add(*key, score))
                    * scale;
                let score = softcap.map_or(score, |cap| (score / cap).tanh() * cap);
                mask_data.map_or(score, |mask| score + mask[row * key_len + key_index])
            };
            // A sink logit takes part in the row's softmax and weighs no value.
            let sink = sinks_data.map(|sinks| sinks[batch_head % heads.max(1)]);
            let max_score = visible
                .clone()
                .map(score)
                .chain(sink)
                .fold(T::neg_infinity(), T::max);
            let score_sum = visible
                .clone()
                .map(score)
                .chain(sink)
                .map(|score| (score - max_score).exp())
                .fold(T::zero(), |sum, weight| sum + weight);
            for (feature, output) in output.iter_mut().enumerate() {
                *output = visible.clone().fold(T::zero(), |sum, key_index| {
                    let weight = (score(key_index) - max_score).exp() / score_sum;
                    weight.mul_add(values[key_index * value_dim + feature], sum)
                });
            }
            Ok(())
        },
    )?;
    Ok(output_data)
}

pub fn scaled_dot_product_attention_shape(
    query: &Shape,
    key: &Shape,
    value: &Shape,
) -> Result<(i64, i64, i64, i64, i64, i64)> {
    let problem = spec::attention_problem(query, key, value, "cpu scaled_dot_product_attention")?;
    Ok((
        problem.batch,
        problem.heads,
        problem.query_len,
        problem.key_len,
        problem.head_dim,
        problem.value_dim,
    ))
}

// Attends ragged queries `[T, H, D]` over paged key and value caches `[P, S, G, D]`, as FlashInfer batches them.
// Sequence `b` holds queries `query_offsets[b]..query_offsets[b + 1]` and the keys of pages `pages[page_offsets[b]..page_offsets[b + 1]]`, all full but the last, which holds `last_page_lengths[b]`.
// Its queries are its last tokens, so a causal mask lets each see the keys up to its own position; a query that sees no key yields zeros.
pub fn paged_attention(
    inputs: &[&Tensor],
    attention: ScaledDotProductAttentionAttributes,
) -> Result<Vec<Tensor>> {
    let &[
        query,
        key_cache,
        value_cache,
        query_offsets,
        page_offsets,
        pages,
        last_page_lengths,
    ] = inputs
    else {
        return Err(Error::missing_input("cpu paged_attention"));
    };
    if let Some(actual) = inputs
        .iter()
        .map(|input| input.device())
        .find(|device| *device != Device::Cpu)
    {
        return Err(cpu_device_mismatch("cpu paged_attention input", actual));
    }

    let data_type = query.data_type();
    if matches!(data_type, DataType::F16 | DataType::Bf16) {
        let wide = |tensor: &Tensor| tensor.cast(DataType::F32);
        let (query, key_cache, value_cache) = (wide(query)?, wide(key_cache)?, wide(value_cache)?);
        let mut outputs = paged_attention(
            &[
                &query,
                &key_cache,
                &value_cache,
                query_offsets,
                page_offsets,
                pages,
                last_page_lengths,
            ],
            attention,
        )?;
        outputs[0] = outputs[0].cast(data_type)?;
        return Ok(outputs);
    }

    let problem = spec::paged_attention_problem(
        query.shape(),
        key_cache.shape(),
        value_cache.shape(),
        [
            (query_offsets.shape(), query_offsets.data_type()),
            (page_offsets.shape(), page_offsets.data_type()),
            (pages.shape(), pages.data_type()),
            (last_page_lengths.shape(), last_page_lengths.data_type()),
        ],
        "cpu paged_attention",
    )?;
    let [query_offsets, page_offsets, pages, last_page_lengths] = checked_paged_values(
        [query_offsets, page_offsets, pages, last_page_lengths],
        problem.tokens,
        key_cache.shape().dimensions()[0],
        problem.page_size,
    )?;
    let geometry = PagedGeometry {
        heads: to_usize(problem.heads, "cpu paged_attention heads")?,
        kv_heads: to_usize(problem.kv_heads, "cpu paged_attention kv heads")?,
        head_dim: to_usize(problem.head_dim, "cpu paged_attention head dimension")?,
        page_size: to_usize(problem.page_size, "cpu paged_attention page size")?,
        query_offsets,
        page_offsets,
        pages,
        last_page_lengths,
    };
    match data_type {
        DataType::F32 => attend_paged::<f32>(query, key_cache, value_cache, &geometry, attention),
        DataType::F64 => attend_paged::<f64>(query, key_cache, value_cache, &geometry, attention),
        _ => Err(Error::unsupported_data_type("cpu paged_attention")),
    }
}

// Reads a paged cache's index tensors, refusing values outside the queries and the cache.
fn checked_paged_values(
    indices: [&Tensor; 4],
    tokens: i64,
    cache_pages: i64,
    page_size: i64,
) -> Result<[Vec<usize>; 4]> {
    let read = |tensor: &Tensor| tensor.cast(DataType::I64)?.to_vec::<i64>();
    let [query_offsets, page_offsets, pages, last_page_lengths] = indices;
    let values = [
        read(query_offsets)?,
        read(page_offsets)?,
        read(pages)?,
        read(last_page_lengths)?,
    ];
    spec::check_paged_values(
        [&values[0], &values[1], &values[2], &values[3]],
        tokens,
        cache_pages,
        page_size,
    )?;
    // What the check let through unread, such as the last page length of a sequence with no pages, is never used.
    let convert = |values: Vec<i64>| {
        values
            .into_iter()
            .map(|value| to_usize(value.max(0), "cpu paged attention index"))
            .collect::<Result<Vec<_>>>()
    };
    let [query_offsets, page_offsets, pages, last_page_lengths] = values;
    Ok([
        convert(query_offsets)?,
        convert(page_offsets)?,
        convert(pages)?,
        convert(last_page_lengths)?,
    ])
}

struct PagedGeometry {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    query_offsets: Vec<usize>,
    page_offsets: Vec<usize>,
    pages: Vec<usize>,
    last_page_lengths: Vec<usize>,
}

fn attend_paged<T: NativeFloat>(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    geometry: &PagedGeometry,
    attention: ScaledDotProductAttentionAttributes,
) -> Result<Vec<Tensor>> {
    let PagedGeometry {
        heads,
        kv_heads,
        head_dim,
        page_size,
        ..
    } = *geometry;
    let scale = attention
        .scale
        .map_or_else(
            || <T as NumCast>::from(head_dim).map(|head_dim| head_dim.sqrt().recip()),
            <T as NumCast>::from,
        )
        .ok_or_else(|| Error::unsupported_operation("cpu paged_attention scale"))?;
    let softcap = attention.softcap.and_then(<T as NumCast>::from);
    let window = attention
        .window
        .map(|window| to_usize(window, "cpu paged_attention window"))
        .transpose()?;
    let (query, key_cache, value_cache) = (
        query.contiguous()?,
        key_cache.contiguous()?,
        value_cache.contiguous()?,
    );
    let query_data = logical_values::<T>(&query)?;
    let key_data = logical_values::<T>(&key_cache)?;
    let value_data = logical_values::<T>(&value_cache)?;
    let mut output = vec![T::zero(); query_data.len()];
    let mut lse = vec![f32::NEG_INFINITY; query_data.len() / head_dim.max(1)];
    let groups = heads / kv_heads.max(1);
    for sequence in 0..geometry.last_page_lengths.len() {
        let (first_query, end_query) = (
            geometry.query_offsets[sequence],
            geometry.query_offsets[sequence + 1],
        );
        let sequence_pages = geometry
            .pages
            .get(geometry.page_offsets[sequence]..geometry.page_offsets[sequence + 1])
            .ok_or_else(|| Error::out_of_range("cpu paged_attention page offsets"))?;
        let keys = sequence_pages.len().saturating_sub(1) * page_size
            + if sequence_pages.is_empty() {
                0
            } else {
                geometry.last_page_lengths[sequence]
            };
        // The slot of key `position` in the caches, counted in rows of `head_dim` values per key head.
        let slot = |position: usize| {
            sequence_pages[position / page_size] * page_size + position % page_size
        };
        let queries = end_query.saturating_sub(first_query);
        for token in first_query..end_query {
            // Queries are the sequence's last tokens.
            let position = (keys + token - first_query).saturating_sub(queries);
            let end = if attention.causal {
                keys.min(position + 1)
            } else {
                keys
            };
            let start = window.map_or(0, |window| (position + 1).saturating_sub(window));
            for head in 0..heads {
                let kv_head = head / groups.max(1);
                let row = &query_data[(token * heads + head) * head_dim..][..head_dim];
                let score = |key: usize| {
                    let offset = (slot(key) * kv_heads + kv_head) * head_dim;
                    let score = row
                        .iter()
                        .zip(&key_data[offset..offset + head_dim])
                        .fold(T::zero(), |sum, (query, key)| query.mul_add(*key, sum))
                        * scale;
                    softcap.map_or(score, |cap| (score / cap).tanh() * cap)
                };
                let visible = start..end.max(start);
                if visible.is_empty() {
                    continue;
                }

                let largest = visible.clone().map(score).fold(T::neg_infinity(), T::max);
                let weights = visible
                    .clone()
                    .map(|key| (score(key) - largest).exp())
                    .collect::<Vec<_>>();
                let total = weights.iter().fold(T::zero(), |sum, &weight| sum + weight);
                lse[token * heads + head] = <f32 as NumCast>::from(largest + total.ln())
                    .ok_or_else(|| Error::unsupported_operation("cpu paged_attention lse"))?
                    / F32_LN_2;
                let out = &mut output[(token * heads + head) * head_dim..][..head_dim];
                for (key, weight) in visible.zip(&weights) {
                    let offset = (slot(key) * kv_heads + kv_head) * head_dim;
                    for (out, value) in out.iter_mut().zip(&value_data[offset..offset + head_dim]) {
                        *out = (*weight / total).mul_add(*value, *out);
                    }
                }
            }
        }
    }

    drop(value_data);
    drop(key_data);
    drop(query_data);
    let lse_shape = Shape::create([
        i64::try_from(lse.len() / heads.max(1)).map_err(|_| Error::IndexOutOfBounds)?,
        i64::try_from(heads).map_err(|_| Error::IndexOutOfBounds)?,
    ])?;
    Ok(vec![
        Tensor::from_vec(Device::Cpu, query.shape().clone(), output)?,
        Tensor::from_vec(Device::Cpu, lse_shape, lse)?,
    ])
}

// Latent attention over paged caches, in `F64`: each head's scores are its latent query against the latent cache plus its positional query against the positional cache, and its output weighs the latent cache.
// Returns the output, of the queries' type, and each row's base-two log-sum-exp of its scaled scores in `F32`, negative infinity where a row sees no key.
pub fn paged_mla_attention(
    inputs: &[&Tensor],
    attention: ScaledDotProductAttentionAttributes,
) -> Result<Vec<Tensor>> {
    let &[
        query,
        query_positions,
        latent_cache,
        position_cache,
        query_offsets,
        page_offsets,
        pages,
        last_page_lengths,
    ] = inputs
    else {
        return Err(Error::missing_input("cpu paged_mla_attention"));
    };
    if let Some(actual) = inputs
        .iter()
        .map(|input| input.device())
        .find(|device| *device != Device::Cpu)
    {
        return Err(cpu_device_mismatch("cpu paged_mla_attention input", actual));
    }

    let problem = spec::paged_mla_attention_problem(
        query.shape(),
        query_positions.shape(),
        latent_cache.shape(),
        position_cache.shape(),
        [
            (query_offsets.shape(), query_offsets.data_type()),
            (page_offsets.shape(), page_offsets.data_type()),
            (pages.shape(), pages.data_type()),
            (last_page_lengths.shape(), last_page_lengths.data_type()),
        ],
        "cpu paged_mla_attention",
    )?;
    let extent = |value: i64| to_usize(value, "cpu paged_mla_attention extent");
    let (heads, latent, positional, page_size) = (
        extent(problem.heads)?,
        extent(problem.latent_dim)?,
        extent(problem.position_dim)?,
        extent(problem.page_size)?,
    );
    let [query_offsets, page_offsets, pages, last_page_lengths] = checked_paged_values(
        [query_offsets, page_offsets, pages, last_page_lengths],
        problem.tokens,
        latent_cache.shape().dimensions()[0],
        problem.page_size,
    )?;
    let values = |tensor: &Tensor| tensor.cast(DataType::F64)?.contiguous()?.to_vec::<f64>();
    let (query_data, position_data, latent_data, cached_positions) = (
        values(query)?,
        values(query_positions)?,
        values(latent_cache)?,
        values(position_cache)?,
    );
    #[expect(
        clippy::cast_precision_loss,
        reason = "head dimensions are far below 2^52"
    )]
    let scale = attention.scale.map_or_else(
        || 1.0 / ((latent + positional) as f64).sqrt(),
        <f64 as From<f32>>::from,
    );
    let mut output = vec![0.0; query_data.len()];
    let mut lse = vec![f32::NEG_INFINITY; query_data.len() / latent.max(1)];
    for sequence in 0..last_page_lengths.len() {
        let (first_query, end_query) = (query_offsets[sequence], query_offsets[sequence + 1]);
        let sequence_pages = pages
            .get(page_offsets[sequence]..page_offsets[sequence + 1])
            .ok_or_else(|| Error::out_of_range("cpu paged_mla_attention page offsets"))?;
        let keys = match sequence_pages.len() {
            0 => 0,
            count => (count - 1) * page_size + last_page_lengths[sequence],
        };
        let slot = |key: usize| sequence_pages[key / page_size] * page_size + key % page_size;
        let queries = end_query.saturating_sub(first_query);
        for token in first_query..end_query {
            // Queries are the sequence's last tokens.
            let position = (keys + token - first_query).saturating_sub(queries);
            let end = if attention.causal {
                keys.min(position + 1)
            } else {
                keys
            };
            if end == 0 {
                continue;
            }
            for head in 0..heads {
                let row = token * heads + head;
                let latent_query = &query_data[row * latent..][..latent];
                let position_query = &position_data[row * positional..][..positional];
                let scores = (0..end)
                    .map(|key| {
                        let slot = slot(key);
                        let dot = |query: &[f64], cache: &[f64], width: usize| {
                            query
                                .iter()
                                .zip(&cache[slot * width..][..width])
                                .map(|(query, key)| query * key)
                                .sum::<f64>()
                        };
                        (dot(latent_query, &latent_data, latent)
                            + dot(position_query, &cached_positions, positional))
                            * scale
                    })
                    .collect::<Vec<_>>();
                let largest = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - largest).exp())
                    .collect::<Vec<_>>();
                let total = weights.iter().sum::<f64>();
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the log-sum-exp output is F32"
                )]
                {
                    lse[row] = ((largest + total.ln()) / LN_2) as f32;
                }

                let out = &mut output[row * latent..][..latent];
                for (key, weight) in weights.iter().enumerate() {
                    let values = &latent_data[slot(key) * latent..][..latent];
                    for (out, value) in out.iter_mut().zip(values) {
                        *out += weight / total * value;
                    }
                }
            }
        }
    }
    Ok(vec![
        Tensor::from_vec(Device::Cpu, query.shape().clone(), output)?.cast(query.data_type())?,
        Tensor::from_vec(Device::Cpu, problem.lse_shape, lse)?,
    ])
}
