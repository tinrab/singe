use crate::{
    backend::spec::neural::{
        ensure_non_negative_dimensions, ensure_positive_dimensions, fixed_attention_rank,
        fixed_rank_dimensions,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    utility::{checked_mul_i64, to_usize},
};

// Whether fused attention kernels take queries, keys, and values of `data_type` with these shapes: `F16` or `BF16` `[B, H, L, D]` queries and `[B, G, S, D]` keys and values with `G` dividing `H`, nothing empty, and a head dimension that is a multiple of 16 up to 128.
#[must_use]
pub fn fuses_attention(data_type: DataType, query: &Shape, key: &Shape, value: &Shape) -> bool {
    let (q, k) = (query.dimensions(), key.dimensions());
    matches!(data_type, DataType::F16 | DataType::Bf16)
        && q.len() == 4
        && k.len() == 4
        && key == value
        && q[0] == k[0]
        && q[3] == k[3]
        && q.iter().chain(k).all(|&extent| extent > 0)
        && q[1] % k[1] == 0
        && q[3] % 16 == 0
        && q[3] <= 128
}

/// Validated dimensions for repeating key or value heads in a rank-four tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepeatKvProblem {
    /// Shape `[batch, output_heads, sequence, head_dim]`.
    pub output_shape: Shape,
    /// Number of sequences in the batch.
    pub batch: i64,
    /// Number of heads in each input sequence.
    pub heads: i64,
    /// Input head count multiplied by the repeat count.
    pub output_heads: i64,
    /// Number of positions per sequence.
    pub sequence: i64,
    /// Number of elements in each input head.
    pub head_dim: i64,
    /// Positive number of consecutive output heads copied from each input head.
    pub repeats: i64,
}

/// Infers repetition of each head in a `[batch, heads, sequence, head_dim]` tensor.
///
/// `diagnostic_prefix` identifies the operation in errors.
///
/// # Errors
///
/// Returns an error if input rank is not four, repeats are nonpositive, any extent is negative, multiplying head count by repeats overflows `i64`, or constructing the output shape fails.
pub fn repeat_kv_problem(
    input: &Shape,
    repeats: i64,
    diagnostic_prefix: &'static str,
) -> Result<RepeatKvProblem> {
    if repeats <= 0 {
        return Err(Error::invalid_argument(format!(
            "{diagnostic_prefix} repeats"
        )));
    }

    let [batch, heads, sequence, head_dim] =
        fixed_rank_dimensions(diagnostic_prefix, "input", input.dimensions(), 4, "BHQD")?;
    ensure_non_negative_dimensions(
        diagnostic_prefix,
        &[
            ("batch", batch),
            ("heads", heads),
            ("sequence", sequence),
            ("head dim", head_dim),
        ],
    )?;
    let output_heads =
        checked_mul_i64(heads, repeats, &format!("{diagnostic_prefix} output heads"))?;
    Ok(RepeatKvProblem {
        output_shape: Shape::create([batch, output_heads, sequence, head_dim])?,
        batch,
        heads,
        output_heads,
        sequence,
        head_dim,
        repeats,
    })
}

/// Validated input and factor dimensions for adjacent-pair rotary embeddings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotaryEmbeddingProblem {
    /// Output shape, identical to the input shape.
    pub output_shape: Shape,
    /// Number of sequences in the batch.
    pub batch: i64,
    /// Number of heads in each input sequence.
    pub heads: i64,
    /// Number of positions per sequence.
    pub sequence: i64,
    /// Number of elements in each input head.
    pub head_dim: i64,
    /// Number of adjacent element pairs rotated in each head.
    pub rotary_pairs: i64,
    /// Number of rotated head elements, equal to twice the pair count.
    pub rotary_dim: i64,
}

/// Checks rotary input shape `[batch, heads, sequence, head_dim]` against cosine and sine shapes `[sequence, rotary_pairs]`.
///
/// The factors must have identical shapes and cover every input position.
/// Twice the pair count must not exceed the head width.
/// `diagnostic_prefix` identifies the operation in errors.
///
/// # Errors
///
/// Returns an error if ranks, factor shapes, or sequence lengths disagree, an extent is negative, the rotated width overflows `i64` or exceeds the head width, or converting a reported extent to `usize` fails.
pub fn rotary_embedding_problem(
    input: &Shape,
    cos: &Shape,
    sin: &Shape,
    diagnostic_prefix: &'static str,
) -> Result<RotaryEmbeddingProblem> {
    let [batch, heads, sequence, head_dim] =
        fixed_rank_dimensions(diagnostic_prefix, "input", input.dimensions(), 4, "BHSK")?;
    let [cos_sequence, rotary_pairs] =
        fixed_rank_dimensions(diagnostic_prefix, "cos", cos.dimensions(), 2, "SP")?;
    fixed_rank_dimensions::<2>(diagnostic_prefix, "sin", sin.dimensions(), 2, "SP")?;
    ensure_non_negative_dimensions(
        diagnostic_prefix,
        &[
            ("batch", batch),
            ("heads", heads),
            ("sequence", sequence),
            ("head dim", head_dim),
            ("cos sequence", cos_sequence),
            ("rotary pairs", rotary_pairs),
        ],
    )?;

    if sin.dimensions() != cos.dimensions() {
        return Err(Error::shape_mismatch(
            format!("{diagnostic_prefix} sin shape"),
            cos.dimensions().to_vec(),
            sin.dimensions().to_vec(),
        ));
    }

    if cos_sequence != sequence {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} sequence"),
            to_usize(sequence, "backend spec dimension")?,
            to_usize(cos_sequence, "backend spec dimension")?,
        ));
    }

    let rotary_dim = checked_mul_i64(
        rotary_pairs,
        2,
        &format!("{diagnostic_prefix} rotary dimension"),
    )?;
    if rotary_dim > head_dim {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} rotary dimension"),
            to_usize(head_dim, "backend spec dimension")?,
            to_usize(rotary_dim, "backend spec dimension")?,
        ));
    }

    Ok(RotaryEmbeddingProblem {
        output_shape: input.clone(),
        batch,
        heads,
        sequence,
        head_dim,
        rotary_pairs,
        rotary_dim,
    })
}

/// Validated rank-four query, key, and value dimensions for scaled dot-product attention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionProblem {
    /// Shape `[batch, heads, query_len, value_dim]`.
    pub output_shape: Shape,
    /// Batch count shared by query, key, and value.
    pub batch: i64,
    /// Head count shared by query, key, and value.
    pub heads: i64,
    /// Number of query positions and output positions per head.
    pub query_len: i64,
    /// Positive sequence length shared by key and value.
    pub key_len: i64,
    /// Positive head width shared by query and key.
    pub head_dim: i64,
    /// Number of value elements produced per query position.
    pub value_dim: i64,
}

/// Validates rank-four attention shapes and infers the output dimensions.
///
/// Batch and head counts must match across all inputs.
/// Query and key head widths must match, as must key and value sequence lengths.
/// The key sequence and query head width must be positive.
/// `diagnostic_prefix` identifies the operation in errors.
///
/// # Errors
///
/// Returns an error if ranks, shared extents, positivity requirements, or nonnegative extents are invalid, converting a reported extent to `usize` fails, or constructing the output shape fails.
pub fn attention_problem(
    query: &Shape,
    key: &Shape,
    value: &Shape,
    diagnostic_prefix: &'static str,
) -> Result<AttentionProblem> {
    let query_dims = query.dimensions();
    let key_dims = key.dimensions();
    let value_dims = value.dimensions();
    let [batch, heads, query_len, head_dim] =
        fixed_attention_rank(diagnostic_prefix, "query", query_dims)?;
    let [key_batch, key_heads, key_len, key_head_dim] =
        fixed_attention_rank(diagnostic_prefix, "key", key_dims)?;
    let [value_batch, value_heads, value_len, value_dim] =
        fixed_attention_rank(diagnostic_prefix, "value", value_dims)?;
    ensure_non_negative_dimensions(
        diagnostic_prefix,
        &[
            ("batch", batch),
            ("heads", heads),
            ("query length", query_len),
            ("value dimension", value_dim),
        ],
    )?;
    ensure_positive_dimensions(
        diagnostic_prefix,
        &[("head dimension", head_dim), ("key length", key_len)],
    )?;
    if batch != key_batch {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} key batch"),
            to_usize(batch, "backend spec dimension")?,
            to_usize(key_batch, "backend spec dimension")?,
        ));
    }

    if batch != value_batch {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} value batch"),
            to_usize(batch, "backend spec dimension")?,
            to_usize(value_batch, "backend spec dimension")?,
        ));
    }

    // Grouped-query attention shares each key and value head among `heads / key_heads` query heads, as PyTorch's `enable_gqa` does.
    if key_heads == 0 || heads % key_heads != 0 {
        return Err(Error::invalid_argument(format!(
            "{diagnostic_prefix} key heads must divide the query heads"
        )));
    }

    if key_heads != value_heads {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} value heads"),
            to_usize(key_heads, "backend spec dimension")?,
            to_usize(value_heads, "backend spec dimension")?,
        ));
    }

    if head_dim != key_head_dim {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} key head dimension"),
            to_usize(head_dim, "backend spec dimension")?,
            to_usize(key_head_dim, "backend spec dimension")?,
        ));
    }

    if key_len != value_len {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} value length"),
            to_usize(key_len, "backend spec dimension")?,
            to_usize(value_len, "backend spec dimension")?,
        ));
    }

    Ok(AttentionProblem {
        output_shape: Shape::create([batch, heads, query_len, value_dim])?,
        batch,
        heads,
        query_len,
        key_len,
        head_dim,
        value_dim,
    })
}

// Checks that an attention mask of `mask`'s shape broadcasts to the scores of `problem`, `[batch, heads, query_len, key_len]`, and returns that shape.
pub fn attention_mask_shape(
    problem: &AttentionProblem,
    mask: &Shape,
    diagnostic_prefix: &'static str,
) -> Result<Shape> {
    let scores = [
        problem.batch,
        problem.heads,
        problem.query_len,
        problem.key_len,
    ];
    let fits = mask.dimensions().len() <= scores.len()
        && mask
            .dimensions()
            .iter()
            .rev()
            .zip(scores.iter().rev())
            .all(|(&mask, &score)| mask == 1 || mask == score);
    if !fits {
        return Err(Error::shape_mismatch(
            format!("{diagnostic_prefix} mask"),
            scores.to_vec(),
            mask.dimensions().to_vec(),
        ));
    }
    Shape::create(scores)
}

// The shape of attention over a paged key and value cache, as FlashInfer batches it: ragged queries `[T, H, D]`, whose sequences `query_offsets` delimit, over caches `[P, S, G, D]` of `P` pages of `S` slots, whose pages `page_offsets` delimit in `pages` and whose last page holds `last_page_lengths` keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedAttentionProblem {
    pub tokens: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub page_size: i64,
    pub sequences: i64,
    pub output_shape: Shape,
}

// Checks the shapes and index types of paged attention's operands.
pub fn paged_attention_problem(
    query: &Shape,
    key_cache: &Shape,
    value_cache: &Shape,
    indices: [(&Shape, DataType); 4],
    diagnostic_prefix: &'static str,
) -> Result<PagedAttentionProblem> {
    let invalid = |what: &str| Error::invalid_argument(format!("{diagnostic_prefix} {what}"));
    let (&[tokens, heads, head_dim], &[_, page_size, kv_heads, key_dim]) =
        (query.dimensions(), key_cache.dimensions())
    else {
        return Err(invalid("query or cache rank"));
    };
    if key_cache != value_cache {
        return Err(invalid("value cache shape"));
    }

    if key_dim != head_dim || kv_heads <= 0 || heads % kv_heads != 0 || page_size <= 0 {
        return Err(invalid("heads, head dimension, or page size"));
    }

    let sequences = paged_sequences(indices, diagnostic_prefix)?;
    Ok(PagedAttentionProblem {
        tokens,
        heads,
        kv_heads,
        head_dim,
        page_size,
        sequences,
        output_shape: query.clone(),
    })
}

// Checks that paging values stay inside the queries, the page table, the pages, and the cache, as the CUDA kernels check them while they run.
// `tokens` and `cache_pages` are the first extents of the queries and the cache.
pub fn check_paged_values(
    [query_offsets, page_offsets, pages, last_page_lengths]: [&[i64]; 4],
    tokens: i64,
    cache_pages: i64,
    page_size: i64,
) -> Result<()> {
    let table = i64::try_from(pages.len()).map_err(|_| Error::IndexOutOfBounds)?;
    let ordered = |offsets: &[i64], end: i64, sequence: usize| {
        let (first, last) = (offsets[sequence], offsets[sequence + 1]);
        0 <= first && first <= last && last <= end
    };
    for sequence in 0..last_page_lengths.len() {
        if !ordered(query_offsets, tokens, sequence) || !ordered(page_offsets, table, sequence) {
            return Err(Error::IndexOutOfBounds);
        }

        let (first, last) = (page_offsets[sequence], page_offsets[sequence + 1]);
        // A sequence reads only its own pages, and a sequence with none has no last page.
        if first == last {
            continue;
        }

        let range =
            to_usize(first, "paged attention pages")?..to_usize(last, "paged attention pages")?;
        if !(0..=page_size).contains(&last_page_lengths[sequence])
            || !pages[range]
                .iter()
                .all(|page| (0..cache_pages).contains(page))
        {
            return Err(Error::IndexOutOfBounds);
        }
    }
    Ok(())
}

// The number of sequences the index tensors of a paged cache delimit: `query_offsets` and `page_offsets` of one more than the sequences, the cache's `pages`, and each sequence's `last_page_lengths`, all `I32` or `I64`.
fn paged_sequences(
    indices: [(&Shape, DataType); 4],
    diagnostic_prefix: &'static str,
) -> Result<i64> {
    let invalid = |what: &str| Error::invalid_argument(format!("{diagnostic_prefix} {what}"));
    let [
        (query_offsets, query_type),
        (page_offsets, offsets_type),
        (pages, table_type),
        (last_page_lengths, last_type),
    ] = indices;
    if [query_type, offsets_type, table_type, last_type]
        .iter()
        .any(|data_type| !matches!(data_type, DataType::I32 | DataType::I64))
    {
        return Err(Error::unsupported_data_type(diagnostic_prefix));
    }

    let (&[bounds], &[page_bounds], &[_], &[sequences]) = (
        query_offsets.dimensions(),
        page_offsets.dimensions(),
        pages.dimensions(),
        last_page_lengths.dimensions(),
    ) else {
        return Err(invalid("index rank"));
    };
    if bounds != sequences + 1 || page_bounds != sequences + 1 {
        return Err(invalid("offset count"));
    }
    Ok(sequences)
}

// The shape of latent attention over paged caches, as DeepSeek's absorbed MLA computes it: ragged queries split into a latent part `[T, H, C]` and a positional part `[T, H, R]` attend over one latent cache `[P, S, C]`, which also holds the values, and one positional cache `[P, S, R]`, both shared by every head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedMlaAttentionProblem {
    pub tokens: i64,
    pub heads: i64,
    pub latent_dim: i64,
    pub position_dim: i64,
    pub page_size: i64,
    pub sequences: i64,
    pub output_shape: Shape,
    pub lse_shape: Shape,
}

pub fn paged_mla_attention_problem(
    query: &Shape,
    query_positions: &Shape,
    latent_cache: &Shape,
    position_cache: &Shape,
    indices: [(&Shape, DataType); 4],
    diagnostic_prefix: &'static str,
) -> Result<PagedMlaAttentionProblem> {
    let invalid = |what: &str| Error::invalid_argument(format!("{diagnostic_prefix} {what}"));
    let (
        &[tokens, heads, latent_dim],
        &[position_tokens, position_heads, position_dim],
        &[pages, page_size, cached_latent],
        &[position_pages, position_page_size, cached_position],
    ) = (
        query.dimensions(),
        query_positions.dimensions(),
        latent_cache.dimensions(),
        position_cache.dimensions(),
    )
    else {
        return Err(invalid("query or cache rank"));
    };
    if (position_tokens, position_heads) != (tokens, heads)
        || (position_pages, position_page_size) != (pages, page_size)
        || cached_latent != latent_dim
        || cached_position != position_dim
        || page_size <= 0
        || latent_dim <= 0
    {
        return Err(invalid("query or cache shapes"));
    }

    let sequences = paged_sequences(indices, diagnostic_prefix)?;
    Ok(PagedMlaAttentionProblem {
        tokens,
        heads,
        latent_dim,
        position_dim,
        page_size,
        sequences,
        output_shape: query.clone(),
        lse_shape: Shape::create([tokens, heads])?,
    })
}
