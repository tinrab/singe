use std::f32::consts::LOG2_E;

use singe_codegen::emit::cuda_c::spelling::store_expression;

use crate::{
    backend::{
        cuda::{
            cudnn_backend::{CudnnPaging, cudnn_takes_attention},
            kernel_plan::source::{
                KernelInput, KernelPlan,
                paging::{PagedSequence, paged_indices, report_index_error},
                writer::KernelWriter,
            },
        },
        kernel_lowering::element_type,
        registry::{Operation, ScaledDotProductAttentionAttributes},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    utility::to_usize,
    value::ValueSpec,
};

// What cuDNN needs to run paged attention of `inputs`, the operands of `paged_attention`, or `None` when it cannot.
pub fn cudnn_paging(
    inputs: &[KernelInput],
    attention: &ScaledDotProductAttentionAttributes,
) -> Result<Option<CudnnPaging>> {
    let [
        query,
        key_cache,
        value_cache,
        query_offsets,
        page_offsets,
        pages,
        last_page_lengths,
    ] = inputs
    else {
        return Ok(None);
    };
    let head_dimension = query.layout.dimensions().last().copied().unwrap_or(0);
    if !cudnn_takes_attention(query.data_type, head_dimension, attention)
        || key_cache.data_type != query.data_type
        || value_cache.data_type != query.data_type
        || inputs.iter().any(|input| !input.layout.is_contiguous())
    {
        return Ok(None);
    }

    let problem = spec::paged_attention_problem(
        query.layout.shape(),
        key_cache.layout.shape(),
        value_cache.layout.shape(),
        paged_indices([query_offsets, page_offsets, pages, last_page_lengths]),
        "cuda paged attention",
    )?;
    let (Some(&table), Some(&cache_pages)) = (
        pages.layout.dimensions().first(),
        key_cache.layout.dimensions().first(),
    ) else {
        return Err(Error::invalid_argument(
            "cuda paged attention page table and key cache dimensions",
        ));
    };
    // cuDNN only takes pages of a power of two.
    // cuDNN's paged attention spreads one sequence's heads across the GPU but runs a step's sequences one after another: at 32 sequences a decoding call took 84 µs on 9 blocks, where the generated kernel takes 25 µs, so it runs only steps of one sequence.
    if problem.tokens == 0
        || problem.sequences != 1
        || table == 0
        || cache_pages == 0
        || problem.page_size.count_ones() != 1
    {
        return Ok(None);
    }
    Ok(Some(CudnnPaging {
        sequences: problem.sequences,
        tokens: problem.tokens,
        heads: problem.heads,
        kv_heads: problem.kv_heads,
        dim: problem.head_dim,
        page_size: problem.page_size,
        cache_pages,
        table,
    }))
}

// Plans the kernel that turns `indices`, the query offsets, page offsets, pages, and last page lengths, into cuDNN's packed offsets, query lengths, key lengths, and page table.
// It refuses what the generated kernels refuse, and what it writes stays inside every buffer even then, since a graph runs on until it reads the status.
pub fn plan_cudnn_paging_indices(
    paging: &CudnnPaging,
    indices: &[KernelInput],
) -> Result<KernelPlan> {
    let [query_offsets, page_offsets, pages, last_page_lengths] = indices else {
        return Err(Error::missing_input("cudnn paging indices"));
    };
    let extent = |value: i64| to_usize(value, "cudnn paging indices");
    let (sequences, tokens, table) = (
        extent(paging.sequences)?,
        extent(paging.tokens)?,
        extent(paging.table)?,
    );
    let (page_size, cache_pages, rows_per_query, row) = (
        extent(paging.page_size)?,
        extent(paging.cache_pages)?,
        extent(paging.heads)?,
        extent(paging.query_elements())?,
    );
    let mut writer = KernelWriter::new();
    let qo = writer.input(query_offsets.data_type, "query_offsets")?;
    let po = writer.input(page_offsets.data_type, "page_offsets")?;
    let table_in = writer.input(pages.data_type, "pages")?;
    let lp = writer.input(last_page_lengths.data_type, "last_page_lengths")?;
    let offsets = writer.output(DataType::I32, "packed_offsets")?;
    let query_lengths = writer.output(DataType::I32, "query_lengths")?;
    let key_lengths = writer.output(DataType::I32, "key_lengths")?;
    let page_table = writer.output(DataType::I32, "page_table")?;
    writer.status();
    let launch = writer.elements(sequences * table);
    let report = report_index_error();
    let sequence_values = PagedSequence {
        query_offsets: &qo,
        page_offsets: &po,
        last_page_lengths: &lp,
        tokens,
        page_table: table,
        page_size,
        rows_per_query,
    }
    .checked_values();
    writer.lines(&format!(
        "
const unsigned long long sequence = index / {table}ULL, entry = index % {table}ULL;
{page_table}[index] = 0;
if (entry == 0ULL) {{
    long long running = 0LL, sequence_start = 0LL;
    for (unsigned long long other = 0; other <= sequence + 1ULL; ++other) {{
        long long offset = static_cast<long long>({qo}[other]);
        offset = offset < 0LL ? 0LL : (offset > {tokens}LL ? {tokens}LL : offset);
        if (offset > running) running = offset;
        if (other == sequence) sequence_start = running;
    }}
    {offsets}[sequence] = static_cast<int>(sequence_start * {row}LL);
    if (sequence + 1ULL == {sequences}ULL) {offsets}[sequence + 1ULL] = static_cast<int>(running * {row}LL);
    {query_lengths}[sequence] = static_cast<int>(running - sequence_start);
    {key_lengths}[sequence] = 0;
}}
{sequence_values}
if (entry == 0ULL) {key_lengths}[sequence] = static_cast<int>(keys);
const long long table_entry = page_start + static_cast<long long>(entry);
if (table_entry < page_end) {{
    const long long page = static_cast<long long>({table_in}[table_entry]);
    if (page < 0LL || page >= {cache_pages}LL) {{
        {report}
        return;
    }}
    {page_table}[index] = static_cast<int>(page);
}}"
    ));
    let ([offset_count, ..], _) = paging.offsets_shape();
    let output = |dimensions: Vec<i64>| -> Result<ValueSpec> {
        Ok(ValueSpec::new(DataType::I32, Shape::create(dimensions)?))
    };
    let specs = vec![
        output(vec![offset_count])?,
        output(vec![paging.sequences])?,
        output(vec![paging.sequences])?,
        output(vec![paging.sequences * paging.table])?,
    ];
    Ok(writer.finish(
        "paged_cudnn_indices",
        launch,
        specs,
        indices,
        &[Operation::PagedAttention],
    ))
}

// Plans the kernel that writes the query lengths cuDNN reads with dense attention's key lengths, `batch` of them, each `queries`.
// A program writes them each run, since it holds no constants of its own making.
pub fn plan_cudnn_query_lengths(batch: i64, queries: i64) -> Result<KernelPlan> {
    let count = to_usize(batch, "cudnn query lengths")?;
    let queries =
        i32::try_from(queries).map_err(|_| Error::out_of_range("cudnn attention queries"))?;
    let mut writer = KernelWriter::new();
    let lengths = writer.output(DataType::I32, "lengths")?;
    let launch = writer.elements(count);
    writer.lines(&format!("{lengths}[index] = {queries};"));
    Ok(writer.finish(
        "cudnn_query_lengths",
        launch,
        vec![ValueSpec::new(DataType::I32, Shape::create([batch])?)],
        &[],
        &[Operation::ScaledDotProductAttention],
    ))
}

// Plans the kernel that follows cuDNN: `inputs` are the query offsets, page offsets, last page lengths, and cuDNN's statistics `[B, H, S]`.
// Each row's natural-log statistic becomes a base-two `lse`, and a row that belongs to no sequence or sees no key gets the zero output and infinite `lse` that generated kernels give it, since cuDNN leaves those rows alone.
pub fn plan_cudnn_paging_results(
    paging: &CudnnPaging,
    attention: &ScaledDotProductAttentionAttributes,
    data_type: DataType,
    inputs: &[KernelInput],
) -> Result<KernelPlan> {
    let [query_offsets, page_offsets, last_page_lengths, stats] = inputs else {
        return Err(Error::missing_input("cudnn paging results"));
    };
    let extent = |value: i64| to_usize(value, "cudnn paging results");
    let (sequences, tokens, heads, dim, page_size) = (
        extent(paging.sequences)?,
        extent(paging.tokens)?,
        extent(paging.heads)?,
        extent(paging.dim)?,
        extent(paging.page_size)?,
    );
    let mut writer = KernelWriter::new();
    let qo = writer.input(query_offsets.data_type, "query_offsets")?;
    let po = writer.input(page_offsets.data_type, "page_offsets")?;
    let lp = writer.input(last_page_lengths.data_type, "last_page_lengths")?;
    let statistics = writer.input(stats.data_type, "statistics")?;
    let output = writer.output(data_type, "output")?;
    let lse = writer.output(DataType::F32, "lse")?;
    let launch = writer.elements(tokens * heads);
    let zero = store_expression(element_type(data_type)?, "0.0f");
    let causal = attention.causal;
    writer.lines(&format!(
        "
const long long token = static_cast<long long>(index / {heads}ULL);
const unsigned long long head = index % {heads}ULL;
for (unsigned long long sequence = 0; sequence < {sequences}ULL; ++sequence) {{
    const long long query_start = static_cast<long long>({qo}[sequence]);
    const long long query_end = static_cast<long long>({qo}[sequence + 1ULL]);
    if (query_start < 0LL || token < query_start || token >= query_end) continue;
    const long long page_start = static_cast<long long>({po}[sequence]);
    const long long page_end = static_cast<long long>({po}[sequence + 1ULL]);
    const long long keys = page_end > page_start ? (page_end - page_start - 1LL) * {page_size}LL + static_cast<long long>({lp}[sequence]) : 0LL;
    const long long position = keys - (query_end - query_start) + (token - query_start);
    if (keys > 0LL && (!{causal} || position >= 0LL)) {{
        {lse}[index] = {statistics}[((sequence * {heads}ULL + head) * {tokens}ULL) + static_cast<unsigned long long>(token - query_start)] * {LOG2_E:?}f;
        return;
    }}
    break;
}}
for (unsigned long long d = 0; d < {dim}ULL; ++d) {output}[index * {dim}ULL + d] = {zero};
{lse}[index] = -__uint_as_float(0x7f800000U);"
    ));
    let specs = vec![
        ValueSpec::new(
            data_type,
            Shape::create([paging.tokens, paging.heads, paging.dim])?,
        ),
        ValueSpec::new(DataType::F32, Shape::create([paging.tokens, paging.heads])?),
    ];
    Ok(writer.finish(
        "paged_cudnn_results",
        launch,
        specs,
        inputs,
        &[Operation::PagedAttention],
    ))
}
