use crate::{
    backend::{
        cuda::kernel_plan::source::KernelInput, kernel_lowering::STATUS_INDEX_OUT_OF_BOUNDS,
    },
    data_type::DataType,
    shape::Shape,
};

// The part of a paged kernel's body that reads one sequence's paging values, reporting and returning when any would address outside the kernel's buffers.
pub struct PagedSequence<'a> {
    pub query_offsets: &'a str,
    pub page_offsets: &'a str,
    pub last_page_lengths: &'a str,
    pub tokens: usize,
    pub page_table: usize,
    pub page_size: usize,
    pub rows_per_query: usize,
}

// The statement that reports an index out of range through the status word, which is null under stream capture.
pub fn report_index_error() -> String {
    format!("if (status != nullptr) atomicCAS(status, 0U, {STATUS_INDEX_OUT_OF_BOUNDS}U);")
}

impl PagedSequence<'_> {
    // Reads `sequence`'s offsets and last page length, reporting and returning when any would address outside the kernel's buffers.
    fn checked_bounds(&self) -> String {
        let Self {
            query_offsets: qo,
            page_offsets: po,
            last_page_lengths: lp,
            tokens,
            page_table,
            page_size,
            ..
        } = self;
        let report = report_index_error();
        format!(
            "const long long query_start = static_cast<long long>({qo}[sequence]);
const long long query_end = static_cast<long long>({qo}[sequence + 1]);
const long long page_start = static_cast<long long>({po}[sequence]);
const long long page_end = static_cast<long long>({po}[sequence + 1]);
const long long last_length = static_cast<long long>({lp}[sequence]);
if (query_start < 0LL || query_end < query_start || query_end > {tokens}LL
    || page_start < 0LL || page_end < page_start || page_end > {page_table}LL
    || (page_end > page_start && (last_length < 0LL || last_length > {page_size}LL))) {{
    {report}
    return;
}}"
        )
    }

    // The checked values of `sequence`, for code that goes on to other sequences rather than returning at the first rows past this one's.
    pub fn checked_values(&self) -> String {
        format!(
            "{}
const long long keys = page_end > page_start ? (page_end - page_start - 1LL) * {page_size}LL + last_length : 0LL;",
            self.checked_bounds(),
            page_size = self.page_size,
        )
    }

    // The checked values of `sequence`, for a block that takes a tile of its rows from `first_row`, and returns past them.
    pub fn source(&self) -> String {
        format!(
            "{}
const unsigned int queries = static_cast<unsigned int>(query_end - query_start);
const unsigned int packed = queries * {rows_per_query}U;
if (first_row >= packed) return;
const unsigned int page_count = static_cast<unsigned int>(page_end - page_start);
const unsigned int keys = page_count == 0U ? 0U : (page_count - 1U) * {page_size}U + static_cast<unsigned int>(last_length);",
            self.checked_bounds(),
            rows_per_query = self.rows_per_query,
            page_size = self.page_size,
        )
    }
}

// The shape and type of each paging operand, as `spec`'s problems read them.
pub fn paged_indices(operands: [&KernelInput; 4]) -> [(&Shape, DataType); 4] {
    operands.map(|input| (input.layout.shape(), input.data_type))
}
