use std::iter;

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    index::TensorIndex,
    nn::{CacheSize, decode::input_group},
    shape::Shape,
    tensor::{Paging, Tensor},
};

input_group! {
    // One decode step's tokens, and where they fall in a paged cache.
    //
    // Everything that changes between steps is a tensor rather than an attribute, so one captured step serves every position.
    pub struct DecodeStep {
        tokens,
        positions,
        write_pages,
        write_slots,
        query_offsets,
        page_offsets,
        pages,
        last_page_lengths,
        last_rows,
    }
}

impl DecodeStep {
    // Returns a step of `tokens` tokens over `sequences` sequences and a page table of `table` entries, carrying shapes and no values, for capturing a program.
    pub fn meta(tokens: i64, sequences: i64, table: i64) -> Result<Self> {
        let vector =
            |length: i64| Tensor::zeroed_on(Device::Meta, DataType::I64, Shape::create([length])?);
        Ok(Self {
            tokens: vector(tokens)?,
            positions: vector(tokens)?,
            write_pages: vector(tokens)?,
            write_slots: vector(tokens)?,
            query_offsets: vector(sequences + 1)?,
            page_offsets: vector(sequences + 1)?,
            pages: vector(table)?,
            last_page_lengths: vector(sequences)?,
            last_rows: vector(sequences)?,
        })
    }

    // Returns the token ids, `[T]`.
    #[must_use]
    pub const fn tokens(&self) -> &Tensor {
        &self.tokens
    }

    // Returns each token's position in its sequence, `[T]`.
    #[must_use]
    pub const fn positions(&self) -> &Tensor {
        &self.positions
    }

    // Returns the row of each sequence's last token, `[S]`.
    #[must_use]
    pub const fn last_rows(&self) -> &Tensor {
        &self.last_rows
    }

    // Returns where this step's queries and their keys sit, as attention reads them.
    #[must_use]
    pub const fn paging(&self) -> Paging<'_> {
        Paging {
            query_offsets: &self.query_offsets,
            page_offsets: &self.page_offsets,
            pages: &self.pages,
            last_page_lengths: &self.last_page_lengths,
        }
    }

    // Returns the index entries that address this step's cache slots, one page and slot per token.
    #[must_use]
    pub fn slots(&self) -> [TensorIndex; 2] {
        [
            TensorIndex::Indices(self.write_pages.clone()),
            TensorIndex::Indices(self.write_slots.clone()),
        ]
    }
}

// Where a page table places a step's tokens, which a loop outside Singe needs to write the same paging tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageLayout {
    pub sequences: usize,
    // The most tokens each sequence holds.
    pub capacity: i64,
    // The slots per page.
    pub page_size: i64,
    // The page held and idle rows write to, after every page a sequence can hold.
    pub scratch_page: i64,
    // The length of a step's page table.
    pub table_length: i64,
}

// Hands out cache pages to sequences, and writes each step's paging tensors.
// A sequence takes free pages in increasing order from page zero.
#[derive(Debug, Clone)]
pub struct PageTable {
    page_size: i64,
    capacity: i64,
    // The pages sequences can hold, and after them one scratch page, which held and idle rows and padding write to and nothing reads as a sequence's key.
    pool: i64,
    // The most tokens a step may be padded by, which the step's page table makes room for.
    padding: i64,
    free: Vec<i64>,
    sequences: Vec<SequencePages>,
}

#[derive(Debug, Clone, Default)]
struct SequencePages {
    pages: Vec<i64>,
    length: i64,
}

impl SequencePages {
    // Returns the page and slot of `position`, taking a free page when it starts a new one.
    fn place(&mut self, position: i64, table: &mut PageSlots<'_>) -> Result<(i64, i64)> {
        let page = usize::try_from(position / table.page_size)
            .map_err(|_| Error::out_of_range("decoder cache page"))?;
        if page == self.pages.len() {
            let fresh = table
                .free
                .pop()
                .ok_or_else(|| Error::out_of_range("decoder cache pages"))?;
            self.pages.push(fresh);
        }
        Ok((self.pages[page], position % table.page_size))
    }
}

struct PageSlots<'a> {
    page_size: i64,
    free: &'a mut Vec<i64>,
}

impl PageTable {
    // Plans `sequences` sequences of up to `capacity` tokens over pages of `page_size` slots, with enough pages for all of them at once.
    pub fn create(sequences: usize, capacity: i64, page_size: i64) -> Result<Self> {
        if sequences == 0 || capacity <= 0 || page_size <= 0 {
            return Err(Error::invalid_argument("page table size"));
        }

        let count = i64::try_from(sequences).map_err(|_| Error::out_of_range("sequences"))?;
        let pool = count
            .checked_mul((capacity - 1) / page_size + 1)
            .ok_or_else(|| Error::out_of_range("page table size"))?;
        Ok(Self::with_pool(sequences, capacity, page_size, pool))
    }

    // Plans sequences as `create` does, sharing pages that hold `cache_tokens` tokens, so a cache can hold less than every sequence's capacity at once.
    pub fn create_shared(
        sequences: usize,
        capacity: i64,
        page_size: i64,
        cache_tokens: i64,
    ) -> Result<Self> {
        if sequences == 0 || capacity <= 0 || page_size <= 0 || cache_tokens <= 0 {
            return Err(Error::invalid_argument("page table size"));
        }
        Ok(Self::with_pool(
            sequences,
            capacity,
            page_size,
            (cache_tokens - 1) / page_size + 1,
        ))
    }

    fn with_pool(sequences: usize, capacity: i64, page_size: i64, pool: i64) -> Self {
        let mut plan = Self {
            page_size,
            capacity,
            pool,
            padding: 0,
            free: Vec::new(),
            sequences: vec![SequencePages::default(); sequences],
        };
        plan.reset();
        plan
    }

    // Lets a step be padded by up to `padding` tokens, which lengthens every step's page table.
    pub fn with_padding(mut self, padding: usize) -> Result<Self> {
        self.padding = i64::try_from(padding).map_err(|_| Error::out_of_range("step padding"))?;
        Ok(self)
    }

    // Forgets every sequence, so the next step starts at position zero.
    // The cache keeps its contents, which a sequence's lengths hide.
    pub fn reset(&mut self) {
        self.free = (0..self.pool).rev().collect();
        for sequence in &mut self.sequences {
            *sequence = SequencePages::default();
        }
    }

    // Forgets one sequence and returns its pages, so its row starts over at position zero and others can take them.
    // The cache keeps their contents, which the next holder's lengths hide.
    pub fn release(&mut self, sequence: usize) -> Result<()> {
        let sequence = self
            .sequences
            .get_mut(sequence)
            .ok_or_else(|| Error::out_of_range("decoder sequence"))?;
        self.free.extend(sequence.pages.drain(..).rev());
        sequence.length = 0;
        Ok(())
    }

    // Returns the pages no sequence holds.
    #[must_use]
    pub const fn free_pages(&self) -> usize {
        self.free.len()
    }

    // Returns the number of pages a cache needs.
    #[must_use]
    pub const fn page_count(&self) -> i64 {
        self.pool + 1
    }

    // Returns what a decoder's caches hold for this table.
    pub fn cache_size(&self) -> Result<CacheSize> {
        Ok(CacheSize {
            pages: self.page_count(),
            page_size: self.page_size(),
            sequences: i64::try_from(self.sequences.len())
                .map_err(|_| Error::out_of_range("sequences"))?,
        })
    }

    // Returns the length of a step's page table, which has room for every held page, an idle row's scratch page, and the scratch pages padding reads.
    pub fn table_length(&self) -> Result<i64> {
        let padding = (self.padding + self.page_size - 1) / self.page_size;
        i64::try_from(self.sequences.len())
            .map(|sequences| self.pool + sequences + padding)
            .map_err(|_| Error::out_of_range("sequences"))
    }

    // Returns where this table places tokens.
    //
    // # Errors
    //
    // Returns an error if the table's length overflows.
    pub fn layout(&self) -> Result<PageLayout> {
        Ok(PageLayout {
            sequences: self.sequences.len(),
            capacity: self.capacity,
            page_size: self.page_size,
            scratch_page: self.pool,
            table_length: self.table_length()?,
        })
    }

    // Returns the slots per page.
    #[must_use]
    pub const fn page_size(&self) -> i64 {
        self.page_size
    }

    // Returns the tokens per sequence the cache holds.
    #[must_use]
    pub const fn capacity(&self) -> i64 {
        self.capacity
    }

    // Returns the number of sequences.
    #[must_use]
    pub fn sequences(&self) -> usize {
        self.sequences.len()
    }

    // Returns how many rows a step of `tokens` takes: one per token, and one for each empty slice.
    #[must_use]
    pub fn step_tokens(tokens: &[&[i64]]) -> usize {
        tokens.iter().map(|tokens| tokens.len().max(1)).sum()
    }

    // Takes pages for `tokens`, one slice per sequence, and writes the step that places them.
    //
    // An empty slice holds a sequence or leaves a row idle: it takes one row, as a compiled step's shape requires, written to the scratch page, so it takes no page or slot, and its logits mean nothing.
    // A step that fails leaves the table as it was.
    pub fn step(&mut self, device: Device, tokens: &[&[i64]]) -> Result<DecodeStep> {
        self.step_padded(device, tokens, Self::step_tokens(tokens))
    }

    // Writes the step `step` would, padded to `total` rows, so it fits a step compiled for that many.
    //
    // The padding follows the last sequence with tokens, as further queries of that sequence whose keys count toward its length in this step alone.
    // Causal attention keeps the sequence's own tokens from reading them, they write the scratch page, and the sequence's last row stays its last real token, so its logits are those of the unpadded step.
    pub fn step_padded(
        &mut self,
        device: Device,
        tokens: &[&[i64]],
        total: usize,
    ) -> Result<DecodeStep> {
        if tokens.len() != self.sequences.len() || tokens.iter().all(|tokens| tokens.is_empty()) {
            return Err(Error::invalid_argument(
                "decode step needs a slice for every sequence, and tokens in one",
            ));
        }

        let padding = total
            .checked_sub(Self::step_tokens(tokens))
            .ok_or_else(|| {
                Error::invalid_argument("a decode step is padded to fewer rows than it has")
            })?;
        let padding = i64::try_from(padding).map_err(|_| Error::out_of_range("step padding"))?;
        if padding > self.padding {
            return Err(Error::invalid_argument(format!(
                "a decode step is padded by {padding} tokens, past the {} its page table allows",
                self.padding
            )));
        }

        let padded = tokens
            .iter()
            .rposition(|tokens| !tokens.is_empty())
            .ok_or_else(|| Error::invalid_argument("decode step has no tokens"))?;

        for (sequence, tokens) in self.sequences.iter().zip(tokens) {
            let added =
                i64::try_from(tokens.len()).map_err(|_| Error::out_of_range("step tokens"))?;
            if sequence.length + added > self.capacity {
                return Err(Error::out_of_range("decoder cache capacity"));
            }
        }

        // Checked before anything moves, so a step that overruns the pool changes nothing.
        let mut fresh = 0;
        for (sequence, tokens) in self.sequences.iter().zip(tokens) {
            if !tokens.is_empty() {
                let end = sequence.length
                    + i64::try_from(tokens.len())
                        .map_err(|_| Error::out_of_range("step tokens"))?;
                let held = i64::try_from(sequence.pages.len())
                    .map_err(|_| Error::out_of_range("step tokens"))?;
                fresh += ((end - 1) / self.page_size + 1 - held).max(0);
            }
        }

        if fresh > i64::try_from(self.free.len()).map_err(|_| Error::out_of_range("step tokens"))? {
            return Err(Error::out_of_range("decoder cache pages"));
        }

        let mut flat = Vec::with_capacity(total);
        let mut positions = Vec::with_capacity(total);
        let mut write_pages = Vec::with_capacity(total);
        let mut write_slots = Vec::with_capacity(total);
        let mut query_offsets = vec![0_i64];
        let mut last_rows = Vec::with_capacity(tokens.len());
        let mut visible = Vec::with_capacity(tokens.len());
        let mut slots = PageSlots {
            page_size: self.page_size,
            free: &mut self.free,
        };
        for (row, (sequence, tokens)) in self.sequences.iter_mut().zip(tokens).enumerate() {
            for &token in *tokens {
                let (page, slot) = sequence.place(sequence.length, &mut slots)?;
                flat.push(token);
                positions.push(sequence.length);
                write_pages.push(page);
                write_slots.push(slot);
                sequence.length += 1;
            }

            if tokens.is_empty() {
                flat.push(0);
                positions.push(sequence.length.min(self.capacity - 1));
                write_pages.push(self.pool);
                write_slots.push(0);
            }

            let last =
                i64::try_from(flat.len()).map_err(|_| Error::out_of_range("step tokens"))? - 1;
            let mut keys = sequence.length;
            if row == padded {
                for position in sequence.length..sequence.length + padding {
                    flat.push(0);
                    positions.push(position.min(self.capacity - 1));
                    write_pages.push(self.pool);
                    write_slots.push(position % self.page_size);
                }

                keys += padding;
            }

            visible.push(keys);
            let end = i64::try_from(flat.len()).map_err(|_| Error::out_of_range("step tokens"))?;
            query_offsets.push(end);
            last_rows.push(last);
        }

        // Unused trailing entries keep the table's shape, so one compiled step serves every length.
        let mut pages = Vec::new();
        let mut page_offsets = vec![0_i64];
        let mut last_page_lengths = Vec::with_capacity(self.sequences.len());
        for (sequence, &seen) in self.sequences.iter().zip(&visible) {
            if seen == 0 {
                // An idle row reads the scratch page, since every attention kernel takes a key.
                pages.push(self.pool);
                last_page_lengths.push(1);
            } else {
                let used = usize::try_from((seen - 1) / self.page_size + 1)
                    .map_err(|_| Error::out_of_range("page table"))?;
                // Padding's keys past the sequence's own pages read the scratch page.
                let held = used.min(sequence.pages.len());
                pages.extend_from_slice(&sequence.pages[..held]);
                pages.extend(iter::repeat_n(self.pool, used - held));
                last_page_lengths.push((seen - 1) % self.page_size + 1);
            }

            page_offsets
                .push(i64::try_from(pages.len()).map_err(|_| Error::out_of_range("page table"))?);
        }

        pages.resize(
            usize::try_from(self.table_length()?).map_err(|_| Error::out_of_range("page table"))?,
            0,
        );
        let vector = |values: Vec<i64>| {
            let length =
                i64::try_from(values.len()).map_err(|_| Error::out_of_range("step tensor"))?;
            Tensor::from_vec(device, Shape::create([length])?, values)
        };
        Ok(DecodeStep {
            tokens: vector(flat)?,
            positions: vector(positions)?,
            write_pages: vector(write_pages)?,
            write_slots: vector(write_slots)?,
            query_offsets: vector(query_offsets)?,
            page_offsets: vector(page_offsets)?,
            pages: vector(pages)?,
            last_page_lengths: vector(last_page_lengths)?,
            last_rows: vector(last_rows)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integers(tensor: &Tensor) -> Vec<i64> {
        tensor.to_vec::<i64>().unwrap()
    }

    #[test]
    fn test_page_tables_give_each_sequence_its_own_pages_and_lengths() {
        let mut table = PageTable::create(2, 5, 2).unwrap();
        // Three pages each, and the scratch page.
        assert_eq!(table.page_count(), 7);
        let step = table.step(Device::Cpu, &[&[1, 2, 3], &[4]]).unwrap();
        assert_eq!(integers(step.tokens()), [1, 2, 3, 4]);
        assert_eq!(integers(step.positions()), [0, 1, 2, 0]);
        assert_eq!(integers(step.last_rows()), [2, 3]);
        let paging = step.paging();
        assert_eq!(integers(paging.query_offsets), [0, 3, 4]);
        assert_eq!(integers(paging.page_offsets), [0, 2, 3]);
        assert_eq!(integers(paging.last_page_lengths), [1, 1]);
        // Pages are unique to their sequence, and the table's trailing entries pad it.
        let pages = integers(paging.pages);
        assert_eq!(pages.len(), 8);
        assert_ne!(pages[0], pages[1]);
        assert_ne!(pages[1], pages[2]);
        assert_eq!(integers(&step.write_slots), [0, 1, 0, 0]);
        assert_eq!(
            integers(&step.write_pages),
            [pages[0], pages[0], pages[1], pages[2]]
        );
    }

    #[test]
    fn test_a_padded_step_adds_rows_that_only_padding_reads() {
        let mut table = PageTable::create(2, 5, 2).unwrap().with_padding(6).unwrap();
        let scratch = table.layout().unwrap().scratch_page;
        // Three pages each, the idle rows' scratch page, and three for padding.
        assert_eq!(table.table_length().unwrap(), 6 + 2 + 3);
        let step = table
            .step_padded(Device::Cpu, &[&[1, 2, 3], &[4]], 8)
            .unwrap();
        assert_eq!(integers(step.tokens()), [1, 2, 3, 4, 0, 0, 0, 0]);
        // Padding's positions stop at the last the tables hold.
        assert_eq!(integers(step.positions()), [0, 1, 2, 0, 1, 2, 3, 4]);
        assert_eq!(integers(step.last_rows()), [2, 3]);
        let paging = step.paging();
        assert_eq!(integers(paging.query_offsets), [0, 3, 8]);
        // The second sequence's five keys take its own page and two scratch pages.
        assert_eq!(integers(paging.page_offsets), [0, 2, 5]);
        let pages = integers(paging.pages);
        assert_eq!(pages[3..5], [scratch, scratch]);
        assert_eq!(integers(paging.last_page_lengths), [1, 1]);
        assert_eq!(integers(&step.write_pages)[4..], [scratch; 4]);
        assert_eq!(integers(&step.write_slots)[4..], [1, 0, 1, 0]);

        // The padding took no capacity, so the next step continues each sequence.
        let step = table.step(Device::Cpu, &[&[5], &[6]]).unwrap();
        assert_eq!(integers(step.positions()), [3, 1]);
    }

    #[test]
    fn test_padding_stays_within_what_the_table_allows() {
        let mut table = PageTable::create(1, 8, 2).unwrap().with_padding(2).unwrap();
        assert!(table.step_padded(Device::Cpu, &[&[1, 2]], 1).is_err());
        assert!(table.step_padded(Device::Cpu, &[&[1, 2]], 5).is_err());
        let step = table.step_padded(Device::Cpu, &[&[1, 2]], 4).unwrap();
        assert_eq!(integers(step.last_rows()), [1]);
        assert_eq!(integers(step.paging().query_offsets), [0, 4]);
    }

    #[test]
    fn test_page_tables_refuse_tokens_past_capacity_and_start_over_after_reset() {
        let mut table = PageTable::create(1, 2, 2).unwrap();
        table.step(Device::Cpu, &[&[1, 2]]).unwrap();
        assert!(table.step(Device::Cpu, &[&[3]]).is_err());
        table.reset();
        let step = table.step(Device::Cpu, &[&[3]]).unwrap();
        assert_eq!(integers(step.positions()), [0]);
    }

    #[test]
    fn test_page_tables_refuse_steps_without_a_slice_per_sequence_or_any_token() {
        let mut table = PageTable::create(2, 2, 2).unwrap();
        assert!(table.step(Device::Cpu, &[&[1]]).is_err());
        assert!(table.step(Device::Cpu, &[&[], &[]]).is_err());
    }

    // A later sequence over capacity must not leave an earlier one ahead of its cache.
    #[test]
    fn test_a_refused_step_leaves_every_sequence_where_it_was() {
        let mut table = PageTable::create(2, 2, 2).unwrap();
        table.step(Device::Cpu, &[&[1], &[2, 3]]).unwrap();
        assert!(table.step(Device::Cpu, &[&[4], &[5]]).is_err());
        let step = table.step(Device::Cpu, &[&[4], &[]]).unwrap();
        assert_eq!(integers(step.positions())[0], 1);
    }

    #[test]
    fn test_a_held_sequence_takes_one_row_and_no_capacity() {
        let mut table = PageTable::create(2, 3, 2).unwrap();
        table.step(Device::Cpu, &[&[1, 2], &[3]]).unwrap();
        for _ in 0..4 {
            let step = table.step(Device::Cpu, &[&[], &[4]]);
            // The second sequence runs out, and the held one never does.
            if let Ok(step) = step {
                assert_eq!(integers(step.last_rows()), [0, 1]);
                assert_eq!(integers(step.positions())[0], 2);
                assert_eq!(integers(step.paging().last_page_lengths)[0], 2);
                // It writes to the scratch page, and takes nothing.
                assert_eq!(integers(&step.write_pages)[0], 4);
            }
        }
        // It resumes at the slot it was holding.
        let step = table.step(Device::Cpu, &[&[9], &[]]).unwrap();
        assert_eq!(integers(step.positions()), [2, 2]);
    }

    // A hold at a page boundary must not take a page the running sequences need.
    #[test]
    fn test_holding_takes_no_page() {
        let mut table = PageTable::create_shared(2, 8, 2, 4).unwrap();
        table.step(Device::Cpu, &[&[1, 2], &[3]]).unwrap();
        assert_eq!(table.free_pages(), 0);
        let step = table.step(Device::Cpu, &[&[], &[4]]).unwrap();
        assert_eq!(table.free_pages(), 0);
        assert_eq!(integers(&step.write_pages)[0], 2);
    }

    #[test]
    fn test_a_released_sequence_starts_over_and_its_pages_serve_another() {
        let mut table = PageTable::create_shared(2, 8, 2, 4).unwrap();
        let first = table.step(Device::Cpu, &[&[1, 2, 3], &[]]).unwrap();
        let taken = integers(first.paging().pages)[..2].to_vec();
        assert_eq!(table.free_pages(), 0);
        // Nothing is left for the other row until the first lets go.
        assert!(table.step(Device::Cpu, &[&[], &[5]]).is_err());
        table.release(0).unwrap();
        assert_eq!(table.free_pages(), 2);
        let step = table.step(Device::Cpu, &[&[], &[5]]).unwrap();
        assert_eq!(integers(step.positions()), [0, 0]);
        assert!(taken.contains(&integers(&step.write_pages)[1]));
        // The idle row reads the scratch page, and the running one its own.
        let paging = step.paging();
        assert_eq!(integers(paging.pages)[0], 2);
        assert_eq!(integers(paging.page_offsets), [0, 1, 2]);
        assert_eq!(integers(paging.last_page_lengths), [1, 1]);
        assert!(table.release(2).is_err());
    }

    // A later sequence that overruns the pool must not leave an earlier one holding pages it never wrote.
    #[test]
    fn test_a_step_that_overruns_the_pool_changes_nothing() {
        let mut table = PageTable::create_shared(2, 8, 2, 2).unwrap();
        assert!(table.step(Device::Cpu, &[&[1, 2], &[3]]).is_err());
        assert_eq!(table.free_pages(), 1);
        let step = table.step(Device::Cpu, &[&[1, 2], &[]]).unwrap();
        assert_eq!(integers(step.positions()), [0, 1, 0]);
    }

    #[test]
    fn test_a_full_sequence_holds_in_its_last_slot() {
        let mut table = PageTable::create(2, 2, 2).unwrap();
        table.step(Device::Cpu, &[&[1, 2], &[3]]).unwrap();
        let step = table.step(Device::Cpu, &[&[], &[4]]).unwrap();
        assert_eq!(integers(step.positions()), [1, 1]);
        assert_eq!(integers(step.paging().last_page_lengths), [2, 2]);
    }
}
