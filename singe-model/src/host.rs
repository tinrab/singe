pub(crate) fn rows<T: Clone>(values: &[T], width: usize) -> Vec<Vec<T>> {
    values.chunks(width).map(<[T]>::to_vec).collect()
}

pub(crate) fn best_first<T>(items: &mut Vec<T>, count: usize, score: impl Fn(&T) -> f32) {
    items.sort_by(|left, right| score(right).total_cmp(&score(left)));
    items.truncate(count);
}
