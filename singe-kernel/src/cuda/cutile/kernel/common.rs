#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn fill_f16(out: *mut f16, value: f16, len: i32) {
        fill_impl(out, value, len);
    }

    #[cutile::entry()]
    pub unsafe fn fill_f32(out: *mut f32, value: f32, len: i32) {
        fill_impl(out, value, len);
    }

    #[cutile::entry()]
    pub unsafe fn fill_f64(out: *mut f64, value: f64, len: i32) {
        fill_impl(out, value, len);
    }

    #[cutile::entry()]
    pub unsafe fn fill_u8(out: *mut u8, value: u8, len: i32) {
        fill_impl(out, value, len);
    }

    #[cutile::entry()]
    pub unsafe fn fill_i8(out: *mut i8, value: i8, len: i32) {
        fill_impl(out, value, len);
    }

    #[cutile::entry()]
    pub unsafe fn fill_u32(out: *mut u32, value: u32, len: i32) {
        fill_impl(out, value, len);
    }

    #[cutile::entry()]
    pub unsafe fn fill_u64(out: *mut u64, value: u64, len: i32) {
        fill_impl(out, value, len);
    }

    #[cutile::entry()]
    pub unsafe fn fill_i32(out: *mut i32, value: i32, len: i32) {
        fill_impl(out, value, len);
    }

    #[cutile::entry()]
    pub unsafe fn fill_i64(out: *mut i64, value: i64, len: i32) {
        fill_impl(out, value, len);
    }

    #[cutile::entry()]
    pub unsafe fn neg_i8(out: *mut i8, input: *mut i8, len: i32) {
        neg_impl(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn neg_i32(out: *mut i32, input: *mut i32, len: i32) {
        neg_impl(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn neg_i64(out: *mut i64, input: *mut i64, len: i32) {
        neg_impl(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn abs_i8(out: *mut i8, input: *mut i8, len: i32) {
        abs_signed_impl(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn abs_i32(out: *mut i32, input: *mut i32, len: i32) {
        abs_signed_impl(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn abs_i64(out: *mut i64, input: *mut i64, len: i32) {
        abs_signed_impl(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn sign_i8(out: *mut i8, input: *mut i8, len: i32) {
        sign_signed_impl(out, input, 0i8, 1i8, -1i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn sign_i32(out: *mut i32, input: *mut i32, len: i32) {
        sign_signed_impl(out, input, 0i32, 1i32, -1i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn sign_i64(out: *mut i64, input: *mut i64, len: i32) {
        sign_signed_impl(out, input, 0i64, 1i64, -1i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn sign_u8(out: *mut u8, input: *mut u8, len: i32) {
        sign_unsigned_impl(out, input, 0u8, 1u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn sign_u32(out: *mut u32, input: *mut u32, len: i32) {
        sign_unsigned_impl(out, input, 0u32, 1u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn sign_u64(out: *mut u64, input: *mut u64, len: i32) {
        sign_unsigned_impl(out, input, 0u64, 1u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_not_u8(out: *mut u8, input: *mut u8, len: i32) {
        bitwise_not_impl(out, input, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_not_i8(out: *mut i8, input: *mut i8, len: i32) {
        bitwise_not_impl(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_not_u32(out: *mut u32, input: *mut u32, len: i32) {
        bitwise_not_impl(out, input, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_not_i32(out: *mut i32, input: *mut i32, len: i32) {
        bitwise_not_impl(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_not_u64(out: *mut u64, input: *mut u64, len: i32) {
        bitwise_not_impl(out, input, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_not_i64(out: *mut i64, input: *mut i64, len: i32) {
        bitwise_not_impl(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u8_i8(out: *mut i8, input: *mut u8, len: i32) {
        cast_int_impl(out, input, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u8_u8(out: *mut u8, input: *mut u8, len: i32) {
        cast_int_impl(out, input, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u8_u32(out: *mut u32, input: *mut u8, len: i32) {
        cast_int_impl(out, input, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u8_i32(out: *mut i32, input: *mut u8, len: i32) {
        cast_int_impl(out, input, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u8_u64(out: *mut u64, input: *mut u8, len: i32) {
        cast_int_impl(out, input, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u8_i64(out: *mut i64, input: *mut u8, len: i32) {
        cast_int_impl(out, input, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_bool_u8(out: *mut u8, input: *mut u8, len: i32) {
        cast_bool_int_impl(out, input, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_bool_i8(out: *mut i8, input: *mut u8, len: i32) {
        cast_bool_int_impl(out, input, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_bool_u32(out: *mut u32, input: *mut u8, len: i32) {
        cast_bool_int_impl(out, input, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_bool_i32(out: *mut i32, input: *mut u8, len: i32) {
        cast_bool_int_impl(out, input, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_bool_u64(out: *mut u64, input: *mut u8, len: i32) {
        cast_bool_int_impl(out, input, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_bool_i64(out: *mut i64, input: *mut u8, len: i32) {
        cast_bool_int_impl(out, input, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i8_u8(out: *mut u8, input: *mut i8, len: i32) {
        cast_int_impl(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i8_i8(out: *mut i8, input: *mut i8, len: i32) {
        cast_int_impl(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i8_u32(out: *mut u32, input: *mut i8, len: i32) {
        cast_int_impl(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i8_i32(out: *mut i32, input: *mut i8, len: i32) {
        cast_int_impl(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i8_u64(out: *mut u64, input: *mut i8, len: i32) {
        cast_int_impl(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i8_i64(out: *mut i64, input: *mut i8, len: i32) {
        cast_int_impl(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u32_u8(out: *mut u8, input: *mut u32, len: i32) {
        cast_int_impl(out, input, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u32_u32(out: *mut u32, input: *mut u32, len: i32) {
        cast_int_impl(out, input, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u32_i8(out: *mut i8, input: *mut u32, len: i32) {
        cast_int_impl(out, input, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u32_i32(out: *mut i32, input: *mut u32, len: i32) {
        cast_int_impl(out, input, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u32_u64(out: *mut u64, input: *mut u32, len: i32) {
        cast_int_impl(out, input, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u32_i64(out: *mut i64, input: *mut u32, len: i32) {
        cast_int_impl(out, input, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i32_u8(out: *mut u8, input: *mut i32, len: i32) {
        cast_int_impl(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i32_i32(out: *mut i32, input: *mut i32, len: i32) {
        cast_int_impl(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i32_i8(out: *mut i8, input: *mut i32, len: i32) {
        cast_int_impl(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i32_u32(out: *mut u32, input: *mut i32, len: i32) {
        cast_int_impl(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i32_u64(out: *mut u64, input: *mut i32, len: i32) {
        cast_int_impl(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i32_i64(out: *mut i64, input: *mut i32, len: i32) {
        cast_int_impl(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u64_u8(out: *mut u8, input: *mut u64, len: i32) {
        cast_int_impl(out, input, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u64_u64(out: *mut u64, input: *mut u64, len: i32) {
        cast_int_impl(out, input, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u64_i8(out: *mut i8, input: *mut u64, len: i32) {
        cast_int_impl(out, input, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u64_u32(out: *mut u32, input: *mut u64, len: i32) {
        cast_int_impl(out, input, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u64_i32(out: *mut i32, input: *mut u64, len: i32) {
        cast_int_impl(out, input, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_u64_i64(out: *mut i64, input: *mut u64, len: i32) {
        cast_int_impl(out, input, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i64_u8(out: *mut u8, input: *mut i64, len: i32) {
        cast_int_impl(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i64_i64(out: *mut i64, input: *mut i64, len: i32) {
        cast_int_impl(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i64_i8(out: *mut i8, input: *mut i64, len: i32) {
        cast_int_impl(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i64_u32(out: *mut u32, input: *mut i64, len: i32) {
        cast_int_impl(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i64_i32(out: *mut i32, input: *mut i64, len: i32) {
        cast_int_impl(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn cast_i64_u64(out: *mut u64, input: *mut i64, len: i32) {
        cast_int_impl(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_zero_bool_u8(out: *mut u8, input: *mut u8, len: i32) {
        compare_scalar_impl::<0, u8>(out, input, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_nonzero_bool_u8(out: *mut u8, input: *mut u8, len: i32) {
        compare_scalar_impl::<1, u8>(out, input, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_positive_bool_u8(out: *mut u8, input: *mut u8, len: i32) {
        compare_scalar_impl::<4, u8>(out, input, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_zero_bool_i8(out: *mut u8, input: *mut i8, len: i32) {
        compare_scalar_impl::<0, i8>(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_nonzero_bool_i8(out: *mut u8, input: *mut i8, len: i32) {
        compare_scalar_impl::<1, i8>(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_positive_bool_i8(out: *mut u8, input: *mut i8, len: i32) {
        compare_scalar_impl::<4, i8>(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_negative_bool_i8(out: *mut u8, input: *mut i8, len: i32) {
        compare_scalar_impl::<2, i8>(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_zero_bool_u32(out: *mut u8, input: *mut u32, len: i32) {
        compare_scalar_impl::<0, u32>(out, input, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_nonzero_bool_u32(out: *mut u8, input: *mut u32, len: i32) {
        compare_scalar_impl::<1, u32>(out, input, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_positive_bool_u32(out: *mut u8, input: *mut u32, len: i32) {
        compare_scalar_impl::<4, u32>(out, input, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_zero_bool_i32(out: *mut u8, input: *mut i32, len: i32) {
        compare_scalar_impl::<0, i32>(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_nonzero_bool_i32(out: *mut u8, input: *mut i32, len: i32) {
        compare_scalar_impl::<1, i32>(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_positive_bool_i32(out: *mut u8, input: *mut i32, len: i32) {
        compare_scalar_impl::<4, i32>(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_negative_bool_i32(out: *mut u8, input: *mut i32, len: i32) {
        compare_scalar_impl::<2, i32>(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_zero_bool_u64(out: *mut u8, input: *mut u64, len: i32) {
        compare_scalar_impl::<0, u64>(out, input, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_nonzero_bool_u64(out: *mut u8, input: *mut u64, len: i32) {
        compare_scalar_impl::<1, u64>(out, input, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_positive_bool_u64(out: *mut u8, input: *mut u64, len: i32) {
        compare_scalar_impl::<4, u64>(out, input, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_zero_bool_i64(out: *mut u8, input: *mut i64, len: i32) {
        compare_scalar_impl::<0, i64>(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_nonzero_bool_i64(out: *mut u8, input: *mut i64, len: i32) {
        compare_scalar_impl::<1, i64>(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_positive_bool_i64(out: *mut u8, input: *mut i64, len: i32) {
        compare_scalar_impl::<4, i64>(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn is_negative_bool_i64(out: *mut u8, input: *mut i64, len: i32) {
        compare_scalar_impl::<2, i64>(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_u8(out: *mut u8, input: *mut u8, indices: *mut u32, len: i32) {
        gather_impl(out, input, indices, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_i8(out: *mut i8, input: *mut i8, indices: *mut u32, len: i32) {
        gather_impl(out, input, indices, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_u32(out: *mut u32, input: *mut u32, indices: *mut u32, len: i32) {
        gather_impl(out, input, indices, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_i32(out: *mut i32, input: *mut i32, indices: *mut u32, len: i32) {
        gather_impl(out, input, indices, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_u64(out: *mut u64, input: *mut u64, indices: *mut u32, len: i32) {
        gather_impl(out, input, indices, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_i64(out: *mut i64, input: *mut i64, indices: *mut u32, len: i32) {
        gather_impl(out, input, indices, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_rows_u8(
        out: *mut u8,
        input: *mut u8,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        gather_rows_impl(out, input, row_indices, 0u8, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_rows_i8(
        out: *mut i8,
        input: *mut i8,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        gather_rows_impl(out, input, row_indices, 0i8, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_rows_u32(
        out: *mut u32,
        input: *mut u32,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        gather_rows_impl(out, input, row_indices, 0u32, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_rows_i32(
        out: *mut i32,
        input: *mut i32,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        gather_rows_impl(out, input, row_indices, 0i32, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_rows_u64(
        out: *mut u64,
        input: *mut u64,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        gather_rows_impl(out, input, row_indices, 0u64, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_rows_i64(
        out: *mut i64,
        input: *mut i64,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        gather_rows_impl(out, input, row_indices, 0i64, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn gather_row_u8(out: *mut u8, input: *mut u8, row_index: i32, cols: i32) {
        gather_row_impl(out, input, row_index, 0u8, cols);
    }

    #[cutile::entry()]
    pub unsafe fn gather_row_i8(out: *mut i8, input: *mut i8, row_index: i32, cols: i32) {
        gather_row_impl(out, input, row_index, 0i8, cols);
    }

    #[cutile::entry()]
    pub unsafe fn gather_row_u32(out: *mut u32, input: *mut u32, row_index: i32, cols: i32) {
        gather_row_impl(out, input, row_index, 0u32, cols);
    }

    #[cutile::entry()]
    pub unsafe fn gather_row_i32(out: *mut i32, input: *mut i32, row_index: i32, cols: i32) {
        gather_row_impl(out, input, row_index, 0i32, cols);
    }

    #[cutile::entry()]
    pub unsafe fn gather_row_u64(out: *mut u64, input: *mut u64, row_index: i32, cols: i32) {
        gather_row_impl(out, input, row_index, 0u64, cols);
    }

    #[cutile::entry()]
    pub unsafe fn gather_row_i64(out: *mut i64, input: *mut i64, row_index: i32, cols: i32) {
        gather_row_impl(out, input, row_index, 0i64, cols);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_u8(out: *mut u8, input: *mut u8, indices: *mut u32, len: i32) {
        scatter_impl(out, input, indices, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_i8(out: *mut i8, input: *mut i8, indices: *mut u32, len: i32) {
        scatter_impl(out, input, indices, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_u32(out: *mut u32, input: *mut u32, indices: *mut u32, len: i32) {
        scatter_impl(out, input, indices, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_i32(out: *mut i32, input: *mut i32, indices: *mut u32, len: i32) {
        scatter_impl(out, input, indices, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_u64(out: *mut u64, input: *mut u64, indices: *mut u32, len: i32) {
        scatter_impl(out, input, indices, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_i64(out: *mut i64, input: *mut i64, indices: *mut u32, len: i32) {
        scatter_impl(out, input, indices, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_rows_u8(
        out: *mut u8,
        input: *mut u8,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        scatter_rows_impl(out, input, row_indices, 0u8, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_rows_i8(
        out: *mut i8,
        input: *mut i8,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        scatter_rows_impl(out, input, row_indices, 0i8, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_rows_u32(
        out: *mut u32,
        input: *mut u32,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        scatter_rows_impl(out, input, row_indices, 0u32, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_rows_i32(
        out: *mut i32,
        input: *mut i32,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        scatter_rows_impl(out, input, row_indices, 0i32, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_rows_u64(
        out: *mut u64,
        input: *mut u64,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        scatter_rows_impl(out, input, row_indices, 0u64, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn scatter_rows_i64(
        out: *mut i64,
        input: *mut i64,
        row_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        scatter_rows_impl(out, input, row_indices, 0i64, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn copy_indexed_rows_u8(
        out: *mut u8,
        input: *mut u8,
        source_indices: *mut u32,
        output_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        copy_indexed_rows_impl(out, input, source_indices, output_indices, 0u8, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn copy_indexed_rows_i8(
        out: *mut i8,
        input: *mut i8,
        source_indices: *mut u32,
        output_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        copy_indexed_rows_impl(out, input, source_indices, output_indices, 0i8, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn copy_indexed_rows_u32(
        out: *mut u32,
        input: *mut u32,
        source_indices: *mut u32,
        output_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        copy_indexed_rows_impl(out, input, source_indices, output_indices, 0u32, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn copy_indexed_rows_i32(
        out: *mut i32,
        input: *mut i32,
        source_indices: *mut u32,
        output_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        copy_indexed_rows_impl(out, input, source_indices, output_indices, 0i32, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn copy_indexed_rows_u64(
        out: *mut u64,
        input: *mut u64,
        source_indices: *mut u32,
        output_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        copy_indexed_rows_impl(out, input, source_indices, output_indices, 0u64, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn copy_indexed_rows_i64(
        out: *mut i64,
        input: *mut i64,
        source_indices: *mut u32,
        output_indices: *mut u32,
        cols: i32,
        len: i32,
    ) {
        copy_indexed_rows_impl(out, input, source_indices, output_indices, 0i64, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn where_bool_u8(
        out: *mut u8,
        condition: *mut u8,
        x: *mut u8,
        y: *mut u8,
        len: i32,
    ) {
        where_bool_impl(out, condition, x, y, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn where_bool_i8(
        out: *mut i8,
        condition: *mut u8,
        x: *mut i8,
        y: *mut i8,
        len: i32,
    ) {
        where_bool_impl(out, condition, x, y, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn where_bool_u32(
        out: *mut u32,
        condition: *mut u8,
        x: *mut u32,
        y: *mut u32,
        len: i32,
    ) {
        where_bool_impl(out, condition, x, y, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn where_bool_i32(
        out: *mut i32,
        condition: *mut u8,
        x: *mut i32,
        y: *mut i32,
        len: i32,
    ) {
        where_bool_impl(out, condition, x, y, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn where_bool_u64(
        out: *mut u64,
        condition: *mut u8,
        x: *mut u64,
        y: *mut u64,
        len: i32,
    ) {
        where_bool_impl(out, condition, x, y, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn where_bool_i64(
        out: *mut i64,
        condition: *mut u8,
        x: *mut i64,
        y: *mut i64,
        len: i32,
    ) {
        where_bool_impl(out, condition, x, y, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn masked_fill_u8(out: *mut u8, input: *mut u8, mask: *mut u8, value: u8, len: i32) {
        masked_fill_impl(out, input, mask, value, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn masked_fill_i8(out: *mut i8, input: *mut i8, mask: *mut u8, value: i8, len: i32) {
        masked_fill_impl(out, input, mask, value, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn masked_fill_u32(
        out: *mut u32,
        input: *mut u32,
        mask: *mut u8,
        value: u32,
        len: i32,
    ) {
        masked_fill_impl(out, input, mask, value, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn masked_fill_i32(
        out: *mut i32,
        input: *mut i32,
        mask: *mut u8,
        value: i32,
        len: i32,
    ) {
        masked_fill_impl(out, input, mask, value, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn masked_fill_u64(
        out: *mut u64,
        input: *mut u64,
        mask: *mut u8,
        value: u64,
        len: i32,
    ) {
        masked_fill_impl(out, input, mask, value, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn masked_fill_i64(
        out: *mut i64,
        input: *mut i64,
        mask: *mut u8,
        value: i64,
        len: i32,
    ) {
        masked_fill_impl(out, input, mask, value, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn clamp_u8(out: *mut u8, input: *mut u8, min: u8, max: u8, len: i32) {
        clamp_impl(out, input, min, max, len);
    }

    #[cutile::entry()]
    pub unsafe fn clamp_i8(out: *mut i8, input: *mut i8, min: i8, max: i8, len: i32) {
        clamp_impl(out, input, min, max, len);
    }

    #[cutile::entry()]
    pub unsafe fn clamp_u32(out: *mut u32, input: *mut u32, min: u32, max: u32, len: i32) {
        clamp_impl(out, input, min, max, len);
    }

    #[cutile::entry()]
    pub unsafe fn clamp_i32(out: *mut i32, input: *mut i32, min: i32, max: i32, len: i32) {
        clamp_impl(out, input, min, max, len);
    }

    #[cutile::entry()]
    pub unsafe fn clamp_u64(out: *mut u64, input: *mut u64, min: u64, max: u64, len: i32) {
        clamp_impl(out, input, min, max, len);
    }

    #[cutile::entry()]
    pub unsafe fn clamp_i64(out: *mut i64, input: *mut i64, min: i64, max: i64, len: i32) {
        clamp_impl(out, input, min, max, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        arithmetic_impl::<0, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        arithmetic_impl::<1, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        arithmetic_impl::<2, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        arithmetic_impl::<0, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        arithmetic_impl::<1, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        arithmetic_impl::<2, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        arithmetic_impl::<0, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        arithmetic_impl::<1, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        arithmetic_impl::<2, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        arithmetic_impl::<0, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        arithmetic_impl::<1, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        arithmetic_impl::<2, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        arithmetic_impl::<0, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        arithmetic_impl::<1, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        arithmetic_impl::<2, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        arithmetic_impl::<0, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        arithmetic_impl::<1, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        arithmetic_impl::<2, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        arithmetic_impl::<3, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        arithmetic_impl::<4, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        arithmetic_impl::<3, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        arithmetic_impl::<4, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        arithmetic_impl::<3, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        arithmetic_impl::<4, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        arithmetic_impl::<3, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        arithmetic_impl::<4, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        arithmetic_impl::<3, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        arithmetic_impl::<4, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        arithmetic_impl::<3, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        arithmetic_impl::<4, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        bitwise_impl::<0, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        bitwise_impl::<1, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        bitwise_impl::<2, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        bitwise_impl::<0, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        bitwise_impl::<1, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        bitwise_impl::<2, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        bitwise_impl::<0, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        bitwise_impl::<1, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        bitwise_impl::<2, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        bitwise_impl::<0, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        bitwise_impl::<1, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        bitwise_impl::<2, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        bitwise_impl::<0, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        bitwise_impl::<1, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        bitwise_impl::<2, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        bitwise_impl::<0, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        bitwise_impl::<1, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        bitwise_impl::<2, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        shift_impl::<0, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        shift_impl::<1, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        shift_impl::<0, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        shift_impl::<1, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        shift_impl::<0, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        shift_impl::<1, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        shift_impl::<0, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        shift_impl::<1, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        shift_impl::<0, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        shift_impl::<1, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        shift_impl::<0, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        shift_impl::<1, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_bool_u32(out: *mut u8, lhs: *mut u32, rhs: *mut u32, len: i32) {
        compare_u32_impl::<0>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_bool_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        compare_u8_impl::<0>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_bool_i8(out: *mut u8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        compare_i8_impl::<0>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_bool_i32(out: *mut u8, lhs: *mut i32, rhs: *mut i32, len: i32) {
        compare_impl::<0, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_bool_u64(out: *mut u8, lhs: *mut u64, rhs: *mut u64, len: i32) {
        compare_impl::<0, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_bool_i64(out: *mut u8, lhs: *mut i64, rhs: *mut i64, len: i32) {
        compare_impl::<0, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_bool_u32(out: *mut u8, lhs: *mut u32, rhs: *mut u32, len: i32) {
        compare_u32_impl::<1>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_bool_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        compare_u8_impl::<1>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_bool_i8(out: *mut u8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        compare_i8_impl::<1>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_bool_i32(out: *mut u8, lhs: *mut i32, rhs: *mut i32, len: i32) {
        compare_impl::<1, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_bool_u64(out: *mut u8, lhs: *mut u64, rhs: *mut u64, len: i32) {
        compare_impl::<1, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_bool_i64(out: *mut u8, lhs: *mut i64, rhs: *mut i64, len: i32) {
        compare_impl::<1, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_bool_u32(out: *mut u8, lhs: *mut u32, rhs: *mut u32, len: i32) {
        compare_u32_impl::<2>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_bool_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        compare_u8_impl::<2>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_bool_i8(out: *mut u8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        compare_i8_impl::<2>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_bool_i32(out: *mut u8, lhs: *mut i32, rhs: *mut i32, len: i32) {
        compare_impl::<2, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_bool_u64(out: *mut u8, lhs: *mut u64, rhs: *mut u64, len: i32) {
        compare_impl::<2, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_bool_i64(out: *mut u8, lhs: *mut i64, rhs: *mut i64, len: i32) {
        compare_impl::<2, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_bool_u32(out: *mut u8, lhs: *mut u32, rhs: *mut u32, len: i32) {
        compare_u32_impl::<3>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_bool_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        compare_u8_impl::<3>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_bool_i8(out: *mut u8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        compare_i8_impl::<3>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_bool_i32(out: *mut u8, lhs: *mut i32, rhs: *mut i32, len: i32) {
        compare_impl::<3, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_bool_u64(out: *mut u8, lhs: *mut u64, rhs: *mut u64, len: i32) {
        compare_impl::<3, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_bool_i64(out: *mut u8, lhs: *mut i64, rhs: *mut i64, len: i32) {
        compare_impl::<3, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_bool_u32(out: *mut u8, lhs: *mut u32, rhs: *mut u32, len: i32) {
        compare_u32_impl::<4>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_bool_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        compare_u8_impl::<4>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_bool_i8(out: *mut u8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        compare_i8_impl::<4>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_bool_i32(out: *mut u8, lhs: *mut i32, rhs: *mut i32, len: i32) {
        compare_impl::<4, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_bool_u64(out: *mut u8, lhs: *mut u64, rhs: *mut u64, len: i32) {
        compare_impl::<4, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_bool_i64(out: *mut u8, lhs: *mut i64, rhs: *mut i64, len: i32) {
        compare_impl::<4, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_bool_u32(out: *mut u8, lhs: *mut u32, rhs: *mut u32, len: i32) {
        compare_u32_impl::<5>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_bool_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        compare_u8_impl::<5>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_bool_i8(out: *mut u8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        compare_i8_impl::<5>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_bool_i32(out: *mut u8, lhs: *mut i32, rhs: *mut i32, len: i32) {
        compare_impl::<5, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_bool_u64(out: *mut u8, lhs: *mut u64, rhs: *mut u64, len: i32) {
        compare_impl::<5, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_bool_i64(out: *mut u8, lhs: *mut i64, rhs: *mut i64, len: i32) {
        compare_impl::<5, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_scalar_bool_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        compare_scalar_impl::<0, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_scalar_bool_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        compare_scalar_impl::<1, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_scalar_bool_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        compare_scalar_impl::<2, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_scalar_bool_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        compare_scalar_impl::<3, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_scalar_bool_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        compare_scalar_impl::<4, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_scalar_bool_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        compare_scalar_impl::<5, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_scalar_bool_i8(out: *mut u8, input: *mut i8, scalar: i8, len: i32) {
        compare_scalar_impl::<0, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_scalar_bool_i8(out: *mut u8, input: *mut i8, scalar: i8, len: i32) {
        compare_scalar_impl::<1, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_scalar_bool_i8(out: *mut u8, input: *mut i8, scalar: i8, len: i32) {
        compare_scalar_impl::<2, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_scalar_bool_i8(out: *mut u8, input: *mut i8, scalar: i8, len: i32) {
        compare_scalar_impl::<3, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_scalar_bool_i8(out: *mut u8, input: *mut i8, scalar: i8, len: i32) {
        compare_scalar_impl::<4, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_scalar_bool_i8(out: *mut u8, input: *mut i8, scalar: i8, len: i32) {
        compare_scalar_impl::<5, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_scalar_bool_u32(out: *mut u8, input: *mut u32, scalar: u32, len: i32) {
        compare_scalar_impl::<0, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_scalar_bool_u32(out: *mut u8, input: *mut u32, scalar: u32, len: i32) {
        compare_scalar_impl::<1, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_scalar_bool_u32(out: *mut u8, input: *mut u32, scalar: u32, len: i32) {
        compare_scalar_impl::<2, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_scalar_bool_u32(out: *mut u8, input: *mut u32, scalar: u32, len: i32) {
        compare_scalar_impl::<3, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_scalar_bool_u32(out: *mut u8, input: *mut u32, scalar: u32, len: i32) {
        compare_scalar_impl::<4, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_scalar_bool_u32(
        out: *mut u8,
        input: *mut u32,
        scalar: u32,
        len: i32,
    ) {
        compare_scalar_impl::<5, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_scalar_bool_i32(out: *mut u8, input: *mut i32, scalar: i32, len: i32) {
        compare_scalar_impl::<0, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_scalar_bool_i32(out: *mut u8, input: *mut i32, scalar: i32, len: i32) {
        compare_scalar_impl::<1, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_scalar_bool_i32(out: *mut u8, input: *mut i32, scalar: i32, len: i32) {
        compare_scalar_impl::<2, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_scalar_bool_i32(out: *mut u8, input: *mut i32, scalar: i32, len: i32) {
        compare_scalar_impl::<3, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_scalar_bool_i32(out: *mut u8, input: *mut i32, scalar: i32, len: i32) {
        compare_scalar_impl::<4, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_scalar_bool_i32(
        out: *mut u8,
        input: *mut i32,
        scalar: i32,
        len: i32,
    ) {
        compare_scalar_impl::<5, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_scalar_bool_u64(out: *mut u8, input: *mut u64, scalar: u64, len: i32) {
        compare_scalar_impl::<0, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_scalar_bool_u64(out: *mut u8, input: *mut u64, scalar: u64, len: i32) {
        compare_scalar_impl::<1, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_scalar_bool_u64(out: *mut u8, input: *mut u64, scalar: u64, len: i32) {
        compare_scalar_impl::<2, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_scalar_bool_u64(out: *mut u8, input: *mut u64, scalar: u64, len: i32) {
        compare_scalar_impl::<3, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_scalar_bool_u64(out: *mut u8, input: *mut u64, scalar: u64, len: i32) {
        compare_scalar_impl::<4, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_scalar_bool_u64(
        out: *mut u8,
        input: *mut u64,
        scalar: u64,
        len: i32,
    ) {
        compare_scalar_impl::<5, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn equal_scalar_bool_i64(out: *mut u8, input: *mut i64, scalar: i64, len: i32) {
        compare_scalar_impl::<0, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn not_equal_scalar_bool_i64(out: *mut u8, input: *mut i64, scalar: i64, len: i32) {
        compare_scalar_impl::<1, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_scalar_bool_i64(out: *mut u8, input: *mut i64, scalar: i64, len: i32) {
        compare_scalar_impl::<2, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn less_equal_scalar_bool_i64(out: *mut u8, input: *mut i64, scalar: i64, len: i32) {
        compare_scalar_impl::<3, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_scalar_bool_i64(out: *mut u8, input: *mut i64, scalar: i64, len: i32) {
        compare_scalar_impl::<4, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn greater_equal_scalar_bool_i64(
        out: *mut u8,
        input: *mut i64,
        scalar: i64,
        len: i32,
    ) {
        compare_scalar_impl::<5, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        arithmetic_scalar_impl::<0, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        arithmetic_scalar_impl::<1, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rsub_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        arithmetic_scalar_impl::<2, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        arithmetic_scalar_impl::<3, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        arithmetic_scalar_impl::<0, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        arithmetic_scalar_impl::<1, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rsub_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        arithmetic_scalar_impl::<2, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        arithmetic_scalar_impl::<3, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        arithmetic_scalar_impl::<0, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        arithmetic_scalar_impl::<1, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rsub_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        arithmetic_scalar_impl::<2, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        arithmetic_scalar_impl::<3, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        arithmetic_scalar_impl::<0, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        arithmetic_scalar_impl::<1, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rsub_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        arithmetic_scalar_impl::<2, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        arithmetic_scalar_impl::<3, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        arithmetic_scalar_impl::<0, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        arithmetic_scalar_impl::<1, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rsub_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        arithmetic_scalar_impl::<2, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        arithmetic_scalar_impl::<3, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn add_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        arithmetic_scalar_impl::<0, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn sub_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        arithmetic_scalar_impl::<1, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rsub_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        arithmetic_scalar_impl::<2, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn mul_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        arithmetic_scalar_impl::<3, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        arithmetic_scalar_impl::<4, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rdiv_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        arithmetic_scalar_impl::<5, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        arithmetic_scalar_impl::<6, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rmodulo_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        arithmetic_scalar_impl::<7, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        arithmetic_scalar_impl::<4, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rdiv_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        arithmetic_scalar_impl::<5, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        arithmetic_scalar_impl::<6, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rmodulo_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        arithmetic_scalar_impl::<7, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        arithmetic_scalar_impl::<4, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rdiv_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        arithmetic_scalar_impl::<5, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        arithmetic_scalar_impl::<6, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rmodulo_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        arithmetic_scalar_impl::<7, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        arithmetic_scalar_impl::<4, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rdiv_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        arithmetic_scalar_impl::<5, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        arithmetic_scalar_impl::<6, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rmodulo_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        arithmetic_scalar_impl::<7, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        arithmetic_scalar_impl::<4, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rdiv_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        arithmetic_scalar_impl::<5, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        arithmetic_scalar_impl::<6, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rmodulo_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        arithmetic_scalar_impl::<7, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn div_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        arithmetic_scalar_impl::<4, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rdiv_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        arithmetic_scalar_impl::<5, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn modulo_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        arithmetic_scalar_impl::<6, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn rmodulo_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        arithmetic_scalar_impl::<7, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        bitwise_scalar_impl::<0, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        bitwise_scalar_impl::<1, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        bitwise_scalar_impl::<2, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        bitwise_scalar_impl::<0, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        bitwise_scalar_impl::<1, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        bitwise_scalar_impl::<2, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        bitwise_scalar_impl::<0, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        bitwise_scalar_impl::<1, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        bitwise_scalar_impl::<2, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        bitwise_scalar_impl::<0, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        bitwise_scalar_impl::<1, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        bitwise_scalar_impl::<2, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        bitwise_scalar_impl::<0, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        bitwise_scalar_impl::<1, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        bitwise_scalar_impl::<2, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_and_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        bitwise_scalar_impl::<0, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_or_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        bitwise_scalar_impl::<1, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn bitwise_xor_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        bitwise_scalar_impl::<2, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        shift_scalar_impl::<0, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        shift_scalar_impl::<1, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        shift_scalar_impl::<0, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        shift_scalar_impl::<1, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        shift_scalar_impl::<0, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        shift_scalar_impl::<1, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        shift_scalar_impl::<0, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        shift_scalar_impl::<1, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        shift_scalar_impl::<0, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        shift_scalar_impl::<1, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_left_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        shift_scalar_impl::<0, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn shift_right_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        shift_scalar_impl::<1, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        minmax_impl::<0, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_u32(out: *mut u32, lhs: *mut u32, rhs: *mut u32, len: i32) {
        minmax_impl::<1, u32>(out, lhs, rhs, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        minmax_impl::<0, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        minmax_impl::<1, u8>(out, lhs, rhs, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        minmax_impl::<0, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_i8(out: *mut i8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        minmax_impl::<1, i8>(out, lhs, rhs, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        minmax_impl::<0, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_i32(out: *mut i32, lhs: *mut i32, rhs: *mut i32, len: i32) {
        minmax_impl::<1, i32>(out, lhs, rhs, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        minmax_impl::<0, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_u64(out: *mut u64, lhs: *mut u64, rhs: *mut u64, len: i32) {
        minmax_impl::<1, u64>(out, lhs, rhs, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        minmax_impl::<0, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_i64(out: *mut i64, lhs: *mut i64, rhs: *mut i64, len: i32) {
        minmax_impl::<1, i64>(out, lhs, rhs, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        minmax_scalar_impl::<0, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_scalar_u32(out: *mut u32, input: *mut u32, scalar: u32, len: i32) {
        minmax_scalar_impl::<1, u32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        minmax_scalar_impl::<0, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_scalar_u8(out: *mut u8, input: *mut u8, scalar: u8, len: i32) {
        minmax_scalar_impl::<1, u8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        minmax_scalar_impl::<0, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_scalar_i8(out: *mut i8, input: *mut i8, scalar: i8, len: i32) {
        minmax_scalar_impl::<1, i8>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        minmax_scalar_impl::<0, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_scalar_i32(out: *mut i32, input: *mut i32, scalar: i32, len: i32) {
        minmax_scalar_impl::<1, i32>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        minmax_scalar_impl::<0, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_scalar_u64(out: *mut u64, input: *mut u64, scalar: u64, len: i32) {
        minmax_scalar_impl::<1, u64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn min_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        minmax_scalar_impl::<0, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn max_scalar_i64(out: *mut i64, input: *mut i64, scalar: i64, len: i32) {
        minmax_scalar_impl::<1, i64>(out, input, scalar, len);
    }

    #[cutile::entry()]
    pub unsafe fn mask_not_u8(out: *mut u8, input: *mut u8, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, 0u8);
        let zero_u8 = constant(0u8, tile_shape);
        let one_u8 = constant(1u8, tile_shape);
        let keep = cmpi(input_tile, zero_u8, predicate::Equal);
        store_vector(out, offsets, select(keep, one_u8, zero_u8), output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn mask_and_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        mask_binary_impl::<0>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn mask_or_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        mask_binary_impl::<1>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn mask_xor_u8(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        mask_binary_impl::<2>(out, lhs, rhs, len);
    }

    #[cutile::entry()]
    pub unsafe fn tril_mask_u8(out: *mut u8, rows: i32, cols: i32, diagonal: i32, len: i32) {
        triangular_mask_impl::<0>(out, rows, cols, diagonal, len);
    }

    #[cutile::entry()]
    pub unsafe fn triu_mask_u8(out: *mut u8, rows: i32, cols: i32, diagonal: i32, len: i32) {
        triangular_mask_impl::<1>(out, rows, cols, diagonal, len);
    }

    #[cutile::entry()]
    pub unsafe fn causal_mask_fill_f32(
        out: *mut f32,
        input: *mut f32,
        fill_value: f32,
        seq_len: i32,
        len: i32,
    ) {
        causal_mask_fill_impl(out, input, fill_value, 0.0f32, seq_len, len);
    }

    #[cutile::entry()]
    pub unsafe fn causal_mask_zero_f32(out: *mut f32, input: *mut f32, seq_len: i32, len: i32) {
        causal_mask_fill_impl(out, input, 0.0f32, 0.0f32, seq_len, len);
    }

    #[cutile::entry()]
    pub unsafe fn causal_mask_fill_f16(
        out: *mut f16,
        input: *mut f16,
        fill_value: f16,
        seq_len: i32,
        len: i32,
    ) {
        causal_mask_fill_impl(out, input, fill_value, f16::from_f32(0.0), seq_len, len);
    }

    #[cutile::entry()]
    pub unsafe fn causal_mask_zero_f16(out: *mut f16, input: *mut f16, seq_len: i32, len: i32) {
        causal_mask_fill_impl(
            out,
            input,
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            seq_len,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn causal_mask_fill_bf16(
        out: *mut bf16,
        input: *mut bf16,
        fill_value: bf16,
        seq_len: i32,
        len: i32,
    ) {
        causal_mask_fill_impl(out, input, fill_value, bf16::from_f32(0.0), seq_len, len);
    }

    #[cutile::entry()]
    pub unsafe fn causal_mask_zero_bf16(out: *mut bf16, input: *mut bf16, seq_len: i32, len: i32) {
        causal_mask_fill_impl(
            out,
            input,
            bf16::from_f32(0.0),
            bf16::from_f32(0.0),
            seq_len,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn sequence_mask_u8(out: *mut u8, lengths: *mut u32, cols: i32, len: i32) {
        sequence_mask_impl(out, lengths, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn tril_sequence_mask_u8(
        out: *mut u8,
        lengths: *mut u32,
        rows: i32,
        cols: i32,
        diagonal: i32,
        len: i32,
    ) {
        triangular_sequence_mask_impl::<0>(out, lengths, rows, cols, diagonal, len);
    }

    #[cutile::entry()]
    pub unsafe fn triu_sequence_mask_u8(
        out: *mut u8,
        lengths: *mut u32,
        rows: i32,
        cols: i32,
        diagonal: i32,
        len: i32,
    ) {
        triangular_sequence_mask_impl::<1>(out, lengths, rows, cols, diagonal, len);
    }

    #[cutile::entry()]
    pub fn mask_any_u8_1(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<0, 1>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_any_u8_8(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<0, 8>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_any_u8_32(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<0, 32>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_any_u8_64(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<0, 64>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_any_u8_128(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<0, 128>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_any_u8_256(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<0, 256>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_any_u8_512(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<0, 512>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_any_u8_1024(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<0, 1024>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_all_u8_1(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<1, 1>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_all_u8_8(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<1, 8>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_all_u8_32(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<1, 32>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_all_u8_64(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<1, 64>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_all_u8_128(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<1, 128>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_all_u8_256(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<1, 256>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_all_u8_512(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<1, 512>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_all_u8_1024(out: &mut Tensor<u8, { [1, 1] }>, input: &Tensor<u8, { [-1, -1] }>) {
        mask_reduce_impl::<1, 1024>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_count_nonzero_u8_1(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        mask_count_nonzero_impl::<1>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_count_nonzero_u8_8(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        mask_count_nonzero_impl::<8>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_count_nonzero_u8_32(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        mask_count_nonzero_impl::<32>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_count_nonzero_u8_64(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        mask_count_nonzero_impl::<64>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_count_nonzero_u8_128(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        mask_count_nonzero_impl::<128>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_count_nonzero_u8_256(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        mask_count_nonzero_impl::<256>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_count_nonzero_u8_512(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        mask_count_nonzero_impl::<512>(out, input);
    }

    #[cutile::entry()]
    pub fn mask_count_nonzero_u8_1024(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        mask_count_nonzero_impl::<1024>(out, input);
    }

    #[cutile::entry()]
    pub fn reduce_sum_u8<const BN: i32>(
        out: &mut Tensor<u8, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        reduce_int_impl::<0, BN, u8>(out, input, 0u8, 0u8, u8::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_max_u8<const BN: i32>(
        out: &mut Tensor<u8, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        reduce_int_impl::<1, BN, u8>(out, input, 0u8, 0u8, u8::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_min_u8<const BN: i32>(
        out: &mut Tensor<u8, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        reduce_int_impl::<2, BN, u8>(out, input, 0u8, 0u8, u8::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_sum_i8<const BN: i32>(
        out: &mut Tensor<i8, { [1, 1] }>,
        input: &Tensor<i8, { [-1, -1] }>,
    ) {
        reduce_int_impl::<0, BN, i8>(out, input, 0i8, i8::MIN, i8::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_max_i8<const BN: i32>(
        out: &mut Tensor<i8, { [1, 1] }>,
        input: &Tensor<i8, { [-1, -1] }>,
    ) {
        reduce_int_impl::<1, BN, i8>(out, input, 0i8, i8::MIN, i8::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_min_i8<const BN: i32>(
        out: &mut Tensor<i8, { [1, 1] }>,
        input: &Tensor<i8, { [-1, -1] }>,
    ) {
        reduce_int_impl::<2, BN, i8>(out, input, 0i8, i8::MIN, i8::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_sum_u32<const BN: i32>(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u32, { [-1, -1] }>,
    ) {
        reduce_int_impl::<0, BN, u32>(out, input, 0u32, 0u32, u32::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_max_u32<const BN: i32>(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u32, { [-1, -1] }>,
    ) {
        reduce_int_impl::<1, BN, u32>(out, input, 0u32, 0u32, u32::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_min_u32<const BN: i32>(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u32, { [-1, -1] }>,
    ) {
        reduce_int_impl::<2, BN, u32>(out, input, 0u32, 0u32, u32::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_sum_i32<const BN: i32>(
        out: &mut Tensor<i32, { [1, 1] }>,
        input: &Tensor<i32, { [-1, -1] }>,
    ) {
        reduce_int_impl::<0, BN, i32>(out, input, 0i32, i32::MIN, i32::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_max_i32<const BN: i32>(
        out: &mut Tensor<i32, { [1, 1] }>,
        input: &Tensor<i32, { [-1, -1] }>,
    ) {
        reduce_int_impl::<1, BN, i32>(out, input, 0i32, i32::MIN, i32::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_min_i32<const BN: i32>(
        out: &mut Tensor<i32, { [1, 1] }>,
        input: &Tensor<i32, { [-1, -1] }>,
    ) {
        reduce_int_impl::<2, BN, i32>(out, input, 0i32, i32::MIN, i32::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_sum_u64<const BN: i32>(
        out: &mut Tensor<u64, { [1, 1] }>,
        input: &Tensor<u64, { [-1, -1] }>,
    ) {
        reduce_int_impl::<0, BN, u64>(out, input, 0u64, 0u64, u64::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_max_u64<const BN: i32>(
        out: &mut Tensor<u64, { [1, 1] }>,
        input: &Tensor<u64, { [-1, -1] }>,
    ) {
        reduce_int_impl::<1, BN, u64>(out, input, 0u64, 0u64, u64::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_min_u64<const BN: i32>(
        out: &mut Tensor<u64, { [1, 1] }>,
        input: &Tensor<u64, { [-1, -1] }>,
    ) {
        reduce_int_impl::<2, BN, u64>(out, input, 0u64, 0u64, u64::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_sum_i64<const BN: i32>(
        out: &mut Tensor<i64, { [1, 1] }>,
        input: &Tensor<i64, { [-1, -1] }>,
    ) {
        reduce_int_impl::<0, BN, i64>(out, input, 0i64, i64::MIN, i64::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_max_i64<const BN: i32>(
        out: &mut Tensor<i64, { [1, 1] }>,
        input: &Tensor<i64, { [-1, -1] }>,
    ) {
        reduce_int_impl::<1, BN, i64>(out, input, 0i64, i64::MIN, i64::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_min_i64<const BN: i32>(
        out: &mut Tensor<i64, { [1, 1] }>,
        input: &Tensor<i64, { [-1, -1] }>,
    ) {
        reduce_int_impl::<2, BN, i64>(out, input, 0i64, i64::MIN, i64::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmax_u8<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<0, BN, u8>(out, input, 0u8, u8::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmin_u8<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<1, BN, u8>(out, input, 0u8, u8::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmax_i8<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<i8, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<0, BN, i8>(out, input, i8::MIN, i8::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmin_i8<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<i8, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<1, BN, i8>(out, input, i8::MIN, i8::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmax_u32<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<u32, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<0, BN, u32>(out, input, 0u32, u32::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmin_u32<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<u32, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<1, BN, u32>(out, input, 0u32, u32::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmax_i32<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<i32, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<0, BN, i32>(out, input, i32::MIN, i32::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmin_i32<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<i32, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<1, BN, i32>(out, input, i32::MIN, i32::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmax_u64<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<u64, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<0, BN, u64>(out, input, 0u64, u64::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmin_u64<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<u64, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<1, BN, u64>(out, input, 0u64, u64::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmax_i64<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<i64, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<0, BN, i64>(out, input, i64::MIN, i64::MAX);
    }

    #[cutile::entry()]
    pub fn reduce_argmin_i64<const BN: i32>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<i64, { [-1, -1] }>,
    ) {
        arg_reduce_int_impl::<1, BN, i64>(out, input, i64::MIN, i64::MAX);
    }

    #[cutile::entry()]
    pub unsafe fn copy_u8(out: *mut u8, input: *mut u8, len: i32) {
        copy_impl(out, input, 0u8, len);
    }

    #[cutile::entry()]
    pub unsafe fn copy_i8(out: *mut i8, input: *mut i8, len: i32) {
        copy_impl(out, input, 0i8, len);
    }

    #[cutile::entry()]
    pub unsafe fn unpack_u4_u8(out: *mut u8, packed: *mut u8, len: i32, high_first: i32) {
        let (offsets, mask) = vector_offsets(len);
        let unpacked = unpack_u4_impl(packed, offsets, mask, high_first);
        store_vector(out, offsets, unpacked, mask);
    }

    #[cutile::entry()]
    pub unsafe fn unpack_i4_i8(out: *mut i8, packed: *mut u8, len: i32, high_first: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let unpacked = unpack_u4_impl(packed, offsets, mask, high_first);
        let sign_bit = broadcast_scalar(8u8, tile_shape);
        let sign = andi(unpacked, sign_bit);
        let signed_bits = unpacked + sign * broadcast_scalar(30u8, tile_shape);
        let signed: Tile<i8, { [128] }> = bitcast(signed_bits);
        store_vector(out, offsets, signed, mask);
    }

    #[cutile::entry()]
    pub unsafe fn copy_u32(out: *mut u32, input: *mut u32, len: i32) {
        copy_impl(out, input, 0u32, len);
    }

    #[cutile::entry()]
    pub unsafe fn copy_u64(out: *mut u64, input: *mut u64, len: i32) {
        copy_impl(out, input, 0u64, len);
    }

    #[cutile::entry()]
    pub unsafe fn copy_i32(out: *mut i32, input: *mut i32, len: i32) {
        copy_impl(out, input, 0i32, len);
    }

    #[cutile::entry()]
    pub unsafe fn copy_i64(out: *mut i64, input: *mut i64, len: i32) {
        copy_impl(out, input, 0i64, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_2d_u8(out: *mut u8, input: *mut u8, rows: i32, cols: i32, len: i32) {
        transpose_2d_impl(out, input, 0u8, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_2d_i8(out: *mut i8, input: *mut i8, rows: i32, cols: i32, len: i32) {
        transpose_2d_impl(out, input, 0i8, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_2d_u32(out: *mut u32, input: *mut u32, rows: i32, cols: i32, len: i32) {
        transpose_2d_impl(out, input, 0u32, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_2d_i32(out: *mut i32, input: *mut i32, rows: i32, cols: i32, len: i32) {
        transpose_2d_impl(out, input, 0i32, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_2d_u64(out: *mut u64, input: *mut u64, rows: i32, cols: i32, len: i32) {
        transpose_2d_impl(out, input, 0u64, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_2d_i64(out: *mut i64, input: *mut i64, rows: i32, cols: i32, len: i32) {
        transpose_2d_impl(out, input, 0i64, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank3_u8(
        out: *mut u8,
        input: *mut u8,
        _batch: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank3_impl(out, input, 0u8, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank3_i8(
        out: *mut i8,
        input: *mut i8,
        _batch: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank3_impl(out, input, 0i8, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank3_u32(
        out: *mut u32,
        input: *mut u32,
        _batch: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank3_impl(out, input, 0u32, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank3_i32(
        out: *mut i32,
        input: *mut i32,
        _batch: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank3_impl(out, input, 0i32, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank3_u64(
        out: *mut u64,
        input: *mut u64,
        _batch: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank3_impl(out, input, 0u64, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank3_i64(
        out: *mut i64,
        input: *mut i64,
        _batch: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank3_impl(out, input, 0i64, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank4_u8(
        out: *mut u8,
        input: *mut u8,
        _dim0: i32,
        dim1: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank4_impl(out, input, 0u8, dim1, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank4_i8(
        out: *mut i8,
        input: *mut i8,
        _dim0: i32,
        dim1: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank4_impl(out, input, 0i8, dim1, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank4_u32(
        out: *mut u32,
        input: *mut u32,
        _dim0: i32,
        dim1: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank4_impl(out, input, 0u32, dim1, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank4_i32(
        out: *mut i32,
        input: *mut i32,
        _dim0: i32,
        dim1: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank4_impl(out, input, 0i32, dim1, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank4_u64(
        out: *mut u64,
        input: *mut u64,
        _dim0: i32,
        dim1: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank4_impl(out, input, 0u64, dim1, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn transpose_last2_rank4_i64(
        out: *mut i64,
        input: *mut i64,
        _dim0: i32,
        dim1: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        transpose_last2_rank4_impl(out, input, 0i64, dim1, rows, cols, len);
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank2_u8(
        out: *mut u8,
        input: *mut u8,
        _dim0: i32,
        dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        len: i32,
    ) {
        materialize_rank2_impl(out, input, 0u8, dim1, input_stride0, input_stride1, len);
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank2_i8(
        out: *mut i8,
        input: *mut i8,
        _dim0: i32,
        dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        len: i32,
    ) {
        materialize_rank2_impl(out, input, 0i8, dim1, input_stride0, input_stride1, len);
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank2_u32(
        out: *mut u32,
        input: *mut u32,
        _dim0: i32,
        dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        len: i32,
    ) {
        materialize_rank2_impl(out, input, 0u32, dim1, input_stride0, input_stride1, len);
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank2_i32(
        out: *mut i32,
        input: *mut i32,
        _dim0: i32,
        dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        len: i32,
    ) {
        materialize_rank2_impl(out, input, 0i32, dim1, input_stride0, input_stride1, len);
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank2_u64(
        out: *mut u64,
        input: *mut u64,
        _dim0: i32,
        dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        len: i32,
    ) {
        materialize_rank2_impl(out, input, 0u64, dim1, input_stride0, input_stride1, len);
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank2_i64(
        out: *mut i64,
        input: *mut i64,
        _dim0: i32,
        dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        len: i32,
    ) {
        materialize_rank2_impl(out, input, 0i64, dim1, input_stride0, input_stride1, len);
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank3_u8(
        out: *mut u8,
        input: *mut u8,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        len: i32,
    ) {
        materialize_rank3_impl(
            out,
            input,
            0u8,
            dim1,
            dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank3_i8(
        out: *mut i8,
        input: *mut i8,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        len: i32,
    ) {
        materialize_rank3_impl(
            out,
            input,
            0i8,
            dim1,
            dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank3_u32(
        out: *mut u32,
        input: *mut u32,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        len: i32,
    ) {
        materialize_rank3_impl(
            out,
            input,
            0u32,
            dim1,
            dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank3_i32(
        out: *mut i32,
        input: *mut i32,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        len: i32,
    ) {
        materialize_rank3_impl(
            out,
            input,
            0i32,
            dim1,
            dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank3_u64(
        out: *mut u64,
        input: *mut u64,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        len: i32,
    ) {
        materialize_rank3_impl(
            out,
            input,
            0u64,
            dim1,
            dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank3_i64(
        out: *mut i64,
        input: *mut i64,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        len: i32,
    ) {
        materialize_rank3_impl(
            out,
            input,
            0i64,
            dim1,
            dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank4_u8(
        out: *mut u8,
        input: *mut u8,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        len: i32,
    ) {
        materialize_rank4_impl(
            out,
            input,
            0u8,
            dim1,
            dim2,
            dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank4_i8(
        out: *mut i8,
        input: *mut i8,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        len: i32,
    ) {
        materialize_rank4_impl(
            out,
            input,
            0i8,
            dim1,
            dim2,
            dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank4_u32(
        out: *mut u32,
        input: *mut u32,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        len: i32,
    ) {
        materialize_rank4_impl(
            out,
            input,
            0u32,
            dim1,
            dim2,
            dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank4_i32(
        out: *mut i32,
        input: *mut i32,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        len: i32,
    ) {
        materialize_rank4_impl(
            out,
            input,
            0i32,
            dim1,
            dim2,
            dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank4_u64(
        out: *mut u64,
        input: *mut u64,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        len: i32,
    ) {
        materialize_rank4_impl(
            out,
            input,
            0u64,
            dim1,
            dim2,
            dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn materialize_rank4_i64(
        out: *mut i64,
        input: *mut i64,
        _dim0: i32,
        dim1: i32,
        dim2: i32,
        dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        len: i32,
    ) {
        materialize_rank4_impl(
            out,
            input,
            0i64,
            dim1,
            dim2,
            dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank2_u8(
        out: *mut u8,
        input: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        start0: i32,
        start1: i32,
        len: i32,
    ) {
        slice_rank2_impl(
            out,
            input,
            0u8,
            output_dim1,
            input_stride0,
            input_stride1,
            start0,
            start1,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank2_i8(
        out: *mut i8,
        input: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        start0: i32,
        start1: i32,
        len: i32,
    ) {
        slice_rank2_impl(
            out,
            input,
            0i8,
            output_dim1,
            input_stride0,
            input_stride1,
            start0,
            start1,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank2_u32(
        out: *mut u32,
        input: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        start0: i32,
        start1: i32,
        len: i32,
    ) {
        slice_rank2_impl(
            out,
            input,
            0u32,
            output_dim1,
            input_stride0,
            input_stride1,
            start0,
            start1,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank2_i32(
        out: *mut i32,
        input: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        start0: i32,
        start1: i32,
        len: i32,
    ) {
        slice_rank2_impl(
            out,
            input,
            0i32,
            output_dim1,
            input_stride0,
            input_stride1,
            start0,
            start1,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank2_u64(
        out: *mut u64,
        input: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        start0: i32,
        start1: i32,
        len: i32,
    ) {
        slice_rank2_impl(
            out,
            input,
            0u64,
            output_dim1,
            input_stride0,
            input_stride1,
            start0,
            start1,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank2_i64(
        out: *mut i64,
        input: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        start0: i32,
        start1: i32,
        len: i32,
    ) {
        slice_rank2_impl(
            out,
            input,
            0i64,
            output_dim1,
            input_stride0,
            input_stride1,
            start0,
            start1,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank3_u8(
        out: *mut u8,
        input: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        len: i32,
    ) {
        slice_rank3_impl(
            out,
            input,
            0u8,
            output_dim1,
            output_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            start0,
            start1,
            start2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank3_i8(
        out: *mut i8,
        input: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        len: i32,
    ) {
        slice_rank3_impl(
            out,
            input,
            0i8,
            output_dim1,
            output_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            start0,
            start1,
            start2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank3_u32(
        out: *mut u32,
        input: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        len: i32,
    ) {
        slice_rank3_impl(
            out,
            input,
            0u32,
            output_dim1,
            output_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            start0,
            start1,
            start2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank3_i32(
        out: *mut i32,
        input: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        len: i32,
    ) {
        slice_rank3_impl(
            out,
            input,
            0i32,
            output_dim1,
            output_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            start0,
            start1,
            start2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank3_u64(
        out: *mut u64,
        input: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        len: i32,
    ) {
        slice_rank3_impl(
            out,
            input,
            0u64,
            output_dim1,
            output_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            start0,
            start1,
            start2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank3_i64(
        out: *mut i64,
        input: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        len: i32,
    ) {
        slice_rank3_impl(
            out,
            input,
            0i64,
            output_dim1,
            output_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            start0,
            start1,
            start2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank4_u8(
        out: *mut u8,
        input: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        start3: i32,
        len: i32,
    ) {
        slice_rank4_impl(
            out,
            input,
            0u8,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            start0,
            start1,
            start2,
            start3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank4_i8(
        out: *mut i8,
        input: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        start3: i32,
        len: i32,
    ) {
        slice_rank4_impl(
            out,
            input,
            0i8,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            start0,
            start1,
            start2,
            start3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank4_u32(
        out: *mut u32,
        input: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        start3: i32,
        len: i32,
    ) {
        slice_rank4_impl(
            out,
            input,
            0u32,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            start0,
            start1,
            start2,
            start3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank4_i32(
        out: *mut i32,
        input: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        start3: i32,
        len: i32,
    ) {
        slice_rank4_impl(
            out,
            input,
            0i32,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            start0,
            start1,
            start2,
            start3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank4_u64(
        out: *mut u64,
        input: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        start3: i32,
        len: i32,
    ) {
        slice_rank4_impl(
            out,
            input,
            0u64,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            start0,
            start1,
            start2,
            start3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn slice_rank4_i64(
        out: *mut i64,
        input: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        start3: i32,
        len: i32,
    ) {
        slice_rank4_impl(
            out,
            input,
            0i64,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            start0,
            start1,
            start2,
            start3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank2_u8(
        out: *mut u8,
        input: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_value: u8,
        len: i32,
    ) {
        pad_rank2_impl(
            out,
            input,
            output_dim1,
            input_dim0,
            input_dim1,
            input_stride0,
            input_stride1,
            pad_before0,
            pad_before1,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank2_i8(
        out: *mut i8,
        input: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_value: i8,
        len: i32,
    ) {
        pad_rank2_impl(
            out,
            input,
            output_dim1,
            input_dim0,
            input_dim1,
            input_stride0,
            input_stride1,
            pad_before0,
            pad_before1,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank2_u32(
        out: *mut u32,
        input: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_value: u32,
        len: i32,
    ) {
        pad_rank2_impl(
            out,
            input,
            output_dim1,
            input_dim0,
            input_dim1,
            input_stride0,
            input_stride1,
            pad_before0,
            pad_before1,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank2_i32(
        out: *mut i32,
        input: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_value: i32,
        len: i32,
    ) {
        pad_rank2_impl(
            out,
            input,
            output_dim1,
            input_dim0,
            input_dim1,
            input_stride0,
            input_stride1,
            pad_before0,
            pad_before1,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank2_u64(
        out: *mut u64,
        input: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_value: u64,
        len: i32,
    ) {
        pad_rank2_impl(
            out,
            input,
            output_dim1,
            input_dim0,
            input_dim1,
            input_stride0,
            input_stride1,
            pad_before0,
            pad_before1,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank2_i64(
        out: *mut i64,
        input: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_value: i64,
        len: i32,
    ) {
        pad_rank2_impl(
            out,
            input,
            output_dim1,
            input_dim0,
            input_dim1,
            input_stride0,
            input_stride1,
            pad_before0,
            pad_before1,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank3_u8(
        out: *mut u8,
        input: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_value: u8,
        len: i32,
    ) {
        pad_rank3_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            input_dim0,
            input_dim1,
            input_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank3_i8(
        out: *mut i8,
        input: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_value: i8,
        len: i32,
    ) {
        pad_rank3_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            input_dim0,
            input_dim1,
            input_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank3_u32(
        out: *mut u32,
        input: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_value: u32,
        len: i32,
    ) {
        pad_rank3_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            input_dim0,
            input_dim1,
            input_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank3_i32(
        out: *mut i32,
        input: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_value: i32,
        len: i32,
    ) {
        pad_rank3_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            input_dim0,
            input_dim1,
            input_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank3_u64(
        out: *mut u64,
        input: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_value: u64,
        len: i32,
    ) {
        pad_rank3_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            input_dim0,
            input_dim1,
            input_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank3_i64(
        out: *mut i64,
        input: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_value: i64,
        len: i32,
    ) {
        pad_rank3_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            input_dim0,
            input_dim1,
            input_dim2,
            input_stride0,
            input_stride1,
            input_stride2,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank4_u8(
        out: *mut u8,
        input: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_before3: i32,
        pad_value: u8,
        len: i32,
    ) {
        pad_rank4_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            output_dim3,
            input_dim0,
            input_dim1,
            input_dim2,
            input_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_before3,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank4_i8(
        out: *mut i8,
        input: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_before3: i32,
        pad_value: i8,
        len: i32,
    ) {
        pad_rank4_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            output_dim3,
            input_dim0,
            input_dim1,
            input_dim2,
            input_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_before3,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank4_u32(
        out: *mut u32,
        input: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_before3: i32,
        pad_value: u32,
        len: i32,
    ) {
        pad_rank4_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            output_dim3,
            input_dim0,
            input_dim1,
            input_dim2,
            input_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_before3,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank4_i32(
        out: *mut i32,
        input: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_before3: i32,
        pad_value: i32,
        len: i32,
    ) {
        pad_rank4_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            output_dim3,
            input_dim0,
            input_dim1,
            input_dim2,
            input_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_before3,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank4_u64(
        out: *mut u64,
        input: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_before3: i32,
        pad_value: u64,
        len: i32,
    ) {
        pad_rank4_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            output_dim3,
            input_dim0,
            input_dim1,
            input_dim2,
            input_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_before3,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn pad_rank4_i64(
        out: *mut i64,
        input: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_before3: i32,
        pad_value: i64,
        len: i32,
    ) {
        pad_rank4_impl(
            out,
            input,
            output_dim1,
            output_dim2,
            output_dim3,
            input_dim0,
            input_dim1,
            input_dim2,
            input_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            pad_before0,
            pad_before1,
            pad_before2,
            pad_before3,
            pad_value,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_u8_axis0(
        out: *mut u8,
        lhs: *mut u8,
        rhs: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_impl::<0, u8>(
            out,
            lhs,
            rhs,
            0u8,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_u8_axis1(
        out: *mut u8,
        lhs: *mut u8,
        rhs: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_impl::<1, u8>(
            out,
            lhs,
            rhs,
            0u8,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_i8_axis0(
        out: *mut i8,
        lhs: *mut i8,
        rhs: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_impl::<0, i8>(
            out,
            lhs,
            rhs,
            0i8,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_i8_axis1(
        out: *mut i8,
        lhs: *mut i8,
        rhs: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_impl::<1, i8>(
            out,
            lhs,
            rhs,
            0i8,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_u32_axis0(
        out: *mut u32,
        lhs: *mut u32,
        rhs: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_impl::<0, u32>(
            out,
            lhs,
            rhs,
            0u32,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_u32_axis1(
        out: *mut u32,
        lhs: *mut u32,
        rhs: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_impl::<1, u32>(
            out,
            lhs,
            rhs,
            0u32,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len,
        );
    }

    macro_rules! concat_rank2_common_body {
        (
            $axis:expr,
            $ty:ty,
            $fill:expr,
            $out:ident,
            $lhs:ident,
            $rhs:ident,
            $output_dim1:ident,
            $lhs_dim0:ident,
            $lhs_dim1:ident,
            $lhs_stride0:ident,
            $lhs_stride1:ident,
            $rhs_stride0:ident,
            $rhs_stride1:ident,
            $len:ident
        ) => {
            concat_rank2_impl::<$axis, $ty>(
                $out,
                $lhs,
                $rhs,
                $fill,
                $output_dim1,
                $lhs_dim0,
                $lhs_dim1,
                $lhs_stride0,
                $lhs_stride1,
                $rhs_stride0,
                $rhs_stride1,
                $len,
            );
        };
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_i32_axis0(
        out: *mut i32,
        lhs: *mut i32,
        rhs: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_common_body!(
            0,
            i32,
            0i32,
            out,
            lhs,
            rhs,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_i32_axis1(
        out: *mut i32,
        lhs: *mut i32,
        rhs: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_common_body!(
            1,
            i32,
            0i32,
            out,
            lhs,
            rhs,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_u64_axis0(
        out: *mut u64,
        lhs: *mut u64,
        rhs: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_common_body!(
            0,
            u64,
            0u64,
            out,
            lhs,
            rhs,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_u64_axis1(
        out: *mut u64,
        lhs: *mut u64,
        rhs: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_common_body!(
            1,
            u64,
            0u64,
            out,
            lhs,
            rhs,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_i64_axis0(
        out: *mut i64,
        lhs: *mut i64,
        rhs: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_common_body!(
            0,
            i64,
            0i64,
            out,
            lhs,
            rhs,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank2_i64_axis1(
        out: *mut i64,
        lhs: *mut i64,
        rhs: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        concat_rank2_common_body!(
            1,
            i64,
            0i64,
            out,
            lhs,
            rhs,
            output_dim1,
            lhs_dim0,
            lhs_dim1,
            lhs_stride0,
            lhs_stride1,
            rhs_stride0,
            rhs_stride1,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_u8_axis0(
        out: *mut u8,
        lhs: *mut u8,
        rhs: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_impl::<0, u8>(
            out,
            lhs,
            rhs,
            0u8,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_u8_axis1(
        out: *mut u8,
        lhs: *mut u8,
        rhs: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_impl::<1, u8>(
            out,
            lhs,
            rhs,
            0u8,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_u8_axis2(
        out: *mut u8,
        lhs: *mut u8,
        rhs: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_impl::<2, u8>(
            out,
            lhs,
            rhs,
            0u8,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_i8_axis0(
        out: *mut i8,
        lhs: *mut i8,
        rhs: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_impl::<0, i8>(
            out,
            lhs,
            rhs,
            0i8,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_i8_axis1(
        out: *mut i8,
        lhs: *mut i8,
        rhs: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_impl::<1, i8>(
            out,
            lhs,
            rhs,
            0i8,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_i8_axis2(
        out: *mut i8,
        lhs: *mut i8,
        rhs: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_impl::<2, i8>(
            out,
            lhs,
            rhs,
            0i8,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_u32_axis0(
        out: *mut u32,
        lhs: *mut u32,
        rhs: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_impl::<0, u32>(
            out,
            lhs,
            rhs,
            0u32,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_u32_axis1(
        out: *mut u32,
        lhs: *mut u32,
        rhs: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_impl::<1, u32>(
            out,
            lhs,
            rhs,
            0u32,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_u32_axis2(
        out: *mut u32,
        lhs: *mut u32,
        rhs: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_impl::<2, u32>(
            out,
            lhs,
            rhs,
            0u32,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len,
        );
    }

    macro_rules! concat_rank3_common_body {
        (
            $axis:expr,
            $ty:ty,
            $fill:expr,
            $out:ident,
            $lhs:ident,
            $rhs:ident,
            $output_dim1:ident,
            $output_dim2:ident,
            $lhs_dim0:ident,
            $lhs_dim1:ident,
            $lhs_dim2:ident,
            $lhs_stride0:ident,
            $lhs_stride1:ident,
            $lhs_stride2:ident,
            $rhs_stride0:ident,
            $rhs_stride1:ident,
            $rhs_stride2:ident,
            $len:ident
        ) => {
            concat_rank3_impl::<$axis, $ty>(
                $out,
                $lhs,
                $rhs,
                $fill,
                $output_dim1,
                $output_dim2,
                $lhs_dim0,
                $lhs_dim1,
                $lhs_dim2,
                $lhs_stride0,
                $lhs_stride1,
                $lhs_stride2,
                $rhs_stride0,
                $rhs_stride1,
                $rhs_stride2,
                $len,
            );
        };
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_i32_axis0(
        out: *mut i32,
        lhs: *mut i32,
        rhs: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_common_body!(
            0,
            i32,
            0i32,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_i32_axis1(
        out: *mut i32,
        lhs: *mut i32,
        rhs: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_common_body!(
            1,
            i32,
            0i32,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_i32_axis2(
        out: *mut i32,
        lhs: *mut i32,
        rhs: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_common_body!(
            2,
            i32,
            0i32,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_u64_axis0(
        out: *mut u64,
        lhs: *mut u64,
        rhs: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_common_body!(
            0,
            u64,
            0u64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_u64_axis1(
        out: *mut u64,
        lhs: *mut u64,
        rhs: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_common_body!(
            1,
            u64,
            0u64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_u64_axis2(
        out: *mut u64,
        lhs: *mut u64,
        rhs: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_common_body!(
            2,
            u64,
            0u64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_i64_axis0(
        out: *mut i64,
        lhs: *mut i64,
        rhs: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_common_body!(
            0,
            i64,
            0i64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_i64_axis1(
        out: *mut i64,
        lhs: *mut i64,
        rhs: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_common_body!(
            1,
            i64,
            0i64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank3_i64_axis2(
        out: *mut i64,
        lhs: *mut i64,
        rhs: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        concat_rank3_common_body!(
            2,
            i64,
            0i64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u8_axis0(
        out: *mut u8,
        lhs: *mut u8,
        rhs: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<0, u8>(
            out,
            lhs,
            rhs,
            0u8,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u8_axis1(
        out: *mut u8,
        lhs: *mut u8,
        rhs: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<1, u8>(
            out,
            lhs,
            rhs,
            0u8,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u8_axis2(
        out: *mut u8,
        lhs: *mut u8,
        rhs: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<2, u8>(
            out,
            lhs,
            rhs,
            0u8,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u8_axis3(
        out: *mut u8,
        lhs: *mut u8,
        rhs: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<3, u8>(
            out,
            lhs,
            rhs,
            0u8,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i8_axis0(
        out: *mut i8,
        lhs: *mut i8,
        rhs: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<0, i8>(
            out,
            lhs,
            rhs,
            0i8,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i8_axis1(
        out: *mut i8,
        lhs: *mut i8,
        rhs: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<1, i8>(
            out,
            lhs,
            rhs,
            0i8,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i8_axis2(
        out: *mut i8,
        lhs: *mut i8,
        rhs: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<2, i8>(
            out,
            lhs,
            rhs,
            0i8,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i8_axis3(
        out: *mut i8,
        lhs: *mut i8,
        rhs: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<3, i8>(
            out,
            lhs,
            rhs,
            0i8,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u32_axis0(
        out: *mut u32,
        lhs: *mut u32,
        rhs: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<0, u32>(
            out,
            lhs,
            rhs,
            0u32,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u32_axis1(
        out: *mut u32,
        lhs: *mut u32,
        rhs: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<1, u32>(
            out,
            lhs,
            rhs,
            0u32,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u32_axis2(
        out: *mut u32,
        lhs: *mut u32,
        rhs: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<2, u32>(
            out,
            lhs,
            rhs,
            0u32,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u32_axis3(
        out: *mut u32,
        lhs: *mut u32,
        rhs: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_impl::<3, u32>(
            out,
            lhs,
            rhs,
            0u32,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len,
        );
    }

    macro_rules! concat_rank4_common_body {
        (
            $axis:expr,
            $ty:ty,
            $fill:expr,
            $out:ident,
            $lhs:ident,
            $rhs:ident,
            $output_dim1:ident,
            $output_dim2:ident,
            $output_dim3:ident,
            $lhs_dim0:ident,
            $lhs_dim1:ident,
            $lhs_dim2:ident,
            $lhs_dim3:ident,
            $lhs_stride0:ident,
            $lhs_stride1:ident,
            $lhs_stride2:ident,
            $lhs_stride3:ident,
            $rhs_stride0:ident,
            $rhs_stride1:ident,
            $rhs_stride2:ident,
            $rhs_stride3:ident,
            $len:ident
        ) => {
            concat_rank4_impl::<$axis, $ty>(
                $out,
                $lhs,
                $rhs,
                $fill,
                $output_dim1,
                $output_dim2,
                $output_dim3,
                $lhs_dim0,
                $lhs_dim1,
                $lhs_dim2,
                $lhs_dim3,
                $lhs_stride0,
                $lhs_stride1,
                $lhs_stride2,
                $lhs_stride3,
                $rhs_stride0,
                $rhs_stride1,
                $rhs_stride2,
                $rhs_stride3,
                $len,
            );
        };
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i32_axis0(
        out: *mut i32,
        lhs: *mut i32,
        rhs: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            0,
            i32,
            0i32,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i32_axis1(
        out: *mut i32,
        lhs: *mut i32,
        rhs: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            1,
            i32,
            0i32,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i32_axis2(
        out: *mut i32,
        lhs: *mut i32,
        rhs: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            2,
            i32,
            0i32,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i32_axis3(
        out: *mut i32,
        lhs: *mut i32,
        rhs: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            3,
            i32,
            0i32,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u64_axis0(
        out: *mut u64,
        lhs: *mut u64,
        rhs: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            0,
            u64,
            0u64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u64_axis1(
        out: *mut u64,
        lhs: *mut u64,
        rhs: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            1,
            u64,
            0u64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u64_axis2(
        out: *mut u64,
        lhs: *mut u64,
        rhs: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            2,
            u64,
            0u64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_u64_axis3(
        out: *mut u64,
        lhs: *mut u64,
        rhs: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            3,
            u64,
            0u64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i64_axis0(
        out: *mut i64,
        lhs: *mut i64,
        rhs: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            0,
            i64,
            0i64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i64_axis1(
        out: *mut i64,
        lhs: *mut i64,
        rhs: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            1,
            i64,
            0i64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i64_axis2(
        out: *mut i64,
        lhs: *mut i64,
        rhs: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            2,
            i64,
            0i64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn concat_rank4_i64_axis3(
        out: *mut i64,
        lhs: *mut i64,
        rhs: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        concat_rank4_common_body!(
            3,
            i64,
            0i64,
            out,
            lhs,
            rhs,
            output_dim1,
            output_dim2,
            output_dim3,
            lhs_dim0,
            lhs_dim1,
            lhs_dim2,
            lhs_dim3,
            lhs_stride0,
            lhs_stride1,
            lhs_stride2,
            lhs_stride3,
            rhs_stride0,
            rhs_stride1,
            rhs_stride2,
            rhs_stride3,
            len
        );
    }

    #[cutile::entry()]
    pub unsafe fn repeat_kv_rank4_u8(
        out: *mut u8,
        input: *mut u8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        repeats: i32,
        len: i32,
    ) {
        repeat_kv_rank4_impl(
            out,
            input,
            0u8,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            repeats,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn repeat_kv_rank4_i8(
        out: *mut i8,
        input: *mut i8,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        repeats: i32,
        len: i32,
    ) {
        repeat_kv_rank4_impl(
            out,
            input,
            0i8,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            repeats,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn repeat_kv_rank4_u32(
        out: *mut u32,
        input: *mut u32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        repeats: i32,
        len: i32,
    ) {
        repeat_kv_rank4_impl(
            out,
            input,
            0u32,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            repeats,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn repeat_kv_rank4_i32(
        out: *mut i32,
        input: *mut i32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        repeats: i32,
        len: i32,
    ) {
        repeat_kv_rank4_impl(
            out,
            input,
            0i32,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            repeats,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn repeat_kv_rank4_u64(
        out: *mut u64,
        input: *mut u64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        repeats: i32,
        len: i32,
    ) {
        repeat_kv_rank4_impl(
            out,
            input,
            0u64,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            repeats,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn repeat_kv_rank4_i64(
        out: *mut i64,
        input: *mut i64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        repeats: i32,
        len: i32,
    ) {
        repeat_kv_rank4_impl(
            out,
            input,
            0i64,
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            repeats,
            len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn arange_f32(out: *mut f32, start: f32, step: f32, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let offsets_f32: Tile<f32, { [128] }> = convert_tile(offsets);
        let values =
            broadcast_scalar(start, tile_shape) + offsets_f32 * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, values, mask);
    }

    #[cutile::entry()]
    pub unsafe fn arange_f64(out: *mut f64, start: f64, step: f64, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let offsets_f64: Tile<f64, { [128] }> = convert_tile(offsets);
        let values =
            broadcast_scalar(start, tile_shape) + offsets_f64 * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, values, mask);
    }

    #[cutile::entry()]
    pub unsafe fn arange_f16(out: *mut f16, start: f16, step: f16, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let offsets_f32: Tile<f32, { [128] }> = convert_tile(offsets);
        let start: f32 = convert_scalar(start);
        let step: f32 = convert_scalar(step);
        let values =
            broadcast_scalar(start, tile_shape) + offsets_f32 * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, convert_tile(values), mask);
    }

    #[cutile::entry()]
    pub unsafe fn arange_u8(out: *mut u8, start: u8, step: u8, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let start: i32 = convert_scalar(start);
        let step: i32 = convert_scalar(step);
        let values =
            broadcast_scalar(start, tile_shape) + offsets * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, trunci(values, overflow::None), mask);
    }

    #[cutile::entry()]
    pub unsafe fn arange_i8(out: *mut i8, start: i8, step: i8, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let start: i32 = convert_scalar(start);
        let step: i32 = convert_scalar(step);
        let values =
            broadcast_scalar(start, tile_shape) + offsets * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, trunci(values, overflow::None), mask);
    }

    #[cutile::entry()]
    pub unsafe fn arange_i32(out: *mut i32, start: i32, step: i32, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let values =
            broadcast_scalar(start, tile_shape) + offsets * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, values, mask);
    }

    #[cutile::entry()]
    pub unsafe fn arange_i64(out: *mut i64, start: i64, step: i64, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let offsets_i64: Tile<i64, { [128] }> = convert_tile(offsets);
        let values =
            broadcast_scalar(start, tile_shape) + offsets_i64 * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, values, mask);
    }

    #[cutile::entry()]
    pub unsafe fn arange_u32(out: *mut u32, start: u32, step: u32, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let offsets_u32: Tile<u32, { [128] }> = convert_tile(offsets);
        let values =
            broadcast_scalar(start, tile_shape) + offsets_u32 * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, values, mask);
    }

    #[cutile::entry()]
    pub unsafe fn arange_u64(out: *mut u64, start: u64, step: u64, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let offsets_u64: Tile<u64, { [128] }> = convert_tile(offsets);
        let values =
            broadcast_scalar(start, tile_shape) + offsets_u64 * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, values, mask);
    }

    #[cutile::entry()]
    pub unsafe fn linspace_f16(out: *mut f16, start: f16, step: f16, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let offsets_f32: Tile<f32, { [128] }> = convert_tile(offsets);
        let start: f32 = convert_scalar(start);
        let step: f32 = convert_scalar(step);
        let values =
            broadcast_scalar(start, tile_shape) + offsets_f32 * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, convert_tile(values), mask);
    }

    #[cutile::entry()]
    pub unsafe fn linspace_f32(out: *mut f32, start: f32, step: f32, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let offsets_f32: Tile<f32, { [128] }> = convert_tile(offsets);
        let values =
            broadcast_scalar(start, tile_shape) + offsets_f32 * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, values, mask);
    }

    #[cutile::entry()]
    pub unsafe fn linspace_f64(out: *mut f64, start: f64, step: f64, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let offsets_f64: Tile<f64, { [128] }> = convert_tile(offsets);
        let values =
            broadcast_scalar(start, tile_shape) + offsets_f64 * broadcast_scalar(step, tile_shape);
        store_vector(out, offsets, values, mask);
    }

    #[cutile::entry()]
    pub unsafe fn block_argmax_f16(
        block_max: *mut f32,
        block_idx: *mut i32,
        input: *mut f16,
        len: i32,
    ) {
        let values_f16 = load_block_argmax_tile(input, f16::NEG_INFINITY, len);
        let values: Tile<f32, { [256] }> = convert_tile(values_f16);
        block_argmax_f32_values(block_max, block_idx, values, len);
    }

    #[cutile::entry()]
    pub unsafe fn block_argmax_f32(
        block_max: *mut f32,
        block_idx: *mut i32,
        input: *mut f32,
        len: i32,
    ) {
        let values = load_block_argmax_tile(input, -1.0e30f32, len);
        block_argmax_f32_values(block_max, block_idx, values, len);
    }

    #[cutile::entry()]
    pub unsafe fn block_argmax_f64(
        block_max: *mut f64,
        block_idx: *mut i32,
        input: *mut f64,
        len: i32,
    ) {
        let values = load_block_argmax_tile(input, -1.7976931348623157e308f64, len);
        block_argmax_f64_values(block_max, block_idx, values, len);
    }

    #[cutile::entry()]
    pub unsafe fn block_argmin_f16(
        block_min: *mut f32,
        block_idx: *mut i32,
        input: *mut f16,
        len: i32,
    ) {
        let values_f16 = load_block_argmax_tile(input, f16::INFINITY, len);
        let values: Tile<f32, { [256] }> = convert_tile(values_f16);
        block_argmin_f32_values(block_min, block_idx, values, len);
    }

    #[cutile::entry()]
    pub unsafe fn block_argmin_f32(
        block_min: *mut f32,
        block_idx: *mut i32,
        input: *mut f32,
        len: i32,
    ) {
        let values = load_block_argmax_tile(input, 1.0e30f32, len);
        block_argmin_f32_values(block_min, block_idx, values, len);
    }

    #[cutile::entry()]
    pub unsafe fn block_argmin_f64(
        block_min: *mut f64,
        block_idx: *mut i32,
        input: *mut f64,
        len: i32,
    ) {
        let values = load_block_argmax_tile(input, 1.7976931348623157e308f64, len);
        block_argmin_f64_values(block_min, block_idx, values, len);
    }

    #[cutile::entry()]
    pub unsafe fn reduce_block_argmax_f32(
        out: *mut i32,
        block_max: *mut f32,
        block_idx: *mut i32,
        num_blocks: i32,
    ) {
        let tile_shape = const_shape![1024];
        let offsets: Tile<i32, { [1024] }> = iota(tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(num_blocks, tile_shape),
            predicate::LessThan,
        );
        let values = load_vector_1024(block_max, offsets, valid, -1.0e30f32);
        let indices = load_vector_1024(block_idx, offsets, valid, 0i32);
        let best_value: Tile<f32, { [1] }> = reduce_max(values, 0i32);
        let best = best_value.reshape(const_shape![1i32]).broadcast(tile_shape);
        let matches = valid & cmpf(values, best, predicate::Equal, cmp_ordering::Ordered);
        let invalid: Tile<i32, { [1024] }> = constant(2_147_483_647i32, tile_shape);
        let candidate_indices = select(matches, indices, invalid);
        let winner: Tile<i32, { [1] }> = reduce_min(candidate_indices, 0i32);
        store_first(out, winner);
    }

    #[cutile::entry()]
    pub unsafe fn reduce_block_argmax_f64(
        out: *mut i32,
        block_max: *mut f64,
        block_idx: *mut i32,
        num_blocks: i32,
    ) {
        let tile_shape = const_shape![1024];
        let offsets: Tile<i32, { [1024] }> = iota(tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(num_blocks, tile_shape),
            predicate::LessThan,
        );
        let values = load_vector_1024(block_max, offsets, valid, -1.7976931348623157e308f64);
        let indices = load_vector_1024(block_idx, offsets, valid, 0i32);
        let best_value: Tile<f64, { [1] }> = reduce_max(values, 0i32);
        let best = best_value.reshape(const_shape![1i32]).broadcast(tile_shape);
        let matches = valid & cmpf(values, best, predicate::Equal, cmp_ordering::Ordered);
        let invalid: Tile<i32, { [1024] }> = constant(2_147_483_647i32, tile_shape);
        let candidate_indices = select(matches, indices, invalid);
        let winner: Tile<i32, { [1] }> = reduce_min(candidate_indices, 0i32);
        store_first(out, winner);
    }

    #[cutile::entry()]
    pub unsafe fn reduce_block_argmin_f32(
        out: *mut i32,
        block_min: *mut f32,
        block_idx: *mut i32,
        num_blocks: i32,
    ) {
        let tile_shape = const_shape![1024];
        let offsets: Tile<i32, { [1024] }> = iota(tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(num_blocks, tile_shape),
            predicate::LessThan,
        );
        let values = load_vector_1024(block_min, offsets, valid, 1.0e30f32);
        let indices = load_vector_1024(block_idx, offsets, valid, 0i32);
        let best_value: Tile<f32, { [1] }> = reduce_min(values, 0i32);
        let best = best_value.reshape(const_shape![1i32]).broadcast(tile_shape);
        let matches = valid & cmpf(values, best, predicate::Equal, cmp_ordering::Ordered);
        let invalid: Tile<i32, { [1024] }> = constant(2_147_483_647i32, tile_shape);
        let candidate_indices = select(matches, indices, invalid);
        let winner: Tile<i32, { [1] }> = reduce_min(candidate_indices, 0i32);
        store_first(out, winner);
    }

    #[cutile::entry()]
    pub unsafe fn reduce_block_argmin_f64(
        out: *mut i32,
        block_min: *mut f64,
        block_idx: *mut i32,
        num_blocks: i32,
    ) {
        let tile_shape = const_shape![1024];
        let offsets: Tile<i32, { [1024] }> = iota(tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(num_blocks, tile_shape),
            predicate::LessThan,
        );
        let values = load_vector_1024(block_min, offsets, valid, 1.7976931348623157e308f64);
        let indices = load_vector_1024(block_idx, offsets, valid, 0i32);
        let best_value: Tile<f64, { [1] }> = reduce_min(values, 0i32);
        let best = best_value.reshape(const_shape![1i32]).broadcast(tile_shape);
        let matches = valid & cmpf(values, best, predicate::Equal, cmp_ordering::Ordered);
        let invalid: Tile<i32, { [1024] }> = constant(2_147_483_647i32, tile_shape);
        let candidate_indices = select(matches, indices, invalid);
        let winner: Tile<i32, { [1] }> = reduce_min(candidate_indices, 0i32);
        store_first(out, winner);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_partial_max_f16(
        partial: *mut f32,
        input: *mut f16,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load_f16(input, cols);
        let values = select(
            softmax_chunk_valid(cols),
            values,
            constant(f32::NEG_INFINITY, const_shape![1024]),
        );
        let value: Tile<f32, { [1] }> = reduce_max(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_partial_max_f32(
        partial: *mut f32,
        input: *mut f32,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load(input, cols, f32::NEG_INFINITY);
        let value: Tile<f32, { [1] }> = reduce_max(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_partial_max_f64(
        partial: *mut f64,
        input: *mut f64,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load(input, cols, f64::NEG_INFINITY);
        let value: Tile<f64, { [1] }> = reduce_max(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_reduce_max_f32(row_max: *mut f32, partial: *mut f32, chunks: i32) {
        let values = softmax_row_scratch_load(partial, chunks, f32::NEG_INFINITY);
        let value: Tile<f32, { [1] }> = reduce_max(values, 0i32);
        let pid: (i32, i32, i32) = get_tile_block_id();
        store_scalar_at(row_max, broadcast_scalar(pid.0, const_shape![1]), value);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_reduce_max_f64(row_max: *mut f64, partial: *mut f64, chunks: i32) {
        let values = softmax_row_scratch_load(partial, chunks, f64::NEG_INFINITY);
        let value: Tile<f64, { [1] }> = reduce_max(values, 0i32);
        let pid: (i32, i32, i32) = get_tile_block_id();
        store_scalar_at(row_max, broadcast_scalar(pid.0, const_shape![1]), value);
    }

    #[cutile::entry()]
    pub unsafe fn softmin_partial_min_f16(
        partial: *mut f32,
        input: *mut f16,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load_f16(input, cols);
        let values = select(
            softmax_chunk_valid(cols),
            values,
            constant(f32::INFINITY, const_shape![1024]),
        );
        let value: Tile<f32, { [1] }> = reduce_min(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmin_partial_min_f32(
        partial: *mut f32,
        input: *mut f32,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load(input, cols, f32::INFINITY);
        let value: Tile<f32, { [1] }> = reduce_min(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmin_partial_min_f64(
        partial: *mut f64,
        input: *mut f64,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load(input, cols, f64::INFINITY);
        let value: Tile<f64, { [1] }> = reduce_min(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmin_reduce_min_f32(row_min: *mut f32, partial: *mut f32, chunks: i32) {
        let values = softmax_row_scratch_load(partial, chunks, f32::INFINITY);
        let value: Tile<f32, { [1] }> = reduce_min(values, 0i32);
        let pid: (i32, i32, i32) = get_tile_block_id();
        store_scalar_at(row_min, broadcast_scalar(pid.0, const_shape![1]), value);
    }

    #[cutile::entry()]
    pub unsafe fn softmin_reduce_min_f64(row_min: *mut f64, partial: *mut f64, chunks: i32) {
        let values = softmax_row_scratch_load(partial, chunks, f64::INFINITY);
        let value: Tile<f64, { [1] }> = reduce_min(values, 0i32);
        let pid: (i32, i32, i32) = get_tile_block_id();
        store_scalar_at(row_min, broadcast_scalar(pid.0, const_shape![1]), value);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_partial_sum_f16(
        partial: *mut f32,
        input: *mut f16,
        row_max: *mut f32,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load_f16(input, cols);
        let row_max_values = softmax_chunk_row_value(row_max, 0.0f32);
        let exp_values = exp(values - row_max_values);
        let values = select(
            softmax_chunk_valid(cols),
            exp_values,
            constant(0.0f32, const_shape![1024]),
        );
        let value: Tile<f32, { [1] }> = reduce_sum(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_partial_sum_f32(
        partial: *mut f32,
        input: *mut f32,
        row_max: *mut f32,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load(input, cols, 0.0f32);
        let row_max_values = softmax_chunk_row_value(row_max, 0.0f32);
        let exp_values = exp(values - row_max_values);
        let values = select(
            softmax_chunk_valid(cols),
            exp_values,
            constant(0.0f32, const_shape![1024]),
        );
        let value: Tile<f32, { [1] }> = reduce_sum(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_partial_sum_f64(
        partial: *mut f64,
        input: *mut f64,
        row_max: *mut f64,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load(input, cols, 0.0f64);
        let row_max_values = softmax_chunk_row_value(row_max, 0.0f64);
        let exp_values = exp(values - row_max_values);
        let values = select(
            softmax_chunk_valid(cols),
            exp_values,
            constant(0.0f64, const_shape![1024]),
        );
        let value: Tile<f64, { [1] }> = reduce_sum(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmin_partial_sum_f16(
        partial: *mut f32,
        input: *mut f16,
        row_min: *mut f32,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load_f16(input, cols);
        let row_min_values = softmax_chunk_row_value(row_min, 0.0f32);
        let exp_values = exp(row_min_values - values);
        let values = select(
            softmax_chunk_valid(cols),
            exp_values,
            constant(0.0f32, const_shape![1024]),
        );
        let value: Tile<f32, { [1] }> = reduce_sum(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmin_partial_sum_f32(
        partial: *mut f32,
        input: *mut f32,
        row_min: *mut f32,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load(input, cols, 0.0f32);
        let row_min_values = softmax_chunk_row_value(row_min, 0.0f32);
        let exp_values = exp(row_min_values - values);
        let values = select(
            softmax_chunk_valid(cols),
            exp_values,
            constant(0.0f32, const_shape![1024]),
        );
        let value: Tile<f32, { [1] }> = reduce_sum(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmin_partial_sum_f64(
        partial: *mut f64,
        input: *mut f64,
        row_min: *mut f64,
        cols: i32,
        chunks: i32,
    ) {
        let values = softmax_chunk_load(input, cols, 0.0f64);
        let row_min_values = softmax_chunk_row_value(row_min, 0.0f64);
        let exp_values = exp(row_min_values - values);
        let values = select(
            softmax_chunk_valid(cols),
            exp_values,
            constant(0.0f64, const_shape![1024]),
        );
        let value: Tile<f64, { [1] }> = reduce_sum(values, 0i32);
        store_softmax_partial(partial, chunks, value);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_reduce_sum_f32(row_sum: *mut f32, partial: *mut f32, chunks: i32) {
        let values = softmax_row_scratch_load(partial, chunks, 0.0f32);
        let value: Tile<f32, { [1] }> = reduce_sum(values, 0i32);
        let pid: (i32, i32, i32) = get_tile_block_id();
        store_scalar_at(row_sum, broadcast_scalar(pid.0, const_shape![1]), value);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_reduce_sum_f64(row_sum: *mut f64, partial: *mut f64, chunks: i32) {
        let values = softmax_row_scratch_load(partial, chunks, 0.0f64);
        let value: Tile<f64, { [1] }> = reduce_sum(values, 0i32);
        let pid: (i32, i32, i32) = get_tile_block_id();
        store_scalar_at(row_sum, broadcast_scalar(pid.0, const_shape![1]), value);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_lse_from_parts_f32(
        out: *mut f32,
        row_max: *mut f32,
        row_sum: *mut f32,
        rows: i32,
    ) {
        let (offsets, mask) = vector_offsets(rows);
        let max_values = load_vector(row_max, offsets, mask, 0.0f32);
        let sum_values = load_vector(row_sum, offsets, mask, 1.0f32);
        store_vector(out, offsets, max_values + log(sum_values), mask);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_lse_from_parts_f64(
        out: *mut f64,
        row_max: *mut f64,
        row_sum: *mut f64,
        rows: i32,
    ) {
        let (offsets, mask) = vector_offsets(rows);
        let max_values = load_vector(row_max, offsets, mask, 0.0f64);
        let sum_values = load_vector(row_sum, offsets, mask, 1.0f64);
        store_vector(out, offsets, max_values + log(sum_values), mask);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_normalize_f16(
        out: *mut f16,
        input: *mut f16,
        row_max: *mut f32,
        row_sum: *mut f32,
        cols: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let values = load_vector(input, offsets, mask, f16::from_f32(0.0));
        let values: Tile<f32, { [128] }> = convert_tile(values);
        let row_values = softmax_vector_rows(offsets, cols);
        let max_values = load_vector(row_max, row_values, mask, 0.0f32);
        let sum_values = load_vector(row_sum, row_values, mask, 1.0f32);
        let output = exp(values - max_values) / sum_values;
        store_vector(out, offsets, convert_tile(output), mask);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_normalize_f32(
        out: *mut f32,
        input: *mut f32,
        row_max: *mut f32,
        row_sum: *mut f32,
        cols: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let values = load_vector(input, offsets, mask, 0.0f32);
        let row_values = softmax_vector_rows(offsets, cols);
        let max_values = load_vector(row_max, row_values, mask, 0.0f32);
        let sum_values = load_vector(row_sum, row_values, mask, 1.0f32);
        let output = exp(values - max_values) / sum_values;
        store_vector(out, offsets, output, mask);
    }

    #[cutile::entry()]
    pub unsafe fn softmax_normalize_f64(
        out: *mut f64,
        input: *mut f64,
        row_max: *mut f64,
        row_sum: *mut f64,
        cols: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let values = load_vector(input, offsets, mask, 0.0f64);
        let row_values = softmax_vector_rows(offsets, cols);
        let max_values = load_vector(row_max, row_values, mask, 0.0f64);
        let sum_values = load_vector(row_sum, row_values, mask, 1.0f64);
        let output = exp(values - max_values) / sum_values;
        store_vector(out, offsets, output, mask);
    }

    #[cutile::entry()]
    pub unsafe fn softmin_normalize_f16(
        out: *mut f16,
        input: *mut f16,
        row_min: *mut f32,
        row_sum: *mut f32,
        cols: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let values = load_vector(input, offsets, mask, f16::from_f32(0.0));
        let values: Tile<f32, { [128] }> = convert_tile(values);
        let row_values = softmax_vector_rows(offsets, cols);
        let min_values = load_vector(row_min, row_values, mask, 0.0f32);
        let sum_values = load_vector(row_sum, row_values, mask, 1.0f32);
        let output = exp(min_values - values) / sum_values;
        store_vector(out, offsets, convert_tile(output), mask);
    }

    #[cutile::entry()]
    pub unsafe fn softmin_normalize_f32(
        out: *mut f32,
        input: *mut f32,
        row_min: *mut f32,
        row_sum: *mut f32,
        cols: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let values = load_vector(input, offsets, mask, 0.0f32);
        let row_values = softmax_vector_rows(offsets, cols);
        let min_values = load_vector(row_min, row_values, mask, 0.0f32);
        let sum_values = load_vector(row_sum, row_values, mask, 1.0f32);
        let output = exp(min_values - values) / sum_values;
        store_vector(out, offsets, output, mask);
    }

    #[cutile::entry()]
    pub unsafe fn softmin_normalize_f64(
        out: *mut f64,
        input: *mut f64,
        row_min: *mut f64,
        row_sum: *mut f64,
        cols: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let values = load_vector(input, offsets, mask, 0.0f64);
        let row_values = softmax_vector_rows(offsets, cols);
        let min_values = load_vector(row_min, row_values, mask, 0.0f64);
        let sum_values = load_vector(row_sum, row_values, mask, 1.0f64);
        let output = exp(min_values - values) / sum_values;
        store_vector(out, offsets, output, mask);
    }

    #[cutile::entry()]
    pub unsafe fn log_softmax_normalize_f16(
        out: *mut f16,
        input: *mut f16,
        row_max: *mut f32,
        row_sum: *mut f32,
        cols: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let values = load_vector(input, offsets, mask, f16::from_f32(0.0));
        let values: Tile<f32, { [128] }> = convert_tile(values);
        let row_values = softmax_vector_rows(offsets, cols);
        let max_values = load_vector(row_max, row_values, mask, 0.0f32);
        let sum_values = load_vector(row_sum, row_values, mask, 1.0f32);
        let output = values - max_values - log(sum_values);
        store_vector(out, offsets, convert_tile(output), mask);
    }

    #[cutile::entry()]
    pub unsafe fn log_softmax_normalize_f32(
        out: *mut f32,
        input: *mut f32,
        row_max: *mut f32,
        row_sum: *mut f32,
        cols: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let values = load_vector(input, offsets, mask, 0.0f32);
        let row_values = softmax_vector_rows(offsets, cols);
        let max_values = load_vector(row_max, row_values, mask, 0.0f32);
        let sum_values = load_vector(row_sum, row_values, mask, 1.0f32);
        let output = values - max_values - log(sum_values);
        store_vector(out, offsets, output, mask);
    }

    #[cutile::entry()]
    pub unsafe fn log_softmax_normalize_f64(
        out: *mut f64,
        input: *mut f64,
        row_max: *mut f64,
        row_sum: *mut f64,
        cols: i32,
        len: i32,
    ) {
        let (offsets, mask) = vector_offsets(len);
        let values = load_vector(input, offsets, mask, 0.0f64);
        let row_values = softmax_vector_rows(offsets, cols);
        let max_values = load_vector(row_max, row_values, mask, 0.0f64);
        let sum_values = load_vector(row_sum, row_values, mask, 1.0f64);
        let output = values - max_values - log(sum_values);
        store_vector(out, offsets, output, mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_f16(
        out: *mut f16,
        input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(
            input,
            args.current_offsets,
            args.output_mask,
            f16::from_f32(0.0),
        );
        let other = load_vector(
            input,
            args.other_offsets,
            args.in_rotary,
            f16::from_f32(0.0),
        );
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, f16::from_f32(1.0));
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, f16::from_f32(0.0));
        let current: Tile<f32, { [128] }> = convert_tile(current);
        let other: Tile<f32, { [128] }> = convert_tile(other);
        let cos_values: Tile<f32, { [128] }> = convert_tile(cos_values);
        let sin_values: Tile<f32, { [128] }> = convert_tile(sin_values);
        let first_values = current * cos_values - other * sin_values;
        let second_values = current * cos_values + other * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, convert_tile(values), args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_f32(
        out: *mut f32,
        input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f32);
        let other = load_vector(input, args.other_offsets, args.in_rotary, 0.0f32);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f32);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f32);
        let first_values = current * cos_values - other * sin_values;
        let second_values = current * cos_values + other * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_f64(
        out: *mut f64,
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f64);
        let other = load_vector(input, args.other_offsets, args.in_rotary, 0.0f64);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f64);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f64);
        let first_values = current * cos_values - other * sin_values;
        let second_values = current * cos_values + other * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_positioned_f16(
        out: *mut f16,
        input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: i32,
        len: i32,
    ) {
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(
            input,
            args.current_offsets,
            args.output_mask,
            f16::from_f32(0.0),
        );
        let other = load_vector(
            input,
            args.other_offsets,
            args.in_rotary,
            f16::from_f32(0.0),
        );
        let current_f32: Tile<f32, { [128] }> = convert_tile(current);
        let other_f32: Tile<f32, { [128] }> = convert_tile(other);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, f16::from_f32(1.0));
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, f16::from_f32(0.0));
        let cos_values: Tile<f32, { [128] }> = convert_tile(cos_values);
        let sin_values: Tile<f32, { [128] }> = convert_tile(sin_values);
        let first_values = current_f32 * cos_values - other_f32 * sin_values;
        let second_values = current_f32 * cos_values + other_f32 * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current_f32);
        store_vector(out, args.offsets, convert_tile(values), args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_dynpos_f16(
        out: *mut f16,
        input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: *mut u32,
        len: i32,
    ) {
        let position_start = load_position_start(position_start);
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(
            input,
            args.current_offsets,
            args.output_mask,
            f16::from_f32(0.0),
        );
        let other = load_vector(
            input,
            args.other_offsets,
            args.in_rotary,
            f16::from_f32(0.0),
        );
        let current_f32: Tile<f32, { [128] }> = convert_tile(current);
        let other_f32: Tile<f32, { [128] }> = convert_tile(other);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, f16::from_f32(1.0));
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, f16::from_f32(0.0));
        let cos_values: Tile<f32, { [128] }> = convert_tile(cos_values);
        let sin_values: Tile<f32, { [128] }> = convert_tile(sin_values);
        let first_values = current_f32 * cos_values - other_f32 * sin_values;
        let second_values = current_f32 * cos_values + other_f32 * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current_f32);
        store_vector(out, args.offsets, convert_tile(values), args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_positioned_f16(
        out: *mut f16,
        input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(
            input,
            args.current_offsets,
            args.output_mask,
            f16::from_f32(0.0),
        );
        let even = load_vector(input, args.even_offsets, args.in_rotary, f16::from_f32(0.0));
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, f16::from_f32(0.0));
        let current_f32: Tile<f32, { [128] }> = convert_tile(current);
        let even_f32: Tile<f32, { [128] }> = convert_tile(even);
        let odd_f32: Tile<f32, { [128] }> = convert_tile(odd);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, f16::from_f32(1.0));
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, f16::from_f32(0.0));
        let cos_values: Tile<f32, { [128] }> = convert_tile(cos_values);
        let sin_values: Tile<f32, { [128] }> = convert_tile(sin_values);
        let even_values = even_f32 * cos_values - odd_f32 * sin_values;
        let odd_values = odd_f32 * cos_values + even_f32 * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current_f32);
        store_vector(out, args.offsets, convert_tile(values), args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_dynpos_f16(
        out: *mut f16,
        input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: *mut u32,
        len: i32,
    ) {
        let position_start = load_position_start(position_start);
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(
            input,
            args.current_offsets,
            args.output_mask,
            f16::from_f32(0.0),
        );
        let even = load_vector(input, args.even_offsets, args.in_rotary, f16::from_f32(0.0));
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, f16::from_f32(0.0));
        let current_f32: Tile<f32, { [128] }> = convert_tile(current);
        let even_f32: Tile<f32, { [128] }> = convert_tile(even);
        let odd_f32: Tile<f32, { [128] }> = convert_tile(odd);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, f16::from_f32(1.0));
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, f16::from_f32(0.0));
        let cos_values: Tile<f32, { [128] }> = convert_tile(cos_values);
        let sin_values: Tile<f32, { [128] }> = convert_tile(sin_values);
        let even_values = even_f32 * cos_values - odd_f32 * sin_values;
        let odd_values = odd_f32 * cos_values + even_f32 * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current_f32);
        store_vector(out, args.offsets, convert_tile(values), args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_positioned_f32(
        out: *mut f32,
        input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: i32,
        len: i32,
    ) {
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f32);
        let other = load_vector(input, args.other_offsets, args.in_rotary, 0.0f32);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f32);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f32);
        let first_values = current * cos_values - other * sin_values;
        let second_values = current * cos_values + other * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_dynpos_f32(
        out: *mut f32,
        input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: *mut u32,
        len: i32,
    ) {
        let position_start = load_position_start(position_start);
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f32);
        let other = load_vector(input, args.other_offsets, args.in_rotary, 0.0f32);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f32);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f32);
        let first_values = current * cos_values - other * sin_values;
        let second_values = current * cos_values + other * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_positioned_f32(
        out: *mut f32,
        input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f32);
        let even = load_vector(input, args.even_offsets, args.in_rotary, 0.0f32);
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, 0.0f32);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f32);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f32);
        let even_values = even * cos_values - odd * sin_values;
        let odd_values = odd * cos_values + even * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_dynpos_f32(
        out: *mut f32,
        input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: *mut u32,
        len: i32,
    ) {
        let position_start = load_position_start(position_start);
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f32);
        let even = load_vector(input, args.even_offsets, args.in_rotary, 0.0f32);
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, 0.0f32);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f32);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f32);
        let even_values = even * cos_values - odd * sin_values;
        let odd_values = odd * cos_values + even * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_positioned_f64(
        out: *mut f64,
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: i32,
        len: i32,
    ) {
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f64);
        let other = load_vector(input, args.other_offsets, args.in_rotary, 0.0f64);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f64);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f64);
        let first_values = current * cos_values - other * sin_values;
        let second_values = current * cos_values + other * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_dynpos_f64(
        out: *mut f64,
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: *mut u32,
        len: i32,
    ) {
        let position_start = load_position_start(position_start);
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f64);
        let other = load_vector(input, args.other_offsets, args.in_rotary, 0.0f64);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f64);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f64);
        let first_values = current * cos_values - other * sin_values;
        let second_values = current * cos_values + other * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_positioned_f64(
        out: *mut f64,
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f64);
        let even = load_vector(input, args.even_offsets, args.in_rotary, 0.0f64);
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, 0.0f64);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f64);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f64);
        let even_values = even * cos_values - odd * sin_values;
        let odd_values = odd * cos_values + even * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_dynpos_f64(
        out: *mut f64,
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        position_start: *mut u32,
        len: i32,
    ) {
        let position_start = load_position_start(position_start);
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            position_start,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f64);
        let even = load_vector(input, args.even_offsets, args.in_rotary, 0.0f64);
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, 0.0f64);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f64);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f64);
        let even_values = even * cos_values - odd * sin_values;
        let odd_values = odd * cos_values + even * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_batched_f32(
        out: *mut f32,
        input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_batch_stride: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_batch_stride: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            cos_batch_stride,
            sin_batch_stride,
            trig_batch,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f32);
        let even = load_vector(input, args.even_offsets, args.in_rotary, 0.0f32);
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, 0.0f32);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f32);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f32);
        let even_values = even * cos_values - odd * sin_values;
        let odd_values = odd * cos_values + even * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_batched_f16(
        out: *mut f16,
        input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_batch_stride: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_batch_stride: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            cos_batch_stride,
            sin_batch_stride,
            trig_batch,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(
            input,
            args.current_offsets,
            args.output_mask,
            f16::from_f32(0.0),
        );
        let even = load_vector(input, args.even_offsets, args.in_rotary, f16::from_f32(0.0));
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, f16::from_f32(0.0));
        let current_f32: Tile<f32, { [128] }> = convert_tile(current);
        let even_f32: Tile<f32, { [128] }> = convert_tile(even);
        let odd_f32: Tile<f32, { [128] }> = convert_tile(odd);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, f16::from_f32(1.0));
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, f16::from_f32(0.0));
        let cos_values: Tile<f32, { [128] }> = convert_tile(cos_values);
        let sin_values: Tile<f32, { [128] }> = convert_tile(sin_values);
        let even_values = even_f32 * cos_values - odd_f32 * sin_values;
        let odd_values = odd_f32 * cos_values + even_f32 * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current_f32);
        store_vector(out, args.offsets, convert_tile(values), args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_batched_f64(
        out: *mut f64,
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_batch_stride: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_batch_stride: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            cos_batch_stride,
            sin_batch_stride,
            trig_batch,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f64);
        let even = load_vector(input, args.even_offsets, args.in_rotary, 0.0f64);
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, 0.0f64);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f64);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f64);
        let even_values = even * cos_values - odd * sin_values;
        let odd_values = odd * cos_values + even * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_qk_f32(
        q_out: *mut f32,
        k_out: *mut f32,
        q_input: *mut f32,
        k_input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _q_output_dim0: i32,
        q_output_dim1: i32,
        q_output_dim2: i32,
        q_output_dim3: i32,
        q_input_stride0: i32,
        q_input_stride1: i32,
        q_input_stride2: i32,
        q_input_stride3: i32,
        _k_output_dim0: i32,
        k_output_dim1: i32,
        k_output_dim2: i32,
        k_output_dim3: i32,
        k_input_stride0: i32,
        k_input_stride1: i32,
        k_input_stride2: i32,
        k_input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        q_len: i32,
        k_len: i32,
    ) {
        rotary_interleaved_f32(
            q_out,
            q_input,
            cos,
            sin,
            q_output_dim1,
            q_output_dim2,
            q_output_dim3,
            q_input_stride0,
            q_input_stride1,
            q_input_stride2,
            q_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            q_len,
        );
        rotary_interleaved_f32(
            k_out,
            k_input,
            cos,
            sin,
            k_output_dim1,
            k_output_dim2,
            k_output_dim3,
            k_input_stride0,
            k_input_stride1,
            k_input_stride2,
            k_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            k_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_qk_f16(
        q_out: *mut f16,
        k_out: *mut f16,
        q_input: *mut f16,
        k_input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        _q_output_dim0: i32,
        q_output_dim1: i32,
        q_output_dim2: i32,
        q_output_dim3: i32,
        q_input_stride0: i32,
        q_input_stride1: i32,
        q_input_stride2: i32,
        q_input_stride3: i32,
        _k_output_dim0: i32,
        k_output_dim1: i32,
        k_output_dim2: i32,
        k_output_dim3: i32,
        k_input_stride0: i32,
        k_input_stride1: i32,
        k_input_stride2: i32,
        k_input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        q_len: i32,
        k_len: i32,
    ) {
        rotary_interleaved_f16(
            q_out,
            q_input,
            cos,
            sin,
            q_output_dim1,
            q_output_dim2,
            q_output_dim3,
            q_input_stride0,
            q_input_stride1,
            q_input_stride2,
            q_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            q_len,
        );
        rotary_interleaved_f16(
            k_out,
            k_input,
            cos,
            sin,
            k_output_dim1,
            k_output_dim2,
            k_output_dim3,
            k_input_stride0,
            k_input_stride1,
            k_input_stride2,
            k_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            k_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_qk_f64(
        q_out: *mut f64,
        k_out: *mut f64,
        q_input: *mut f64,
        k_input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        _q_output_dim0: i32,
        q_output_dim1: i32,
        q_output_dim2: i32,
        q_output_dim3: i32,
        q_input_stride0: i32,
        q_input_stride1: i32,
        q_input_stride2: i32,
        q_input_stride3: i32,
        _k_output_dim0: i32,
        k_output_dim1: i32,
        k_output_dim2: i32,
        k_output_dim3: i32,
        k_input_stride0: i32,
        k_input_stride1: i32,
        k_input_stride2: i32,
        k_input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        q_len: i32,
        k_len: i32,
    ) {
        rotary_interleaved_f64(
            q_out,
            q_input,
            cos,
            sin,
            q_output_dim1,
            q_output_dim2,
            q_output_dim3,
            q_input_stride0,
            q_input_stride1,
            q_input_stride2,
            q_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            q_len,
        );
        rotary_interleaved_f64(
            k_out,
            k_input,
            cos,
            sin,
            k_output_dim1,
            k_output_dim2,
            k_output_dim3,
            k_input_stride0,
            k_input_stride1,
            k_input_stride2,
            k_input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            rotary_pairs,
            k_len,
        );
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_batched_f32(
        out: *mut f32,
        input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_batch_stride: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_batch_stride: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            cos_batch_stride,
            sin_batch_stride,
            trig_batch,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f32);
        let other = load_vector(input, args.other_offsets, args.in_rotary, 0.0f32);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f32);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f32);
        let first_values = current * cos_values - other * sin_values;
        let second_values = current * cos_values + other * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_batched_f16(
        out: *mut f16,
        input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_batch_stride: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_batch_stride: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            cos_batch_stride,
            sin_batch_stride,
            trig_batch,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(
            input,
            args.current_offsets,
            args.output_mask,
            f16::from_f32(0.0),
        );
        let other = load_vector(
            input,
            args.other_offsets,
            args.in_rotary,
            f16::from_f32(0.0),
        );
        let current: Tile<f32, { [128] }> = convert_tile(current);
        let other: Tile<f32, { [128] }> = convert_tile(other);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, f16::from_f32(1.0));
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, f16::from_f32(0.0));
        let cos_values: Tile<f32, { [128] }> = convert_tile(cos_values);
        let sin_values: Tile<f32, { [128] }> = convert_tile(sin_values);
        let first_values = current * cos_values - other * sin_values;
        let second_values = current * cos_values + other * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, convert_tile(values), args.output_mask);
    }

    #[cutile::entry()]
    pub unsafe fn rotary_embedding_chunked_batched_f64(
        out: *mut f64,
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        _output_dim0: i32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_batch_stride: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_batch_stride: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_chunked_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            cos_batch_stride,
            sin_batch_stride,
            trig_batch,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f64);
        let other = load_vector(input, args.other_offsets, args.in_rotary, 0.0f64);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f64);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f64);
        let first_values = current * cos_values - other * sin_values;
        let second_values = current * cos_values + other * sin_values;
        let rotated = select(args.first_half, first_values, second_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    fn fill_impl<T: ElementType>(out: *mut T, value: T, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let values: Tile<T, { [128] }> = broadcast_scalar(value, tile_shape);
        store_vector(out, offsets, values, mask);
    }

    fn copy_impl<T: ElementType>(out: *mut T, input: *mut T, fill: T, len: i32) {
        let (offsets, mask) = vector_offsets(len);
        let values = load_vector(input, offsets, mask, fill);
        store_vector(out, offsets, values, mask);
    }

    fn transpose_2d_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        fill: T,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));
        let rows_tile = broadcast_scalar(rows, tile_shape);
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let output_rows = safe_offsets / rows_tile;
        let output_cols = safe_offsets - output_rows * rows_tile;
        let input_offsets = output_cols * cols_tile + output_rows;
        let values = load_vector(input, input_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn transpose_last2_rank3_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        fill: T,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));
        let rows_tile = broadcast_scalar(rows, tile_shape);
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let matrix_len = rows_tile * cols_tile;
        let batch_index = safe_offsets / matrix_len;
        let matrix_offset = safe_offsets - batch_index * matrix_len;
        let output_rows = matrix_offset / rows_tile;
        let output_cols = matrix_offset - output_rows * rows_tile;
        let input_offsets = batch_index * matrix_len + output_cols * cols_tile + output_rows;
        let values = load_vector(input, input_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn transpose_last2_rank4_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        fill: T,
        dim1: i32,
        rows: i32,
        cols: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));
        let dim1_tile = broadcast_scalar(dim1, tile_shape);
        let rows_tile = broadcast_scalar(rows, tile_shape);
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let matrix_len = rows_tile * cols_tile;
        let dim1_matrix_len = dim1_tile * matrix_len;
        let index0 = safe_offsets / dim1_matrix_len;
        let remaining0 = safe_offsets - index0 * dim1_matrix_len;
        let index1 = remaining0 / matrix_len;
        let matrix_offset = remaining0 - index1 * matrix_len;
        let output_rows = matrix_offset / rows_tile;
        let output_cols = matrix_offset - output_rows * rows_tile;
        let input_offsets =
            index0 * dim1_matrix_len + index1 * matrix_len + output_cols * cols_tile + output_rows;
        let values = load_vector(input, input_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn materialize_rank2_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        fill: T,
        dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));
        let dim1_tile = broadcast_scalar(dim1, tile_shape);
        let index0 = safe_offsets / dim1_tile;
        let index1 = safe_offsets - index0 * dim1_tile;
        let input_offsets = index0 * broadcast_scalar(input_stride0, tile_shape)
            + index1 * broadcast_scalar(input_stride1, tile_shape);
        let values = load_vector(input, input_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn materialize_rank3_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        fill: T,
        dim1: i32,
        dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));
        let dim1_dim2 = dim1 * dim2;
        let dim1_dim2_tile = broadcast_scalar(dim1_dim2, tile_shape);
        let dim2_tile = broadcast_scalar(dim2, tile_shape);
        let index0 = safe_offsets / dim1_dim2_tile;
        let remaining0 = safe_offsets - index0 * dim1_dim2_tile;
        let index1 = remaining0 / dim2_tile;
        let index2 = remaining0 - index1 * dim2_tile;
        let input_offsets = index0 * broadcast_scalar(input_stride0, tile_shape)
            + index1 * broadcast_scalar(input_stride1, tile_shape)
            + index2 * broadcast_scalar(input_stride2, tile_shape);
        let values = load_vector(input, input_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn materialize_rank4_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        fill: T,
        dim1: i32,
        dim2: i32,
        dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));
        let dim1_dim2_dim3 = dim1 * dim2 * dim3;
        let dim2_dim3 = dim2 * dim3;
        let dim1_dim2_dim3_tile = broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let dim2_dim3_tile = broadcast_scalar(dim2_dim3, tile_shape);
        let dim3_tile = broadcast_scalar(dim3, tile_shape);
        let index0 = safe_offsets / dim1_dim2_dim3_tile;
        let remaining0 = safe_offsets - index0 * dim1_dim2_dim3_tile;
        let index1 = remaining0 / dim2_dim3_tile;
        let remaining1 = remaining0 - index1 * dim2_dim3_tile;
        let index2 = remaining1 / dim3_tile;
        let index3 = remaining1 - index2 * dim3_tile;
        let input_offsets = index0 * broadcast_scalar(input_stride0, tile_shape)
            + index1 * broadcast_scalar(input_stride1, tile_shape)
            + index2 * broadcast_scalar(input_stride2, tile_shape)
            + index3 * broadcast_scalar(input_stride3, tile_shape);
        let values = load_vector(input, input_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn slice_rank2_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        fill: T,
        output_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        start0: i32,
        start1: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));
        let output_dim1_tile = broadcast_scalar(output_dim1, tile_shape);
        let coord0 = safe_offsets / output_dim1_tile;
        let coord1 = safe_offsets - coord0 * output_dim1_tile;
        let source_offsets = (coord0 + broadcast_scalar(start0, tile_shape))
            * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 + broadcast_scalar(start1, tile_shape))
                * broadcast_scalar(input_stride1, tile_shape);
        let values = load_vector(input, source_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn slice_rank3_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        fill: T,
        output_dim1: i32,
        output_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));
        let dim1_dim2 = output_dim1 * output_dim2;
        let dim1_dim2_tile = broadcast_scalar(dim1_dim2, tile_shape);
        let dim2_tile = broadcast_scalar(output_dim2, tile_shape);
        let coord0 = safe_offsets / dim1_dim2_tile;
        let remaining0 = safe_offsets - coord0 * dim1_dim2_tile;
        let coord1 = remaining0 / dim2_tile;
        let coord2 = remaining0 - coord1 * dim2_tile;
        let source_offsets = (coord0 + broadcast_scalar(start0, tile_shape))
            * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 + broadcast_scalar(start1, tile_shape))
                * broadcast_scalar(input_stride1, tile_shape)
            + (coord2 + broadcast_scalar(start2, tile_shape))
                * broadcast_scalar(input_stride2, tile_shape);
        let values = load_vector(input, source_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn slice_rank4_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        fill: T,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        start0: i32,
        start1: i32,
        start2: i32,
        start3: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));
        let dim1_dim2_dim3 = output_dim1 * output_dim2 * output_dim3;
        let dim2_dim3 = output_dim2 * output_dim3;
        let dim1_dim2_dim3_tile = broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let dim2_dim3_tile = broadcast_scalar(dim2_dim3, tile_shape);
        let dim3_tile = broadcast_scalar(output_dim3, tile_shape);
        let coord0 = safe_offsets / dim1_dim2_dim3_tile;
        let remaining0 = safe_offsets - coord0 * dim1_dim2_dim3_tile;
        let coord1 = remaining0 / dim2_dim3_tile;
        let remaining1 = remaining0 - coord1 * dim2_dim3_tile;
        let coord2 = remaining1 / dim3_tile;
        let coord3 = remaining1 - coord2 * dim3_tile;
        let source_offsets = (coord0 + broadcast_scalar(start0, tile_shape))
            * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 + broadcast_scalar(start1, tile_shape))
                * broadcast_scalar(input_stride1, tile_shape)
            + (coord2 + broadcast_scalar(start2, tile_shape))
                * broadcast_scalar(input_stride2, tile_shape)
            + (coord3 + broadcast_scalar(start3, tile_shape))
                * broadcast_scalar(input_stride3, tile_shape);
        let values = load_vector(input, source_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn pad_rank2_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        output_dim1: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_stride0: i32,
        input_stride1: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_value: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));
        let output_dim1_tile = broadcast_scalar(output_dim1, tile_shape);
        let coord0 = safe_offsets / output_dim1_tile;
        let coord1 = safe_offsets - coord0 * output_dim1_tile;

        let pad0 = broadcast_scalar(pad_before0, tile_shape);
        let pad1 = broadcast_scalar(pad_before1, tile_shape);
        let input_dim0_end = broadcast_scalar(pad_before0 + input_dim0, tile_shape);
        let input_dim1_end = broadcast_scalar(pad_before1 + input_dim1, tile_shape);
        let valid0 = cmpi(coord0, pad0, predicate::GreaterThanOrEqual)
            & cmpi(coord0, input_dim0_end, predicate::LessThan);
        let valid1 = cmpi(coord1, pad1, predicate::GreaterThanOrEqual)
            & cmpi(coord1, input_dim1_end, predicate::LessThan);
        let input_mask = output_mask & valid0 & valid1;

        let source_offsets = (coord0 - pad0) * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 - pad1) * broadcast_scalar(input_stride1, tile_shape);
        let values = load_vector(input, source_offsets, input_mask, pad_value);
        store_vector(out, offsets, values, output_mask);
    }

    fn pad_rank3_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        output_dim1: i32,
        output_dim2: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_value: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));

        let dim1_dim2 = output_dim1 * output_dim2;
        let dim1_dim2_tile = broadcast_scalar(dim1_dim2, tile_shape);
        let dim2_tile = broadcast_scalar(output_dim2, tile_shape);
        let coord0 = safe_offsets / dim1_dim2_tile;
        let remaining0 = safe_offsets - coord0 * dim1_dim2_tile;
        let coord1 = remaining0 / dim2_tile;
        let coord2 = remaining0 - coord1 * dim2_tile;

        let pad0 = broadcast_scalar(pad_before0, tile_shape);
        let pad1 = broadcast_scalar(pad_before1, tile_shape);
        let pad2 = broadcast_scalar(pad_before2, tile_shape);
        let input_dim0_end = broadcast_scalar(pad_before0 + input_dim0, tile_shape);
        let input_dim1_end = broadcast_scalar(pad_before1 + input_dim1, tile_shape);
        let input_dim2_end = broadcast_scalar(pad_before2 + input_dim2, tile_shape);
        let valid0 = cmpi(coord0, pad0, predicate::GreaterThanOrEqual)
            & cmpi(coord0, input_dim0_end, predicate::LessThan);
        let valid1 = cmpi(coord1, pad1, predicate::GreaterThanOrEqual)
            & cmpi(coord1, input_dim1_end, predicate::LessThan);
        let valid2 = cmpi(coord2, pad2, predicate::GreaterThanOrEqual)
            & cmpi(coord2, input_dim2_end, predicate::LessThan);
        let input_mask = output_mask & valid0 & valid1 & valid2;

        let source_offsets = (coord0 - pad0) * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 - pad1) * broadcast_scalar(input_stride1, tile_shape)
            + (coord2 - pad2) * broadcast_scalar(input_stride2, tile_shape);
        let values = load_vector(input, source_offsets, input_mask, pad_value);
        store_vector(out, offsets, values, output_mask);
    }

    fn pad_rank4_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_dim0: i32,
        input_dim1: i32,
        input_dim2: i32,
        input_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        pad_before0: i32,
        pad_before1: i32,
        pad_before2: i32,
        pad_before3: i32,
        pad_value: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let safe_offsets = select(output_mask, offsets, constant(0i32, tile_shape));

        let dim1_dim2_dim3 = output_dim1 * output_dim2 * output_dim3;
        let dim2_dim3 = output_dim2 * output_dim3;
        let dim1_dim2_dim3_tile = broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let dim2_dim3_tile = broadcast_scalar(dim2_dim3, tile_shape);
        let dim3_tile = broadcast_scalar(output_dim3, tile_shape);
        let coord0 = safe_offsets / dim1_dim2_dim3_tile;
        let remaining0 = safe_offsets - coord0 * dim1_dim2_dim3_tile;
        let coord1 = remaining0 / dim2_dim3_tile;
        let remaining1 = remaining0 - coord1 * dim2_dim3_tile;
        let coord2 = remaining1 / dim3_tile;
        let coord3 = remaining1 - coord2 * dim3_tile;

        let pad0 = broadcast_scalar(pad_before0, tile_shape);
        let pad1 = broadcast_scalar(pad_before1, tile_shape);
        let pad2 = broadcast_scalar(pad_before2, tile_shape);
        let pad3 = broadcast_scalar(pad_before3, tile_shape);
        let input_dim0_end = broadcast_scalar(pad_before0 + input_dim0, tile_shape);
        let input_dim1_end = broadcast_scalar(pad_before1 + input_dim1, tile_shape);
        let input_dim2_end = broadcast_scalar(pad_before2 + input_dim2, tile_shape);
        let input_dim3_end = broadcast_scalar(pad_before3 + input_dim3, tile_shape);
        let valid0 = cmpi(coord0, pad0, predicate::GreaterThanOrEqual)
            & cmpi(coord0, input_dim0_end, predicate::LessThan);
        let valid1 = cmpi(coord1, pad1, predicate::GreaterThanOrEqual)
            & cmpi(coord1, input_dim1_end, predicate::LessThan);
        let valid2 = cmpi(coord2, pad2, predicate::GreaterThanOrEqual)
            & cmpi(coord2, input_dim2_end, predicate::LessThan);
        let valid3 = cmpi(coord3, pad3, predicate::GreaterThanOrEqual)
            & cmpi(coord3, input_dim3_end, predicate::LessThan);
        let input_mask = output_mask & valid0 & valid1 & valid2 & valid3;

        let source_offsets = (coord0 - pad0) * broadcast_scalar(input_stride0, tile_shape)
            + (coord1 - pad1) * broadcast_scalar(input_stride1, tile_shape)
            + (coord2 - pad2) * broadcast_scalar(input_stride2, tile_shape)
            + (coord3 - pad3) * broadcast_scalar(input_stride3, tile_shape);
        let values = load_vector(input, source_offsets, input_mask, pad_value);
        store_vector(out, offsets, values, output_mask);
    }

    fn concat_rank2_impl<const AXIS: i32, T: ElementType>(
        out: *mut T,
        lhs: *mut T,
        rhs: *mut T,
        fill: T,
        output_dim1: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let zero_offsets = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero_offsets);

        let output_dim1_tile = broadcast_scalar(output_dim1, tile_shape);
        let coord0 = safe_offsets / output_dim1_tile;
        let coord1 = safe_offsets - coord0 * output_dim1_tile;

        let lhs_axis_size = if AXIS == 0 { lhs_dim0 } else { lhs_dim1 };
        let axis_coord = if AXIS == 0 { coord0 } else { coord1 };
        let lhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::LessThan,
            );
        let rhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::GreaterThanOrEqual,
            );

        let rhs_coord0 = if AXIS == 0 {
            coord0 - broadcast_scalar(lhs_axis_size, tile_shape)
        } else {
            coord0
        };
        let rhs_coord1 = if AXIS == 1 {
            coord1 - broadcast_scalar(lhs_axis_size, tile_shape)
        } else {
            coord1
        };

        let lhs_offsets = coord0 * broadcast_scalar(lhs_stride0, tile_shape)
            + coord1 * broadcast_scalar(lhs_stride1, tile_shape);
        let rhs_offsets = rhs_coord0 * broadcast_scalar(rhs_stride0, tile_shape)
            + rhs_coord1 * broadcast_scalar(rhs_stride1, tile_shape);
        let lhs_offsets = select(lhs_mask, lhs_offsets, zero_offsets);
        let rhs_offsets = select(rhs_mask, rhs_offsets, zero_offsets);
        let lhs_values = load_vector(lhs, lhs_offsets, lhs_mask, fill);
        let rhs_values = load_vector(rhs, rhs_offsets, rhs_mask, fill);
        store_vector(
            out,
            offsets,
            select(lhs_mask, lhs_values, rhs_values),
            output_mask,
        );
    }

    fn concat_rank3_impl<const AXIS: i32, T: ElementType>(
        out: *mut T,
        lhs: *mut T,
        rhs: *mut T,
        fill: T,
        output_dim1: i32,
        output_dim2: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let zero_offsets = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero_offsets);

        let dim1_dim2 = output_dim1 * output_dim2;
        let dim1_dim2_tile = broadcast_scalar(dim1_dim2, tile_shape);
        let dim2_tile = broadcast_scalar(output_dim2, tile_shape);
        let coord0 = safe_offsets / dim1_dim2_tile;
        let remaining0 = safe_offsets - coord0 * dim1_dim2_tile;
        let coord1 = remaining0 / dim2_tile;
        let coord2 = remaining0 - coord1 * dim2_tile;

        let lhs_axis_size = if AXIS == 0 {
            lhs_dim0
        } else if AXIS == 1 {
            lhs_dim1
        } else {
            lhs_dim2
        };
        let axis_coord = if AXIS == 0 {
            coord0
        } else if AXIS == 1 {
            coord1
        } else {
            coord2
        };
        let lhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::LessThan,
            );
        let rhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::GreaterThanOrEqual,
            );

        let rhs_coord0 = if AXIS == 0 {
            coord0 - broadcast_scalar(lhs_axis_size, tile_shape)
        } else {
            coord0
        };
        let rhs_coord1 = if AXIS == 1 {
            coord1 - broadcast_scalar(lhs_axis_size, tile_shape)
        } else {
            coord1
        };
        let rhs_coord2 = if AXIS == 2 {
            coord2 - broadcast_scalar(lhs_axis_size, tile_shape)
        } else {
            coord2
        };

        let lhs_offsets = coord0 * broadcast_scalar(lhs_stride0, tile_shape)
            + coord1 * broadcast_scalar(lhs_stride1, tile_shape)
            + coord2 * broadcast_scalar(lhs_stride2, tile_shape);
        let rhs_offsets = rhs_coord0 * broadcast_scalar(rhs_stride0, tile_shape)
            + rhs_coord1 * broadcast_scalar(rhs_stride1, tile_shape)
            + rhs_coord2 * broadcast_scalar(rhs_stride2, tile_shape);
        let lhs_offsets = select(lhs_mask, lhs_offsets, zero_offsets);
        let rhs_offsets = select(rhs_mask, rhs_offsets, zero_offsets);
        let lhs_values = load_vector(lhs, lhs_offsets, lhs_mask, fill);
        let rhs_values = load_vector(rhs, rhs_offsets, rhs_mask, fill);
        store_vector(
            out,
            offsets,
            select(lhs_mask, lhs_values, rhs_values),
            output_mask,
        );
    }

    fn concat_rank4_impl<const AXIS: i32, T: ElementType>(
        out: *mut T,
        lhs: *mut T,
        rhs: *mut T,
        fill: T,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        lhs_dim0: i32,
        lhs_dim1: i32,
        lhs_dim2: i32,
        lhs_dim3: i32,
        lhs_stride0: i32,
        lhs_stride1: i32,
        lhs_stride2: i32,
        lhs_stride3: i32,
        rhs_stride0: i32,
        rhs_stride1: i32,
        rhs_stride2: i32,
        rhs_stride3: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let zero_offsets = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero_offsets);

        let dim1_dim2_dim3 = output_dim1 * output_dim2 * output_dim3;
        let dim2_dim3 = output_dim2 * output_dim3;
        let dim1_dim2_dim3_tile = broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let dim2_dim3_tile = broadcast_scalar(dim2_dim3, tile_shape);
        let dim3_tile = broadcast_scalar(output_dim3, tile_shape);
        let coord0 = safe_offsets / dim1_dim2_dim3_tile;
        let remaining0 = safe_offsets - coord0 * dim1_dim2_dim3_tile;
        let coord1 = remaining0 / dim2_dim3_tile;
        let remaining1 = remaining0 - coord1 * dim2_dim3_tile;
        let coord2 = remaining1 / dim3_tile;
        let coord3 = remaining1 - coord2 * dim3_tile;

        let lhs_axis_size = if AXIS == 0 {
            lhs_dim0
        } else if AXIS == 1 {
            lhs_dim1
        } else if AXIS == 2 {
            lhs_dim2
        } else {
            lhs_dim3
        };
        let axis_coord = if AXIS == 0 {
            coord0
        } else if AXIS == 1 {
            coord1
        } else if AXIS == 2 {
            coord2
        } else {
            coord3
        };
        let lhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::LessThan,
            );
        let rhs_mask = output_mask
            & cmpi(
                axis_coord,
                broadcast_scalar(lhs_axis_size, tile_shape),
                predicate::GreaterThanOrEqual,
            );

        let axis_offset = broadcast_scalar(lhs_axis_size, tile_shape);
        let rhs_coord0 = if AXIS == 0 {
            coord0 - axis_offset
        } else {
            coord0
        };
        let rhs_coord1 = if AXIS == 1 {
            coord1 - axis_offset
        } else {
            coord1
        };
        let rhs_coord2 = if AXIS == 2 {
            coord2 - axis_offset
        } else {
            coord2
        };
        let rhs_coord3 = if AXIS == 3 {
            coord3 - axis_offset
        } else {
            coord3
        };

        let lhs_offsets = coord0 * broadcast_scalar(lhs_stride0, tile_shape)
            + coord1 * broadcast_scalar(lhs_stride1, tile_shape)
            + coord2 * broadcast_scalar(lhs_stride2, tile_shape)
            + coord3 * broadcast_scalar(lhs_stride3, tile_shape);
        let rhs_offsets = rhs_coord0 * broadcast_scalar(rhs_stride0, tile_shape)
            + rhs_coord1 * broadcast_scalar(rhs_stride1, tile_shape)
            + rhs_coord2 * broadcast_scalar(rhs_stride2, tile_shape)
            + rhs_coord3 * broadcast_scalar(rhs_stride3, tile_shape);
        let lhs_offsets = select(lhs_mask, lhs_offsets, zero_offsets);
        let rhs_offsets = select(rhs_mask, rhs_offsets, zero_offsets);
        let lhs_values = load_vector(lhs, lhs_offsets, lhs_mask, fill);
        let rhs_values = load_vector(rhs, rhs_offsets, rhs_mask, fill);
        store_vector(
            out,
            offsets,
            select(lhs_mask, lhs_values, rhs_values),
            output_mask,
        );
    }

    fn repeat_kv_rank4_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        fill: T,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        repeats: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let zero_offsets = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero_offsets);

        let dim1_dim2_dim3 = output_dim1 * output_dim2 * output_dim3;
        let dim2_dim3 = output_dim2 * output_dim3;
        let dim1_dim2_dim3_tile = broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let dim2_dim3_tile = broadcast_scalar(dim2_dim3, tile_shape);
        let dim3_tile = broadcast_scalar(output_dim3, tile_shape);
        let coord0 = safe_offsets / dim1_dim2_dim3_tile;
        let remaining0 = safe_offsets - coord0 * dim1_dim2_dim3_tile;
        let coord1 = remaining0 / dim2_dim3_tile;
        let remaining1 = remaining0 - coord1 * dim2_dim3_tile;
        let coord2 = remaining1 / dim3_tile;
        let coord3 = remaining1 - coord2 * dim3_tile;
        let input_head = coord1 / broadcast_scalar(repeats, tile_shape);

        let source_offsets = coord0 * broadcast_scalar(input_stride0, tile_shape)
            + input_head * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + coord3 * broadcast_scalar(input_stride3, tile_shape);
        let values = load_vector(input, source_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn where_bool_impl<T: ElementType>(
        out: *mut T,
        condition: *mut u8,
        x: *mut T,
        y: *mut T,
        fill: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let condition_tile = load_vector(condition, offsets, output_mask, 0u8);
        let x_tile = load_vector(x, offsets, output_mask, fill);
        let y_tile = load_vector(y, offsets, output_mask, fill);
        let zero = constant(0u8, tile_shape);
        let mask = cmpi(condition_tile, zero, predicate::NotEqual);
        store_vector(out, offsets, select(mask, x_tile, y_tile), output_mask);
    }

    fn masked_fill_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        mask: *mut u8,
        value: T,
        fill: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, fill);
        let mask_tile = load_vector(mask, offsets, output_mask, 0u8);
        let zero = constant(0u8, tile_shape);
        let keep = cmpi(mask_tile, zero, predicate::NotEqual);
        let value = broadcast_scalar(value, tile_shape);
        store_vector(out, offsets, select(keep, value, input_tile), output_mask);
    }

    fn clamp_impl<T: ElementType>(out: *mut T, input: *mut T, min: T, max: T, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, min);
        let min_tile = broadcast_scalar(min, tile_shape);
        let max_tile = broadcast_scalar(max, tile_shape);
        let above_min = select(
            cmpi(input_tile, min_tile, predicate::GreaterThanOrEqual),
            input_tile,
            min_tile,
        );
        let clamped = select(
            cmpi(above_min, max_tile, predicate::LessThanOrEqual),
            above_min,
            max_tile,
        );
        store_vector(out, offsets, clamped, output_mask);
    }

    fn arithmetic_impl<const OP: i32, T: ElementType>(
        out: *mut T,
        lhs: *mut T,
        rhs: *mut T,
        fill: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let lhs_tile = load_vector(lhs, offsets, output_mask, fill);
        let rhs_tile = load_vector(rhs, offsets, output_mask, fill);
        let value = if OP == 0 {
            lhs_tile + rhs_tile
        } else if OP == 1 {
            lhs_tile - rhs_tile
        } else if OP == 2 {
            lhs_tile * rhs_tile
        } else if OP == 3 {
            lhs_tile / rhs_tile
        } else {
            lhs_tile % rhs_tile
        };
        store_vector(out, offsets, value, output_mask);
    }

    fn arithmetic_scalar_impl<const OP: i32, T: ElementType>(
        out: *mut T,
        input: *mut T,
        scalar: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, scalar);
        let scalar_tile = broadcast_scalar(scalar, tile_shape);
        let value = if OP == 0 {
            input_tile + scalar_tile
        } else if OP == 1 {
            input_tile - scalar_tile
        } else if OP == 2 {
            scalar_tile - input_tile
        } else if OP == 3 {
            input_tile * scalar_tile
        } else if OP == 4 {
            input_tile / scalar_tile
        } else if OP == 5 {
            scalar_tile / input_tile
        } else if OP == 6 {
            input_tile % scalar_tile
        } else {
            scalar_tile % input_tile
        };
        store_vector(out, offsets, value, output_mask);
    }

    fn bitwise_impl<const OP: i32, T: ElementType>(
        out: *mut T,
        lhs: *mut T,
        rhs: *mut T,
        fill: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let lhs_tile = load_vector(lhs, offsets, output_mask, fill);
        let rhs_tile = load_vector(rhs, offsets, output_mask, fill);
        let value = if OP == 0 {
            andi(lhs_tile, rhs_tile)
        } else if OP == 1 {
            ori(lhs_tile, rhs_tile)
        } else {
            xori(lhs_tile, rhs_tile)
        };
        store_vector(out, offsets, value, output_mask);
    }

    fn bitwise_scalar_impl<const OP: i32, T: ElementType>(
        out: *mut T,
        input: *mut T,
        scalar: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, scalar);
        let scalar_tile = broadcast_scalar(scalar, tile_shape);
        let value = if OP == 0 {
            andi(input_tile, scalar_tile)
        } else if OP == 1 {
            ori(input_tile, scalar_tile)
        } else {
            xori(input_tile, scalar_tile)
        };
        store_vector(out, offsets, value, output_mask);
    }

    fn bitwise_not_impl<T: ElementType>(out: *mut T, input: *mut T, fill: T, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let input_tile = load_vector(input, offsets, output_mask, fill);
        store_vector(out, offsets, noti(input_tile), output_mask);
    }

    fn cast_int_impl<T: ElementType, U: ElementType>(
        out: *mut U,
        input: *mut T,
        fill: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let input_tile = load_vector(input, offsets, output_mask, fill);
        let output_tile: Tile<U, { [128] }> = convert_tile(input_tile);
        store_vector(out, offsets, output_tile, output_mask);
    }

    fn cast_bool_int_impl<U: ElementType>(out: *mut U, input: *mut u8, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, 0u8);
        let zero_u8 = constant(0u8, tile_shape);
        let one_u8 = constant(1u8, tile_shape);
        let canonical_u8 = select(
            cmpi(input_tile, zero_u8, predicate::NotEqual),
            one_u8,
            zero_u8,
        );
        let output_tile: Tile<U, { [128] }> = convert_tile(canonical_u8);
        store_vector(out, offsets, output_tile, output_mask);
    }

    fn shift_impl<const OP: i32, T: ElementType>(
        out: *mut T,
        lhs: *mut T,
        rhs: *mut T,
        fill: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let lhs_tile = load_vector(lhs, offsets, output_mask, fill);
        let rhs_tile = load_vector(rhs, offsets, output_mask, fill);
        let value = if OP == 0 {
            shli(lhs_tile, rhs_tile, overflow::None)
        } else {
            shri(lhs_tile, rhs_tile)
        };
        store_vector(out, offsets, value, output_mask);
    }

    fn shift_scalar_impl<const OP: i32, T: ElementType>(
        out: *mut T,
        input: *mut T,
        scalar: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, scalar);
        let scalar_tile = broadcast_scalar(scalar, tile_shape);
        let value = if OP == 0 {
            shli(input_tile, scalar_tile, overflow::None)
        } else {
            shri(input_tile, scalar_tile)
        };
        store_vector(out, offsets, value, output_mask);
    }

    fn neg_impl<T: ElementType>(out: *mut T, input: *mut T, zero: T, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let input_tile = load_vector(input, offsets, output_mask, zero);
        let zero_tile = constant(zero, const_shape![128]);
        store_vector(out, offsets, zero_tile - input_tile, output_mask);
    }

    fn abs_signed_impl<T: ElementType>(out: *mut T, input: *mut T, zero: T, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, zero);
        let zero_tile = constant(zero, tile_shape);
        let negative = cmpi(input_tile, zero_tile, predicate::LessThan);
        store_vector(
            out,
            offsets,
            select(negative, zero_tile - input_tile, input_tile),
            output_mask,
        );
    }

    fn sign_signed_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        zero: T,
        one: T,
        negative_one: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, zero);
        let zero_tile = constant(zero, tile_shape);
        let one_tile = constant(one, tile_shape);
        let negative_one_tile = constant(negative_one, tile_shape);
        let positive = cmpi(input_tile, zero_tile, predicate::GreaterThan);
        let negative = cmpi(input_tile, zero_tile, predicate::LessThan);
        let value = select(positive, one_tile, zero_tile);
        let value = select(negative, negative_one_tile, value);
        store_vector(out, offsets, value, output_mask);
    }

    fn sign_unsigned_impl<T: ElementType>(out: *mut T, input: *mut T, zero: T, one: T, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, zero);
        let zero_tile = constant(zero, tile_shape);
        let one_tile = constant(one, tile_shape);
        let nonzero = cmpi(input_tile, zero_tile, predicate::NotEqual);
        store_vector(
            out,
            offsets,
            select(nonzero, one_tile, zero_tile),
            output_mask,
        );
    }

    fn compare_u32_impl<const OP: i32>(out: *mut u8, lhs: *mut u32, rhs: *mut u32, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let lhs_tile = load_vector(lhs, offsets, output_mask, 0u32);
        let rhs_tile = load_vector(rhs, offsets, output_mask, 0u32);
        let cmp = if OP == 0 {
            cmpi(lhs_tile, rhs_tile, predicate::Equal)
        } else if OP == 1 {
            cmpi(lhs_tile, rhs_tile, predicate::NotEqual)
        } else if OP == 2 {
            cmpi(lhs_tile, rhs_tile, predicate::LessThan)
        } else if OP == 3 {
            cmpi(lhs_tile, rhs_tile, predicate::LessThanOrEqual)
        } else if OP == 4 {
            cmpi(lhs_tile, rhs_tile, predicate::GreaterThan)
        } else {
            cmpi(lhs_tile, rhs_tile, predicate::GreaterThanOrEqual)
        };
        let zero = constant(0u8, tile_shape);
        let one = constant(1u8, tile_shape);
        store_vector(out, offsets, select(cmp, one, zero), output_mask);
    }

    fn compare_u8_impl<const OP: i32>(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let lhs_tile = load_vector(lhs, offsets, output_mask, 0u8);
        let rhs_tile = load_vector(rhs, offsets, output_mask, 0u8);
        let cmp = if OP == 0 {
            cmpi(lhs_tile, rhs_tile, predicate::Equal)
        } else if OP == 1 {
            cmpi(lhs_tile, rhs_tile, predicate::NotEqual)
        } else if OP == 2 {
            cmpi(lhs_tile, rhs_tile, predicate::LessThan)
        } else if OP == 3 {
            cmpi(lhs_tile, rhs_tile, predicate::LessThanOrEqual)
        } else if OP == 4 {
            cmpi(lhs_tile, rhs_tile, predicate::GreaterThan)
        } else {
            cmpi(lhs_tile, rhs_tile, predicate::GreaterThanOrEqual)
        };
        let zero = constant(0u8, tile_shape);
        let one = constant(1u8, tile_shape);
        store_vector(out, offsets, select(cmp, one, zero), output_mask);
    }

    fn compare_i8_impl<const OP: i32>(out: *mut u8, lhs: *mut i8, rhs: *mut i8, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let lhs_tile = load_vector(lhs, offsets, output_mask, 0i8);
        let rhs_tile = load_vector(rhs, offsets, output_mask, 0i8);
        let cmp = if OP == 0 {
            cmpi(lhs_tile, rhs_tile, predicate::Equal)
        } else if OP == 1 {
            cmpi(lhs_tile, rhs_tile, predicate::NotEqual)
        } else if OP == 2 {
            cmpi(lhs_tile, rhs_tile, predicate::LessThan)
        } else if OP == 3 {
            cmpi(lhs_tile, rhs_tile, predicate::LessThanOrEqual)
        } else if OP == 4 {
            cmpi(lhs_tile, rhs_tile, predicate::GreaterThan)
        } else {
            cmpi(lhs_tile, rhs_tile, predicate::GreaterThanOrEqual)
        };
        let zero = constant(0u8, tile_shape);
        let one = constant(1u8, tile_shape);
        store_vector(out, offsets, select(cmp, one, zero), output_mask);
    }

    fn compare_impl<const OP: i32, T: ElementType>(
        out: *mut u8,
        lhs: *mut T,
        rhs: *mut T,
        fill: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let lhs_tile = load_vector(lhs, offsets, output_mask, fill);
        let rhs_tile = load_vector(rhs, offsets, output_mask, fill);
        let cmp = if OP == 0 {
            cmpi(lhs_tile, rhs_tile, predicate::Equal)
        } else if OP == 1 {
            cmpi(lhs_tile, rhs_tile, predicate::NotEqual)
        } else if OP == 2 {
            cmpi(lhs_tile, rhs_tile, predicate::LessThan)
        } else if OP == 3 {
            cmpi(lhs_tile, rhs_tile, predicate::LessThanOrEqual)
        } else if OP == 4 {
            cmpi(lhs_tile, rhs_tile, predicate::GreaterThan)
        } else {
            cmpi(lhs_tile, rhs_tile, predicate::GreaterThanOrEqual)
        };
        let zero = constant(0u8, tile_shape);
        let one = constant(1u8, tile_shape);
        store_vector(out, offsets, select(cmp, one, zero), output_mask);
    }

    fn compare_scalar_impl<const OP: i32, T: ElementType>(
        out: *mut u8,
        input: *mut T,
        scalar: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, scalar);
        let scalar_tile = broadcast_scalar(scalar, tile_shape);
        let cmp = if OP == 0 {
            cmpi(input_tile, scalar_tile, predicate::Equal)
        } else if OP == 1 {
            cmpi(input_tile, scalar_tile, predicate::NotEqual)
        } else if OP == 2 {
            cmpi(input_tile, scalar_tile, predicate::LessThan)
        } else if OP == 3 {
            cmpi(input_tile, scalar_tile, predicate::LessThanOrEqual)
        } else if OP == 4 {
            cmpi(input_tile, scalar_tile, predicate::GreaterThan)
        } else {
            cmpi(input_tile, scalar_tile, predicate::GreaterThanOrEqual)
        };
        let zero = constant(0u8, tile_shape);
        let one = constant(1u8, tile_shape);
        store_vector(out, offsets, select(cmp, one, zero), output_mask);
    }

    fn minmax_impl<const OP: i32, T: ElementType>(
        out: *mut T,
        lhs: *mut T,
        rhs: *mut T,
        fill: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let lhs_tile = load_vector(lhs, offsets, output_mask, fill);
        let rhs_tile = load_vector(rhs, offsets, output_mask, fill);
        let choose_lhs = if OP == 0 {
            cmpi(lhs_tile, rhs_tile, predicate::LessThanOrEqual)
        } else {
            cmpi(lhs_tile, rhs_tile, predicate::GreaterThanOrEqual)
        };
        store_vector(
            out,
            offsets,
            select(choose_lhs, lhs_tile, rhs_tile),
            output_mask,
        );
    }

    fn minmax_scalar_impl<const OP: i32, T: ElementType>(
        out: *mut T,
        input: *mut T,
        scalar: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let input_tile = load_vector(input, offsets, output_mask, scalar);
        let scalar_tile = broadcast_scalar(scalar, tile_shape);
        let choose_input = if OP == 0 {
            cmpi(input_tile, scalar_tile, predicate::LessThanOrEqual)
        } else {
            cmpi(input_tile, scalar_tile, predicate::GreaterThanOrEqual)
        };
        store_vector(
            out,
            offsets,
            select(choose_input, input_tile, scalar_tile),
            output_mask,
        );
    }

    fn gather_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        indices: *mut u32,
        fill: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let zero_offsets = constant(0i32, const_shape![128]);
        let safe_offsets = select(output_mask, offsets, zero_offsets);
        let indices_u32 = load_vector(indices, safe_offsets, output_mask, 0u32);
        let source_offsets: Tile<i32, { [128] }> = bitcast(indices_u32);
        let values = load_vector(input, source_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn gather_rows_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        row_indices: *mut u32,
        fill: T,
        cols: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let output_rows = offsets / cols_tile;
        let columns = offsets - output_rows * cols_tile;
        let zero_offsets = constant(0i32, tile_shape);
        let safe_rows = select(output_mask, output_rows, zero_offsets);
        let source_rows_u32 = load_vector(row_indices, safe_rows, output_mask, 0u32);
        let source_rows: Tile<i32, { [128] }> = bitcast(source_rows_u32);
        let source_offsets = source_rows * cols_tile + columns;
        let values = load_vector(input, source_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn gather_row_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        row_index: i32,
        fill: T,
        cols: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(cols);
        let tile_shape = const_shape![128];
        let source_offsets = broadcast_scalar(row_index * cols, tile_shape) + offsets;
        let values = load_vector(input, source_offsets, output_mask, fill);
        store_vector(out, offsets, values, output_mask);
    }

    fn scatter_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        indices: *mut u32,
        fill: T,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let zero_offsets = constant(0i32, const_shape![128]);
        let safe_offsets = select(output_mask, offsets, zero_offsets);
        let values = load_vector(input, safe_offsets, output_mask, fill);
        let indices_u32 = load_vector(indices, safe_offsets, output_mask, 0u32);
        let dest_offsets: Tile<i32, { [128] }> = bitcast(indices_u32);
        store_vector(out, dest_offsets, values, output_mask);
    }

    fn scatter_rows_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        row_indices: *mut u32,
        fill: T,
        cols: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let source_rows = offsets / cols_tile;
        let columns = offsets - source_rows * cols_tile;
        let zero_offsets = constant(0i32, tile_shape);
        let safe_rows = select(output_mask, source_rows, zero_offsets);
        let values = load_vector(input, offsets, output_mask, fill);
        let dest_rows_u32 = load_vector(row_indices, safe_rows, output_mask, 0u32);
        let dest_rows: Tile<i32, { [128] }> = bitcast(dest_rows_u32);
        let dest_offsets = dest_rows * cols_tile + columns;
        store_vector(out, dest_offsets, values, output_mask);
    }

    fn copy_indexed_rows_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        source_indices: *mut u32,
        output_indices: *mut u32,
        fill: T,
        cols: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let logical_rows = offsets / cols_tile;
        let columns = offsets - logical_rows * cols_tile;
        let zero_offsets = constant(0i32, tile_shape);
        let safe_rows = select(output_mask, logical_rows, zero_offsets);
        let source_rows_u32 = load_vector(source_indices, safe_rows, output_mask, 0u32);
        let source_rows: Tile<i32, { [128] }> = bitcast(source_rows_u32);
        let source_offsets = source_rows * cols_tile + columns;
        let values = load_vector(input, source_offsets, output_mask, fill);
        let output_rows_u32 = load_vector(output_indices, safe_rows, output_mask, 0u32);
        let output_rows: Tile<i32, { [128] }> = bitcast(output_rows_u32);
        let output_offsets = output_rows * cols_tile + columns;
        store_vector(out, output_offsets, values, output_mask);
    }

    fn mask_binary_impl<const OP: i32>(out: *mut u8, lhs: *mut u8, rhs: *mut u8, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let lhs_tile = load_vector(lhs, offsets, output_mask, 0u8);
        let rhs_tile = load_vector(rhs, offsets, output_mask, 0u8);
        let zero_u8 = constant(0u8, tile_shape);
        let one_u8 = constant(1u8, tile_shape);
        let lhs_mask = cmpi(lhs_tile, zero_u8, predicate::NotEqual);
        let rhs_mask = cmpi(rhs_tile, zero_u8, predicate::NotEqual);
        let mask = if OP == 0 {
            lhs_mask & rhs_mask
        } else if OP == 1 {
            lhs_mask | rhs_mask
        } else {
            let lhs_value = select(lhs_mask, one_u8, zero_u8);
            let rhs_value = select(rhs_mask, one_u8, zero_u8);
            cmpi(lhs_value, rhs_value, predicate::NotEqual)
        };
        store_vector(out, offsets, select(mask, one_u8, zero_u8), output_mask);
    }

    fn triangular_mask_impl<const OP: i32>(
        out: *mut u8,
        _rows: i32,
        cols: i32,
        diagonal: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let row = offsets / cols_tile;
        let col = offsets - row * cols_tile;
        let diagonal_tile = broadcast_scalar(diagonal, tile_shape);
        let keep = if OP == 0 {
            cmpi(col, row + diagonal_tile, predicate::LessThanOrEqual)
        } else {
            cmpi(col, row + diagonal_tile, predicate::GreaterThanOrEqual)
        };
        let zero_u8 = constant(0u8, tile_shape);
        let one_u8 = constant(1u8, tile_shape);
        store_vector(out, offsets, select(keep, one_u8, zero_u8), output_mask);
    }

    fn causal_mask_fill_impl<T: ElementType>(
        out: *mut T,
        input: *mut T,
        fill_value: T,
        zero_value: T,
        seq_len: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let seq_len_tile = broadcast_scalar(seq_len, tile_shape);
        let row_major_offset = offsets % (seq_len_tile * seq_len_tile);
        let row = row_major_offset / seq_len_tile;
        let col = row_major_offset - row * seq_len_tile;
        let keep = cmpi(col, row, predicate::LessThanOrEqual);
        let input_tile = load_vector(input, offsets, output_mask, zero_value);
        let fill_tile = broadcast_scalar(fill_value, tile_shape);
        store_vector(
            out,
            offsets,
            select(keep, input_tile, fill_tile),
            output_mask,
        );
    }

    fn sequence_mask_impl(out: *mut u8, lengths: *mut u32, cols: i32, len: i32) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let row = offsets / cols_tile;
        let col = offsets - row * cols_tile;
        let zero_offsets = constant(0i32, tile_shape);
        let safe_rows = select(output_mask, row, zero_offsets);
        let lengths_tile = load_vector(lengths, safe_rows, output_mask, 0u32);
        let col_u32: Tile<u32, { [128] }> = convert_tile(col);
        let keep = cmpi(col_u32, lengths_tile, predicate::LessThan);
        let zero_u8 = constant(0u8, tile_shape);
        let one_u8 = constant(1u8, tile_shape);
        store_vector(out, offsets, select(keep, one_u8, zero_u8), output_mask);
    }

    fn triangular_sequence_mask_impl<const OP: i32>(
        out: *mut u8,
        lengths: *mut u32,
        _rows: i32,
        cols: i32,
        diagonal: i32,
        len: i32,
    ) {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let cols_tile = broadcast_scalar(cols, tile_shape);
        let row = offsets / cols_tile;
        let col = offsets - row * cols_tile;
        let zero_offsets = constant(0i32, tile_shape);
        let safe_rows = select(output_mask, row, zero_offsets);
        let lengths_tile = load_vector(lengths, safe_rows, output_mask, 0u32);
        let col_u32: Tile<u32, { [128] }> = convert_tile(col);
        let valid_length = cmpi(col_u32, lengths_tile, predicate::LessThan);
        let diagonal_tile = broadcast_scalar(diagonal, tile_shape);
        let valid_triangle = if OP == 0 {
            cmpi(col, row + diagonal_tile, predicate::LessThanOrEqual)
        } else {
            cmpi(col, row + diagonal_tile, predicate::GreaterThanOrEqual)
        };
        let keep = valid_length & valid_triangle;
        let zero_u8 = constant(0u8, tile_shape);
        let one_u8 = constant(1u8, tile_shape);
        store_vector(out, offsets, select(keep, one_u8, zero_u8), output_mask);
    }

    fn mask_reduce_impl<const OP: i32, const BN: i32>(
        out: &mut Tensor<u8, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1i32, BN];
        let input_tile: Tile<u8, { [1, BN] }> = input.partition(tile_shape).load([pid.0, 0i32]);
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota.reshape(tile_shape);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], tile_shape);
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let zero_u8 = constant(0u8, tile_shape);
        let nonzero = cmpi(input_tile, zero_u8, predicate::NotEqual);
        let one_i32 = constant(1i32, tile_shape);
        let zero_i32 = constant(0i32, tile_shape);
        let true_count: Tile<i32, { [1] }> =
            reduce_sum(select(valid_cols & nonzero, one_i32, zero_i32), 1i32);
        let out_shape = const_shape![1i32, 1i32];
        let true_count = true_count.reshape(out_shape);
        let keep = if OP == 0 {
            cmpi(
                true_count,
                constant(0i32, out_shape),
                predicate::GreaterThan,
            )
        } else {
            cmpi(
                true_count,
                broadcast_scalar(input_shape[1], out_shape),
                predicate::Equal,
            )
        };
        let one_u8: Tile<u8, { [1, 1] }> = constant(1u8, out_shape);
        let zero_u8: Tile<u8, { [1, 1] }> = constant(0u8, out_shape);
        out.store(select(keep, one_u8, zero_u8));
    }

    fn mask_count_nonzero_impl<const BN: i32>(
        out: &mut Tensor<u32, { [1, 1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1i32, BN];
        let input_tile: Tile<u8, { [1, BN] }> = input.partition(tile_shape).load([pid.0, 0i32]);
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota.reshape(tile_shape);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], tile_shape);
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let zero_u8 = constant(0u8, tile_shape);
        let nonzero = cmpi(input_tile, zero_u8, predicate::NotEqual);
        let one_i32 = constant(1i32, tile_shape);
        let zero_i32 = constant(0i32, tile_shape);
        let true_count: Tile<i32, { [1] }> =
            reduce_sum(select(valid_cols & nonzero, one_i32, zero_i32), 1i32);
        let true_count = true_count.reshape(const_shape![1i32, 1i32]);
        let true_count: Tile<u32, { [1, 1] }> = convert_tile(true_count);
        out.store(true_count);
    }

    fn reduce_int_impl<const OP: i32, const BN: i32, T: ElementType>(
        out: &mut Tensor<T, { [1, 1] }>,
        input: &Tensor<T, { [-1, -1] }>,
        zero: T,
        min_value: T,
        max_value: T,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<T, { [1, BN] }> = input.partition(tile_shape).load([pid.0, 0i32]);
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota.reshape(tile_shape);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], tile_shape);
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let value: Tile<T, { [1] }> = if OP == 0 {
            let fill = constant(zero, tile_shape);
            reduce_sum(select(valid_cols, input_tile_raw, fill), 1i32)
        } else if OP == 1 {
            let fill = constant(min_value, tile_shape);
            reduce_max(select(valid_cols, input_tile_raw, fill), 1i32)
        } else {
            let fill = constant(max_value, tile_shape);
            reduce_min(select(valid_cols, input_tile_raw, fill), 1i32)
        };
        out.store(value.reshape(const_shape![1i32, 1i32]));
    }

    fn arg_reduce_int_impl<const OP: i32, const BN: i32, T: ElementType>(
        out: &mut Tensor<i32, { [1, 2] }>,
        input: &Tensor<T, { [-1, -1] }>,
        min_value: T,
        max_value: T,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1i32, BN];
        let input_tile_raw: Tile<T, { [1, BN] }> = input.partition(tile_shape).load([pid.0, 0i32]);
        let col_iota: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let col_iota = col_iota.reshape(tile_shape);
        let input_shape: [i32; 2] = get_tensor_shape(input);
        let cols_tile = broadcast_scalar(input_shape[1], tile_shape);
        let valid_cols = cmpi(col_iota, cols_tile, predicate::LessThan);
        let candidates = if OP == 0 {
            let fill = constant(min_value, tile_shape);
            select(valid_cols, input_tile_raw, fill)
        } else {
            let fill = constant(max_value, tile_shape);
            select(valid_cols, input_tile_raw, fill)
        };
        let best_value: Tile<T, { [1] }> = if OP == 0 {
            reduce_max(candidates, 1i32)
        } else {
            reduce_min(candidates, 1i32)
        };
        let best_value = best_value
            .reshape(const_shape![1i32, 1i32])
            .broadcast(tile_shape);
        let best = valid_cols & cmpi(input_tile_raw, best_value, predicate::Equal);
        let sentinel: Tile<i32, { [1, BN] }> = broadcast_scalar(input_shape[1], tile_shape);
        let best_indices = select(best, col_iota, sentinel);
        let best_index: Tile<i32, { [1] }> = reduce_min(best_indices, 1i32);
        let lane_iota: Tile<i32, { [2] }> = iota(const_shape![2]);
        let lane_iota = lane_iota.reshape(const_shape![1i32, 2i32]);
        let zero_pair: Tile<i32, { [1, 2] }> = constant(0i32, const_shape![1i32, 2i32]);
        let first_lane = cmpi(lane_iota, zero_pair, predicate::Equal);
        let best_index = best_index
            .reshape(const_shape![1i32, 1i32])
            .broadcast(const_shape![1i32, 2i32]);
        out.store(select(first_lane, best_index, zero_pair));
    }

    fn vector_offsets(len: i32) -> (Tile<i32, { [128] }>, Tile<bool, { [128] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![128];
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 128i32, tile_shape);
        let len_tile = broadcast_scalar(len, tile_shape);
        let mask = cmpi(offsets, len_tile, predicate::LessThan);
        (offsets, mask)
    }

    fn unpack_u4_impl(
        packed: *mut u8,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        high_first: i32,
    ) -> Tile<u8, { [128] }> {
        let tile_shape = const_shape![128];
        let two = broadcast_scalar(2i32, tile_shape);
        let byte_offsets = offsets / two;
        let within_pair = offsets - byte_offsets * two;
        let packed_bytes = load_vector(packed, byte_offsets, mask, 0u8);
        let nibble_mask = broadcast_scalar(15u8, tile_shape);
        let high = andi(
            shri(packed_bytes, broadcast_scalar(4u8, tile_shape)),
            nibble_mask,
        );
        let low = andi(packed_bytes, nibble_mask);

        let zero = broadcast_scalar(0i32, tile_shape);
        let one = broadcast_scalar(1i32, tile_shape);
        let high_first = cmpi(
            broadcast_scalar(high_first, tile_shape),
            one,
            predicate::Equal,
        );
        let odd = cmpi(within_pair, one, predicate::Equal);
        let even = cmpi(within_pair, zero, predicate::Equal);
        let take_high = select(high_first, even, odd);
        select(take_high, high, low)
    }

    fn load_block_argmax_tile<T: ElementType>(
        input: *mut T,
        fill: T,
        len: i32,
    ) -> Tile<T, { [256] }> {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![256];
        let offsets: Tile<i32, { [256] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 256i32, tile_shape);
        let mask = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        load_vector_256(input, offsets, mask, fill)
    }

    fn block_argmax_f32_values(
        block_max: *mut f32,
        block_idx: *mut i32,
        values: Tile<f32, { [256] }>,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![256];
        let offsets: Tile<i32, { [256] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 256i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let fill: Tile<f32, { [256] }> = constant(-1.0e30f32, tile_shape);
        let masked_values = select(valid, values, fill);
        let best_value: Tile<f32, { [1] }> = reduce_max(masked_values, 0i32);
        let best = best_value.reshape(const_shape![1i32]).broadcast(tile_shape);
        let matches = valid & cmpf(masked_values, best, predicate::Equal, cmp_ordering::Ordered);
        let invalid: Tile<i32, { [256] }> = constant(2_147_483_647i32, tile_shape);
        let candidate_indices = select(matches, offsets, invalid);
        let winner: Tile<i32, { [1] }> = reduce_min(candidate_indices, 0i32);
        store_block_scalar(block_max, best_value);
        store_block_scalar(block_idx, winner);
    }

    fn block_argmax_f64_values(
        block_max: *mut f64,
        block_idx: *mut i32,
        values: Tile<f64, { [256] }>,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![256];
        let offsets: Tile<i32, { [256] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 256i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let fill: Tile<f64, { [256] }> = constant(-1.7976931348623157e308f64, tile_shape);
        let masked_values = select(valid, values, fill);
        let best_value: Tile<f64, { [1] }> = reduce_max(masked_values, 0i32);
        let best = best_value.reshape(const_shape![1i32]).broadcast(tile_shape);
        let matches = valid & cmpf(masked_values, best, predicate::Equal, cmp_ordering::Ordered);
        let invalid: Tile<i32, { [256] }> = constant(2_147_483_647i32, tile_shape);
        let candidate_indices = select(matches, offsets, invalid);
        let winner: Tile<i32, { [1] }> = reduce_min(candidate_indices, 0i32);
        store_block_scalar(block_max, best_value);
        store_block_scalar(block_idx, winner);
    }

    fn block_argmin_f32_values(
        block_min: *mut f32,
        block_idx: *mut i32,
        values: Tile<f32, { [256] }>,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![256];
        let offsets: Tile<i32, { [256] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 256i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let fill: Tile<f32, { [256] }> = constant(1.0e30f32, tile_shape);
        let masked_values = select(valid, values, fill);
        let best_value: Tile<f32, { [1] }> = reduce_min(masked_values, 0i32);
        let best = best_value.reshape(const_shape![1i32]).broadcast(tile_shape);
        let matches = valid & cmpf(masked_values, best, predicate::Equal, cmp_ordering::Ordered);
        let invalid: Tile<i32, { [256] }> = constant(2_147_483_647i32, tile_shape);
        let candidate_indices = select(matches, offsets, invalid);
        let winner: Tile<i32, { [1] }> = reduce_min(candidate_indices, 0i32);
        store_block_scalar(block_min, best_value);
        store_block_scalar(block_idx, winner);
    }

    fn block_argmin_f64_values(
        block_min: *mut f64,
        block_idx: *mut i32,
        values: Tile<f64, { [256] }>,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![256];
        let offsets: Tile<i32, { [256] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 256i32, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(len, tile_shape),
            predicate::LessThan,
        );
        let fill: Tile<f64, { [256] }> = constant(1.7976931348623157e308f64, tile_shape);
        let masked_values = select(valid, values, fill);
        let best_value: Tile<f64, { [1] }> = reduce_min(masked_values, 0i32);
        let best = best_value.reshape(const_shape![1i32]).broadcast(tile_shape);
        let matches = valid & cmpf(masked_values, best, predicate::Equal, cmp_ordering::Ordered);
        let invalid: Tile<i32, { [256] }> = constant(2_147_483_647i32, tile_shape);
        let candidate_indices = select(matches, offsets, invalid);
        let winner: Tile<i32, { [1] }> = reduce_min(candidate_indices, 0i32);
        store_block_scalar(block_min, best_value);
        store_block_scalar(block_idx, winner);
    }

    fn softmax_chunk_offsets(cols: i32) -> (Tile<i32, { [1024] }>, Tile<bool, { [1024] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let columns: Tile<i32, { [1024] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * 1024i32, tile_shape);
        let valid = cmpi(
            columns,
            broadcast_scalar(cols, tile_shape),
            predicate::LessThan,
        );
        let offsets = broadcast_scalar(pid.1 * cols, tile_shape) + columns;
        (offsets, valid)
    }

    fn softmax_chunk_valid(cols: i32) -> Tile<bool, { [1024] }> {
        softmax_chunk_offsets(cols).1
    }

    fn softmax_chunk_load<T: ElementType>(
        input: *mut T,
        cols: i32,
        fill: T,
    ) -> Tile<T, { [1024] }> {
        let (offsets, valid) = softmax_chunk_offsets(cols);
        load_vector_1024(input, offsets, valid, fill)
    }

    fn softmax_chunk_load_f16(input: *mut f16, cols: i32) -> Tile<f32, { [1024] }> {
        let values = softmax_chunk_load(input, cols, f16::from_f32(0.0));
        convert_tile(values)
    }

    fn softmax_chunk_row_value<T: ElementType>(row_values: *mut T, fill: T) -> Tile<T, { [1024] }> {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [1024] }> = broadcast_scalar(pid.1, const_shape![1024]);
        let mask: Tile<bool, { [1024] }> = constant(true, const_shape![1024]);
        load_vector_1024(row_values, offsets, mask, fill)
    }

    fn softmax_row_scratch_load<T: ElementType>(
        scratch: *mut T,
        chunks: i32,
        fill: T,
    ) -> Tile<T, { [1024] }> {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![1024];
        let chunk_offsets: Tile<i32, { [1024] }> = iota(tile_shape);
        let valid = cmpi(
            chunk_offsets,
            broadcast_scalar(chunks, tile_shape),
            predicate::LessThan,
        );
        let offsets = broadcast_scalar(pid.0 * chunks, tile_shape) + chunk_offsets;
        load_vector_1024(scratch, offsets, valid, fill)
    }

    fn store_softmax_partial<T: ElementType>(
        partial: *mut T,
        chunks: i32,
        value: Tile<T, { [1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offset: Tile<i32, { [1] }> = broadcast_scalar(pid.1 * chunks + pid.0, const_shape![1]);
        store_scalar_at(partial, offset, value);
    }

    fn softmax_vector_rows(offsets: Tile<i32, { [128] }>, cols: i32) -> Tile<i32, { [128] }> {
        offsets / broadcast_scalar(cols, const_shape![128])
    }

    fn load_vector_256<T: ElementType>(
        input: *mut T,
        offsets: Tile<i32, { [256] }>,
        mask: Tile<bool, { [256] }>,
        fill: T,
    ) -> Tile<T, { [256] }> {
        let input_base: PointerTile<*mut T, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut T, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut T, { [256] }> = input_base.broadcast(const_shape![256]);
        let input_ptrs: PointerTile<*mut T, { [256] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<T, { [256] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(fill),
            None,
            Latency::<0>,
        );
        result.0
    }

    fn load_vector_1024<T: ElementType>(
        input: *mut T,
        offsets: Tile<i32, { [1024] }>,
        mask: Tile<bool, { [1024] }>,
        fill: T,
    ) -> Tile<T, { [1024] }> {
        let input_base: PointerTile<*mut T, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut T, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut T, { [1024] }> = input_base.broadcast(const_shape![1024]);
        let input_ptrs: PointerTile<*mut T, { [1024] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<T, { [1024] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(fill),
            None,
            Latency::<0>,
        );
        result.0
    }

    fn store_block_scalar<T: ElementType>(out: *mut T, value: Tile<T, { [1] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        store_scalar_at_scalar_offset(out, pid.0, value);
    }

    fn store_first<T: ElementType>(out: *mut T, value: Tile<T, { [1] }>) {
        store_scalar_at_scalar_offset(out, 0i32, value);
    }

    fn store_scalar_at_scalar_offset<T: ElementType>(
        out: *mut T,
        offset: i32,
        value: Tile<T, { [1] }>,
    ) {
        let tile_shape = const_shape![128];
        let lanes: Tile<i32, { [128] }> = iota(const_shape![128]);
        let offsets = broadcast_scalar(offset, tile_shape);
        let values = value.reshape(const_shape![1]).broadcast(tile_shape);
        let zero: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let mask: Tile<bool, { [128] }> = cmpi(lanes, zero, predicate::Equal);
        store_vector(out, offsets, values, mask);
    }

    fn store_scalar_at<T: ElementType>(
        out: *mut T,
        offset: Tile<i32, { [1] }>,
        value: Tile<T, { [1] }>,
    ) {
        let out_base: PointerTile<*mut T, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut T, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut T, { [1] }> = out_base.offset_tile(offset);
        let mask: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        store_ptr_tko(
            out_ptrs,
            value,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    fn rotary_interleaved_f32(
        out: *mut f32,
        input: *mut f32,
        cos: *mut f32,
        sin: *mut f32,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f32);
        let even = load_vector(input, args.even_offsets, args.in_rotary, 0.0f32);
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, 0.0f32);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f32);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f32);
        let even_values = even * cos_values - odd * sin_values;
        let odd_values = odd * cos_values + even * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    fn rotary_interleaved_f16(
        out: *mut f16,
        input: *mut f16,
        cos: *mut f16,
        sin: *mut f16,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(
            input,
            args.current_offsets,
            args.output_mask,
            f16::from_f32(0.0),
        );
        let even = load_vector(input, args.even_offsets, args.in_rotary, f16::from_f32(0.0));
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, f16::from_f32(0.0));
        let current_f32: Tile<f32, { [128] }> = convert_tile(current);
        let even_f32: Tile<f32, { [128] }> = convert_tile(even);
        let odd_f32: Tile<f32, { [128] }> = convert_tile(odd);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, f16::from_f32(1.0));
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, f16::from_f32(0.0));
        let cos_values: Tile<f32, { [128] }> = convert_tile(cos_values);
        let sin_values: Tile<f32, { [128] }> = convert_tile(sin_values);
        let even_values = even_f32 * cos_values - odd_f32 * sin_values;
        let odd_values = odd_f32 * cos_values + even_f32 * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current_f32);
        store_vector(out, args.offsets, convert_tile(values), args.output_mask);
    }

    fn rotary_interleaved_f64(
        out: *mut f64,
        input: *mut f64,
        cos: *mut f64,
        sin: *mut f64,
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        rotary_pairs: i32,
        len: i32,
    ) {
        let args = rotary_interleaved_offsets(
            output_dim1,
            output_dim2,
            output_dim3,
            input_stride0,
            input_stride1,
            input_stride2,
            input_stride3,
            cos_stride0,
            cos_stride1,
            sin_stride0,
            sin_stride1,
            0i32,
            0i32,
            1i32,
            rotary_pairs,
            0i32,
            len,
        );
        let current = load_vector(input, args.current_offsets, args.output_mask, 0.0f64);
        let even = load_vector(input, args.even_offsets, args.in_rotary, 0.0f64);
        let odd = load_vector(input, args.odd_offsets, args.in_rotary, 0.0f64);
        let cos_values = load_vector(cos, args.cos_offsets, args.in_rotary, 1.0f64);
        let sin_values = load_vector(sin, args.sin_offsets, args.in_rotary, 0.0f64);
        let even_values = even * cos_values - odd * sin_values;
        let odd_values = odd * cos_values + even * sin_values;
        let rotated = select(args.is_even, even_values, odd_values);
        let values = select(args.in_rotary, rotated, current);
        store_vector(out, args.offsets, values, args.output_mask);
    }

    struct RotaryInterleavedOffsets {
        offsets: Tile<i32, { [128] }>,
        output_mask: Tile<bool, { [128] }>,
        in_rotary: Tile<bool, { [128] }>,
        is_even: Tile<bool, { [128] }>,
        current_offsets: Tile<i32, { [128] }>,
        even_offsets: Tile<i32, { [128] }>,
        odd_offsets: Tile<i32, { [128] }>,
        cos_offsets: Tile<i32, { [128] }>,
        sin_offsets: Tile<i32, { [128] }>,
    }

    fn rotary_interleaved_offsets(
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        cos_batch_stride: i32,
        sin_batch_stride: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        position_start: i32,
        len: i32,
    ) -> RotaryInterleavedOffsets {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let zero = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero);

        let dim1_dim2_dim3 = output_dim1 * output_dim2 * output_dim3;
        let dim2_dim3 = output_dim2 * output_dim3;
        let coord0: Tile<i32, { [128] }> =
            safe_offsets / broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let remaining0: Tile<i32, { [128] }> =
            safe_offsets - coord0 * broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let coord1: Tile<i32, { [128] }> = remaining0 / broadcast_scalar(dim2_dim3, tile_shape);
        let remaining1: Tile<i32, { [128] }> =
            remaining0 - coord1 * broadcast_scalar(dim2_dim3, tile_shape);
        let coord2: Tile<i32, { [128] }> = remaining1 / broadcast_scalar(output_dim3, tile_shape);
        let coord3: Tile<i32, { [128] }> =
            remaining1 - coord2 * broadcast_scalar(output_dim3, tile_shape);

        let current_offsets = coord0 * broadcast_scalar(input_stride0, tile_shape)
            + coord1 * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + coord3 * broadcast_scalar(input_stride3, tile_shape);
        let pair = coord3 / broadcast_scalar(2i32, tile_shape);
        let even_coord3 = pair * broadcast_scalar(2i32, tile_shape);
        let odd_coord3 = even_coord3 + broadcast_scalar(1i32, tile_shape);
        let in_rotary = output_mask
            & cmpi(
                pair,
                broadcast_scalar(rotary_pairs, tile_shape),
                predicate::LessThan,
            );
        let is_even = cmpi(coord3, even_coord3, predicate::Equal);
        let even_offsets = coord0 * broadcast_scalar(input_stride0, tile_shape)
            + coord1 * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + even_coord3 * broadcast_scalar(input_stride3, tile_shape);
        let odd_offsets = coord0 * broadcast_scalar(input_stride0, tile_shape)
            + coord1 * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + odd_coord3 * broadcast_scalar(input_stride3, tile_shape);
        let trig_coord0 = if trig_batch == 1i32 { zero } else { coord0 };
        let position = coord2 + broadcast_scalar(position_start, tile_shape);
        let cos_offsets = trig_coord0 * broadcast_scalar(cos_batch_stride, tile_shape)
            + position * broadcast_scalar(cos_stride0, tile_shape)
            + pair * broadcast_scalar(cos_stride1, tile_shape);
        let sin_offsets = trig_coord0 * broadcast_scalar(sin_batch_stride, tile_shape)
            + position * broadcast_scalar(sin_stride0, tile_shape)
            + pair * broadcast_scalar(sin_stride1, tile_shape);

        RotaryInterleavedOffsets {
            offsets,
            output_mask,
            in_rotary,
            is_even,
            current_offsets: select(output_mask, current_offsets, zero),
            even_offsets: select(in_rotary, even_offsets, zero),
            odd_offsets: select(in_rotary, odd_offsets, zero),
            cos_offsets: select(in_rotary, cos_offsets, zero),
            sin_offsets: select(in_rotary, sin_offsets, zero),
        }
    }

    struct RotaryChunkedOffsets {
        offsets: Tile<i32, { [128] }>,
        output_mask: Tile<bool, { [128] }>,
        in_rotary: Tile<bool, { [128] }>,
        first_half: Tile<bool, { [128] }>,
        current_offsets: Tile<i32, { [128] }>,
        other_offsets: Tile<i32, { [128] }>,
        cos_offsets: Tile<i32, { [128] }>,
        sin_offsets: Tile<i32, { [128] }>,
    }

    fn rotary_chunked_offsets(
        output_dim1: i32,
        output_dim2: i32,
        output_dim3: i32,
        input_stride0: i32,
        input_stride1: i32,
        input_stride2: i32,
        input_stride3: i32,
        cos_stride0: i32,
        cos_stride1: i32,
        sin_stride0: i32,
        sin_stride1: i32,
        cos_batch_stride: i32,
        sin_batch_stride: i32,
        trig_batch: i32,
        rotary_pairs: i32,
        position_start: i32,
        len: i32,
    ) -> RotaryChunkedOffsets {
        let (offsets, output_mask) = vector_offsets(len);
        let tile_shape = const_shape![128];
        let zero = constant(0i32, tile_shape);
        let safe_offsets = select(output_mask, offsets, zero);

        let dim1_dim2_dim3 = output_dim1 * output_dim2 * output_dim3;
        let dim2_dim3 = output_dim2 * output_dim3;
        let coord0: Tile<i32, { [128] }> =
            safe_offsets / broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let remaining0: Tile<i32, { [128] }> =
            safe_offsets - coord0 * broadcast_scalar(dim1_dim2_dim3, tile_shape);
        let coord1: Tile<i32, { [128] }> = remaining0 / broadcast_scalar(dim2_dim3, tile_shape);
        let remaining1: Tile<i32, { [128] }> =
            remaining0 - coord1 * broadcast_scalar(dim2_dim3, tile_shape);
        let coord2: Tile<i32, { [128] }> = remaining1 / broadcast_scalar(output_dim3, tile_shape);
        let coord3: Tile<i32, { [128] }> =
            remaining1 - coord2 * broadcast_scalar(output_dim3, tile_shape);

        let current_offsets = coord0 * broadcast_scalar(input_stride0, tile_shape)
            + coord1 * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + coord3 * broadcast_scalar(input_stride3, tile_shape);

        let rotary_lanes = broadcast_scalar(rotary_pairs * 2i32, tile_shape);
        let in_rotary = output_mask & cmpi(coord3, rotary_lanes, predicate::LessThan);
        let first_half = cmpi(
            coord3,
            broadcast_scalar(rotary_pairs, tile_shape),
            predicate::LessThan,
        );
        let pair = select(
            first_half,
            coord3,
            coord3 - broadcast_scalar(rotary_pairs, tile_shape),
        );
        let other_coord3 = select(
            first_half,
            coord3 + broadcast_scalar(rotary_pairs, tile_shape),
            coord3 - broadcast_scalar(rotary_pairs, tile_shape),
        );
        let other_offsets = coord0 * broadcast_scalar(input_stride0, tile_shape)
            + coord1 * broadcast_scalar(input_stride1, tile_shape)
            + coord2 * broadcast_scalar(input_stride2, tile_shape)
            + other_coord3 * broadcast_scalar(input_stride3, tile_shape);
        let trig_coord0 = if trig_batch == 1i32 { zero } else { coord0 };
        let position = coord2 + broadcast_scalar(position_start, tile_shape);
        let cos_offsets = trig_coord0 * broadcast_scalar(cos_batch_stride, tile_shape)
            + position * broadcast_scalar(cos_stride0, tile_shape)
            + pair * broadcast_scalar(cos_stride1, tile_shape);
        let sin_offsets = trig_coord0 * broadcast_scalar(sin_batch_stride, tile_shape)
            + position * broadcast_scalar(sin_stride0, tile_shape)
            + pair * broadcast_scalar(sin_stride1, tile_shape);

        RotaryChunkedOffsets {
            offsets,
            output_mask,
            in_rotary,
            first_half,
            current_offsets: select(output_mask, current_offsets, zero),
            other_offsets: select(in_rotary, other_offsets, zero),
            cos_offsets: select(in_rotary, cos_offsets, zero),
            sin_offsets: select(in_rotary, sin_offsets, zero),
        }
    }

    fn load_position_start(position_start: *mut u32) -> i32 {
        let offset: Tile<i32, { [1] }> = constant(0i32, const_shape![1]);
        let mask: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        let base: PointerTile<*mut u32, { [] }> = pointer_to_tile(position_start);
        let base: PointerTile<*mut u32, { [1] }> = base.reshape(const_shape![1]);
        let ptrs: PointerTile<*mut u32, { [1] }> = base.offset_tile(offset);
        let result: (Tile<u32, { [1] }>, Token) = load_ptr_tko(
            ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u32),
            None,
            Latency::<0>,
        );
        let position_u32 = result.0;
        let position_i32: Tile<i32, { [1] }> = bitcast(position_u32);
        tile_to_scalar(position_i32.reshape(const_shape![]))
    }

    fn load_vector<T: ElementType>(
        input: *mut T,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
        fill: T,
    ) -> Tile<T, { [128] }> {
        let input_base: PointerTile<*mut T, { [] }> = pointer_to_tile(input);
        let input_base: PointerTile<*mut T, { [1] }> = input_base.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut T, { [128] }> = input_base.broadcast(const_shape![128]);
        let input_ptrs: PointerTile<*mut T, { [128] }> = input_ptrs.offset_tile(offsets);
        let result: (Tile<T, { [128] }>, Token) = load_ptr_tko(
            input_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(fill),
            None,
            Latency::<0>,
        );
        result.0
    }

    fn store_vector<T: ElementType>(
        out: *mut T,
        offsets: Tile<i32, { [128] }>,
        values: Tile<T, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let out_base: PointerTile<*mut T, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut T, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut T, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut T, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }
}

pub use kernels::*;
