//! Helpers every emitted CUDA C++ kernel defines before its function.

/// Conversions between storage and compute types (rounding to BF16 in hardware from `sm_80`, and to the 8-bit floating types in software), index checks that report a status code, NaN-propagating minimum and maximum, and block-wide reductions for row schedules, whose blocks have a multiple of 32 threads.
///
/// `singe_combine` selects a combiner by tag: 0 sums, 1 multiplies, 2 takes the maximum, 3 the minimum, and 4 the log of the sum of exponentials.
/// `singe_warp_reduce` gives every lane of each aligned group of `Lanes` lanes the same result, and `singe_block_reduce` every thread of a block, each combined in the same order on every launch.
/// `singe_softmax_step` folds one logit and value into a running maximum, a sum of exponentials, and a weighted sum, rescaling both sums when the maximum grows; `singe_warp_softmax` and `singe_block_softmax` merge the threads' triples and return the weighted sum over the sum.
pub const PRELUDE: &str = r#"__device__ __forceinline__ float load_half(unsigned short bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}
__device__ __forceinline__ unsigned short store_half(float value) {
    unsigned short bits;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    return bits;
}
// 8-bit floats with `E` exponent and `M` mantissa bits; `Finite` formats have no infinities and spend the all-ones exponent on finite values but NaN, as `float8_e4m3fn` does.
// Stores round to nearest even; values rounding above the largest finite value become NaN in finite formats and infinities otherwise, as PyTorch converts them.
// Shifts as PyTorch defines them: zero for a left shift outside `0..bits`, the sign's fill for a right one; the left shift is computed unsigned, so it wraps.
// Every amount is clamped in 64 bits before it shifts, since nvcc may fold a range check into PTX's clamping shift, whose amount has only 32 bits.
// A negative amount converts to a huge unsigned one, so it clamps as a large one does.
template <typename T> __device__ __forceinline__ T singe_shift_left(T value, T shift) {
    const unsigned long long amount = min(static_cast<unsigned long long>(shift), 64ULL);
    // Amounts from the type's width to 63 shift every bit of a narrower value out of the bits it keeps.
    return static_cast<T>(amount >= 64ULL ? 0ULL : static_cast<unsigned long long>(value) << amount);
}
template <typename T> __device__ __forceinline__ T singe_shift_right(T value, T shift) {
    constexpr unsigned long long bits = sizeof(T) * 8ULL;
    if constexpr (static_cast<T>(-1) < T(0)) {
        // Shifting a signed value by one less than its width leaves the sign's fill.
        return static_cast<T>(value >> min(static_cast<unsigned long long>(shift), bits - 1ULL));
    } else {
        const unsigned long long amount = min(static_cast<unsigned long long>(shift), bits);
        return amount >= bits ? T(0) : static_cast<T>(value >> amount);
    }
}
// An integer power, wrapping; a negative exponent gives the integer part of the reciprocal power, zero but for bases of one and minus one.
template <typename T> __device__ __forceinline__ T singe_integer_power(T base, T exponent) {
    if (exponent < T(0)) {
        if (base == T(1)) return T(1);
        if (base == static_cast<T>(-1)) return (exponent & T(1)) ? static_cast<T>(-1) : T(1);
        return T(0);
    }
    unsigned long long result = 1ULL, factor = static_cast<unsigned long long>(base);
    for (unsigned long long remaining = static_cast<unsigned long long>(exponent); remaining != 0ULL; remaining >>= 1) {
        if (remaining & 1ULL) result *= factor;
        factor *= factor;
    }
    return static_cast<T>(result);
}
template <int E, int M, bool Finite> __device__ __forceinline__ float singe_load_fp8(unsigned char bits) {
    const unsigned int exponent = (bits >> M) & ((1U << E) - 1U);
    const unsigned int mantissa = bits & ((1U << M) - 1U);
    const int bias = (1 << (E - 1)) - 1;
    float magnitude;
    if (exponent == (1U << E) - 1U && (!Finite || mantissa == (1U << M) - 1U)) {
        magnitude = (Finite || mantissa != 0U) ? __uint_as_float(0x7fc00000U) : __uint_as_float(0x7f800000U);
    } else if (exponent == 0U) {
        magnitude = ldexpf(static_cast<float>(mantissa), 1 - bias - M);
    } else {
        magnitude = ldexpf(static_cast<float>(mantissa | (1U << M)), static_cast<int>(exponent) - bias - M);
    }
    return (bits & 0x80U) ? -magnitude : magnitude;
}
template <int E, int M, bool Finite> __device__ __forceinline__ unsigned char singe_store_fp8(float value) {
    const unsigned int sign = (__float_as_uint(value) >> 24) & 0x80U;
    // NaN keeps its sign, and in finite formats so does an overflow, which becomes NaN.
    const unsigned char nan = static_cast<unsigned char>(sign | 0x7fU);
    if (value != value) return nan;
    const int bias = (1 << (E - 1)) - 1;
    const unsigned char overflow = Finite ? nan : static_cast<unsigned char>(sign | (((1U << E) - 1U) << M));
    const float magnitude = fabsf(value);
    if (magnitude == __uint_as_float(0x7f800000U)) return overflow;
    if (magnitude == 0.0f) return static_cast<unsigned char>(sign);
    const float largest = Finite ? ldexpf(static_cast<float>((2U << M) - 2U), (1 << E) - 1 - bias - M)
                                 : ldexpf(static_cast<float>((2U << M) - 1U), (1 << E) - 2 - bias - M);
    int binade;
    frexpf(magnitude, &binade);
    int exponent = binade - 1 < 1 - bias ? 1 - bias : binade - 1;
    const float units = rintf(ldexpf(magnitude, M - exponent));
    if (ldexpf(units, exponent - M) > largest) return overflow;
    unsigned int whole = static_cast<unsigned int>(units);
    if (whole >= (2U << M)) {
        whole >>= 1;
        exponent += 1;
    }
    const unsigned int field = whole < (1U << M) ? whole : ((static_cast<unsigned int>(exponent + bias) << M) | (whole - (1U << M)));
    return static_cast<unsigned char>(sign | field);
}
__device__ __forceinline__ float load_bfloat(unsigned short bits) {
    return __uint_as_float(static_cast<unsigned int>(bits) << 16);
}
__device__ __forceinline__ unsigned short store_bfloat(float value) {
#if __CUDA_ARCH__ >= 800
    unsigned short bits;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    return bits;
#else
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U)
        return static_cast<unsigned short>((bits >> 16) | 0x0040U);
    return static_cast<unsigned short>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
#endif
}
// Atomically combines an 8- or 16-bit element with `combine` by swapping the aligned 32-bit word that holds it.
template <typename Combine> __device__ __forceinline__ void singe_atomic_narrow(void* address, unsigned int bytes, Combine combine) {
    const unsigned long long byte_address = reinterpret_cast<unsigned long long>(address);
    unsigned int* word = reinterpret_cast<unsigned int*>(byte_address & ~3ULL);
    const unsigned int shift = static_cast<unsigned int>(byte_address & 3ULL) * 8U;
    const unsigned int mask = (bytes == 1U ? 0xffU : 0xffffU) << shift;
    unsigned int old = *word;
    unsigned int assumed;
    do {
        assumed = old;
        const unsigned int next = (combine((assumed & mask) >> shift) << shift) & mask;
        old = atomicCAS(word, assumed, (assumed & ~mask) | next);
    } while (assumed != old);
}
struct singe_add_integer {
    unsigned int value;
    __device__ unsigned int operator()(unsigned int current) const { return current + value; }
};
struct singe_add_half {
    float value;
    __device__ unsigned int operator()(unsigned int current) const { return store_half(load_half(static_cast<unsigned short>(current)) + value); }
};
struct singe_add_bfloat {
    float value;
    __device__ unsigned int operator()(unsigned int current) const { return store_bfloat(load_bfloat(static_cast<unsigned short>(current)) + value); }
};
template <int E, int M, bool Finite> struct singe_add_fp8 {
    float value;
    __device__ unsigned int operator()(unsigned int current) const { return singe_store_fp8<E, M, Finite>(singe_load_fp8<E, M, Finite>(static_cast<unsigned char>(current)) + value); }
};
struct singe_or_bool {
    unsigned int value;
    __device__ unsigned int operator()(unsigned int current) const { return (current | value) != 0U ? 1U : 0U; }
};
__device__ __forceinline__ unsigned long long singe_check_index(unsigned int* status, long long value, unsigned long long extent, bool wrap, unsigned int code) {
    if (wrap && value < 0) value += static_cast<long long>(extent);
    if (value < 0 || static_cast<unsigned long long>(value) >= extent) {
        if (status != nullptr) atomicCAS(status, 0U, code);
        return 0;
    }
    return static_cast<unsigned long long>(value);
}
__device__ __forceinline__ void singe_require(unsigned int* status, bool condition, unsigned int code) {
    if (!condition && status != nullptr) atomicCAS(status, 0U, code);
}
template <typename T> __device__ __forceinline__ T singe_max(T a, T b) {
    return (a != a) ? a : ((b != b || b > a) ? b : a);
}
template <typename T> __device__ __forceinline__ T singe_min(T a, T b) {
    return (a != a) ? a : ((b != b || b < a) ? b : a);
}
__device__ __forceinline__ float singe_exp(float value) { return expf(value); }
__device__ __forceinline__ double singe_exp(double value) { return exp(value); }
__device__ __forceinline__ float singe_log1p(float value) { return log1pf(value); }
__device__ __forceinline__ double singe_log1p(double value) { return log1p(value); }
__device__ __forceinline__ bool singe_is_negative_infinity(float value) { return value == -__uint_as_float(0x7f800000U); }
__device__ __forceinline__ bool singe_is_negative_infinity(double value) { return value == -__longlong_as_double(0x7ff0000000000000LL); }
template <typename T> __device__ __forceinline__ T singe_log_sum_exp(T a, T b) {
    const T largest = singe_max(a, b);
    // Two positive infinities would subtract to NaN, and a largest value of either infinity is the result.
    if (singe_is_negative_infinity(largest) || singe_is_negative_infinity(-largest)) return largest;
    return largest + singe_log1p(singe_exp(singe_min(a, b) - largest));
}
template <typename T, int Combiner> __device__ __forceinline__ T singe_combine(T a, T b) {
    if constexpr (Combiner == 0) return a + b;
    else if constexpr (Combiner == 1) return a * b;
    else if constexpr (Combiner == 2) return singe_max(a, b);
    else if constexpr (Combiner == 3) return singe_min(a, b);
    else return singe_log_sum_exp(a, b);
}
template <typename T> __device__ __forceinline__ void singe_softmax_step(T& largest, T& total, T& weighted, T logit, T value) {
    if (singe_is_negative_infinity(logit)) return;
    if (!(logit <= largest)) {
        const T scale = singe_exp(largest - logit);
        total *= scale;
        weighted *= scale;
        largest = logit;
    }
    const T weight = singe_exp(logit - largest);
    total += weight;
    weighted += weight * value;
}
template <typename T> __device__ __forceinline__ void singe_softmax_merge(T& largest, T& total, T& weighted, T other_largest, T other_total, T other_weighted) {
    const T next = singe_max(largest, other_largest);
    if (singe_is_negative_infinity(next)) return;
    const T mine = singe_exp(largest - next);
    const T theirs = singe_exp(other_largest - next);
    total = total * mine + other_total * theirs;
    weighted = weighted * mine + other_weighted * theirs;
    largest = next;
}
template <typename T, int Lanes> __device__ __forceinline__ void singe_warp_softmax_merge(T& largest, T& total, T& weighted) {
    for (int offset = Lanes / 2; offset > 0; offset /= 2) {
        const T other_largest = __shfl_xor_sync(0xffffffffu, largest, offset);
        const T other_total = __shfl_xor_sync(0xffffffffu, total, offset);
        const T other_weighted = __shfl_xor_sync(0xffffffffu, weighted, offset);
        singe_softmax_merge(largest, total, weighted, other_largest, other_total, other_weighted);
    }
}
template <typename T, int Lanes = 32> __device__ __forceinline__ T singe_warp_softmax(T largest, T total, T weighted) {
    singe_warp_softmax_merge<T, Lanes>(largest, total, weighted);
    return weighted / total;
}
template <typename T> __device__ T singe_block_softmax(T largest, T total, T weighted, T* partials) {
    singe_warp_softmax_merge<T, 32>(largest, total, weighted);
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        partials[3 * (threadIdx.x / 32)] = largest;
        partials[3 * (threadIdx.x / 32) + 1] = total;
        partials[3 * (threadIdx.x / 32) + 2] = weighted;
    }
    __syncthreads();
    largest = partials[0];
    total = partials[1];
    weighted = partials[2];
    for (unsigned int warp = 1; warp < blockDim.x / 32; ++warp) {
        singe_softmax_merge(largest, total, weighted, partials[3 * warp], partials[3 * warp + 1], partials[3 * warp + 2]);
    }
    return weighted / total;
}
template <typename T, bool Maximum> __device__ __forceinline__ bool singe_arg_better(T value, long long index, T best, long long best_index) {
    if (index < 0) return false;
    if (best_index < 0) return true;
    const bool unordered = value != value, best_unordered = best != best;
    if (unordered || best_unordered) return unordered && (!best_unordered || index < best_index);
    if (Maximum ? value > best : value < best) return true;
    return value == best && index < best_index;
}
template <typename T, bool Maximum> __device__ __forceinline__ void singe_arg_step(T& best, long long& best_index, T value, long long index) {
    if (singe_arg_better<T, Maximum>(value, index, best, best_index)) {
        best = value;
        best_index = index;
    }
}
template <typename T, bool Maximum, int Lanes> __device__ __forceinline__ void singe_warp_arg_merge(T& best, long long& best_index) {
    for (int offset = Lanes / 2; offset > 0; offset /= 2) {
        const T other = __shfl_xor_sync(0xffffffffu, best, offset);
        const long long other_index = __shfl_xor_sync(0xffffffffu, best_index, offset);
        singe_arg_step<T, Maximum>(best, best_index, other, other_index);
    }
}
template <typename T, bool Maximum, int Lanes = 32> __device__ __forceinline__ long long singe_warp_arg(T best, long long best_index) {
    singe_warp_arg_merge<T, Maximum, Lanes>(best, best_index);
    return best_index;
}
template <typename T, bool Maximum> __device__ long long singe_block_arg(T best, long long best_index, T* values, long long* indices) {
    singe_warp_arg_merge<T, Maximum, 32>(best, best_index);
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        values[threadIdx.x / 32] = best;
        indices[threadIdx.x / 32] = best_index;
    }
    __syncthreads();
    best = values[0];
    best_index = indices[0];
    for (unsigned int warp = 1; warp < blockDim.x / 32; ++warp) {
        singe_arg_step<T, Maximum>(best, best_index, values[warp], indices[warp]);
    }
    return best_index;
}
// Each thread's combination of the values of the threads before it in the block, `start` for the first; every thread of the block must call it.
template <typename T, int Combiner> __device__ T singe_block_exclusive_scan(T value, T* partials, T start) {
    const unsigned int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    T inclusive = value;
    for (int offset = 1; offset < 32; offset *= 2) {
        const T other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive = singe_combine<T, Combiner>(other, inclusive);
    }
    __syncthreads();
    if (lane == 31) partials[warp] = inclusive;
    __syncthreads();
    T before = start;
    for (unsigned int earlier = 0; earlier < warp; ++earlier) before = singe_combine<T, Combiner>(before, partials[earlier]);
    const T within = __shfl_up_sync(0xffffffffu, inclusive, 1);
    return lane == 0 ? before : singe_combine<T, Combiner>(before, within);
}
template <typename T, int Combiner, int Lanes = 32> __device__ __forceinline__ T singe_warp_reduce(T value) {
    for (int offset = Lanes / 2; offset > 0; offset /= 2) {
        value = singe_combine<T, Combiner>(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}
template <typename T, int Combiner> __device__ T singe_block_reduce(T value, T* partials) {
    value = singe_warp_reduce<T, Combiner>(value);
    __syncthreads();
    if (threadIdx.x % 32 == 0) partials[threadIdx.x / 32] = value;
    __syncthreads();
    value = partials[0];
    for (unsigned int warp = 1; warp < blockDim.x / 32; ++warp) {
        value = singe_combine<T, Combiner>(value, partials[warp]);
    }
    return value;
}
"#;
