#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

    /** Owns an nvCOMP C++ manager and its CPU/GPU execution settings. */
    typedef struct SingeNvcompManager SingeNvcompManager;

    /** Enumeration that defines how a buffer gets compressed by an nvCOMP manager. */
    typedef enum SingeNvcompBitstreamKind {
        /** Each input buffer is chunked according to manager setting and compressed in parallel.
     * Allows computation of checksums.
     * Adds custom header with nvCOMP metadata at the beginning of the compressed data. */
        SingeNvcompNative = 0,
        /** Compresses input data as is, just using underlying compression algorithm.
     * Does not add header with nvCOMP metadata. */
        SingeNvcompRaw = 1,
        /** Similar to RAW, but adds custom header with just uncompressed size at the beginning of the compressed data. */
        SingeNvcompWithUncompressedSize = 2
    } SingeNvcompBitstreamKind;

    /** Enumeration that defines how an nvCOMP manager trades off kernel latency
 * against concurrency with other work on the device. */
    typedef enum SingeNvcompExecutionPolicy {
        /** Use implementation optimized for concurrency. */
        SingeNvcompConcurrency = 0,
        /** Use implementation optimized for latency. */
        SingeNvcompLatency = 1
    } SingeNvcompExecutionPolicy;

    /** Enumeration that defines the checksum policy used by an nvCOMP manager. */
    typedef enum SingeNvcompChecksumPolicy {
        /** During compression, do not compute checksums.
     * During decompression, do not verify checksums. */
        SingeNvcompNoComputeNoVerify = 0,
        /** During compression, compute checksums.
     * During decompression, do not attempt to verify checksums. */
        SingeNvcompComputeAndNoVerify = 1,
        /** During compression, do not compute checksums.
     * During decompression, verify checksums if they were included. */
        SingeNvcompNoComputeAndVerifyIfPresent = 2,
        /** During compression, compute checksums.
     * During decompression, verify checksums if they were included. */
        SingeNvcompComputeAndVerifyIfPresent = 3,
        /** During compression, compute checksums.
     * During decompression, verify checksums.
     * A runtime error will be thrown upon configure_decompression if
     * checksums were not included in the compressed buffer. */
        SingeNvcompComputeAndVerify = 4
    } SingeNvcompChecksumPolicy;

    /** nvCOMP supported compression formats. */
    typedef enum SingeNvcompFormat {
        /** LZ4 block compression. */
        SingeNvcompLz4 = 0,
        /** Snappy compression. */
        SingeNvcompSnappy = 1,
        /** Asymmetric numeral systems compression. */
        SingeNvcompAns = 2,
        /** GDeflate compression. */
        SingeNvcompGdeflate = 3,
        /** Cascaded integer compression. */
        SingeNvcompCascaded = 4,
        /** Bitcomp compression. */
        SingeNvcompBitcomp = 5,
        /** Zstandard compression. */
        SingeNvcompZstd = 6,
        /** Deflate compression. */
        SingeNvcompDeflate = 7,
        /** Gzip compression. */
        SingeNvcompGzip = 8
    } SingeNvcompFormat;

    /* The bridge returns zero on success. It catches every C++ exception and copies
 * its message into thread-local storage, valid until the next bridge call. */
    /** Returns the calling thread's last bridge error as a NUL-terminated string.
     * The pointer remains valid until another guarded bridge call on that thread.
     */
    const char* singe_nvcomp_last_error(void);
    /** Creates a GPU manager for format and writes the owned handle to output.
     * compress_options and decompress_options must point to the option structures
     * for that format. chunk_size is in bytes. Initialization on stream completes
     * before return. Returns 0 on success or -1 after recording a C++ exception.
     */
    int singe_nvcomp_manager_create(
        SingeNvcompFormat format,
        size_t chunk_size,
        const void* compress_options,
        const void* decompress_options,
        void* stream,
        SingeNvcompChecksumPolicy checksum,
        SingeNvcompExecutionPolicy execution,
        SingeNvcompBitstreamKind bitstream,
        SingeNvcompManager** output
    );
    /** Creates an LZ4 CPU manager with the requested chunk bytes, compression level,
     * and worker count. Writes the owned handle to output. Returns 0 on success or
     * -1 after recording a C++ exception.
     */
    int singe_nvcomp_cpu_create(
        size_t chunk_size,
        int level,
        unsigned threads,
        SingeNvcompManager** output
    );
    /** Construct a manager from a given compressed buffer. This operation synchronizes the host with the stream. */
    int singe_nvcomp_manager_from_buffer(
        const uint8_t* buffer,
        void* stream,
        int cpu,
        unsigned threads,
        SingeNvcompChecksumPolicy checksum,
        SingeNvcompExecutionPolicy execution,
        int backend,
        int sort,
        int algorithm,
        SingeNvcompManager** output
    );
    /** Returns the compression format for a given HLIF compressed buffer. */
    int singe_nvcomp_format(
        const uint8_t* buffer,
        void* stream,
        int cpu,
        int* format
    );
    /** Releases a manager; NULL is accepted. GPU work is drained before destruction.
     * The process terminates if stream completion cannot be established after retry.
     */
    void singe_nvcomp_manager_destroy(SingeNvcompManager* manager);
    /** Configure the compression for a single buffer. */
    int singe_nvcomp_compress_bound(
        SingeNvcompManager* manager,
        size_t input_bytes,
        size_t* output_bytes
    );
    /** Perform compression for a single buffer and wait for completion. */
    int singe_nvcomp_compress(
        SingeNvcompManager* manager,
        const uint8_t* input,
        size_t input_bytes,
        uint8_t* output,
        size_t capacity,
        size_t* written
    );
    /** Configure the decompression for a single buffer using a compressed buffer. */
    int singe_nvcomp_decompress_size(
        SingeNvcompManager* manager,
        const uint8_t* input,
        size_t input_bytes,
        size_t* output_bytes
    );
    /** Perform decompression for a single buffer and wait for completion. */
    int singe_nvcomp_decompress(
        SingeNvcompManager* manager,
        const uint8_t* input,
        size_t input_bytes,
        uint8_t* output,
        size_t capacity,
        size_t* written
    );
    /** Get the required alignment of input, output, and scratch buffers. */
    int singe_nvcomp_alignments(
        SingeNvcompManager* manager,
        int decompress,
        size_t* input,
        size_t* output,
        size_t* scratch
    );
    /** Deallocate the GPU memory associated with this manager. */
    int singe_nvcomp_deallocate(SingeNvcompManager* manager);

    /** Streaming input callback taking context, writable bytes, capacity, and an
     * output byte count. Return 0 and set the count to at most capacity on success;
     * a count of zero indicates end of input. A nonzero return reports failure.
     */
    typedef int (*SingeNvcompRead)(void*, uint8_t*, size_t, size_t*);
    /** Streaming output callback taking context, readable bytes, and byte count.
     * Return 0 after consuming all bytes or nonzero on failure.
     */
    typedef int (*SingeNvcompWrite)(void*, const uint8_t*, size_t);
    /** Compute the number of device bytes to allocate for the compression workspace. */
    int singe_nvcomp_gzip_temp_size(int compress, int algorithm, size_t* bytes);
    /** Perform synchronous compression or decompression streaming from stream to stream.
 * Does not support multi-member gzip streams. */
    int singe_nvcomp_gzip_stream(
        int operation,
        int algorithm,
        SingeNvcompRead read,
        void* input,
        SingeNvcompWrite write,
        void* output,
        void* workspace,
        size_t workspace_bytes,
        void* stream,
        size_t* decompressed_size
    );
#ifdef __cplusplus
}
#endif
