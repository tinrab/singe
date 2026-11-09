#include "bridge.h"
#include <nvcomp/nvcompManagerFactory.hpp>
#include <nvcomp/nvcompCPUManagerFactory.hpp>
#include <nvcomp/lz4_cpu.hpp>
#include <nvcomp/native/streaming_gzip.hpp>
#include <array>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <streambuf>

struct SingeNvcompManager {
    std::shared_ptr<nvcomp::nvcompManagerBase> value;
    cudaStream_t stream;
    bool cpu;
    bool native;
};

namespace {
thread_local char error_text[4096] = {};

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

void status_check(nvcompStatus_t status) {
    if (status != nvcompSuccess)
        throw std::runtime_error(nvcompGetStatusString(status));
}

template<class F>
int guard(F&& call) noexcept {
    error_text[0] = 0;
    try {
        call();
        return 0;
    } catch (const std::exception& error) {
        std::strncpy(error_text, error.what(), sizeof(error_text) - 1);
    } catch (...) {
        std::strncpy(
            error_text,
            "unknown nvcomp c++ exception",
            sizeof(error_text) - 1
        );
    }
    error_text[sizeof(error_text) - 1] = 0;
    return -1;
}

// Completion is a prerequisite to releasing borrowed host state or native
// scratch. Retry once after an error, then preserve resources by terminating.
void drain_before_release(cudaStream_t stream) noexcept {
    if (cudaStreamSynchronize(stream) != cudaSuccess &&
        cudaStreamSynchronize(stream) != cudaSuccess)
        std::abort();
}

// These values stay allocated until the stream is synchronized, including failure paths.
struct SizeValue {
    size_t host;
    size_t* device = nullptr;
    cudaStream_t stream;

    explicit SizeValue(SingeNvcompManager& manager, size_t value = 0) :
        host(value),
        stream(manager.stream) {
        if (!manager.cpu) {
            cuda_check(
                cudaMalloc(reinterpret_cast<void**>(&device), sizeof(size_t))
            );
            auto status = cudaMemcpyAsync(
                device,
                &host,
                sizeof(size_t),
                cudaMemcpyHostToDevice,
                stream
            );
            if (status != cudaSuccess) {
                drain_before_release(stream);
                cudaFree(device);
                device = nullptr;
                cuda_check(status);
            }
        }
    }

    ~SizeValue() {
        if (device) {
            drain_before_release(stream);
            cudaFree(device);
        }
    }

    size_t* get() {
        return device ? device : &host;
    }

    size_t read() {
        if (device)
            cuda_check(cudaMemcpyAsync(
                &host,
                device,
                sizeof(size_t),
                cudaMemcpyDeviceToHost,
                stream
            ));
        if (device)
            cuda_check(cudaStreamSynchronize(stream));
        return host;
    }
};

struct StreamCompletion {
    SingeNvcompManager& manager;

    ~StreamCompletion() {
        if (!manager.cpu)
            drain_before_release(manager.stream);
    }

    void wait() {
        if (!manager.cpu)
            cuda_check(cudaStreamSynchronize(manager.stream));
    }
};

class InputBuffer: public std::streambuf {
    SingeNvcompRead read;
    void* context;
    std::array<char, 65536> buffer {};

  public:
    InputBuffer(SingeNvcompRead read, void* context) :
        read(read),
        context(context) {}

    int_type underflow() override {
        size_t count = 0;
        if (read(
                context,
                reinterpret_cast<uint8_t*>(buffer.data()),
                buffer.size(),
                &count
            ) != 0
            || count > buffer.size())
            throw std::runtime_error("gzip input callback failed");
        if (!count)
            return traits_type::eof();
        setg(buffer.data(), buffer.data(), buffer.data() + count);
        return traits_type::to_int_type(*gptr());
    }
};

class OutputBuffer: public std::streambuf {
    SingeNvcompWrite write;
    void* context;

  public:
    OutputBuffer(SingeNvcompWrite write, void* context) :
        write(write),
        context(context) {}

    std::streamsize xsputn(const char* data, std::streamsize count) override {
        if (count < 0
            || write(
                   context,
                   reinterpret_cast<const uint8_t*>(data),
                   static_cast<size_t>(count)
               ) != 0)
            throw std::runtime_error("gzip output callback failed");
        return count;
    }

    int_type overflow(int_type value) override {
        if (traits_type::eq_int_type(value, traits_type::eof()))
            return traits_type::not_eof(value);
        char byte = traits_type::to_char_type(value);
        xsputn(&byte, 1);
        return value;
    }
};
} // namespace

extern "C" {
    const char* singe_nvcomp_last_error() {
        return error_text;
    }

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
    ) {
        return guard([&] {
            auto handle = std::make_unique<SingeNvcompManager>();
            handle->stream = static_cast<cudaStream_t>(stream);
            handle->cpu = false;
            handle->native = bitstream == SingeNvcompNative;
            auto c = static_cast<nvcomp::ChecksumPolicy>(checksum);
            auto e = static_cast<nvcomp::ExecutionPolicy>(execution);
            auto b = static_cast<nvcomp::BitstreamKind>(bitstream);
            StreamCompletion completion {*handle};
            switch (format) {
#define CODEC(id, Type, Opts)                                           \
    case id:                                                            \
        handle->value = std::make_shared<nvcomp::Type>(                 \
            chunk_size,                                                 \
            *static_cast<const nvcompBatched##Opts##CompressOpts_t*>(   \
                compress_options                                        \
            ),                                                          \
            *static_cast<const nvcompBatched##Opts##DecompressOpts_t*>( \
                decompress_options                                      \
            ),                                                          \
            handle->stream,                                             \
            c,                                                          \
            e,                                                          \
            b                                                           \
        );                                                              \
        break;
                CODEC(SingeNvcompLz4, LZ4Manager, LZ4)
                CODEC(SingeNvcompSnappy, SnappyManager, Snappy)
                CODEC(SingeNvcompAns, ANSManager, ANS)
                CODEC(SingeNvcompGdeflate, GdeflateManager, Gdeflate)
                CODEC(SingeNvcompCascaded, CascadedManager, Cascaded)
                CODEC(SingeNvcompBitcomp, BitcompManager, Bitcomp)
                CODEC(SingeNvcompZstd, ZstdManager, Zstd)
                CODEC(SingeNvcompDeflate, DeflateManager, Deflate)
                CODEC(SingeNvcompGzip, GzipManager, Gzip)
#undef CODEC
                default:
                    throw std::runtime_error(
                        "unsupported nvcomp manager format"
                    );
            }
            completion.wait();
            *output = handle.release();
        });
    }

    int singe_nvcomp_cpu_create(
        size_t chunk_size,
        int level,
        unsigned threads,
        SingeNvcompManager** output
    ) {
        return guard([&] {
            auto manager = std::make_unique<SingeNvcompManager>();
            manager->cpu = true;
            manager->native = true;
            manager->stream = nullptr;
            manager->value = std::make_shared<nvcomp::LZ4CPUManager>(
                chunk_size,
                level,
                threads
            );
            *output = manager.release();
        });
    }

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
    ) {
        return guard([&] {
            auto manager = std::make_unique<SingeNvcompManager>();
            manager->cpu = cpu != 0;
            manager->native = true;
            manager->stream = static_cast<cudaStream_t>(stream);
            StreamCompletion completion {*manager};
            manager->value = cpu
                ? nvcomp::create_cpu_manager(buffer, threads)
                : nvcomp::create_manager(
                      buffer,
                      manager->stream,
                      static_cast<nvcomp::ChecksumPolicy>(checksum),
                      static_cast<nvcomp::ExecutionPolicy>(execution),
                      static_cast<nvcompDecompressBackend_t>(backend),
                      sort != 0,
                      algorithm
                  );
            completion.wait();
            *output = manager.release();
        });
    }

    int singe_nvcomp_format(
        const uint8_t* buffer,
        void* stream,
        int cpu,
        int* format
    ) {
        return guard([&] {
            *format = static_cast<int>(
                cpu ? nvcomp::get_cpu_compression_format(buffer)
                    : nvcomp::get_compression_format(
                          buffer,
                          static_cast<cudaStream_t>(stream)
                      )
            );
        });
    }

    void singe_nvcomp_manager_destroy(SingeNvcompManager* manager) {
        guard([&] {
            if (manager && !manager->cpu)
                drain_before_release(manager->stream);
            delete manager;
        });
    }

    int singe_nvcomp_compress_bound(
        SingeNvcompManager* manager,
        size_t input_bytes,
        size_t* output_bytes
    ) {
        return guard([&] {
            *output_bytes = manager->value->configure_compression(input_bytes)
                                .max_compressed_buffer_size;
        });
    }

    int singe_nvcomp_compress(
        SingeNvcompManager* manager,
        const uint8_t* input,
        size_t input_bytes,
        uint8_t* output,
        size_t capacity,
        size_t* written
    ) {
        return guard([&] {
            auto config = manager->value->configure_compression(input_bytes);
            if (capacity < config.max_compressed_buffer_size)
                throw std::runtime_error(
                    "compression output buffer is too small"
                );
            SizeValue size(*manager);
            StreamCompletion completion {*manager};
            manager->value->compress(input, output, config, size.get());
            completion.wait();
            status_check(*config.get_status());
            *written = manager->native
                ? manager->value->get_compressed_output_size(output)
                : size.read();
            if (*written > capacity)
                throw std::runtime_error(
                    "invalid compressed size returned by nvcomp"
                );
        });
    }

    int singe_nvcomp_decompress_size(
        SingeNvcompManager* manager,
        const uint8_t* input,
        size_t input_bytes,
        size_t* output_bytes
    ) {
        return guard([&] {
            SizeValue size(*manager, input_bytes);
            *output_bytes =
                manager->value->configure_decompression(input, size.get())
                    .decomp_data_size;
        });
    }

    int singe_nvcomp_decompress(
        SingeNvcompManager* manager,
        const uint8_t* input,
        size_t input_bytes,
        uint8_t* output,
        size_t capacity,
        size_t* written
    ) {
        return guard([&] {
            SizeValue size(*manager, input_bytes);
            auto config =
                manager->value->configure_decompression(input, size.get());
            if (capacity < config.decomp_data_size)
                throw std::runtime_error(
                    "decompression output buffer is too small"
                );
            StreamCompletion completion {*manager};
            manager->value->decompress(output, input, config, size.get());
            completion.wait();
            status_check(*config.get_status());
            *written = config.decomp_data_size;
        });
    }

    int singe_nvcomp_alignments(
        SingeNvcompManager* manager,
        int decompress,
        size_t* input,
        size_t* output,
        size_t* scratch
    ) {
        return guard([&] {
            auto value = decompress
                ? manager->value->get_required_decompression_alignments()
                : manager->value->get_required_compression_alignments();
            *input = value.input;
            *output = value.output;
            *scratch = value.temp;
        });
    }

    int singe_nvcomp_deallocate(SingeNvcompManager* manager) {
        return guard([&] {
            StreamCompletion completion {*manager};
            completion.wait();
            manager->value->deallocate_gpu_mem();
        });
    }

    int singe_nvcomp_gzip_temp_size(
        int compress,
        int algorithm,
        size_t* bytes
    ) {
        return guard([&] {
            auto options = nvcompBatchedGzipCompressDefaultOpts;
            options.algorithm = algorithm;
            status_check(
                compress
                    ? nvcompGzipStreamingCompressGetTempSize(options, bytes)
                    : nvcompGzipStreamingDecompressGetTempSize(bytes)
            );
        });
    }

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
    ) {
        return guard([&] {
            InputBuffer in_buffer(read, input);
            std::istream in(&in_buffer);
            in.exceptions(std::ios::badbit);
            auto cuda_stream = static_cast<cudaStream_t>(stream);
            SingeNvcompManager context {nullptr, cuda_stream, false, false};
            if (operation == 2) {
                StreamCompletion completion {context};
                status_check(nvcompGzipStreamingGetDecompressSize(
                    in,
                    workspace_bytes,
                    workspace,
                    decompressed_size,
                    cuda_stream
                ));
                completion.wait();
                return;
            }
            OutputBuffer out_buffer(write, output);
            std::ostream out(&out_buffer);
            out.exceptions(std::ios::badbit | std::ios::failbit);
            auto options = nvcompBatchedGzipCompressDefaultOpts;
            options.algorithm = algorithm;
            StreamCompletion completion {context};
            status_check(
                operation == 1 ? nvcompGzipStreamingCompress(
                                     in,
                                     out,
                                     workspace_bytes,
                                     workspace,
                                     options,
                                     cuda_stream
                                 )
                               : nvcompGzipStreamingDecompress(
                                     in,
                                     out,
                                     workspace_bytes,
                                     workspace,
                                     cuda_stream
                                 )
            );
            completion.wait();
        });
    }
}
