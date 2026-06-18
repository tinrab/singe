#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void check(CUresult result, const char *expr) {
    if (result != CUDA_SUCCESS) {
        const char *name = "unknown";
        const char *message = "unknown";
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &message);
        fprintf(stderr, "%s failed: %s: %s\n", expr, name, message);
        exit(1);
    }
}

static int exists(const char *path) {
    return access(path, F_OK) == 0;
}

static void write_file(const char *path, const char *contents) {
    FILE *file = fopen(path, "w");
    if (!file) {
        perror(path);
        exit(1);
    }
    fputs(contents, file);
    fclose(file);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <test-dir>\n", argv[0]);
        return 2;
    }

    const char *test_dir = argv[1];
    CUdevice device;
    CUcontext context;
    CUdeviceptr counter;
    int value = 100;

    check(cuInit(0), "cuInit");
    check(cuDeviceGet(&device, 0), "cuDeviceGet");
    check(cuCtxCreate(&context, NULL, 0, device), "cuCtxCreate");
    check(cuMemAlloc(&counter, sizeof(value)), "cuMemAlloc");
    check(cuMemcpyHtoD(counter, &value, sizeof(value)), "cuMemcpyHtoD");

    char path[4096];
    snprintf(path, sizeof(path), "%s/ready", test_dir);
    write_file(path, "");

    int sequence = 1;
    while (1) {
        snprintf(path, sizeof(path), "%s/release", test_dir);
        if (exists(path)) {
            break;
        }

        snprintf(path, sizeof(path), "%s/request-%d", test_dir, sequence);
        if (exists(path)) {
            check(cuMemcpyDtoH(&value, counter, sizeof(value)), "cuMemcpyDtoH");
            value++;
            check(cuMemcpyHtoD(counter, &value, sizeof(value)), "cuMemcpyHtoD");

            char response[4096];
            char contents[32];
            snprintf(response, sizeof(response), "%s/response-%d", test_dir, sequence);
            snprintf(contents, sizeof(contents), "%d", value);
            write_file(response, contents);
            sequence++;
        } else {
            usleep(25000);
        }
    }

    cuMemFree(counter);
    cuCtxDestroy(context);
    return 0;
}
