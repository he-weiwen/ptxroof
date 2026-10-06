#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(CUresult result, const char *what) {
    const char *message = "unknown error";
    if (result == CUDA_SUCCESS)
        return;
    cuGetErrorString(result, &message);
    fprintf(stderr, "launch: %s: %s\n", what, message);
    exit(2);
}

static void usage(const char *what) {
    fprintf(stderr, "launch: %s\nusage: launch CUBIN KERNEL GRID BLOCK [u32:N | f32:X | ptr:BYTES]...\n", what);
    exit(2);
}

int main(int argc, char **argv) {
    unsigned grid[3] = {1, 1, 1}, block[3] = {1, 1, 1};
    if (argc < 5 || sscanf(argv[3], "%u,%u,%u", &grid[0], &grid[1], &grid[2]) < 1 ||
        sscanf(argv[4], "%u,%u,%u", &block[0], &block[1], &block[2]) < 1)
        usage("expected a cubin, a kernel name, and GRID and BLOCK as x,y,z");
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    check(cuInit(0), "cuInit");
    check(cuDeviceGet(&device, 0), "cuDeviceGet");
    check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    check(cuModuleLoad(&module, argv[1]), argv[1]);
    check(cuModuleGetFunction(&kernel, module, argv[2]), argv[2]);
    union { unsigned u32; float f32; CUdeviceptr ptr; } *values = calloc(argc, sizeof *values);
    void **args = calloc(argc, sizeof *args);
    for (int i = 0; i < argc - 5; i++) {
        const char *param = argv[5 + i];
        size_t bytes = strtoull(param + 4, NULL, 0);
        args[i] = &values[i];
        if (!strncmp(param, "u32:", 4)) {
            values[i].u32 = strtoul(param + 4, NULL, 0);
        } else if (!strncmp(param, "f32:", 4)) {
            values[i].f32 = strtof(param + 4, NULL);
        } else if (!strncmp(param, "ptr:", 4)) {
            check(cuMemAlloc(&values[i].ptr, bytes), param);
            check(cuMemsetD8(values[i].ptr, 0, bytes), param);
        } else {
            usage(param);
        }
    }
    check(cuLaunchKernel(kernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, NULL, args, NULL),
          "cuLaunchKernel");
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    return 0;
}
