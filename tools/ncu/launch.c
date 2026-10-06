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
    fprintf(stderr, "launch: %s\nusage: launch CUBIN KERNEL GRID BLOCK [u32:N | f32:X | ptr:BYTES]...\n"
                    "       launch --device\n", what);
    exit(2);
}

int main(int argc, char **argv) {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    check(cuInit(0), "cuInit");
    check(cuDeviceGet(&device, 0), "cuDeviceGet");
    if (argc == 2 && !strcmp(argv[1], "--device")) {
        char name[256];
        int major, minor;
        check(cuDeviceGetName(name, sizeof name, device), "cuDeviceGetName");
        check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "major");
        check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "minor");
        printf("%s\nsm_%d%d\n", name, major, minor);
        return 0;
    }
    unsigned grid[3] = {1, 1, 1}, block[3] = {1, 1, 1};
    if (argc < 5 || sscanf(argv[3], "%u,%u,%u", &grid[0], &grid[1], &grid[2]) < 1 ||
        sscanf(argv[4], "%u,%u,%u", &block[0], &block[1], &block[2]) < 1)
        usage("expected a cubin, a kernel name, and GRID and BLOCK as x,y,z, or --device");
    check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    check(cuModuleLoad(&module, argv[1]), argv[1]);
    check(cuModuleGetFunction(&kernel, module, argv[2]), argv[2]);
    union { unsigned u32; float f32; CUdeviceptr ptr; } *values = calloc(argc, sizeof *values);
    void **args = calloc(argc, sizeof *args);
    for (int i = 0; i < argc - 5; i++) {
        const char *param = argv[5 + i], *text = strlen(param) > 4 ? param + 4 : "";
        char *end = NULL;
        unsigned long long number = strtoull(text, NULL, 10);
        float real = strtof(text, &end);
        int decimal = *text && strspn(text, "0123456789") == strlen(text);
        args[i] = &values[i];
        if (!strncmp(param, "u32:", 4) && decimal && number <= 0xffffffffULL) {
            values[i].u32 = number;
        } else if (!strncmp(param, "f32:", 4) && *text && !*end) {
            values[i].f32 = real;
        } else if (!strncmp(param, "ptr:", 4) && decimal) {
            check(cuMemAlloc(&values[i].ptr, number), param);
            check(cuMemsetD8(values[i].ptr, 0, number), param);
        } else {
            usage(param);
        }
    }
    check(cuLaunchKernel(kernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, NULL, args, NULL),
          "cuLaunchKernel");
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    return 0;
}
