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
    fprintf(stderr, "launch: %s\nusage: launch CUBIN KERNEL GRID BLOCK SHARED [u32:N | f32:X | ptr:BYTES]...\n"
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
    unsigned grid[3] = {1, 1, 1}, block[3] = {1, 1, 1}, shared = 0;
    if (argc < 6 || sscanf(argv[3], "%u,%u,%u", &grid[0], &grid[1], &grid[2]) < 1 ||
        sscanf(argv[4], "%u,%u,%u", &block[0], &block[1], &block[2]) < 1 ||
        !*argv[5] || strspn(argv[5], "0123456789") != strlen(argv[5]) || strlen(argv[5]) > 9)
        usage("expected a cubin, a kernel name, GRID and BLOCK as x,y,z and the dynamic shared bytes, or --device");
    shared = strtoul(argv[5], NULL, 10);
    check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    check(cuModuleLoad(&module, argv[1]), argv[1]);
    check(cuModuleGetFunction(&kernel, module, argv[2]), argv[2]);
    // A launch asking for more dynamic shared memory than this attribute fails, and it
    // defaults to MAX_SHARED_MEMORY_PER_BLOCK less the static bytes (cuda.h,
    // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES).
    check(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared), "dynamic shared bytes");
    union { unsigned u32; float f32; CUdeviceptr ptr; } *values = calloc(argc, sizeof *values);
    void **args = calloc(argc, sizeof *args);
    for (int i = 0; i < argc - 6; i++) {
        const char *param = argv[6 + i], *text = strlen(param) > 4 ? param + 4 : "";
        char *end = NULL;
        unsigned long long number = strtoull(text, NULL, 10);
        float real = strtof(text, &end);
        int decimal = *text && strspn(text, "0123456789") == strlen(text);
        args[i] = &values[i];
        if (!strncmp(param, "u32:", 4) && decimal && number <= 0xffffffffULL) {
            values[i].u32 = number;
        } else if (!strncmp(param, "f32:", 4) && *text && !*end) {
            values[i].f32 = real;
        } else if (!strcmp(param, "ptr:0")) {
            values[i].ptr = 0;
        } else if (!strncmp(param, "ptr:", 4) && decimal) {
            check(cuMemAlloc(&values[i].ptr, number), param);
            check(cuMemsetD8(values[i].ptr, 0, number), param);
        } else {
            usage(param);
        }
    }
    check(cuLaunchKernel(kernel, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared, NULL, args, NULL),
          "cuLaunchKernel");
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    return 0;
}
