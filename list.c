#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120

#include "ocl_boiler.h"

void list_platforms(int details);

void list_devices(int details, cl_platform_id);

void list_device_info(int details, cl_device_id);

// List every opencl platform
void list_platforms(int details) {
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    ocl_check(err, "get number of platforms");

    cl_platform_id platforms[num_platforms];
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    ocl_check(err, "get platform ids");

    printf("Number of platforms: %u\n", num_platforms);

    for (int i = 0; i < num_platforms; i++) {
        char name[1024];
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
        ocl_check(err, "get platform name");
        printf("Platform %d: %s\n", i, name);

        if (details) {
            list_devices(details - 1, platforms[i]);
        }
    }
}

// List every opencl device of a platform
void list_devices(int details, cl_platform_id platform) {
    cl_uint num_devices;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    ocl_check(err, "get number of devices");

    cl_device_id devices[num_devices];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    ocl_check(err, "get device ids");

    printf("\tNumber of devices: %u\n", num_devices);

    for (int i = 0; i < num_devices; i++) {
        char name[1024];
        err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
        ocl_check(err, "get device name");
        printf("\tDevice %d: %s\n", i, name);

        if (details) {
            list_device_info(details - 1, devices[i]);
        }
    }
}

// List every opencl device info of a device
void list_device_info(int details, cl_device_id device) {
    cl_int err;

    cl_device_type device_type;
    err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    ocl_check(err, "get device type");
    printf("\t\tDevice type: %llu\n", device_type);

    cl_uint max_compute_units;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, NULL);
    ocl_check(err, "get max compute units");
    printf("\t\tMax compute units: %u\n", max_compute_units);

    cl_uint max_work_item_dimensions;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions),
                          &max_work_item_dimensions, NULL);
    ocl_check(err, "get max work item dimensions");
    printf("\t\tMax work item dimensions: %u\n", max_work_item_dimensions);

    size_t max_work_item_sizes[max_work_item_dimensions];
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), &max_work_item_sizes,
                          NULL);
    ocl_check(err, "get max work item sizes");
    printf("\t\tMax work item sizes: ");
    for (int i = 0; i < max_work_item_dimensions; i++) {
        printf("%zu\t", max_work_item_sizes[i]);
    }
    printf("\n");

    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size,
                          NULL);
    ocl_check(err, "get max work group size");
    printf("\t\tMax work group size: %zu\n", max_work_group_size);

    cl_uint address_bits;
    err = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(address_bits), &address_bits, NULL);
    ocl_check(err, "get address bits");
    printf("\t\tAddress bits: %u\n", address_bits);

    cl_ulong max_mem_alloc_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
    ocl_check(err, "get max mem alloc size");
    printf("\t\tMax mem alloc size: %llu\n", max_mem_alloc_size);

    cl_ulong global_mem_size;
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    ocl_check(err, "get global mem size");
    printf("\t\tGlobal mem size: %llu\n", global_mem_size);

    cl_ulong local_mem_size;
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
    ocl_check(err, "get local mem size");
    printf("\t\tLocal mem size: %llu\n", local_mem_size);

    cl_bool image_support;
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
    ocl_check(err, "get image support");
    if (image_support) {
        cl_uint max_read_image_args;
        cl_uint max_write_image_args;
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args,
                              NULL);
        ocl_check(err, "get max read image args");
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args),
                              &max_write_image_args,
                              NULL);
        ocl_check(err, "get max write image args");
        printf("\t\tMax read/write image args: %u/%u\n", max_read_image_args, max_write_image_args);

        size_t image2d_max_width;
        size_t image2d_max_height;
        err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(image2d_max_width), &image2d_max_width, NULL);
        ocl_check(err, "get image2d max width");
        err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(image2d_max_height), &image2d_max_height,
                              NULL);
        ocl_check(err, "get image2d max height");
        printf("\t\tImage2d max width, height: %zu, %zu\n", image2d_max_width, image2d_max_height);

        size_t image3d_max_width;
        size_t image3d_max_height;
        size_t image3d_max_depth;
        err = clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(image3d_max_width), &image3d_max_width, NULL);
        ocl_check(err, "get image3d max width");
        err = clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(image3d_max_height), &image3d_max_height,
                              NULL);
        ocl_check(err, "get image3d max height");
        err = clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(image3d_max_depth), &image3d_max_depth, NULL);
        ocl_check(err, "get image3d max depth");
        printf("\t\tImage3d max width, height, depth: %zu, %zu, %zu\n", image3d_max_width, image3d_max_height,
               image3d_max_depth);

        size_t max_samplers;
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_SAMPLERS, sizeof(max_samplers), &max_samplers, NULL);
        ocl_check(err, "get max samplers");
        printf("\t\tMax samplers: %zu\n", max_samplers);
    }
}

int main(int argc, char *argv[]) {
    if (argc > 2) {
        printf("Usage: %s [details]\n", argv[0]);
    }

    int details = 0;
    if (argc == 2) {
        details = atoi(argv[1]);
    }

    list_platforms(details);
    return 0;
}
