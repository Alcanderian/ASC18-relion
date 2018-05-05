#include "src/gpu_utils/sysu_cuda_bundle.h"
#include <cuda_runtime.h>

void SysuCudaBundle::setupMemory()
{
    //get device on host
    HANDLE_ERROR(cudaGetDeviceCount(&device_on_host));

    int mem_div = rank_on_host / device_on_host;
    if(rank_on_host % device_on_host > 0)
    {
        if(my_rank  % device_on_host < rank_on_host % device_on_host)
        {
            mem_div ++;
        }
    }
    device_id = my_rank % device_on_host;
	HANDLE_ERROR(cudaSetDevice(device_id));

    int memAlignmentSize;
	cudaDeviceGetAttribute ( &memAlignmentSize, cudaDevAttrTextureAlignment, device_id );
	allocator = new CudaCustomAllocator(0, memAlignmentSize);

    size_t free_mem, total, allocationSize;
	HANDLE_ERROR(cudaMemGetInfo( &free_mem, &total ));

    allocationSize = (total / mem_div) * 0.95;

    allocator->resize(allocationSize);

    printf("[Sysu Note]: Rank %d get %llu MB Mem from gpu %d\n", my_rank, allocationSize >> 20, device_id);
}

void SysuCudaBundle::releaseMemory()
{
    if(allocator != NULL)
    {
        printf("[Sysu Note]: Rank %d free Mem from gpu %d\n", my_rank, device_id);
        delete allocator;
        allocator = NULL;

        HANDLE_ERROR(cudaSetDevice(device_id));
		HANDLE_ERROR(cudaDeviceReset());
    }
}

SysuCudaBundle * SysuCudaBundle::instance = NULL;