#ifndef _SYSU_CUDA_BUNDLE_H_
#define _SYSU_CUDA_BUNDLE_H_

#include "src/gpu_utils/cuda_mem_utils.h"

class SysuCudaBundle
{
private:
    SysuCudaBundle():
        my_rank(-1),
        rank_on_host(-1),
        device_id(-1),
        allocator(NULL)
    {

    }

public:
    int my_rank;
    int rank_on_host;
    int device_on_host;
    int device_id;
    CudaCustomAllocator * allocator;

    static SysuCudaBundle * instance;

    static bool canBeUse()
    {
        return instance != NULL;
    }

    static SysuCudaBundle * newInstance(int my_rank, int rank_on_host)
    {
        if(instance != NULL)
            delete instance;
        instance = new SysuCudaBundle();
        instance->my_rank = my_rank;
        instance->rank_on_host = rank_on_host;
        return instance;
    }

    static SysuCudaBundle * getInstance()
    {
        return instance;
    }

    static void freeInstance()
    {
        if(instance != NULL)
        {
            delete instance;
            instance = NULL;
        }
    }

    void setupMemory();

    void releaseMemory();

    ~SysuCudaBundle()
    {
        releaseMemory();
    }
};

#endif /* _SYSU_CUDA_BUNDLE_H_ */
