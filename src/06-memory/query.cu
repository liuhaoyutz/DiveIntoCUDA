// 这段代码展示了通过cudaGetDeviceProperties取得设备信息

/*
基于3090平台，执行结果如下：
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/06-memory$ nvcc query.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/06-memory$ ./a.out 
Device id:                                 0
Device name:                               NVIDIA GeForce RTX 3090
Compute capability:                        8.6
Amount of global memory:                   23.569 GB
Amount of constant memory:                 64 KB
Maximum grid size:                         2147483647 65535 65535
Maximum block size:                        1024 1024 64
Number of SMs:                             82
Maximum amount of shared memory per block: 48 KB
Maximum amount of shared memory per SM:    100 KB
Maximum number of registers per block:     64 K
Maximum number of registers per SM:        64 K
Maximum number of threads per block:       1024
Maximum number of threads per SM:          1536
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/06-memory$ 
*/

#include "error.cuh"
#include <stdio.h>

int main(int argc, char *argv[])
{
    int device_id = 0;
    if (argc > 1) device_id = atoi(argv[1]);
    CHECK(cudaSetDevice(device_id));

    // 定义cudaDeviceProp类型的变量 prop。
    cudaDeviceProp prop;
    // 利用CUDA运行时API函数cudaGetDeviceProperties得到编号为device_id = 0的设备的性质，存放在结构体变量prop中。
    CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    // 将变量prop中的某些成员的值打印出来。
    printf("Device id:                                 %d\n",
        device_id);
    printf("Device name:                               %s\n",
        prop.name);
    printf("Compute capability:                        %d.%d\n",
        prop.major, prop.minor);
    printf("Amount of global memory:                   %g GB\n",
        prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("Amount of constant memory:                 %g KB\n",
        prop.totalConstMem  / 1024.0);
    printf("Maximum grid size:                         %d %d %d\n",
        prop.maxGridSize[0], 
        prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block size:                        %d %d %d\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], 
        prop.maxThreadsDim[2]);
    printf("Number of SMs:                             %d\n",
        prop.multiProcessorCount);
    printf("Maximum amount of shared memory per block: %g KB\n",
        prop.sharedMemPerBlock / 1024.0);
    printf("Maximum amount of shared memory per SM:    %g KB\n",
        prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Maximum number of registers per block:     %d K\n",
        prop.regsPerBlock / 1024);
    printf("Maximum number of registers per SM:        %d K\n",
        prop.regsPerMultiprocessor / 1024);
    printf("Maximum number of threads per block:       %d\n",
        prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM:          %d\n",
        prop.maxThreadsPerMultiProcessor);

    return 0;
}

