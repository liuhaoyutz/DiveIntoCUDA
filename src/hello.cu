#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;  // 线程块ID
    const int tid = threadIdx.x;  // 线程ID
    printf("Hello World from block %d and thread %d!\n", bid, tid);
}

int main(void)
{
    // 执行配置（grid_size = 2，block_size = 4），即2个线程块，每个线程块包含4个线程，共8个线程
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}

