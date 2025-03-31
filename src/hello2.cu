#include <stdio.h>

__global__ void hello_from_gpu()
{
    // blockIdx, threadIdx是uint3类型变量
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    printf("Hello World from block-(%d, %d, %d) and thread-(%d, %d, %d)!\n", bx, by, bz, tx, ty, tz);
}

int main(void)
{
    // grid_size和block_size是dim3类型变量
    const dim3 grid_size(2, 2, 2);
    const dim3 block_size(2, 2, 3);
    hello_from_gpu<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}

