/*
在第9章几个版本的数组归约函数中，核函数并没有做全部的计算，而只是将一个长一些的数组 d_x变成了一个短一些的数组d_y，
后者中的每个元素为前者中若干元素的和。在调用核函数之后，将短一些的数组复制到主机，然后在主机中完成了余下的求和。

有两种方法能够在 GPU 中得到最终结果，一是用另一个核函数将较短的数组进一步归约，得到最终的结果（一个数值）；
二是在先前的核函数的末尾利用原子函数进行归约，直接得到最终结果。本章仅讨论第二种方法，在第 11 章将讨论第一种方法。
*/

#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

void timing(const real *d_x);

int main(void)
{
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\nusing atomicAdd:\n");
    timing(d_x);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    // 使用原子操作归约
    if (tid == 0)
    {
        atomicAdd(d_y, s_y[0]);
    }
}

real reduce(const real *d_x)
{
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    reduce<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

void timing(const real *d_x)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x); 

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}


