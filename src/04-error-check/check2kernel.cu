// 这个程序演示了怎样检查核函数执行有没有出错

#include "error.cuh"  // 定义CUDA运行时API检查宏CHECK
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;

    // 为host数组分配内存
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    // 初始化host数组
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    // 为device数组分配显存
    double *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **)&d_x, M));
    CHECK(cudaMalloc((void **)&d_y, M));
    CHECK(cudaMalloc((void **)&d_z, M));

    // 将host数组h_x, h_y的内容拷贝到device数组d_x, d_y
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    const int block_size = 1280;
    const int grid_size = (N + block_size - 1) / block_size;
    // 调用核函数add
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

    // 捕捉cudaDeviceSynchronize语句之前的最后一个错误
    CHECK(cudaGetLastError());
    // 同步host与device
    CHECK(cudaDeviceSynchronize());

    // 将核函数计算结果device d_z拷贝到host h_z
    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    // 检查计算结果是否正确
    check(h_z, N);

    // 释放分配的hos内存
    free(h_x);
    free(h_y);
    free(h_z);
    // 释放分配的device显存
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

// 定义核函数
void __global__ add(const double *x, const double *y, double *z, const int N)
{
    // 计算每个线程对应的一维向量index
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    // 实际上我们创建的线程个数是N + block_size - 1，只允许前N个线程执行计算，后面的线程不做任何计算。
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}

// 检查计算结果是否正确
void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

