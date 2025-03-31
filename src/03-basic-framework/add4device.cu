#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add1(const double *x, const double *y, double *z, const int N);
void __global__ add2(const double *x, const double *y, double *z, const int N);
void __global__ add3(const double *x, const double *y, double *z, const int N);
void check(const double *z, int N);

int main(void)
{
    const int N = 100000001;  // 这里故意多加1，让N/128除不尽
    const int M = sizeof(double) * N;

    // 分配host内存
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    // 初始化host数组，h_x元素值为1.23, h_y元素值为2.34
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    // 分配device显存
    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);

    // 将host数组h_x, h_y的内容拷贝到device数组d_x, d_y
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;

    // 调用核函数add1
    add1<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    // 将核函数计算结果从device d_z拷贝到host h_z
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    // 检查计算结果是否正确
    check(h_z, N);

    // 调用核函数add2
    add2<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    // 将核函数计算结果从device d_z拷贝到host h_z
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    // 检查计算结果是否正确
    check(h_z, N);

    // 调用核函数add3
    add3<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    // 将核函数计算结果从device d_z拷贝到host h_z
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    // 检查计算结果是否正确
    check(h_z, N);

    // 释放分配的内存
    free(h_x);
    free(h_y);
    free(h_z);
    // 释放分配的显存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

// 定义设备函数add1_device，这个函数有返回值。
double __device__ add1_device(const double x, const double y)
{
    return (x + y);
}

// 定义设备函数add2_device
void __device__ add2_device(const double x, const double y, double *z)
{
    *z = x + y;
}

// 定义设备函数add3_device
void __device__ add3_device(const double x, const double y, double &z)
{
    z = x + y;
}

// 定义核函数add1
void __global__ add1(const double *x, const double *y, double *z, const int N)
{
    // 计算每个线程对应的一维向量index
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    // 实际上我们创建的线程个数是N + block_size - 1，只允许前N个线程执行计算，后面的线程不做任何计算。
    if (n < N)
    {
        z[n] = add1_device(x[n], y[n]);  // 设备函数add1_deivce将结果返回
    }
}

// 定义核函数add2
void __global__ add2(const double *x, const double *y, double *z, const int N)
{
    // 计算每个线程对应的一维向量index
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    // 实际上我们创建的线程个数是N + block_size - 1，只允许前N个线程执行计算，后面的线程不做任何计算。
    if (n < N)
    {
        add2_device(x[n], y[n], &z[n]);  // 通过指针传递z数组
    }
}

// 定义核函数add3
void __global__ add3(const double *x, const double *y, double *z, const int N)
{
    // 计算每个线程对应的一维向量index
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    // 实际上我们创建的线程个数是N + block_size - 1，只允许前N个线程执行计算，后面的线程不做任何计算。
    if (n < N)
    {
        add3_device(x[n], y[n], z[n]);  // 通过引用传递z数组
    }
}

// 检查计算结果
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

