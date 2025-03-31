#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    // 为host数组分配内存空间
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    // 初始化host数组， h_x的元素为1.23, h_y的元素为2.34
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    // 为device数组分配显存
    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);

    // 将host数组h_x, h_y拷贝到device数组d_x, d_y
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = N / block_size;
    // 调用核函数。
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    // 将核函数计算结果d_z拷贝到host h_z
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    // 检查计算结果是否正确
    check(h_z, N);

    // 稍许host内存
    free(h_x);
    free(h_y);
    free(h_z);
    // 释放device显存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

// 定义核函数add
void __global__ add(const double *x, const double *y, double *z)
{
    // blockDim.x是block size，即每个block有多少个线程。
    // blockIdx.x是block ID，即第几个block。
    // threadIdx.x是线程ID。
    // 所以计算得到的n就是每个线程要处理的元素index。
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    // 执行数组对应元素加法，结果保存在数组z中（即d_z)。
    z[n] = x[n] + y[n];
}

// 检查结果中否正确，z的元素值应该是3.57
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
