// 这段代码演示了怎样使用CUDA提供的基于CUDA事件（CUDA event）的计时，计算函数add使用GPU
/*
执行结果如下：
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/05-prerequisites-for-speedup$ nvcc add2gpu.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/05-prerequisites-for-speedup$ ./a.out 
Time = 1.91437 ms.
Time = 1.41187 ms.
Time = 1.45376 ms.
Time = 1.42746 ms.
Time = 1.41197 ms.
Time = 1.42621 ms.
Time = 1.42234 ms.
Time = 1.42234 ms.
Time = 1.42336 ms.
Time = 1.41005 ms.
Time = 1.42512 ms.
Time = 1.42345 +- 0.0117998 ms.
No errors
*/

#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP  // 使用双精度浮点数
    typedef double real;
    const real EPSILON = 1.0e-15;
#else  // 使用单精度浮点数
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

const int NUM_REPEATS = 10;
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;
void __global__ add(const real *x, const real *y, real *z, const int N);
void check(const real *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    // 为host数组分配内存
    real *h_x = (real*) malloc(M);
    real *h_y = (real*) malloc(M);
    real *h_z = (real*) malloc(M);

    // 初始化host数组
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    // 为device数组分配显存
    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **)&d_x, M));
    CHECK(cudaMalloc((void **)&d_y, M));
    CHECK(cudaMalloc((void **)&d_z, M));

    // 将host数组内容拷贝到device数组
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        // 创建CUDA event变量start和stop。
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        // 将start传递给cudaEventRecord，在需要计时的代码块之前，记录一个代表开始的事件。
        CHECK(cudaEventRecord(start));
        /*
        对处于 TCC 驱动模式的 GPU 来说可以省略，但对处于 WDDM 驱动模式的 GPU 来说必须保留。
        这是因为，在处于 WDDM 驱动模式的 GPU 中，一个 CUDA 流（CUDA stream）中的操作
        （如这里的 cudaEventRecord 函数）并不是直接提交给 GPU 执行，而是先提交到一个软件队列，
        需要添加一条对该流的 cudaEventQuery 操作（或者 cudaEventSynchronize）刷新队列，才能促使前面的操作在 GPU 执行。
        */
        cudaEventQuery(start);

        // 调用核函数
        add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

        // 将stop传递给cudaEventRecord。在需要计时的代码块之后记录一个代表结束的事件。
        CHECK(cudaEventRecord(stop));
        // 调用cudaEventSynchronize 函数让主机等待事件 stop 被记录完毕。
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        //调用 cudaEventElapsedTime 函数计算 start 和 stop 之间的时间差(单位ms)。
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        // 调用 cudaEventDestroy 函数销毁 start 和 stop 这两个 CUDA 事件。
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

void __global__ add(const real *x, const real *y, real *z, const int N)
{
    // 计算每个线程对应的index
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    // 实际上我们创建的线程个数是N + block_size - 1，只允许前N个线程执行计算，后面的线程不做任何计算。
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const real *z, const int N)
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

