// 这个程序展示了进行更复杂的浮点运算用时情况，这个例子的计算使用GPU

/*
执行结果如下：
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/05-prerequisites-for-speedup$ nvcc arithmetic2gpu.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/05-prerequisites-for-speedup$ ./a.out 10000
Time = 0.638976 ms.
Time = 0.452608 ms.
Time = 0.452512 ms.
Time = 0.449536 ms.
Time = 0.448512 ms.
Time = 0.449536 ms.
Time = 0.448512 ms.
Time = 0.448544 ms.
Time = 0.448512 ms.
Time = 0.449536 ms.
Time = 0.448512 ms.
Time = 0.449632 +- 0.00152466 ms.
*/

#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const real x0 = 100.0;
void __global__ arithmetic(real *x, const real x0, const int N);

int main(int argc, char **argv)
{
    if (argc != 2) 
    {
        printf("usage: %s N\n", argv[0]);
        exit(1);
    }
    const int N = atoi(argv[1]);
    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;

    const int M = sizeof(real) * N;
    // 为host数组h_x分配内存
    real *h_x = (real*) malloc(M);
    // 为device数组d_x分配显存
    real *d_x;
    CHECK(cudaMalloc((void **)&d_x, M));

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        // 初始化host数组
        for (int n = 0; n < N; ++n)
        {
            h_x[n] = 0.0;
        }
        // 将host数组内容拷贝到device数组
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

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

        // 调用核函数执行复杂计算
        arithmetic<<<grid_size, block_size>>>(d_x, x0, N);

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

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ arithmetic(real *d_x, const real x0, const int N)
{
    // 计算每个线程对应的index
    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    // 实际上我们创建的线程个数是N + block_size - 1，只允许前N个线程执行计算，后面的线程不做任何计算。
    if (n < N)
    {
        real x_tmp = d_x[n];
        while (sqrt(x_tmp) < x0)
        {
            ++x_tmp;
        }
        d_x[n] = x_tmp;
    }
}


