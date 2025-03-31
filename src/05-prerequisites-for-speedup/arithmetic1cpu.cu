// 这个程序展示了进行更复杂的浮点运算用时情况，这个例子的计算使用CPU
/*
执行结果如下：
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/05-prerequisites-for-speedup$ nvcc arithmetic1cpu.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/05-prerequisites-for-speedup$ ./a.out 
Time = 495.245 ms.
Time = 494.284 ms.
Time = 493.772 ms.
Time = 491.962 ms.
Time = 494.696 ms.
Time = 493.166 ms.
Time = 490.79 ms.
Time = 490.545 ms.
Time = 495.776 ms.
Time = 490.499 ms.
Time = 491.385 ms.
Time = 492.687 +- 1.8071 ms.
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
void arithmetic(real *x, const real x0, const int N);

int main(void)
{
    const int N = 10000;
    const int M = sizeof(real) * N;
    //为x数组分配内存
    real *x = (real*) malloc(M);

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        // 初始化x数组为0.0
        for (int n = 0; n < N; ++n)
        {
            x[n] = 0.0;
        }

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

        // 执行复杂运算，使用CPU
        arithmetic(x, x0, N);

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

    free(x);
    return 0;
}

void arithmetic(real *x, const real x0, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        real x_tmp = x[n];
        while (sqrt(x_tmp) < x0)
        {
            ++x_tmp;
        }
        x[n] = x_tmp;
    }
}


