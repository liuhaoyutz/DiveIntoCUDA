// CPU版本数组归约实现

/*
这种方法计算会出错，这是因为，sum是浮点数，在累加计算中出现了所谓的“大数吃小数”的现象。
单精度浮点数只有 6、7 位精确的有效数字。在上面的函数 reduce 中，将变量 sum 的值累加到 3000多万后，
再将它和 1.23 相加，其值就不再增加了（小数被大数“吃掉了”，但大数并没有变化）。
*/

#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 20;
void timing(const real *x, const int N);
real reduce(const real *x, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *x = (real *) malloc(M);
    // x是包含N个元素的一维数组，初始化x的元素值为1.23
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.23;
    }

    timing(x, N);

    free(x);
    return 0;
}

void timing(const real *x, const int N)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        // x是包含N个元素的一维数组，reduce函数求数组x所有元素之和
        sum = reduce(x, N);

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

real reduce(const real *x, const int N)
{
    real sum = 0.0;
    for (int n = 0; n < N; ++n)
    {
        sum += x[n];
    }
    return sum;
}


