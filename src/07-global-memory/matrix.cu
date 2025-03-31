// 这段程序展示了怎样通过global memory进行矩阵复制和矩阵转置

#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP  // 使用双精度浮点数
    typedef double real;
#else  // 使用单精度浮点数
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;  // 定义线程块的大小（32x32）

void timing(const real *d_A, real *d_B, const int N, const int task);
__global__ void copy(const real *A, real *B, const int N);
__global__ void transpose1(const real *A, real *B, const int N);
__global__ void transpose2(const real *A, real *B, const int N);
__global__ void transpose3(const real *A, real *B, const int N);
void print_matrix(const int N, const real *A);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: %s N\n", argv[0]);  // N是矩阵的维度
        exit(1);
    }
    const int N = atoi(argv[1]);

    const int N2 = N * N;
    const int M = sizeof(real) * N2;

    // 为2个大小为NxN的矩阵h_A和h_B分配内存
    real *h_A = (real *) malloc(M);
    real *h_B = (real *) malloc(M);

    // 初始化h_A矩阵
    for (int n = 0; n < N2; ++n)
    {
        h_A[n] = n;
    }

    // 为2个大小为NxN的矩阵d_A和d_B分配显存
    real *d_A, *d_B;
    CHECK(cudaMalloc(&d_A, M));
    CHECK(cudaMalloc(&d_B, M));

    // 将host矩阵h_A的内容拷贝到device矩阵d_A中
    CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));

    // 调用timing函数，分别测试4种操作的性能：矩阵复制（copy)、转置（三种实现方式：transpose1, transpose2, transpose3）
    printf("\ncopy:\n");
    timing(d_A, d_B, N, 0);
    printf("\ntranspose with coalesced read:\n");
    timing(d_A, d_B, N, 1);
    printf("\ntranspose with coalesced write:\n");
    timing(d_A, d_B, N, 2);
    printf("\ntranspose with coalesced write and __ldg read:\n");
    timing(d_A, d_B, N, 3);

    // 将device矩阵d_B拷贝到host矩阵h_B
    CHECK(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));

    // 如果矩阵维度小于等于10，则打印矩阵h_A和h_B
    if (N <= 10)
    {
        printf("A =\n");
        print_matrix(N, h_A);
        printf("\nB =\n");
        print_matrix(N, h_B);
    }

    // 释放host内存
    free(h_A);
    free(h_B);
    // 释放device显存
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    return 0;
}

// 测试矩阵运算性能
void timing(const real *d_A, real *d_B, const int N, const int task)
{
    // 计算grid_size和block_size
    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_size_y = grid_size_x;
    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size(grid_size_x, grid_size_y);

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);  // 开始计时

        // 调用核函数
        switch (task)
        {
            case 0:
                // 调用核函数copy
                copy<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 1:
                // 调用核函数transpose1
                transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 2:
                // 调用核函数transpose2
                transpose2<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 3:
                // 调用核函数transpose3
                transpose3<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            default:
                printf("Error: wrong task\n");
                exit(1);
                break;
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));  // 结束计时
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));  // 统计矩阵运算耗时
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}

// 实现矩阵复制的核函数
__global__ void copy(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    const int index = ny * N + nx;
    if (nx < N && ny < N)
    {
        B[index] = A[index];
    }
}

// 矩阵转置核函数实现方法1
__global__ void transpose1(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx];
    }
}

// 矩阵转置核函数实现方法2
__global__ void transpose2(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny];
    }
}

// 矩阵转置核函数实现方法3
__global__ void transpose3(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = __ldg(&A[nx * N + ny]);
    }
}

// 打印矩阵
void print_matrix(const int N, const real *A)
{
    for (int ny = 0; ny < N; ny++)
    {
        for (int nx = 0; nx < N; nx++)
        {
            printf("%g\t", A[ny * N + nx]);
        }
        printf("\n");
    }
}

/*
执行结果如下：

(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/07-global-memory$ nvcc matrix.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/07-global-memory$ ./a.out 8

copy:
Time = 18.692 ms.
Time = 0.007168 ms.
Time = 0.00512 ms.
Time = 0.00512 ms.
Time = 0.004096 ms.
Time = 0.004928 ms.
Time = 0.00496 ms.
Time = 0.004032 ms.
Time = 0.004096 ms.
Time = 0.004096 ms.
Time = 0.004064 ms.
Time = 0.004768 +- 0.000919908 ms.

transpose with coalesced read:
Time = 0.017408 ms.
Time = 0.006144 ms.
Time = 0.004096 ms.
Time = 0.003296 ms.
Time = 0.004096 ms.
Time = 0.005056 ms.
Time = 0.004096 ms.
Time = 0.003232 ms.
Time = 0.004096 ms.
Time = 0.00432 ms.
Time = 0.004096 ms.
Time = 0.0042528 +- 0.000794151 ms.

transpose with coalesced write:
Time = 0.013312 ms.
Time = 0.00512 ms.
Time = 0.004096 ms.
Time = 0.004128 ms.
Time = 0.004096 ms.
Time = 0.003968 ms.
Time = 0.004096 ms.
Time = 0.004096 ms.
Time = 0.004096 ms.
Time = 0.004096 ms.
Time = 0.004096 ms.
Time = 0.0041888 +- 0.000313026 ms.

transpose with coalesced write and __ldg read:
Time = 0.013312 ms.
Time = 0.004096 ms.
Time = 0.004096 ms.
Time = 0.004096 ms.
Time = 0.003232 ms.
Time = 0.003072 ms.
Time = 0.004096 ms.
Time = 0.004096 ms.
Time = 0.003072 ms.
Time = 0.004096 ms.
Time = 0.004096 ms.
Time = 0.0038048 +- 0.000446729 ms.
A =
0	1	2	3	4	5	6	7	
8	9	10	11	12	13	14	15	
16	17	18	19	20	21	22	23	
24	25	26	27	28	29	30	31	
32	33	34	35	36	37	38	39	
40	41	42	43	44	45	46	47	
48	49	50	51	52	53	54	55	
56	57	58	59	60	61	62	63	

B =
0	8	16	24	32	40	48	56	
1	9	17	25	33	41	49	57	
2	10	18	26	34	42	50	58	
3	11	19	27	35	43	51	59	
4	12	20	28	36	44	52	60	
5	13	21	29	37	45	53	61	
6	14	22	30	38	46	54	62	
7	15	23	31	39	47	55	63	
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/07-global-memory$ 
*/