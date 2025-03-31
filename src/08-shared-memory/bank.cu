// 这段代码是一个基于CUDA的矩阵转置示例，主要展示了两种不同的共享内存实现方式（有和没有共享内存银行冲突）对性能的影响。

#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;

void timing(const real *d_A, real *d_B, const int N, const int task);
__global__ void transpose1(const real *A, real *B, const int N);
__global__ void transpose2(const real *A, real *B, const int N);
void print_matrix(const int N, const real *A);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: %s N\n", argv[0]);
        exit(1);
    }
    const int N = atoi(argv[1]);

    const int N2 = N * N;
    const int M = sizeof(real) * N2;
    real *h_A = (real *) malloc(M);
    real *h_B = (real *) malloc(M);
    for (int n = 0; n < N2; ++n)
    {
        h_A[n] = n;
    }
    real *d_A, *d_B;
    CHECK(cudaMalloc(&d_A, M));
    CHECK(cudaMalloc(&d_B, M));
    CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));

    // 调用 timing 函数，分别测试两种转置实现的性能：
    // transpose1：存在共享内存bank冲突。
    // transpose2：优化以避免共享内存bank冲突。
    printf("\ntranspose with shared memory bank conflict:\n");
    timing(d_A, d_B, N, 1);
    printf("\ntranspose without shared memory bank conflict:\n");
    timing(d_A, d_B, N, 2);

    CHECK(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));
    if (N <= 10)
    {
        printf("A =\n");
        print_matrix(N, h_A);
        printf("\nB =\n");
        print_matrix(N, h_B);
    }

    free(h_A);
    free(h_B);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    return 0;
}

void timing(const real *d_A, real *d_B, const int N, const int task)
{
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
        cudaEventQuery(start);

        switch (task)
        {
            case 1:
                transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 2:
                transpose2<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            default:
                printf("Error: wrong task\n");
                exit(1);
                break;
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
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

__global__ void transpose1(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 < N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose2(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM + 1];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 < N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

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

(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/08-shared-memory$ nvcc bank.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/08-shared-memory$ ./a.out 8

transpose with shared memory bank conflict:
Time = 0.191488 ms.
Time = 0.007168 ms.
Time = 0.00512 ms.
Time = 0.005984 ms.
Time = 0.004928 ms.
Time = 0.004096 ms.
Time = 0.0048 ms.
Time = 0.004096 ms.
Time = 0.003296 ms.
Time = 0.004096 ms.
Time = 0.004928 ms.
Time = 0.0048512 +- 0.00104245 ms.

transpose without shared memory bank conflict:
Time = 0.017408 ms.
Time = 0.00512 ms.
Time = 0.004096 ms.
Time = 0.004096 ms.
Time = 0.00512 ms.
Time = 0.003808 ms.
Time = 0.004096 ms.
Time = 0.004096 ms.
Time = 0.004064 ms.
Time = 0.005824 ms.
Time = 0.003968 ms.
Time = 0.0044288 +- 0.000638432 ms.
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
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/08-shared-memory$ 
*/