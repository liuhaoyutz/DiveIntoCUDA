#include <stdio.h>

__global__ void hello_from_gpu()
{
    // blockIdx和threadIdx都是uint3类型，是3维的
    const int bx = blockIdx.x;  // block ID.x
    const int by = blockIdx.y;  // block ID.y
    const int bz = blockIdx.z;  // block ID.z
    const int tx = threadIdx.x;  // thread ID.x
    const int ty = threadIdx.y;  // thread ID.y
    const int tz = threadIdx.z;  // thread ID.z
    printf("Hello World from block-(%d, %d, %d) and thread-(%d, %d, %d)!\n", bx, by, bz, tx, ty, tz);
}

int main(void)
{
    // grid_size和block_size都是dim3类型，是3维的，如果某个维度不指定，则默认值为1
    const dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}

// 执行结果如下：
/*
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ nvcc hello5.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ ls
a.out  hello1.cu  hello2.cu  hello3.cu  hello4.cu  hello5.cu  hello.cpp  readme.md
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ ./a.out 
Hello World from block-(0, 0, 0) and thread-(0, 0, 0)!
Hello World from block-(0, 0, 0) and thread-(1, 0, 0)!
Hello World from block-(0, 0, 0) and thread-(0, 1, 0)!
Hello World from block-(0, 0, 0) and thread-(1, 1, 0)!
Hello World from block-(0, 0, 0) and thread-(0, 2, 0)!
Hello World from block-(0, 0, 0) and thread-(1, 2, 0)!
Hello World from block-(0, 0, 0) and thread-(0, 3, 0)!
Hello World from block-(0, 0, 0) and thread-(1, 3, 0)!
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ 
*/