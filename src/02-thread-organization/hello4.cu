#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;  // block ID
    const int tid = threadIdx.x;  // thread ID
    printf("Hello World from block %d and thread %d!\n", bid, tid);
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}

/*
执行结果如下：
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ nvcc hello4.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ ./a.out 
Hello World from block 1 and thread 0!
Hello World from block 1 and thread 1!
Hello World from block 1 and thread 2!
Hello World from block 1 and thread 3!
Hello World from block 0 and thread 0!
Hello World from block 0 and thread 1!
Hello World from block 0 and thread 2!
Hello World from block 0 and thread 3!
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ 
*/