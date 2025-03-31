#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    // (grid_size为2, block_size为4)，所以共8个线程，每个线程都调用核函数打印一条语句，共打印8条语句。
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}

/*
执行结果如下：
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ nvcc hello3.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ ./a.out 
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ 
*/