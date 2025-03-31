#include <stdio.h>

// CUDA核函数与C++函数类似，但一个显著的差别是：它必须被限定词__global__修饰。
// 另外，核函数的返回类型必须是空类型（即void）。限定符__global__和void的次序可随意。
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    // 调用核函数，(grid_size为1, block_size为1)，所以共1个线程
    hello_from_gpu<<<1, 1>>>();

    // 同步主机与设备，能够促使缓冲区刷新。
    cudaDeviceSynchronize();
    return 0;
}

/*
执行结果如下：
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ nvcc hello2.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ ./a.out 
Hello World from the GPU!
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ 
*/