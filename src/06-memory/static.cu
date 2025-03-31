// 这段代码展示了怎样定义和使用静态全局内存变量

// 修饰符__device__说明该变量是设备中的变量，而不是主机中的变量

/*
在核函数中，可直接对静态全局内存变量进行访问，并不需要将它们以参数的形式传给核
函数。不可在主机函数中直接访问静态全局内存变量，但可以用cudaMemcpyToSymbol函
数和cudaMemcpyFromSymbol函数在静态全局内存与主机内存之间传输数据。
*/

/*
执行结果如下：
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/06-memory$ nvcc static.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/06-memory$ ./a.out 
d_x = 1, d_y[0] = 11, d_y[1] = 21.
h_y[0] = 11, h_y[1] = 21.
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/06-memory$
*/

#include "error.cuh"
#include <stdio.h>

// 定义单个静态全局内存变量
__device__ int d_x = 1;
// 定义固定长度的静态全局内存变量数组
__device__ int d_y[2];

// 定义核函数
void __global__ my_kernel(void)
{
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);
}

int main(void)
{
    int h_y[2] = {10, 20};
    // 将host数组h_y的内容复制给静态全局内存变量数组d_y
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));
    
    my_kernel<<<1, 1>>>();
    // 等待核函数运行结束
    CHECK(cudaDeviceSynchronize());
    
    // 将静态全局变量数组d_y的内容复制给host数组h_y
    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);
    
    return 0;
}

