// 核函数要加描述符__global__，返回类型必须是void，即没有返回值
__global__ void add2_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n)
{
    // 该核函数的作用是将数组a和数组b的对应元素相加，并将结果存储到数组c中。
    // i是线程对应的索引。
    // n是数组大小。
    // gridDim.x * blockDim.x是每个grid包含的线程数。
    // 如果n <= gridDim.x * blockDim.x, 则只循环一次就能全部处理完。
    // 如果n > gridDim.x * blockDim.x, 则要通过多次循环处理。
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        c[i] = a[i] + b[i];
    }
}

// host函数，用于调用核函数
void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    // grid_size = (N + block_size - 1) / block_size
    dim3 block(1024);
    dim3 grid((n + 1023) / 1024);

    // 调用核函数，grid是线程块个数((n+1023)/1024)，block是每个线程块包含的线程个数（1024）
    // 注意launch_add2是异步调用add2_kernel的，调用完控制权立即返还给CPU，所以后面分析add2_kernel用时时要小心，不要只统计调用时间。
    add2_kernel<<<grid, block>>>(c, a, b, n);
}