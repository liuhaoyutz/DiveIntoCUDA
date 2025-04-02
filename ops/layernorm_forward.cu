/*
fork from https://github.com/karpathy/llm.c/blob/master/dev/cuda/layernorm_forward.cu

Kernels for layernorm forward pass.

Compile example:
nvcc -O3 --use_fast_math layernorm_forward.cu -o layernorm_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./layernorm_forward 1

version 2 parallelizes over all of B,T,C
./layernorm_forward 2

version 3 uses cooperative groups to parallelize over all of B,T,C
./layernorm_forward 3

version 4 uses a more clever way to estimate variance, var(x) = mean(x**2) - mean(x)**2
          (allowing us to do a single pass over x on load)
./layernorm_forward 4

verstion 5 allocates blocks per row instead of warps per row, same alg as 4 otherwise
./layernorm_forward 5
*/

/*
v1: 每个线程负责以for循环的方式，从全局内存读取并求C个数据的平均值和方差。

v2: 每个线程负责以for循环的方式，从全局内存读取并求C / block_size个数据的平均值和方差（thread coarsening），中间结果保存在共享内存中。然后对共享内存中的中间结果通过for循环进行归约（求和），再得到平均值和方差。相对于v1，以求mean为例，从v1的一个for循环变成v2的2个for循环。v2的第一个for循环负责处理C / block_size个数据，第二个for循环基于shared memory做reduce得到全部数据的mean。
假设有102400个数据，1024个线程，
对于v1，for循环将基于global memory循环102400次。
对于v2，第一个for循环每个线程负责100个数据，将基于global memory循环100次，第二个for循环每个线程基于shared memory循环10次。

关于warp内操作函数及协作组，参考https://github.com/brucefan1983/CUDA-Programming中第11章，线程束基本函数与协作组

v3: 每个线程负责以for循环的方式，从全局内存读取并求C / warp.size()个数据的平均值和方差（thread coarsening），然后通过协作组函数sum = cg::reduce(warp, sum, cg::plus<float>{})求和，再得到平均值和方差。相比v2，v3在归约步骤使用的是cg::reduce函数，而不是自己写的for循环，所以更高效。

总的来说，优化的思想是一共C个数据，有n个线程，每个线程负责C / n个数据，得到中间结果，然后再通过协作组或intra-warp reductions或共享内存方式进一步reduce n个线程的中间结果，得到最终数据。
*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// 层归一化前向传播的CPU实现
// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {  // 外层循环每次处理一个样本，每个batch有8个样本
        for (int t = 0; t < T; t++) {  // 内层循环每次处理一个time step，每个样本有1024个time step
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;  // x代表当前time step，包含768个特征值
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {  // 这个for循环计算当前time step的所有特征值之和，暂存在m中
                m += x[i];
            }
            m = m/C;  // 计算当前time step特征值的均值，赋值给m
            
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;  // v代表方差
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);  // s代表rstd

            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output  对当前time step的每一个特征值进行归一化
                float o = n * weight[i] + bias[i]; // scale and shift it   通过weight进行缩放，通过bias进行平移
                out_bt[i] = o; // write  写到输出中
            }
            
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels


/*
对于layernorm_forward_kernel1，总结一下：
B=8, T=1024, C=768，输入数据共有8192个样本，block_size为32，则调用layernorm_forward_kernel1核函数时，grid_size为256，即有256个block，每个block有32个线程，每个线程处理一个样本。
每个线程中：
为计算均值m，需要读取访问全局内存C次。
为计算方差v，因为数据被加载到寄存器或缓存中，后续操作可以直接使用这些已经加载的数据，而不需要再次访问全局内存。
为计算归一化输出n，也不再需要访问全局内存。
权重 (weight) 和 偏置 (bias)：这些通常是常量数组，通常会被加载到共享内存或寄存器中，以减少全局内存访问。因此，每个线程只会读取一次。
*/

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
__global__ void layernorm_forward_kernel1(float* out, float* mean, float* rstd,
                                 const float* inp, const float* weight, const float* bias,
                                 int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if (idx < N) {  // N等于B*T，即共有N个time step，所以只需要N个线程，每个线程处理一个time step，实际创建的线程可能比N多，剩余的线程不做任何计算
        // seek to the input position inp[idx,:]
        const float* x = inp + idx * C;  // x代表当前time step，包含768维特征值
        
        // 计算当前time step特征值的均值
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];  // 每次取x[i]都要访问global memory，共访问C次，所以效率低
        }
        m = m / C;

        // 计算当前time step特征值的方差
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;

        // 计算当前time step特征值的rstd
        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);

        // seek to the output position in out[idx,:]
        float* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m)); // normalized output  对当前time step的每一个特征值进行归一化
            float o = n * weight[i] + bias[i]; // scale and shift it  每个特征值使用weight进行缩放，通过bias进行偏移
            out_idx[i] = o; // write  写入output，每次写入都要访问global memory，共访问C次，所以效率低
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

/*
核函数mean_kernel实现了高效的均值计算，利用了共享内存来加速线程间的通信，并通过线程粗化和归约操作来最小化全局内存访问次数，从而提高了性能。
这种方法特别适合大规模数据集的并行处理，尤其是在需要频繁计算统计量（如均值）的情况下。

layernorm_forward_kernel1核函数中，每个线程处理一个样本，为了计算mean，需要C次访问global memory。而全局内存访问相对较慢，因此这种实现方式效率不高。

mean_kernel核函数，每个block处理一个样本，所以每个线程负责C / block_size个元素。所以，为了计算mean，每个线程只需要C / block_size次读取全局内存。
*/
__global__ void mean_kernel(float* mean, const float* inp, int N, int C, int block_size) {
    // 要使用动态定义的共享内存，第一，必须加上extern限定词；第二，不能指定数组大小。
    extern __shared__ float shared[];

    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    const float* x = inp + idx * C;  // x代表当前time step，包括768维特征值

    // 线程粗化（Thread Coarsening），每个线程负责累加一部分元素（从 tid 开始，每次跳跃 block_size 个元素），即每个线程负责C / block_size个元素，
    // 这样可以利用更多的线程来分担工作，提高并行度。
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }

    // 累加的结果存储在共享内存中，并调用 __syncthreads() 来确保所有线程都完成了它们的部分累加。
    shared[tid] = sum;
    __syncthreads();

    // 使用归约算法来计算线程块内所有线程累加结果的总和。每次迭代将相邻两个元素相加，并将结果存回第一个元素的位置。
    // stride初始为block_size 的一半，并在每次迭代后减半，直到变为1。
    // 只有当tid小于当前stride的线程才会执行加法操作，以避免越界访问。
    // reductions
    for (int stride = block_size >> 1; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }

    // 最终，只有线程0会将共享内存中的累积结果除以C特征维度），并将计算出的均值写入全局内存中的mean数组。
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

__global__ void rstd_kernel(float* rstd, const float* inp, const float* mean, int N, int C, int block_size) {
    // 要使用动态定义的共享内存，第一，必须加上extern限定词；第二，不能指定数组大小。
    extern __shared__ float shared[];

    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    const float* x = inp + idx * C;
    float m = mean[idx];

    // 线程粗化（Thread Coarsening），每个线程负责累加一部分元素（从 tid 开始，每次跳跃 block_size 个元素），这样可以利用更多的线程来分担工作，提高并行度。
    // 每个线程计算其负责部分的平方差和，并将结果存储在共享内存中。
    // 调用 __syncthreads() 来确保所有线程都完成了它们的部分累加。
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    
    // 使用归约算法来计算线程块内所有线程累加结果的总和。每次迭代将相邻两个元素相加，并将结果存回第一个元素的位置。
    // stride初始为block_size的一半，并在每次迭代后减半，直到变为1。
    // 只有当tid小于当前stride的线程才会执行加法操作，以避免越界访问。
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }

    // 最终，只有线程0会将共享内存中的累积结果除以C并加上一个小的常数1e-5f以防止除零错误，然后取平方根的倒数，
    // 将计算出的倒数标准差写入全局内存中的rstd数组。
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}

// 将输入数据归一化，并应用权重和偏置以生成最终的输出。
__global__ void normalization_kernel(float* out, const float* inp, float* mean, float* rstd,
                                     const float* weight, const float* bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程的全局索引idx

    int bt = idx / C;  // 当前处理的样本索引，范围是从0到 B * T - 1。
    int c = idx % C;  // 当前处理的特征维度索引，范围是从0到 C - 1。

    float m = mean[bt];  // 从mean数组中读取当前样本的均值。
    float s = rstd[bt];  // 从rstd数组中读取当前样本的倒数标准差。
    float xi = inp[idx];  // 从inp数组中读取当前元素的值。
    float n = s * (xi - m);  // 计算归一化后的值，公式为 n=rstd×(input−mean)。
    float o = n * weight[c] + bias[c];  // 应用权重和偏置，公式为 o=n×weight[c]+bias[c]。

    out[idx] = o;  // 将归一化并加权后的结果写入out数组。
}

__global__ void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    // 引入cooperative_groups命名空间，简化后续调用。
    namespace cg = cooperative_groups;

    // 获取当前线程块的引用。
    cg::thread_block block = cg::this_thread_block();
    // 将线程块划分为多个瓦片（warp），每个瓦片包含32个线程，以支持更细粒度的协作。
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // 计算当前处理的样本索引，范围是从0到N - 1。
    // meta_group_size is the number of warps in a block, and meta_group_rank is the warp index
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;  // x代表当前要处理的time step，包括768个特征值

    // mean
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    // 使用 cg::reduce 对瓦片内的线程进行归约操作，计算总和。注意这个cg::reduce函数的调用，这个就是协作组高效的地方。
    // 调用cg:reduce函数就完成了之前v2版本mean_kernel中for循环的工作，并且cg:reduce是硬件加速的，更加高效。
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;  // 求均值
    // 由瓦片中的第一个线程写入mean数组（如果提供了mean数组）。
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);  // __stcs函数的作用是将数据写入全局内存，并指定特殊的缓存行为。
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

// same as kernel 3 but uses var(x) == mean(x**2) - mean(x)**2
__global__ void layernorm_forward_kernel4(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // thread coarsening through the row, reduce the sum in series
    float sum = 0.0; // stores sum(x)
    float sum2 = 0.0; // stores sum(x**2)
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float xi = x[i];
        sum += xi;
        sum2 += xi * xi;
    }
    // warp-level reduction at the end
    sum = cg::reduce(warp, sum, cg::plus<float>{}); // sum(x)
    sum2 = cg::reduce(warp, sum2, cg::plus<float>{}); // sum(x**2)
    sum /= C; // mean(x)
    sum2 /= C; // mean(x**2)

    // mean, var, rstd
    float m = sum;
    float var = sum2 - sum * sum;
    float s = rsqrtf(var + 1e-5f);

    // store the mean, no need to cache it
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    // store the rstd, no need to cache it
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

// like 4, but in kernel 5 we have each block doing one row, not just a single warp
__global__ void layernorm_forward_kernel5(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum[32]; // block_size max is 1024 = 32 * 32 warps
    __shared__ float shared_sum2[32]; // warps will be writing into shared memeory after warp-reduce
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int idx = blockIdx.x; // simpoy one block per row
    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;
    // thread coarsening through the row, reduce the sum in series
    float thread_sum = 0.0; // stores sum(x)
    float thread_sum2 = 0.0; // stores sum(x**2)
    // for (int i = C + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = x[i];
        thread_sum += xi;
        thread_sum2 += xi * xi;
    }
    // warp-level reduction
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{}); // sum(x)
    float warp_sum2 = cg::reduce(warp, thread_sum2, cg::plus<float>{}); // sum(x**2)
    // store the warp-level reduction in shared memory (we could have lane_id == 0 guard but not needed)
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();
    // load results from shared memory to threads, pad with zeros for threads that are out of bounds
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    // now reduce the warp-level reductions
    float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{}); // sum(x)
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{}); // sum(x**2)
    // mean, var, rstd
    block_sum /= C; // mean(x)
    block_sum2 /= C; // mean(x**2)
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = rsqrtf(var + 1e-5f);
    // store the mean, no need to cache it
    if(threadIdx.x == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    // store the rstd, no need to cache it
    if(threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n = s * (__ldcs(x+i) - m);
        __stcs(o+i, n * weight[i] + bias[i]);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void layernorm_forward1(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C,
                           const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size); // 进1法取整：(dividend + divisor-1) / divisor
    layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward2(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    int N = B * T;

    // 核函数执行配置的第三个参数block_size * sizeof(float)代表每个线程块需要定义的共享内存的字节数
    // in mean and rstd, threads cooperate within blocks via reductions
    mean_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(mean, inp, N, C, block_size);
    cudaCheck(cudaGetLastError());
    rstd_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(rstd, inp, mean, N, C, block_size);
    cudaCheck(cudaGetLastError());
    // in the normalization, everything just gets flattened out
    const int block_size2 = 256;
    const int grid_size = ceil_div(B * T * C, block_size2);
    normalization_kernel<<<grid_size, block_size2>>>(out, inp, mean, rstd, weight, bias, B, T, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward3(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward4(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    layernorm_forward_kernel4<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward5(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = N;
    layernorm_forward_kernel5<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void layernorm_forward(int kernel_num,
                    float* out, float* mean, float* rstd,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 1:
            layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            layernorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 3:
            layernorm_forward3(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 4:
            layernorm_forward4(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 5:
            layernorm_forward5(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;  // batch size为8，模型一次处理8个样本
    int T = 1024;  // 每个样本包含1024个time step
    int C = 768;  // 每个time step的特征向量维度是768

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));  // 输出
    float* mean = (float*)malloc(B * T * sizeof(float));  // 每个time step激活值的均值
    float* rstd = (float*)malloc(B * T * sizeof(float));  // 每个time step的激活值逆标准差（reciprocal standard deviation）
    float* inp = make_random_float(B * T * C);  // 输入
    float* weight = make_random_float(C);  // 归一化层的权重
    float* bias = make_random_float(C);  // 归一化层的偏置

    // move to GPU
    float* d_out;
    float* d_mean;
    float* d_rstd;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));  // 输出
    cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(float)));  // 每个time step激活值的均值
    cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(float)));  // 每个time step激活值的逆标准差
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));  // 输入
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));  // 归一化层的权重
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));  // 归一化层的偏置

    // 将输入、权重、偏置从CPU拷贝到GPU
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    /*
    // 用于将GPU的相应值保存到CPU，没有用到
    float* out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float* mean_gpu = (float*)malloc(B * T * sizeof(float));
    float* rstd_gpu = (float*)malloc(B * T * sizeof(float));
    */

    // 基于CPU的层归一化前向传播
    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        // 基于GPU的层归一化前向传播，kernel_num指定使用哪种核函数实现方案
        layernorm_forward(kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);

        // 验证GPU和CPU归一化结果是否一致
        validate_result(d_out, out, "out", B * T * C, 1e-5f);
        validate_result(d_mean, mean, "mean", B * T, 1e-5f);
        validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
    }

    printf("\nAll results match. Starting benchmarks.\n");

    // 统计在不同block size情况下，核函数进行层归一化前向传播的耗时
    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm_forward,
                                              kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_mean));
    cudaCheck(cudaFree(d_rstd));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));

    return 0;
}