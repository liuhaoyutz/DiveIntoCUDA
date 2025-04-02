/*
Kernels for softmax forward pass.

Compile example:
nvcc -O3 --use_fast_math softmax_forward.cu -o softmax_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./softmax_forward 1

version 2 is a fused kernel that parallelizes over all of B,T,C
./softmax_forward 2

version 3 uses intra-warp reductions for maxval and sumval, must use block_size=32
./softmax_forward 3

version 4 uses both intra-warp reductions and shared memory for inter-warp reductions
so it can tolerate any block_size % 32 == 0. this is hopefully the most efficient version
./softmax_forward 4

version 5 is naive port from CPU code (softmax_online) to kernel: parallelizes over B,T, loops over C
./softmax_forward 5

version 6 is softmax_online that parallelizes over all of B,T,C
./softmax_forward 6

version 7 is softmax optimized for very large C.
./softmax_forward 7
*/

/*
Softmax 是一种基本的归一化函数，它可以将一个数值向量归一化为一个概率分布向量，且各个概率之和为1。Softmax 可以用来作为神经网络的最后一层，用于多分类问题的输出。

Softmax溢出问题是指在计算Softmax函数时，由于输入值过大导致指数运算结果超出计算机浮点数表示范围的问题。为了避免溢出，通常要先减最大值。

Softmax是self-attention里计算score的关键环节，而最关键的实际上是两个reduce操作，即max和sum。

v1: 每个线程负责以for循环的方式，从全局内存读取并求C个数据的maxval和sum。

v2: 每个线程负责以for循环的方式，从全局内存读取并求C / block_size个数据的maxval，中间结果放入共享内存，然后对共享内存中的中间结果归约得到block范围内的maxval。然后求C / block_size个数据的sumval，中间结果放入共享内存，然后对共享内存中的中间结果归约得到block范围内的sumval。最后计算softmax结果。

关于warp内操作函数及协作组，参考https://github.com/brucefan1983/CUDA-Programming中第11章，线程束基本函数与协作组

v3: 每个线程负责以for循环的方式，从全局内存读取并求C / blockDim.x（50257 / 32）个数据的maxval，然后用__shfl_down_sync 内建函数进行warp内的数据操作，进而取得warp内的maxval。然后求C / blockDim.x个数据的expf和sumval。然后用__shfl_down_sync 内建函数进行warp内的数据共享，进而取得warp内的sumval。最后计算softmax结果。
也就是说一个block固定为32个线程，即一个block对应一个warp。一个block处理50257个数据，所以每个线程处理50257 / 32个数据，每个线程取得这些数据的最大值（或和），然后调用__shfl_down_sync在warp内提取这32个线程的最大值（或和），最后计算softmax结果。

v2归约使用的共享内存，而v3使用更高效的__shfl_down_sync 内建函数进行warp内的数据共享。

总的来说，优化的思想是一共C个数据，有n个线程，每个线程负责C / n个数据，得到中间结果，然后再通过协作组或intra-warp reductions或共享内存方式进一步reduce n个线程的中间结果，得到最终数据。
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference
void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    // N=B*T，即N代表time step个数。
    // C代表每个time step有C个特征值，C为50257，即每个time step有50257个特征值
    for (int i = 0; i < N; i++) {  // 外层循环每次处理一个time step
        const float* inp_row = inp + i * C;  // inp_row对应一个time step
        float* out_row = out + i * C;  // out_row也对应一个time step

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {  // 遍历当前time step的所有特征值，找到其中的最大值，赋值给maxval
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }

        // Note: since we want to ensure that the CUDA-kernels are accurate,
        // we do this accumulation in higher precision, so we can be assured
        // that our ground-truth is of high quality.
        double sum = 0.0;
        for (int j = 0; j < C; j++) {  // 遍历当前time step
            out_row[j] = expf(inp_row[j] - maxval);  // 每个特征值 - 最大特征值，然后做指数运算
            sum += out_row[j];  // 对所有特征值的指数运算结果求和
        }
        float norm = 1.f / (float)sum;  // 将所有特征值的指数运算结果之和的倒数，赋值给norm
        for (int j = 0; j < C; j++) {  // 遍历当前time step
            out_row[j] *= norm;  // 求每个特征值的softmax值
        }
    }
}


// online version of softmax on CPU from the paper "Online normalizer calculation for softmax"
void softmax_forward_online_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    // N=B*T，即N代表time step个数。
    // C代表每个time step有C个特征值，C为50257，即每个time step有50257个特征值
    for (int i = 0; i < N; i++) {  // 外层循环每次处理一个time step
        const float* inp_row = inp + i * C;  // inp_row对应一个time step
        float* out_row = out + i * C;  // out_row也对应一个time step

        float maxval = -INFINITY;
        float sum = 0.0f;
		for (int j = 0; j < C; j++) {  // 遍历当前time step的所有特征值，找到其中的最大值，赋值给maxval，并分情况计算sum
			float maxval_prev = maxval;
			if (inp_row[j] > maxval) {
				maxval = inp_row[j];
				sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
			} else {
				sum += expf(inp_row[j] - maxval);
			}
		}

        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval) / sum;  // 求每个特征值的softmax值
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void softmax_forward_kernel1(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    // N = B * T，即N为time step个数
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // i是每个线程对应的index
    if (i < N) {  // 每个线程对应一个time step，处理该time step对应的C个特征值
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {  // 遍历当前时间步对应的所有特征值，找出最大特征值
            if (inp_row[j] > maxval) {  // 每一次访问inp_row[j]都需要访问global memory，共循环C次，所以效率低
                maxval = inp_row[j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);  // 每一次访问out_row[j]都要访问global memory，共循环C次，所以效率低
            sum += out_row[j];
        }

        for (int j = 0; j < C; j++) {
            out_row[j] /= (float)sum;  // 计算每个特征值对应的softmax值，每次保存out_row[j]值都要访问global memory
        }
    }
}

__global__ void softmax_forward_kernel2(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // in each row of C elements, first calculates maxval, then returns expf(val - maxval)

    // 要使用动态定义的共享内存，第一，必须加上extern限定词；第二，不能指定数组大小。
    extern __shared__ float shared[];

    int idx = blockIdx.x; // ranges [0, N)
    int tid = threadIdx.x; // ranges [0, block_size)
    int block_size = blockDim.x;
    const float* x = inp + idx * C; // idx-th row of inp, inp[idx,:]
    
    // 线程粗化（Thread Coarsening），每个线程负责累加一部分元素（从 tid 开始，每次跳跃 block_size 个元素），即每个线程负责C / block_size个元素，
    // 这样可以利用更多的线程来分担工作，提高并行度。
    // thread coarsening
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size) {
        maxval = fmaxf(maxval, x[i]);
    }
    shared[tid] = maxval;  // 将当前线程负责的maxval写入共享内存
    __syncthreads();
    
    // 通过共享内存对中间maxval进行归约，求得全部特征值的最大值
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();
    float offset = shared[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = expf(x[i] - offset);
    }
    __syncthreads();

    // 线程粗化，每个线程负责C / block_size个特征值。计算每个线程负责的特征值的sum。
    // thread coarsening again, for the sum
    x = out + idx * C; // idx-th row of out
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sumval += x[i];
    }
    shared[tid] = sumval;  // 将当前线程负责的C / block_size个特征值的sumval保存在共享内存中
    __syncthreads();

    // 在共享内存中对每个线程的sum进行归约（累加），得到所有特征值的sum。
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // broadcast the sum to all threads in the block
    __syncthreads();
    float sum = shared[0];

    // divide the input values by the sum
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = x[i] / sum;  // 计算每个特征值的softmax值
    }
}

// 用 __device__ 修饰的函数称为设备函数，只能被核函数或其他设备函数调用，在设备中执行。

/*
__shfl_down_sync(mask, v, d, w)：标号为 t 的参与线程返回标号为 t + d 的线程
中变量 v 的值。标号满足 t + d >= w 的线程返回原来的 v。例如：当 w = 8，d = 2 时，
该函数将第 2-7 号线程中变量 v 的值传送到第 0-5 号线程，而第 6-7 号线程返回它们
原来的 v。形象地说，这是一种将数据向下平移的操作。
*/

/*
假设当前线程的线程号为tid，则__shfl_down_sync(0xFFFFFFFF, val, offset)的作用是返回线程号为tid+offset的线程的val值。
第一轮循环：offset为16。
对于第0号线程，__shfl_down_sync返回的是0+16号线程的val值，所以，fmaxf是比较0号线程和16号线程的val值，并将较大的那个保存在0号线程的的val变量中。
对于第1号线程，__shfl_down_sync返回的是1+16号线程的val值，所以，fmaxf是比较1号线程和17号线程的val值，并将较大的那个保存在1号线程的的val变量中。
... ...
对于第15号线程，__shfl_down_sync返回的是15+16号线程的val值，所以，fmaxf是比较15号线程和31号线程的val值，并将较大的那个保存在15号线程的的val变量中。
对于第16到31号线程，tid+offset > warp_size了，__shfl_down_sync返回自己的val，相当于不变。形象的说，__shfl_down_sync是一种数据向下平移的操作。


第二轮循环，offset为8。
对于第0号线程，__shfl_down_sync返回的是0+8号线程的val值，所以，fmaxf是比较0号线程和8号线程的val值，并将较大的那个保存在0号线程的的val变量中。
... ...
对于第7号线程，__shfl_down_sync返回的是7+8号线程的val值，所以，fmaxf是比较7号线程和15号线程的val值，并将较大的那个保存在7号线程的的val变量中。

第三轮循环，offset为4。
对于第0号线程，__shfl_down_sync返回的是0+4号线程的val值，所以，fmaxf是比较0号线程和4号线程的val值，并将较大的那个保存在0号线程的的val变量中。
... ...
对于第3号线程，__shfl_down_sync返回的是3+4号线程的val值，所以，fmaxf是比较3号线程和7号线程的val值，并将较大的那个保存在3号线程的的val变量中。

第四轮循环，offset为2。
对于第0号线程，__shfl_down_sync返回的是0+2号线程的val值，所以，fmaxf是比较0号线程和2号线程的val值，并将较大的那个保存在0号线程的的val变量中。
... ...
对于第1号线程，__shfl_down_sync返回的是1+2号线程的val值，所以，fmaxf是比较1号线程和3号线程的val值，并将较大的那个保存在1号线程的的val变量中。

第五轮循环，offset为1。
对于第0号线程，__shfl_down_sync返回的是0+1号线程的val值，所以，fmaxf是比较0号线程和1号线程的val值，并将较大的那个保存在0号线程的的val变量中。

循环结束。这样，0号线程的val就是整个warp的最大val。
*/

// 该函数查找并返回线程束范围内的最大值
// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// 用 __device__ 修饰的函数称为设备函数，只能被核函数或其他设备函数调用，在设备中执行。
// 该函数返回线程束范围内所有val之和
// warp-level reduction for summing values
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel3(float* out, const float* inp, int N, int C) {
    // kernel must use block size of 32
    // 要使用动态定义的共享内存，第一，必须加上extern限定词；第二，不能指定数组大小。
    extern __shared__ float shared[];  // 这个共享内存没有使用

    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = inp + idx * C;  // x代表当前time step

    // 线程粗化，计算局部max
    // Thread coarsening and within-warp reduction for maxval
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    maxval = warpReduceMax(maxval);  // 调用设备函数warpReduceMax取得warp内最大值

    // 这个__shfl_sync调用在warp内每个线程中，都返回0号线程的maxval。所以这一行的使用是把0号线程的maxval广播到每个线程，并保存为offset。
    // Broadcast maxval within the warp
    float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);

    // 计算expf
    // Compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    // 线程粗化，计算局部sum
    // Thread coarsening and within-warp reduction for sumval
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    sumval = warpReduceSum(sumval);  // 调用设备函数warpReduceSum取得warp内sum

    // 在warp内广播sumval
    // Broadcast sumval within the warp
    float sum = __shfl_sync(0xFFFFFFFF, sumval, 0);

    // Divide the input values by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;  // 计算每个特征值的softmax值
    }
}

__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}

__global__ void softmax_forward_online_kernel1(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            float maxval_prev = maxval;
			if (inp_row[j] > maxval) {
				maxval = inp_row[j];
				sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
			}
			else {
				sum += expf(inp_row[j] - maxval);
			}
		}

        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}

// struct for the reduction operation, guarantees 8-byte alignment
struct __align__(8) SumMax
{
    float maxval;
    float sum;
};

// forceinline helps avoid function call overhead
__device__ __forceinline__ SumMax reduce_sum_max_op(SumMax a, SumMax b) {
    bool a_bigger = (a.maxval > b.maxval);
    SumMax bigger_m = a_bigger ? a : b;
    SumMax smaller_m = a_bigger ? b : a;
    SumMax res;
    res.maxval = bigger_m.maxval;
    res.sum = bigger_m.sum + smaller_m.sum * expf(smaller_m.maxval - bigger_m.maxval);
    return res;
}

__global__ void softmax_forward_online_kernel2(float* out, const float* inp, int N, int C) {
	namespace cg = cooperative_groups;
	cg::thread_block block = cg::this_thread_block();
	cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
	int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
	if (idx >= N) {
		return;
	}

	// one row of inp, i.e. inp[idx, :] of shape (C,)
	const float* x = inp + idx * C;

    // base case for the reduction
    SumMax sm_partial;
	sm_partial.maxval = -INFINITY;
	sm_partial.sum = 0.0f;

	// first, thread coarsening by directly accessing global memory in series
	for (int i = warp.thread_rank(); i < C; i += warp.size()) {
		sm_partial = reduce_sum_max_op(sm_partial, { x[i], 1.0f });
	}

    // second, the reduction
	SumMax sm_total = cg::reduce(warp, sm_partial, reduce_sum_max_op);

	// divide the whole row by the sum
	for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        // the below is equivalent to
        // out[idx * C + i] = expf(x[i] - sm_total.maxval) / sm_total.sum;
        // but uses special instruction that bypasses the cache
        __stcs(out + idx * C + i, expf(x[i] - sm_total.maxval) / sm_total.sum);
	}
}

__global__ void softmax_forward_kernel7(float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel4, but optimised for very large Cs with advanced unrolling

    // The trick is to read into a register array (all indices known at compile time)
    // and always read UNROLL_FACTOR values to maximise memory level parallelism
    // even if we would be out of bounds, we set the index to min(C-1, idx)
    // so we just do some unnecessary reads (obviously bad for small C)
    // the writes are in a separate loop with a conditional check for out of bounds
    // making it separate is necessary to convince the compiler to do the right thing
    const int UNROLL_FACTOR = 8;
    const int warpsPerBlock = blockDim.x / 32;

    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }

    const float* x = inp + idx * C; // input
    float* y = out + idx * C; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            maxval = fmaxf(maxval, x[min(C - 1, i + u*blockDim.x)]);
        }
    }

    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);
    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();
    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    // + thread coarsening for sum
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = __ldcs(&x[min(C - 1, i + u*blockDim.x)]);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                float output = expf(reg_array[u] - offset);
                y[min(C - 1, i + u*blockDim.x)] = output; // compiler likes redundant min()?!
                sumval += output; // combined into the same loop unlike kernel3
            }
        }
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);
    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();
    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[min(C - 1, i + u*blockDim.x)];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                y[i + u*blockDim.x] = reg_array[u] / sum;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void softmax_forward1(float* out, const float* inp, int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    softmax_forward_kernel1<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward2(float* out, const float* inp, int N, int C, const int block_size) {
    int grid_size = N;
    size_t shared_mem_size = block_size * sizeof(float);
    softmax_forward_kernel2<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
}

void softmax_forward3(float* out, const float* inp, int N, int C, int block_size) {
    block_size = 32; // awkward but ok. this one only works with block size 32
    int grid_size = N;
    size_t shared_mem_size = block_size * sizeof(float);
    softmax_forward_kernel3<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
}

void softmax_forward4(float* out, const float* inp, int N, int C, int block_size) {
    int grid_size = N;
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
}

void softmax_forward_online1(float* out, const float* inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N, block_size);
    softmax_forward_online_kernel1 <<<grid_size, block_size >>> (out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward_online2(float* out, const float* inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N * 32, block_size);
    softmax_forward_online_kernel2 <<<grid_size, block_size >>> (out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward7(float* out, const float* inp, int N, int C, int block_size) {
    int grid_size = N;
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    softmax_forward_kernel7<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
}

// kernel version dispatch
void softmax_forward(int kernel_num, float* out, const float* inp, int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1:
            softmax_forward1(out, inp, N, C, block_size);
            break;
        case 2:
            softmax_forward2(out, inp, N, C, block_size);
            break;
        case 3:
            softmax_forward3(out, inp, N, C, block_size);
            break;
        case 4:
            softmax_forward4(out, inp, N, C, block_size);
            break;
        case 5:
            softmax_forward_online1(out, inp, N, C, block_size);
            break;
        case 6:
            softmax_forward_online2(out, inp, N, C, block_size);
            break;
        case 7:
            softmax_forward7(out, inp, N, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;  // 每个batch包含8个样本
    int T = 1024;  // 每个样本包含1024个time step
    int V = 50257;  // 每个time step包含50257个特征值

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * V * sizeof(float));
    float* inp = make_random_float(B * T * V);

    // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
    // and the tests are not actually meaningful.
    const int* outliers = make_random_int(B * T * 3, V);
    for(int k = 0; k < 3; ++k) {
        for(int j = 0; j < B * T; ++j) {
            inp[j * V +  outliers[j*3 + k]] *= 20;
        }
    }


    // move to GPU
    float* d_out;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * V * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * V * sizeof(float), cudaMemcpyHostToDevice));  // 将CPU上的inp拷贝到GPU上的d_inp

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    // softmax forward的CPU实现
    softmax_forward_cpu(out, inp, B * T, V);
    {
        float max_el = -INFINITY;
        for(int i = 0; i <  B * T * V; ++i) {
            max_el = max(max_el, out[i]);
        }
        assert(max_el > 1e-4);
        printf("Largest output is: %f\n", max_el);
    }

    // first check the correctness of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        softmax_forward(kernel_num, d_out, d_inp, B * T, V, block_size);  // softmax forward的GPU实现
        validate_result(d_out, out, "out", B * T * V, 1e-4f);  // 比较CPU实现和GPU实现结果是否相同
    }

    printf("All results match. Starting benchmarks.\n\n");

    // 对softmax的GPU实现进行性能评估
    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, softmax_forward,
                                              kernel_num, d_out, d_inp, B * T, V, block_size
                                              );

        printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (B*T));
    }

    // free memory
    free(out);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));

    return 0;
}