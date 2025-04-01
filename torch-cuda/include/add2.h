// 对launch_add2函数的声明, 函数内容定义在torch-cuda/kernel/add2_kernel.cu文件中
void launch_add2(float *c,
                 const float *a,
                 const float *b,
                 int n);