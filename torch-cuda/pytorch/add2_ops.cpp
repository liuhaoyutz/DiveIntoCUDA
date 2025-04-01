// 这段代码展示了怎样创建PyTorch自定义算子

// 包含了PyTorch C++扩展API的头文件，允许开发者创建自定义算子
#include <torch/extension.h>
#include "add2.h"

/*
torch_launch_add2函数传入的是C++版本的torch tensor，然后使用data_ptr()方法来获取张量底层数据的指针，
并将其转换为float*类型传递给launch_add2来执行核函数。
*/
void torch_launch_add2(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n) {
    launch_add2((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

/*
注意：这是添加Python模块的通用操作，通过Python模块，让Python代码可以访问C++代码（例如这里的torch_launch_add2函数就是C++代码）

定义了一个新的Python模块，模块的名字TORCH_EXTENSION_NAME会在编译时自动设置。
m是这个模块的对象，通过它向模块中添加了torch_launch_add2函数。
*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
}

/*
注意：这是添加PyTorch算子库，与前面的添加Python模块不一样，这是PyTorch专用的操作。

注册一个新的PyTorch算子库，其namespace为add2，该namespace下添加了一个算子torch_launch_add2。

参考https://pytorch.org/cppdocs/api/define_library_8h_1a0bd5fb09d25dfb58e750d712fc5afb84.html
TORCH_LIBRARY(ns, m)
Macro for defining a function that will be run at static initialization time to define a library of operators in the namespace ns (must be a valid C++ identifier, no quotes).
Use this macro when you want to define a new set of custom operators that do not already exist in PyTorch.
*/
TORCH_LIBRARY(add2, m) {
    m.def("torch_launch_add2", torch_launch_add2);
}

/*
通过cmake/jit/setuptools方式对pytorch/add2_ops.cpp和kernel/add2_kernel.cu文件进行编译，就可得到libadd2.so动态库。
*/