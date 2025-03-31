// 下面是一个只有host代码的CUDA程序，打印Hello World!
// 可以看到，只有host代码的CUDA程序和标准C++程序完全一样，这是因为，CUDA程序的编译器nvcc支持编译纯粹的C++代码。
// 一般来说，一个标准的CUDA程序中既有纯粹的C++代码，也有不属于C++的真正的CUDA代码。
// nvcc在编译一个CUDA程序时，会将纯粹的C++代码交给C++编译器（例如g++），它自己则负责编译剩下的部分。
#include <stdio.h>

int main(void)
{
    printf("Hello World!\n");
    return 0;
}

/*
执行结果如下：
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ nvcc hello1.cu 
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ ./a.out 
Hello World!
(INT8) haoyu@thinker:~/work/code/DiveIntoCUDA/src/02-thread-organization$ 
*/