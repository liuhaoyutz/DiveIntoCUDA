# DiveIntoCUDA
  
算子编译命令：  
cd ops  
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_forward.cu -o layernorm_forward  
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_backward.cu -o layernorm_backward  
nvcc -O3 --use_fast_math -lcublas -lcublasLt softmax_forward.cu -o softmax_forward  

cd torch-cuda

测试JIT方式：
python pytorch/time.py --compiler jit

测试setup方式：
python pytorch/setup.py install
python pytorch/time.py --compiler setup

测试cmake方式：
mkdir build
cd build
cmake ../pytorch/
make
cd ../
python pytorch/time.py --compiler cmake

Reference:  
https://github.com/brucefan1983/CUDA-Programming  
https://github.com/godweiyang/torch-cuda-example  
https://godweiyang.com/2021/03/18/torch-cpp-cuda/  
