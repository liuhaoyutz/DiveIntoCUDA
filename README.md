# DiveIntoCUDA
  
算子编译命令：  
cd ops  
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_forward.cu -o layernorm_forward  
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_backward.cu -o layernorm_backward  
nvcc -O3 --use_fast_math -lcublas -lcublasLt softmax_forward.cu -o softmax_forward  

Reference:  
https://github.com/brucefan1983/CUDA-Programming  
