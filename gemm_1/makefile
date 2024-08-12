default:
	nvcc gemm.cu -O2 -arch=sm_89 -std=c++17  -I ../ -I ../thirdparty/cutlass/include -I ../thirdparty/cutlass/tools/util/include --expt-relaxed-constexpr -lcublas