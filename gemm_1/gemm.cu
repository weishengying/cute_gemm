#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <float.h>
#include "utils.h"

using T = cute::half_t;
using namespace cute;

template<typename T, int bM, int bN, int bK, typename TiledMMA>
__global__ void gemm_device(const T* Aptr, const T* Bptr, T* Cptr, 
                            int m, int n, int k) {
  using namespace cute;
  using TA = float;
  using TB = float;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{})); //(m,k) row-major
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{})); //(n,k) row-major
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{})); //(m,n) row-major

  // Get the appropriate blocks for this thread block
  int ix = blockIdx.x;
  int iy = blockIdx.y;             
  Tensor gA = local_tile(A, make_tile(Int<bM>{}, Int<bK>{}), make_coord(iy, _));  // (b_M,b_K,num_tile_k)
  Tensor gB = local_tile(B, make_tile(Int<bN>{}, Int<bK>{}), make_coord(ix, _));  // (b_N,b_K,num_tile_k)
  Tensor gC = local_tile(C, make_tile(Int<bM>{}, Int<bN>{}), make_coord(iy, ix)); // (b_M,b_N)
  
  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
  Tensor tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)
  

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_K, MMA_N)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)

  clear(tCrC); //将寄存器中的数据初始化为0
  int num_tile_k = size<2>(gA);

  #pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC); 
}

template<typename T>
void gemm_v1(const T* A, const T* B, T* C, int M, int N, int K) {
  const int bM = 128;
  const int bN = 256;
  const int bK = 32;

  using mma_op = SM80_16x8x16_F16F16F16F16_TN; // A(M,K) row-major. B(K,N) col-major 等同于 B(N,K) row-major
  using mma_traits = MMA_Traits<mma_op>;
  using MMA_Atom_Arch = MMA_Atom<mma_traits>;

  static constexpr int kNWarps = 4;

  using TiledMMA = TiledMMA<MMA_Atom_Arch,
                      Layout<Shape<Int<kNWarps>,_1,_1>>,
                      Tile<Int<16 * kNWarps>, _16, _16>>;
//   using TiledMMA = decltype(make_tiled_mma(MMA_Atom_Arch{}, 
//                       make_layout(Shape<_2, _2, _1>{}), 
//                       make_layout(Shape<_4, _1, _1>{})));
  // 上面两种写法等同
  // print(TiledMMA{});
  dim3 dimGrid(size(ceil_div(N, bN)), 
               size(ceil_div(M, bM)));
  dim3 dimBlock(size(TiledMMA{}));

  gemm_device<T, bM, bN, bK, TiledMMA><<<dimGrid, dimBlock, 0, 0>>>(A, B, C,  M, N, K);
}

int main() {

    const int test_num = 64;
    int M_list[test_num];
    int N_list[test_num];
    int K_list[test_num];

    for (int i = 0; i < test_num; i++) {
        M_list[i] = (i + 1) * 256;
        N_list[i] = (i + 1) * 256;
        K_list[i] = (i + 1) * 256;
    }

    const int outer_repeat = 10, inner_repeat = 1;

    printf("\nalgo = Cute_HGEMM_V1\n");
    int test_maxerror_num = 5;
    for (int j = 0; j < test_maxerror_num; j++) {
        int M = M_list[j], N = N_list[j], K = K_list[j];
        float max_error = testF16F16GemmMaxError_V2<T>(gemm_v1, M, N, K);
        printf("M N K = %6d %6d %6d, ", M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    for (int j = 0; j < test_num; j++) {
        int M = M_list[j], N = N_list[j], K = K_list[j];
 
        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = testF16F16GemmPerformance<T>(
                gemm_v1, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, ", M, N, K);
        printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
        printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
    }


    return 0;
}