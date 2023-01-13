// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/common/gsl.h"
#include "contrib_ops/cuda/bert/attention_fmha.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "41_fused_multi_head_attention/kernel_forward.h" // in cutlass examples

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define ASSIGN_NO_OVERFLOW(A, B)                                       \
  {                                                                    \
    A = B;                                                             \
    ORT_ENFORCE(                                                       \
        B < std::numeric_limits<decltype(A)>::max(), #B " overflows"); \
  }


template<typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block,
         bool single_value_iteration>
void LaunchCutlassFmha(const FmhaParams& params) {
  using Attention = AttentionKernel<T, ArchTag, is_aligned, queries_per_block, keys_per_block, single_value_iteration>;
  typename Attention::Params p;
  { // set parameters
    p.query_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.query));
    p.key_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.key));
    p.value_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.value));
    // p.cu_seqlens_q_ptr = params.cu_seqlens_q;
    // p.cu_seqlens_k_ptr = params.cu_seqlens_k;

    p.logsumexp_ptr = nullptr; // [num_heads, num_queries] for backward or nullptr for forward
    p.output_ptr = reinterpret_cast<T*>(params.output);
    if (Attention::kNeedsOutputAccumulatorBuffer) {
      using Acc = typename Attention::accum_t;
      //ORT_ENFORCE(params.workspace_size >= params.batch_size * params.sequence_length * params.num_heads * params.v_head_size * sizeof(Acc));
      p.output_accum_ptr = reinterpret_cast<Acc*>(params.workspace);
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.num_heads = params.num_heads;
    p.num_batches = params.batch_size;
    p.head_dim = params.qk_head_size;
    p.head_dim_value = params.v_head_size;

    // When params.cu_seqlens_q is provided, num_queries is max_seq_q and num_keys will be set inside the kernel
    p.num_queries = params.sequence_length;
    p.num_keys = params.kv_sequence_length;

    p.causal = params.causal;

    // q/k/v_strideB is needed only when cu_seqlens_q and cu_seqlens_k are not avaiable
    // These might overflow for big tensors.
    ASSIGN_NO_OVERFLOW(p.q_strideB, params.num_heads * params.qk_head_size * params.sequence_length);
    ASSIGN_NO_OVERFLOW(p.k_strideB, params.num_heads * params.qk_head_size * params.kv_sequence_length);
    ASSIGN_NO_OVERFLOW(p.v_strideB, params.num_heads * params.v_head_size * params.kv_sequence_length);
    ASSIGN_NO_OVERFLOW(p.o_strideB, params.num_heads * params.v_head_size * params.sequence_length);

    p.q_strideM = params.num_heads * params.qk_head_size;
    p.k_strideM = params.num_heads * params.qk_head_size;
    p.v_strideM = params.num_heads * params.v_head_size;

    p.q_strideH = params.qk_head_size;
    p.k_strideH = params.qk_head_size;
    p.v_strideH = params.v_head_size;
    p.o_strideH = params.v_head_size;

    p.causal = params.causal;
  }

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    ORT_ENFORCE(params.sm >= 70, "This kernel requires too much shared memory on this machine!");
    static bool once = [&]() {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      return true;
    }();
  }

  ORT_ENFORCE(Attention::check_supported(p));
  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, params.stream>>>(p);
}


template<typename T, typename ArchTag, int queries_per_block, int keys_per_block, bool single_value_iteration>
void DispatchIsAligned(const FmhaParams& params){

  using AlignedAK = AttentionKernel<T, ArchTag, true, queries_per_block, keys_per_block, single_value_iteration>;

   // Run a more efficient kernel with `isAligned=True` when memory is correctly aligned.
  bool is_aligned = params.qk_head_size % AlignedAK::kAlignmentQ == 0 &&
                   params.qk_head_size % AlignedAK::kAlignmentK == 0 &&
                   params.v_head_size % AlignedAK::kAlignmentV == 0;

  DISPATCH_BOOL(is_aligned, kIsAligned, ([&]() {
    LaunchCutlassFmha<T, ArchTag, kIsAligned, queries_per_block, keys_per_block, single_value_iteration>(params);
  }));
}


template<typename T, typename ArchTag>
void DispatchBlockSize(const FmhaParams& params) {
  if (params.v_head_size <= 64) {
    DispatchIsAligned<T, ArchTag, 64, 64, true>(params);
  } else if (params.v_head_size <= 128) {
    DispatchIsAligned<T, ArchTag, 32, 128, true>(params);
  } else {
    DispatchIsAligned<T, ArchTag, 32, 128, false>(params);
  }
}

//TODO: split each SM to a cu file to speed up compiling.
template<typename T>
void DispatchArchTag(const FmhaParams& params) {
  const int32_t &sm = params.sm;
  if (sm == 80 || sm == 86 || sm == 89) {
    DispatchBlockSize<T, cutlass::arch::Sm80>(params);
  } else if (sm == 75) {
      DispatchBlockSize<T, cutlass::arch::Sm75>(params);
    } else if (sm == 70) {
      DispatchBlockSize<T, cutlass::arch::Sm70>(params);
    } else {
      ORT_ENFORCE(!"not implemented");
  }
}

void run_cutlass_fused_attention(const FmhaParams& params) {
  if (params.is_float16) {
    DispatchArchTag<cutlass::half_t>(params);
  } else {
    DispatchArchTag<float>(params);
  }
}

}
}
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
