// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cuda/cuda_common.h"
#include "42_fused_multi_head_attention/kernel_forward.h" // in cutlass examples

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct FmhaParams {
  int32_t sm;
  bool is_float16;
  int64_t batch_size;
  int64_t num_heads;
  int64_t sequence_length;
  int64_t kv_sequence_length;
  int64_t qk_head_size;
  int64_t v_head_size;
  bool causal;
  const void* query; // [B, S, N, H], or [B, M, n_heads, K] in xFormers
  const void* key;   // [B, L, N, H], or [B, N, n_heads, K] in xFormers
  const void* value; // [B, L, N, H_v], or [B, N, n_heads, Kv] in xFormers

  // int32_t* cu_seqlens_q;
  // int32_t* cu_seqlens_k;

  void* output;         // [B, S, N, H_v]
  void* workspace;      // [B, S, N, H_v] when kNeedsOutputAccumulatorBuffer
  int64_t workspace_size;

  cudaStream_t  stream;

  int32_t QueryHiddenSize() const {return num_heads * qk_head_size;}
  int32_t ValueHiddenSize() const {return num_heads * v_head_size;}
};

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
      ORT_ENFORCE(params.workspace_size >= params.batch_size * params.sequence_length * params.num_heads * params.v_head_size * sizeof(Acc));
      p.output_accum_ptr = reinterpret_cast<Acc*>(params.workspace);
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.num_heads = params.num_heads;
    p.num_batches = params.batch_size;
    p.head_dim = params.qk_head_size;
    p.head_dim_value = params.v_head_size;

    // When params.cu_seqlens_q is provided, num_queries and num_keys will be set inside the kernel
    p.num_queries = params.batch_size * params.sequence_length;
    p.num_keys = params.batch_size * params.kv_sequence_length;

    p.causal = params.causal;

    p.q_strideH = params.qk_head_size;
    p.k_strideH = params.qk_head_size;
    p.v_strideH = params.v_head_size;
    p.o_strideH = params.v_head_size;

    // These might overflow for big tensors
    p.q_strideM = params.num_heads * params.qk_head_size;
    p.k_strideM = params.num_heads * params.qk_head_size;
    p.v_strideM = params.num_heads * params.v_head_size;

    // q/k/v_strideB is needed only when cu_seqlens_q and cu_seqlens_k are not avaiable
    p.q_strideB = p.q_strideM * params.sequence_length;
    p.k_strideB = p.k_strideM * params.kv_sequence_length;
    p.v_strideB = p.v_strideM * params.kv_sequence_length;
    p.o_strideB = p.v_strideM * params.sequence_length;

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

template<typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block>
void DispatchSingleValueIteration(const FmhaParams& params) {
  if (params.v_head_size <= keys_per_block) {
    LaunchCutlassFmha<T, ArchTag, is_aligned, queries_per_block, keys_per_block, true>(params);
  } else {
    LaunchCutlassFmha<T, ArchTag, is_aligned, queries_per_block, keys_per_block, false>(params);
  }
}

template<typename T, typename ArchTag, bool is_aligned>
void DispatchKeysPerBlock(const FmhaParams& params) {
  if (params.v_head_size <= 64) {
    DispatchSingleValueIteration<T, ArchTag, is_aligned, 64, 64>(params);
  } else {
    DispatchSingleValueIteration<T, ArchTag, is_aligned, 32, 128>(params);
  }
}

template<typename T, typename ArchTag>
void DispatchIsAligned(const FmhaParams& params) {
  if (reinterpret_cast<uintptr_t>(params.query) % 16 == 0
      && reinterpret_cast<uintptr_t>(params.key) % 16 == 0
      && params.QueryHiddenSize() % (16 / sizeof(T)) == 0
      && params.ValueHiddenSize() % (16 / sizeof(T)) == 0) {
    DispatchKeysPerBlock<T, ArchTag, true>(params);
  } else {
    DispatchKeysPerBlock<T, ArchTag, false>(params);
  }
}

template<typename T>
void DispatchArchTag(const FmhaParams& params) {
  const int32_t &sm = params.sm;
  if (sm == 80 || sm == 86 || sm == 89) {
    DispatchIsAligned<T, cutlass::arch::Sm80>(params);
  } else if (sm == 75) {
      DispatchIsAligned<T, cutlass::arch::Sm75>(params);
    } else if (sm == 70) {
      DispatchIsAligned<T, cutlass::arch::Sm70>(params);
    } else {
      ORT_ENFORCE(!"not implemented");
  }
}

void DispatchCutlassFmha(const FmhaParams& params) {
  if (params.is_float16) {
    DispatchArchTag<cutlass::half_t>(params);
  } else {
    DispatchArchTag<cutlass::tfloat32_t>(params);
  }
}

}
}
}
