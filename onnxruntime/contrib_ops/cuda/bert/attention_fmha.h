// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/cuda/cuda_common.h"

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
  //int64_t workspace_size;

  cudaStream_t  stream;
};

void run_cutlass_fused_attention(const FmhaParams& params);

inline bool has_cutlass_fused_attention(int32_t sm){
  return (sm==70 || sm==75 || sm==80 || sm==86 || sm==89);
}

}
}
}
