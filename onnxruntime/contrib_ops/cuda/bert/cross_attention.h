// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cross_attention/fmha_cross_attention.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class CrossAttention final : public CudaKernel {
 public:
  CrossAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  int num_heads_;  // number of attention heads
  bool disable_fused_runner_;
  bool enable_flash_attention_;
  bool disable_fused_cross_attention_;
  mutable std::unique_ptr<MHARunner> fused_fp16_runner_;
  mutable const FusedMultiHeadCrossAttentionKernel* fused_fp16_cross_attention_kernel_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime