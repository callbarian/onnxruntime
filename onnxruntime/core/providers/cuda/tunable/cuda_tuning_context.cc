// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tunable/cuda_tunable.h"

#include "core/framework/tuning_context.h"
#define TUNING_CONTEXT_IMPL
#include "core/framework/tuning_context_impl.h"
#undef TUNING_CONTEXT_IMPL

namespace onnxruntime {
namespace cuda {
namespace tunable {

CudaTuningContext::CudaTuningContext(TunableOpInfo* info) : info_(info) {}

void CudaTuningContext::EnableTunableOp() {
  LOGS_DEFAULT(INFO) << "Enable TunableOp for CUDA Execution Provider";
  info_->enabled = true;
}

void CudaTuningContext::DisableTunableOp() {
  LOGS_DEFAULT(INFO) << "Disable TunableOp for CUDA Execution Provider";
  info_->enabled = false;
}

bool CudaTuningContext::IsTunableOpEnabled() const {
  return info_->enabled;
}

TuningResultsManager& CudaTuningContext::GetTuningResultsManager() {
  return manager_;
}

const TuningResultsManager& CudaTuningContext::GetTuningResultsManager() const {
  return manager_;
}

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
